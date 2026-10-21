"""Thread de capture : ouverture, lecture, détection de perte de signal, reconnexion."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional, Tuple

from app.config_manager import VideoSettings
from app.events import EventKind, WorkerEvent, post_event
from app.utils.image_utils import is_valid_frame
from app.utils.latest_queue import LatestQueue
from app.utils.metrics import FpsCounter
from app.video.base_source import SourceNotSupportedError, SourceError, VideoSource
from app.video.frame import FramePacket

logger = logging.getLogger(__name__)

SourceFactory = Callable[[], VideoSource]


class VideoCaptureWorker(threading.Thread):
    """Capture les images d'une source et les dépose dans une ``LatestQueue``.

    - Ne touche jamais à l'interface : l'état est publié via ``event_queue``.
    - Reconnexion avec délai progressif (exponentiel, plafonné).
    - ``request_stop()`` est non bloquant ; la libération de la source se fait
      dans le thread worker (``finally``).
    """

    def __init__(
        self,
        source_factory: SourceFactory,
        frame_queue: "LatestQueue[FramePacket]",
        event_queue: "queue.Queue[WorkerEvent]",
        settings: VideoSettings,
    ) -> None:
        super().__init__(name="VideoCaptureWorker", daemon=True)
        self._factory = source_factory
        self._frames = frame_queue
        self._events = event_queue
        self._settings = settings
        self._stop_event = threading.Event()
        self._fps = FpsCounter()
        self._frame_id = 0
        self._source_name = ""
        self._resolution: Optional[Tuple[int, int]] = None

    # ---- API publique (thread-safe) --------------------------------------
    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def capture_fps(self) -> float:
        return self._fps.fps

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def resolution(self) -> Optional[Tuple[int, int]]:
        return self._resolution

    # ---- Boucle principale -----------------------------------------------
    def run(self) -> None:
        attempt = 0
        try:
            while not self._stop_event.is_set():
                source: Optional[VideoSource] = None
                got_frames = False
                fatal = False
                try:
                    source = self._factory()
                    self._source_name = source.name
                    self._emit(EventKind.CONNECTING, f"Connexion à {source.name}...")
                    source.open()
                    self._resolution = source.resolution
                    self._emit(EventKind.CONNECTED, f"Connecté : {source.name}")
                    got_frames = self._read_loop(source)
                except SourceNotSupportedError as exc:
                    logger.warning("Source non prise en charge : %s", exc)
                    self._emit(EventKind.ERROR, str(exc))
                    fatal = True
                except SourceError as exc:
                    logger.warning("Erreur de source : %s", exc)
                    self._emit(EventKind.ERROR, str(exc))
                except Exception:
                    logger.exception("Erreur inattendue dans la capture")
                    self._emit(EventKind.ERROR, "Erreur inattendue pendant la capture (voir les logs).")
                finally:
                    self._safe_release(source)

                if fatal or self._stop_event.is_set():
                    break
                if not self._settings.auto_reconnect:
                    break

                attempt = 1 if got_frames else attempt + 1
                limit = self._settings.reconnect_max_attempts
                if limit and attempt > limit:
                    self._emit(
                        EventKind.ERROR,
                        f"Reconnexion abandonnée après {limit} tentatives.",
                    )
                    break

                delay = min(
                    self._settings.reconnect_initial_delay_s * (2 ** (attempt - 1)),
                    self._settings.reconnect_max_delay_s,
                )
                self._emit(
                    EventKind.RECONNECTING,
                    f"Reconnexion dans {delay:.1f} s (tentative {attempt})...",
                    {"attempt": attempt, "delay_s": delay},
                )
                if self._stop_event.wait(delay):
                    break
        finally:
            self._fps.reset()
            self._emit(EventKind.STOPPED, "Capture arrêtée.")

    def _read_loop(self, source: VideoSource) -> bool:
        """Lit jusqu'à arrêt ou perte de signal. Retourne True si au moins une image valide."""
        failures = 0
        got_frames = False
        max_failures = self._settings.max_consecutive_read_failures
        while not self._stop_event.is_set():
            ok, frame = source.read()
            if not ok or not is_valid_frame(frame):
                failures += 1
                if failures >= max_failures:
                    logger.warning("Signal perdu (%d lectures échouées)", failures)
                    self._emit(EventKind.SIGNAL_LOST, "Signal vidéo perdu.")
                    return got_frames
                self._stop_event.wait(0.01)
                continue
            failures = 0
            got_frames = True
            self._frame_id += 1
            self._fps.tick()
            if self._stop_event.is_set():
                break
            self._frames.put(
                FramePacket(frame, time.monotonic(), self._frame_id, self._source_name)  # type: ignore[arg-type]
            )
        return got_frames

    # ---- Outils internes ---------------------------------------------------
    def _emit(self, kind: EventKind, message: str, data: Optional[dict] = None) -> None:
        post_event(
            self._events,
            WorkerEvent(kind, message, self._source_name, data or {}),
        )

    @staticmethod
    def _safe_release(source: Optional[VideoSource]) -> None:
        if source is None:
            return
        try:
            source.release()
        except Exception:
            logger.exception("Erreur lors de la libération de la source")
