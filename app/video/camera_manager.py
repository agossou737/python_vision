"""Gestion du cycle de vie des sources vidéo et du worker de capture."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config_manager import VideoSettings
from app.events import WorkerEvent
from app.utils.latest_queue import LatestQueue
from app.video.base_source import SourceError, VideoSource
from app.video.frame import FramePacket
from app.video.local_camera import LocalCameraSource, CameraInfo, scan_local_cameras_async
from app.video.network_camera import NetworkCameraSource
from app.video.smartphone_source import SmartphoneSource
from app.video.video_capture_worker import SourceFactory, VideoCaptureWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceTestResult:
    ok: bool
    message: str
    resolution: Optional[Tuple[int, int]] = None


def build_source(settings: VideoSettings) -> VideoSource:
    """Fabrique la source correspondant aux paramètres courants."""
    if settings.source_type == "local":
        return LocalCameraSource(settings.camera_index)
    if settings.source_type == "network":
        return NetworkCameraSource(settings.stream_url, settings.connect_timeout_s)
    if settings.source_type == "smartphone":
        return SmartphoneSource(
            settings.smartphone_method,
            camera_index=settings.camera_index,
            url=settings.stream_url,
            timeout_s=settings.connect_timeout_s,
            adb_local_port=settings.adb_local_port,
            adb_remote_port=settings.adb_remote_port,
            adb_stream_path=settings.adb_stream_path,
        )
    raise SourceError(f"Type de source inconnu : {settings.source_type}")


class CameraManager:
    """Possède la file d'images, la file d'événements et le worker de capture.

    Garantit qu'un seul worker est actif : changer de source arrête et libère
    l'ancienne avant de démarrer la nouvelle.
    """

    def __init__(
        self,
        settings: VideoSettings,
        event_queue: Optional["queue.Queue[WorkerEvent]"] = None,
    ) -> None:
        self.settings = settings
        self.frame_queue: "LatestQueue[FramePacket]" = LatestQueue(settings.frame_queue_size)
        self.events: "queue.Queue[WorkerEvent]" = event_queue or queue.Queue(maxsize=256)
        self._worker: Optional[VideoCaptureWorker] = None
        self._lock = threading.Lock()

    # ---- Cycle de vie ------------------------------------------------------
    @property
    def is_running(self) -> bool:
        w = self._worker
        return w is not None and w.is_alive()

    @property
    def capture_fps(self) -> float:
        w = self._worker
        return w.capture_fps if w else 0.0

    @property
    def active_source_name(self) -> str:
        w = self._worker
        return w.source_name if w else ""

    @property
    def resolution(self) -> Optional[Tuple[int, int]]:
        w = self._worker
        return w.resolution if w else None

    def start(self, factory: Optional[SourceFactory] = None) -> None:
        """Démarre la capture. ``factory`` est rappelée à chaque (re)connexion."""
        with self._lock:
            self._stop_locked(wait=True, timeout=3.0)
            self.frame_queue.clear()
            if factory is None:
                settings = self.settings
                factory = lambda: build_source(settings)  # noqa: E731
            worker = VideoCaptureWorker(factory, self.frame_queue, self.events, self.settings)
            self._worker = worker
            worker.start()

    def stop(self, wait: bool = True, timeout: float = 3.0) -> bool:
        """Arrête la capture. Retourne True si le thread est bien terminé.

        Depuis l'interface, préférer ``wait=False`` (non bloquant) puis surveiller
        ``is_running``.
        """
        with self._lock:
            return self._stop_locked(wait, timeout)

    def _stop_locked(self, wait: bool, timeout: float) -> bool:
        worker, self._worker = self._worker, None
        if worker is None:
            return True
        worker.request_stop()
        if wait:
            worker.join(timeout)
            if worker.is_alive():
                logger.warning(
                    "Le worker de capture ne s'est pas arrêté en %.1f s (lecture bloquante ?).",
                    timeout,
                )
                return False
        return not worker.is_alive()

    def shutdown(self) -> None:
        """Nettoyage centralisé à la fermeture de l'application."""
        self.stop(wait=True, timeout=3.0)
        self.frame_queue.clear()

    # ---- Test d'une source avant démarrage ---------------------------------
    @staticmethod
    def test_source(source: VideoSource, timeout_s: float = 8.0) -> SourceTestResult:
        """Ouvre la source, lit une image et libère. BLOQUANT : appeler depuis un thread."""
        result: List[SourceTestResult] = []

        def _probe() -> None:
            try:
                source.open()
                for _ in range(10):
                    ok, frame = source.read()
                    if ok and frame is not None:
                        h, w = frame.shape[:2]
                        result.append(SourceTestResult(True, f"Source OK ({w}x{h}).", (w, h)))
                        return
                result.append(SourceTestResult(False, "Source ouverte mais aucune image reçue."))
            except SourceError as exc:
                result.append(SourceTestResult(False, str(exc)))
            except Exception:
                logger.exception("Échec du test de source")
                result.append(SourceTestResult(False, "Erreur inattendue pendant le test."))
            finally:
                try:
                    source.release()
                except Exception:
                    logger.exception("Erreur de libération après test")

        thread = threading.Thread(target=_probe, name="SourceTest", daemon=True)
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive() or not result:
            return SourceTestResult(False, f"Délai dépassé ({timeout_s:.0f} s) : source injoignable.")
        return result[0]

    def scan_cameras(self, callback) -> threading.Thread:
        """Détection non bloquante des caméras locales (callback appelé depuis un thread)."""
        return scan_local_cameras_async(callback, self.settings.scan_max_index)
