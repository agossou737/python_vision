"""Thread d'inférence : consomme les images, produit des DetectionResult."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Tuple

from app.config_manager import DetectionSettings
from app.detection.base_detector import InferenceError
from app.detection.detection_engine import DetectionEngine
from app.detection.types import Detection, DetectionResult
from app.events import EventKind, WorkerEvent, post_event
from app.utils.latest_queue import LatestQueue
from app.utils.metrics import FpsCounter, LatencyTracker
from app.video.frame import FramePacket

logger = logging.getLogger(__name__)

ERROR_EVENT_INTERVAL_S = 5.0


class InferenceWorker(threading.Thread):
    """Prend toujours l'image la plus récente (``get_latest``) : la latence reste bornée.

    - Chaque image reçue produit un résultat pour l'affichage, y compris les
      images ignorées par ``frame_skip`` (détections de la dernière inférence).
    - Sans modèle prêt ou après une erreur : résultat SANS détection, jamais de
      résultat inventé.
    - FPS d'inférence et latence ne comptent que les vraies inférences.
    """

    def __init__(
        self,
        engine: DetectionEngine,
        frame_queue: "LatestQueue[FramePacket]",
        result_queue: "LatestQueue[DetectionResult]",
        event_queue: "queue.Queue[WorkerEvent]",
        settings: DetectionSettings,
    ) -> None:
        super().__init__(name="InferenceWorker", daemon=True)
        self._engine = engine
        self._frames = frame_queue
        self._results = result_queue
        self._events = event_queue
        self._settings = settings
        self._stop_event = threading.Event()
        self._fps = FpsCounter()
        self._latency = LatencyTracker()
        self._last_count = 0
        self._last_error_event = 0.0

    # ---- API publique ----------------------------------------------------------
    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def inference_fps(self) -> float:
        return self._fps.fps

    @property
    def avg_latency_ms(self) -> float:
        return self._latency.average_ms

    @property
    def last_latency_ms(self) -> float:
        return self._latency.last_ms

    @property
    def last_detection_count(self) -> int:
        return self._last_count

    # ---- Boucle ------------------------------------------------------------------
    def run(self) -> None:
        # Démarre « à échéance » : la toute première image est toujours analysée.
        skipped = self._settings.frame_skip
        last_dets: Tuple[Detection, ...] = ()
        try:
            while not self._stop_event.is_set():
                packet = self._frames.get_latest(timeout=0.1)
                if packet is None:
                    continue

                if not self._engine.is_ready:
                    last_dets = ()
                    self._publish(packet, (), 0.0, inferred=False, ready=False)
                    continue

                if skipped < self._settings.frame_skip:
                    skipped += 1
                    self._publish(packet, last_dets, 0.0, inferred=False, ready=True)
                    continue
                skipped = 0

                start = time.perf_counter()
                try:
                    detections = self._engine.detect(packet.frame)
                except InferenceError as exc:
                    last_dets = ()
                    self._report_error(str(exc))
                    self._publish(packet, (), 0.0, inferred=False, ready=False)
                    continue
                except Exception:
                    logger.exception("Erreur inattendue dans l'inférence")
                    last_dets = ()
                    self._report_error("Erreur inattendue pendant l'analyse (voir les logs).")
                    self._publish(packet, (), 0.0, inferred=False, ready=False)
                    continue

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._latency.add(elapsed_ms)
                self._fps.tick()
                last_dets = tuple(detections)
                self._publish(packet, last_dets, elapsed_ms, inferred=True, ready=True)
        finally:
            self._fps.reset()
            self._latency.reset()
            self._last_count = 0

    # ---- Internes ----------------------------------------------------------------
    def _publish(
        self,
        packet: FramePacket,
        detections: Tuple[Detection, ...],
        ms: float,
        *,
        inferred: bool,
        ready: bool,
    ) -> None:
        self._last_count = len(detections)
        self._results.put(DetectionResult(packet, detections, ms, inferred, ready))

    def _report_error(self, message: str) -> None:
        """Limite les événements d'erreur (une cause persistante ne doit pas inonder l'UI)."""
        now = time.monotonic()
        if now - self._last_error_event >= ERROR_EVENT_INTERVAL_S:
            self._last_error_event = now
            post_event(self._events, WorkerEvent(EventKind.INFERENCE_ERROR, message))
