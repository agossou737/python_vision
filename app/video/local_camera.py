"""Webcams locales (intégrées ou USB) + détection non bloquante des périphériques."""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2

from app.video.base_source import CvCaptureSource

logger = logging.getLogger(__name__)


def _preferred_backend() -> int:
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW      # ouverture plus rapide et plus fiable que MSMF
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class LocalCameraSource(CvCaptureSource):
    """Webcam accessible par index (``cv2.VideoCapture(index)``)."""

    def __init__(self, index: int = 0, backend: Optional[int] = None) -> None:
        super().__init__()
        self.index = int(index)
        self._backend = _preferred_backend() if backend is None else backend

    @property
    def name(self) -> str:
        return f"Caméra locale #{self.index}"

    def _create_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.index, self._backend)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # limite la latence (si supporté)
        except Exception:
            pass
        return cap

    def _unavailable_message(self) -> str:
        return (
            f"Caméra #{self.index} indisponible : vérifiez qu'elle est branchée "
            "et qu'aucun autre programme ne l'utilise."
        )


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    width: int
    height: int


def scan_local_cameras(
    max_index: int = 5, stop_event: Optional[threading.Event] = None
) -> List[CameraInfo]:
    """Teste les index 0..max_index. BLOQUANT : l'appeler depuis un thread.

    Une caméra est retenue si elle s'ouvre ET fournit une image.
    """
    found: List[CameraInfo] = []
    for index in range(max_index + 1):
        if stop_event is not None and stop_event.is_set():
            break
        source = LocalCameraSource(index)
        try:
            source.open()
            ok, frame = source.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                found.append(CameraInfo(index, f"Caméra #{index} ({w}x{h})", w, h))
        except Exception as exc:
            logger.debug("Index %d non disponible : %s", index, exc)
        finally:
            source.release()
    return found


def scan_local_cameras_async(
    callback: Callable[[List[CameraInfo]], None],
    max_index: int = 5,
    stop_event: Optional[threading.Event] = None,
) -> threading.Thread:
    """Lance le scan dans un thread daemon.

    ``callback`` est appelé DEPUIS CE THREAD : l'interface doit donc re-planifier
    le traitement dans le thread principal (queue + ``root.after``).
    """

    def _run() -> None:
        try:
            result = scan_local_cameras(max_index, stop_event)
        except Exception:
            logger.exception("Échec du scan des caméras")
            result = []
        callback(result)

    thread = threading.Thread(target=_run, name="CameraScan", daemon=True)
    thread.start()
    return thread
