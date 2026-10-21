"""Flux réseau (HTTP/MJPEG, RTSP...) lisibles par OpenCV/FFmpeg."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import cv2

from app.utils.validators import validate_stream_url
from app.video.base_source import CvCaptureSource, SourceError

logger = logging.getLogger(__name__)


class NetworkCameraSource(CvCaptureSource):
    """Source réseau : l'URL doit pointer vers un flux réellement exposé.

    Les exemples d'URL (``http://IP:8080/video``, ``rtsp://IP:554/stream``) ne sont
    pas universels : ils dépendent de l'application émettrice.
    """

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        super().__init__()
        self.url = (url or "").strip()
        self.timeout_s = float(timeout_s)

    @property
    def name(self) -> str:
        parsed = urlparse(self.url)
        host = parsed.netloc or self.url
        return f"Flux réseau {host}"

    def open(self) -> None:
        check = validate_stream_url(self.url)
        if not check:
            raise SourceError(check.message)
        super().open()

    def _create_capture(self) -> cv2.VideoCapture:
        if self.url.lower().startswith("rtsp"):
            # Doit être défini avant la première utilisation de FFmpeg dans le processus.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;tcp|stimeout;{int(self.timeout_s * 1_000_000)}",
            )
        timeout_ms = int(self.timeout_s * 1000)
        params = []
        # Propriétés de timeout disponibles selon la version d'OpenCV.
        for prop in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
            if hasattr(cv2, prop):
                params.extend([getattr(cv2, prop), timeout_ms])
        try:
            if params:
                return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            logger.debug("Signature VideoCapture avec paramètres non supportée, repli simple.")
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _unavailable_message(self) -> str:
        return (
            "Impossible d'ouvrir le flux réseau : vérifiez l'adresse, que l'émetteur "
            "est démarré et que les deux appareils sont sur le même réseau."
        )
