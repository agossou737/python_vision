"""Source smartphone : méthodes réellement prises en charge.

Méthodes implémentées :
  - ``virtual_camera`` : scrcpy/DroidCam/etc. exposent une webcam virtuelle -> index local.
  - ``ip_app``         : application de streaming (IP Webcam, DroidCam...) -> URL réseau.
  - ``adb_forward``    : redirige un port USB (adb forward) vers un service vidéo
                         DÉJÀ lancé sur le téléphone, puis lit http://127.0.0.1:<port>.
Non implémentées (explicites) : ``bluetooth``, ``ios_direct``.
ADB ne fournit PAS de flux vidéo par lui-même.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import List, Optional, Tuple

import numpy as np

from app.video.base_source import (
    SourceNotSupportedError,
    SourceUnavailableError,
    VideoSource,
)
from app.video.local_camera import LocalCameraSource
from app.video.network_camera import NetworkCameraSource

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("virtual_camera", "ip_app", "adb_forward")
UNSUPPORTED_METHODS = {
    "bluetooth": (
        "Le Bluetooth n'expose pas de flux vidéo exploitable par OpenCV. "
        "Utilisez USB (caméra virtuelle) ou Wi-Fi (application de streaming)."
    ),
    "ios_direct": (
        "L'accès direct à la caméra d'un iPhone n'est pas disponible. "
        "Utilisez une application de streaming ou la Continuity Camera (macOS)."
    ),
}


def adb_available() -> bool:
    return shutil.which("adb") is not None


def _run_adb(args: List[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["adb", *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def list_adb_devices() -> List[Tuple[str, str]]:
    """Retourne [(serial, état)] ; état : device | unauthorized | offline."""
    if not adb_available():
        return []
    try:
        proc = _run_adb(["devices"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("adb devices a échoué : %s", exc)
        return []
    devices: List[Tuple[str, str]] = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            devices.append((parts[0], parts[1]))
    return devices


class SmartphoneSource(VideoSource):
    """Délègue à la source concrète correspondant à la méthode choisie."""

    def __init__(
        self,
        method: str,
        *,
        camera_index: int = 0,
        url: str = "",
        timeout_s: float = 5.0,
        adb_local_port: int = 8080,
        adb_remote_port: int = 8080,
        adb_stream_path: str = "/video",
    ) -> None:
        self.method = method
        self._camera_index = camera_index
        self._url = url
        self._timeout_s = timeout_s
        self._adb_local = adb_local_port
        self._adb_remote = adb_remote_port
        self._adb_path = adb_stream_path
        self._inner: Optional[VideoSource] = None
        self._forward_active = False

    @property
    def name(self) -> str:
        return f"Smartphone ({self.method})"

    def open(self) -> None:
        self.release()
        if self.method in UNSUPPORTED_METHODS:
            raise SourceNotSupportedError(UNSUPPORTED_METHODS[self.method])
        if self.method not in SUPPORTED_METHODS:
            raise SourceNotSupportedError(f"Méthode smartphone inconnue : {self.method}")

        if self.method == "virtual_camera":
            inner: VideoSource = LocalCameraSource(self._camera_index)
        elif self.method == "ip_app":
            inner = NetworkCameraSource(self._url, self._timeout_s)
        else:  # adb_forward
            self._setup_adb_forward()
            inner = NetworkCameraSource(
                f"http://127.0.0.1:{self._adb_local}{self._adb_path}", self._timeout_s
            )
        try:
            inner.open()
        except Exception:
            self._remove_adb_forward()
            raise
        self._inner = inner

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self._inner.read() if self._inner else (False, None)

    def release(self) -> None:
        inner, self._inner = self._inner, None
        if inner is not None:
            inner.release()
        self._remove_adb_forward()

    @property
    def is_opened(self) -> bool:
        return self._inner is not None and self._inner.is_opened

    @property
    def resolution(self) -> Optional[Tuple[int, int]]:
        return self._inner.resolution if self._inner else None

    # ---- ADB -------------------------------------------------------------
    def _setup_adb_forward(self) -> None:
        if not adb_available():
            raise SourceUnavailableError(
                "ADB est introuvable : installez les platform-tools Android et ajoutez-les au PATH."
            )
        devices = list_adb_devices()
        if not devices:
            raise SourceUnavailableError(
                "Aucun appareil ADB détecté : branchez le téléphone et activez le débogage USB."
            )
        if not any(state == "device" for _, state in devices):
            raise SourceUnavailableError(
                "Appareil détecté mais non autorisé : acceptez la demande de débogage USB sur le téléphone."
            )
        try:
            proc = _run_adb(
                ["forward", f"tcp:{self._adb_local}", f"tcp:{self._adb_remote}"]
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SourceUnavailableError("La commande 'adb forward' a échoué.") from exc
        if proc.returncode != 0:
            logger.warning("adb forward : %s", proc.stderr.strip())
            raise SourceUnavailableError(
                "Impossible de rediriger le port ADB (port local déjà utilisé ?)."
            )
        self._forward_active = True

    def _remove_adb_forward(self) -> None:
        if not self._forward_active:
            return
        self._forward_active = False
        try:
            _run_adb(["forward", "--remove", f"tcp:{self._adb_local}"])
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("Impossible de retirer la redirection ADB tcp:%d", self._adb_local)
