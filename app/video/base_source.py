"""Abstraction commune des sources vidéo.

Le moteur de détection et l'interface ne dépendent que de ``VideoSource`` :
ils ignorent s'il s'agit d'une webcam, d'un flux réseau ou d'un smartphone.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SourceError(Exception):
    """Erreur générique de source vidéo (message destiné à l'utilisateur)."""


class SourceUnavailableError(SourceError):
    """Source inaccessible (périphérique absent/occupé, réseau coupé...). Réessayable."""


class SourceNotSupportedError(SourceError):
    """Fonctionnalité non implémentée ou impossible. NON réessayable."""


class VideoSource(ABC):
    """Interface d'une source vidéo."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom lisible (affiché dans l'interface)."""

    @abstractmethod
    def open(self) -> None:
        """Ouvre la source. Lève SourceError en cas d'échec."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lit une image. Retourne (ok, image BGR) ; ne doit pas lever d'exception."""

    @abstractmethod
    def release(self) -> None:
        """Libère les ressources. Idempotent."""

    @property
    @abstractmethod
    def is_opened(self) -> bool: ...

    @property
    def resolution(self) -> Optional[Tuple[int, int]]:
        """(largeur, hauteur) si connue."""
        return None

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class CvCaptureSource(VideoSource):
    """Base commune des sources reposant sur ``cv2.VideoCapture``."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None

    @abstractmethod
    def _create_capture(self) -> cv2.VideoCapture:
        """Crée l'objet VideoCapture (appel potentiellement bloquant)."""

    @abstractmethod
    def _unavailable_message(self) -> str:
        """Message utilisateur si l'ouverture échoue."""

    def open(self) -> None:
        if self._cap is not None:
            self.release()
        cap: Optional[cv2.VideoCapture] = None
        try:
            cap = self._create_capture()
        except Exception as exc:  # backend OpenCV imprévisible
            logger.exception("Exception à l'ouverture de %s", self.name)
            if cap is not None:
                cap.release()
            raise SourceUnavailableError(self._unavailable_message()) from exc
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            raise SourceUnavailableError(self._unavailable_message())
        self._cap = cap

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        cap = self._cap
        if cap is None:
            return False, None
        try:
            ok, frame = cap.read()
        except cv2.error as exc:
            logger.warning("Erreur de lecture OpenCV (%s) : %s", self.name, exc)
            return False, None
        return (True, frame) if ok else (False, None)

    def release(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:  # ne jamais empêcher le nettoyage
                logger.exception("Erreur lors de la libération de %s", self.name)

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Optional[Tuple[int, int]]:
        if self._cap is None:
            return None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h) if w > 0 and h > 0 else None
