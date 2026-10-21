"""Interface commune des modèles de détection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from app.detection.types import Detection


class DetectionError(Exception):
    """Erreur du moteur de détection (message destiné à l'utilisateur)."""


class ModelLoadError(DetectionError):
    """Échec de chargement du modèle (poids, dépendance ou matériel)."""


class InferenceError(DetectionError):
    """Échec pendant l'inférence."""


def resolve_device(requested: str = "auto") -> str:
    """Traduit ``auto`` en cuda:0 / mps / cpu selon ce qui est réellement disponible.

    Sans PyTorch installé ou sans GPU compatible, retombe sur le CPU.
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch  # import tardif : lourd, et facultatif pour les tests
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class BaseDetector(ABC):
    """Contrat que tout modèle doit respecter pour être utilisé par DetectionEngine."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Noms des classes, indexés par class_id."""

    @property
    def device(self) -> str:
        return "cpu"

    @abstractmethod
    def load(self, model_name: str, device: str = "auto") -> None:
        """Charge le modèle (BLOQUANT). Lève ModelLoadError."""

    @abstractmethod
    def detect(self, frame: np.ndarray, confidence: float, imgsz: int) -> List[Detection]:
        """Inférence sur une image BGR. Le seuil est passé à chaque appel (réglage à chaud)."""

    @abstractmethod
    def unload(self) -> None:
        """Libère le modèle et la mémoire associée. Idempotent."""
