"""Moteur de détection : indépendant de l'interface graphique et de la capture."""

from __future__ import annotations

import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from app.config_manager import DetectionSettings
from app.detection.base_detector import (
    BaseDetector,
    InferenceError,
    ModelLoadError,
)
from app.detection.types import Detection
from app.utils.validators import clamp

logger = logging.getLogger(__name__)

DetectorFactory = Callable[[str], BaseDetector]


class EngineState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


def default_detector_factory(models_dir: Union[str, Path]) -> DetectorFactory:
    """Choisit le détecteur selon le nom du modèle (import tardif : dépendances lourdes)."""

    def _factory(model_name: str) -> BaseDetector:
        if model_name.lower().startswith("fasterrcnn"):
            try:
                from app.detection.faster_rcnn_detector import FasterRcnnDetector
            except ImportError as exc:
                raise ModelLoadError(
                    "Le détecteur Faster R-CNN n'est pas disponible (torchvision manquant "
                    "ou module absent dans cette version)."
                ) from exc
            return FasterRcnnDetector(models_dir)
        from app.detection.yolo_detector import YoloDetector

        return YoloDetector(models_dir)

    return _factory


class DetectionEngine:
    """Charge le modèle une fois, puis exécute l'inférence avec des réglages modifiables à chaud.

    Les paramètres (seuil, imgsz, classes) sont lus dans ``settings`` à CHAQUE
    inférence : les modifier ne recharge jamais le modèle.

    Concurrence :
      - ``load_model`` (BLOQUANT, à appeler hors thread graphique) charge le nouveau
        détecteur sans verrou d'inférence, puis l'échange atomiquement ;
      - ``detect`` et l'échange/déchargement partagent un verrou : l'ancien modèle
        n'est jamais libéré pendant qu'il sert.
    """

    def __init__(
        self,
        settings: DetectionSettings,
        models_dir: Union[str, Path] = "assets/models",
        detector_factory: Optional[DetectorFactory] = None,
    ) -> None:
        self.settings = settings
        self._factory = detector_factory or default_detector_factory(models_dir)
        self._detector: Optional[BaseDetector] = None
        self._lock = threading.Lock()          # protège _detector + inférence
        self._load_lock = threading.Lock()     # sérialise les chargements
        self._state_lock = threading.Lock()
        self._state = EngineState.UNLOADED
        self._message = "Aucun modèle chargé."
        self._model_name = ""

    # ---- État ----------------------------------------------------------------
    @property
    def state(self) -> EngineState:
        with self._state_lock:
            return self._state

    @property
    def status_message(self) -> str:
        with self._state_lock:
            return self._message

    @property
    def is_ready(self) -> bool:
        return self.state == EngineState.READY

    @property
    def model_name(self) -> str:
        with self._state_lock:
            return self._model_name

    @property
    def class_names(self) -> List[str]:
        det = self._detector
        return det.class_names if det is not None else []

    @property
    def device(self) -> str:
        det = self._detector
        return det.device if det is not None else ""

    def _set_state(self, state: EngineState, message: str, model_name: Optional[str] = None) -> None:
        with self._state_lock:
            self._state = state
            self._message = message
            if model_name is not None:
                self._model_name = model_name

    # ---- Chargement ------------------------------------------------------------
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Charge (ou remplace) le modèle. BLOQUANT. Lève ModelLoadError.

        En cas d'échec, l'ancien modèle reste actif s'il y en avait un.
        """
        name = (model_name or self.settings.model_name).strip()
        with self._load_lock:
            previous_ready = self._detector is not None
            self._set_state(EngineState.LOADING, f"Chargement du modèle {name}...")
            try:
                detector = self._factory(name)
                detector.load(name, self.settings.device)
            except ModelLoadError as exc:
                self._on_load_failure(str(exc), previous_ready)
                raise
            except Exception as exc:
                logger.exception("Erreur inattendue au chargement de %s", name)
                self._on_load_failure(f"Erreur inattendue au chargement de « {name} ».", previous_ready)
                raise ModelLoadError(f"Erreur inattendue au chargement de « {name} ».") from exc

            with self._lock:
                old, self._detector = self._detector, detector
                if old is not None:
                    try:
                        old.unload()
                    except Exception:
                        logger.exception("Erreur au déchargement de l'ancien modèle")
            self.settings.model_name = name
            self._set_state(
                EngineState.READY, f"Modèle prêt : {name} ({detector.device})", name
            )
            logger.info("Moteur prêt : %s sur %s", name, detector.device)

    def _on_load_failure(self, message: str, previous_ready: bool) -> None:
        logger.error("Chargement du modèle échoué : %s", message)
        if previous_ready:
            self._set_state(EngineState.READY, f"{message} Ancien modèle conservé.")
        else:
            self._set_state(EngineState.ERROR, message)

    def unload(self) -> None:
        """Libère le modèle (fermeture de l'application ou changement radical)."""
        with self._lock:
            det, self._detector = self._detector, None
            if det is not None:
                try:
                    det.unload()
                except Exception:
                    logger.exception("Erreur au déchargement du modèle")
        self._set_state(EngineState.UNLOADED, "Aucun modèle chargé.")

    # ---- Réglages à chaud --------------------------------------------------------
    def set_confidence(self, value: float) -> None:
        self.settings.confidence = clamp(float(value), 0.05, 0.95)

    def set_imgsz(self, value: int) -> None:
        self.settings.imgsz = int(clamp(int(value), 160, 1920))

    def set_enabled_classes(self, labels: List[str]) -> None:
        """Liste vide = toutes les classes."""
        self.settings.enabled_classes = list(labels)

    # ---- Inférence -----------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Exécute l'inférence. Lève InferenceError ; ne renvoie jamais de faux résultats."""
        confidence = self.settings.confidence
        imgsz = self.settings.imgsz
        allowed = frozenset(self.settings.enabled_classes)
        with self._lock:
            detector = self._detector
            if detector is None:
                raise InferenceError("Aucun modèle chargé.")
            try:
                detections = detector.detect(frame, confidence, imgsz)
            except Exception as exc:
                logger.exception("Erreur d'inférence")
                raise InferenceError("Erreur pendant l'analyse de l'image.") from exc
        if allowed:
            detections = [d for d in detections if d.label in allowed]
        return detections
