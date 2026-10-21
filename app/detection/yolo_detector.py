"""Détecteur basé sur Ultralytics YOLO (YOLOv8n par défaut, ou toute version compatible)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

from app.detection.base_detector import BaseDetector, ModelLoadError, resolve_device
from app.detection.types import Detection

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    """Convertit un tenseur torch (éventuellement sur GPU) ou une liste en ndarray."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_yolo_result(result: Any, frame_shape: tuple, fallback_names: List[str]) -> List[Detection]:
    """Convertit un résultat Ultralytics en liste de Detection.

    Ultralytics renvoie les boîtes dans le repère de l'image passée en entrée (donc
    l'image originale, pas l'image redimensionnée) ; on borne seulement aux limites.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    conf = _to_numpy(boxes.conf).reshape(-1)
    cls = _to_numpy(boxes.cls).reshape(-1).astype(int)
    names = getattr(result, "names", None) or {}
    h, w = int(frame_shape[0]), int(frame_shape[1])

    detections: List[Detection] = []
    for (x1, y1, x2, y2), score, cid in zip(xyxy, conf, cls):
        if isinstance(names, dict):
            label = names.get(int(cid))
        else:
            label = names[int(cid)] if 0 <= int(cid) < len(names) else None
        if label is None:
            label = fallback_names[int(cid)] if 0 <= int(cid) < len(fallback_names) else str(cid)
        detections.append(
            Detection(
                class_id=int(cid),
                label=str(label),
                confidence=float(score),
                x1=int(max(0, min(w - 1, round(float(x1))))),
                y1=int(max(0, min(h - 1, round(float(y1))))),
                x2=int(max(0, min(w - 1, round(float(x2))))),
                y2=int(max(0, min(h - 1, round(float(y2))))),
            )
        )
    return detections


class YoloDetector(BaseDetector):
    """Charge un modèle Ultralytics une seule fois ; le seuil est passé à chaque appel."""

    def __init__(self, models_dir: Union[str, Path] = "assets/models") -> None:
        self._models_dir = Path(models_dir)
        self._model: Optional[Any] = None
        self._names: List[str] = []
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> List[str]:
        return list(self._names)

    @property
    def device(self) -> str:
        return self._device

    def load(self, model_name: str, device: str = "auto") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError(
                "Ultralytics n'est pas installé. Exécutez : pip install ultralytics"
            ) from exc

        self.unload()
        weights = self._resolve_weights(model_name)
        target = resolve_device(device)
        try:
            model = YOLO(weights)
            self._warm_up(model, target)
        except ModelLoadError:
            raise
        except Exception as exc:
            logger.exception("Échec de chargement du modèle %s sur %s", weights, target)
            if target != "cpu":
                logger.warning("Nouvel essai sur CPU après l'échec sur %s", target)
                try:
                    model = YOLO(weights)
                    self._warm_up(model, "cpu")
                    target = "cpu"
                except Exception as exc2:
                    raise ModelLoadError(
                        f"Impossible de charger le modèle « {model_name} » (poids invalides, "
                        "réseau indisponible pour le téléchargement, ou version incompatible)."
                    ) from exc2
            else:
                raise ModelLoadError(
                    f"Impossible de charger le modèle « {model_name} » (poids invalides, "
                    "réseau indisponible pour le téléchargement, ou version incompatible)."
                ) from exc

        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            self._names = [str(names[k]) for k in sorted(names)]
        else:
            self._names = [str(n) for n in names]
        self._model = model
        self._device = target
        logger.info("Modèle %s chargé sur %s (%d classes)", model_name, target, len(self._names))

    def detect(self, frame: np.ndarray, confidence: float, imgsz: int) -> List[Detection]:
        model = self._model
        if model is None:
            raise RuntimeError("Modèle non chargé")
        results = model.predict(
            frame, conf=confidence, imgsz=imgsz, device=self._device, verbose=False
        )
        if not results:
            return []
        return parse_yolo_result(results[0], frame.shape, self._names)

    def unload(self) -> None:
        model, self._model = self._model, None
        self._names = []
        if model is not None:
            del model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ---- Internes ----------------------------------------------------------
    def _resolve_weights(self, model_name: str) -> str:
        """Cherche les poids localement ; sinon tente un téléchargement dans models_dir."""
        direct = Path(model_name)
        if direct.is_absolute() or direct.exists():
            return str(direct)
        local = self._models_dir / model_name
        if local.exists():
            return str(local)
        try:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(str(local))
        except Exception as exc:
            logger.warning("Téléchargement dans %s impossible (%s).", self._models_dir, exc)
        if local.exists():
            return str(local)
        # Repli : Ultralytics gère lui-même le nom (téléchargement dans le dossier courant).
        logger.warning("Poids absents de %s ; repli sur le nom brut %s.", self._models_dir, model_name)
        return model_name

    @staticmethod
    def _warm_up(model: Any, device: str) -> None:
        """Première inférence à blanc : détecte tôt les erreurs de device et évite le pic de latence."""
        model.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, device=device, verbose=False)
