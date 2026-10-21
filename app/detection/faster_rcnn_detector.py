"""Détecteur Faster R-CNN (torchvision, poids COCO pré-entraînés).

Plus précis que YOLO nano sur certains objets, mais nettement plus lent sur CPU :
privilégier ``fasterrcnn_mobilenet_v3_large_320_fpn`` sans GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from app.detection.base_detector import (
    BaseDetector,
    ModelLoadError,
    resolve_device,
    to_numpy,
)
from app.detection.types import Detection

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = (
    "fasterrcnn_mobilenet_v3_large_320_fpn",   # le plus léger
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
)
_IGNORED_CATEGORIES = ("__background__", "N/A")


def parse_torchvision_output(
    output: Any, frame_shape: tuple, categories: Sequence[str], confidence: float
) -> List[Detection]:
    """Convertit la sortie torchvision ({boxes, labels, scores}) en liste de Detection.

    torchvision restitue déjà les boîtes dans le repère de l'image d'entrée
    (l'image originale) ; on borne simplement aux limites de l'image.
    """
    boxes = to_numpy(output["boxes"]).reshape(-1, 4)
    scores = to_numpy(output["scores"]).reshape(-1)
    labels = to_numpy(output["labels"]).reshape(-1).astype(int)
    h, w = int(frame_shape[0]), int(frame_shape[1])

    detections: List[Detection] = []
    for (x1, y1, x2, y2), score, cid in zip(boxes, scores, labels):
        if float(score) < confidence:
            continue
        label = categories[int(cid)] if 0 <= int(cid) < len(categories) else str(int(cid))
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


class FasterRcnnDetector(BaseDetector):
    def __init__(self, models_dir: Union[str, Path] = "assets/models") -> None:
        self._models_dir = Path(models_dir)
        self._model: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._categories: List[str] = []
        self._device = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> List[str]:
        return [c for c in self._categories if c not in _IGNORED_CATEGORIES]

    @property
    def device(self) -> str:
        return self._device

    def load(self, model_name: str, device: str = "auto") -> None:
        name = model_name.strip().lower()
        if name not in SUPPORTED_MODELS:
            raise ModelLoadError(
                f"Modèle Faster R-CNN non pris en charge : « {model_name} ». "
                f"Disponibles : {', '.join(SUPPORTED_MODELS)}."
            )
        try:
            import torch
            import torchvision
        except ImportError as exc:
            raise ModelLoadError(
                "PyTorch / torchvision ne sont pas installés (voir README, section PyTorch)."
            ) from exc

        self.unload()
        try:
            # Les poids téléchargés restent dans assets/models/torch (ignoré par Git).
            self._models_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(self._models_dir / "torch"))
        except OSError:
            logger.warning("Dossier de cache torch non inscriptible, cache par défaut utilisé.")

        try:
            weights = torchvision.models.get_model_weights(name).DEFAULT
            model = torchvision.models.get_model(name, weights=weights)
            categories = [str(c) for c in weights.meta["categories"]]
        except Exception as exc:
            logger.exception("Échec de construction de %s", name)
            raise ModelLoadError(
                f"Impossible de charger « {name} » (téléchargement des poids impossible "
                "ou version de torchvision incompatible)."
            ) from exc

        target = resolve_device(device)
        try:
            self._prepare(model, target)
        except Exception:
            if target == "cpu":
                logger.exception("Échec d'initialisation du modèle sur CPU")
                raise ModelLoadError(f"Échec d'initialisation de « {name} » sur CPU.") from None
            logger.warning("Échec sur %s, nouvel essai sur CPU", target, exc_info=True)
            target = "cpu"
            try:
                self._prepare(model, target)
            except Exception as exc2:
                raise ModelLoadError(f"Échec d'initialisation de « {name} ».") from exc2

        self._torch = torch
        self._model = model
        self._categories = categories
        self._device = target
        logger.info("Faster R-CNN %s chargé sur %s", name, target)

    def detect(self, frame: np.ndarray, confidence: float, imgsz: int) -> List[Detection]:
        model, torch = self._model, self._torch
        if model is None or torch is None:
            raise RuntimeError("Modèle non chargé")
        # imgsz = côté court après redimensionnement interne (réglage à chaud).
        model.transform.min_size = (int(imgsz),)
        model.transform.max_size = int(imgsz * 1333 / 800)
        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(self._device)
        with torch.inference_mode():
            output = model([tensor])[0]
        return parse_torchvision_output(output, frame.shape, self._categories, confidence)

    def unload(self) -> None:
        model, torch = self._model, self._torch
        self._model = None
        self._categories = []
        if model is not None:
            del model
            try:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    @staticmethod
    def _prepare(model: Any, device: str) -> None:
        """Place le modèle sur le périphérique et fait une inférence à blanc."""
        import torch

        model.to(device).eval()
        with torch.inference_mode():
            model([torch.zeros(3, 64, 64, device=device)])
