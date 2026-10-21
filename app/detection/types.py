"""Types de données de la détection (aucune dépendance à l'interface ni à un modèle)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from app.video.frame import FramePacket


@dataclass(frozen=True)
class Detection:
    """Un objet détecté. Coordonnées en pixels dans le repère de l'image ORIGINALE."""

    class_id: int
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class DetectionResult:
    """Image + détections associées, transmise à l'interface par l'InferenceWorker.

    - ``inferred`` : True si le modèle a tourné sur CETTE image ; False si les
      détections sont reprises de la dernière inférence (frame_skip).
    - ``engine_ready`` : False si aucun modèle n'est chargé ou si l'inférence a
      échoué ; ``detections`` est alors vide (aucun résultat inventé).
    """

    packet: FramePacket
    detections: Tuple[Detection, ...]
    inference_ms: float
    inferred: bool
    engine_ready: bool
