"""Utilitaires image indépendants de l'interface graphique."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def is_valid_frame(frame: Any) -> bool:
    """Vrai si ``frame`` est une image BGR uint8 non vide (H, W, 3)."""
    return (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] == 3
        and frame.shape[0] > 0
        and frame.shape[1] > 0
        and frame.dtype == np.uint8
    )


def compute_fit_size(
    src_w: int, src_h: int, box_w: int, box_h: int
) -> Tuple[int, int, int, int]:
    """Calcule la taille d'affichage qui préserve les proportions.

    Retourne ``(new_w, new_h, offset_x, offset_y)`` où l'offset centre l'image
    dans la zone ``box_w x box_h``. Utile aussi pour convertir des coordonnées
    d'affichage vers l'image source.
    """
    if src_w <= 0 or src_h <= 0 or box_w <= 0 or box_h <= 0:
        return 0, 0, 0, 0
    scale = min(box_w / src_w, box_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    return new_w, new_h, (box_w - new_w) // 2, (box_h - new_h) // 2
