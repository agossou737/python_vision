"""Palette centralisée : changer de thème = modifier ce fichier ou le mode d'apparence.

Les couleurs « paire » sont (clair, sombre), format natif de CustomTkinter.
Les widgets Tk/ttk (Canvas, Treeview) n'acceptent pas les paires : utiliser ``resolve``.
"""

from __future__ import annotations

from typing import Tuple

import customtkinter as ctk

ColorPair = Tuple[str, str]

ACCENT = "#3b82f6"
OK = "#22c55e"
WARN = "#f59e0b"
ERROR = "#ef4444"
MUTED = "#9ca3af"

BG_PANEL: ColorPair = ("#f3f4f6", "#111827")
BG_CARD: ColorPair = ("#e5e7eb", "#1f2937")
BG_VIDEO: ColorPair = ("#d1d5db", "#0b0f19")
BG_BAR: ColorPair = ("#e5e7eb", "#0f172a")
TEXT: ColorPair = ("#111827", "#f9fafb")
TEXT_MUTED: ColorPair = ("#4b5563", "#9ca3af")

LEVEL_COLORS = {"ok": OK, "warn": WARN, "error": ERROR, "idle": MUTED}

# Couleurs des boîtes en BGR (OpenCV), assignées par class_id.
BOX_COLORS_BGR = [
    (246, 130, 59),
    (94, 197, 34),
    (68, 68, 239),
    (11, 158, 245),
    (246, 92, 139),
    (212, 182, 6),
    (22, 115, 249),
    (166, 85, 168),
    (128, 128, 250),
    (163, 230, 52),
]


def is_dark() -> bool:
    return ctk.get_appearance_mode() == "Dark"


def resolve(pair: ColorPair) -> str:
    """Couleur effective pour les widgets qui n'acceptent pas les paires."""
    return pair[1] if is_dark() else pair[0]


def box_color(class_id: int) -> Tuple[int, int, int]:
    return BOX_COLORS_BGR[class_id % len(BOX_COLORS_BGR)]
