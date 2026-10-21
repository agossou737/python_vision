"""Type d'échange entre capture et inférence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """Une image capturée (BGR uint8) et ses métadonnées."""

    frame: np.ndarray
    timestamp: float      # time.monotonic() au moment de la capture
    frame_id: int         # compteur croissant propre au worker
    source_name: str

    @property
    def height(self) -> int:
        return int(self.frame.shape[0])

    @property
    def width(self) -> int:
        return int(self.frame.shape[1])
