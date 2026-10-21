"""Sources simulées pour les tests (aucune caméra physique requise)."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from app.video.base_source import SourceUnavailableError, VideoSource


class FakeSource(VideoSource):
    """Source scriptée : ``frames`` images valides puis des échecs de lecture."""

    def __init__(
        self,
        frames: int = 10,
        fail_open: bool = False,
        shape: Tuple[int, int, int] = (48, 64, 3),
        read_delay: float = 0.0,
        label: str = "fake",
    ) -> None:
        self.frames = frames
        self.fail_open = fail_open
        self.shape = shape
        self.read_delay = read_delay
        self.label = label
        self.opened = False
        self.released = False
        self.open_calls = 0
        self._served = 0

    @property
    def name(self) -> str:
        return f"Fake {self.label}"

    def open(self) -> None:
        self.open_calls += 1
        if self.fail_open:
            raise SourceUnavailableError("fake indisponible")
        self.opened = True
        self.released = False
        self._served = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.read_delay:
            time.sleep(self.read_delay)
        if not self.opened or self._served >= self.frames:
            return False, None
        self._served += 1
        return True, np.full(self.shape, 128, dtype=np.uint8)

    def release(self) -> None:
        self.opened = False
        self.released = True

    @property
    def is_opened(self) -> bool:
        return self.opened

    @property
    def resolution(self) -> Optional[Tuple[int, int]]:
        return (self.shape[1], self.shape[0])


class FakeFactory:
    """Fabrique qui mémorise toutes les sources créées."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.sources: List[FakeSource] = []

    def __call__(self) -> FakeSource:
        src = FakeSource(**self.kwargs)
        self.sources.append(src)
        return src


def wait_until(predicate, timeout: float = 3.0, step: float = 0.01) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(step)
    return predicate()
