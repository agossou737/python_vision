"""Mesures de performance thread-safe (FPS, latence)."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque


class FpsCounter:
    """Calcule un FPS glissant sur une fenêtre de temps.

    Chaque composant (capture, inférence, affichage) possède SON compteur afin
    de ne jamais confondre les cadences.
    """

    def __init__(self, window_s: float = 2.0) -> None:
        self._window = float(window_s)
        self._ticks: Deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        limit = now - self._window
        while self._ticks and self._ticks[0] < limit:
            self._ticks.popleft()

    def tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._ticks.append(now)
            self._prune(now)

    def reset(self) -> None:
        with self._lock:
            self._ticks.clear()

    @property
    def fps(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if len(self._ticks) < 2:
                return 0.0
            span = self._ticks[-1] - self._ticks[0]
            return (len(self._ticks) - 1) / span if span > 0 else 0.0


class LatencyTracker:
    """Moyenne glissante de durées en millisecondes."""

    def __init__(self, size: int = 30) -> None:
        self._values: Deque[float] = deque(maxlen=max(1, size))
        self._last = 0.0
        self._lock = threading.Lock()

    def add(self, ms: float) -> None:
        with self._lock:
            self._values.append(float(ms))
            self._last = float(ms)

    def reset(self) -> None:
        with self._lock:
            self._values.clear()
            self._last = 0.0

    @property
    def average_ms(self) -> float:
        with self._lock:
            return sum(self._values) / len(self._values) if self._values else 0.0

    @property
    def last_ms(self) -> float:
        with self._lock:
            return self._last
