"""File bornée « dernier gagne » pour le temps réel.

Quand la file est pleine, l'élément le plus ancien est abandonné : le consommateur
travaille toujours sur des données récentes et la latence reste bornée.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Generic, Optional, TypeVar

T = TypeVar("T")


class LatestQueue(Generic[T]):
    """File thread-safe, de taille fixe, qui écarte les plus anciens éléments."""

    def __init__(self, maxsize: int = 2) -> None:
        self._maxsize = max(1, int(maxsize))
        self._items: Deque[T] = deque()
        self._cond = threading.Condition()
        self._dropped = 0

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def dropped(self) -> int:
        """Nombre cumulé d'éléments écartés faute de place."""
        with self._cond:
            return self._dropped

    def put(self, item: T) -> bool:
        """Ajoute un élément. Retourne True si un ancien élément a été écarté."""
        with self._cond:
            overflow = len(self._items) >= self._maxsize
            if overflow:
                self._items.popleft()
                self._dropped += 1
            self._items.append(item)
            self._cond.notify()
            return overflow

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Retire le plus ancien élément (FIFO). None si délai dépassé."""
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            return self._items.popleft() if self._items else None

    def get_latest(self, timeout: Optional[float] = None) -> Optional[T]:
        """Retire l'élément le plus récent et vide le reste (idéal pour l'inférence)."""
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            if not self._items:
                return None
            item = self._items.pop()
            self._dropped += len(self._items)
            self._items.clear()
            return item

    def clear(self) -> None:
        with self._cond:
            self._items.clear()

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)
