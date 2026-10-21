"""Événements échangés entre les workers (threads) et l'interface.

Les workers n'appellent JAMAIS Tkinter : ils déposent des WorkerEvent dans une
queue.Queue que le thread principal vide périodiquement avec ``root.after``.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class EventKind(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SIGNAL_LOST = "signal_lost"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    INFERENCE_ERROR = "inference_error"
    STOPPED = "stopped"


@dataclass(frozen=True)
class WorkerEvent:
    """Message d'état émis par un worker."""

    kind: EventKind
    message: str = ""
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def post_event(event_queue: "queue.Queue[WorkerEvent]", event: WorkerEvent) -> bool:
    """Dépose un événement sans jamais bloquer. Retourne False si la file est pleine."""
    try:
        event_queue.put_nowait(event)
        return True
    except queue.Full:
        return False
