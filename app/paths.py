"""Emplacements des fichiers modifiables, en mode normal ou « gelé » (PyInstaller)."""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """Vrai si l'application tourne depuis un exécutable PyInstaller."""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Dossier de travail de l'application (config/, logs/, assets/models/).

    - Code source : racine du projet.
    - Exécutable : dossier contenant l'exécutable (modifiable, contrairement au
      dossier temporaire d'extraction de PyInstaller).
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent
