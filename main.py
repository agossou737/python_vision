"""Point d'entrée : python main.py"""

from __future__ import annotations

import os
import sys


def _ensure_std_streams() -> None:
    """Exécutable PyInstaller fenêtré : stdout/stderr valent None et feraient planter tqdm & co."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def main() -> int:
    _ensure_std_streams()
    try:
        import tkinter  # noqa: F401
    except ImportError as exc:
        print(
            "Tkinter est introuvable ou inutilisable.\n"
            f"  Python utilisé : {sys.executable} ({sys.version.split()[0]})\n"
            f"  Erreur réelle  : {exc!r}\n"
            "  Ubuntu/Debian : sudo apt install python3-tk "
            f"(ou python{sys.version_info.major}.{sys.version_info.minor}-tk)\n"
            "  Windows       : réinstallez Python en cochant « tcl/tk and IDLE »."
        )
        return 1
    try:
        from app.application import Application
    except ImportError as exc:
        print(
            f"Dépendance manquante : {exc.name or exc}\n"
            "Installez les dépendances : pip install -r requirements.txt"
        )
        return 1

    Application().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
