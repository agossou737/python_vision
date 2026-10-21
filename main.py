"""Point d'entrée : python main.py"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "Tkinter est introuvable.\n"
            "  Ubuntu/Debian : sudo apt install python3-tk\n"
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
