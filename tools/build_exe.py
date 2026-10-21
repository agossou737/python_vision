"""Génère un exécutable desktop avec PyInstaller (mode « dossier », pas « fichier unique »).

Usage (depuis la racine du projet, environnement virtuel actif) :
    pip install pyinstaller
    python tools/build_exe.py              # fenêtré (Windows : sans console)
    python tools/build_exe.py --console    # avec console, pour diagnostiquer un problème

Résultat : dist/VisionDesktop/ (à copier tel quel). Les dossiers config/, logs/ et
assets/models/ sont créés À CÔTÉ de l'exécutable : ils restent modifiables.

Pourquoi « onedir » : le mode « onefile » ré-extrait toute l'application (dont PyTorch,
plusieurs centaines de Mo) dans un dossier temporaire à chaque lancement.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Construit l'exécutable Vision Desktop")
    parser.add_argument("--name", default="VisionDesktop")
    parser.add_argument("--console", action="store_true", help="conserver la console (debug)")
    args = parser.parse_args()

    try:
        import PyInstaller.__main__ as pyinstaller
    except ImportError:
        print("PyInstaller n'est pas installé : pip install pyinstaller")
        return 1

    options = [
        str(ROOT / "main.py"),
        "--name", args.name,
        "--noconfirm", "--clean", "--onedir",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
        "--paths", str(ROOT),
        # Thèmes et ressources de CustomTkinter (sinon : fenêtre sans style ou erreur au lancement).
        "--collect-all", "customtkinter",
        # Imports tardifs (dépendances lourdes) : on force leur prise en compte.
        "--hidden-import", "app.detection.yolo_detector",
        "--hidden-import", "app.detection.faster_rcnn_detector",
        "--console" if args.console else "--windowed",
    ]
    if importlib.util.find_spec("ultralytics") is not None:
        options += ["--collect-all", "ultralytics"]  # fichiers de config YAML des modèles
    else:
        print("Note : Ultralytics n'est pas installé ici ; l'exécutable ne contiendra pas YOLO.")

    print("Lancement de PyInstaller…")
    pyinstaller.run(options)

    target = ROOT / "dist" / args.name
    if not target.exists():
        print("Échec : le dossier de sortie est introuvable.")
        return 1
    for sub in ("config", "logs", "assets/models"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    default_config = ROOT / "config" / "settings.json"
    if default_config.exists():
        shutil.copy2(default_config, target / "config" / "settings.json")

    print(f"\nExécutable prêt : {target}")
    print("Premier lancement : les poids YOLO se téléchargent dans assets/models/ (réseau requis).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
