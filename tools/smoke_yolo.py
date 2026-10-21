"""Vérification de bout en bout du moteur YOLO (à lancer sur votre machine).

Usage :
    python tools/smoke_yolo.py                      # image synthétique (charge + inférence)
    python tools/smoke_yolo.py chemin/image.jpg     # vraie image : affiche les détections

Valide : chargement des poids (téléchargement dans assets/models si absents),
choix du device, inférence, changement de seuil SANS rechargement, coordonnées.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config_manager import DetectionSettings  # noqa: E402
from app.detection.base_detector import ModelLoadError  # noqa: E402
from app.detection.detection_engine import DetectionEngine  # noqa: E402


def main() -> int:
    settings = DetectionSettings()
    engine = DetectionEngine(settings, models_dir=ROOT / "assets" / "models")

    t0 = time.perf_counter()
    try:
        engine.load_model()
    except ModelLoadError as exc:
        print(f"ECHEC chargement : {exc}")
        return 1
    print(f"Modèle prêt sur {engine.device} en {time.perf_counter() - t0:.1f} s "
          f"({len(engine.class_names)} classes)")

    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print("Image illisible.")
            return 1
    else:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    h, w = frame.shape[:2]
    for conf in (0.5, 0.25, 0.05):
        engine.set_confidence(conf)
        t = time.perf_counter()
        dets = engine.detect(frame)
        ms = (time.perf_counter() - t) * 1000
        print(f"\nseuil={conf:.2f} -> {len(dets)} détection(s) en {ms:.0f} ms (image {w}x{h})")
        for d in dets[:10]:
            inside = 0 <= d.x1 < d.x2 < w and 0 <= d.y1 < d.y2 < h
            print(f"  {d.label:<14} {d.confidence:.2f}  ({d.x1},{d.y1})-({d.x2},{d.y2})  "
                  f"{'OK' if inside else 'HORS IMAGE'}")

    engine.unload()
    print("\nTerminé : le modèle n'a été chargé qu'une fois pour les 3 seuils.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
