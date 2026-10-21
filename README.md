# Vision Desktop

Application desktop de **détection d'objets en temps réel** : webcam, flux réseau ou smartphone,
analyse avec Ultralytics YOLO (ou Faster R-CNN), interface CustomTkinter.

- Interface sombre/claire, vidéo avec boîtes, labels et scores, tableau des détections, statistiques.
- Seuil de confiance, résolution, images ignorées et classes affichées **modifiables à chaud**.
- Capture et inférence dans des threads séparés ; files bornées (latence maîtrisée) ; arrêt propre.
- Perte de signal détectée, reconnexion automatique à délai progressif.
- Windows et Linux (Ubuntu).

> **Ce qui a été vérifié, et ce qui ne l'a pas été.** La logique (capture, reconnexion, files, moteur,
> interface, arrêt) est couverte par 52 tests automatiques avec des sources et un détecteur simulés,
> et l'interface a été lancée sous écran virtuel Linux. **Non vérifiés par l'auteur** : le chargement
> réel de YOLO/Faster R-CNN, une vraie webcam, Windows, Python 3.14. Voir `docs/VALIDATION.md`.

---

## 1. Architecture

```
 Source vidéo (webcam / réseau / smartphone)           interface : thread principal uniquement
        │  VideoSource.read()                                   ▲
        ▼                                                       │ root.after (15 ms)
 VideoCaptureWorker ──LatestQueue(2)──► InferenceWorker ──LatestQueue(2)──► MainWindow
   (thread, reconnexion)                  (thread)                          (Canvas + ImageTk)
        │                                    │ DetectionEngine
        │                                    ▼
        │                         BaseDetector : YoloDetector | FasterRcnnDetector
        └── événements (queue.Queue) ───────────────────────────────────► barre d'état, indicateurs
```

Principes :

- **Une seule source de vérité par composant** : la capture ne connaît ni le modèle ni l'interface ;
  le moteur ne connaît ni la caméra ni l'interface.
- **`LatestQueue`** : file bornée qui écarte les images les plus anciennes ; l'inférence prend toujours
  la plus récente. Si le modèle est plus lent que la caméra, des images sont ignorées, pas accumulées.
- **Le thread principal est le seul à toucher Tkinter.** Les threads (scan, test, chargement du modèle)
  renvoient leur résultat via une file consommée par `root.after`.
- **Aucun résultat inventé** : sans modèle prêt ou après une erreur d'inférence, la vidéo continue
  mais sans détection.

## 2. Arborescence et rôle des fichiers

```
vision_desktop/
├── main.py                       Point d'entrée (vérifie Tkinter et les dépendances)
├── requirements.txt              Dépendances (obligatoires / optionnelles)
├── pytest.ini                    Configuration des tests
├── config/settings.json          Configuration utilisateur (créée au premier lancement)
├── assets/models/                Poids des modèles (ignorés par Git)
├── assets/icons/                 Réservé aux icônes
├── logs/                         app.log (rotation, 2 Mo x 5)
├── docs/
│   ├── SMARTPHONE.md             Guide webcams et smartphones (Windows / Ubuntu / iOS / Bluetooth)
│   └── VALIDATION.md             Scénarios de validation et leur couverture
├── tools/
│   ├── smoke_yolo.py             Vérifie YOLO de bout en bout sur votre machine
│   └── build_exe.py              Génère l'exécutable PyInstaller
├── app/
│   ├── application.py            Contrôleur : relie UI, capture, inférence ; démarrage/arrêt non bloquants
│   ├── config_manager.py         Chargement/sauvegarde JSON tolérants, bornes de sécurité
│   ├── logger.py                 Journalisation avec rotation
│   ├── events.py                 WorkerEvent / EventKind (workers → interface)
│   ├── paths.py                  Dossiers modifiables (normal ou exécutable gelé)
│   ├── video/
│   │   ├── base_source.py        VideoSource (interface), erreurs, base OpenCV commune
│   │   ├── local_camera.py       Webcams par index + scan non bloquant
│   │   ├── network_camera.py     Flux HTTP/RTSP (FFmpeg), délais configurables
│   │   ├── smartphone_source.py  Caméra virtuelle, application IP, ADB forward ; Bluetooth/iOS = non pris en charge
│   │   ├── camera_manager.py     Cycle de vie : un seul worker actif, test de source
│   │   ├── video_capture_worker.py  Thread de capture, perte de signal, reconnexion
│   │   └── frame.py              FramePacket (image + métadonnées)
│   ├── detection/
│   │   ├── types.py              Detection, DetectionResult
│   │   ├── base_detector.py      Interface commune des modèles, erreurs, choix du périphérique
│   │   ├── yolo_detector.py      Ultralytics YOLO
│   │   ├── faster_rcnn_detector.py  Faster R-CNN (torchvision)
│   │   └── detection_engine.py   Chargement unique, réglages à chaud, filtre de classes
│   ├── workers/inference_worker.py  Thread d'inférence, frame_skip, FPS et latence
│   ├── ui/                       main_window, camera_panel, video_panel, detection_panel,
│   │                             statistics_panel, settings_dialog, theme
│   └── utils/                    image_utils, latest_queue, metrics, validators
└── tests/                        52 tests (sources simulées dans tests/fakes.py)
```

## 3. Installation

Pré-requis : **Python 3.10 à 3.13** (3.12 recommandé). Python 3.14 n'est pas validé.

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install python3 python3-venv python3-tk
cd vision_desktop
python3 -m venv .venv
source .venv/bin/activate

# Sans GPU NVIDIA : PyTorch CPU d'abord (évite des Go de bibliothèques CUDA inutiles)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python -c "import tkinter; print('Tkinter OK', tkinter.TkVersion)"
python main.py
```

Accès à la webcam : `sudo usermod -aG video $USER`, puis se reconnecter.

### Windows (PowerShell)

Installez Python depuis python.org (Tkinter est inclus par défaut : ne décochez pas « tcl/tk and IDLE »).

```powershell
cd vision_desktop
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # si bloqué : Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Sans GPU NVIDIA : PyTorch CPU d'abord
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python -c "import tkinter; print('Tkinter OK', tkinter.TkVersion)"
python main.py
```

Si le dossier du projet contient des espaces, placez le chemin entre guillemets dans vos commandes.

### PyTorch : CPU ou GPU

| Situation | Commande |
|---|---|
| **CPU seul** (aucun GPU NVIDIA) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| **GPU NVIDIA** | Générez la commande exacte sur https://pytorch.org (choix de la version CUDA selon votre pilote), du type `pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX` |
| **Mac Apple Silicon** | `pip install torch torchvision` ; l'accélération `mps` est détectée automatiquement |

À savoir : sous Linux, `pip install torch` depuis PyPI embarque CUDA (plusieurs Go) ; sous Windows, le
paquet PyPI est une version CPU. CUDA n'est **jamais** imposé : `device = "auto"` choisit `cuda:0`, `mps`
ou `cpu` selon ce qui est réellement disponible, et repasse sur le CPU si le GPU échoue.
Vérification : `python -c "import torch; print(torch.cuda.is_available())"`.

### Tkinter : dépannage

| Message | Cause | Solution |
|---|---|---|
| `Tkinter est introuvable` | Python sans Tkinter | Ubuntu : `sudo apt install python3-tk` (ou `python3.X-tk`). Le message affiche le Python utilisé et l'erreur réelle. |
| Idem après l'installation | Environnement virtuel créé avant, ou Python hors système (pyenv, conda, deadsnakes, 3.14) | Recréer le venv avec `/usr/bin/python3`, ou installer le paquet Tk de ce Python (pyenv : `tk-dev` puis réinstaller ; conda : `conda install tk`) |

## 4. Modèles

- Premier lancement : le modèle choisi (`yolov8n.pt` par défaut) est **téléchargé dans `assets/models/`**
  (connexion requise). Un échec est signalé dans la barre d'état ; l'application reste utilisable.
- Hors ligne : téléchargez le fichier `.pt` ailleurs et copiez-le dans `assets/models/`.
- Faster R-CNN (`fasterrcnn_mobilenet_v3_large_320_fpn`, `fasterrcnn_resnet50_fpn_v2`, …) : poids COCO
  téléchargés par torchvision dans `assets/models/torch/`. Sans GPU, préférez la variante `mobilenet_320`.
- Les poids ne sont jamais versionnés (`.gitignore`).
- Vérification rapide : `python tools/smoke_yolo.py chemin/image.jpg`.

## 5. Utilisation

1. **Source** : Webcam locale (bouton ⟳ pour détecter les caméras), Flux réseau (URL) ou Smartphone.
   Voir `docs/SMARTPHONE.md` pour obtenir une URL de flux ou une caméra virtuelle.
2. **Tester la source** (facultatif), puis **▶ Démarrer**. Pendant l'analyse, le bouton devient
   « Redémarrer » : choisir une autre source puis cliquer dessus libère l'ancienne avant d'ouvrir la nouvelle.
3. **Réglages à chaud** : seuil de confiance, résolution d'inférence, images ignorées, boîtes/labels,
   classes affichées (le filtre agit à l'affichage ; l'inférence reste calculée pour toutes les classes).
4. **Changer de modèle** : sélectionner ou saisir un nom puis Entrée. En cas d'échec, l'ancien modèle reste actif.
5. **⚙ Paramètres** : thème, appareil d'inférence, délais et tentatives de reconnexion, taille de file.

Exemples d'URL (non universels, voir l'application émettrice) :
`http://192.168.1.20:8080/video` · `rtsp://192.168.1.20:554/stream`.

## 6. Configuration (`config/settings.json`)

Fichier JSON tolérant : clés inconnues ignorées, types invalides remplacés par les défauts, valeurs
ramenées dans des bornes sûres, fichier corrompu sauvegardé en `.bak`.

| Section | Clé | Défaut | Rôle |
|---|---|---|---|
| video | `connect_timeout_s` | 5.0 | Délai de connexion aux flux réseau |
| video | `auto_reconnect` | true | Reconnexion automatique |
| video | `reconnect_initial_delay_s` / `reconnect_max_delay_s` | 1.0 / 15.0 | Délai progressif (x2 à chaque échec, plafonné) |
| video | `reconnect_max_attempts` | 0 | 0 = illimité |
| video | `max_consecutive_read_failures` | 30 | Lectures ratées avant « signal perdu » |
| video | `frame_queue_size` | 2 | Taille de la file d'images (appliquée au démarrage) |
| video | `scan_max_index` | 5 | Dernier index testé par le scan (max 15) |
| detection | `model_name` | yolov8n.pt | Modèle |
| detection | `confidence` / `imgsz` / `frame_skip` | 0.5 / 640 / 0 | Réglages à chaud |
| detection | `device` | auto | `auto`, `cpu`, `cuda:0` |
| appearance | `theme` | dark | `dark`, `light`, `system` |

## 7. Tests

```bash
python -m pytest                      # tous les tests (ceux de l'interface sont ignorés sans Tkinter/écran)
xvfb-run -a python -m pytest          # Linux sans écran (sudo apt install xvfb)
```

Les tests utilisent des sources et détecteurs simulés (`tests/fakes.py`) : aucune caméra ni GPU requis.

## 8. Exécutable (PyInstaller)

```bash
pip install pyinstaller
python tools/build_exe.py              # ajoutez --console pour diagnostiquer un problème
```

Résultat : `dist/VisionDesktop/` (mode « dossier », 250 Mo sans PyTorch ; bien plus avec). `config/`,
`logs/` et `assets/models/` sont créés **à côté** de l'exécutable, modifiables par l'utilisateur.
Le build est à faire sur chaque système cible (un exécutable Windows se construit sous Windows).
Construit et lancé avec succès sous Linux **sans** Ultralytics ni PyTorch ; le build complet
(avec modèles) est à valider sur votre machine, et peut demander des options supplémentaires
(`--collect-all torch`) selon vos versions.

## 9. Limites connues

- **Lecture bloquante** : OpenCV ne peut pas interrompre un `read()` bloqué (câble débranché en pleine
  lecture, réseau figé). Le thread est « daemon » : l'arrêt signale l'échec après un délai
  (4 s) au lieu de geler l'interface, mais la ressource peut rester occupée jusqu'à la fin du système.
- **Une seule source à la fois** (l'architecture permet d'en ajouter, voir optimisations).
- **Scan des caméras** : il ouvre brièvement chaque index ; il est désactivé pendant une analyse.
  Sous Windows, les caméras sont identifiées par index, pas par nom.
- **Réseau** : les délais FFmpeg dépendent de la version d'OpenCV ; l'option RTSP « TCP » est fixée au
  premier flux RTSP du processus.
- **Latence de rendu** : le dessin des boîtes est fait dans le thread principal ; au-delà de la 1080p
  il peut devenir significatif (réduire la fenêtre ou la cadence).
- **Faster R-CNN** : `imgsz` correspond au côté court interne du modèle ; beaucoup plus lent que YOLO sur CPU.
- **Fonctions non incluses** : enregistrement vidéo, historique des détections, suivi d'objets,
  multi-caméras, Bluetooth, accès direct à la caméra d'un iPhone (cf. `docs/SMARTPHONE.md`).

## 10. Optimisations possibles

- **Modèle** : export ONNX / OpenVINO (CPU Intel) ou TensorRT via Ultralytics ; demi-précision sur GPU ;
  `imgsz` plus petit ; `frame_skip` pour garder une interface fluide.
- **Multi-caméras** : une instance de `CameraManager` + `InferenceWorker` par source (files séparées).
- **Enregistrement** : worker dédié avec `cv2.VideoWriter` alimenté par une file bornée.
- **Suivi d'objets** : `model.track()` (ByteTrack) dans `YoloDetector`, identifiant ajouté à `Detection`.
- **Historique** : worker d'écriture SQLite, découplé de l'affichage.
- **Charge CPU** : processus séparé pour l'inférence (`multiprocessing`) si le GIL devient limitant.
