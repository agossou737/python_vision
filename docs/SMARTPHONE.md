# Guide : webcams et smartphones

Vision Desktop lit des images avec OpenCV. Il accepte donc :

1. une **webcam** (intégrée, USB, ou **caméra virtuelle** vue comme une webcam) → type de source « Webcam locale » ;
2. un **flux réseau** HTTP/RTSP → « Flux réseau », ou « Smartphone > Application Wi-Fi » ;
3. un smartphone via **ADB forward** → « Smartphone > ADB forward ».

> Les commandes, options et URL ci-dessous dépendent des versions de scrcpy, d'Android et des
> applications. En cas de doute, la référence est `scrcpy --help` et l'écran de l'application de streaming.

| Méthode | Android | iOS | Dans Vision Desktop |
|---|---|---|---|
| A. scrcpy + caméra virtuelle | oui (USB) | non | Webcam locale (ou Smartphone > Caméra virtuelle) |
| B. ADB forward | oui (USB) | **non** | Smartphone > ADB forward |
| C. Application de streaming (Wi-Fi) | oui | oui (apps tierces) | Flux réseau (ou Smartphone > Application Wi-Fi) |
| Bluetooth | — | — | **Non pris en charge** (voir plus bas) |

---

## 0. Préparer le téléphone Android (méthodes A et B)

1. **Réglages > À propos du téléphone** : touchez 7 fois « Numéro de build » pour activer le mode développeur.
2. **Réglages > Options pour les développeurs** : activez **Débogage USB**.
3. Branchez le câble USB (un vrai câble de données) et acceptez la fenêtre « Autoriser le débogage USB ? ».

### Installer ADB

- **Ubuntu** : `sudo apt install adb scrcpy`
- **Windows** : téléchargez « SDK Platform-Tools » sur developer.android.com, décompressez-le, ajoutez le
  dossier au `PATH`. scrcpy : archive Windows sur la page GitHub du projet (Genymobile/scrcpy).

### Vérifier la connexion

```bash
adb devices
```

| État affiché | Signification | Action |
|---|---|---|
| `device` | prêt | rien |
| `unauthorized` | non autorisé | acceptez la fenêtre sur le téléphone (ou révoquez les autorisations USB puis rebranchez) |
| `offline` / rien | câble, mode USB ou pilote | autre câble/port ; mode « Transfert de fichiers » ; Windows : pilote USB du constructeur |

---

## A. scrcpy et caméra virtuelle

scrcpy affiche (et peut diffuser) l'écran **ou la caméra** du téléphone. Pour que Vision Desktop le voie,
il faut une **caméra virtuelle** que le système présente comme une webcam.

### Ubuntu : v4l2loopback

```bash
sudo apt install v4l2loopback-dkms v4l2loopback-utils
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Smartphone" exclusive_caps=1

# Caméra arrière du téléphone (Android 12+, scrcpy récent) vers la caméra virtuelle :
scrcpy --video-source=camera --camera-facing=back --v4l2-sink=/dev/video10 --no-playback
```

Dans Vision Desktop : « Webcam locale », saisissez ou choisissez **Caméra #10** (le champ est modifiable ;
le scan ⟳ ne teste que 0 à `scan_max_index`, 5 par défaut, réglable jusqu'à 15 dans `settings.json`).
Pour dupliquer l'**écran** du téléphone au lieu de sa caméra, retirez `--video-source=camera`.

### Windows : OBS Virtual Camera

scrcpy n'a pas de sortie « webcam » sous Windows. Méthode courante :

1. Lancez `scrcpy` (écran ou `--video-source=camera`) : une fenêtre s'ouvre.
2. Dans **OBS Studio**, ajoutez une source « Capture de fenêtre » sur cette fenêtre.
3. OBS : **Démarrer la caméra virtuelle**.
4. Dans Vision Desktop : « Webcam locale », ⟳ ; la caméra « OBS Virtual Camera » apparaît à un index (0, 1, 2…).

**Alternative sans scrcpy** : une application type DroidCam installe un pilote de webcam sur le PC.

---

## B. ADB et transfert de ports

**ADB ne fournit pas de flux vidéo par lui-même.** Il sert seulement à créer un tunnel TCP par le câble USB.
Il n'a de sens que si **une application sur le téléphone expose déjà un serveur vidéo** (par exemple IP Webcam,
qui diffuse sur le port 8080 du téléphone). Le tunnel permet alors de s'y connecter par USB, sans Wi-Fi.

### `adb forward` ou `adb reverse` ?

| Commande | Sens | Quand l'utiliser |
|---|---|---|
| `adb forward tcp:8080 tcp:8080` | le **PC** se connecte à un service qui tourne **sur le téléphone** | **Notre cas** : Vision Desktop est un client qui lit le flux du téléphone |
| `adb reverse tcp:5000 tcp:5000` | le **téléphone** se connecte à un service qui tourne **sur le PC** | Pas utilisé ici : concerne une application du téléphone qui enverrait vers un serveur du PC |

(Syntaxe : `adb forward tcp:<port PC> tcp:<port téléphone>` ; `adb reverse tcp:<port téléphone> tcp:<port PC>`.)

### Exemple complet avec IP Webcam

1. Sur le téléphone : ouvrez IP Webcam, **Démarrer le serveur** (port 8080 par défaut).
2. Dans Vision Desktop : **Smartphone > ADB forward (USB…)**, port local `8080`, port téléphone `8080`,
   chemin `/video` → **Tester la source** → **Démarrer**.

Vision Desktop exécute lui-même : vérification d'ADB, vérification que l'appareil est **autorisé**,
`adb forward tcp:8080 tcp:8080`, lecture de `http://127.0.0.1:8080/video`, puis `adb forward --remove tcp:8080`
à l'arrêt. À la main :

```bash
adb forward tcp:8080 tcp:8080      # créer le tunnel
adb forward --list                 # lister
adb forward --remove tcp:8080      # le retirer
```

Si le port local est occupé, choisissez-en un autre (ex. 8081) pour le « port local ».

---

## C. Application de streaming par Wi-Fi

Le téléphone et le PC doivent être sur le **même réseau local**.

1. Installez une application de caméra réseau (ex. **IP Webcam**, **DroidCam**) et démarrez son serveur.
2. L'application affiche l'**adresse** à utiliser (ex. `http://192.168.1.20:8080`).
3. Dans Vision Desktop : **Flux réseau**, collez l'URL du **flux vidéo**, **Tester la source**, **Démarrer**.

URL fréquentes, **à adapter à votre application et à sa version** :

| Application | Exemple d'URL de flux |
|---|---|
| IP Webcam (Android) | `http://192.168.1.20:8080/video` (MJPEG) |
| DroidCam (Android) | `http://192.168.1.20:4747/video` (le chemin varie selon la version) |
| Caméra IP / application RTSP | `rtsp://192.168.1.20:554/stream` |

Si l'URL de l'interface web fonctionne dans un navigateur mais pas ici, c'est que ce n'est pas l'URL du *flux*.

### Réseau et pare-feu

- Vision Desktop **se connecte** au téléphone (connexion sortante) : en général, **aucune règle entrante** n'est
  nécessaire sur le PC. C'est le pare-feu/réseau qui peut bloquer : Wi-Fi « invité » avec isolation des clients,
  VPN actif, hotspot restrictif.
- Ne **redirigez jamais** ces ports vers Internet : un flux de caméra sans mot de passe est public. Activez
  l'authentification de l'application si elle existe.
- Windows : si une alerte de pare-feu s'affiche pour Python, autorisez **réseau privé** uniquement.
- Un flux perdu est détecté ; la reconnexion automatique réessaie avec un délai croissant (⚙ Paramètres).

---

## iOS (iPhone / iPad)

- **ADB n'existe pas pour iOS** : les méthodes A et B ne s'appliquent pas.
- Il n'y a **pas d'accès direct** à la caméra d'un iPhone depuis un PC Windows/Linux.
- Solutions possibles :
  - **Application de streaming** qui expose un flux MJPEG/RTSP sur le réseau local → méthode C ;
  - **Application avec pilote PC** (ex. Camo, EpocCam, iVCam : disponibilité, version et licence à vérifier) qui
    crée une webcam virtuelle → « Webcam locale » ;
  - **Mac** : la « Caméra de continuité » d'Apple expose l'iPhone comme webcam du Mac.

## Bluetooth

Le Bluetooth **seul ne fournit pas de flux vidéo exploitable par OpenCV** : il n'existe pas de « webcam
Bluetooth » lisible par `cv2.VideoCapture`. Le Bluetooth peut servir au contrôle, mais l'image doit passer
par **USB** (méthodes A/B) ou **Wi-Fi** (méthode C). Vision Desktop l'indique explicitement (« Non pris en
charge ») au lieu de simuler une connexion.

---

## Dépannage

| Symptôme | Piste |
|---|---|
| « Caméra #N indisponible » | Un autre programme l'utilise (visioconférence, navigateur) ; Linux : groupe `video` ; autre index |
| La caméra virtuelle n'apparaît pas | Linux : `v4l2loopback` chargé ? (`ls /dev/video*`) ; Windows : caméra virtuelle d'OBS démarrée ? |
| « Appareil détecté mais non autorisé » | Acceptez la fenêtre de débogage USB sur le téléphone |
| « ADB est introuvable » | Ajoutez les platform-tools au `PATH`, rouvrez le terminal |
| « Impossible de rediriger le port ADB » | Port local déjà utilisé : changez-le |
| Test OK mais flux saccadé/lent | Réduisez la résolution dans l'application émettrice ; Wi-Fi 5 GHz ; augmentez « images ignorées » |
| « Signal vidéo perdu » en boucle | Téléphone en veille/économie d'énergie (désactivez l'optimisation de batterie pour l'application) |
