"""Gestion de la configuration (JSON) avec valeurs par défaut et validation."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar, Union

from app.utils.validators import clamp

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"

T = TypeVar("T")


@dataclass
class AppearanceSettings:
    theme: str = "dark"            # dark | light | system
    color_theme: str = "blue"      # blue | green | dark-blue


@dataclass
class VideoSettings:
    source_type: str = "local"     # local | network | smartphone
    camera_index: int = 0
    stream_url: str = ""
    connect_timeout_s: float = 5.0
    auto_reconnect: bool = True
    reconnect_initial_delay_s: float = 1.0
    reconnect_max_delay_s: float = 15.0
    reconnect_max_attempts: int = 0          # 0 = illimité
    max_consecutive_read_failures: int = 30  # échecs de lecture avant « signal perdu »
    frame_queue_size: int = 2
    scan_max_index: int = 5
    # Smartphone : virtual_camera | ip_app | adb_forward
    smartphone_method: str = "ip_app"
    adb_local_port: int = 8080
    adb_remote_port: int = 8080
    adb_stream_path: str = "/video"


@dataclass
class DetectionSettings:
    model_name: str = "yolov8n.pt"
    confidence: float = 0.5
    imgsz: int = 640
    frame_skip: int = 0            # images ignorées entre deux inférences
    enabled_classes: List[str] = field(default_factory=list)  # vide = toutes
    show_boxes: bool = True
    show_labels: bool = True
    device: str = "auto"           # auto | cpu | cuda:0 ...


@dataclass
class PathSettings:
    models_dir: str = "assets/models"
    logs_dir: str = "logs"


@dataclass
class AppConfig:
    appearance: AppearanceSettings = field(default_factory=AppearanceSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    paths: PathSettings = field(default_factory=PathSettings)

    def validate(self) -> None:
        """Ramène les valeurs dans des bornes sûres (appelé après chargement)."""
        v, d = self.video, self.detection
        if v.source_type not in ("local", "network", "smartphone"):
            v.source_type = "local"
        if v.smartphone_method not in ("virtual_camera", "ip_app", "adb_forward"):
            v.smartphone_method = "ip_app"
        v.camera_index = int(clamp(v.camera_index, 0, 63))
        v.connect_timeout_s = clamp(v.connect_timeout_s, 1.0, 60.0)
        v.reconnect_initial_delay_s = clamp(v.reconnect_initial_delay_s, 0.1, 60.0)
        v.reconnect_max_delay_s = clamp(
            v.reconnect_max_delay_s, v.reconnect_initial_delay_s, 300.0
        )
        v.reconnect_max_attempts = int(clamp(v.reconnect_max_attempts, 0, 10_000))
        v.max_consecutive_read_failures = int(clamp(v.max_consecutive_read_failures, 1, 1000))
        v.frame_queue_size = int(clamp(v.frame_queue_size, 1, 16))
        v.scan_max_index = int(clamp(v.scan_max_index, 0, 15))
        v.adb_local_port = int(clamp(v.adb_local_port, 1, 65535))
        v.adb_remote_port = int(clamp(v.adb_remote_port, 1, 65535))
        if not v.adb_stream_path.startswith("/"):
            v.adb_stream_path = "/" + v.adb_stream_path

        d.confidence = clamp(d.confidence, 0.05, 0.95)
        d.imgsz = int(clamp(d.imgsz, 160, 1920))
        d.frame_skip = int(clamp(d.frame_skip, 0, 30))
        if self.appearance.theme not in ("dark", "light", "system"):
            self.appearance.theme = "dark"


def _build(cls: Type[T], data: Any) -> T:
    """Construit un dataclass depuis un dict en ignorant clés inconnues et types invalides."""
    obj = cls()  # type: ignore[call-arg]
    if not isinstance(data, dict):
        return obj
    for f in fields(obj):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        default = getattr(obj, f.name)
        value = data[f.name]
        if hasattr(default, "__dataclass_fields__"):
            setattr(obj, f.name, _build(type(default), value))
        elif isinstance(default, bool):
            if isinstance(value, bool):
                setattr(obj, f.name, value)
        elif isinstance(default, int):
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(obj, f.name, value)
        elif isinstance(default, float):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(obj, f.name, float(value))
        elif isinstance(default, str):
            if isinstance(value, str):
                setattr(obj, f.name, value)
        elif isinstance(default, list):
            if isinstance(value, list):
                setattr(obj, f.name, [str(x) for x in value])
    return obj


class ConfigManager:
    """Charge / sauvegarde la configuration JSON de manière tolérante et atomique."""

    def __init__(self, path: Union[str, Path, None] = None) -> None:
        self.path = Path(path) if path else DEFAULT_CONFIG_PATH
        self.config = AppConfig()
        self._lock = threading.RLock()

    def load(self) -> AppConfig:
        """Charge le fichier. Fichier absent ou corrompu -> défauts (corrompu sauvegardé en .bak)."""
        with self._lock:
            if not self.path.exists():
                logger.info("Configuration absente, création des valeurs par défaut : %s", self.path)
                self.config = AppConfig()
                self.config.validate()
                self.save()
                return self.config
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.config = _build(AppConfig, raw)
            except (OSError, ValueError) as exc:
                logger.error("Configuration illisible (%s). Valeurs par défaut utilisées.", exc)
                self._backup_corrupted()
                self.config = AppConfig()
            self.config.validate()
            return self.config

    def save(self) -> bool:
        """Écriture atomique (fichier temporaire + remplacement)."""
        with self._lock:
            self.config.validate()
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(
                    json.dumps(asdict(self.config), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                os.replace(tmp, self.path)
                return True
            except OSError as exc:
                logger.error("Échec de sauvegarde de la configuration : %s", exc)
                return False

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return asdict(self.config)

    def _backup_corrupted(self) -> None:
        try:
            os.replace(self.path, self.path.with_suffix(self.path.suffix + ".bak"))
        except OSError:
            pass
