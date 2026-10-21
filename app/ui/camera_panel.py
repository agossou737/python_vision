"""Panneau latéral de configuration : source, modèle, réglages de détection."""

from __future__ import annotations

import re
import tkinter as tk
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import customtkinter as ctk

from app.config_manager import AppConfig
from app.ui import theme
from app.video.local_camera import CameraInfo


def _noop(*_args: Any) -> None:
    return None


@dataclass
class CameraPanelCallbacks:
    """Actions déclenchées par le panneau (câblées par ``Application``)."""

    on_start: Callable[[], None] = _noop
    on_stop: Callable[[], None] = _noop
    on_test: Callable[[], None] = _noop
    on_scan: Callable[[], None] = _noop
    on_confidence: Callable[[float], None] = _noop
    on_model: Callable[[str], None] = _noop
    on_imgsz: Callable[[int], None] = _noop
    on_frame_skip: Callable[[int], None] = _noop
    on_display_options: Callable[[bool, bool], None] = _noop
    on_classes: Callable[[], None] = _noop


SOURCE_TYPES: Dict[str, str] = {
    "Webcam locale (USB / intégrée)": "local",
    "Flux réseau (HTTP / RTSP)": "network",
    "Smartphone": "smartphone",
}
SMARTPHONE_METHODS: Dict[str, str] = {
    "Caméra virtuelle (scrcpy / DroidCam USB)": "virtual_camera",
    "Application Wi-Fi (IP Webcam / DroidCam)": "ip_app",
    "ADB forward (USB, service vidéo requis)": "adb_forward",
}
MODEL_CHOICES = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"]
IMGSZ_CHOICES = ["320", "416", "640", "960", "1280"]


def _key_for(mapping: Dict[str, str], value: str) -> str:
    return next((k for k, v in mapping.items() if v == value), next(iter(mapping)))


class CameraPanel(ctk.CTkScrollableFrame):
    def __init__(self, master: tk.Misc, config: AppConfig, callbacks: CameraPanelCallbacks) -> None:
        super().__init__(master, corner_radius=10, fg_color=theme.BG_CARD)
        self._cb = callbacks
        self._config = config
        self._cameras: List[CameraInfo] = []
        self._running = False

        v, d = config.video, config.detection

        # ---------------- Source ----------------
        self._title("Source vidéo")
        self._type_menu = ctk.CTkOptionMenu(
            self, values=list(SOURCE_TYPES), command=lambda _v: self._refresh_visibility()
        )
        self._type_menu.set(_key_for(SOURCE_TYPES, v.source_type))
        self._type_menu.pack(fill="x", padx=10, pady=(0, 6))

        self._params = ctk.CTkFrame(self, fg_color="transparent")
        self._params.pack(fill="x")

        self._method_row = ctk.CTkFrame(self._params, fg_color="transparent")
        self._method_menu = ctk.CTkOptionMenu(
            self._method_row, values=list(SMARTPHONE_METHODS),
            command=lambda _v: self._refresh_visibility(),
        )
        self._method_menu.set(_key_for(SMARTPHONE_METHODS, v.smartphone_method))
        self._method_menu.pack(fill="x", padx=10, pady=2)

        self._cam_row = ctk.CTkFrame(self._params, fg_color="transparent")
        self._cam_combo = ctk.CTkComboBox(self._cam_row, values=[f"Caméra #{v.camera_index}"])
        self._cam_combo.set(f"Caméra #{v.camera_index}")
        self._cam_combo.pack(side="left", fill="x", expand=True, padx=(10, 4), pady=2)
        self._scan_btn = ctk.CTkButton(
            self._cam_row, text="⟳", width=34, command=lambda: self._cb.on_scan()
        )
        self._scan_btn.pack(side="right", padx=(0, 10), pady=2)

        self._url_row = ctk.CTkFrame(self._params, fg_color="transparent")
        self._url_entry = ctk.CTkEntry(
            self._url_row, placeholder_text="http://192.168.1.20:8080/video"
        )
        if v.stream_url:
            self._url_entry.insert(0, v.stream_url)
        self._url_entry.pack(fill="x", padx=10, pady=2)

        self._adb_row = ctk.CTkFrame(self._params, fg_color="transparent")
        self._adb_local = self._labeled_entry(self._adb_row, "Port local", str(v.adb_local_port))
        self._adb_remote = self._labeled_entry(self._adb_row, "Port téléphone", str(v.adb_remote_port))
        self._adb_path = self._labeled_entry(self._adb_row, "Chemin du flux", v.adb_stream_path)

        self._test_btn = ctk.CTkButton(
            self, text="Tester la source", fg_color="transparent", border_width=1,
            text_color=theme.TEXT, command=lambda: self._cb.on_test(),
        )
        self._test_btn.pack(fill="x", padx=10, pady=(6, 4))

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=10, pady=(4, 8))
        buttons.grid_columnconfigure((0, 1), weight=1)
        self._start_btn = ctk.CTkButton(
            buttons, text="▶ Démarrer", fg_color=theme.OK, hover_color="#16a34a",
            text_color="#08130b", command=lambda: self._cb.on_start(),
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._stop_btn = ctk.CTkButton(
            buttons, text="■ Arrêter", fg_color=theme.ERROR, hover_color="#dc2626",
            command=lambda: self._cb.on_stop(), state="disabled",
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # ---------------- Détection ----------------
        self._title("Modèle de détection")
        self._model_combo = ctk.CTkComboBox(
            self, values=MODEL_CHOICES, command=lambda name: self._cb.on_model(name.strip())
        )
        self._model_combo.set(d.model_name)
        self._model_combo.bind("<Return>", lambda _e: self._cb.on_model(self._model_combo.get().strip()))
        self._model_combo.pack(fill="x", padx=10, pady=(0, 2))
        ctk.CTkLabel(
            self, text="Entrée ou sélection pour charger", text_color=theme.TEXT_MUTED,
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", padx=12)

        self._conf_label = ctk.CTkLabel(self, text=f"Seuil de confiance : {d.confidence:.2f}")
        self._conf_label.pack(anchor="w", padx=12, pady=(10, 0))
        self._conf_slider = ctk.CTkSlider(
            self, from_=0.05, to=0.95, number_of_steps=90, command=self._on_conf
        )
        self._conf_slider.set(d.confidence)
        self._conf_slider.pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkLabel(self, text="Résolution d'inférence").pack(anchor="w", padx=12, pady=(6, 0))
        self._imgsz_menu = ctk.CTkOptionMenu(
            self, values=IMGSZ_CHOICES, command=lambda v_: self._cb.on_imgsz(int(v_))
        )
        self._imgsz_menu.set(str(d.imgsz) if str(d.imgsz) in IMGSZ_CHOICES else "640")
        self._imgsz_menu.pack(fill="x", padx=10, pady=(0, 4))

        self._skip_label = ctk.CTkLabel(self, text=f"Images ignorées entre 2 analyses : {d.frame_skip}")
        self._skip_label.pack(anchor="w", padx=12, pady=(6, 0))
        self._skip_slider = ctk.CTkSlider(
            self, from_=0, to=10, number_of_steps=10, command=self._on_skip
        )
        self._skip_slider.set(d.frame_skip)
        self._skip_slider.pack(fill="x", padx=10, pady=(0, 4))

        # ---------------- Affichage ----------------
        self._title("Affichage")
        self._boxes_var = tk.BooleanVar(value=d.show_boxes)
        self._labels_var = tk.BooleanVar(value=d.show_labels)
        ctk.CTkSwitch(
            self, text="Boîtes englobantes", variable=self._boxes_var, command=self._on_display
        ).pack(anchor="w", padx=12, pady=3)
        ctk.CTkSwitch(
            self, text="Labels et scores", variable=self._labels_var, command=self._on_display
        ).pack(anchor="w", padx=12, pady=3)
        self._classes_btn = ctk.CTkButton(
            self, text="Classes affichées…", fg_color="transparent", border_width=1,
            text_color=theme.TEXT, command=lambda: self._cb.on_classes(),
        )
        self._classes_btn.pack(fill="x", padx=10, pady=(8, 10))

        self._refresh_visibility()

    # ---- Construction ------------------------------------------------------------
    def _title(self, text: str) -> None:
        ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=10, pady=(10, 4)
        )

    @staticmethod
    def _labeled_entry(parent: ctk.CTkFrame, label: str, value: str) -> ctk.CTkEntry:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(row)
        entry.insert(0, value)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    def _refresh_visibility(self) -> None:
        """Affiche uniquement les champs utiles pour le type de source choisi."""
        for row in (self._method_row, self._cam_row, self._url_row, self._adb_row):
            row.pack_forget()
        source = SOURCE_TYPES[self._type_menu.get()]
        if source == "local":
            self._cam_row.pack(fill="x")
        elif source == "network":
            self._url_row.pack(fill="x")
        else:
            self._method_row.pack(fill="x")
            method = SMARTPHONE_METHODS[self._method_menu.get()]
            if method == "virtual_camera":
                self._cam_row.pack(fill="x")
            elif method == "ip_app":
                self._url_row.pack(fill="x")
            else:
                self._adb_row.pack(fill="x")

    # ---- Événements internes -------------------------------------------------------
    def _on_conf(self, value: float) -> None:
        value = round(float(value), 2)
        self._conf_label.configure(text=f"Seuil de confiance : {value:.2f}")
        self._cb.on_confidence(value)

    def _on_skip(self, value: float) -> None:
        n = int(round(value))
        self._skip_label.configure(text=f"Images ignorées entre 2 analyses : {n}")
        self._cb.on_frame_skip(n)

    def _on_display(self) -> None:
        self._cb.on_display_options(self._boxes_var.get(), self._labels_var.get())

    # ---- API pour Application ------------------------------------------------------
    def get_source_params(self) -> Dict[str, Any]:
        """Valeurs saisies ; ``camera_index`` et ports valent -1 si illisibles."""
        match = re.search(r"\d+", self._cam_combo.get())
        return {
            "source_type": SOURCE_TYPES[self._type_menu.get()],
            "smartphone_method": SMARTPHONE_METHODS[self._method_menu.get()],
            "camera_index": int(match.group()) if match else -1,
            "stream_url": self._url_entry.get().strip(),
            "adb_local_port": self._to_int(self._adb_local.get()),
            "adb_remote_port": self._to_int(self._adb_remote.get()),
            "adb_stream_path": self._adb_path.get().strip() or "/video",
        }

    @staticmethod
    def _to_int(text: str) -> int:
        try:
            return int(text.strip())
        except ValueError:
            return -1

    def set_cameras(self, cameras: Sequence[CameraInfo]) -> None:
        self._cameras = list(cameras)
        if not self._cameras:
            return
        labels = [c.name for c in self._cameras]
        current = re.search(r"\d+", self._cam_combo.get())
        self._cam_combo.configure(values=labels)
        keep = next((c.name for c in self._cameras if current and c.index == int(current.group())), labels[0])
        self._cam_combo.set(keep)

    def set_running(self, running: bool) -> None:
        self._running = running
        self._start_btn.configure(text="⟳ Redémarrer" if running else "▶ Démarrer")
        self._stop_btn.configure(state="normal" if running else "disabled")

    def set_busy_test(self, busy: bool) -> None:
        self._test_btn.configure(
            state="disabled" if busy else "normal",
            text="Test en cours…" if busy else "Tester la source",
        )

    def set_scan_busy(self, busy: bool) -> None:
        self._scan_btn.configure(state="disabled" if busy else "normal")

    def set_model_text(self, name: str) -> None:
        self._model_combo.set(name)

    @property
    def model_text(self) -> str:
        return self._model_combo.get().strip()
