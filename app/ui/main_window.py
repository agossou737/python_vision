"""Fenêtre principale : barre supérieure, panneaux, barre d'état."""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Dict

import customtkinter as ctk

from app import __version__
from app.config_manager import AppConfig
from app.ui import theme
from app.ui.camera_panel import CameraPanel, CameraPanelCallbacks
from app.ui.detection_panel import DetectionPanel
from app.ui.statistics_panel import StatisticsPanel
from app.ui.video_panel import VideoPanel


class Indicator(ctk.CTkFrame):
    """Pastille colorée + texte (état caméra / modèle / connexion)."""

    def __init__(self, master: tk.Misc, text: str) -> None:
        super().__init__(master, fg_color="transparent")
        self._dot = ctk.CTkLabel(self, text="●", text_color=theme.MUTED, width=14)
        self._dot.pack(side="left")
        self._label = ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=12))
        self._label.pack(side="left", padx=(2, 0))

    def set(self, text: str, level: str = "idle") -> None:
        self._dot.configure(text_color=theme.LEVEL_COLORS.get(level, theme.MUTED))
        self._label.configure(text=text)


class MainWindow(ctk.CTk):
    """Construit l'interface ; ne contient AUCUNE logique de capture ou d'inférence."""

    def __init__(
        self,
        config: AppConfig,
        callbacks: CameraPanelCallbacks,
        on_settings: Callable[[], None],
    ) -> None:
        super().__init__()
        self.title("Vision Desktop — Détection d'objets en temps réel")
        self.geometry("1380x820")
        self.minsize(1100, 680)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---- Barre supérieure ----
        top = ctk.CTkFrame(self, corner_radius=0, fg_color=theme.BG_BAR, height=48)
        top.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(top, text="Vision Desktop", font=ctk.CTkFont(size=18, weight="bold")).pack(
            side="left", padx=16, pady=10
        )
        ctk.CTkButton(top, text="⚙ Paramètres", width=110, command=on_settings).pack(
            side="right", padx=12
        )
        self._indicators: Dict[str, Indicator] = {}
        for key, text in (
            ("connection", "Connexion : —"),
            ("model", "Modèle : non chargé"),
            ("camera", "Caméra : arrêtée"),
        ):
            ind = Indicator(top, text)
            ind.pack(side="right", padx=12)
            self._indicators[key] = ind

        # ---- Corps ----
        body = ctk.CTkFrame(self, fg_color=theme.BG_PANEL, corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        self.camera_panel = CameraPanel(body, config, callbacks)
        self.camera_panel.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)
        self.camera_panel.configure(width=290)

        center = ctk.CTkFrame(body, fg_color="transparent")
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)
        self.video = VideoPanel(center)
        self.video.grid(row=0, column=0, sticky="nsew")
        self.statistics = StatisticsPanel(center)
        self.statistics.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        self.detections = DetectionPanel(body)
        self.detections.grid(row=0, column=2, sticky="ns", padx=(5, 10), pady=10)
        self.detections.configure(width=370)

        # ---- Barre d'état ----
        bar = ctk.CTkFrame(self, corner_radius=0, fg_color=theme.BG_BAR, height=28)
        bar.grid(row=2, column=0, sticky="ew")
        self._status = ctk.CTkLabel(bar, text="Prêt.", anchor="w")
        self._status.pack(side="left", padx=12, pady=3)
        ctk.CTkLabel(bar, text=f"v{__version__}", text_color=theme.TEXT_MUTED).pack(
            side="right", padx=12
        )

    # ---- API ----
    def set_indicator(self, key: str, text: str, level: str = "idle") -> None:
        self._indicators[key].set(text, level)

    def set_status(self, message: str, level: str = "idle") -> None:
        color = theme.TEXT if level == "idle" else theme.LEVEL_COLORS.get(level, theme.TEXT)
        self._status.configure(text=message, text_color=color)

    def restyle(self) -> None:
        """Recolore les widgets Tk/ttk après un changement de thème."""
        self.video.restyle()
        self.detections.restyle()
