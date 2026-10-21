"""Boîtes de dialogue : paramètres généraux et filtre de classes."""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Dict, List, Sequence

import customtkinter as ctk

from app.config_manager import AppConfig
from app.ui import theme


class _Dialog(ctk.CTkToplevel):
    def __init__(self, master: tk.Misc, title: str, width: int, height: int) -> None:
        super().__init__(master)
        self.title(title)
        self.geometry(f"{width}x{height}")
        self.transient(master)  # type: ignore[arg-type]
        self.after(150, self._grab)

    def _grab(self) -> None:
        try:
            self.grab_set()
            self.focus_set()
        except tk.TclError:
            pass  # fenêtre fermée entre-temps


class SettingsDialog(_Dialog):
    """Paramètres généraux. ``on_save`` est appelé après écriture dans ``config``."""

    def __init__(self, master: tk.Misc, config: AppConfig, on_save: Callable[[], None]) -> None:
        super().__init__(master, "Paramètres", 440, 560)
        self._config = config
        self._on_save = on_save
        v, d, a = config.video, config.detection, config.appearance

        body = ctk.CTkScrollableFrame(self)
        body.pack(fill="both", expand=True, padx=12, pady=(12, 4))

        self._theme = self._menu(body, "Thème", ["dark", "light", "system"], a.theme)
        self._device = self._menu(
            body, "Appareil d'inférence (au prochain chargement)",
            ["auto", "cpu", "cuda:0"], d.device,
        )
        self._timeout = self._entry(body, "Délai de connexion (s)", v.connect_timeout_s)
        self._auto = tk.BooleanVar(value=v.auto_reconnect)
        ctk.CTkSwitch(body, text="Reconnexion automatique", variable=self._auto).pack(
            anchor="w", padx=6, pady=8
        )
        self._delay0 = self._entry(body, "Reconnexion : délai initial (s)", v.reconnect_initial_delay_s)
        self._delay1 = self._entry(body, "Reconnexion : délai maximal (s)", v.reconnect_max_delay_s)
        self._attempts = self._entry(body, "Tentatives max (0 = illimité)", v.reconnect_max_attempts)
        self._failures = self._entry(
            body, "Lectures échouées avant « signal perdu »", v.max_consecutive_read_failures
        )
        self._queue = self._entry(body, "Taille de la file d'images (au prochain démarrage)", v.frame_queue_size)

        self._error = ctk.CTkLabel(self, text="", text_color=theme.ERROR)
        self._error.pack(pady=(2, 0))
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=12, pady=10)
        ctk.CTkButton(bar, text="Annuler", fg_color="transparent", border_width=1,
                      text_color=theme.TEXT, command=self.destroy).pack(side="right", padx=(6, 0))
        ctk.CTkButton(bar, text="Enregistrer", command=self._save).pack(side="right")

    @staticmethod
    def _menu(parent: ctk.CTkFrame, label: str, values: List[str], current: str) -> ctk.CTkOptionMenu:
        ctk.CTkLabel(parent, text=label, anchor="w").pack(fill="x", padx=6, pady=(8, 0))
        menu = ctk.CTkOptionMenu(parent, values=values)
        menu.set(current if current in values else values[0])
        menu.pack(fill="x", padx=6)
        return menu

    @staticmethod
    def _entry(parent: ctk.CTkFrame, label: str, value: object) -> ctk.CTkEntry:
        ctk.CTkLabel(parent, text=label, anchor="w").pack(fill="x", padx=6, pady=(8, 0))
        entry = ctk.CTkEntry(parent)
        entry.insert(0, str(value))
        entry.pack(fill="x", padx=6)
        return entry

    def _save(self) -> None:
        try:
            timeout = float(self._timeout.get().replace(",", "."))
            d0 = float(self._delay0.get().replace(",", "."))
            d1 = float(self._delay1.get().replace(",", "."))
            attempts = int(self._attempts.get())
            failures = int(self._failures.get())
            queue_size = int(self._queue.get())
        except ValueError:
            self._error.configure(text="Valeur numérique invalide.")
            return
        v = self._config.video
        v.connect_timeout_s, v.reconnect_initial_delay_s, v.reconnect_max_delay_s = timeout, d0, d1
        v.reconnect_max_attempts, v.max_consecutive_read_failures = attempts, failures
        v.frame_queue_size = queue_size
        v.auto_reconnect = self._auto.get()
        self._config.detection.device = self._device.get()
        self._config.appearance.theme = self._theme.get()
        self._config.validate()  # ramène dans les bornes
        self._on_save()
        self.destroy()


class ClassFilterDialog(_Dialog):
    """Choix des catégories affichées. Aucune case cochée = toutes les classes."""

    def __init__(
        self,
        master: tk.Misc,
        class_names: Sequence[str],
        enabled: Sequence[str],
        on_apply: Callable[[List[str]], None],
    ) -> None:
        super().__init__(master, "Classes affichées", 360, 520)
        self._on_apply = on_apply
        self._vars: Dict[str, tk.BooleanVar] = {}
        active = set(enabled)

        ctk.CTkLabel(
            self, text="Aucune case cochée = toutes les classes sont affichées.",
            text_color=theme.TEXT_MUTED, wraplength=320,
        ).pack(padx=12, pady=(12, 4))
        tools = ctk.CTkFrame(self, fg_color="transparent")
        tools.pack(fill="x", padx=12)
        ctk.CTkButton(tools, text="Tout cocher", width=110,
                      command=lambda: self._set_all(True)).pack(side="left")
        ctk.CTkButton(tools, text="Tout décocher", width=110, fg_color="transparent",
                      border_width=1, text_color=theme.TEXT,
                      command=lambda: self._set_all(False)).pack(side="left", padx=6)

        box = ctk.CTkScrollableFrame(self)
        box.pack(fill="both", expand=True, padx=12, pady=8)
        for name in class_names:
            var = tk.BooleanVar(value=name in active)
            self._vars[name] = var
            ctk.CTkCheckBox(box, text=name, variable=var).pack(anchor="w", pady=2)

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(bar, text="Annuler", fg_color="transparent", border_width=1,
                      text_color=theme.TEXT, command=self.destroy).pack(side="right", padx=(6, 0))
        ctk.CTkButton(bar, text="Appliquer", command=self._apply).pack(side="right")

    def _set_all(self, value: bool) -> None:
        for var in self._vars.values():
            var.set(value)

    def _apply(self) -> None:
        selected = [name for name, var in self._vars.items() if var.get()]
        self._on_apply([] if len(selected) == len(self._vars) else selected)
        self.destroy()
