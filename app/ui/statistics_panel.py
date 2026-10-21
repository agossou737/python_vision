"""Cartes de statistiques temps réel (FPS distincts, latence, résolution, état)."""

from __future__ import annotations

import tkinter as tk
from typing import Dict, Optional, Tuple

import customtkinter as ctk

from app.ui import theme


class StatCard(ctk.CTkFrame):
    def __init__(self, master: tk.Misc, title: str, value: str = "—") -> None:
        super().__init__(master, corner_radius=10, fg_color=theme.BG_CARD)
        ctk.CTkLabel(
            self, text=title, text_color=theme.TEXT_MUTED, font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10, pady=(6, 0))
        self._value = ctk.CTkLabel(self, text=value, font=ctk.CTkFont(size=17, weight="bold"))
        self._value.pack(anchor="w", padx=10, pady=(0, 6))

    def set(self, text: str) -> None:
        self._value.configure(text=text)


class StatisticsPanel(ctk.CTkFrame):
    CARDS = (
        ("fps_capture", "FPS capture"),
        ("fps_infer", "FPS inférence"),
        ("fps_display", "FPS affichage"),
        ("latency", "Latence moy."),
        ("count", "Détections"),
        ("resolution", "Résolution"),
        ("state", "État du flux"),
        ("source", "Source active"),
    )

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, fg_color="transparent")
        self._cards: Dict[str, StatCard] = {}
        columns = 4
        for i, (key, title) in enumerate(self.CARDS):
            card = StatCard(self, title)
            card.grid(row=i // columns, column=i % columns, sticky="nsew", padx=4, pady=4)
            self._cards[key] = card
        for c in range(columns):
            self.grid_columnconfigure(c, weight=1, uniform="stat")

    def update_stats(
        self,
        *,
        capture_fps: float,
        inference_fps: float,
        display_fps: float,
        latency_ms: float,
        count: int,
        resolution: Optional[Tuple[int, int]],
        state: str,
        source: str,
    ) -> None:
        c = self._cards
        c["fps_capture"].set(f"{capture_fps:.1f}")
        c["fps_infer"].set(f"{inference_fps:.1f}")
        c["fps_display"].set(f"{display_fps:.1f}")
        c["latency"].set(f"{latency_ms:.0f} ms" if latency_ms > 0 else "—")
        c["count"].set(str(count))
        c["resolution"].set(f"{resolution[0]}×{resolution[1]}" if resolution else "—")
        c["state"].set(state)
        c["source"].set(source or "—")
