"""Tableau des détections, mis à jour en place (sans reconstruction complète)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Sequence

import customtkinter as ctk

from app.detection.types import Detection
from app.ui import theme

MAX_ROWS = 100
COLUMNS = (
    ("label", "Classe", 85, "w"),
    ("conf", "Score", 52, "center"),
    ("box", "Boîte x1,y1 → x2,y2", 170, "w"),
    ("cid", "ID", 32, "center"),
)


class DetectionPanel(ctk.CTkFrame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, corner_radius=10, fg_color=theme.BG_CARD)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self, text="Détections", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 0))
        self._counter = ctk.CTkLabel(
            self, text="Objets détectés : 0", text_color=theme.TEXT_MUTED
        )
        self._counter.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))

        holder = tk.Frame(self, bd=0, highlightthickness=0)
        self._holder = holder
        holder.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)

        self._style_name = "Det.Treeview"
        self.restyle()
        self._tree = ttk.Treeview(
            holder,
            columns=[c[0] for c in COLUMNS],
            show="headings",
            style=self._style_name,
            selectmode="browse",
        )
        for key, title, width, anchor in COLUMNS:
            self._tree.heading(key, text=title)
            self._tree.column(key, width=width, anchor=anchor, stretch=(key == "box"))
        scroll = ttk.Scrollbar(holder, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self._row_ids: List[str] = []

    def restyle(self) -> None:
        """Applique les couleurs du thème courant au Treeview (ttk ne suit pas CTk)."""
        dark = theme.is_dark()
        bg = "#111827" if dark else "#ffffff"
        fg = "#f9fafb" if dark else "#111827"
        head = "#374151" if dark else "#d1d5db"
        style = ttk.Style()
        try:
            style.theme_use("clam")  # seul thème ttk qui respecte les couleurs de fond
        except tk.TclError:
            pass
        style.configure(
            self._style_name, background=bg, foreground=fg, fieldbackground=bg,
            borderwidth=0, rowheight=24, bordercolor=bg, lightcolor=bg, darkcolor=bg,
        )
        if hasattr(self, "_holder"):
            self._holder.configure(bg=bg)
        style.configure(
            f"{self._style_name}.Heading", background=head, foreground=fg,
            relief="flat", borderwidth=0,
        )
        style.map(self._style_name, background=[("selected", theme.ACCENT)])

    def update_detections(self, detections: Sequence[Detection]) -> None:
        """Met à jour les lignes existantes, n'ajoute/supprime que la différence."""
        rows = sorted(detections, key=lambda d: d.confidence, reverse=True)[:MAX_ROWS]
        self._counter.configure(text=f"Objets détectés : {len(detections)}")

        for index, det in enumerate(rows):
            values = (
                det.label,
                f"{det.confidence:.0%}",
                f"{det.x1},{det.y1} → {det.x2},{det.y2}",
                det.class_id,
            )
            if index < len(self._row_ids):
                self._tree.item(self._row_ids[index], values=values)
            else:
                self._row_ids.append(self._tree.insert("", "end", values=values))
        while len(self._row_ids) > len(rows):
            self._tree.delete(self._row_ids.pop())

    def clear(self) -> None:
        self.update_detections(())
