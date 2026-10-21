"""Zone vidéo : affichage de l'image avec boîtes de détection, proportions préservées."""

from __future__ import annotations

import tkinter as tk
from typing import Optional, Sequence

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from app.detection.types import Detection, DetectionResult
from app.ui import theme
from app.utils.image_utils import compute_fit_size


class VideoPanel(ctk.CTkFrame):
    """Affiche les images dans un Canvas.

    Un Canvas (et non un Label) évite la boucle de rétroaction « l'image agrandit
    le widget qui agrandit l'image ». Toutes les méthodes sont à appeler depuis
    le thread principal uniquement.
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, corner_radius=10, fg_color=theme.BG_VIDEO)
        self._canvas = tk.Canvas(
            self, highlightthickness=0, bd=0, bg=theme.resolve(theme.BG_VIDEO)
        )
        self._canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self._photo: Optional[ImageTk.PhotoImage] = None  # référence obligatoire (sinon GC)
        self._image_item: Optional[int] = None
        self._text_item = self._canvas.create_text(
            0, 0, text="", fill=theme.MUTED, font=("Segoe UI", 14), justify="center"
        )
        self._fps_item = self._canvas.create_text(
            12, 10, text="", fill="#ffffff", anchor="nw", font=("Consolas", 11, "bold")
        )
        self._last: Optional[tuple] = None  # (result, show_boxes, show_labels, fps)
        self._placeholder = "Aucun flux vidéo.\nChoisissez une source puis cliquez sur « Démarrer »."
        self._canvas.bind("<Configure>", self._on_resize)
        self._show_placeholder_item()

    # ---- API ---------------------------------------------------------------
    def show_placeholder(self, text: Optional[str] = None) -> None:
        """Efface l'image et affiche un message explicite."""
        if text is not None:
            self._placeholder = text
        self._last = None
        self._photo = None
        if self._image_item is not None:
            self._canvas.delete(self._image_item)
            self._image_item = None
        self._canvas.itemconfigure(self._fps_item, text="")
        self._show_placeholder_item()

    def show_result(
        self,
        result: DetectionResult,
        show_boxes: bool,
        show_labels: bool,
        display_fps: float,
    ) -> None:
        self._last = (result, show_boxes, show_labels, display_fps)
        self._render()

    def restyle(self) -> None:
        """À appeler après un changement de thème (le Canvas n'est pas piloté par CTk)."""
        self._canvas.configure(bg=theme.resolve(theme.BG_VIDEO))

    # ---- Rendu ---------------------------------------------------------------
    def _on_resize(self, _event: tk.Event) -> None:
        if self._last is not None:
            self._render()
        else:
            self._show_placeholder_item()

    def _show_placeholder_item(self) -> None:
        w, h = self._canvas.winfo_width(), self._canvas.winfo_height()
        self._canvas.coords(self._text_item, w // 2, h // 2)
        self._canvas.itemconfigure(self._text_item, text=self._placeholder, state="normal")
        self._canvas.tag_raise(self._text_item)

    def _render(self) -> None:
        if self._last is None:
            return
        result, show_boxes, show_labels, fps = self._last
        box_w, box_h = self._canvas.winfo_width(), self._canvas.winfo_height()
        if box_w < 20 or box_h < 20:
            return
        frame = result.packet.frame
        src_h, src_w = frame.shape[:2]
        new_w, new_h, _, _ = compute_fit_size(src_w, src_h, box_w, box_h)
        interp = cv2.INTER_AREA if new_w < src_w else cv2.INTER_LINEAR
        image = cv2.resize(frame, (new_w, new_h), interpolation=interp)

        if show_boxes and result.detections:
            self._draw_detections(image, result.detections, new_w / src_w, show_labels)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        cx, cy = box_w // 2, box_h // 2
        if self._image_item is None:
            self._image_item = self._canvas.create_image(cx, cy, image=self._photo, anchor="center")
        else:
            self._canvas.coords(self._image_item, cx, cy)
            self._canvas.itemconfigure(self._image_item, image=self._photo)
        self._canvas.itemconfigure(self._text_item, state="hidden")
        self._canvas.itemconfigure(self._fps_item, text=f"{fps:4.1f} FPS (affichage)")
        self._canvas.tag_raise(self._fps_item)

    @staticmethod
    def _draw_detections(
        image, detections: Sequence[Detection], scale: float, show_labels: bool
    ) -> None:
        """Dessine sur l'image REDIMENSIONNÉE : coordonnées source * échelle (texte net)."""
        height, width = image.shape[:2]
        for det in detections:
            color = theme.box_color(det.class_id)
            p1 = (int(det.x1 * scale), int(det.y1 * scale))
            p2 = (int(det.x2 * scale), int(det.y2 * scale))
            cv2.rectangle(image, p1, p2, color, 2)
            if not show_labels:
                continue
            text = f"{det.label} {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(0, p1[1] - th - baseline - 4)
            left = min(max(0, p1[0]), max(0, width - tw - 6))
            cv2.rectangle(image, (left, top), (left + tw + 6, top + th + baseline + 4), color, -1)
            cv2.putText(
                image, text, (left + 3, top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
