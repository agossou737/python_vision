"""Contrôleur de l'application : relie l'interface, la capture et l'inférence.

Règles de concurrence :
  - seul le thread principal touche aux widgets ;
  - les threads (scan, test de source, chargement du modèle) renvoient leur résultat
    via ``_post_ui`` (file consommée par la boucle ``after``) ;
  - aucun arrêt ne bloque l'interface : ``_begin_stop`` + ``_poll_stop`` surveillent
    la fin des workers toutes les 50 ms.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk

from app.config_manager import PROJECT_ROOT, ConfigManager
from app.detection.base_detector import ModelLoadError
from app.detection.detection_engine import DetectionEngine, DetectorFactory, EngineState
from app.detection.types import DetectionResult
from app.events import EventKind, WorkerEvent
from app.logger import setup_logging
from app.ui.camera_panel import CameraPanelCallbacks
from app.ui.main_window import MainWindow
from app.ui.settings_dialog import ClassFilterDialog, SettingsDialog
from app.utils.latest_queue import LatestQueue
from app.utils.metrics import FpsCounter
from app.utils.validators import validate_camera_index, validate_stream_url
from app.video.camera_manager import CameraManager, SourceTestResult, build_source
from app.video.local_camera import CameraInfo
from app.video.video_capture_worker import SourceFactory
from app.workers.inference_worker import InferenceWorker

logger = logging.getLogger(__name__)


def _resolve_dir(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


class Application:
    TICK_MS = 15
    TABLE_INTERVAL_S = 0.15
    STATS_INTERVAL_S = 0.5
    STOP_TIMEOUT_S = 4.0

    def __init__(
        self,
        config_path: Optional[Path] = None,
        detector_factory: Optional[DetectorFactory] = None,
        source_factory: Optional[SourceFactory] = None,
        scan_on_start: bool = True,
        load_model_on_start: bool = True,
    ) -> None:
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load()
        setup_logging(_resolve_dir(self.config.paths.logs_dir))
        ctk.set_appearance_mode(self.config.appearance.theme)
        ctk.set_default_color_theme(self.config.appearance.color_theme)

        self._source_factory = source_factory
        self._scan_on_start = scan_on_start
        self._load_model_on_start = load_model_on_start

        self.events: "queue.Queue[WorkerEvent]" = queue.Queue(maxsize=256)
        self.camera_manager = CameraManager(self.config.video, self.events)
        self.engine = DetectionEngine(
            self.config.detection, _resolve_dir(self.config.paths.models_dir), detector_factory
        )
        self.results: "LatestQueue[DetectionResult]" = LatestQueue(2)
        self._inference_worker: Optional[InferenceWorker] = None
        self._ui_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()

        # État de l'interface
        self._running = False
        self._stopping = False
        self._pending_start = False
        self._stop_deadline = 0.0
        self._final_message: Optional[str] = "Arrêté."
        self._stream_state = "Arrêté"
        self._camera_hint = "arrêtée"  # état affiché tant que rien ne tourne (résultat du scan)
        self._loading_model = False
        self._closing = False
        self._after_id: Optional[str] = None
        self._last_result: Optional[DetectionResult] = None
        self._display_fps = FpsCounter()
        self._last_table = 0.0
        self._last_stats = 0.0

        callbacks = CameraPanelCallbacks(
            on_start=self._on_start,
            on_stop=self._on_stop,
            on_test=self._on_test,
            on_scan=self._scan_cameras,
            on_confidence=self.engine.set_confidence,
            on_model=self._load_model_async,
            on_imgsz=self.engine.set_imgsz,
            on_frame_skip=self._on_frame_skip,
            on_display_options=self._on_display_options,
            on_classes=self._on_classes,
        )
        self.window = MainWindow(self.config, callbacks, self._on_settings)

    # ------------------------------------------------------------------ cycle de vie
    def prepare(self) -> None:
        """Initialise l'interface et la boucle périodique (sans lancer mainloop)."""
        self.window.protocol("WM_DELETE_WINDOW", self.shutdown)
        self._set_status("Prêt. Choisissez une source puis cliquez sur « Démarrer ».")
        self._update_indicators()
        if self._scan_on_start:
            self._scan_cameras()
        if self._load_model_on_start:
            self._load_model_async(self.config.detection.model_name)
        self._after_id = self.window.after(self.TICK_MS, self._tick)

    def run(self) -> None:
        self.prepare()
        self.window.mainloop()

    def shutdown(self) -> None:
        """Nettoyage centralisé : arrêt des workers, libération caméra et modèle, sauvegarde."""
        if self._closing:
            return
        self._closing = True
        logger.info("Fermeture de l'application")
        if self._after_id is not None:
            try:
                self.window.after_cancel(self._after_id)
            except tk.TclError:
                pass
        worker = self._inference_worker
        if worker is not None:
            worker.request_stop()
        self.camera_manager.stop(wait=True, timeout=1.5)
        self.camera_manager.frame_queue.clear()
        if worker is not None:
            worker.join(1.0)
        if worker is None or not worker.is_alive():
            self.engine.unload()
        self.config_manager.save()
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ boucle UI
    def _post_ui(self, fn: Callable[[], None]) -> None:
        """Planifie ``fn`` dans le thread principal (appelable depuis n'importe quel thread)."""
        self._ui_queue.put(fn)

    def _tick(self) -> None:
        if self._closing:
            return
        try:
            self._drain_ui_queue()
            self._drain_events()
            self._render_latest()
            now = time.monotonic()
            if now - self._last_stats >= self.STATS_INTERVAL_S:
                self._last_stats = now
                self._update_stats()
                self._update_indicators()
        except tk.TclError:
            return  # fenêtre détruite
        except Exception:
            logger.exception("Erreur dans la boucle d'interface")
        try:
            self._after_id = self.window.after(self.TICK_MS, self._tick)
        except tk.TclError:
            pass

    def _drain_ui_queue(self) -> None:
        for _ in range(20):
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception:
                logger.exception("Erreur dans un callback d'interface")

    def _drain_events(self) -> None:
        for _ in range(50):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            self._handle_event(event)

    def _render_latest(self) -> None:
        result = self.results.get_latest(timeout=0.0)
        if result is None or not self._running or self._stopping:
            return
        self._last_result = result
        self._display_fps.tick()
        cfg = self.config.detection
        self.window.video.show_result(result, cfg.show_boxes, cfg.show_labels, self._display_fps.fps)
        now = time.monotonic()
        if result.inferred and now - self._last_table >= self.TABLE_INTERVAL_S:
            self._last_table = now
            self.window.detections.update_detections(result.detections)
        elif not result.engine_ready and now - self._last_table >= self.TABLE_INTERVAL_S:
            self._last_table = now
            self.window.detections.clear()

    def _update_stats(self) -> None:
        worker = self._inference_worker
        res = self._last_result
        active = self._running and not self._stopping
        resolution = (res.packet.width, res.packet.height) if res else self.camera_manager.resolution
        self.window.statistics.update_stats(
            capture_fps=self.camera_manager.capture_fps,
            inference_fps=worker.inference_fps if worker else 0.0,
            display_fps=self._display_fps.fps,
            latency_ms=worker.avg_latency_ms if worker else 0.0,
            count=worker.last_detection_count if worker else 0,
            resolution=resolution if active else None,
            state=self._stream_state,
            source=self.camera_manager.active_source_name if active else "",
        )

    def _update_indicators(self) -> None:
        state = self._stream_state
        level = {"Actif": "ok", "Signal perdu": "error", "Erreur": "error"}.get(
            state, "warn" if state not in ("Arrêté",) else "idle"
        )
        if state == "Arrêté":
            warn = self._camera_hint.startswith("aucune")
            self.window.set_indicator("camera", f"Caméra : {self._camera_hint}", "warn" if warn else "idle")
        else:
            self.window.set_indicator("camera", f"Caméra : {state.lower()}", level)
        is_net = self.config.video.source_type != "local"
        conn_text = f"Connexion : {state.lower()}" if (is_net and self._running) else "Connexion : —"
        self.window.set_indicator("connection", conn_text, level if (is_net and self._running) else "idle")

        es = self.engine.state
        if es == EngineState.READY:
            self.window.set_indicator("model", f"Modèle : {self.engine.model_name} ({self.engine.device})", "ok")
        elif es == EngineState.LOADING:
            self.window.set_indicator("model", "Modèle : chargement…", "warn")
        elif es == EngineState.ERROR:
            self.window.set_indicator("model", "Modèle : erreur", "error")
        else:
            self.window.set_indicator("model", "Modèle : non chargé", "idle")

    def _set_status(self, message: str, level: str = "idle") -> None:
        self.window.set_status(message, level)

    # ------------------------------------------------------------------ événements workers
    def _handle_event(self, event: WorkerEvent) -> None:
        kind = event.kind
        if self._stopping and kind != EventKind.STOPPED:
            return
        video = self.window.video
        if kind == EventKind.CONNECTING:
            self._stream_state = "Connexion…"
            self._set_status(event.message, "warn")
        elif kind == EventKind.CONNECTED:
            self._stream_state = "Actif"
            self._set_status(f"Analyse en cours — {event.message}", "ok")
        elif kind == EventKind.SIGNAL_LOST:
            self._stream_state = "Signal perdu"
            suffix = "\nTentative de reconnexion…" if self.config.video.auto_reconnect else ""
            video.show_placeholder("Signal vidéo perdu." + suffix)
            self._set_status("Signal vidéo perdu.", "warn")
        elif kind == EventKind.RECONNECTING:
            self._stream_state = "Reconnexion…"
            self._set_status(event.message, "warn")
        elif kind == EventKind.ERROR:
            self._stream_state = "Erreur"
            video.show_placeholder(event.message)
            self._set_status(event.message, "error")
        elif kind == EventKind.INFERENCE_ERROR:
            self._set_status(f"Analyse : {event.message}", "error")
        elif kind == EventKind.STOPPED:
            # Un STOPPED périmé d'un ancien worker ne doit pas couper un nouveau démarrage.
            if self._running and not self._stopping and not self.camera_manager.is_running:
                self._begin_stop(final_message=None)  # capture terminée d'elle-même

    # ------------------------------------------------------------------ démarrage / arrêt
    def _capture_alive(self) -> bool:
        return self.camera_manager.is_running

    def _inference_alive(self) -> bool:
        return self._inference_worker is not None and self._inference_worker.is_alive()

    def _validate_params(self, p: Dict[str, Any]) -> Optional[str]:
        source, method = p["source_type"], p["smartphone_method"]
        needs_cam = source == "local" or (source == "smartphone" and method == "virtual_camera")
        needs_url = source == "network" or (source == "smartphone" and method == "ip_app")
        if needs_cam:
            check = validate_camera_index(p["camera_index"])
            return None if check else check.message
        if needs_url:
            check = validate_stream_url(p["stream_url"])
            return None if check else check.message
        for key in ("adb_local_port", "adb_remote_port"):
            if not (1 <= p[key] <= 65535):
                return "Port ADB invalide (1-65535)."
        return None

    def _apply_params(self, p: Dict[str, Any]) -> None:
        v = self.config.video
        v.source_type = p["source_type"]
        v.smartphone_method = p["smartphone_method"]
        if p["camera_index"] >= 0:
            v.camera_index = p["camera_index"]
        v.stream_url = p["stream_url"]
        if p["adb_local_port"] > 0:
            v.adb_local_port = p["adb_local_port"]
        if p["adb_remote_port"] > 0:
            v.adb_remote_port = p["adb_remote_port"]
        v.adb_stream_path = p["adb_stream_path"]

    def _on_start(self) -> None:
        params = self.window.camera_panel.get_source_params()
        error = self._validate_params(params)
        if error:
            self._set_status(error, "error")
            return
        self._apply_params(params)
        self.config_manager.save()
        if self._stopping:
            self._pending_start = True  # un arrêt est déjà en cours : on enchaînera
            return
        if self._capture_alive() or self._inference_alive():
            # Changement de source en cours d'analyse : arrêt non bloquant puis redémarrage.
            self._pending_start = True
            self._begin_stop("Changement de source : libération de l'ancienne…", "Arrêté.")
            return
        self._do_start()

    def _on_stop(self) -> None:
        self._pending_start = False
        if self._running and not self._stopping:
            self._begin_stop("Arrêt en cours…", "Arrêté.")

    def _do_start(self) -> None:
        self.results.clear()
        self._display_fps.reset()
        self._last_result = None
        if self.engine.state in (EngineState.UNLOADED, EngineState.ERROR):
            self._load_model_async(self.config.detection.model_name)

        worker = InferenceWorker(
            self.engine, self.camera_manager.frame_queue, self.results, self.events,
            self.config.detection,
        )
        self._inference_worker = worker
        worker.start()
        self.camera_manager.start(self._source_factory)

        self._running = True
        self._stopping = False
        self._stream_state = "Connexion…"
        self.window.camera_panel.set_running(True)
        self.window.video.show_placeholder("Connexion à la source…")
        self._set_status("Démarrage de l'analyse…", "warn")
        self._update_indicators()

    def _begin_stop(self, message: Optional[str] = None, final_message: Optional[str] = "Arrêté.") -> None:
        self._stopping = True
        self._final_message = final_message
        self._stream_state = "Arrêt…"
        if message:
            self._set_status(message, "warn")
        self.camera_manager.stop(wait=False)
        if self._inference_worker is not None:
            self._inference_worker.request_stop()
        self._stop_deadline = time.monotonic() + self.STOP_TIMEOUT_S
        self.window.after(50, self._poll_stop)

    def _poll_stop(self) -> None:
        if self._closing:
            return
        if self._capture_alive() or self._inference_alive():
            if time.monotonic() > self._stop_deadline:
                logger.error("Les workers ne se sont pas arrêtés dans le délai imparti")
                self._pending_start = False
                self._finalize_stop()
                self._set_status(
                    "La source ne répond pas à l'arrêt (lecture bloquante). Réessayez ou relancez l'application.",
                    "error",
                )
                return
            self.window.after(50, self._poll_stop)
            return
        self._finalize_stop()

    def _finalize_stop(self) -> None:
        self.results.clear()
        self.camera_manager.frame_queue.clear()
        self._last_result = None
        self._display_fps.reset()
        self.window.detections.clear()
        self.window.video.show_placeholder("Flux arrêté." if self._final_message else None)
        self._running = False
        self._stopping = False
        self._stream_state = "Arrêté"
        self.window.camera_panel.set_running(False)
        if self._final_message:
            self._set_status(self._final_message)
        self._update_indicators()
        if self._pending_start:
            self._pending_start = False
            self._do_start()

    # ------------------------------------------------------------------ actions secondaires
    def _scan_cameras(self) -> None:
        if self._capture_alive():
            self._set_status("Arrêtez l'analyse avant de rechercher les caméras.", "warn")
            return
        self.window.camera_panel.set_scan_busy(True)
        self._set_status("Recherche des caméras…", "warn")

        def done(cameras: List[CameraInfo]) -> None:
            self._post_ui(lambda: self._on_scan_done(cameras))

        self.camera_manager.scan_cameras(done)

    def _on_scan_done(self, cameras: List[CameraInfo]) -> None:
        panel = self.window.camera_panel
        panel.set_scan_busy(False)
        if cameras:
            panel.set_cameras(cameras)
            self._camera_hint = f"{len(cameras)} détectée(s)"
            self._set_status(f"{len(cameras)} caméra(s) détectée(s).", "ok")
        else:
            self._camera_hint = "aucune détectée"
            self._set_status(
                "Aucune caméra détectée : branchez-en une, ou utilisez un flux réseau / smartphone.", "warn"
            )
        self._update_indicators()

    def _on_test(self) -> None:
        if self._capture_alive():
            self._set_status("Arrêtez l'analyse avant de tester une source.", "warn")
            return
        params = self.window.camera_panel.get_source_params()
        error = self._validate_params(params)
        if error:
            self._set_status(error, "error")
            return
        settings = dataclasses.replace(self.config.video, **params)
        source = self._source_factory() if self._source_factory else build_source(settings)
        timeout = settings.connect_timeout_s + 3.0
        panel = self.window.camera_panel
        panel.set_busy_test(True)
        self._set_status("Test de la source en cours…", "warn")

        def work() -> None:
            result = CameraManager.test_source(source, timeout)
            self._post_ui(lambda: self._on_test_done(result))

        threading.Thread(target=work, name="SourceTestRunner", daemon=True).start()

    def _on_test_done(self, result: SourceTestResult) -> None:
        self.window.camera_panel.set_busy_test(False)
        self._set_status(result.message, "ok" if result.ok else "error")

    def _load_model_async(self, name: str) -> None:
        name = (name or "").strip()
        if not name:
            return
        if self._loading_model:
            self._set_status("Un chargement de modèle est déjà en cours.", "warn")
            return
        self._loading_model = True
        self._set_status(f"Initialisation du modèle {name}…", "warn")

        def work() -> None:
            error: Optional[str] = None
            try:
                self.engine.load_model(name)
            except ModelLoadError as exc:
                error = str(exc)
            except Exception:
                logger.exception("Erreur inattendue au chargement du modèle")
                error = "Erreur inattendue au chargement du modèle (voir les logs)."
            self._post_ui(lambda: self._on_model_loaded(error))

        threading.Thread(target=work, name="ModelLoader", daemon=True).start()

    def _on_model_loaded(self, error: Optional[str]) -> None:
        self._loading_model = False
        if error:
            self._set_status(error, "error")
        else:
            self._set_status(self.engine.status_message, "ok")
        if self.engine.model_name:
            self.window.camera_panel.set_model_text(self.engine.model_name)
        self._update_indicators()

    def _on_frame_skip(self, value: int) -> None:
        self.config.detection.frame_skip = int(value)

    def _on_display_options(self, show_boxes: bool, show_labels: bool) -> None:
        self.config.detection.show_boxes = show_boxes
        self.config.detection.show_labels = show_labels
        if self._last_result is not None and self._running:
            self.window.video.show_result(
                self._last_result, show_boxes, show_labels, self._display_fps.fps
            )

    def _on_classes(self) -> None:
        names = self.engine.class_names
        if not names:
            self._set_status("Chargez d'abord un modèle pour choisir les classes.", "warn")
            return
        ClassFilterDialog(
            self.window, names, self.config.detection.enabled_classes, self._apply_classes
        )

    def _apply_classes(self, selected: List[str]) -> None:
        self.engine.set_enabled_classes(selected)
        self._set_status(
            f"{len(selected)} classe(s) affichée(s)." if selected else "Toutes les classes sont affichées.",
            "ok",
        )

    def _on_settings(self) -> None:
        SettingsDialog(self.window, self.config, self._on_settings_saved)

    def _on_settings_saved(self) -> None:
        self.config_manager.save()
        ctk.set_appearance_mode(self.config.appearance.theme)
        self.window.restyle()
        self._set_status("Paramètres enregistrés.", "ok")
