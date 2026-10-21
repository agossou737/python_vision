"""Test de fumée de l'interface avec sources et détecteur simulés.

Nécessite un affichage (sous Linux sans écran : ``xvfb-run -a python -m pytest``).
Ignoré automatiquement si Tkinter, CustomTkinter ou l'affichage sont indisponibles.
"""

import json
import threading
import time

import pytest

tk = pytest.importorskip("tkinter")
pytest.importorskip("customtkinter")
pytest.importorskip("PIL")


def _display_available() -> bool:
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


if not _display_available():
    pytest.skip("Aucun affichage disponible", allow_module_level=True)

from app.application import Application  # noqa: E402
from app.detection.detection_engine import EngineState  # noqa: E402
from tests.fakes import FakeFactory  # noqa: E402
from tests.test_detection_engine import FakeDetector  # noqa: E402


def pump(app: Application, until, timeout: float = 6.0) -> bool:
    """Fait tourner la boucle Tk (sans mainloop) jusqu'à ce que la condition soit vraie."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.window.update()
        if until():
            return True
        time.sleep(0.005)
    app.window.update()
    return until()


def make_app(tmp_path, factory, scan=False, load_model=True) -> Application:
    config = {
        "video": {
            "reconnect_initial_delay_s": 0.1,
            "reconnect_max_delay_s": 0.2,
            "max_consecutive_read_failures": 3,
        },
        "paths": {"logs_dir": str(tmp_path / "logs"), "models_dir": str(tmp_path / "models")},
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(config))
    FakeDetector.instances = []
    app = Application(
        config_path=path,
        detector_factory=lambda name: FakeDetector(),
        source_factory=factory,
        scan_on_start=scan,
        load_model_on_start=load_model,
    )
    app.prepare()
    return app


@pytest.fixture
def make(tmp_path):
    apps = []

    def _make(factory, **kw):
        app = make_app(tmp_path, factory, **kw)
        apps.append(app)
        return app

    yield _make
    for app in apps:
        if not app._closing:
            app.shutdown()


def worker_threads():
    return [
        t for t in threading.enumerate()
        if t.name in ("VideoCaptureWorker", "InferenceWorker") and t.is_alive()
    ]


def test_startup_without_camera_stays_open_with_clear_message(make):
    app = make(FakeFactory(), scan=True)
    camera_label = app.window._indicators["camera"]._label
    assert pump(app, lambda: "aucune détectée" in camera_label.cget("text"))
    assert pump(app, lambda: app.engine.state == EngineState.READY)
    assert app.window.winfo_exists()


def test_full_pipeline_display_table_and_stats(make):
    app = make(FakeFactory(frames=10**6, read_delay=0.01))
    assert pump(app, lambda: app.engine.is_ready)
    app._on_start()
    assert app._running
    assert pump(app, lambda: app.window.video._image_item is not None)
    assert pump(app, lambda: len(app.window.detections._tree.get_children()) == 1)
    assert "1" in app.window.detections._counter.cget("text")

    # Seuil modifié à chaud : plus de rechargement, davantage de résultats.
    app.window.camera_panel._conf_slider.set(0.05)
    app.window.camera_panel._on_conf(0.05)
    assert pump(app, lambda: len(app.window.detections._tree.get_children()) == 2)
    assert len(FakeDetector.instances) == 1 and FakeDetector.instances[0].load_count == 1

    # Les trois FPS sont distincts et alimentés.
    assert pump(app, lambda: app.camera_manager.capture_fps > 5)
    assert pump(app, lambda: app._inference_worker.inference_fps > 0)
    assert pump(app, lambda: app._display_fps.fps > 0)
    assert app._stream_state == "Actif"


def test_stop_releases_camera_and_threads(make):
    factory = FakeFactory(frames=10**6, read_delay=0.01)
    app = make(factory)
    app._on_start()
    assert pump(app, lambda: app.window.video._image_item is not None)
    app._on_stop()
    assert pump(app, lambda: not app._running)
    assert not app.camera_manager.is_running
    assert factory.sources[0].released
    assert pump(app, lambda: not worker_threads())
    assert app.window.video._image_item is None  # placeholder affiché
    assert app.window.camera_panel._stop_btn.cget("state") == "disabled"


def test_signal_loss_then_reconnection(make):
    factory = FakeFactory(frames=5, read_delay=0.005)
    app = make(factory)
    app._on_start()
    assert pump(app, lambda: len(factory.sources) >= 2)  # reconnexion effective
    assert app._running
    assert pump(app, lambda: app._stream_state in ("Actif", "Signal perdu", "Reconnexion…"))
    assert factory.sources[0].released


def test_changing_source_while_running_frees_previous_first(make):
    factory = FakeFactory(frames=10**6, read_delay=0.01)
    app = make(factory)
    app._on_start()
    assert pump(app, lambda: len(factory.sources) == 1 and factory.sources[0].opened)
    app._on_start()  # redémarrage sur la source choisie
    assert pump(app, lambda: len(factory.sources) == 2 and factory.sources[1].opened)
    assert factory.sources[0].released
    assert len(worker_threads()) == 2  # 1 capture + 1 inférence, pas de doublon


def test_invalid_url_is_rejected_without_starting(make):
    app = make(FakeFactory())
    panel = app.window.camera_panel
    panel._type_menu.set("Flux réseau (HTTP / RTSP)")
    panel._refresh_visibility()
    panel._url_entry.insert(0, "pas une url")
    app._on_start()
    assert not app._running
    assert "espaces" in app.window._status.cget("text") or "invalide" in app.window._status.cget("text").lower()


def test_model_failure_keeps_app_open(tmp_path):
    from app.detection.base_detector import ModelLoadError

    class Broken(FakeDetector):
        def load(self, model_name, device="auto"):
            raise ModelLoadError("Ultralytics n'est pas installé.")

    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"paths": {"logs_dir": str(tmp_path / "l"), "models_dir": str(tmp_path / "m")}}))
    app = Application(config_path=config, detector_factory=lambda n: Broken(),
                      source_factory=FakeFactory(frames=10**6, read_delay=0.01), scan_on_start=False)
    app.prepare()
    try:
        assert pump(app, lambda: app.engine.state == EngineState.ERROR)
        assert pump(app, lambda: "Ultralytics" in app.window._status.cget("text"))
        app._on_start()  # la vidéo doit quand même s'afficher, sans détection inventée
        assert pump(app, lambda: app.window.video._image_item is not None)
        assert app.window.detections._counter.cget("text").endswith("0")
    finally:
        app.shutdown()


def test_shutdown_while_running_leaves_no_threads(make):
    app = make(FakeFactory(frames=10**6, read_delay=0.01))
    app._on_start()
    assert pump(app, lambda: app.window.video._image_item is not None)
    app.shutdown()
    deadline = time.monotonic() + 3
    while worker_threads() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not worker_threads()
