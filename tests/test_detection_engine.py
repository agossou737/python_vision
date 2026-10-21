import queue
import time
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from app.config_manager import DetectionSettings
from app.detection.base_detector import (
    BaseDetector,
    InferenceError,
    ModelLoadError,
    resolve_device,
)
from app.detection.detection_engine import DetectionEngine, EngineState
from app.detection.types import Detection
from app.detection.yolo_detector import parse_yolo_result
from app.events import EventKind
from app.utils.latest_queue import LatestQueue
from app.video.frame import FramePacket
from app.workers.inference_worker import InferenceWorker
from tests.fakes import wait_until

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


class FakeDetector(BaseDetector):
    """Détecteur scripté : renvoie person@0.9 et car@0.3, filtré par le seuil reçu."""

    instances: List["FakeDetector"] = []

    def __init__(self, fail_load: bool = False, fail_detect: bool = False) -> None:
        self.fail_load = fail_load
        self.fail_detect = fail_detect
        self.loaded = False
        self.load_count = 0
        self.unloaded = False
        self.detect_calls = 0
        self.model_name = ""
        FakeDetector.instances.append(self)

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @property
    def class_names(self) -> List[str]:
        return ["person", "car"]

    def load(self, model_name: str, device: str = "auto") -> None:
        if self.fail_load:
            raise ModelLoadError("poids invalides")
        self.load_count += 1
        self.loaded = True
        self.model_name = model_name

    def detect(self, frame, confidence, imgsz):
        self.detect_calls += 1
        if self.fail_detect:
            raise RuntimeError("boom")
        raw = [Detection(0, "person", 0.9, 1, 2, 30, 40), Detection(1, "car", 0.3, 5, 5, 20, 20)]
        return [d for d in raw if d.confidence >= confidence]

    def unload(self) -> None:
        self.loaded = False
        self.unloaded = True


def make_engine(**detector_kwargs) -> DetectionEngine:
    FakeDetector.instances = []
    settings = DetectionSettings()
    settings.confidence = 0.5
    return DetectionEngine(settings, detector_factory=lambda name: FakeDetector(**detector_kwargs))


# ---------------------------------------------------------------- engine
def test_load_and_ready():
    engine = make_engine()
    assert engine.state == EngineState.UNLOADED
    engine.load_model("yolov8n.pt")
    assert engine.is_ready
    assert engine.class_names == ["person", "car"]
    assert engine.model_name == "yolov8n.pt"


def test_confidence_changes_without_reload():
    engine = make_engine()
    engine.load_model("m.pt")
    assert [d.label for d in engine.detect(FRAME)] == ["person"]
    engine.set_confidence(0.05)
    assert [d.label for d in engine.detect(FRAME)] == ["person", "car"]
    assert len(FakeDetector.instances) == 1
    assert FakeDetector.instances[0].load_count == 1


def test_confidence_is_clamped():
    engine = make_engine()
    engine.set_confidence(5)
    assert engine.settings.confidence == 0.95
    engine.set_confidence(0)
    assert engine.settings.confidence == 0.05


def test_class_filter():
    engine = make_engine()
    engine.load_model("m.pt")
    engine.set_confidence(0.05)
    engine.set_enabled_classes(["car"])
    assert [d.label for d in engine.detect(FRAME)] == ["car"]
    engine.set_enabled_classes([])
    assert len(engine.detect(FRAME)) == 2


def test_detect_without_model_raises():
    engine = make_engine()
    with pytest.raises(InferenceError):
        engine.detect(FRAME)


def test_detector_exception_becomes_inference_error():
    engine = make_engine(fail_detect=True)
    engine.load_model("m.pt")
    with pytest.raises(InferenceError):
        engine.detect(FRAME)


def test_load_failure_sets_error_state():
    engine = make_engine(fail_load=True)
    with pytest.raises(ModelLoadError):
        engine.load_model("bad.pt")
    assert engine.state == EngineState.ERROR
    assert "poids invalides" in engine.status_message


def test_failed_switch_keeps_previous_model():
    engine = make_engine()
    engine.load_model("a.pt")
    engine._factory = lambda name: FakeDetector(fail_load=True)
    with pytest.raises(ModelLoadError):
        engine.load_model("b.pt")
    assert engine.is_ready
    assert engine.model_name == "a.pt"
    assert engine.detect(FRAME)  # l'ancien modèle fonctionne toujours


def test_model_switch_unloads_old_one():
    engine = make_engine()
    engine.load_model("a.pt")
    first = FakeDetector.instances[0]
    engine.load_model("b.pt")
    assert first.unloaded
    assert engine.model_name == "b.pt"
    engine.unload()
    assert engine.state == EngineState.UNLOADED
    assert FakeDetector.instances[1].unloaded


def test_resolve_device_explicit_and_fallback():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("auto") in ("cpu", "cuda:0", "mps")


# ---------------------------------------------------------------- parsing YOLO
def test_parse_yolo_result_coordinates_and_labels():
    class FakeBoxes:
        xyxy = np.array([[10.4, 20.6, 100.0, 90.0], [-5.0, -5.0, 999.0, 999.0]])
        conf = np.array([0.88, 0.51])
        cls = np.array([0.0, 2.0])

        def __len__(self) -> int:
            return 2

    boxes = FakeBoxes()
    result = SimpleNamespace(boxes=boxes, names={0: "person", 2: "car"})
    dets = parse_yolo_result(result, (480, 640, 3), [])
    assert dets[0] == Detection(0, "person", pytest.approx(0.88), 10, 21, 100, 90)
    assert dets[1].label == "car"
    assert (dets[1].x1, dets[1].y1, dets[1].x2, dets[1].y2) == (0, 0, 639, 479)


def test_parse_yolo_result_empty():
    assert parse_yolo_result(SimpleNamespace(boxes=None, names={}), (10, 10, 3), []) == []


# ---------------------------------------------------------------- worker
class Harness:
    def __init__(self, engine: DetectionEngine, frame_skip: int = 0) -> None:
        self.settings = engine.settings
        self.settings.frame_skip = frame_skip
        self.frames: LatestQueue = LatestQueue(4)
        self.results: LatestQueue = LatestQueue(64)
        self.events: queue.Queue = queue.Queue(maxsize=64)
        self.worker = InferenceWorker(engine, self.frames, self.results, self.events, self.settings)
        self.worker.start()
        self._id = 0

    def push_and_get(self):
        self._id += 1
        self.frames.put(FramePacket(FRAME.copy(), time.monotonic(), self._id, "test"))
        return self.results.get(2.0)

    def stop(self):
        self.worker.request_stop()
        self.worker.join(2)
        assert not self.worker.is_alive()


def test_worker_produces_detections_and_metrics():
    engine = make_engine()
    engine.load_model("m.pt")
    h = Harness(engine)
    for _ in range(5):
        res = h.push_and_get()
        assert res is not None and res.inferred and res.engine_ready
        assert [d.label for d in res.detections] == ["person"]
        assert res.packet.width == 64
    assert h.worker.avg_latency_ms > 0
    assert h.worker.last_detection_count == 1
    h.stop()


def test_worker_frame_skip_reuses_last_detections():
    engine = make_engine()
    engine.load_model("m.pt")
    h = Harness(engine, frame_skip=2)
    pattern = [h.push_and_get().inferred for _ in range(6)]
    assert pattern == [True, False, False, True, False, False]
    assert FakeDetector.instances[0].detect_calls == 2
    h.stop()


def test_worker_without_model_returns_no_detections():
    engine = make_engine()  # modèle non chargé
    h = Harness(engine)
    res = h.push_and_get()
    assert not res.engine_ready and not res.inferred and res.detections == ()
    h.stop()


def test_worker_inference_error_gives_no_fake_results_and_one_event():
    engine = make_engine(fail_detect=True)
    engine.load_model("m.pt")
    h = Harness(engine)
    for _ in range(4):
        res = h.push_and_get()
        assert res.detections == () and not res.engine_ready
    kinds = []
    while not h.events.empty():
        kinds.append(h.events.get_nowait().kind)
    assert kinds == [EventKind.INFERENCE_ERROR]  # limité en fréquence
    h.stop()


def test_worker_recovers_after_model_loaded_later():
    engine = make_engine()
    h = Harness(engine)
    assert not h.push_and_get().engine_ready
    engine.load_model("m.pt")
    assert wait_until(lambda: h.push_and_get().engine_ready)
    h.stop()


def test_worker_threshold_change_applies_live():
    engine = make_engine()
    engine.load_model("m.pt")
    h = Harness(engine)
    assert len(h.push_and_get().detections) == 1
    engine.set_confidence(0.05)
    assert len(h.push_and_get().detections) == 2
    h.stop()
