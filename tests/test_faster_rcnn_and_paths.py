import sys
from pathlib import Path

import numpy as np
import pytest

from app import paths
from app.detection.base_detector import ModelLoadError
from app.detection.detection_engine import default_detector_factory
from app.detection.faster_rcnn_detector import (
    SUPPORTED_MODELS,
    FasterRcnnDetector,
    parse_torchvision_output,
)
from app.detection.types import Detection

CATEGORIES = ["__background__", "person", "bicycle", "car"]


def test_parse_filters_by_confidence_and_clamps():
    output = {
        "boxes": np.array([[10.4, 20.6, 100.0, 90.0], [-5.0, -5.0, 999.0, 999.0], [1, 1, 5, 5]]),
        "scores": np.array([0.92, 0.55, 0.10]),
        "labels": np.array([1, 3, 2]),
    }
    dets = parse_torchvision_output(output, (480, 640, 3), CATEGORIES, 0.5)
    assert len(dets) == 2  # le score 0.10 est filtré
    assert dets[0] == Detection(1, "person", pytest.approx(0.92), 10, 21, 100, 90)
    assert (dets[1].label, dets[1].x1, dets[1].y2, dets[1].x2) == ("car", 0, 479, 639)


def test_parse_unknown_label_and_empty():
    output = {"boxes": np.array([[0, 0, 4, 4]]), "scores": np.array([0.9]), "labels": np.array([99])}
    assert parse_torchvision_output(output, (10, 10, 3), CATEGORIES, 0.5)[0].label == "99"
    empty = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0)}
    assert parse_torchvision_output(empty, (10, 10, 3), CATEGORIES, 0.5) == []


def test_unsupported_model_name_gives_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="non pris en charge"):
        FasterRcnnDetector(tmp_path).load("fasterrcnn_inconnu")


def test_factory_routes_by_model_name(tmp_path):
    factory = default_detector_factory(tmp_path)
    assert isinstance(factory(SUPPORTED_MODELS[0]), FasterRcnnDetector)
    from app.detection.yolo_detector import YoloDetector

    assert isinstance(factory("yolov8n.pt"), YoloDetector)


def test_class_names_hide_background():
    det = FasterRcnnDetector()
    det._categories = ["__background__", "person", "N/A", "car"]
    assert det.class_names == ["person", "car"]


def test_app_dir_normal_and_frozen(monkeypatch):
    assert paths.app_dir() == Path(paths.__file__).resolve().parent.parent
    assert not paths.is_frozen()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/VisionDesktop/VisionDesktop")
    assert paths.is_frozen()
    assert paths.app_dir() == Path("/opt/VisionDesktop")
