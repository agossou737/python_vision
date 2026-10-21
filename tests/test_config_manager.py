import json

from app.config_manager import AppConfig, ConfigManager
from app.utils.validators import validate_camera_index, validate_stream_url


def test_defaults_created_when_missing(tmp_path):
    path = tmp_path / "settings.json"
    cfg = ConfigManager(path).load()
    assert path.exists()
    assert cfg.detection.confidence == 0.5
    assert cfg.appearance.theme == "dark"


def test_roundtrip(tmp_path):
    mgr = ConfigManager(tmp_path / "s.json")
    mgr.load()
    mgr.config.video.stream_url = "http://192.168.1.20:8080/video"
    mgr.config.detection.enabled_classes = ["person", "car"]
    assert mgr.save()
    other = ConfigManager(tmp_path / "s.json").load()
    assert other.video.stream_url == "http://192.168.1.20:8080/video"
    assert other.detection.enabled_classes == ["person", "car"]


def test_values_are_clamped(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps({"detection": {"confidence": 5, "imgsz": 10}, "video": {"frame_queue_size": 999}})
    )
    cfg = ConfigManager(path).load()
    assert cfg.detection.confidence == 0.95
    assert cfg.detection.imgsz == 160
    assert cfg.video.frame_queue_size == 16


def test_wrong_types_and_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"video": {"camera_index": "abc", "inconnu": 1}, "zzz": {}}))
    cfg = ConfigManager(path).load()
    assert cfg.video.camera_index == 0


def test_corrupted_file_falls_back_and_is_backed_up(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{ pas du json")
    cfg = ConfigManager(path).load()
    assert isinstance(cfg, AppConfig)
    assert (tmp_path / "s.json.bak").exists()


def test_url_validation():
    assert validate_stream_url("http://192.168.1.20:8080/video")
    assert validate_stream_url("rtsp://192.168.1.20:554/stream")
    assert not validate_stream_url("")
    assert not validate_stream_url("ftp://host/x")
    assert not validate_stream_url("http://")
    assert not validate_stream_url("http://host:99999/x")
    assert not validate_stream_url("http://ho st/x")


def test_camera_index_validation():
    assert validate_camera_index(0)
    assert validate_camera_index("2")
    assert not validate_camera_index(-1)
    assert not validate_camera_index("x")
