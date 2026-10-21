import queue

from app.config_manager import VideoSettings
from app.events import EventKind
from app.utils.image_utils import compute_fit_size, is_valid_frame
from app.utils.latest_queue import LatestQueue
from app.utils.metrics import FpsCounter
from app.video.base_source import SourceNotSupportedError
from app.video.camera_manager import CameraManager
from app.video.smartphone_source import SmartphoneSource
from app.video.video_capture_worker import VideoCaptureWorker
from tests.fakes import FakeFactory, FakeSource, wait_until

import numpy as np
import pytest


def fast_settings(**kw) -> VideoSettings:
    base = dict(
        reconnect_initial_delay_s=0.1,
        reconnect_max_delay_s=0.2,
        max_consecutive_read_failures=3,
        frame_queue_size=2,
    )
    base.update(kw)
    s = VideoSettings()
    for k, v in base.items():
        setattr(s, k, v)
    return s


def drain(q: "queue.Queue"):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_latest_queue_is_bounded_and_drops_oldest():
    q = LatestQueue(2)
    for i in range(10):
        q.put(i)
    assert len(q) == 2
    assert q.dropped == 8
    assert q.get(0.01) == 8
    assert q.get_latest(0.01) == 9
    assert q.get(0.01) is None


def test_frame_validation_and_fit():
    assert is_valid_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    assert not is_valid_frame(None)
    assert not is_valid_frame(np.zeros((4, 4), dtype=np.uint8))
    assert not is_valid_frame(np.zeros((0, 4, 3), dtype=np.uint8))
    assert compute_fit_size(1920, 1080, 960, 960) == (960, 540, 0, 210)


def test_fps_counter_zero_when_idle():
    assert FpsCounter().fps == 0.0


def test_worker_streams_frames_then_signal_lost_and_reconnects():
    settings = fast_settings()
    frames = LatestQueue(2)
    events: "queue.Queue" = queue.Queue(maxsize=256)
    factory = FakeFactory(frames=5)
    worker = VideoCaptureWorker(factory, frames, events, settings)
    worker.start()
    assert wait_until(lambda: len(factory.sources) >= 2)  # reconnexion effective
    worker.request_stop()
    worker.join(2)
    assert not worker.is_alive()
    kinds = [e.kind for e in drain(events)]
    assert EventKind.CONNECTED in kinds
    assert EventKind.SIGNAL_LOST in kinds
    assert EventKind.RECONNECTING in kinds
    assert kinds[-1] == EventKind.STOPPED
    assert all(s.released for s in factory.sources)


def test_worker_without_auto_reconnect_stops_after_loss():
    settings = fast_settings(auto_reconnect=False)
    events: "queue.Queue" = queue.Queue(maxsize=256)
    worker = VideoCaptureWorker(FakeFactory(frames=3), LatestQueue(2), events, settings)
    worker.start()
    worker.join(3)
    assert not worker.is_alive()
    kinds = [e.kind for e in drain(events)]
    assert EventKind.RECONNECTING not in kinds
    assert kinds[-1] == EventKind.STOPPED


def test_worker_reports_open_failure_and_retries_until_stopped():
    settings = fast_settings()
    events: "queue.Queue" = queue.Queue(maxsize=256)
    factory = FakeFactory(fail_open=True)
    worker = VideoCaptureWorker(factory, LatestQueue(2), events, settings)
    worker.start()
    assert wait_until(lambda: len(factory.sources) >= 3)
    worker.request_stop()
    worker.join(2)
    assert not worker.is_alive()
    kinds = [e.kind for e in drain(events)]
    assert kinds.count(EventKind.ERROR) >= 2


def test_worker_gives_up_after_max_attempts():
    settings = fast_settings(reconnect_max_attempts=2)
    events: "queue.Queue" = queue.Queue(maxsize=256)
    factory = FakeFactory(fail_open=True)
    worker = VideoCaptureWorker(factory, LatestQueue(2), events, settings)
    worker.start()
    worker.join(3)
    assert not worker.is_alive()
    assert len(factory.sources) == 3  # 1 essai initial + 2 tentatives de reconnexion


def test_queue_stays_bounded_under_load():
    settings = fast_settings(auto_reconnect=False)
    frames = LatestQueue(2)
    worker = VideoCaptureWorker(
        FakeFactory(frames=500), frames, queue.Queue(maxsize=256), settings
    )
    worker.start()
    worker.join(5)
    assert len(frames) <= 2
    assert frames.dropped > 0


def test_manager_start_stop_releases_source():
    mgr = CameraManager(fast_settings())
    factory = FakeFactory(frames=10_000, read_delay=0.001)
    mgr.start(factory)
    assert wait_until(lambda: len(mgr.frame_queue) > 0)
    packet = mgr.frame_queue.get_latest(1)
    assert packet is not None and packet.width == 64 and packet.height == 48
    assert mgr.stop(wait=True, timeout=2)
    assert not mgr.is_running
    assert factory.sources[0].released


def test_manager_switch_source_releases_previous():
    mgr = CameraManager(fast_settings())
    first = FakeFactory(frames=10_000, read_delay=0.001, label="A")
    second = FakeFactory(frames=10_000, read_delay=0.001, label="B")
    mgr.start(first)
    assert wait_until(lambda: len(first.sources) == 1 and first.sources[0].opened)
    mgr.start(second)
    assert first.sources[0].released
    assert wait_until(lambda: len(second.sources) == 1 and second.sources[0].opened)
    mgr.shutdown()
    assert second.sources[0].released
    assert not mgr.is_running


def test_manager_test_source():
    ok = CameraManager.test_source(FakeSource(frames=3))
    assert ok.ok and ok.resolution == (64, 48)
    bad = CameraManager.test_source(FakeSource(fail_open=True))
    assert not bad.ok
    empty = CameraManager.test_source(FakeSource(frames=0))
    assert not empty.ok


def test_smartphone_unsupported_methods_are_explicit():
    for method in ("bluetooth", "ios_direct", "inconnue"):
        with pytest.raises(SourceNotSupportedError):
            SmartphoneSource(method).open()


def test_unsupported_source_does_not_retry():
    from app.video.smartphone_source import SmartphoneSource as S

    settings = fast_settings()
    events: "queue.Queue" = queue.Queue(maxsize=256)
    worker = VideoCaptureWorker(lambda: S("bluetooth"), LatestQueue(2), events, settings)
    worker.start()
    worker.join(2)
    assert not worker.is_alive()
    kinds = [e.kind for e in drain(events)]
    assert EventKind.RECONNECTING not in kinds
    assert EventKind.ERROR in kinds
