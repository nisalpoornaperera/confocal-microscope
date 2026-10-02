"""Cooperative scan control signals and live progress / ETA tracking."""

from __future__ import annotations

import asyncio

import pytest

from confocal.models import Position, ScanState
from confocal.scanning.control import ScanControl, ScanStopRequested, StopKind
from confocal.scanning.plan import GridPoint
from confocal.scanning.progress import EventThrottle, LiveEventConfig, ProgressTracker


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


# --------------------------------------------------------------------------- control
def test_checkpoint_passes_until_a_stop_is_requested() -> None:
    control = ScanControl()
    control.checkpoint()
    control.request_cancel()
    with pytest.raises(ScanStopRequested) as info:
        control.checkpoint()
    assert info.value.request.kind is StopKind.CANCEL
    assert str(info.value) == "cancelled by user"


def test_abort_overrides_cancel_but_not_the_reverse() -> None:
    control = ScanControl()
    control.request_cancel()
    control.request_abort("emergency stop: button")
    assert control.stop_request is not None
    assert control.stop_request.kind is StopKind.ABORT
    control.request_cancel()
    assert control.stop_request.message == "emergency stop: button"


def test_pause_is_ignored_once_stopping() -> None:
    control = ScanControl()
    control.request_abort("emergency stop: x")
    control.request_pause()
    assert not control.pause_requested


async def test_wait_while_paused_returns_on_resume() -> None:
    control = ScanControl()
    control.request_pause()
    waiter = asyncio.create_task(control.wait_while_paused())
    await asyncio.sleep(0.01)
    assert not waiter.done()
    control.request_resume()
    await asyncio.wait_for(waiter, 1.0)


async def test_wait_while_paused_raises_when_stopped() -> None:
    control = ScanControl()
    control.request_pause()
    waiter = asyncio.create_task(control.wait_while_paused())
    await asyncio.sleep(0)
    control.request_abort("emergency stop: test")
    with pytest.raises(ScanStopRequested):
        await asyncio.wait_for(waiter, 1.0)


async def test_wait_while_paused_is_immediate_without_pause() -> None:
    await asyncio.wait_for(ScanControl().wait_while_paused(), 0.5)


# --------------------------------------------------------------------------- progress
def test_throttle_limits_the_event_rate() -> None:
    clock = FakeClock()
    throttle = EventThrottle(0.5, clock)
    assert throttle.ready()
    assert not throttle.ready()
    clock.now += 0.49
    assert not throttle.ready()
    clock.now += 0.02
    assert throttle.ready()
    always = EventThrottle(0.0, clock)
    assert always.ready()
    assert always.ready()


def test_live_event_config_validation() -> None:
    with pytest.raises(ValueError, match="intervals"):
        LiveEventConfig(progress_interval_s=-1.0)
    with pytest.raises(ValueError, match="eta_window"):
        LiveEventConfig(eta_window_points=0)


def test_eta_is_a_moving_average_of_recent_points() -> None:
    clock = FakeClock()
    tracker = ProgressTracker("s", 10, eta_window_points=3, clock=clock)
    assert tracker.eta_s() is None
    for duration in (10.0, 1.0, 1.0, 1.0):
        tracker.finish_point(duration)
    # window holds the last three (1 s each); 6 points remain
    assert tracker.eta_s() == pytest.approx(6.0)
    assert tracker.progress == pytest.approx(0.4)


def test_snapshot_reports_position_intensity_and_elapsed_time() -> None:
    clock = FakeClock()
    tracker = ProgressTracker("scan", 4, clock=clock)
    tracker.state = ScanState.SCANNING
    tracker.start()
    clock.now += 12.5
    tracker.begin_point(GridPoint(2, 1, 0, 10.0, 0.0))
    tracker.update(Position(x_um=10.0, y_um=0.0, z_um=3.25), float("nan"))
    snapshot = tracker.snapshot("hello")
    assert snapshot.state is ScanState.SCANNING
    assert snapshot.current_point_id == 2
    assert snapshot.current_z_um == 3.25
    assert snapshot.current_intensity is None  # NaN never reaches the API
    assert snapshot.elapsed_s == pytest.approx(12.5)
    assert snapshot.message == "hello"
    tracker.update(Position(x_um=10.0, y_um=0.0, z_um=3.5), 0.75)
    assert tracker.snapshot().current_intensity == 0.75
