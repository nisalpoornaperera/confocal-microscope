"""ScanManager: cancel, hardware failure, emergency stop, refusals and shutdown."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from tests.unit.scanning.conftest import (
    FakeController,
    InMemoryRepository,
    MoveGate,
    make_bench,
    small_config,
    wait_for,
)

from confocal.api.deps import require_no_active_scan
from confocal.errors import (
    EmergencyStopActiveError,
    LimitViolationError,
    ScanConflictError,
    StorageError,
)
from confocal.models import (
    AdcSamples,
    CalibrationState,
    DarkCalibrationRequest,
    PointStatus,
    ProfileAnalysis,
    ProfileData,
    ScanPoint,
    ScanState,
    ScanSummary,
)
from confocal.scanning import manager as manager_module
from confocal.scanning.manager import SHUTDOWN_MESSAGE
from confocal.scanning.plan import ScanPlan

S = ScanState
_DARK_V1 = CalibrationState(version=1, dark_v=0.01)


async def test_cancel_ends_cancelled_and_keeps_the_partial_point() -> None:
    bench = make_bench()
    manager = bench.manager
    created = await manager.create_scan(small_config())

    async def cancel() -> None:
        await manager.cancel(created.id)

    bench.controller.move_hooks[30] = cancel  # inside the fine sweep of point 0
    final = await manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.CANCELLED
    assert final.interrupted
    assert final.error_message is None
    assert final.finished_at is not None
    stored = bench.repository.stored(created.id)
    assert [s.point.status for s in stored] == [PointStatus.ABORTED]
    assert stored[0].profile is not None
    assert stored[0].profile.n_positions == 29  # raw data of the interrupted point kept
    assert bench.controller.moves == 30  # stopped at the next Z step
    assert bench.controller.emergency_stops == []  # cancel never latches the e-stop
    assert "scan_cancelled" in bench.repository.event_kinds(created.id)
    assert not bench.manager.is_active
    assert bench.repository.surfaces == []


async def test_cancel_while_paused() -> None:
    bench = make_bench()
    manager = bench.manager
    created = await manager.create_scan(small_config())

    async def pause() -> None:
        await manager.pause(created.id)

    bench.controller.move_hooks[3] = pause
    await wait_for(lambda: manager.active_scan_state is S.PAUSED)
    await manager.cancel(created.id)
    final = await manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.CANCELLED
    assert final.interrupted
    assert final.completed_points == 1
    assert bench.repository.state_log[created.id][-2:] == [S.PAUSED, S.CANCELLED]


async def test_hardware_failure_stops_the_stage_and_ends_in_error() -> None:
    controller = FakeController(fail_at_move=45)  # inside point 1
    bench = make_bench(controller)
    created = await bench.manager.create_scan(small_config())
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.ERROR
    assert final.interrupted
    assert final.error_message is not None
    assert final.error_message.startswith("hardware failure: MotionError")
    assert len(controller.emergency_stops) == 1  # the stage was stopped
    assert controller.estop_engaged
    statuses = [s.point.status for s in bench.repository.stored(created.id)]
    assert statuses == [PointStatus.VALID, PointStatus.ABORTED]
    assert "hardware_failure" in bench.repository.event_kinds(created.id)
    assert not bench.manager.is_active


async def test_emergency_stop_ends_the_scan_in_error() -> None:
    controller = FakeController(estop_at_move=10)
    bench = make_bench(controller)
    created = await bench.manager.create_scan(small_config())
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.ERROR
    assert final.error_message == "emergency stop: test e-stop"
    assert final.interrupted
    assert controller.emergency_stops == ["test e-stop"]  # not stopped a second time
    assert bench.repository.stored(created.id)[0].point.status is PointStatus.ABORTED
    assert "emergency_stop" in bench.repository.event_kinds(created.id)


async def test_emergency_stop_wakes_a_paused_scan() -> None:
    bench = make_bench()
    manager = bench.manager
    created = await manager.create_scan(small_config())

    async def pause() -> None:
        await manager.pause(created.id)

    bench.controller.move_hooks[3] = pause
    await wait_for(lambda: manager.active_scan_state is S.PAUSED)
    await bench.controller.emergency_stop("operator button")
    final = await manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.ERROR
    assert final.error_message == "emergency stop: operator button"
    assert bench.repository.state_log[created.id][-2:] == [S.PAUSED, S.ERROR]


async def test_storage_failure_ends_in_error_without_latching_the_estop() -> None:
    class FailingRepository(InMemoryRepository):
        def append_point(
            self,
            scan_id: str,
            point: ScanPoint,
            profile: ProfileData | None,
            analysis: ProfileAnalysis | None,
        ) -> None:
            raise StorageError("disk full")

    bench = make_bench(repository=FailingRepository())
    created = await bench.manager.create_scan(small_config())
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.ERROR
    assert final.error_message == "unexpected error: StorageError: disk full"
    assert final.interrupted
    assert bench.controller.emergency_stops == []  # nothing was moving


async def test_unexpected_bug_during_acquisition_stops_the_stage() -> None:
    controller = FakeController()

    async def broken(n_samples: int) -> AdcSamples:
        raise ValueError("driver bug")

    controller.acquire = broken  # type: ignore[method-assign]
    bench = make_bench(controller)
    created = await bench.manager.create_scan(small_config())
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.ERROR
    assert final.error_message == "unexpected error: ValueError: driver bug"
    assert len(controller.emergency_stops) == 1


# --------------------------------------------------------------------------- refusals
async def test_a_second_scan_is_refused_while_one_is_active() -> None:
    bench = make_bench()
    gate = MoveGate()
    bench.controller.move_hooks[2] = gate
    created = await bench.manager.create_scan(small_config())
    await asyncio.wait_for(gate.entered.wait(), 5.0)
    with pytest.raises(ScanConflictError, match=created.id):
        await bench.manager.create_scan(small_config())
    assert len(bench.repository.scans) == 1
    await bench.manager.cancel(created.id)
    gate.release()
    await bench.manager.wait_until_finished(created.id, timeout=5.0)
    second = await bench.manager.create_scan(small_config())
    assert (await bench.manager.wait_until_finished(second.id, timeout=5.0)).state is S.COMPLETE


async def test_concurrent_create_requests_start_only_one_scan() -> None:
    bench = make_bench()
    results = await asyncio.gather(
        bench.manager.create_scan(small_config()),
        bench.manager.create_scan(small_config()),
        return_exceptions=True,
    )
    assert sum(isinstance(r, ScanConflictError) for r in results) == 1
    assert len(bench.repository.scans) == 1
    scan_id = bench.manager.active_scan_id
    assert scan_id is not None
    await bench.manager.wait_until_finished(scan_id, timeout=5.0)


async def test_limit_violation_is_refused_before_anything_is_persisted_or_moved() -> None:
    bench = make_bench()
    with pytest.raises(LimitViolationError) as info:
        await bench.manager.create_scan(small_config(x_stop_um=2000.0, xy_step_um=500.0))
    assert info.value.violations
    assert bench.repository.scans == {}
    assert bench.controller.moves == 0
    assert not bench.manager.is_active


async def test_latched_emergency_stop_refuses_new_scans() -> None:
    bench = make_bench()
    await bench.controller.emergency_stop("earlier stop")
    with pytest.raises(EmergencyStopActiveError):
        await bench.manager.create_scan(small_config())
    assert bench.repository.scans == {}


# --------------------------------------------------------------------------- shutdown
async def test_shutdown_records_the_running_scan_as_interrupted() -> None:
    bench = make_bench()
    gate = MoveGate()
    bench.controller.move_hooks[50] = gate  # hangs inside point 1
    created = await bench.manager.create_scan(small_config())
    await asyncio.wait_for(gate.entered.wait(), 5.0)
    await bench.manager.shutdown()
    summary = bench.repository.get_scan(created.id)
    assert summary.state is S.ERROR
    assert summary.error_message == SHUTDOWN_MESSAGE
    assert summary.interrupted
    statuses = [s.point.status for s in bench.repository.stored(created.id)]
    assert statuses == [PointStatus.VALID, PointStatus.ABORTED]
    assert not bench.manager.is_active
    assert "scan_interrupted" in bench.repository.event_kinds(created.id)
    with pytest.raises(ScanConflictError, match="shutting down"):
        await bench.manager.create_scan(small_config())


async def test_shutdown_before_the_task_started_still_finalises_the_scan() -> None:
    bench = make_bench()
    created = await bench.manager.create_scan(small_config())
    await bench.manager.shutdown()  # the task never got to run
    summary = bench.repository.get_scan(created.id)
    assert summary.state is S.ERROR
    assert summary.error_message == SHUTDOWN_MESSAGE
    assert summary.interrupted
    assert bench.controller.moves == 0
    assert not bench.manager.is_active


async def test_shutdown_without_a_scan_is_a_no_op() -> None:
    bench = make_bench()
    await bench.manager.shutdown()
    assert bench.repository.scans == {}


# --------------------------------------------------------------------------- creation race
class _BlockingCreateRepository(InMemoryRepository):
    """``create_scan`` blocks its worker thread until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def create_scan(self, **kwargs: Any) -> ScanSummary:
        self.entered.set()
        if not self.release.wait(5.0):
            raise AssertionError("create_scan was never released")
        return super().create_scan(**kwargs)


async def _wait_thread_event(event: threading.Event) -> None:
    await asyncio.wait_for(asyncio.to_thread(event.wait, 5.0), 6.0)


async def test_instrument_is_reserved_while_create_scan_persists() -> None:
    """repro_create_race: a manual command must be refused before the scan is registered."""
    repository = _BlockingCreateRepository()
    bench = make_bench(controller=FakeController(calibration=_DARK_V1), repository=repository)
    task = asyncio.create_task(bench.manager.create_scan(small_config()))
    await _wait_thread_event(repository.entered)
    try:
        assert bench.manager.is_active
        assert bench.manager.active_scan_id is not None
        assert bench.manager.active_scan_state is S.PREPARING
        with pytest.raises(ScanConflictError, match="manual hardware control is refused"):
            require_no_active_scan(bench.manager)
    finally:
        repository.release.set()
    created = await task
    assert created.id == bench.manager.active_scan_id
    assert (await bench.manager.wait_until_finished(created.id, timeout=5.0)).state is S.COMPLETE
    assert not bench.manager.is_active


async def test_failed_creation_releases_the_reservation() -> None:
    bench = make_bench()
    with pytest.raises(LimitViolationError):
        await bench.manager.create_scan(small_config(x_stop_um=2000.0, xy_step_um=500.0))
    assert not bench.manager.is_active
    assert bench.manager.active_scan_id is None
    assert bench.manager.active_scan_state is None
    require_no_active_scan(bench.manager)  # manual control is allowed again
    created = await bench.manager.create_scan(small_config())
    assert (await bench.manager.wait_until_finished(created.id, timeout=5.0)).state is S.COMPLETE


async def test_calibration_changed_during_creation_is_recorded_in_the_scan_row() -> None:
    """A calibration that completes while the scan row is written is the one applied."""
    repository = _BlockingCreateRepository()
    controller = FakeController(calibration=_DARK_V1)
    bench = make_bench(controller=controller, repository=repository)
    task = asyncio.create_task(bench.manager.create_scan(small_config()))
    await _wait_thread_event(repository.entered)
    newer = await controller.calibrate_dark(DarkCalibrationRequest(beam_blocked_confirmed=True))
    repository.release.set()
    created = await task
    assert created.calibration_version == 1
    done = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert newer.version == 2
    assert done.calibration_version == 2
    assert done.calibration is not None
    assert done.calibration.dark_v == newer.dark_v


async def test_unchanged_calibration_is_not_rewritten() -> None:
    bench = make_bench(controller=FakeController(calibration=_DARK_V1))
    calls: list[dict[str, object]] = []
    original = bench.repository.update_scan

    def spy(scan_id: str, **kwargs: Any) -> ScanSummary:
        calls.append(kwargs)
        return original(scan_id, **kwargs)

    bench.repository.update_scan = spy  # type: ignore[method-assign]
    created = await bench.manager.create_scan(small_config())
    done = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert done.calibration_version == 1
    assert not any("calibration" in kwargs for kwargs in calls)


# --------------------------------------------------------------------------- off-loop planning
async def test_plan_construction_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread = threading.get_ident()
    threads: list[int] = []
    real = ScanPlan.from_config

    def recording(*args: Any, **kwargs: Any) -> ScanPlan:
        threads.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(manager_module.ScanPlan, "from_config", recording)
    bench = make_bench()
    await bench.manager.estimate(small_config())
    created = await bench.manager.create_scan(small_config())
    await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert len(threads) == 2
    assert loop_thread not in threads
