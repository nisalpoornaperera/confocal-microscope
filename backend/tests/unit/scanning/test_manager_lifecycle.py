"""ScanManager: normal lifecycle, calibration / homing phases, pause / resume, post-processing."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence

import pytest
from tests.unit.scanning.conftest import (
    FakeController,
    FakeMLAnalyser,
    FakeReconstructor,
    MoveGate,
    make_bench,
    small_config,
    wait_for,
)

from confocal.errors import CalibrationError, ScanConflictError
from confocal.models import (
    CalibrationState,
    MLAnalysisRequest,
    MLResult,
    PointStatus,
    ScanEvent,
    ScanEventType,
    ScanMode,
    ScanPoint,
    ScanState,
)
from confocal.scanning.events import Subscription

S = ScanState


async def _collect(subscription: Subscription) -> list[ScanEvent]:
    async def run() -> list[ScanEvent]:
        return [event async for event in subscription]

    return await asyncio.wait_for(run(), 5.0)


async def test_full_lifecycle_to_complete_with_surface_saved() -> None:
    bench = make_bench()
    manager, repository = bench.manager, bench.repository
    created = await manager.create_scan(small_config(name="demo"))
    assert created.state is S.IDLE  # returned before the task ran
    assert manager.is_active
    assert manager.active_scan_id == created.id
    assert len(created.id) == 32
    subscription = await manager.subscribe(created.id)

    final = await manager.wait_until_finished(created.id, timeout=5.0)
    events = await _collect(subscription)

    assert final.state is S.COMPLETE
    assert not final.interrupted
    assert final.error_message is None
    assert final.completed_points == 6
    assert final.progress == 1.0
    assert final.started_at is not None
    assert final.finished_at is not None
    assert final.has_surface
    assert final.software_version == "0.0.0-test"
    assert repository.state_log[created.id] == [
        S.IDLE,
        S.PREPARING,
        S.SCANNING,
        S.PROCESSING,
        S.SURFACE_RECONSTRUCTION,
        S.COMPLETE,
    ]
    assert all(p.point.status is PointStatus.VALID for p in repository.stored(created.id))
    assert bench.reconstructor.calls == [(created.id, 6, 10.0)]
    assert repository.surfaces[0].scan_id == created.id
    assert not manager.is_active
    assert manager.active_scan_id is None

    states = [e.progress.state for e in events if e.type is ScanEventType.STATE]
    assert states == [S.PREPARING, S.SCANNING, S.PROCESSING, S.SURFACE_RECONSTRUCTION, S.COMPLETE]
    assert sum(e.type is ScanEventType.POINT for e in events) == 6
    assert events[-1].type is ScanEventType.STATE
    assert subscription.finished
    assert "scan_state" in repository.event_kinds(created.id)


async def test_manual_laser_gets_no_automatic_dark_calibration() -> None:
    bench = make_bench(FakeController(laser_controllable=False))
    created = await bench.manager.create_scan(small_config())
    subscription = await bench.manager.subscribe(created.id)
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    events = await _collect(subscription)
    assert final.state is S.COMPLETE
    assert S.CALIBRATING not in bench.repository.state_log[created.id]
    assert bench.controller.dark_calibrations == 0
    assert bench.controller.laser_switches == []  # a manual laser is never commanded
    warnings = [e for e in events if e.type is ScanEventType.ERROR]
    assert warnings
    assert warnings[0].message is not None
    assert warnings[0].message.startswith("warning: no dark calibration")
    assert "warning" in bench.repository.event_kinds(created.id)


async def test_controllable_laser_without_dark_calibration_calibrates_first() -> None:
    controller = FakeController(laser_controllable=True, laser_on=False)
    bench = make_bench(controller)
    created = await bench.manager.create_scan(small_config())
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.COMPLETE
    assert bench.repository.state_log[created.id][:4] == [
        S.IDLE,
        S.PREPARING,
        S.CALIBRATING,
        S.SCANNING,
    ]
    assert controller.dark_calibrations == 1
    assert controller.laser_switches == [True]  # switched on for the scan
    assert final.calibration_version == 1
    profile = bench.repository.stored(created.id)[0].profile
    assert profile is not None
    assert profile.dark_v == controller.surface.dark_v
    assert profile.calibration_version == 1


async def test_existing_dark_calibration_is_reused() -> None:
    controller = FakeController(
        laser_controllable=True, calibration=CalibrationState(version=4, dark_v=0.01)
    )
    bench = make_bench(controller)
    created = await bench.manager.create_scan(small_config())
    await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert controller.dark_calibrations == 0
    assert S.CALIBRATING not in bench.repository.state_log[created.id]


async def test_calibrate_dark_before_scan_forces_a_calibration() -> None:
    controller = FakeController(
        laser_controllable=True, calibration=CalibrationState(version=4, dark_v=0.01)
    )
    bench = make_bench(controller)
    created = await bench.manager.create_scan(small_config(calibrate_dark_before_scan=True))
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert controller.dark_calibrations == 1
    assert final.calibration_version == 5


async def test_calibrate_dark_before_scan_with_a_manual_laser_is_refused() -> None:
    bench = make_bench(FakeController(laser_controllable=False))
    with pytest.raises(CalibrationError, match="manual laser"):
        await bench.manager.create_scan(small_config(calibrate_dark_before_scan=True))
    assert bench.repository.scans == {}
    assert not bench.manager.is_active


async def test_home_before_scan_runs_the_homing_phase() -> None:
    bench = make_bench()
    created = await bench.manager.create_scan(small_config(home_before_scan=True))
    await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert bench.controller.homes == 1
    assert bench.repository.state_log[created.id][:4] == [S.IDLE, S.PREPARING, S.HOMING, S.SCANNING]


async def test_pause_applies_after_the_current_point_and_resume_continues() -> None:
    bench = make_bench()
    manager, controller = bench.manager, bench.controller
    created = await manager.create_scan(small_config())

    async def pause() -> None:
        await manager.pause(created.id)

    controller.move_hooks[5] = pause  # during point 0
    await wait_for(lambda: manager.active_scan_state is S.PAUSED)
    assert len(bench.repository.stored(created.id)) == 1  # point 0 was completed first
    moves = controller.moves
    await asyncio.sleep(0.02)
    assert controller.moves == moves  # nothing moves while paused
    paused = await manager.get_scan(created.id)
    assert paused.state is S.PAUSED
    assert paused.completed_points == 1
    assert (await manager.pause(created.id)).state is S.PAUSED  # idempotent

    await manager.resume(created.id)
    final = await manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.COMPLETE
    log = bench.repository.state_log[created.id]
    assert log[:5] == [S.IDLE, S.PREPARING, S.SCANNING, S.PAUSED, S.SCANNING]
    assert final.completed_points == 6


async def test_fixed_z_scan_skips_reconstruction() -> None:
    bench = make_bench()
    created = await bench.manager.create_scan(small_config(mode=ScanMode.FIXED_Z, z_center_um=2.0))
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.COMPLETE
    assert bench.repository.state_log[created.id] == [
        S.IDLE,
        S.PREPARING,
        S.SCANNING,
        S.PROCESSING,
        S.COMPLETE,
    ]
    assert bench.reconstructor.calls == []
    assert {p.point.status for p in bench.repository.stored(created.id)} == {PointStatus.MEASURED}


async def test_reconstruction_disabled_skips_the_phase() -> None:
    bench = make_bench()
    created = await bench.manager.create_scan(small_config(reconstruct_on_complete=False))
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.COMPLETE
    assert S.SURFACE_RECONSTRUCTION not in bench.repository.state_log[created.id]
    assert not final.has_surface


async def test_reconstruction_error_is_a_warning_not_a_failure() -> None:
    bench = make_bench(reconstructor=FakeReconstructor(fail=True))
    created = await bench.manager.create_scan(small_config())
    subscription = await bench.manager.subscribe(created.id)
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    events = await _collect(subscription)
    assert final.state is S.COMPLETE
    assert not final.interrupted
    assert not final.has_surface
    messages = [e.message or "" for e in events if e.type is ScanEventType.ERROR]
    assert any(m.startswith("warning: surface reconstruction failed") for m in messages)


async def test_ml_runs_when_a_model_is_deployed() -> None:
    ml = FakeMLAnalyser(available=True)
    bench = make_bench(ml=ml)
    created = await bench.manager.create_scan(small_config(ml_on_complete=True))
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.COMPLETE
    assert bench.repository.state_log[created.id][-3:] == [
        S.SURFACE_RECONSTRUCTION,
        S.ML_PROCESSING,
        S.COMPLETE,
    ]
    assert ml.calls == 1
    assert final.has_ml_result
    assert bench.repository.ml_results[0].advisory


async def test_ml_is_skipped_without_a_deployed_model() -> None:
    bench = make_bench(ml=FakeMLAnalyser(available=False))
    created = await bench.manager.create_scan(small_config(ml_on_complete=True))
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.COMPLETE
    assert S.ML_PROCESSING not in bench.repository.state_log[created.id]


async def test_ml_failure_never_fails_a_measured_scan() -> None:
    class BrokenML(FakeMLAnalyser):
        def analyse(
            self, scan_id: str, points: Sequence[ScanPoint], request: MLAnalysisRequest
        ) -> MLResult:
            raise RuntimeError("model file corrupt")

    bench = make_bench(ml=BrokenML())
    created = await bench.manager.create_scan(small_config(ml_on_complete=True))
    final = await bench.manager.wait_until_finished(created.id, timeout=5.0)
    assert final.state is S.COMPLETE
    assert "warning" in bench.repository.event_kinds(created.id)


async def test_other_coroutines_keep_running_while_a_scan_runs() -> None:
    bench = make_bench()
    manager = bench.manager
    gate = MoveGate()
    bench.controller.move_hooks[60] = gate  # a "slow move" in point 1
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    background = asyncio.create_task(ticker())
    created = await manager.create_scan(small_config())
    await asyncio.wait_for(gate.entered.wait(), 5.0)
    before = ticks
    # While the scan task waits on hardware the API stays responsive.
    live = await manager.get_scan(created.id)
    progress = await manager.get_progress(created.id)
    listed = await manager.list_scans()
    estimate = await manager.estimate(small_config())
    await asyncio.sleep(0.01)
    assert ticks > before
    assert live.state is S.SCANNING
    assert live.completed_points == 1
    assert progress.state is S.SCANNING
    assert progress.current_point_id == 1
    assert listed[0].state is S.SCANNING
    assert estimate.total_points == 6
    with pytest.raises(ScanConflictError):
        await manager.create_scan(small_config())
    gate.release()
    final = await manager.wait_until_finished(created.id, timeout=5.0)
    background.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await background
    assert final.state is S.COMPLETE
    assert ticks > 100
