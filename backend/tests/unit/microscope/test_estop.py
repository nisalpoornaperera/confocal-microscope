"""Latching emergency stop: pre-emption, listeners, laser handling, reset."""

from __future__ import annotations

import asyncio

import pytest
from tests.unit.microscope.conftest import RecordingStage, RigFactory

from confocal.config import MotionConfig
from confocal.errors import (
    EmergencyStopActiveError,
    LaserError,
    MotionAbortedError,
    MotionError,
)
from confocal.hardware.simulation import SimulationLaser
from confocal.models.calibration import DarkCalibrationRequest, ReferenceCalibrationRequest
from confocal.models.common import StageLimits


async def test_estop_aborts_an_in_flight_move_and_latches(make_rig: RigFactory) -> None:
    # 400 um at 40 um/s in real time = 10 s: the e-stop must not wait for the lock.
    rig = await make_rig(stage_time_scale=1.0, motion=MotionConfig(xy_speed_um_s=40.0))
    move = asyncio.create_task(rig.controller.move_to(x_um=400.0))
    for _ in range(100):
        if rig.stage.moving:
            break
        await asyncio.sleep(0.005)
    assert rig.stage.moving
    await asyncio.wait_for(rig.controller.emergency_stop("operator pressed STOP"), timeout=1.0)
    with pytest.raises(MotionAbortedError):
        await asyncio.wait_for(move, timeout=1.0)
    assert rig.controller.estop_engaged
    assert rig.stage.current_position.x_um < 400.0
    status = await rig.controller.status()
    assert status.estop_engaged
    assert status.stage.estop_engaged
    assert status.estop_reason == "operator pressed STOP"


async def test_motion_refused_until_reset(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.move_to(z_um=10.0)
    await rig.controller.emergency_stop("test")
    with pytest.raises(EmergencyStopActiveError):
        await rig.controller.move_to(z_um=11.0)
    with pytest.raises(EmergencyStopActiveError):
        await rig.controller.move_relative(dz_um=1.0)
    with pytest.raises(EmergencyStopActiveError):
        await rig.controller.home()
    with pytest.raises(EmergencyStopActiveError):
        await rig.controller.set_laser(True)
    with pytest.raises(EmergencyStopActiveError):
        await rig.controller.calibrate_dark(DarkCalibrationRequest())
    with pytest.raises(EmergencyStopActiveError):
        await rig.controller.calibrate_reference(ReferenceCalibrationRequest())
    assert rig.stage.moves_started == 1

    position = await rig.controller.reset_emergency_stop()
    assert position.z_um == pytest.approx(10.0)
    assert not rig.controller.estop_engaged
    assert (await rig.controller.status()).estop_reason is None
    assert (await rig.controller.move_to(z_um=11.0)).z_um == pytest.approx(11.0)


async def test_listeners_are_awaited_and_a_failing_one_is_isolated(make_rig: RigFactory) -> None:
    rig = await make_rig()
    calls: list[str] = []

    async def failing(reason: str) -> None:
        calls.append(f"failing:{reason}")
        raise RuntimeError("listener bug")

    async def recording(reason: str) -> None:
        await asyncio.sleep(0)
        calls.append(f"recording:{reason}")

    rig.controller.add_estop_listener(failing)
    rig.controller.add_estop_listener(recording)
    await rig.controller.emergency_stop("door opened")  # must not raise
    assert calls == ["failing:door opened", "recording:door opened"]
    assert rig.controller.estop_engaged


async def test_controllable_laser_is_switched_off(make_rig: RigFactory) -> None:
    rig = await make_rig()
    assert await rig.laser.is_enabled() is True
    await rig.controller.emergency_stop("test")
    assert await rig.laser.is_enabled() is False
    status = await rig.controller.status()
    assert status.laser.enabled is False
    assert status.laser.last_error is None


async def test_manual_laser_is_reported_as_not_switched_off(make_rig: RigFactory) -> None:
    rig = await make_rig(manual_laser=True)
    await rig.controller.emergency_stop("test")
    status = await rig.controller.status()
    assert status.estop_reason is not None
    assert status.estop_reason.startswith("test; ")
    assert "switch it off by hand" in status.estop_reason
    assert status.laser.last_error is not None
    assert "manually" in status.laser.last_error
    await rig.controller.reset_emergency_stop()
    assert (await rig.controller.status()).laser.last_error is None


async def test_manual_laser_cannot_be_switched_by_software(make_rig: RigFactory) -> None:
    rig = await make_rig(manual_laser=True)
    with pytest.raises(LaserError, match="manually"):
        await rig.controller.set_laser(False)


async def test_estop_never_raises_even_if_the_stage_stop_fails(make_rig: RigFactory) -> None:
    rig = await make_rig()

    async def broken_stop() -> None:
        raise RuntimeError("stop failed")

    rig.stage.stop = broken_stop  # type: ignore[method-assign]
    await rig.controller.emergency_stop("test")
    assert rig.controller.estop_engaged
    assert isinstance(rig.laser, SimulationLaser)
    assert not rig.laser.enabled


async def test_second_estop_keeps_the_first_reason(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.emergency_stop("first")
    await rig.controller.emergency_stop("second")
    assert rig.controller.estop_reason == "first"


async def test_set_laser_waits_the_scaled_warmup(make_rig: RigFactory) -> None:
    rig = await make_rig(laser_warmup_s=100.0)  # controller time_scale = 0
    await rig.controller.set_laser(False)
    await asyncio.wait_for(rig.controller.set_laser(True), timeout=1.0)
    assert isinstance(rig.laser, SimulationLaser)
    assert rig.laser.enabled


class _FirstMoveFails(RecordingStage):
    """The first move fails, leaving the controller's position unknown (re-read next time)."""

    def __init__(self, limits: StageLimits, *, motion: MotionConfig, time_scale: float) -> None:
        super().__init__(limits, motion=motion, time_scale=time_scale, fault_after_moves=0)


class _DeafStage(RecordingStage):
    """Completes every move even when stop() is called meanwhile (a firmware ignoring STOP)."""

    async def stop(self) -> None:
        self.stop_calls += 1
        await asyncio.sleep(0)


@pytest.mark.parametrize("position_known", [True, False])
@pytest.mark.parametrize("yields", range(8))
async def test_estop_racing_the_start_of_a_move_is_never_lost(
    make_rig: RigFactory, position_known: bool, yields: int
) -> None:
    # 3 um at 30 um/s in real time = 0.1 s: long enough to be stopped part-way.
    stage_cls = RecordingStage if position_known else _FirstMoveFails
    rig = await make_rig(stage_cls=stage_cls, stage_time_scale=1.0)
    if not position_known:
        with pytest.raises(MotionError):
            await rig.controller.move_to(z_um=1.0)
        assert rig.controller.last_position is None
    move = asyncio.create_task(rig.controller.move_to(z_um=3.0))
    for _ in range(yields):
        await asyncio.sleep(0)
    await rig.controller.emergency_stop("operator pressed STOP")
    with pytest.raises((EmergencyStopActiveError, MotionAbortedError)):
        await asyncio.wait_for(move, timeout=2.0)
    assert rig.stage.current_position.z_um < 3.0
    assert rig.controller.estop_engaged
    assert rig.controller.last_position is None


async def test_move_completing_despite_an_estop_is_reported_as_aborted(
    make_rig: RigFactory,
) -> None:
    rig = await make_rig(stage_cls=_DeafStage, stage_time_scale=1.0)
    stage = rig.stage
    assert isinstance(stage, _DeafStage)
    move = asyncio.create_task(rig.controller.move_to(z_um=3.0))
    for _ in range(100):
        if stage.moving:
            break
        await asyncio.sleep(0.001)
    assert stage.moving
    await rig.controller.emergency_stop("operator pressed STOP")
    with pytest.raises(MotionAbortedError, match="emergency stop"):
        await asyncio.wait_for(move, timeout=2.0)
    assert stage.stop_calls >= 2  # the e-stop's own stop, then the controller's after the move
    assert rig.controller.last_position is None
    status = await rig.controller.status()
    assert status.stage.last_error is not None
    assert "emergency stop" in status.stage.last_error
