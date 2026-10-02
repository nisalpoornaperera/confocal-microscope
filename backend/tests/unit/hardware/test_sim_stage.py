"""SimulationStage: timing, interruption, limits, kinematic quantisation, faults."""

from __future__ import annotations

import asyncio

import pytest

from confocal.config import MotionConfig
from confocal.errors import (
    HardwareNotConnectedError,
    LimitViolationError,
    MotionAbortedError,
    MotionError,
)
from confocal.hardware.kinematics import StageKinematics
from confocal.hardware.simulation import SimulationStage
from confocal.models.common import Axis, AxisLimits, Position, StageLimits
from confocal.models.hardware import StageState

LIMITS = StageLimits(
    x=AxisLimits(min_um=-500.0, max_um=500.0),
    y=AxisLimits(min_um=-500.0, max_um=500.0),
    z=AxisLimits(min_um=-100.0, max_um=100.0),
)
MOTION = MotionConfig(xy_speed_um_s=100.0, z_speed_um_s=50.0)
DELTA = StageKinematics.openflexure_delta(um_per_step=(0.06, 0.06, 0.06))


async def _stage(
    time_scale: float = 0.0,
    *,
    kinematics: StageKinematics | None = None,
    fault_after_moves: int | None = None,
) -> SimulationStage:
    stage = SimulationStage(
        LIMITS,
        motion=MOTION,
        time_scale=time_scale,
        kinematics=kinematics,
        fault_after_moves=fault_after_moves,
    )
    await stage.connect()
    return stage


async def test_move_reports_target_and_updates_position() -> None:
    stage = await _stage()
    target = Position(x_um=12.5, y_um=-3.0, z_um=4.0)
    assert await stage.move_to(target) == target
    assert await stage.get_position() == target
    assert stage.current_position == target
    status = await stage.status()
    assert status.state is StageState.IDLE
    assert status.connected
    assert status.position == target
    assert not stage.motors_released


async def test_requires_connection() -> None:
    stage = SimulationStage(LIMITS, motion=MOTION, time_scale=0.0)
    with pytest.raises(HardwareNotConnectedError):
        await stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=0.0))
    assert (await stage.status()).state is StageState.DISCONNECTED


async def test_limits_are_enforced_and_nothing_moves() -> None:
    stage = await _stage()
    with pytest.raises(LimitViolationError) as excinfo:
        await stage.move_to(Position(x_um=600.0, y_um=0.0, z_um=-200.0))
    assert len(excinfo.value.violations) == 2
    assert stage.current_position == Position(x_um=0.0, y_um=0.0, z_um=0.0)
    assert stage.moves_started == 0


async def test_kinematic_quantisation() -> None:
    stage = await _stage(kinematics=DELTA)
    target = Position(x_um=10.01, y_um=-4.33, z_um=1.234)
    reached = await stage.move_to(target)
    assert reached == DELTA.quantize(target)
    assert reached != target
    assert reached.max_axis_error(target) <= DELTA.max_quantization_error_um()
    assert stage.kinematics is DELTA


async def test_move_duration_scales_with_time_scale() -> None:
    stage = await _stage(time_scale=0.01)
    loop = asyncio.get_running_loop()
    started = loop.time()
    await stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=100.0))  # 2 s at 50 um/s -> 20 ms
    assert loop.time() - started >= 0.015


async def test_stop_aborts_an_in_flight_move_at_the_partial_position() -> None:
    stage = await _stage(time_scale=1.0)
    target = Position(x_um=400.0, y_um=-200.0, z_um=0.0)  # 4 s
    task = asyncio.create_task(stage.move_to(target))
    await asyncio.sleep(0.05)
    assert stage.moves_started == 1
    midway = stage.current_position
    assert 0.0 < midway.x_um < 400.0
    await stage.stop()
    with pytest.raises(MotionAbortedError, match="stopped at"):
        await task
    stopped = stage.current_position
    assert 0.0 < stopped.x_um < 100.0
    assert stopped.y_um == pytest.approx(-0.5 * stopped.x_um)  # still on the line
    assert not stage.moving
    assert (await stage.status()).state is StageState.STOPPED
    # The stage is usable again afterwards.
    assert await stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=0.0)) is not None


@pytest.mark.parametrize("time_scale", [0.0, 1.0])
@pytest.mark.parametrize("homing", [False, True])
async def test_stop_arriving_as_the_move_starts_aborts_it(time_scale: float, homing: bool) -> None:
    stage = await _stage(time_scale)
    start = Position(x_um=2.0, y_um=0.0, z_um=0.0)
    await stage.move_to(start)  # at most 20 ms
    target = Position(x_um=400.0, y_um=0.0, z_um=0.0)
    task = asyncio.create_task(stage.home() if homing else stage.move_to(target))
    await asyncio.sleep(0)  # the move has begun but not completed its first sleep
    await stage.stop()
    with pytest.raises(MotionAbortedError, match="stopped at"):
        await task
    assert stage.current_position.x_um == pytest.approx(start.x_um, abs=0.1)
    assert not stage.moving
    assert (await stage.status()).state is StageState.STOPPED


async def test_stop_when_idle_is_harmless() -> None:
    stage = await _stage()
    await stage.stop()
    assert (await stage.status()).state is StageState.IDLE


async def test_cancelled_move_freezes_in_place() -> None:
    stage = await _stage(time_scale=1.0)
    task = asyncio.create_task(stage.move_to(Position(x_um=400.0, y_um=0.0, z_um=0.0)))
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert 0.0 < stage.current_position.x_um < 400.0
    assert not stage.moving


async def test_concurrent_move_is_refused() -> None:
    stage = await _stage(time_scale=1.0)
    task = asyncio.create_task(stage.move_to(Position(x_um=100.0, y_um=0.0, z_um=0.0)))
    await asyncio.sleep(0.01)
    with pytest.raises(MotionError, match="in progress"):
        await stage.move_to(Position(x_um=0.0, y_um=10.0, z_um=0.0))
    with pytest.raises(MotionError, match="release"):
        await stage.release()
    await stage.stop()
    with pytest.raises(MotionAbortedError):
        await task


async def test_home_returns_to_origin() -> None:
    stage = await _stage(kinematics=DELTA)
    await stage.move_to(Position(x_um=50.0, y_um=40.0, z_um=30.0))
    partial = await stage.home([Axis.Z])
    assert partial.z_um == pytest.approx(0.0, abs=DELTA.max_quantization_error_um())
    assert partial.x_um == pytest.approx(50.0, abs=DELTA.max_quantization_error_um())
    assert not (await stage.status()).homed
    assert await stage.home() == DELTA.quantize(Position(x_um=0.0, y_um=0.0, z_um=0.0))
    assert (await stage.status()).homed


async def test_fault_injection_fails_the_next_move_once() -> None:
    stage = await _stage(fault_after_moves=1)
    first = Position(x_um=1.0, y_um=0.0, z_um=0.0)
    await stage.move_to(first)
    with pytest.raises(MotionError, match="injected"):
        await stage.move_to(Position(x_um=2.0, y_um=0.0, z_um=0.0))
    assert stage.current_position == first
    status = await stage.status()
    assert status.state is StageState.ERROR
    assert status.last_error is not None
    assert await stage.move_to(Position(x_um=3.0, y_um=0.0, z_um=0.0)) is not None


async def test_release_and_close() -> None:
    stage = await _stage()
    await stage.move_to(Position(x_um=1.0, y_um=0.0, z_um=0.0))
    await stage.release()
    assert stage.motors_released
    await stage.close()
    assert not stage.connected
    assert stage.version() == "simulation"


def test_invalid_construction() -> None:
    with pytest.raises(ValueError, match="time_scale"):
        SimulationStage(LIMITS, motion=MOTION, time_scale=-1.0)
    with pytest.raises(ValueError, match="fault_after_moves"):
        SimulationStage(LIMITS, motion=MOTION, time_scale=0.0, fault_after_moves=-1)
