"""Motion safety: limits before motion, verification, timeout, failure handling."""

from __future__ import annotations

import math
import time

import pytest
from tests.unit.microscope.conftest import (
    LIMITS,
    CountingStage,
    FaultyStage,
    HangingStage,
    InMemoryCalibrationStore,
    LyingStage,
    RigFactory,
    ScriptedADC,
)

from confocal.config import MotionConfig
from confocal.errors import (
    LimitViolationError,
    MotionError,
    MotionTimeoutError,
    MotionVerificationError,
)
from confocal.hardware.kinematics import StageKinematics
from confocal.hardware.simulation import ManualLaser, NullCamera, SimulationStage
from confocal.microscope import StandardMicroscopeController
from confocal.models.common import Axis, Position, StageLimits


async def test_move_keeps_unspecified_axes_and_returns_reported_position(
    make_rig: RigFactory,
) -> None:
    rig = await make_rig()
    first = await rig.controller.move_to(x_um=100.0, y_um=-50.0, z_um=3.0)
    assert first == Position(x_um=100.0, y_um=-50.0, z_um=3.0)
    second = await rig.controller.move_to(z_um=7.5)
    assert second == Position(x_um=100.0, y_um=-50.0, z_um=7.5)
    assert rig.controller.last_position == second


async def test_move_relative_offsets_the_verified_position(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.move_to(x_um=10.0, y_um=10.0, z_um=10.0)
    position = await rig.controller.move_relative(dx_um=1.0, dz_um=-2.0)
    assert position == Position(x_um=11.0, y_um=10.0, z_um=8.0)


@pytest.mark.parametrize(
    "target",
    [{"x_um": 1000.5}, {"y_um": -2000.0}, {"z_um": 250.0}],
)
async def test_limit_violation_never_calls_the_stage(
    make_rig: RigFactory, target: dict[str, float]
) -> None:
    rig = await make_rig(stage_cls=CountingStage)
    stage = rig.stage
    assert isinstance(stage, CountingStage)
    with pytest.raises(LimitViolationError) as info:
        await rig.controller.move_to(**target)
    assert info.value.violations
    assert stage.targets == []
    assert stage.moves_started == 0


async def test_relative_move_beyond_limits_is_refused(make_rig: RigFactory) -> None:
    rig = await make_rig(stage_cls=CountingStage)
    await rig.controller.move_to(z_um=195.0)
    with pytest.raises(LimitViolationError):
        await rig.controller.move_relative(dz_um=10.0)
    assert isinstance(rig.stage, CountingStage)
    assert len(rig.stage.targets) == 1


@pytest.mark.parametrize("value", [math.nan, math.inf])
async def test_non_finite_target_is_a_limit_violation(make_rig: RigFactory, value: float) -> None:
    rig = await make_rig(stage_cls=CountingStage)
    with pytest.raises(LimitViolationError):
        await rig.controller.move_to(x_um=value)
    with pytest.raises(LimitViolationError):
        await rig.controller.move_relative(dy_um=value)
    assert isinstance(rig.stage, CountingStage)
    assert rig.stage.targets == []


async def test_lying_stage_fails_verification_and_is_stopped(make_rig: RigFactory) -> None:
    rig = await make_rig(stage_cls=LyingStage, motion=MotionConfig(position_tolerance_um=0.5))
    stage = rig.stage
    assert isinstance(stage, LyingStage)
    with pytest.raises(MotionVerificationError, match=r"x off by 5.000 um"):
        await rig.controller.move_to(x_um=20.0)
    assert stage.stop_calls >= 1
    assert rig.controller.last_position is None
    status = await rig.controller.status()
    assert status.stage.last_error is not None
    assert "x off by" in status.stage.last_error


async def test_error_within_tolerance_is_accepted(make_rig: RigFactory) -> None:
    rig = await make_rig(stage_cls=LyingStage, motion=MotionConfig(position_tolerance_um=0.5))
    assert isinstance(rig.stage, LyingStage)
    rig.stage.error_um = 0.3
    position = await rig.controller.move_to(x_um=20.0)
    assert position.x_um == pytest.approx(20.3)


async def test_timeout_stops_the_stage(make_rig: RigFactory) -> None:
    rig = await make_rig(stage_cls=HangingStage, motion=MotionConfig(move_timeout_s=0.05))
    stage = rig.stage
    assert isinstance(stage, HangingStage)
    with pytest.raises(MotionTimeoutError):
        await rig.controller.move_to(z_um=1.0)
    assert len(stage.targets) == 1
    assert stage.stop_calls >= 1
    # The lock was released: the controller is usable again.
    assert await rig.controller.get_position() == Position(x_um=0.0, y_um=0.0, z_um=0.0)


async def test_hardware_error_during_motion_stops_and_reraises(make_rig: RigFactory) -> None:
    rig = await make_rig(stage_cls=FaultyStage)
    await rig.controller.move_to(z_um=1.0)
    with pytest.raises(MotionError, match="injected"):
        await rig.controller.move_to(z_um=2.0)
    assert isinstance(rig.stage, FaultyStage)
    assert rig.stage.stop_calls >= 1
    assert rig.controller.last_position is None
    # The next move re-reads the device position for the unspecified axes.
    assert await rig.controller.move_to(z_um=3.0) == Position(x_um=0.0, y_um=0.0, z_um=3.0)


async def test_home_verifies_the_origin(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.move_to(x_um=50.0, y_um=40.0, z_um=30.0)
    position = await rig.controller.home([Axis.Z])
    assert position == Position(x_um=50.0, y_um=40.0, z_um=0.0)
    assert await rig.controller.home() == Position(x_um=0.0, y_um=0.0, z_um=0.0)


async def test_home_that_misses_the_origin_fails_verification(make_rig: RigFactory) -> None:
    class OffsetHomeStage(SimulationStage):
        async def home(self, axes: object = None) -> Position:
            await super().home()
            return Position(x_um=0.0, y_um=0.0, z_um=2.0)

    rig = await make_rig(stage_cls=OffsetHomeStage)
    with pytest.raises(MotionVerificationError, match="z off by"):
        await rig.controller.home()


async def test_wait_settle_scales_and_validates(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.wait_settle(10.0)  # time_scale 0: returns at once
    with pytest.raises(ValueError, match="settle time"):
        await rig.controller.wait_settle(-1.0)


@pytest.mark.parametrize(("warmup", "scale"), [(-1.0, 1.0), (0.0, -1.0), (math.inf, 1.0)])
def test_constructor_rejects_bad_timing(warmup: float, scale: float) -> None:
    stage = SimulationStage(LIMITS, motion=MotionConfig(), time_scale=0.0)
    with pytest.raises(ValueError, match="must be finite"):
        StandardMicroscopeController(
            stage,
            ScriptedADC([0]),
            ManualLaser(),
            NullCamera(),
            limits=LIMITS,
            motion=MotionConfig(),
            calibration_store=InMemoryCalibrationStore(),
            laser_warmup_s=warmup,
            time_scale=scale,
        )


# --------------------------------------------------------------------------- limit faces

DELTA = StageKinematics.openflexure_delta(um_per_step=(0.06, 0.06, 0.06))


class _QuantisingStage(SimulationStage):
    """Simulation stage on the delta step grid (reports quantised positions), recording targets."""

    def __init__(self, limits: StageLimits, *, motion: MotionConfig, time_scale: float) -> None:
        super().__init__(limits, motion=motion, time_scale=time_scale, kinematics=DELTA)
        self.targets: list[Position] = []

    async def move_to(self, target: Position) -> Position:
        self.targets.append(target)
        return await super().move_to(target)


def _face_point_quantised_outside() -> Position:
    """A point on the +X limit face whose step-grid position lies just outside the limits."""
    for i in range(400):
        for j in range(20):
            point = Position(x_um=LIMITS.x.max_um, y_um=-50.0 + 0.0371 * i, z_um=3.0 + 0.0173 * j)
            if DELTA.quantize(point).x_um > LIMITS.x.max_um:
                return point
    raise AssertionError("no quantised point outside the face found")


async def test_unspecified_axes_come_from_the_commanded_target_on_a_limit_face(
    make_rig: RigFactory,
) -> None:
    rig = await make_rig(stage_cls=_QuantisingStage)
    stage = rig.stage
    assert isinstance(stage, _QuantisingStage)
    face = _face_point_quantised_outside()
    reported = await rig.controller.move_to(x_um=face.x_um, y_um=face.y_um, z_um=face.z_um)
    assert reported.x_um > LIMITS.x.max_um  # the read-back is quantised outside the limit...
    assert rig.controller.last_position == reported  # ...and is what is reported
    # ...but the next moves keep X from the commanded target and are not refused.
    assert (await rig.controller.move_to(z_um=1.0)).z_um == pytest.approx(1.0, abs=0.1)
    await rig.controller.move_relative(dz_um=1.0)
    assert [t.x_um for t in stage.targets] == [LIMITS.x.max_um] * 3
    assert stage.targets[-1].z_um == pytest.approx(2.0)
    assert all(LIMITS.contains(t) for t in stage.targets)


async def test_read_back_quantised_outside_a_limit_is_pulled_onto_it(
    make_rig: RigFactory,
) -> None:
    rig = await make_rig(stage_cls=_QuantisingStage)
    stage = rig.stage
    assert isinstance(stage, _QuantisingStage)
    face = _face_point_quantised_outside()
    await rig.controller.move_to(x_um=face.x_um, y_um=face.y_um, z_um=face.z_um)
    await rig.controller.emergency_stop("test")
    reset = await rig.controller.reset_emergency_stop()  # position known only from the device
    assert reset.x_um > LIMITS.x.max_um
    await rig.controller.move_to(z_um=1.0)
    assert stage.targets[-1].x_um == LIMITS.x.max_um
    assert LIMITS.contains(stage.targets[-1])


async def test_read_back_well_outside_a_limit_is_still_refused(make_rig: RigFactory) -> None:
    class OutsideStage(CountingStage):
        async def get_position(self) -> Position:
            return Position(x_um=LIMITS.x.max_um + 2.0, y_um=0.0, z_um=0.0)

    rig = await make_rig(stage_cls=OutsideStage, motion=MotionConfig(position_tolerance_um=0.5))
    with pytest.raises(LimitViolationError, match="x="):
        await rig.controller.move_to(z_um=1.0)
    assert isinstance(rig.stage, CountingStage)
    assert rig.stage.targets == []


# --------------------------------------------------------------------------- timeouts


def test_move_timeout_is_distance_aware_and_keeps_the_toml_key() -> None:
    motion = MotionConfig.model_validate(
        {"move_timeout_s": 7.0, "xy_speed_um_s": 40.0, "z_speed_um_s": 20.0}
    )
    origin = Position(x_um=0.0, y_um=0.0, z_um=0.0)
    assert motion.move_timeout_for(origin, origin) == pytest.approx(7.0)
    assert motion.expected_move_s(origin, Position(x_um=400.0, y_um=-80.0, z_um=100.0)) == (
        pytest.approx(10.0)
    )
    assert motion.move_timeout_for(origin, Position(x_um=0.0, y_um=0.0, z_um=-100.0)) == (
        pytest.approx(2.0 * 5.0 + 7.0)
    )
    # A full-range XY move of the default machine (10 mm at 40 um/s = 250 s) fits its timeout.
    full = MotionConfig()
    span = Position(x_um=10_000.0, y_um=10_000.0, z_um=0.0)
    assert full.move_timeout_for(origin, span) > full.expected_move_s(origin, span) == 250.0


LONG_MOVE = MotionConfig(move_timeout_s=0.05, xy_speed_um_s=1000.0)  # 300 um: 0.3 s >> 0.05 s


async def test_long_move_is_not_mistaken_for_a_stall(make_rig: RigFactory) -> None:
    rig = await make_rig(stage_time_scale=1.0, motion=LONG_MOVE)
    started = time.monotonic()
    position = await rig.controller.move_to(x_um=300.0)
    assert position.x_um == pytest.approx(300.0)
    assert time.monotonic() - started >= 0.25


async def test_stalled_stage_times_out_after_the_distance_aware_bound(
    make_rig: RigFactory,
) -> None:
    rig = await make_rig(stage_cls=HangingStage, motion=LONG_MOVE)
    started = time.monotonic()
    with pytest.raises(MotionTimeoutError, match="did not finish"):
        await rig.controller.move_to(x_um=300.0)
    elapsed = time.monotonic() - started
    assert 0.6 <= elapsed < 3.0  # 2 x 0.3 s + 0.05 s
    assert isinstance(rig.stage, HangingStage)
    assert rig.stage.stop_calls >= 1
