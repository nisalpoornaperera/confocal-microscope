"""StepMapper (micrometres <-> firmware steps) and backlash planning."""

from __future__ import annotations

import logging

import pytest

from confocal.config import ArduinoConfig, KinematicsConfig, MotorConfig, Settings
from confocal.errors import LimitViolationError
from confocal.hardware.arduino.steps import StepMapper, motor_configs, plan_backlash_moves
from confocal.hardware.kinematics import Motor, MotorSteps, StageKinematics
from confocal.models.common import Axis, AxisLimits, Position, StageLimits

DELTA = StageKinematics.openflexure_delta(um_per_step=(0.06, 0.06, 0.06))


def _motors(**overrides: MotorConfig) -> dict[Motor, MotorConfig]:
    motors = {motor: MotorConfig() for motor in Motor}
    for name, cfg in overrides.items():
        motors[Motor(name)] = cfg
    return motors


def _mapper(**overrides: MotorConfig) -> StepMapper:
    return StepMapper(DELTA, _motors(**overrides))


# --------------------------------------------------------------------------- conversion


def test_pure_z_drives_all_three_motors_equally() -> None:
    steps = _mapper().to_motor_steps(Position(x_um=0.0, y_um=0.0, z_um=3.0))
    assert steps[0] == steps[1] == steps[2]
    assert steps[0] > 0


def test_xy_move_drives_motors_in_opposite_directions() -> None:
    a, b, c = _mapper().to_motor_steps(Position(x_um=10.0, y_um=0.0, z_um=0.0))
    assert a * b < 0
    assert c == 0


def test_invert_flips_only_that_motor() -> None:
    target = Position(x_um=12.0, y_um=-7.0, z_um=4.0)
    plain = _mapper().to_motor_steps(target)
    inverted = _mapper(b=MotorConfig(invert=True)).to_motor_steps(target)
    assert inverted == (plain[0], -plain[1], plain[2])


@pytest.mark.parametrize("invert", [False, True])
def test_round_trip_within_resolution(invert: bool) -> None:
    mapper = _mapper(a=MotorConfig(invert=invert), c=MotorConfig(invert=invert))
    target = Position(x_um=123.4, y_um=-56.7, z_um=8.9)
    reached = mapper.to_position(mapper.to_motor_steps(target))
    assert reached.max_axis_error(target) <= mapper.resolution_um()
    assert mapper.quantize(target) == reached
    assert mapper.to_motor_steps(reached) == mapper.to_motor_steps(target)


def test_resolution_matches_kinematics() -> None:
    assert _mapper().resolution_um() == pytest.approx(DELTA.max_quantization_error_um())
    assert 0.0 < _mapper().resolution_um() < 0.1


def test_missing_motor_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="c"):
        StepMapper(DELTA, {Motor.A: MotorConfig(), Motor.B: MotorConfig()})


def test_from_settings_uses_kinematics_and_motor_sections() -> None:
    settings = Settings(
        kinematics=KinematicsConfig(geometry="cartesian", steps_per_um=(10.0, 10.0, 20.0)),
        arduino=ArduinoConfig(c=MotorConfig(invert=True, backlash_steps=7)),
    )
    mapper = StepMapper.from_settings(settings)
    assert mapper.to_motor_steps(Position(x_um=1.0, y_um=2.0, z_um=3.0)) == (10, 20, -60)
    assert mapper.backlash == {Motor.A: 0, Motor.B: 0, Motor.C: 7}
    assert motor_configs(settings.arduino)[Motor.C].invert


# --------------------------------------------------------------------------- travel


def test_check_travel_accepts_inside_and_edges() -> None:
    mapper = _mapper(a=MotorConfig(min_steps=-100, max_steps=100))
    mapper.check_travel((100, 0, 0))
    mapper.check_travel((-100, 5, -5))


def test_check_travel_lists_every_violating_motor() -> None:
    mapper = _mapper(
        a=MotorConfig(min_steps=-100, max_steps=100), c=MotorConfig(min_steps=-10, max_steps=10)
    )
    with pytest.raises(LimitViolationError) as excinfo:
        mapper.check_travel((101, 0, -11))
    assert len(excinfo.value.violations) == 2
    assert "motor a" in excinfo.value.violations[0]
    assert "motor c" in excinfo.value.violations[1]


def test_limit_violations_honour_invert() -> None:
    limits = StageLimits(
        x=AxisLimits(min_um=-1.0, max_um=1.0),
        y=AxisLimits(min_um=-1.0, max_um=1.0),
        z=AxisLimits(min_um=0.0, max_um=600.0),  # only upwards: needs + steps on every leg
    )
    asymmetric = MotorConfig(min_steps=-200, max_steps=20_000)
    assert _mapper(a=asymmetric, b=asymmetric, c=asymmetric).limit_violations(limits) == []
    inverted = MotorConfig(min_steps=-200, max_steps=20_000, invert=True)
    problems = _mapper(a=asymmetric, b=asymmetric, c=inverted).limit_violations(limits)
    assert len(problems) == 1
    assert problems[0].startswith("motor c")


# --------------------------------------------------------------------------- backlash

BACKLASH = {Motor.A: 30, Motor.B: 30, Motor.C: 30}


def test_all_up_move_is_a_single_waypoint() -> None:
    assert plan_backlash_moves((0, 0, 0), (50, 50, 50), BACKLASH) == [(50, 50, 50)]


def test_no_motion_is_a_single_waypoint() -> None:
    assert plan_backlash_moves((5, 5, 5), (5, 5, 5), BACKLASH) == [(5, 5, 5)]


def test_downward_motors_overshoot_then_come_up() -> None:
    plan = plan_backlash_moves((100, 100, 100), (40, 40, 40), BACKLASH)
    assert plan == [(10, 10, 10), (40, 40, 40)]


def test_mixed_move_overshoots_only_downward_motors() -> None:
    plan = plan_backlash_moves((0, 100, 0), (60, 40, 0), {Motor.A: 30, Motor.B: 20, Motor.C: 9})
    assert plan == [(60, 20, 0), (60, 40, 0)]
    _assert_finishes_upwards((0, 100, 0), plan)


def test_overshoot_is_clamped_to_travel() -> None:
    travel = {Motor.A: (-50, 500), Motor.B: (-500, 500), Motor.C: (-500, 500)}
    plan = plan_backlash_moves((100, 100, 100), (-40, 0, 0), BACKLASH, travel)
    assert plan == [(-50, -30, -30), (-40, 0, 0)]


def test_target_on_travel_bottom_cannot_overshoot() -> None:
    travel = dict.fromkeys(Motor, (-50, 500))
    assert plan_backlash_moves((10, 10, 10), (-50, -50, -50), BACKLASH, travel) == [(-50, -50, -50)]


def test_motor_without_backlash_goes_straight() -> None:
    plan = plan_backlash_moves((10, 10, 10), (0, 0, 0), {Motor.A: 0, Motor.B: 5})
    assert plan == [(0, -5, 0), (0, 0, 0)]


def test_negative_backlash_is_rejected() -> None:
    with pytest.raises(ValueError, match="backlash"):
        plan_backlash_moves((10, 0, 0), (0, 0, 0), {Motor.A: -1})


def test_mapper_plan_moves_uses_configured_backlash_and_travel() -> None:
    mapper = _mapper(a=MotorConfig(backlash_steps=25, min_steps=-10, max_steps=100))
    assert mapper.plan_moves((50, 0, 0), (0, 0, 0), limits=WIDE) == [(-10, 0, 0), (0, 0, 0)]


# Cartesian box used by the backlash/limit tests (the mapper default geometry: 0.06 um/step).
BOX = StageLimits(
    x=AxisLimits(min_um=-5000.0, max_um=5000.0),
    y=AxisLimits(min_um=-5000.0, max_um=5000.0),
    z=AxisLimits(min_um=-2000.0, max_um=2000.0),
)
WIDE = StageLimits(
    x=AxisLimits(min_um=-1e6, max_um=1e6),
    y=AxisLimits(min_um=-1e6, max_um=1e6),
    z=AxisLimits(min_um=-1e6, max_um=1e6),
)
PLAY = MotorConfig(backlash_steps=60)
PLAY_INVERTED = MotorConfig(backlash_steps=60, invert=True)


def _logical(mapper: StepMapper, steps: MotorSteps) -> MotorSteps:
    """Kinematic (logical) motor steps: the firmware steps with ``invert`` undone."""
    signs = [-1 if mapper.motor_config(m).invert else 1 for m in Motor]
    a, b, c = (s * v for s, v in zip(signs, steps, strict=True))
    return (a, b, c)


def _assert_finishes_logically_upwards(
    mapper: StepMapper, start: MotorSteps, plan: list[MotorSteps]
) -> None:
    before_last = _logical(mapper, plan[-2] if len(plan) > 1 else start)
    last = _logical(mapper, plan[-1])
    assert all(end >= begin for begin, end in zip(before_last, last, strict=True))


def _assert_inside(mapper: StepMapper, limits: StageLimits, steps: MotorSteps) -> None:
    """Inside the box, allowing only the step-grid quantisation of a target on a face."""
    position = mapper.to_position(steps)
    margin = mapper.resolution_um()
    for axis in Axis:
        axis_limits = limits.for_axis(axis)
        value = position.get(axis)
        assert axis_limits.min_um - margin <= value <= axis_limits.max_um + margin, (axis, value)


@pytest.mark.parametrize("inverted", ["", "a", "b", "c", "abc"])
def test_increasing_z_never_needs_compensation_whatever_the_inversion(inverted: str) -> None:
    mapper = _mapper(**{m.value: PLAY_INVERTED if m.value in inverted else PLAY for m in Motor})
    for z in (-1500.0, 0.0, 10.0, 1500.0):
        current = mapper.to_motor_steps(Position(x_um=30.0, y_um=-20.0, z_um=z))
        target = mapper.to_motor_steps(Position(x_um=30.0, y_um=-20.0, z_um=z + 0.25))
        assert mapper.plan_moves(current, target, limits=BOX) == [target]


def test_inverted_motor_overshoots_in_its_logical_down_direction() -> None:
    mapper = _mapper(a=PLAY_INVERTED, b=PLAY, c=PLAY)
    current = mapper.to_motor_steps(Position(x_um=0.0, y_um=0.0, z_um=20.0))
    target = mapper.to_motor_steps(Position(x_um=0.0, y_um=0.0, z_um=10.0))
    plan = mapper.plan_moves(current, target, limits=BOX)
    assert len(plan) == 2
    assert plan[-1] == target
    # Motor a is inverted: logically down is firmware +, so its overshoot is above the target.
    assert plan[0] == (target[0] + 60, target[1] - 60, target[2] - 60)
    _assert_finishes_logically_upwards(mapper, current, plan)


def test_overshoot_waypoint_never_leaves_the_cartesian_limits() -> None:
    mapper = _mapper(a=PLAY, b=PLAY, c=PLAY)
    current = mapper.to_motor_steps(Position(x_um=4900.0, y_um=0.0, z_um=-1950.0))
    target = mapper.to_motor_steps(Position(x_um=5000.0, y_um=0.0, z_um=-2000.0))
    plan = mapper.plan_moves(current, target, limits=BOX)
    assert plan[-1] == target
    for waypoint in plan:
        _assert_inside(mapper, BOX, waypoint)


def test_overshoot_is_shrunk_to_fit_inside_the_limits(caplog: pytest.LogCaptureFixture) -> None:
    mapper = _mapper(a=PLAY, b=PLAY, c=PLAY)
    current = mapper.to_motor_steps(Position(x_um=0.0, y_um=0.0, z_um=-1900.0))
    target = mapper.to_motor_steps(Position(x_um=0.0, y_um=0.0, z_um=-1998.0))  # 2 um of room
    full = plan_backlash_moves(current, target, mapper.backlash, mapper.travel)
    assert mapper.to_position(full[0]).z_um < -2000.0  # the full overshoot would leave the box
    with caplog.at_level(logging.WARNING, logger="confocal.hardware.arduino.steps"):
        plan = mapper.plan_moves(current, target, limits=BOX)
    assert len(plan) == 2
    assert plan[-1] == target
    _assert_inside(mapper, BOX, plan[0])
    assert all(t - 60 < w < t for w, t in zip(plan[0], target, strict=True))  # shrunk, not lost
    _assert_finishes_logically_upwards(mapper, current, plan)
    assert "shrunk" in caplog.text


def test_overshoot_without_room_is_skipped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mapper = _mapper(a=PLAY, b=PLAY, c=PLAY)
    current = mapper.to_motor_steps(Position(x_um=0.0, y_um=0.0, z_um=-1900.0))
    target = mapper.to_motor_steps(Position(x_um=0.0, y_um=0.0, z_um=-2000.0))  # on the face
    with caplog.at_level(logging.WARNING, logger="confocal.hardware.arduino.steps"):
        plan = mapper.plan_moves(current, target, limits=BOX)
    assert plan == [target]
    assert "skipped" in caplog.text


def test_plan_signs_and_admissibility_in_the_pure_function() -> None:
    # Motor b inverted: moving firmware + is logically down, so it overshoots firmware +.
    plan = plan_backlash_moves((0, 0, 0), (-40, 40, 40), BACKLASH, signs=(1, -1, 1))
    assert plan == [(-70, 70, 40), (-40, 40, 40)]
    # Nothing admissible except the target itself: the overshoot is dropped.
    plan = plan_backlash_moves(
        (0, 0, 0), (-40, 40, 40), BACKLASH, signs=(1, -1, 1), admissible=lambda s: False
    )
    assert plan == [(-40, 40, 40)]
    with pytest.raises(ValueError, match="sign"):
        plan_backlash_moves((0, 0, 0), (1, 1, 1), BACKLASH, signs=(1, 0, 1))


def _assert_finishes_upwards(start: MotorSteps, plan: list[MotorSteps]) -> None:
    """In the last leg of the plan no motor travels in the - direction."""
    before_last = plan[-2] if len(plan) > 1 else start
    assert all(end >= begin for begin, end in zip(before_last, plan[-1], strict=True))
