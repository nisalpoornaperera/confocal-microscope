"""Cartesian <-> motor-step kinematics, in particular the OpenFlexure Delta Stage."""

from __future__ import annotations

import math

import numpy as np
import pytest

from confocal.hardware.kinematics import MOTORS, Motor, StageKinematics
from confocal.models.common import Axis, AxisLimits, Position, StageLimits

UM_PER_STEP = (0.08, 0.08, 0.05)


@pytest.fixture
def delta() -> StageKinematics:
    return StageKinematics.openflexure_delta(um_per_step=UM_PER_STEP)


def _openflexure_reference(
    steps: np.ndarray, h: float = 80, a: float = 50, b: float = 50
) -> np.ndarray:
    """The OpenFlexure linearised delta equations, written out independently."""
    sa, sb, sc = steps
    x = (2 / math.sqrt(3)) * (b / h) * (sa - sb)
    y = (b / h) * (sc - (sa + sb) / 2)
    z = (1 / 3) * (b / a) * (sa + sb + sc)
    return np.array([x, y, z]) * np.array(UM_PER_STEP)


def test_delta_matches_openflexure_equations(
    delta: StageKinematics, rng: np.random.Generator
) -> None:
    for steps in rng.integers(-50_000, 50_000, size=(20, 3)):
        p = delta.to_position(steps.tolist())
        np.testing.assert_allclose(p.as_tuple(), _openflexure_reference(steps), atol=1e-9)


def test_delta_pure_z_drives_all_three_motors_equally(delta: StageKinematics) -> None:
    steps = delta.exact_steps(Position(x_um=0, y_um=0, z_um=10))
    assert np.all(steps > 0)
    np.testing.assert_allclose(steps, steps[0])


def test_delta_pure_x_moves_a_and_b_oppositely(delta: StageKinematics) -> None:
    a, b, c = delta.exact_steps(Position(x_um=10, y_um=0, z_um=0))
    assert a == pytest.approx(-b)
    assert a > 0
    assert c == pytest.approx(0, abs=1e-9)


def test_delta_pure_y_moves_c_against_a_and_b(delta: StageKinematics) -> None:
    a, b, c = delta.exact_steps(Position(x_um=0, y_um=10, z_um=0))
    assert a == pytest.approx(b)
    assert c > 0 > a
    assert a + b + c == pytest.approx(0, abs=1e-9)  # no net Z


def test_round_trip_and_quantisation_bound(
    delta: StageKinematics, rng: np.random.Generator
) -> None:
    bound = delta.max_quantization_error_um()
    for xyz in rng.uniform(-3000, 3000, size=(50, 3)):
        target = Position(x_um=xyz[0], y_um=xyz[1], z_um=xyz[2])
        exact = delta.to_position(delta.exact_steps(target).tolist())
        assert exact.max_axis_error(target) < 1e-6
        assert delta.quantize(target).max_axis_error(target) <= bound + 1e-12


def test_rounding_is_half_away_from_zero() -> None:
    kin = StageKinematics.cartesian((2.0, 2.0, 2.0))
    assert kin.to_steps(Position(x_um=1.25, y_um=-1.25, z_um=0.24)) == (3, -3, 0)


def test_xy_rotation_rotates_the_xy_frame() -> None:
    base = StageKinematics.openflexure_delta(um_per_step=(0.08, 0.08, 0.08))
    rotated = StageKinematics.openflexure_delta(
        um_per_step=(0.08, 0.08, 0.08), xy_rotation_deg=90.0
    )
    steps = base.to_steps(Position(x_um=100, y_um=0, z_um=0))
    p = rotated.to_position(steps)
    assert p.x_um == pytest.approx(0, abs=0.2)
    assert abs(p.y_um) == pytest.approx(100, abs=0.2)
    assert p.z_um == pytest.approx(0, abs=0.2)


def test_cartesian_is_one_motor_per_axis() -> None:
    kin = StageKinematics.cartesian((14.3, 14.3, 20.0))
    assert kin.to_steps(Position(x_um=10, y_um=0, z_um=0)) == (143, 0, 0)
    assert kin.to_steps(Position(x_um=0, y_um=0, z_um=-1)) == (0, 0, -20)
    np.testing.assert_allclose(kin.um_per_step, np.diag([1 / 14.3, 1 / 14.3, 1 / 20.0]))


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((3, 3)),
        [[1, 2, 3], [2, 4, 6], [0, 0, 1]],
        np.eye(2),
        [[1, 0, 0], [0, 1, 0], [0, 0, np.nan]],
    ],
)
def test_singular_or_malformed_matrix_rejected(matrix: object) -> None:
    with pytest.raises(ValueError, match=r"3 x 3|singular"):
        StageKinematics(matrix)  # type: ignore[arg-type]


def test_matrices_are_read_only(delta: StageKinematics) -> None:
    m = delta.um_per_step
    m[0, 0] = 99.0  # a copy: must not affect the kinematics
    assert delta.um_per_step[0, 0] != 99.0


def test_motor_ranges_and_limit_validation(delta: StageKinematics) -> None:
    limits = StageLimits(
        x=AxisLimits(min_um=-100, max_um=100),
        y=AxisLimits(min_um=-100, max_um=100),
        z=AxisLimits(min_um=-50, max_um=50),
    )
    ranges = delta.motor_ranges(limits)
    assert set(ranges) == set(MOTORS)
    for low, high in ranges.values():
        assert low < 0 < high

    generous = dict.fromkeys(MOTORS, (-1_000_000, 1_000_000))
    assert delta.limit_violations(limits, generous) == []

    tight = dict(generous)
    tight[Motor.C] = (-10, 10)
    problems = delta.limit_violations(limits, tight)
    assert len(problems) == 1
    assert "motor c" in problems[0]


def test_any_point_inside_validated_box_is_reachable(
    delta: StageKinematics, rng: np.random.Generator
) -> None:
    limits = StageLimits(
        x=AxisLimits(min_um=-200, max_um=200),
        y=AxisLimits(min_um=-150, max_um=150),
        z=AxisLimits(min_um=-80, max_um=80),
    )
    motor_limits = {
        m: (math.floor(lo) - 1, math.ceil(hi) + 1)
        for m, (lo, hi) in delta.motor_ranges(limits).items()
    }
    assert delta.limit_violations(limits, motor_limits) == []
    for xyz in rng.uniform([-200, -150, -80], [200, 150, 80], size=(200, 3)):
        steps = delta.to_steps(Position(x_um=xyz[0], y_um=xyz[1], z_um=xyz[2]))
        for motor, value in zip(MOTORS, steps, strict=True):
            low, high = motor_limits[motor]
            assert low <= value <= high


def test_max_speed_is_limited_by_the_fastest_motor(delta: StageKinematics) -> None:
    v_max = 600.0
    for direction in (Axis.X, Axis.Y, Axis.Z, (1.0, 1.0, 0.0)):
        speed = delta.max_speed_um_s(direction, v_max)
        vector = (
            np.array([1.0 if a is direction else 0.0 for a in Axis])
            if isinstance(direction, Axis)
            else np.asarray(direction) / np.linalg.norm(direction)
        )
        rates = np.abs(delta.steps_for_displacement(*(vector * speed)))
        assert rates.max() == pytest.approx(v_max)

    per_motor = {Motor.A: 600.0, Motor.B: 600.0, Motor.C: 300.0}
    assert delta.max_speed_um_s(Axis.Z, per_motor) == pytest.approx(
        delta.max_speed_um_s(Axis.Z, 300.0)
    )
