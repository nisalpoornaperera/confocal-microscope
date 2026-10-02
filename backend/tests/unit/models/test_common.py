"""Coordinate and travel-limit models."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from confocal.models import Axis, AxisLimits, Position, StageLimits, default_stage_limits, utc_now


def _limits() -> StageLimits:
    return StageLimits(
        x=AxisLimits(min_um=-10.0, max_um=10.0),
        y=AxisLimits(min_um=0.0, max_um=5.0),
        z=AxisLimits(min_um=-1.0, max_um=1.0),
    )


# --------------------------------------------------------------------------- Position
def test_position_axis_access_and_copies() -> None:
    p = Position(x_um=1.0, y_um=2.0, z_um=3.0)
    assert [p.get(axis) for axis in Axis] == [1.0, 2.0, 3.0]
    assert p.with_axis(Axis.Y, 7).as_tuple() == (1.0, 7.0, 3.0)
    assert p.with_updates(z_um=-1.0).as_tuple() == (1.0, 2.0, -1.0)
    assert p.with_updates() == p
    assert p.offset(1.0, -2.0, 0.5).as_tuple() == (2.0, 0.0, 3.5)
    assert p.as_tuple() == (1.0, 2.0, 3.0)  # original unchanged


def test_position_distance_and_verification_error() -> None:
    a = Position(x_um=0.0, y_um=0.0, z_um=0.0)
    b = Position(x_um=3.0, y_um=4.0, z_um=-0.5)
    assert a.distance_to(b) == pytest.approx(math.sqrt(25.25))
    assert a.max_axis_error(b) == 4.0


def test_position_is_frozen_finite_and_strict() -> None:
    p = Position(x_um=0.0, y_um=0.0, z_um=0.0)
    with pytest.raises(ValidationError):
        p.x_um = 1.0  # type: ignore[misc]
    for bad in (math.nan, math.inf):
        with pytest.raises(ValidationError):
            Position(x_um=bad, y_um=0.0, z_um=0.0)
    with pytest.raises(ValidationError, match="extra"):
        Position.model_validate({"x_um": 0, "y_um": 0, "z_um": 0, "a_steps": 1})


# --------------------------------------------------------------------------- AxisLimits
def test_axis_limits_require_min_below_max() -> None:
    AxisLimits(min_um=-1.0, max_um=1.0)
    for low, high in ((1.0, 1.0), (2.0, 1.0)):
        with pytest.raises(ValidationError, match="must be <"):
            AxisLimits(min_um=low, max_um=high)
    with pytest.raises(ValidationError):
        AxisLimits(min_um=-math.inf, max_um=1.0)


def test_axis_limits_helpers() -> None:
    limits = AxisLimits(min_um=-2.0, max_um=6.0)
    assert limits.span_um == 8.0
    assert limits.contains(-2.0)
    assert limits.contains(6.0)  # inclusive
    assert not limits.contains(6.0001)
    assert not limits.contains(math.nan)
    assert limits.clamp(10.0) == 6.0
    assert limits.clamp(-10.0) == -2.0
    assert limits.clamp(1.5) == 1.5


# --------------------------------------------------------------------------- StageLimits
def test_stage_limits_violations_name_every_offending_axis() -> None:
    limits = _limits()
    assert limits.violations(Position(x_um=0.0, y_um=1.0, z_um=0.0)) == []
    assert limits.contains(Position(x_um=10.0, y_um=5.0, z_um=-1.0))
    problems = limits.violations(Position(x_um=11.0, y_um=1.0, z_um=-2.0))
    assert problems == [
        "x=11.000 um outside [-10.000, 10.000] um",
        "z=-2.000 um outside [-1.000, 1.000] um",
    ]
    assert not limits.contains(Position(x_um=0.0, y_um=-0.1, z_um=0.0))
    assert limits.for_axis(Axis.Y) == AxisLimits(min_um=0.0, max_um=5.0)


def test_default_stage_limits_are_symmetric_and_conservative() -> None:
    limits = default_stage_limits()
    for axis in Axis:
        axis_limits = limits.for_axis(axis)
        assert axis_limits.min_um == -axis_limits.max_um
    assert limits.z.span_um < limits.x.span_um


def test_utc_now_is_timezone_aware() -> None:
    assert utc_now().utcoffset() is not None
