"""ScanPlan: grid generation, acquisition order, Z sweeps, limit validation and estimates."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from tests.unit.scanning.conftest import FAKE_LIMITS, small_config

from confocal.config import MotionConfig
from confocal.errors import LimitViolationError
from confocal.models import AxisLimits, ProcessingConfig, ScanMode, ScanOrder, StageLimits
from confocal.scanning.plan import (
    MOTOR_RESOLUTION_UM,
    GridPoint,
    ScanPlan,
    axis_positions,
    grid_axes,
    order_points,
    sweep_is_clipped,
    sweep_length,
    z_sweep,
)

Z_LIMITS = AxisLimits(min_um=-50.0, max_um=50.0)


def _plan(**overrides: object) -> ScanPlan:
    return ScanPlan.from_config(small_config(**overrides), FAKE_LIMITS, MotionConfig(), 860)


# --------------------------------------------------------------------------- grid
def test_grid_counts_include_both_ends() -> None:
    xs, ys = grid_axes(small_config(x_stop_um=100.0, y_stop_um=50.0))
    assert xs.size == 11
    assert ys.size == 6
    assert xs[0] == 0.0
    assert xs[-1] == 100.0
    assert ys[-1] == 50.0


def test_grid_is_robust_to_float_steps() -> None:
    positions = axis_positions(0.0, 1.0, 0.1)
    assert positions.size == 11
    assert positions[-1] <= 1.0
    assert positions[-1] == pytest.approx(1.0)
    # start + i * step, never accumulated
    np.testing.assert_array_equal(positions[:10], 0.0 + np.arange(10) * 0.1)


def test_grid_last_position_never_beyond_stop() -> None:
    positions = axis_positions(0.0, 10.5, 2.0)
    np.testing.assert_allclose(positions, [0, 2, 4, 6, 8, 10])
    tiny = axis_positions(0.3, 0.3 + 3 * 0.1, 0.1)
    assert tiny.size == 4
    assert tiny[-1] <= 0.3 + 3 * 0.1


def test_zero_length_axis_gives_one_point() -> None:
    xs, ys = grid_axes(small_config(x_start_um=5.0, x_stop_um=5.0))
    assert xs.tolist() == [5.0]
    assert ys.size == 2


# --------------------------------------------------------------------------- order
def test_serpentine_reverses_odd_rows_and_keeps_neighbours_adjacent() -> None:
    points = order_points(np.arange(4.0), np.arange(3.0), ScanOrder.SERPENTINE)
    assert [p.point_id for p in points] == list(range(12))
    assert [p.ix for p in points[:4]] == [0, 1, 2, 3]
    assert [p.ix for p in points[4:8]] == [3, 2, 1, 0]
    assert [p.ix for p in points[8:]] == [0, 1, 2, 3]
    for a, b in itertools.pairwise(points):
        assert abs(a.ix - b.ix) + abs(a.iy - b.iy) == 1


def test_raster_keeps_every_row_in_the_same_direction() -> None:
    points = order_points([0.0, 5.0, 10.0], [0.0, 5.0], ScanOrder.RASTER)
    assert [(p.ix, p.iy) for p in points] == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    assert points[3] == GridPoint(point_id=3, ix=0, iy=1, x_um=0.0, y_um=5.0)


def test_every_grid_cell_is_visited_exactly_once() -> None:
    plan = _plan(x_stop_um=40.0, y_stop_um=30.0)
    cells = {(p.ix, p.iy) for p in plan.points}
    assert len(cells) == plan.total_points == plan.n_x * plan.n_y == 20
    for p in plan.points:
        assert p.x_um == plan.x_positions[p.ix]
        assert p.y_um == plan.y_positions[p.iy]


def test_order_points_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        order_points([], [0.0], ScanOrder.RASTER)


# --------------------------------------------------------------------------- z sweeps
def test_z_sweep_is_increasing_uniform_and_includes_both_ends() -> None:
    z = z_sweep(10.0, 12.0, 0.25, Z_LIMITS)
    assert z[0] == 4.0
    assert z[-1] == 16.0
    assert z.size == 49 == sweep_length(12.0, 0.25)
    assert np.all(np.diff(z) > 0)
    np.testing.assert_allclose(np.diff(z), 0.25)


def test_z_sweep_never_exceeds_the_requested_step() -> None:
    z = z_sweep(0.0, 10.0, 3.0, Z_LIMITS)
    assert z[0] == -5.0
    assert z[-1] == 5.0
    assert np.max(np.diff(z)) <= 3.0
    np.testing.assert_allclose(np.diff(z), np.diff(z)[0])


def test_z_sweep_is_clipped_to_limits() -> None:
    z = z_sweep(45.0, 20.0, 2.0, Z_LIMITS)
    assert z[0] == 35.0
    assert z[-1] == 50.0
    assert np.all(z <= Z_LIMITS.max_um)
    assert sweep_is_clipped(45.0, 20.0, Z_LIMITS)
    assert not sweep_is_clipped(0.0, 20.0, Z_LIMITS)


def test_z_sweep_degenerate_cases() -> None:
    assert z_sweep(80.0, 20.0, 2.0, Z_LIMITS).size == 0  # entirely outside the limits
    np.testing.assert_array_equal(z_sweep(60.0, 20.0, 2.0, Z_LIMITS), [50.0])  # touches a limit
    with pytest.raises(ValueError, match="finite"):
        z_sweep(float("nan"), 10.0, 1.0, Z_LIMITS)
    with pytest.raises(ValueError, match="width"):
        z_sweep(0.0, 0.0, 1.0, Z_LIMITS)
    with pytest.raises(ValueError, match="step"):
        z_sweep(0.0, 10.0, -1.0, Z_LIMITS)


# --------------------------------------------------------------------------- limits
def test_validate_limits_accepts_a_safe_scan() -> None:
    plan = _plan()
    plan.validate_limits()
    assert plan.limit_violations() == []
    assert plan.sweep_limits() == AxisLimits(min_um=-20.0, max_um=20.0)


def test_validate_limits_reports_the_xy_envelope() -> None:
    plan = _plan(x_start_um=990.0, x_stop_um=1010.0)
    with pytest.raises(LimitViolationError) as info:
        plan.validate_limits()
    assert len(info.value.violations) == 1
    assert info.value.violations[0].startswith("x range")


def test_validate_limits_checks_the_full_z_envelope() -> None:
    plan = _plan(z_center_um=90.0, z_range_um=40.0)  # 70 .. 110, limit 100
    with pytest.raises(LimitViolationError) as info:
        plan.validate_limits()
    assert info.value.violations[0].startswith("z range [70.000, 110.000]")
    with pytest.raises(LimitViolationError):
        _plan(z_center_um=150.0, z_range_um=40.0).sweep_limits()


def test_fixed_z_checks_only_z_center() -> None:
    plan = _plan(mode=ScanMode.FIXED_Z, z_center_um=90.0, z_range_um=40.0)
    plan.validate_limits()
    assert plan.z_envelope() == (90.0, 90.0)
    assert _plan(mode=ScanMode.FIXED_Z, z_center_um=101.0).limit_violations()
    with pytest.raises(ValueError, match="fixed-Z"):
        plan.sweep_limits()


def test_plan_requires_a_positive_data_rate() -> None:
    with pytest.raises(ValueError, match="data_rate"):
        ScanPlan.from_config(small_config(), FAKE_LIMITS, MotionConfig(), 0)


# --------------------------------------------------------------------------- estimate
def test_estimate_confocal_math() -> None:
    motion = MotionConfig(
        xy_speed_um_s=10.0, z_speed_um_s=5.0, per_move_overhead_s=0.1, position_tolerance_um=0.5
    )
    config = small_config(settle_time_ms=20.0, samples_per_z=8)
    plan = ScanPlan.from_config(config, FAKE_LIMITS, motion, 100)
    estimate = plan.estimate()
    first = 21 + 17  # full coarse range + fine
    typical = 9 + 17  # adaptive coarse range + fine
    measurements = first + 5 * typical
    assert estimate.n_x == 3
    assert estimate.n_y == 2
    assert estimate.total_points == 6
    assert estimate.z_positions_per_point == typical
    assert estimate.total_measurements == measurements
    assert estimate.total_adc_samples == measurements * 8
    xy_s = (10 + 10 + 10 + 10 + 10) / 10.0  # serpentine: four X steps and one Y step
    z_s = (2 * (40 + 8) + 5 * 2 * (16 + 8)) / 5.0
    overhead_s = (6 + measurements) * 0.1
    dwell_s = measurements * (0.020 + 8 / 100)
    assert estimate.estimated_duration_s == pytest.approx(xy_s + z_s + overhead_s + dwell_s)
    assert estimate.estimated_data_bytes == measurements * (8 * 12 + 49) + 6 * 24
    assert estimate.within_limits
    assert estimate.limit_violations == []
    assert estimate.warnings == []


def test_estimate_without_adaptive_z_sweeps_the_full_range_everywhere() -> None:
    estimate = _plan(adaptive_z=False).estimate()
    assert estimate.z_positions_per_point == 21 + 17
    assert estimate.total_measurements == 6 * (21 + 17)


def test_estimate_fixed_z_has_one_position_per_point() -> None:
    estimate = _plan(mode=ScanMode.FIXED_Z).estimate()
    assert estimate.z_positions_per_point == 1
    assert estimate.total_measurements == 6


def test_estimate_reports_limit_violations_without_raising() -> None:
    estimate = _plan(x_stop_um=2000.0, xy_step_um=500.0).estimate()
    assert not estimate.within_limits
    assert estimate.limit_violations


def test_estimate_warnings() -> None:
    fine = small_config(fine_z_step_um=MOTOR_RESOLUTION_UM / 2)
    plan = ScanPlan.from_config(fine, FAKE_LIMITS, MotionConfig(), 860)
    assert any("motor resolution" in w for w in plan.estimate().warnings)

    slow = MotionConfig(z_speed_um_s=0.001)
    plan = ScanPlan.from_config(small_config(), FAKE_LIMITS, slow, 860)
    assert any("very long scan" in w for w in plan.estimate().warnings)

    coarse = small_config(processing=ProcessingConfig(expected_fwhm_um=3.0))
    plan = ScanPlan.from_config(coarse, FAKE_LIMITS, MotionConfig(), 860)
    assert any("step over the peak" in w for w in plan.estimate().warnings)


def test_plan_uses_the_given_limits() -> None:
    narrow = StageLimits(
        x=AxisLimits(min_um=0.0, max_um=10.0),
        y=AxisLimits(min_um=0.0, max_um=10.0),
        z=AxisLimits(min_um=-5.0, max_um=5.0),
    )
    plan = ScanPlan.from_config(small_config(), narrow, MotionConfig(), 860)
    assert {v.split(" ")[0] for v in plan.limit_violations()} == {"x", "z"}
