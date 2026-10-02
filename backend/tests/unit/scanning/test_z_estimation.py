"""Adaptive Z estimation: previous point, previous-row neighbour, default."""

from __future__ import annotations

import math

import pytest
from tests.unit.scanning.conftest import small_config

from confocal.models import ScanOrder
from confocal.scanning.plan import GridPoint, grid_axes, order_points
from confocal.scanning.z_estimation import AdaptiveZEstimator


def _setup(
    order: ScanOrder = ScanOrder.SERPENTINE, **overrides: object
) -> tuple[AdaptiveZEstimator, list[GridPoint]]:
    config = small_config(x_stop_um=30.0, y_stop_um=20.0, order=order, **overrides)
    xs, ys = grid_axes(config)
    return AdaptiveZEstimator(config, n_x=xs.size, n_y=ys.size), order_points(xs, ys, order)


def test_first_point_uses_the_default_range() -> None:
    estimator, points = _setup()
    estimate = estimator.estimate(points[0])
    assert estimate.source == "default"
    assert estimate.center_um == 0.0
    assert estimate.width_um == 40.0


def test_previous_valid_point_gives_a_narrow_range() -> None:
    estimator, points = _setup()
    estimator.record(points[0], 3.5)
    estimate = estimator.estimate(points[1])
    assert estimate.source == "previous"
    assert estimate.center_um == 3.5
    assert estimate.width_um == 16.0


def test_invalid_previous_point_falls_back_to_the_previous_row() -> None:
    estimator, points = _setup()
    for p in points[:4]:  # row 0: ix 0..3
        estimator.record(p, 1.0 + p.ix)
    estimator.record(points[4], None)  # row 1 starts at ix 3 (serpentine) and fails
    estimate = estimator.estimate(points[5])  # ix 2, iy 1
    assert estimate.source == "neighbour"
    assert estimate.center_um == 3.0  # directly above: ix 2 of row 0


def test_neighbour_search_prefers_the_cell_above_then_its_sides() -> None:
    estimator, points = _setup()
    row0 = {p.ix: p for p in points[:4]}
    estimator.record(row0[0], 10.0)
    estimator.record(row0[1], None)
    estimator.record(row0[2], 30.0)
    estimator.record(row0[3], None)
    target = next(p for p in points if p.iy == 1 and p.ix == 1)
    estimator.record(points[4], None)  # make the predecessor invalid
    assert estimator.estimate(target).center_um == 10.0  # ix 0 before ix 2 on a tie
    far = next(p for p in points if p.iy == 1 and p.ix == 3)
    assert estimator.estimate(far).center_um == 30.0


def test_no_valid_neighbour_uses_the_default() -> None:
    estimator, points = _setup()
    for p in points[:5]:
        estimator.record(p, None)
    assert estimator.estimate(points[5]).source == "default"


def test_raster_row_start_does_not_use_the_far_predecessor() -> None:
    estimator, points = _setup(ScanOrder.RASTER)
    for p in points[:4]:
        estimator.record(p, 2.0 * p.ix)
    first_of_row1 = points[4]
    assert (first_of_row1.ix, first_of_row1.iy) == (0, 1)
    estimate = estimator.estimate(first_of_row1)
    assert estimate.source == "neighbour"  # the predecessor (ix 3) is not adjacent
    assert estimate.center_um == 0.0


def test_adaptive_z_disabled_always_uses_the_default() -> None:
    estimator, points = _setup(adaptive_z=False)
    estimator.record(points[0], 3.0)
    estimate = estimator.estimate(points[1])
    assert estimate.source == "default"
    assert estimate.width_um == 40.0


def test_non_finite_surface_values_are_ignored() -> None:
    estimator, points = _setup()
    estimator.record(points[0], math.nan)
    assert estimator.surface_z(0, 0) is None
    assert estimator.estimate(points[1]).source == "default"


def test_grid_shape_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AdaptiveZEstimator(small_config(), n_x=0, n_y=1)
