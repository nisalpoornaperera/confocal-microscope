"""Surface reconstruction: classification, interpolation, gaps, mesh, statistics."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable

import numpy as np
import pytest

from confocal.errors import ReconstructionError
from confocal.models.processing import PointStatus
from confocal.models.scan import ScanPoint
from confocal.models.surface import (
    InterpolationMethod,
    OutlierMethod,
    PointClassification,
    ReconstructionRequest,
)
from confocal.surface import (
    classify_points,
    fit_plane,
    global_mad_outliers,
    height_parameters,
    local_mad_outliers,
    reconstruct_surface,
)
from confocal.surface.gaps import missing_grid_positions

STEP = 2.0
HeightFn = Callable[[float, float], float]


def make_points(
    n: int,
    height: HeightFn,
    *,
    step: float = STEP,
    skip: Callable[[int, int], bool] = lambda ix, iy: False,
    confidence: float = 0.9,
) -> list[ScanPoint]:
    points: list[ScanPoint] = []
    for iy in range(n):
        for ix in range(n):
            if skip(ix, iy):
                continue
            x, y = ix * step, iy * step
            points.append(
                ScanPoint(
                    point_id=len(points),
                    ix=ix,
                    iy=iy,
                    x_um=x,
                    y_um=y,
                    status=PointStatus.VALID,
                    surface_z_um=height(x, y),
                    confidence=confidence,
                )
            )
    return points


def plane(x: float, y: float) -> float:
    return 0.05 * x - 0.02 * y + 3.0


def grid_array(values: list[list[float | None]]) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in values])


def assert_json_safe(result: object) -> None:
    dumped = result.model_dump_json()  # type: ignore[attr-defined]
    assert "NaN" not in dumped
    assert "Infinity" not in dumped
    json.loads(dumped)


# --------------------------------------------------------------------------- classification


def test_classification_of_every_category() -> None:
    points = make_points(6, plane)
    points[0] = points[0].model_copy(update={"status": PointStatus.NO_PEAK})
    points[1] = points[1].model_copy(update={"surface_z_um": None})
    points[2] = points[2].model_copy(update={"surface_z_um": math.nan})
    points[3] = points[3].model_copy(update={"confidence": 0.1})
    points[20] = points[20].model_copy(update={"surface_z_um": 50.0})
    classified = classify_points(points, ReconstructionRequest())
    c = classified.classification
    assert c[0] is c[1] is c[2] is PointClassification.INVALID
    assert c[3] is PointClassification.LOW_CONFIDENCE
    assert c[20] is PointClassification.OUTLIER
    assert classified.count(PointClassification.USED) == len(points) - 5


@pytest.mark.parametrize("method", [OutlierMethod.LOCAL_MAD, OutlierMethod.GLOBAL_MAD])
def test_outlier_detection_finds_spikes_only(method: OutlierMethod) -> None:
    rng = np.random.default_rng(11)
    gx, gy = np.meshgrid(np.arange(20) * STEP, np.arange(20) * STEP)
    xy = np.column_stack((gx.ravel(), gy.ravel()))
    z = 0.03 * xy[:, 0] + 0.01 * xy[:, 1] + rng.normal(0.0, 0.02, xy.shape[0])
    spikes = [0, 57, 210, 399]  # includes two corners
    z[spikes] += [2.0, -1.5, 3.0, 1.0]
    if method is OutlierMethod.LOCAL_MAD:
        flagged = local_mad_outliers(xy, z, neighbours=8, threshold=5.0)
    else:
        flagged = global_mad_outliers(xy, z, threshold=5.0)
    assert set(np.flatnonzero(flagged)) == set(spikes)


def test_local_mad_exact_on_tilted_plane_borders() -> None:
    gx, gy = np.meshgrid(np.arange(30) * STEP, np.arange(30) * STEP)
    xy = np.column_stack((gx.ravel(), gy.ravel()))
    z = 0.2 * xy[:, 0] - 0.1 * xy[:, 1]
    assert not local_mad_outliers(xy, z, neighbours=8, threshold=3.5).any()


def test_outlier_method_none_keeps_spikes() -> None:
    points = make_points(6, plane)
    points[14] = points[14].model_copy(update={"surface_z_um": 80.0})
    classified = classify_points(points, ReconstructionRequest(outlier_method=OutlierMethod.NONE))
    assert classified.count(PointClassification.OUTLIER) == 0


# --------------------------------------------------------------------------- interpolation


@pytest.mark.parametrize("method", list(InterpolationMethod))
def test_plane_is_reproduced(method: InterpolationMethod) -> None:
    points = make_points(8, plane, skip=lambda ix, iy: (ix + iy) % 3 == 0 and 0 < ix < 7)
    # The skipped points are gaps; fill_gaps keeps their interpolated values.
    request = ReconstructionRequest(
        method=method, grid_step_um=1.0, max_gap_distance_um=10.0, fill_gaps=True
    )
    result = reconstruct_surface("s", points, request, xy_step_um=STEP)
    assert np.array(result.gap_mask).any()
    z = grid_array(result.z_um)
    gx, gy = np.meshgrid(result.x_um, result.y_um)
    expected = 0.05 * gx - 0.02 * gy + 3.0
    defined = np.isfinite(z)
    assert defined.mean() > 0.95
    # nearest: up to one scan step away on a 0.05 um/um slope.
    tol = 0.11 if method is InterpolationMethod.NEAREST else 1e-6
    np.testing.assert_allclose(z[defined], expected[defined], atol=tol)
    assert_json_safe(result)


def test_nearest_takes_the_closest_point_value() -> None:
    points = make_points(3, lambda x, y: float(x + 10 * y))
    request = ReconstructionRequest(
        method=InterpolationMethod.NEAREST, outlier_method=OutlierMethod.NONE
    )
    result = reconstruct_surface("s", points, request, xy_step_um=STEP)
    np.testing.assert_allclose(grid_array(result.z_um), [[0, 2, 4], [20, 22, 24], [40, 42, 44]])


# --------------------------------------------------------------------------- gaps


def test_gap_detection_on_a_holed_grid() -> None:
    def hole(ix: int, iy: int) -> bool:
        return 3 <= ix <= 5 and 4 <= iy <= 6

    points = make_points(12, plane, skip=hole)
    result = reconstruct_surface("s", points, ReconstructionRequest(), xy_step_um=STEP)
    mask = np.array(result.gap_mask)
    z = grid_array(result.z_um)
    assert mask.shape == (12, 12)
    assert result.statistics.n_gaps == 1
    region = result.gaps[0]
    # Every cell whose own scan point is missing is a gap, nothing else.
    assert region.n_cells == 9
    assert mask[4:7, 3:6].all()
    assert mask.sum() == 9
    assert np.isnan(z[mask]).all()
    assert region.centroid_x_um == pytest.approx(4 * STEP)
    assert region.centroid_y_um == pytest.approx(5 * STEP)
    assert region.area_um2 == pytest.approx(9 * STEP * STEP)
    assert np.isfinite(z[~mask]).all()

    filled = reconstruct_surface(
        "s", points, ReconstructionRequest(fill_gaps=True), xy_step_um=STEP
    )
    assert np.array_equal(np.array(filled.gap_mask), mask)
    assert grid_array(filled.z_um)[5, 4] == pytest.approx(plane(8.0, 10.0))


def _with_status(points: list[ScanPoint], where: Callable[[int, int], bool]) -> list[ScanPoint]:
    """Copy of ``points`` with NO_PEAK (no height) at the selected grid indices."""
    return [
        p.model_copy(update={"status": PointStatus.NO_PEAK, "surface_z_um": None})
        if where(p.ix, p.iy)
        else p
        for p in points
    ]


@pytest.mark.parametrize("fill_gaps", [False, True])
def test_single_rejected_point_is_a_gap(fill_gaps: bool) -> None:
    """repro_gap: one NO_PEAK point in a 5 x 5 grid used to be silently interpolated."""
    points = _with_status(make_points(5, plane), lambda ix, iy: (ix, iy) == (2, 2))
    request = ReconstructionRequest(fill_gaps=fill_gaps)
    result = reconstruct_surface("s", points, request, xy_step_um=STEP)
    mask = np.array(result.gap_mask)
    z = grid_array(result.z_um)
    assert mask[2, 2]
    assert mask.sum() == 1
    assert result.statistics.n_gaps == 1
    assert result.gaps[0].n_cells == 1
    if fill_gaps:
        assert z[2, 2] == pytest.approx(plane(2 * STEP, 2 * STEP))
    else:
        assert math.isnan(z[2, 2])
    assert np.isfinite(z[~mask]).all()
    assert_json_safe(result)


def test_single_missing_point_is_a_gap() -> None:
    points = make_points(5, plane, skip=lambda ix, iy: (ix, iy) == (1, 3))
    result = reconstruct_surface("s", points, ReconstructionRequest(), xy_step_um=STEP)
    mask = np.array(result.gap_mask)
    assert mask[3, 1]
    assert mask.sum() == 1
    assert result.z_um[3][1] is None


@pytest.mark.parametrize(
    "where",
    [
        lambda ix, iy: (ix, iy) == (3, 3),  # rejected outright (no peak)
        lambda ix, iy: False,  # control: nothing rejected
    ],
)
def test_rejected_point_on_a_finer_output_grid(where: Callable[[int, int], bool]) -> None:
    """With grid_step = step / 2 the cells closer to the rejected point are gaps."""
    points = _with_status(make_points(7, plane), where)
    request = ReconstructionRequest(grid_step_um=STEP / 2)
    result = reconstruct_surface("s", points, request, xy_step_um=STEP)
    mask = np.array(result.gap_mask)
    if not any(where(p.ix, p.iy) for p in points):
        assert not mask.any()
        return
    # Cell (6, 6) sits on the rejected point; its 8 neighbours are half a step
    # away from it and at least as close to a used point (ties go to the used point).
    assert mask[6, 6]
    assert mask.sum() == 1


def test_low_confidence_and_outlier_points_are_gaps() -> None:
    points = make_points(7, plane)
    points[10] = points[10].model_copy(update={"confidence": 0.1})  # (3, 1)
    points[24] = points[24].model_copy(update={"surface_z_um": 50.0})  # (3, 3): outlier
    result = reconstruct_surface("s", points, ReconstructionRequest(), xy_step_um=STEP)
    mask = np.array(result.gap_mask)
    assert mask[1, 3]
    assert mask[3, 3]
    assert mask.sum() == 2


def test_missing_grid_positions() -> None:
    ix = np.array([0, 1, 2, 0, 2], dtype=np.int64)  # (1, 1) missing in a 3 x 2 index range
    iy = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    x = 5.0 + ix * STEP
    y = -3.0 + iy * STEP
    np.testing.assert_allclose(
        missing_grid_positions(ix, iy, x, y, STEP), [[5.0 + STEP, -3.0 + STEP]]
    )
    assert missing_grid_positions(ix, iy, x, y, STEP / 2).shape == (0, 2)  # off grid
    shifted = x + np.array([0.0, 0.0, 0.0, 0.9 * STEP, 0.9 * STEP])
    assert missing_grid_positions(ix, iy, shifted, y, STEP).shape == (0, 2)
    empty = np.empty(0, dtype=np.int64)
    assert missing_grid_positions(empty, empty, x[:0], y[:0], STEP).shape == (0, 2)


def test_gap_distance_can_be_tightened() -> None:
    points = make_points(10, plane, skip=lambda ix, iy: ix == 4)
    request = ReconstructionRequest(max_gap_distance_um=1.0)
    result = reconstruct_surface("s", points, request, xy_step_um=STEP)
    mask = np.array(result.gap_mask)
    assert mask[:, 4].all()
    assert not mask[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]].any()
    assert result.statistics.n_gaps == 1  # one connected column


# --------------------------------------------------------------------------- mesh / sections


def test_mesh_is_valid_and_counter_clockwise() -> None:
    points = make_points(6, plane)
    result = reconstruct_surface("s", points, ReconstructionRequest(), xy_step_um=STEP)
    mesh = result.mesh
    assert mesh is not None
    assert len(mesh.vertices) == result.statistics.n_used
    v = np.array(mesh.vertices)
    f = np.array(mesh.faces)
    assert f.min() >= 0
    assert f.max() < len(v)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    assert (cross > 0).all()
    assert cross.sum() / 2 == pytest.approx((5 * STEP) ** 2)  # triangles tile the hull
    no_mesh = reconstruct_surface(
        "s", points, ReconstructionRequest(build_mesh=False), xy_step_um=STEP
    )
    assert no_mesh.mesh is None


def test_cross_sections_follow_the_height_map() -> None:
    points = make_points(9, plane)
    request = ReconstructionRequest(cross_section_count=2)
    result = reconstruct_surface("s", points, request, xy_step_um=STEP)
    along_x = [s for s in result.cross_sections if s.along == "x"]
    along_y = [s for s in result.cross_sections if s.along == "y"]
    assert len(along_x) == len(along_y) == 2
    z = grid_array(result.z_um)
    for section in along_x:
        row = result.y_um.index(section.position_um)
        assert section.coordinate_um == result.x_um
        np.testing.assert_allclose(np.array(section.z_um, dtype=float), z[row])
    for section in along_y:
        col = result.x_um.index(section.position_um)
        np.testing.assert_allclose(np.array(section.z_um, dtype=float), z[:, col])


# --------------------------------------------------------------------------- statistics


def test_statistics_of_a_sinusoid_after_plane_removal() -> None:
    amplitude, wavelength = 0.5, 20.0
    gx, gy = np.meshgrid(np.arange(0.0, 200.0, 0.5), np.arange(0.0, 40.0, 0.5))
    # A cosine over whole periods is orthogonal to the plane terms, so plane
    # removal leaves it untouched.
    z = 0.01 * gx + 0.03 * gy + amplitude * np.cos(2 * np.pi * gx / wavelength)
    params = height_parameters(gx.ravel(), gy.ravel(), z.ravel())
    assert params is not None
    assert params.sq_um == pytest.approx(amplitude / math.sqrt(2), rel=0.01)
    assert params.sa_um == pytest.approx(2 * amplitude / math.pi, rel=0.01)
    assert params.sz_um == pytest.approx(2 * amplitude, rel=0.01)
    assert params.ssk == pytest.approx(0.0, abs=0.02)
    assert params.sku == pytest.approx(1.5, rel=0.02)


def test_plane_fit_and_perfect_plane_statistics() -> None:
    x = np.array([0.0, 1.0, 0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    a, b, c = fit_plane(x, y, 2 * x - 3 * y + 1)  # type: ignore[misc]
    assert (a, b, c) == pytest.approx((2.0, -3.0, 1.0))
    params = height_parameters(x, y, 2 * x - 3 * y + 1)
    assert params is not None
    assert params.sq_um == pytest.approx(0.0, abs=1e-9)
    assert params.ssk is None
    assert fit_plane(x[:2], y[:2], x[:2]) is None


def test_reconstruction_statistics_counts() -> None:
    points = make_points(10, plane)
    points[0] = points[0].model_copy(update={"status": PointStatus.FIT_FAILED})
    points[1] = points[1].model_copy(update={"confidence": 0.2})
    result = reconstruct_surface("s", points, ReconstructionRequest(), xy_step_um=STEP)
    stats = result.statistics
    assert (stats.n_input, stats.n_invalid, stats.n_low_confidence) == (100, 1, 1)
    assert stats.n_used == 98
    assert stats.plane_coefficients == pytest.approx((0.05, -0.02, 3.0), abs=1e-9)
    assert stats.sq_um == pytest.approx(0.0, abs=1e-9)
    assert stats.mean_confidence == pytest.approx(0.9)
    assert len(result.points) == 100
    assert_json_safe(result)


# --------------------------------------------------------------------------- errors


def test_too_few_points_raise() -> None:
    points = make_points(3, plane)
    for p in points:
        p.status = PointStatus.NO_PEAK
    with pytest.raises(ReconstructionError):
        reconstruct_surface("s", points, ReconstructionRequest(), xy_step_um=STEP)
    with pytest.raises(ReconstructionError):
        reconstruct_surface("s", [], ReconstructionRequest(), xy_step_um=STEP)


def test_collinear_points_need_nearest() -> None:
    points = make_points(5, plane, skip=lambda ix, iy: iy != 0)
    with pytest.raises(ReconstructionError):
        reconstruct_surface("s", points, ReconstructionRequest(), xy_step_um=STEP)
    request = ReconstructionRequest(method=InterpolationMethod.NEAREST)
    result = reconstruct_surface("s", points, request, xy_step_um=STEP)
    assert result.mesh is None
    assert_json_safe(result)


def test_invalid_step_is_a_caller_error() -> None:
    with pytest.raises(ValueError, match="xy_step_um"):
        reconstruct_surface("s", make_points(3, plane), ReconstructionRequest(), xy_step_um=0.0)


def test_inputs_are_not_mutated() -> None:
    points = make_points(5, plane)
    before = [p.model_dump() for p in points]
    reconstruct_surface("s", points, ReconstructionRequest(), xy_step_um=STEP)
    assert [p.model_dump() for p in points] == before


# --------------------------------------------------------------------------- performance


@pytest.mark.parametrize("method", [InterpolationMethod.LINEAR, InterpolationMethod.RBF])
def test_100x100_is_fast(method: InterpolationMethod) -> None:
    points = make_points(100, lambda x, y: 0.01 * x + math.sin(y / 10.0))
    start = time.perf_counter()
    result = reconstruct_surface("s", points, ReconstructionRequest(method=method), xy_step_um=STEP)
    assert time.perf_counter() - start < 6.0
    assert result.statistics.n_used >= 9990
    assert result.statistics.coverage_fraction == pytest.approx(1.0)
