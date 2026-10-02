"""Surface reconstruction: classified scan points -> height map, mesh and statistics.

``reconstruct_surface`` implements the ``SurfaceReconstructor`` protocol::

    classify (invalid / low confidence / outlier / used)  -- filtering.py
    -> output grid over the used points' bounding box       -- grid.py
    -> nearest-point distances, rejected / missing points   -- gaps.py
    -> Delaunay triangulation (shared)                      -- mesh.py
    -> heights + confidence map on the grid                 -- interpolation.py
    -> gap masking, cross-sections, mesh, statistics        -- statistics.py

Only the physics results are used: ``surface_z_um`` of points whose status
is VALID or LOW_CONFIDENCE. ML predictions never enter the reconstruction.
The result is strictly JSON-safe: every missing value is None, never NaN.

Gaps (gaps.py): a cell is a gap when its own scan point was rejected or
never measured, or when no used point lies within ``max_gap_distance_um``.

``fill_gaps``: when false (default) gap cells are None. When true they keep
the interpolated value where the method produced one; the method is never
asked to extrapolate further than it does by itself (linear and cubic stay
undefined outside the convex hull of the used points). ``gap_mask`` marks gap
cells either way.

CPU-bound and synchronous: the scan manager runs it in a worker thread.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree

from confocal.errors import ReconstructionError
from confocal.models.scan import ScanPoint
from confocal.models.surface import (
    PointClassification,
    ReconstructionRequest,
    SurfacePoint,
    SurfaceResult,
)
from confocal.surface.filtering import ClassifiedPoints, classify_points
from confocal.surface.gaps import gap_mask, gap_regions, missing_grid_positions
from confocal.surface.grid import OutputGrid, cross_sections, to_json_grid
from confocal.surface.interpolation import (
    check_interpolable,
    interpolate_confidence,
    interpolate_heights,
)
from confocal.surface.mesh import build_mesh, triangulate
from confocal.surface.statistics import surface_statistics

#: Default gap distance in units of the scan XY step.
DEFAULT_GAP_DISTANCE_STEPS = 1.5


def _surface_points(classified: ClassifiedPoints) -> list[SurfacePoint]:
    """The classified point cloud; points without a finite XY cannot be placed and are
    omitted (they are still counted as invalid in the statistics)."""
    out: list[SurfacePoint] = []
    for i, label in enumerate(classified.classification):
        x, y, z = classified.x_um[i], classified.y_um[i], classified.z_um[i]
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        out.append(
            SurfacePoint(
                point_id=int(classified.point_ids[i]),
                x_um=float(x),
                y_um=float(y),
                z_um=float(z) if math.isfinite(z) else None,
                confidence=float(classified.confidence[i]),
                classification=label,
            )
        )
    return out


def _no_usable_points(classified: ClassifiedPoints) -> ReconstructionError:
    return ReconstructionError(
        f"no usable points out of {len(classified.classification)}: "
        f"{classified.count(PointClassification.INVALID)} invalid, "
        f"{classified.count(PointClassification.LOW_CONFIDENCE)} low confidence, "
        f"{classified.count(PointClassification.OUTLIER)} outliers"
    )


def _rejected_positions(
    points: Sequence[ScanPoint], classified: ClassifiedPoints, xy_step_um: float
) -> np.ndarray:
    """XY, shape (k, 2), of the placed points that are not used plus the missing
    scan-grid positions (see :mod:`confocal.surface.gaps`)."""
    placed = np.isfinite(classified.x_um) & np.isfinite(classified.y_um)
    rejected = placed & ~classified.used
    n = len(points)
    ix = np.fromiter((p.ix for p in points), dtype=np.int64, count=n)[placed]
    iy = np.fromiter((p.iy for p in points), dtype=np.int64, count=n)[placed]
    missing = missing_grid_positions(
        ix, iy, classified.x_um[placed], classified.y_um[placed], xy_step_um
    )
    own = np.column_stack((classified.x_um[rejected], classified.y_um[rejected]))
    return np.vstack((own, missing))


def reconstruct_surface(
    scan_id: str,
    points: Sequence[ScanPoint],
    request: ReconstructionRequest,
    *,
    xy_step_um: float,
) -> SurfaceResult:
    """Reconstruct the surface of one scan (see the module docstring).

    ``xy_step_um`` is the scan's XY step: the default output-grid spacing, the
    unit of the gap distance default (1.5 steps) and the coordinate scale of
    the interpolators.

    Raises:
        ReconstructionError: too few usable points for the method, collinear
            points for a method that needs a plane, or an oversized grid.
        ValueError: ``xy_step_um`` is not a positive finite number (caller bug).
    """
    if not (math.isfinite(xy_step_um) and xy_step_um > 0.0):
        raise ValueError(f"xy_step_um must be positive and finite, got {xy_step_um}")
    classified = classify_points(points, request)
    used = classified.used
    if not used.any():
        raise _no_usable_points(classified)
    x, y = classified.x_um[used], classified.y_um[used]
    z, conf = classified.z_um[used], classified.confidence[used]

    origin = np.array([float(np.min(x)), float(np.min(y))])
    xy = (np.column_stack((x, y)) - origin) / xy_step_um
    check_interpolable(xy, request.method, request.rbf_kernel)

    grid = OutputGrid.covering(x, y, request.grid_step_um or xy_step_um)
    targets = (grid.points() - origin) / xy_step_um
    distance, nearest = cKDTree(xy).query(targets, k=1)
    nearest_index = np.asarray(nearest, dtype=np.intp).reshape(-1)
    distance_um = np.asarray(distance, dtype=np.float64).reshape(grid.shape) * xy_step_um

    triangulation = triangulate(xy)
    heights = interpolate_heights(
        request.method,
        xy,
        z,
        targets,
        triangulation=triangulation,
        nearest_index=nearest_index,
        request=request,
    ).reshape(grid.shape)
    confidence = interpolate_confidence(
        conf, targets, triangulation=triangulation, nearest_index=nearest_index
    ).reshape(grid.shape)

    max_gap = request.max_gap_distance_um or DEFAULT_GAP_DISTANCE_STEPS * xy_step_um
    rejected = _rejected_positions(points, classified, xy_step_um)
    rejected_um: np.ndarray | None = None
    if rejected.shape[0]:
        d_rejected, _ = cKDTree((rejected - origin) / xy_step_um).query(targets, k=1)
        rejected_um = np.asarray(d_rejected, dtype=np.float64).reshape(grid.shape) * xy_step_um
    gaps = gap_mask(distance_um, max_gap, rejected_um)
    keep = np.isfinite(heights) & (~gaps if not request.fill_gaps else True)
    heights = np.where(keep, heights, np.nan)
    confidence = np.where(keep, confidence, np.nan)
    regions = gap_regions(gaps, grid)
    grid_x, grid_y = grid.mesh()

    return SurfaceResult(
        scan_id=scan_id,
        request=request,
        x_um=[float(v) for v in grid.x_um],
        y_um=[float(v) for v in grid.y_um],
        z_um=to_json_grid(heights),
        confidence=to_json_grid(confidence),
        gap_mask=gaps.tolist(),
        points=_surface_points(classified),
        mesh=(
            build_mesh(triangulation, x, y, z)
            if request.build_mesh and triangulation is not None
            else None
        ),
        cross_sections=cross_sections(grid, heights, request.cross_section_count),
        gaps=regions,
        statistics=surface_statistics(
            classified=classified,
            grid_x=grid_x,
            grid_y=grid_y,
            heights=heights,
            gaps=gaps,
            n_gaps=len(regions),
        ),
    )
