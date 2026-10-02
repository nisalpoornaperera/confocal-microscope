"""Gap detection on the output grid.

A grid cell is a *gap* when either

* its own scan point was rejected or never measured: the measured position
  nearest to the cell is a point that is not used (no peak, low confidence,
  outlier, ...) or a scan-grid position without any point (an interrupted
  scan), and no used point is as close; or
* no used point lies within ``max_gap_distance_um`` of it (larger holes and
  the area outside the scan).

So a single rejected or missing point already makes the cell at its position
a gap (with an output grid finer than the scan, the cells closer to that
position than to any used point), while the cells of a complete scan grid
are never gaps. A gap cell has no value unless ``fill_gaps`` is requested.

Missing positions are inferred from the points' grid indices ``(ix, iy)``:
every index pair inside the index range of the points that has no point is
missing, at its nominal position ``origin + index x step``. When the points do
not lie on such a grid (positions inconsistent with their indices) nothing is
inferred and only rejected points and the distance rule apply.

Gap cells are grouped into connected regions (8-connectivity, so a diagonal
streak of missing points is one defect) with ``skimage.measure.label``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.measure import label

from confocal.models.surface import GapRegion
from confocal.surface.grid import OutputGrid

#: Max deviation (in scan steps) of a point from its nominal grid position for
#: the grid to be trusted when inferring missing positions.
GRID_CONSISTENCY_STEPS = 0.25


def gap_mask(
    distance_um: NDArray[np.float64],
    max_gap_distance_um: float,
    rejected_distance_um: NDArray[np.float64] | None = None,
) -> NDArray[np.bool_]:
    """Gap cells (see the module docstring).

    ``distance_um`` is each cell's distance to the nearest used point and
    ``rejected_distance_um`` (same shape) to the nearest rejected or missing
    scan position; a cell strictly closer to the latter is a gap (on a tie the
    used point wins).
    """
    gaps = distance_um > max_gap_distance_um
    if rejected_distance_um is not None:
        gaps |= rejected_distance_um < distance_um
    return np.asarray(gaps, dtype=np.bool_)


def missing_grid_positions(
    ix: NDArray[np.int64],
    iy: NDArray[np.int64],
    x_um: NDArray[np.float64],
    y_um: NDArray[np.float64],
    step_um: float,
) -> NDArray[np.float64]:
    """Nominal XY, shape (k, 2), of the scan-grid positions that have no point.

    Only index pairs inside ``[min ix, max ix] x [min iy, max iy]`` of the
    given points are considered. Returns an empty array when there are no
    points or their positions do not match ``origin + index x step``.
    """
    empty = np.empty((0, 2), dtype=np.float64)
    if ix.size == 0:
        return empty
    x0 = float(np.median(x_um - ix * step_um))
    y0 = float(np.median(y_um - iy * step_um))
    tolerance = GRID_CONSISTENCY_STEPS * step_um
    if (
        np.max(np.abs(x_um - (x0 + ix * step_um))) > tolerance
        or np.max(np.abs(y_um - (y0 + iy * step_um))) > tolerance
    ):
        return empty
    ix_lo, iy_lo = int(ix.min()), int(iy.min())
    present = np.zeros((int(iy.max()) - iy_lo + 1, int(ix.max()) - ix_lo + 1), dtype=np.bool_)
    present[iy - iy_lo, ix - ix_lo] = True
    miss_iy, miss_ix = np.nonzero(~present)
    if miss_ix.size == 0:
        return empty
    return np.column_stack(
        (x0 + (miss_ix + ix_lo) * step_um, y0 + (miss_iy + iy_lo) * step_um)
    ).astype(np.float64)


def gap_regions(mask: NDArray[np.bool_], grid: OutputGrid) -> list[GapRegion]:
    """Connected gap regions with size, area, centroid and bounding box.

    ``area_um2`` is the number of cells times the cell area; the centroid and
    ``bbox_um`` refer to the centres of the region's cells.
    """
    labels = np.asarray(label(mask, connectivity=2), dtype=np.int64)  # type: ignore[no-untyped-call]
    n_regions = int(labels.max()) if labels.size else 0
    if n_regions == 0:
        return []
    gx, gy = grid.mesh()
    in_gap = labels > 0
    region = labels[in_gap] - 1
    xs, ys = gx[in_gap], gy[in_gap]

    counts = np.bincount(region, minlength=n_regions)
    cx = np.bincount(region, weights=xs, minlength=n_regions) / counts
    cy = np.bincount(region, weights=ys, minlength=n_regions) / counts
    x_min = np.full(n_regions, np.inf)
    y_min = np.full(n_regions, np.inf)
    x_max = np.full(n_regions, -np.inf)
    y_max = np.full(n_regions, -np.inf)
    np.minimum.at(x_min, region, xs)
    np.minimum.at(y_min, region, ys)
    np.maximum.at(x_max, region, xs)
    np.maximum.at(y_max, region, ys)

    area = grid.cell_area_um2
    return [
        GapRegion(
            label=i + 1,
            n_cells=int(counts[i]),
            area_um2=float(counts[i]) * area,
            centroid_x_um=float(cx[i]),
            centroid_y_um=float(cy[i]),
            bbox_um=(float(x_min[i]), float(y_min[i]), float(x_max[i]), float(y_max[i])),
        )
        for i in range(n_regions)
    ]
