"""Height statistics of the reconstructed surface (ISO 25178 style).

The areal roughness parameters are computed over every defined cell of the
height map (a regular grid, i.e. area-weighted as the standard requires),
after removing the least-squares plane ``z = a x + b y + c`` -- the
"form removal" that separates roughness from sample tilt::

    eta = z - (a x + b y + c)
    Sa  = mean |eta|                 arithmetic mean height
    Sq  = sqrt(mean eta^2)           root-mean-square height
    Sz  = max eta - min eta          maximum height (peak to valley)
    Ssk = mean eta^3 / Sq^3          skewness (0 for a symmetric distribution)
    Sku = mean eta^4 / Sq^4          kurtosis (3 for a Gaussian surface)

These are the parameters of the *sampled, interpolated* map without the
S-/L-filters of the full standard, so they are comparable between scans of the
same step but are not certified ISO values. ``z_min/max/mean/std`` describe
the raw heights (before plane removal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from confocal.models.surface import PointClassification, SurfaceStatistics
from confocal.surface.filtering import ClassifiedPoints

#: Sq below this (in micrometres) makes Ssk / Sku undefined (a perfect plane).
_MIN_SQ_FOR_SHAPE_UM = 1e-9


@dataclass(frozen=True, slots=True)
class HeightParameters:
    sa_um: float
    sq_um: float
    sz_um: float
    ssk: float | None
    sku: float | None
    plane: tuple[float, float, float]  # (a, b, c) of z = a x + b y + c


def fit_plane(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> tuple[float, float, float] | None:
    """Least-squares plane ``z = a x + b y + c``; None for fewer than 3 points.

    Solved on centred coordinates for numerical stability; for collinear points
    (a single grid row) the minimum-norm solution is returned, i.e. no tilt
    across the line.
    """
    if z.shape[0] < 3:
        return None
    xm, ym = float(np.mean(x)), float(np.mean(y))
    design = np.column_stack((x - xm, y - ym, np.ones_like(x)))
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    a, b, c0 = (float(v) for v in coeffs)
    return a, b, c0 - a * xm - b * ym


def height_parameters(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> HeightParameters | None:
    """Sa, Sq, Sz, Ssk, Sku after plane removal; None for fewer than 3 heights."""
    plane = fit_plane(x, y, z)
    if plane is None:
        return None
    a, b, c = plane
    eta = z - (a * x + b * y + c)
    eta = eta - float(np.mean(eta))  # exactly zero-mean despite rounding
    sq = float(np.sqrt(np.mean(eta * eta)))
    shape_defined = sq > _MIN_SQ_FOR_SHAPE_UM
    return HeightParameters(
        sa_um=float(np.mean(np.abs(eta))),
        sq_um=sq,
        sz_um=float(np.max(eta) - np.min(eta)),
        ssk=float(np.mean(eta**3)) / sq**3 if shape_defined else None,
        sku=float(np.mean(eta**4)) / sq**4 if shape_defined else None,
        plane=plane,
    )


def surface_statistics(
    *,
    classified: ClassifiedPoints,
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    heights: NDArray[np.float64],
    gaps: NDArray[np.bool_],
    n_gaps: int,
) -> SurfaceStatistics:
    """Counts, coverage, raw height statistics and roughness of the final map.

    ``grid_x`` / ``grid_y`` / ``heights`` / ``gaps`` all have the grid's
    shape; ``heights`` is NaN where the map has no value.
    """
    defined = np.isfinite(heights)
    n_cells = int(heights.size)
    z = heights[defined]
    params = height_parameters(grid_x[defined], grid_y[defined], z)
    used_conf = classified.confidence[classified.used]
    return SurfaceStatistics(
        n_input=len(classified.classification),
        n_invalid=classified.count(PointClassification.INVALID),
        n_low_confidence=classified.count(PointClassification.LOW_CONFIDENCE),
        n_outliers=classified.count(PointClassification.OUTLIER),
        n_used=classified.count(PointClassification.USED),
        coverage_fraction=float(np.count_nonzero(defined)) / n_cells if n_cells else 0.0,
        gap_fraction=float(np.count_nonzero(gaps)) / n_cells if n_cells else 0.0,
        n_gaps=n_gaps,
        z_min_um=float(np.min(z)) if z.size else None,
        z_max_um=float(np.max(z)) if z.size else None,
        z_mean_um=float(np.mean(z)) if z.size else None,
        z_std_um=float(np.std(z)) if z.size else None,
        sa_um=None if params is None else params.sa_um,
        sq_um=None if params is None else params.sq_um,
        sz_um=None if params is None else params.sz_um,
        ssk=None if params is None else params.ssk,
        sku=None if params is None else params.sku,
        plane_coefficients=None if params is None else params.plane,
        mean_confidence=float(np.mean(used_conf)) if used_conf.size else None,
    )
