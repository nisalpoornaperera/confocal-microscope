"""Per-point feature vectors for the advisory ML models.

Every feature is derived from the stored physics results (``ScanPoint``), so
a model can be applied to any finished scan without re-analysing profiles,
and training (offline) and inference (on the Pi) share this one definition.

Features (column order = :data:`FEATURE_NAMES`)::

    x_um, y_um                  stage position of the point
    coarse_peak_z_um            Z of the coarse-sweep peak
    peak_intensity              smoothed signal at the peak (normalized or volts)
    peak_width_um               axial FWHM
    prominence, snr             peak prominence and SNR
    fit_residual                RMS residual of the fit that defines the height
    fit_disagreement_um         |parabolic vertex - Gaussian centre|
    confidence                  physics confidence score (0..1)
    surface_z_deviation_um      surface Z - median surface Z of the grid neighbours
    neighbour_z_std_um          standard deviation of the neighbours' surface Z
    neighbour_count             neighbours (8-connected on the (ix, iy) grid)
                                that have a usable surface height
    secondary_peak_ratio        second / main peak prominence
    asymmetry                   signed half-width asymmetry of the peak
    *_missing                   1.0 where the corresponding value was missing

Neighbours are the up to eight points at ``|dix| <= 1, |diy| <= 1`` with a
finite surface height and status VALID or LOW_CONFIDENCE -- the same points
that would define the reconstructed surface around this one. A point whose
height disagrees with its neighbourhood is the typical "bad point" (a
secondary reflection or dust picked as the surface).

Imputation (the matrix is always finite): every missing value is replaced by
0.0 and flagged in the matching ``*_missing`` indicator, so tree models can
separate "missing" from "zero". All the imputed quantities are either
non-negative magnitudes for which 0 means "no evidence" (width, prominence,
SNR, residual, disagreement, ratio, spread) or signed offsets centred on 0
(deviation, asymmetry). ``coarse_peak_z_um`` and ``peak_intensity`` are
imputed with 0 as well; their indicators carry the information. Non-finite
inputs are treated as missing.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from confocal.models.processing import PointStatus
from confocal.models.scan import ScanPoint

FEATURE_NAMES: tuple[str, ...] = (
    "x_um",
    "y_um",
    "coarse_peak_z_um",
    "peak_intensity",
    "peak_width_um",
    "prominence",
    "snr",
    "fit_residual",
    "fit_disagreement_um",
    "confidence",
    "surface_z_deviation_um",
    "neighbour_z_std_um",
    "neighbour_count",
    "secondary_peak_ratio",
    "asymmetry",
    "surface_z_missing",
    "coarse_peak_z_missing",
    "peak_intensity_missing",
    "peak_width_missing",
    "snr_missing",
    "fit_disagreement_missing",
    "neighbour_z_missing",
    "profile_shape_missing",
)

#: Statuses whose surface height counts for the neighbour statistics.
_USABLE_STATUSES: frozenset[PointStatus] = frozenset(
    {PointStatus.VALID, PointStatus.LOW_CONFIDENCE}
)

_NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)
)

_COLUMN = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _value(value: float | None) -> float:
    """``value`` as a float, NaN when missing or non-finite."""
    return float(value) if value is not None and math.isfinite(value) else math.nan


def _usable_height(point: ScanPoint) -> float:
    return _value(point.surface_z_um) if point.status in _USABLE_STATUSES else math.nan


def _neighbour_heights(points: Sequence[ScanPoint]) -> NDArray[np.float64]:
    """``(n, 8)`` surface heights of each point's grid neighbours (NaN = none).

    Built on a padded ``(iy, ix)`` raster. If an ``(ix, iy)`` cell was measured
    more than once, the last point in the sequence represents it.
    """
    n = len(points)
    if n == 0:
        return np.empty((0, len(_NEIGHBOUR_OFFSETS)), dtype=np.float64)
    ix = np.fromiter((p.ix for p in points), dtype=np.intp, count=n)
    iy = np.fromiter((p.iy for p in points), dtype=np.intp, count=n)
    raster = np.full((int(iy.max()) + 3, int(ix.max()) + 3), np.nan, dtype=np.float64)
    raster[iy + 1, ix + 1] = [_usable_height(p) for p in points]
    return np.column_stack([raster[iy + 1 + dy, ix + 1 + dx] for dx, dy in _NEIGHBOUR_OFFSETS])


def _neighbour_statistics(
    points: Sequence[ScanPoint],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """(median, standard deviation, count) of the neighbours' heights; NaN without any."""
    heights = _neighbour_heights(points)
    count = np.count_nonzero(np.isfinite(heights), axis=1).astype(np.float64)
    with warnings.catch_warnings():
        # All-NaN rows (no usable neighbour) are expected; they yield NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(heights, axis=1) if heights.size else np.empty(0)
        spread = np.nanstd(heights, axis=1) if heights.size else np.empty(0)
    return (
        np.asarray(median, dtype=np.float64),
        np.asarray(spread, dtype=np.float64),
        count,
    )


def build_feature_matrix(points: Sequence[ScanPoint]) -> NDArray[np.float64]:
    """Finite ``(len(points), len(FEATURE_NAMES))`` matrix; never modifies ``points``."""
    n = len(points)
    raw = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    for row, p in enumerate(points):
        parabolic, gaussian = _value(p.parabolic_z_um), _value(p.gaussian_z_um)
        raw[row, _COLUMN["x_um"]] = _value(p.x_um)
        raw[row, _COLUMN["y_um"]] = _value(p.y_um)
        raw[row, _COLUMN["coarse_peak_z_um"]] = _value(p.coarse_peak_z_um)
        raw[row, _COLUMN["peak_intensity"]] = _value(p.peak_intensity)
        raw[row, _COLUMN["peak_width_um"]] = _value(p.peak_width_um)
        raw[row, _COLUMN["prominence"]] = _value(p.prominence)
        raw[row, _COLUMN["snr"]] = _value(p.snr)
        raw[row, _COLUMN["fit_residual"]] = _value(p.fit_residual)
        raw[row, _COLUMN["fit_disagreement_um"]] = abs(parabolic - gaussian)
        raw[row, _COLUMN["confidence"]] = _value(p.confidence)
        raw[row, _COLUMN["surface_z_deviation_um"]] = _usable_height(p)  # median removed below
        raw[row, _COLUMN["secondary_peak_ratio"]] = _value(p.secondary_peak_ratio)
        raw[row, _COLUMN["asymmetry"]] = _value(p.asymmetry)

    median, spread, count = _neighbour_statistics(points)
    raw[:, _COLUMN["surface_z_deviation_um"]] -= median
    raw[:, _COLUMN["neighbour_z_std_um"]] = spread
    raw[:, _COLUMN["neighbour_count"]] = count

    def missing(*names: str) -> NDArray[np.float64]:
        cols = [_COLUMN[name] for name in names]
        return np.asarray(np.isnan(raw[:, cols]).any(axis=1), dtype=np.float64)

    indicators = {
        "surface_z_missing": np.asarray(
            [math.isnan(_usable_height(p)) for p in points], dtype=np.float64
        ),
        "coarse_peak_z_missing": missing("coarse_peak_z_um"),
        "peak_intensity_missing": missing("peak_intensity"),
        "peak_width_missing": missing("peak_width_um"),
        "snr_missing": missing("snr"),
        "fit_disagreement_missing": missing("fit_disagreement_um"),
        "neighbour_z_missing": np.asarray(count == 0, dtype=np.float64),
        "profile_shape_missing": missing("secondary_peak_ratio", "asymmetry"),
    }
    for name, column in indicators.items():
        raw[:, _COLUMN[name]] = column
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
