"""Classification of scan points before reconstruction.

Every input point gets exactly one :class:`PointClassification`, decided in
this order:

1. ``INVALID``: the point has no usable height -- its status is neither VALID
   nor LOW_CONFIDENCE (no peak, peak at the sweep edge, fit failed, aborted,
   fixed-Z measurement) or its surface Z / XY position is not finite.
2. ``LOW_CONFIDENCE``: confidence below the request's ``min_confidence``.
   (A point with status LOW_CONFIDENCE is still *used* when its confidence
   reaches the request's own, possibly lower, threshold.)
3. ``OUTLIER``: rejected by the robust outlier test.
4. ``USED``: everything else defines the surface.

Outlier tests use robust statistics (median / MAD), because a single dust
particle or a secondary reflection picked as the surface can be tens of
micrometres off and would wreck a mean / standard-deviation test:

``LOCAL_MAD``
    For each point, ``d = z - p``, where ``p`` is the height predicted at the
    point by a plane fitted to its k nearest neighbours (the point itself
    excluded) without the neighbours that deviate by more than
    :data:`LOCAL_PLANE_REJECTION` robust standard deviations, so that a nearby
    outlier cannot drag the prediction of good points. A local *plane* rather
    than the neighbourhood median makes the test exact for tilt also at the
    scan border, where the neighbours lie on one side only; the median there
    is biased by tilt x distance and flags ordinary border points. The scale
    is the larger of the local spread (MAD of the neighbours' own deviations)
    and the global spread of all deviations, so a smooth region with
    by-chance tiny local noise does not flag ordinary points, while a rough
    region tolerates proportionally larger deviations.

``GLOBAL_MAD``
    Residuals from a robust plane (least squares refitted on the inliers until
    the inlier set is stable); the scale is the MAD of all residuals. Suited to
    nominally flat samples; it also catches large connected defects that a
    local test sees as "normal" because all their neighbours share them.

A point is an outlier when ``|deviation| > outlier_threshold * scale``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from confocal.models.processing import PointStatus
from confocal.models.scan import ScanPoint
from confocal.models.surface import OutlierMethod, PointClassification, ReconstructionRequest

#: Converts the MAD of normally distributed data into its standard deviation.
MAD_TO_STD = 1.482602218505602

#: Statuses whose surface height is meaningful.
USABLE_STATUSES: frozenset[PointStatus] = frozenset({PointStatus.VALID, PointStatus.LOW_CONFIDENCE})

#: Deviations are never judged against a scale below 1 pm: on noise-free
#: (synthetic) data the MAD is 0 and rounding errors must not become outliers.
SCALE_FLOOR_UM = 1e-6

#: Below this many candidate points no outlier test is meaningful.
MIN_POINTS_FOR_OUTLIERS = 4

#: Neighbours farther than this many robust standard deviations from a local
#: plane are left out of the next fit; the last of LOCAL_PLANE_FITS fits counts.
LOCAL_PLANE_REJECTION = 3.0
LOCAL_PLANE_FITS = 3

_MAX_PLANE_ITERATIONS = 10


@dataclass(frozen=True, slots=True)
class ClassifiedPoints:
    """Per-point arrays in input order plus the classification of each point."""

    point_ids: NDArray[np.int64]
    x_um: NDArray[np.float64]
    y_um: NDArray[np.float64]
    z_um: NDArray[np.float64]  # NaN where the point has no usable height
    confidence: NDArray[np.float64]
    classification: tuple[PointClassification, ...]

    @property
    def used(self) -> NDArray[np.bool_]:
        return np.array(
            [c is PointClassification.USED for c in self.classification], dtype=np.bool_
        )

    def count(self, classification: PointClassification) -> int:
        return sum(1 for c in self.classification if c is classification)


def _robust_scale(values: NDArray[np.float64]) -> float:
    """1.4826 x MAD around the median (0 for an empty array)."""
    if values.size == 0:
        return 0.0
    return MAD_TO_STD * float(np.median(np.abs(values - np.median(values))))


def _neighbour_indices(xy: NDArray[np.float64], k: int) -> NDArray[np.intp]:
    """Indices of the ``k`` nearest *other* points of every point, shape (n, k).

    Queries ``k + 1`` neighbours and removes the point itself wherever it
    appears (with duplicate XY positions it is not necessarily first).
    """
    n = int(xy.shape[0])
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = np.asarray(idx, dtype=np.intp).reshape(n, k + 1)
    is_self = idx == np.arange(n, dtype=np.intp)[:, None]
    order = np.argsort(is_self, axis=1, kind="stable")  # non-self entries first
    return np.take_along_axis(idx, order, axis=1)[:, :k]


def _robust_inliers(
    residual: NDArray[np.float64], previous: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per row, 1.0 where the residual is within LOCAL_PLANE_REJECTION robust
    standard deviations of the row median; rows that would keep fewer than three
    samples keep ``previous``."""
    centred = np.abs(residual - np.median(residual, axis=1, keepdims=True))
    spread = MAD_TO_STD * np.median(centred, axis=1)
    limit = LOCAL_PLANE_REJECTION * np.maximum(spread, SCALE_FLOOR_UM)
    inliers = np.asarray(centred <= limit[:, None], dtype=np.float64)
    enough = inliers.sum(axis=1) >= 3
    return np.asarray(np.where(enough[:, None], inliers, previous), dtype=np.float64)


def _local_plane_predictions(
    xy: NDArray[np.float64], z: NDArray[np.float64], nbr: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Height at every point predicted by a robust plane through its neighbours.

    Weighted least squares ``z_j = a + b dx_j + c dy_j`` (``dx_j = x_j - x_i``)
    solved for all points at once through their 3 x 3 normal equations; the
    prediction is ``a``. Neighbours far from the neighbourhood median are left
    out of the first fit; the fit is then repeated :data:`LOCAL_PLANE_FITS`
    times, each time without the neighbours whose residual lies more than
    :data:`LOCAL_PLANE_REJECTION` robust standard deviations from the median
    residual (see :func:`_robust_inliers`). The pseudo-inverse keeps collinear
    neighbourhoods (a single scan row) well defined: it returns the
    minimum-norm solution, i.e. no tilt across the line.
    """
    dxy = xy[nbr] - xy[:, None, :]  # (n, k, 2)
    scale = float(np.max(np.abs(dxy))) or 1.0
    design = np.concatenate((np.ones((*nbr.shape, 1)), dxy / scale), axis=2)  # (n, k, 3)
    zn = z[nbr]  # (n, k)
    # Start from the neighbours close to the neighbourhood median: least squares
    # alone has too little breakdown (a spike at a corner of a 3 x 3 neighbourhood
    # tilts the plane so much that its own residual looks ordinary).
    weights = _robust_inliers(zn, np.ones(nbr.shape, dtype=np.float64))
    prediction = np.zeros(z.shape[0], dtype=np.float64)
    for _ in range(LOCAL_PLANE_FITS):
        weighted = design * weights[:, :, None]
        normal = np.einsum("nki,nkj->nij", weighted, design)
        rhs = np.einsum("nki,nk->ni", weighted, zn)
        coeffs = np.einsum("nij,nj->ni", np.linalg.pinv(normal), rhs)
        prediction = coeffs[:, 0]
        weights = _robust_inliers(zn - np.einsum("nki,ni->nk", design, coeffs), weights)
    return np.asarray(prediction, dtype=np.float64)


def local_mad_outliers(
    xy: NDArray[np.float64], z: NDArray[np.float64], *, neighbours: int, threshold: float
) -> NDArray[np.bool_]:
    """Robust k-nearest-neighbour outlier test (see the module docstring)."""
    n = int(z.shape[0])
    k = min(neighbours, n - 1)
    if n < MIN_POINTS_FOR_OUTLIERS or k < 3:
        return np.zeros(n, dtype=np.bool_)
    nbr = _neighbour_indices(xy, k)
    deviation = z - _local_plane_predictions(xy, z, nbr)
    local_scale = MAD_TO_STD * np.median(np.abs(deviation[nbr]), axis=1)
    scale = np.maximum(np.maximum(local_scale, _robust_scale(deviation)), SCALE_FLOOR_UM)
    return np.asarray(np.abs(deviation) > threshold * scale, dtype=np.bool_)


def _plane_design(xy: NDArray[np.float64]) -> NDArray[np.float64]:
    centred = xy - xy.mean(axis=0)
    span = float(np.max(np.abs(centred))) or 1.0
    return np.column_stack((centred / span, np.ones(xy.shape[0], dtype=np.float64)))


def global_mad_outliers(
    xy: NDArray[np.float64], z: NDArray[np.float64], *, threshold: float
) -> NDArray[np.bool_]:
    """Robust-plane residual outlier test (see the module docstring)."""
    n = int(z.shape[0])
    if n < MIN_POINTS_FOR_OUTLIERS:
        return np.zeros(n, dtype=np.bool_)
    design = _plane_design(xy)
    inliers = np.ones(n, dtype=np.bool_)
    outliers = np.zeros(n, dtype=np.bool_)
    for _ in range(_MAX_PLANE_ITERATIONS):
        coeffs, *_ = np.linalg.lstsq(design[inliers], z[inliers], rcond=None)
        residual = z - design @ coeffs
        scale = max(_robust_scale(residual), SCALE_FLOOR_UM)
        outliers = np.abs(residual - np.median(residual)) > threshold * scale
        if int(np.count_nonzero(~outliers)) < 3 or np.array_equal(~outliers, inliers):
            break
        inliers = ~outliers
    return np.asarray(outliers, dtype=np.bool_)


def classify_points(
    points: Sequence[ScanPoint], request: ReconstructionRequest
) -> ClassifiedPoints:
    """Classify every point (see the module docstring); never modifies ``points``."""
    n = len(points)
    ids = np.fromiter((p.point_id for p in points), dtype=np.int64, count=n)
    x = np.fromiter((p.x_um for p in points), dtype=np.float64, count=n)
    y = np.fromiter((p.y_um for p in points), dtype=np.float64, count=n)
    conf = np.fromiter((p.confidence for p in points), dtype=np.float64, count=n)
    z = np.full(n, np.nan, dtype=np.float64)
    labels: list[PointClassification] = []
    for i, p in enumerate(points):
        height = p.surface_z_um
        if (
            height is None
            or not math.isfinite(height)
            or p.status not in USABLE_STATUSES
            or not (math.isfinite(p.x_um) and math.isfinite(p.y_um))
        ):
            labels.append(PointClassification.INVALID)
            continue
        z[i] = height
        if p.confidence < request.min_confidence:
            labels.append(PointClassification.LOW_CONFIDENCE)
        else:
            labels.append(PointClassification.USED)

    candidates = np.flatnonzero([c is PointClassification.USED for c in labels])
    if candidates.size and request.outlier_method is not OutlierMethod.NONE:
        xy = np.column_stack((x[candidates], y[candidates]))
        zc = z[candidates]
        if request.outlier_method is OutlierMethod.LOCAL_MAD:
            flagged = local_mad_outliers(
                xy, zc, neighbours=request.outlier_neighbours, threshold=request.outlier_threshold
            )
        else:
            flagged = global_mad_outliers(xy, zc, threshold=request.outlier_threshold)
        for i in candidates[flagged]:
            labels[int(i)] = PointClassification.OUTLIER

    return ClassifiedPoints(
        point_ids=ids,
        x_um=x,
        y_um=y,
        z_um=z,
        confidence=conf,
        classification=tuple(labels),
    )
