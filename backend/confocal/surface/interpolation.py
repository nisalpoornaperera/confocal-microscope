"""Scattered-data interpolation of heights (and confidence) onto the output grid.

All coordinates passed here are normalised to the scan step (``(x - x0) /
xy_step``): Qhull and the RBF polynomial terms are best conditioned for O(1)
coordinates, and the RBF shape parameter ``epsilon = 1`` then means "one scan
step" for the kernels that need one (gaussian, multiquadric, ...). For the
scale-free kernels (thin-plate spline, linear, cubic, quintic) epsilon has no
effect, and ``rbf_smoothing`` acts in these normalised units.

Methods
-------
nearest   value of the nearest used point; defined everywhere.
linear    barycentric interpolation on the Delaunay triangulation; NaN outside
          the convex hull of the used points.
cubic     Clough-Tocher C1 interpolant on the same triangulation; NaN outside.
rbf       ``scipy.interpolate.RBFInterpolator``; defined everywhere (it
          extrapolates, which gap detection then masks). A global RBF solves a
          dense n x n system -- O(n^3) time, O(n^2) memory -- so above
          :data:`GLOBAL_RBF_MAX_POINTS` points a local RBF over the
          :data:`DEFAULT_LOCAL_RBF_NEIGHBOURS` nearest points is used unless the
          request sets ``rbf_neighbors`` explicitly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RBFInterpolator
from scipy.spatial import Delaunay

from confocal.errors import ReconstructionError
from confocal.models.surface import InterpolationMethod, RbfKernel, ReconstructionRequest
from confocal.surface.mesh import is_two_dimensional

#: Largest number of points interpolated with a global RBF (~1 s on a desktop).
GLOBAL_RBF_MAX_POINTS = 2000

#: Neighbourhood of the automatically selected local RBF.
DEFAULT_LOCAL_RBF_NEIGHBOURS = 32

#: Minimum polynomial degree SciPy requires for a well-posed RBF interpolant
#: (kernels not listed have none and get a constant term).
_RBF_MIN_DEGREE: dict[str, int] = {
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
    "linear": 0,
    "multiquadric": 0,
}


def rbf_degree(kernel: RbfKernel) -> int:
    """Degree of the polynomial term added to the RBF (SciPy's default for the kernel)."""
    return _RBF_MIN_DEGREE.get(kernel, 0)


def minimum_points(method: InterpolationMethod, kernel: RbfKernel) -> int:
    """Smallest number of used points the method can work with."""
    if method is InterpolationMethod.NEAREST:
        return 1
    if method is InterpolationMethod.RBF:
        degree = rbf_degree(kernel)
        return (degree + 1) * (degree + 2) // 2  # monomials of a 2-D polynomial
    return 3


def check_interpolable(
    xy: NDArray[np.float64], method: InterpolationMethod, kernel: RbfKernel
) -> None:
    """Raise :class:`ReconstructionError` if ``method`` cannot use these points."""
    n = int(xy.shape[0])
    needed = minimum_points(method, kernel)
    if n < needed:
        raise ReconstructionError(
            f"{method.value} interpolation needs at least {needed} usable points, got {n}"
        )
    needs_plane = method in (InterpolationMethod.LINEAR, InterpolationMethod.CUBIC) or (
        method is InterpolationMethod.RBF and rbf_degree(kernel) >= 1
    )
    if needs_plane and not is_two_dimensional(xy):
        raise ReconstructionError(
            f"{method.value} interpolation needs usable points that are not all on one line"
        )


def rbf_neighbours(request: ReconstructionRequest, n_points: int) -> int | None:
    """Neighbourhood size for the RBF (None = global), never more than the points."""
    k = request.rbf_neighbors
    if k is None and n_points > GLOBAL_RBF_MAX_POINTS:
        k = DEFAULT_LOCAL_RBF_NEIGHBOURS
    return None if k is None else min(k, n_points)


def _rbf(
    xy: NDArray[np.float64],
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
    request: ReconstructionRequest,
) -> NDArray[np.float64]:
    neighbours = rbf_neighbours(request, int(xy.shape[0]))
    needed = minimum_points(InterpolationMethod.RBF, request.rbf_kernel)
    if neighbours is not None and neighbours < needed:
        raise ReconstructionError(
            f"rbf_neighbors={neighbours} is below the {needed} points the "
            f"'{request.rbf_kernel}' kernel needs"
        )
    try:
        interpolator = RBFInterpolator(
            xy,
            values,
            kernel=request.rbf_kernel,
            smoothing=request.rbf_smoothing,
            neighbors=neighbours,
            epsilon=1.0,
        )
        return np.asarray(interpolator(targets), dtype=np.float64)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ReconstructionError(f"RBF interpolation failed: {exc}") from exc


def interpolate_heights(
    method: InterpolationMethod,
    xy: NDArray[np.float64],
    z: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    triangulation: Delaunay | None,
    nearest_index: NDArray[np.intp],
    request: ReconstructionRequest,
) -> NDArray[np.float64]:
    """Heights at ``targets`` (m, 2); NaN where the method has no value.

    ``nearest_index`` (index of the nearest used point of every target) and
    ``triangulation`` are shared with gap detection and the mesh so that
    neither is computed twice.
    """
    if method is InterpolationMethod.NEAREST:
        return np.asarray(z[nearest_index], dtype=np.float64)
    if method is InterpolationMethod.RBF:
        return _rbf(xy, z, targets, request)
    if triangulation is None:
        raise ReconstructionError(f"{method.value} interpolation needs a triangulation")
    if method is InterpolationMethod.LINEAR:
        interpolator = LinearNDInterpolator(triangulation, z)
    else:
        interpolator = CloughTocher2DInterpolator(triangulation, z)
    return np.asarray(interpolator(targets), dtype=np.float64).reshape(-1)


def interpolate_confidence(
    confidence: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    triangulation: Delaunay | None,
    nearest_index: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Confidence map: linear inside the convex hull, nearest point outside.

    Linear interpolation is a convex combination, so the map stays within the
    range of the measured confidences whatever height method is used.
    """
    nearest = confidence[nearest_index]
    if triangulation is None:
        return np.asarray(nearest, dtype=np.float64)
    linear = np.asarray(
        LinearNDInterpolator(triangulation, confidence)(targets), dtype=np.float64
    ).reshape(-1)
    return np.clip(np.where(np.isfinite(linear), linear, nearest), 0.0, 1.0)
