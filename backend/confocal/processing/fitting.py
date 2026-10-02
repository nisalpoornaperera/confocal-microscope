"""Peak-model fits of the axial response.

The confocal axial response of a reflecting surface is close to a Gaussian
near focus, so the surface height is the centre of a Gaussian fitted to the
I(Z) samples around the peak. A quadratic fitted to the top samples gives an
independent, model-light estimate of the same maximum; the profile analysis
uses their agreement as a sanity check.

Neither fit ever raises on bad data: degenerate, noisy or pathological input
yields ``PeakFit(success=False, message=...)``. Both accept an ``exclude``
mask for samples that are not part of either model -- above all saturated
samples: the OPT101 clips easily, and a flat top fitted as if it were the
peak drags the Gaussian down and widens it, while its flanks still locate the
centre accurately.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeWarning, curve_fit

from confocal.models.processing import FitMethod, PeakFit

#: FWHM of a Gaussian in units of its standard deviation: 2 sqrt(2 ln 2) = 2.3548.
GAUSSIAN_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

MIN_PARABOLIC_POINTS = 3
MIN_GAUSSIAN_POINTS = 5
DEFAULT_GAUSSIAN_MAXFEV = 400

# Relative distance to a bound below which a fitted parameter counts as pinned.
_PINNED_TOLERANCE = 1e-6


def _finite_sorted(
    *arrays: ArrayLike, exclude: ArrayLike | None = None
) -> tuple[NDArray[np.float64], ...]:
    """Float64 copies restricted to samples finite in every array and not excluded,
    sorted by the first array."""
    arrs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs):
        raise ValueError("fit arrays must have equal lengths")
    keep = np.ones(n, dtype=np.bool_)
    if exclude is not None:
        excluded = np.asarray(exclude, dtype=np.bool_).reshape(-1)
        if excluded.shape[0] != n:
            raise ValueError("exclude mask must have the same length as the samples")
        keep &= ~excluded
    for a in arrs:
        keep &= np.isfinite(a)
    order = np.argsort(arrs[0][keep], kind="stable")
    return tuple(a[keep][order] for a in arrs)


def _goodness(
    y: NDArray[np.float64], predicted: NDArray[np.float64]
) -> tuple[float | None, float | None]:
    """(residual RMS, R^2); None where undefined or non-finite."""
    residual = y - predicted
    rms = float(np.sqrt(np.mean(residual * residual)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / ss_tot if ss_tot > 0.0 else None
    return (rms if math.isfinite(rms) else None), (
        r2 if r2 is not None and math.isfinite(r2) else None
    )


def _failed(method: FitMethod, message: str, n_points: int = 0) -> PeakFit:
    return PeakFit(method=method, success=False, n_points=n_points, message=message)


# --------------------------------------------------------------------------- parabolic


def _top_run(select: NDArray[np.float64], peak: int, level: float, min_points: int) -> slice:
    """Contiguous samples around ``peak`` at or above ``level``, grown to ``min_points``."""
    n = int(select.shape[0])
    below = select < level
    left_below = np.flatnonzero(below[:peak])
    right_below = np.flatnonzero(below[peak + 1 :])
    lo = int(left_below[-1]) + 1 if left_below.size else 0
    hi = peak + int(right_below[0]) if right_below.size else n - 1
    while hi - lo + 1 < min_points and (lo > 0 or hi < n - 1):
        if lo > 0:
            lo -= 1
        if hi - lo + 1 < min_points and hi < n - 1:
            hi += 1
    return slice(lo, hi + 1)


def fit_parabolic(
    z: ArrayLike,
    y: ArrayLike,
    *,
    baseline: float | None = None,
    top_fraction: float = 0.5,
    select_by: ArrayLike | None = None,
    exclude: ArrayLike | None = None,
    min_points: int = MIN_PARABOLIC_POINTS,
) -> PeakFit:
    """Vertex of a least-squares quadratic through the top of the peak.

    The *top samples* are the contiguous run around the maximum whose values
    are at least ``base + top_fraction * (max - base)`` (``base`` =
    ``baseline``, or the minimum when it is None), grown to ``min_points`` if
    needed. ``select_by`` (e.g. a smoothed copy of ``y``) chooses those samples
    while the quadratic is fitted to ``y`` itself, which keeps noise from
    breaking the run up at low SNR. ``exclude`` (True = leave out) removes
    samples first; with a clipped top removed, the run spans both flanks just
    below the clipping level and the vertex still lands between them.

    The fit fails when the quadratic is not concave (no maximum) or its vertex
    lies outside the fitted samples (an extrapolated maximum is not measured).
    ``amplitude`` is the vertex height above ``base`` and ``offset`` is ``base``.
    """
    method = FitMethod.PARABOLIC
    try:
        if select_by is None:
            zs, ys = _finite_sorted(z, y, exclude=exclude)
            sel = ys
        else:
            zs, ys, sel = _finite_sorted(z, y, select_by, exclude=exclude)
    except ValueError as exc:
        return _failed(method, str(exc))
    n = int(zs.shape[0])
    if n < max(MIN_PARABOLIC_POINTS, min_points):
        return _failed(method, f"need at least {max(3, min_points)} finite samples, got {n}", n)

    peak = int(np.argmax(sel))
    base = float(baseline) if baseline is not None and math.isfinite(baseline) else float(sel.min())
    level = base + top_fraction * (float(sel[peak]) - base)
    window = _top_run(sel, peak, level, max(MIN_PARABOLIC_POINTS, min_points))
    zt, yt = zs[window], ys[window]
    n_top = int(zt.shape[0])
    if n_top < MIN_PARABOLIC_POINTS:
        return _failed(method, f"only {n_top} samples near the maximum", n_top)

    centre = 0.5 * float(zt[0] + zt[-1])
    scale = 0.5 * float(zt[-1] - zt[0])
    if not scale > 0.0:
        return _failed(method, "top samples span no Z range", n_top)
    u = (zt - centre) / scale
    design = np.column_stack((u * u, u, np.ones_like(u)))
    coeffs, *_ = np.linalg.lstsq(design, yt, rcond=None)
    a, b, c = (float(v) for v in coeffs)
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        return _failed(method, "quadratic fit did not produce finite coefficients", n_top)
    if a >= 0.0:
        return _failed(method, "top of the profile is not concave (no maximum)", n_top)

    u_vertex = -b / (2.0 * a)
    if not -1.0 <= u_vertex <= 1.0:
        return _failed(method, "vertex lies outside the fitted samples", n_top)
    rms, r2 = _goodness(yt, design @ coeffs)
    return PeakFit(
        method=method,
        success=True,
        center_um=centre + scale * u_vertex,
        amplitude=(c - b * b / (4.0 * a)) - base,
        offset=base,
        residual_rms=rms,
        r_squared=r2,
        n_points=n_top,
    )


# --------------------------------------------------------------------------- gaussian


def _gauss(
    u: NDArray[np.float64], offset: float, amplitude: float, centre: float, sigma: float
) -> NDArray[np.float64]:
    return offset + amplitude * np.exp(-0.5 * ((u - centre) / sigma) ** 2)


def _gauss_jac(
    u: NDArray[np.float64], offset: float, amplitude: float, centre: float, sigma: float
) -> NDArray[np.float64]:
    t = (u - centre) / sigma
    e = np.exp(-0.5 * t * t)
    ae = amplitude * e
    return np.column_stack((np.ones_like(u), e, ae * t / sigma, ae * t * t / sigma))


def _moment_sigma(z: NDArray[np.float64], weight: NDArray[np.float64]) -> float | None:
    """Standard deviation of Z weighted by the (positive) signal above its minimum."""
    total = float(weight.sum())
    if not total > 0.0:
        return None
    mean = float((weight * z).sum()) / total
    var = float((weight * (z - mean) ** 2).sum()) / total
    return math.sqrt(var) if var > 0.0 else None


def fit_gaussian(
    z: ArrayLike,
    y: ArrayLike,
    *,
    center_guess: float | None = None,
    sigma_guess: float | None = None,
    offset_guess: float | None = None,
    exclude: ArrayLike | None = None,
    maxfev: int = DEFAULT_GAUSSIAN_MAXFEV,
) -> PeakFit:
    """Bounded least-squares fit of ``offset + amplitude * exp(-(z - c)^2 / (2 sigma^2))``.

    The problem is solved in scaled units (Z relative to the initial centre in
    units of the initial sigma, signal relative to its range) so the optimiser
    is well conditioned for volts and normalized intensity alike, with an
    analytic Jacobian and at most ``maxfev`` function evaluations. Samples
    flagged in ``exclude`` (e.g. saturated) are left out.

    Initial guesses are data driven: centre = ``center_guess`` or the Z of the
    maximum, sigma = ``sigma_guess`` or the signal-weighted spread, offset =
    ``offset_guess`` (e.g. the robust baseline) or the minimum.

    Bounds: amplitude in [0, 10 x range] (a clipped top may hide a higher
    true peak), offset within one range of the data, centre within one span of
    the samples, sigma between half the median sample spacing (narrower is
    unresolvable) and twice the span (wider is not a peak). A parameter pinned
    at a bound means the model does not describe the data, so the fit fails.
    Whether the centre lies *inside* the sweep is the caller's decision.

    FWHM = 2 sqrt(2 ln 2) sigma; ``residual_rms`` and ``r_squared`` are in the
    units of ``y`` over the fitted samples.
    """
    method = FitMethod.GAUSSIAN
    try:
        zs, ys = _finite_sorted(z, y, exclude=exclude)
    except ValueError as exc:
        return _failed(method, str(exc))
    n = int(zs.shape[0])
    if n < MIN_GAUSSIAN_POINTS:
        return _failed(method, f"need at least {MIN_GAUSSIAN_POINTS} finite samples, got {n}", n)

    y_min, y_max = float(ys.min()), float(ys.max())
    y_range = y_max - y_min
    z_span = float(zs[-1] - zs[0])
    spacing = np.diff(zs)
    spacing = spacing[spacing > 0.0]
    if not y_range > 0.0:
        return _failed(method, "constant signal", n)
    if not z_span > 0.0 or spacing.size == 0:
        return _failed(method, "samples span no Z range", n)

    peak = int(np.argmax(ys))
    c0 = float(zs[peak])
    if center_guess is not None and math.isfinite(center_guess):
        c0 = float(np.clip(center_guess, zs[0], zs[-1]))
    sigma_lo = 0.5 * float(np.median(spacing))
    sigma_hi = 2.0 * z_span
    s0 = sigma_guess if sigma_guess is not None and math.isfinite(sigma_guess) else None
    if s0 is None or s0 <= 0.0:
        s0 = _moment_sigma(zs, ys - y_min) or z_span / 6.0
    s0 = float(np.clip(s0, sigma_lo * 1.01, sigma_hi * 0.99))
    o0 = float(offset_guess) if offset_guess is not None and math.isfinite(offset_guess) else y_min

    # Scaled problem: u = (z - c0) / s0, v = (y - y_min) / y_range.
    u = (zs - c0) / s0
    v = (ys - y_min) / y_range
    o0_s = float(np.clip((o0 - y_min) / y_range, -0.99, 0.99))
    a0_s = float(np.clip(float(v[peak]) - o0_s, 1e-3, 9.9))
    span_u = z_span / s0
    lower = (-1.0, 0.0, float(u[0]) - span_u, sigma_lo / s0)
    upper = (1.0, 10.0, float(u[-1]) + span_u, sigma_hi / s0)
    p0 = (o0_s, a0_s, 0.0, 1.0)

    try:
        with warnings.catch_warnings():
            # Covariance estimation is irrelevant here; a singular Jacobian at the
            # solution shows up as a pinned or non-finite parameter instead.
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                _gauss,
                u,
                v,
                p0=p0,
                bounds=(lower, upper),
                jac=_gauss_jac,
                method="trf",
                maxfev=maxfev,
            )
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return _failed(method, f"Gaussian fit did not converge: {exc}", n)

    params = np.asarray(popt, dtype=np.float64)
    if not np.all(np.isfinite(params)):
        return _failed(method, "Gaussian fit produced non-finite parameters", n)
    names = ("offset", "amplitude", "centre", "sigma")
    for name, value, lo, hi in zip(names, params, lower, upper, strict=True):
        tol = _PINNED_TOLERANCE * (hi - lo)
        if name != "offset" and (value - lo <= tol or hi - value <= tol):
            return _failed(method, f"Gaussian {name} pinned at its bound", n)

    o_s, a_s, c_s, s_s = (float(p) for p in params)
    sigma = s_s * s0
    rms, r2 = _goodness(ys, y_min + y_range * _gauss(u, o_s, a_s, c_s, s_s))
    return PeakFit(
        method=method,
        success=True,
        center_um=c0 + c_s * s0,
        amplitude=a_s * y_range,
        fwhm_um=GAUSSIAN_FWHM_PER_SIGMA * sigma,
        offset=y_min + o_s * y_range,
        residual_rms=rms,
        r_squared=r2,
        n_points=n,
    )
