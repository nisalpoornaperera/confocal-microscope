"""Peak detection on a Z-sorted (smoothed) I(Z) profile.

Candidates are the local maxima found by :func:`scipy.signal.find_peaks`
(flat tops count once, at their middle) plus the two sweep ends when the
signal still rises towards them (a surface just outside the sweep, which
``find_peaks`` can never report).

Prominence follows the topographic definition used by SciPy -- the drop to
the *higher* of the two side minima before reaching higher ground -- so a
peak only counts when the signal falls on both sides (except where the sweep
cuts a side off, below). Ties are broken
explicitly: a peak's own flat top is skipped, and of two separate peaks of
exactly equal height only the right-hand one keeps its full prominence (the
other is the lower "child" of the pair). Without that rule both halves of a
notched clipped top would claim full prominence and look like two peaks.

A peak within ``edge_margin_samples`` of a sweep end is truncated by the
sweep, so its prominence is measured on the inner side only; it is then
reported as *at edge* rather than silently rejected, which lets the scan
engine repeat the sweep with a wider range.

The sweep can also truncate a flank of a peak further inside: a side that
reaches the sweep end without meeting higher ground, is still at its lowest
there (within ``min_prominence``) and has not come down to the baseline was
cut off by the sweep, not by a valley. Its minimum says nothing about the
peak, so -- as for a peak at the edge -- the prominence uses the other side
only. Without this rule the real focus peak, whose outer flank the sweep
cuts, loses to a weaker secondary reflection whose flanks are complete. When
both sides are truncated (a sweep narrower than the peak) the conservative
rule above applies unchanged.

The main peak is the most prominent candidate. ``n_peaks`` counts candidates
whose prominence reaches ``min_prominence`` (normally a multiple of the noise),
and ``secondary_peak_ratio`` = second-largest / largest prominence among them:
a confocal profile of a single reflecting surface has exactly one peak, so a
strong secondary peak (internal reflection, dust, a transparent layer) makes
the surface height ambiguous. ``global_max_not_selected`` reports that the
signal somewhere exceeds the main peak by more than ``min_prominence``: the
brightest reflection was *not* chosen, so the selection is ambiguous and the
caller must not trust it as a clean measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


@dataclass(frozen=True, slots=True)
class PeakInfo:
    """One peak of a Z-sorted profile (indices refer to the analysed arrays)."""

    index: int
    z_um: float  # Z of the sample at the maximum
    z_refined_um: float  # sub-sample maximum from a three-point parabola
    height: float  # signal value at the maximum
    prominence: float
    at_edge: bool
    left_half_z_um: float | None  # interpolated half-maximum crossings (None: not reached)
    right_half_z_um: float | None

    @property
    def fwhm_um(self) -> float | None:
        """Full width at half maximum; None unless both crossings lie inside the sweep."""
        if self.left_half_z_um is None or self.right_half_z_um is None:
            return None
        return self.right_half_z_um - self.left_half_z_um

    @property
    def width_estimate_um(self) -> float | None:
        """FWHM, or twice the one-sided half width when only one crossing exists."""
        if self.fwhm_um is not None:
            return self.fwhm_um
        if self.left_half_z_um is not None:
            return 2.0 * (self.z_refined_um - self.left_half_z_um)
        if self.right_half_z_um is not None:
            return 2.0 * (self.right_half_z_um - self.z_refined_um)
        return None

    @property
    def asymmetry(self) -> float | None:
        """``(hw_right - hw_left) / (hw_right + hw_left)`` in [-1, 1]; > 0 = tail towards +Z."""
        if self.left_half_z_um is None or self.right_half_z_um is None:
            return None
        left = self.z_refined_um - self.left_half_z_um
        right = self.right_half_z_um - self.z_refined_um
        total = left + right
        if total <= 0.0:
            return None
        return float(np.clip((right - left) / total, -1.0, 1.0))


@dataclass(frozen=True, slots=True)
class PeakDetection:
    main: PeakInfo | None  # most prominent candidate, even if below min_prominence
    significant: bool  # main.prominence >= min_prominence (and > 0)
    n_peaks: int  # candidates with prominence >= min_prominence
    secondary_peak_ratio: float | None  # None without a significant main peak
    global_max_not_selected: bool = False  # signal exceeds main.height by > min_prominence


NO_DETECTION = PeakDetection(main=None, significant=False, n_peaks=0, secondary_peak_ratio=None)


def is_at_edge(index: int, n: int, margin: int) -> bool:
    """True when ``index`` is within ``margin`` samples of either end of ``n`` samples."""
    return min(index, n - 1 - index) <= margin


def refine_vertex(z: NDArray[np.float64], y: NDArray[np.float64], index: int) -> float:
    """Vertex of the parabola through the maximum and its two neighbours.

    Works for non-uniform spacing; falls back to the sample Z at the ends or
    when the three points are not concave (e.g. the middle of a flat top). The
    result is clamped between the neighbouring samples.
    """
    n = int(z.shape[0])
    if index <= 0 or index >= n - 1:
        return float(z[index])
    x1, x2, x3 = float(z[index - 1]), float(z[index]), float(z[index + 1])
    y1, y2, y3 = float(y[index - 1]), float(y[index]), float(y[index + 1])
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    if not a < 0.0:
        return x2
    return float(np.clip(-b / (2.0 * a), x1, x3))


def _interp_crossing(z0: float, z1: float, y0: float, y1: float, level: float) -> float:
    if y1 == y0:
        return 0.5 * (z0 + z1)
    return z0 + (level - y0) * (z1 - z0) / (y1 - y0)


def half_maximum_crossings(
    z: NDArray[np.float64], y: NDArray[np.float64], index: int, level: float
) -> tuple[float | None, float | None]:
    """Linearly interpolated Z where ``y`` first falls below ``level`` on each side."""
    left: float | None = None
    right: float | None = None
    below_left = np.flatnonzero(y[:index] < level)
    if below_left.size:
        i = int(below_left[-1])
        left = _interp_crossing(float(z[i]), float(z[i + 1]), float(y[i]), float(y[i + 1]), level)
    below_right = np.flatnonzero(y[index + 1 :] < level)
    if below_right.size:
        j = index + 1 + int(below_right[0])
        right = _interp_crossing(float(z[j - 1]), float(z[j]), float(y[j - 1]), float(y[j]), level)
    return left, right


def _edge_plateau_candidate(y: NDArray[np.float64], *, from_start: bool) -> int | None:
    """Middle of a run of equal values at a sweep end that is higher than its neighbour."""
    seq = y if from_start else y[::-1]
    unequal = np.flatnonzero(seq != seq[0])
    if unequal.size == 0:
        return None  # constant profile: no peak at all
    run = int(unequal[0])
    if not seq[run] < seq[0]:
        return None
    mid = (run - 1) // 2
    return mid if from_start else int(y.shape[0]) - 1 - mid


def _candidates(y: NDArray[np.float64]) -> NDArray[np.intp]:
    interior, _ = find_peaks(y)
    found = {int(i) for i in interior}
    for from_start in (True, False):
        edge = _edge_plateau_candidate(y, from_start=from_start)
        if edge is not None:
            found.add(edge)
    return np.asarray(sorted(found), dtype=np.intp)


def _side_minimum(values: list[float], index: int, *, left: bool) -> tuple[float | None, bool]:
    """Lowest value between the peak and the first higher sample on one side.

    The peak's own flat top is skipped first. On the right-hand side a sample
    of *equal* height also ends the search (the tie rule in the module
    docstring). Returns ``(lowest, open)``: ``lowest`` is ``None`` when the
    side has no samples beyond the flat top, and ``open`` is True when the
    search ran into the sweep end without meeting higher ground.

    A plain loop over a list: most candidates of a noisy profile end their
    scan after a few samples, which makes this ~10x faster than NumPy calls on
    ~50-sample profiles.
    """
    height = values[index]
    step = -1 if left else 1
    n = len(values)
    i = index + step
    while 0 <= i < n and values[i] == height:
        i += step
    lowest: float | None = None
    while 0 <= i < n:
        value = values[i]
        if value > height or (not left and value == height):
            return lowest, False
        if lowest is None or value < lowest:
            lowest = value
        i += step
    return lowest, True


def _prominence(
    values: list[float], index: int, margin: int, *, baseline: float, tolerance: float
) -> float:
    n = len(values)
    left_min, left_open = _side_minimum(values, index, left=True)
    right_min, right_open = _side_minimum(values, index, left=False)
    if is_at_edge(index, n, margin):
        # Only the inner side is trustworthy: the sweep truncates the outer one.
        inner = right_min if index < n - 1 - index else left_min
        bases = [] if inner is None else [inner]
    else:
        sides = [
            (left_min, left_open, values[0]),
            (right_min, right_open, values[-1]),
        ]
        bases = [b for b, _, _ in sides if b is not None]
        # A side cut off by the sweep (still falling at the sweep end, above
        # the baseline) does not bound the prominence; see the module docstring.
        complete = [
            b
            for b, is_open, end in sides
            if b is not None and not (is_open and end - b <= tolerance and b > baseline + tolerance)
        ]
        if complete:
            bases = complete
    if not bases:
        return 0.0
    return max(0.0, values[index] - max(bases))


def detect_peaks(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    baseline: float,
    min_prominence: float,
    edge_margin_samples: int,
) -> PeakDetection:
    """Find the main peak of a Z-sorted, finite profile and describe it.

    The half-maximum level is ``baseline + (height - baseline) / 2``; widths
    are interpolated linearly between samples (sub-sample resolution).
    Profiles shorter than 3 samples have no peak by definition.
    """
    n = int(y.shape[0])
    if n < 3:
        return NO_DETECTION
    candidates = _candidates(y)
    if candidates.size == 0:
        return NO_DETECTION

    values = [float(v) for v in y]
    prominences = np.array(
        [
            _prominence(
                values, int(i), edge_margin_samples, baseline=baseline, tolerance=min_prominence
            )
            for i in candidates
        ],
        dtype=np.float64,
    )
    # Most prominent first; ties broken by height, then by position.
    order = np.lexsort((candidates, -y[candidates], -prominences))
    best = int(order[0])
    index = int(candidates[best])
    main_prominence = float(prominences[best])
    significant_mask = prominences >= min_prominence
    significant = bool(significant_mask[best]) and main_prominence > 0.0

    height = float(y[index])
    left: float | None = None
    right: float | None = None
    if height > baseline:
        left, right = half_maximum_crossings(z, y, index, baseline + 0.5 * (height - baseline))
    main = PeakInfo(
        index=index,
        z_um=float(z[index]),
        z_refined_um=refine_vertex(z, y, index),
        height=height,
        prominence=main_prominence,
        at_edge=is_at_edge(index, n, edge_margin_samples),
        left_half_z_um=left,
        right_half_z_um=right,
    )
    if not significant:
        return PeakDetection(main=main, significant=False, n_peaks=0, secondary_peak_ratio=None)
    global_max_not_selected = max(values) - height > max(min_prominence, 0.0)

    strong = np.sort(prominences[significant_mask])
    n_peaks = int(strong.shape[0])
    secondary = float(strong[-2] / main_prominence) if n_peaks > 1 else 0.0
    return PeakDetection(
        main=main,
        significant=True,
        n_peaks=n_peaks,
        secondary_peak_ratio=secondary,
        global_max_not_selected=global_max_not_selected,
    )
