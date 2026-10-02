"""Smoothing and robust baseline / noise estimation for I(Z) profiles.

Filters act in *sample* space. Z sweeps are uniformly stepped, so a window of
``filter_window`` samples spans a fixed Z extent; the window must stay well
below the number of samples per axial FWHM or the filter broadens the peak
(Savitzky-Golay preserves the peak shape best, which is why it is the default).

Noise is estimated from second differences, ``y[i-1] - 2 y[i] + y[i+1]``:
for white noise their standard deviation is ``sqrt(6) * sigma``, while a
slowly varying background or the tails of a well-sampled peak contribute
almost nothing. The median absolute deviation (MAD) makes the estimate
insensitive to the few samples at a sharply curved peak top and to isolated
spikes. Where curvature does leak in (finely sampled, very high SNR peaks) the
noise is *over*-estimated, which only lowers an SNR that is far above any
threshold anyway: the error is conservative.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

from confocal.models.processing import FilterMethod, ProcessingConfig

#: Converts the MAD of normally distributed data into its standard deviation.
MAD_TO_STD = 1.482602218505602

#: Noise is never reported below this fraction of the profile's peak-to-peak
#: range (nor below an absolute 1e-12), so SNR stays finite on noise-free data.
#: Real measurements always carry ADC quantisation noise far above this floor.
NOISE_FLOOR_FRACTION = 1e-6
_ABS_NOISE_FLOOR = 1e-12

#: Minimum number of off-peak samples for an explicit exclusion mask to be used.
MIN_OFF_PEAK_SAMPLES = 5

#: The off-peak region only defines the noise when it yields at least this many
#: second differences: a MAD of n values has a relative error of ~1.2 / sqrt(n),
#: so a handful of samples can under-estimate the noise several-fold.
MIN_NOISE_DIFFERENCES = 16

#: Fewer second differences than this give no usable spread estimate at all.
_MIN_SECOND_DIFFERENCES = 3

#: Profiles up to this length are smoothed with a cached Savitzky-Golay matrix.
_MAX_CACHED_SAVGOL_LENGTH = 256


def _largest_odd_at_most(value: int) -> int:
    return value if value % 2 == 1 else value - 1


def _fill_non_finite(y: NDArray[np.float64], finite: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Linear interpolation (over the sample index) across non-finite samples."""
    if finite.all():
        return y
    idx = np.arange(y.shape[0], dtype=np.float64)
    return np.interp(idx, idx[finite], y[finite])


@lru_cache(maxsize=32)
def _savgol_matrix(n: int, window: int, polyorder: int) -> NDArray[np.float64]:
    """The linear operator of ``savgol_filter(y, window, polyorder, mode="interp")``.

    Computing Savitzky-Golay coefficients and edge fits costs ~0.5 ms per call,
    ten times the filtering itself; every profile of a scan has the same
    length, so the operator is built once and applied as a matrix product.
    """
    matrix = np.asarray(
        savgol_filter(np.eye(n), window, polyorder, axis=0, mode="interp"), dtype=np.float64
    )
    matrix.setflags(write=False)
    return matrix


def filter_signal(signal: ArrayLike, config: ProcessingConfig) -> NDArray[np.float64]:
    """Smooth a profile with ``config.filter_method``; safe for any length.

    * The window is clipped to the largest odd length that fits the profile;
      profiles shorter than 3 samples are returned unfiltered (as a copy).
    * Savitzky-Golay uses ``min(savgol_polyorder, window - 1)``.
    * Non-finite samples are bridged by linear interpolation for the filter
      and returned as NaN, so they never masquerade as measurements.
    """
    y = np.asarray(signal, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("signal must be 1-D")
    n = int(y.shape[0])
    finite = np.isfinite(y)
    if config.filter_method is FilterMethod.NONE or n < 3 or not finite.any():
        return y.copy()

    work = _fill_non_finite(y, finite)
    window = _largest_odd_at_most(min(config.filter_window, n))
    if config.filter_method is FilterMethod.SAVGOL:
        polyorder = min(config.savgol_polyorder, window - 1)
        if n <= _MAX_CACHED_SAVGOL_LENGTH:
            filtered = _savgol_matrix(n, window, polyorder) @ work
        else:
            filtered = savgol_filter(work, window, polyorder, mode="interp")
    elif config.filter_method is FilterMethod.MEDIAN:
        filtered = median_filter(work, size=window, mode="nearest")
    else:
        filtered = gaussian_filter1d(work, sigma=config.gaussian_sigma_samples, mode="nearest")

    out = np.array(filtered, dtype=np.float64)
    out[~finite] = np.nan
    return out


def _mad_std(values: NDArray[np.float64]) -> float:
    """Robust standard deviation: 1.4826 x median absolute deviation from the median."""
    if values.size == 0:
        return 0.0
    return MAD_TO_STD * float(np.median(np.abs(values - np.median(values))))


def _second_differences(y: NDArray[np.float64], keep: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Second differences of adjacent samples whose three samples are all in ``keep``."""
    if y.shape[0] < 3:
        return np.empty(0, dtype=np.float64)
    d2 = y[:-2] - 2.0 * y[1:-1] + y[2:]
    usable = keep[:-2] & keep[1:-1] & keep[2:]
    return np.asarray(d2[usable], dtype=np.float64)


def estimate_baseline_and_noise(
    signal: ArrayLike,
    *,
    exclude: ArrayLike | None = None,
    off_peak_quantile: float = 0.5,
) -> tuple[float, float]:
    """Robust ``(baseline, noise_std)`` of a profile in its own units.

    Off-peak region:
        * when ``exclude`` (True = belongs to the peak) leaves at least
          :data:`MIN_OFF_PEAK_SAMPLES` finite samples, everything outside it;
        * otherwise the samples at or below the ``off_peak_quantile`` of the
          signal (the lower half by default): a confocal peak only ever raises
          the signal, so the lowest values are the defocused background. On a
          fine sweep that hardly leaves the peak this over-estimates the
          baseline, which under-estimates the SNR: conservative.

    ``baseline`` is the median of the off-peak region. ``noise_std`` is the
    MAD of the second differences divided by sqrt(6) (see the module
    docstring): over the off-peak region when it was defined by position
    (``exclude``) and yields at least :data:`MIN_NOISE_DIFFERENCES`
    differences, over the whole profile otherwise -- selecting samples *by
    value* would bias the spread low. Second differences only use runs of
    consecutive finite samples, so NaN-ed samples (e.g. a clipped top) never
    bridge a gap. Profiles too short for second differences fall back to the
    MAD of the off-peak values. The noise is floored at
    :data:`NOISE_FLOOR_FRACTION` of the peak-to-peak range.

    Samples must be in Z order. Non-finite samples are ignored.

    Raises:
        ValueError: if the signal has no finite sample.
    """
    y = np.asarray(signal, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("signal must be 1-D")
    finite = np.isfinite(y)
    if not finite.any():
        raise ValueError("cannot estimate a baseline without finite samples")

    peak_region = (
        np.zeros(y.shape, dtype=np.bool_)
        if exclude is None
        else np.asarray(exclude, dtype=np.bool_)
    )
    if peak_region.shape != y.shape:
        raise ValueError("exclude must have the same shape as the signal")
    by_position = finite & ~peak_region

    values = y[finite]
    filled = np.where(finite, y, 0.0)
    d2 = np.empty(0, dtype=np.float64)
    if exclude is not None and int(by_position.sum()) >= MIN_OFF_PEAK_SAMPLES:
        off_peak = y[by_position]
        d2 = _second_differences(filled, by_position)
    else:
        threshold = float(np.quantile(values, off_peak_quantile))
        off_peak = values[values <= threshold]
    if d2.size < MIN_NOISE_DIFFERENCES:
        d2 = _second_differences(filled, finite)

    baseline = float(np.median(off_peak))
    if d2.size >= _MIN_SECOND_DIFFERENCES:
        noise = _mad_std(d2) / math.sqrt(6.0)
    else:
        noise = _mad_std(off_peak)
    span = float(np.max(values) - np.min(values))
    noise = max(noise, NOISE_FLOOR_FRACTION * span, _ABS_NOISE_FLOOR)
    return baseline, noise
