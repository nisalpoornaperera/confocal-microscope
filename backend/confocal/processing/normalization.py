"""Dark subtraction, normalization and sample aggregation.

These primitives are shared by the microscope layer (single readings) and the
I(Z) profile pipeline so that both apply calibration identically:

    corrected  = V - V_dark                       (V_dark = 0 if not calibrated)
    normalized = (V - V_dark) / (V_ref - V_dark)  (None if no reference)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from confocal.errors import CalibrationError
from confocal.models.hardware import SamplingMethod


def aggregate_samples(
    samples: ArrayLike, method: SamplingMethod, *, axis: int = -1
) -> NDArray[np.float64]:
    """Reduce repeated samples with the mean or the median along ``axis``."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot aggregate an empty sample array")
    if method is SamplingMethod.MEDIAN:
        return np.asarray(np.median(arr, axis=axis), dtype=np.float64)
    return np.asarray(np.mean(arr, axis=axis), dtype=np.float64)


def subtract_dark(voltage: ArrayLike, dark_v: float | None) -> NDArray[np.float64]:
    """Dark-corrected signal. An uncalibrated dark level is treated as 0 V."""
    arr = np.asarray(voltage, dtype=np.float64)
    return arr - (0.0 if dark_v is None else float(dark_v))


def normalize(
    voltage: ArrayLike, dark_v: float | None, reference_v: float | None
) -> NDArray[np.float64] | None:
    """Normalized intensity, 1.0 at the reference level, 0.0 at the dark level.

    Returns ``None`` when no reference calibration exists.

    Raises:
        CalibrationError: if the reference is not above the dark level.
    """
    if reference_v is None:
        return None
    dark = 0.0 if dark_v is None else float(dark_v)
    span = float(reference_v) - dark
    if not np.isfinite(span) or span <= 0.0:
        raise CalibrationError(
            f"reference level ({reference_v:.6f} V) must exceed the dark level ({dark:.6f} V)"
        )
    return subtract_dark(voltage, dark_v) / span
