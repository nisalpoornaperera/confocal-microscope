"""Validation, ordering and de-duplication of raw I(Z) samples.

The analysis works on the Z values *reported by the stage*, so a profile can
arrive unsorted (e.g. a re-centred sweep), contain repeated positions (stage
read-back is quantised to the delta stage's step grid) or carry non-finite
values (a failed conversion stored as NaN). Peak detection and interpolated
widths need a strictly increasing Z axis, so every analysis starts from a
:class:`PreparedProfile`:

* samples whose Z or signal is not finite are dropped;
* the remaining samples are sorted by Z (stable sort);
* samples at exactly the same Z are merged: mean signal, saturated if any of
  the merged samples was saturated.

``source_index`` maps each input sample to its prepared sample (``-1`` when
dropped) so that processed arrays can be handed back in the caller's order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class PreparedProfile:
    """A profile with a strictly increasing Z axis and only finite samples."""

    z_um: NDArray[np.float64]  # (m,) strictly increasing
    signal: NDArray[np.float64]  # (m,) mean signal of the input samples at each Z
    saturated: NDArray[np.bool_]  # (m,) any merged input sample was saturated
    source_index: NDArray[np.intp]  # (n_input,) prepared index per input sample, -1 = dropped
    first_input_index: NDArray[np.intp]  # (m,) first input sample merged into each sample
    n_dropped: int  # input samples with a non-finite Z or signal
    n_merged: int  # input samples merged into another sample at the same Z

    @property
    def n(self) -> int:
        return int(self.z_um.shape[0])

    @property
    def n_input(self) -> int:
        return int(self.source_index.shape[0])

    @property
    def saturated_fraction(self) -> float:
        """Fraction of the analysed (prepared) samples that are saturated."""
        return float(np.mean(self.saturated)) if self.n else 0.0

    def to_input_order(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scatter per-prepared-sample ``values`` back to input order (NaN where dropped)."""
        out = np.full(self.n_input, np.nan, dtype=np.float64)
        kept = self.source_index >= 0
        out[kept] = values[self.source_index[kept]]
        return out


def as_profile_arrays(
    z_um: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert a (Z, value) pair to float64 1-D arrays of equal length.

    Raises:
        ValueError: if either array is not 1-D or their lengths differ. This is
            a caller bug, not bad measurement data, so it is not tolerated.
    """
    z = np.asarray(z_um, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if z.ndim != 1 or y.ndim != 1:
        raise ValueError("profile arrays must be 1-D")
    if z.shape != y.shape:
        raise ValueError(f"z ({z.shape[0]}) and signal ({y.shape[0]}) lengths differ")
    return z, y


def saturation_mask(
    voltage_v: NDArray[np.float64], saturation_v: float | None
) -> NDArray[np.bool_]:
    """Samples whose *raw* voltage is at or above the saturation level.

    Saturation is a property of the detector output (OPT101 rail, ADC full
    scale), so it is judged on the raw voltage, before dark subtraction.
    """
    if saturation_v is None:
        return np.zeros(voltage_v.shape, dtype=np.bool_)
    with np.errstate(invalid="ignore"):
        return np.asarray(voltage_v >= saturation_v, dtype=np.bool_)


def fill_short_gaps(mask: NDArray[np.bool_], max_gap: int) -> NDArray[np.bool_]:
    """Also set runs of at most ``max_gap`` False samples enclosed by True samples.

    A clipped peak top is one physical feature even when noise lets a sample
    in its middle dip just below the saturation threshold; treating that
    sample as a valid measurement would put a spurious notch in the top.
    """
    out = np.array(mask, dtype=np.bool_, copy=True)
    idx = np.flatnonzero(out)
    if idx.size < 2 or max_gap <= 0:
        return out
    gaps = np.diff(idx) - 1
    for start, gap in zip(idx[:-1], gaps, strict=True):
        if 0 < gap <= max_gap:
            out[start + 1 : start + 1 + gap] = True
    return out


def prepare_profile(
    z_um: NDArray[np.float64],
    signal: NDArray[np.float64],
    saturated: NDArray[np.bool_] | None = None,
) -> PreparedProfile:
    """Drop non-finite samples, sort by Z and merge samples at identical Z."""
    z, y = as_profile_arrays(z_um, signal)
    sat = (
        np.zeros(z.shape, dtype=np.bool_)
        if saturated is None
        else np.asarray(saturated, dtype=np.bool_)
    )
    if sat.shape != z.shape:
        raise ValueError("saturation mask must have the same shape as the profile")

    valid_idx = np.flatnonzero(np.isfinite(z) & np.isfinite(y))
    order = valid_idx[np.argsort(z[valid_idx], kind="stable")]
    unique_z, first, inverse, counts = np.unique(
        z[order], return_index=True, return_inverse=True, return_counts=True
    )
    m = int(unique_z.shape[0])
    inverse = inverse.reshape(-1)
    merged = np.bincount(inverse, weights=y[order], minlength=m) / np.maximum(counts, 1)
    sat_merged = np.bincount(inverse, weights=sat[order].astype(np.float64), minlength=m) > 0.0

    source = np.full(z.shape[0], -1, dtype=np.intp)
    source[order] = inverse
    return PreparedProfile(
        z_um=np.asarray(unique_z, dtype=np.float64),
        signal=np.asarray(merged, dtype=np.float64),
        saturated=np.asarray(sat_merged, dtype=np.bool_),
        source_index=source,
        first_input_index=np.asarray(order[first], dtype=np.intp),
        n_dropped=int(z.shape[0] - valid_idx.shape[0]),
        n_merged=int(valid_idx.shape[0] - m),
    )
