"""ADS1115 code <-> voltage conversion and PGA range selection (pure, no I/O).

The ADS1115 is a 16-bit delta-sigma converter with a programmable gain
amplifier (PGA). Every conversion is a signed two's-complement code in
``[-32768, +32767]``; one LSB is ``full_scale_v / 32768``. The OPT101 output is
single-ended (0 V up to its output rail, about 2 V on this machine's 3.3 V
supply), so only the positive half of the code range is used: 15 bits of
effective resolution.

An input beyond the PGA full scale does not wrap: the converter returns the
rail code (+32767 / -32768). A reading containing rail codes is therefore
*saturated* and must not be trusted as an intensity (the true signal is
somewhere above full scale). Independently of the PGA, the analogue inputs must
stay within GND - 0.3 V ... VDD + 0.3 V: with the ADS1115 on 3.3 V the
+/-6.144 V and +/-4.096 V ranges only cost resolution, they never allow larger
inputs. The OPT101 on the same 3.3 V supply cannot exceed that window.

``ADS1115ADC`` (the I2C driver, ``.adc``) builds on these functions; the
simulated ADC uses them too, so simulated and real codes are identical.
"""

from __future__ import annotations

import math
from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from confocal.models.hardware import AdcGain

#: Output data rates (samples per second) supported by the ADS1115.
VALID_DATA_RATES: tuple[int, ...] = (8, 16, 32, 64, 128, 250, 475, 860)

#: Signed 16-bit code range of the converter.
CODE_MIN = -32768
CODE_MAX = 32767

#: PGA settings ordered from the most sensitive (smallest full scale) down.
GAINS_HIGH_TO_LOW: tuple[AdcGain, ...] = (
    AdcGain.G16,
    AdcGain.G8,
    AdcGain.G4,
    AdcGain.G2,
    AdcGain.G1,
    AdcGain.G2_3,
)


@overload
def counts_to_volts(counts: int | np.integer[Any], gain: AdcGain) -> float: ...


@overload
def counts_to_volts(counts: NDArray[Any], gain: AdcGain) -> NDArray[np.float64]: ...


def counts_to_volts(
    counts: int | np.integer[Any] | NDArray[Any], gain: AdcGain
) -> float | NDArray[np.float64]:
    """Convert raw signed codes to volts: ``V = counts * full_scale / 32768``."""
    volts = np.asarray(counts, dtype=np.float64) * gain.lsb_v
    if volts.ndim == 0:
        return float(volts)
    return volts


def volts_to_counts(volts: ArrayLike, gain: AdcGain) -> NDArray[np.int32]:
    """Ideal quantisation of an input voltage to ADS1115 codes.

    Rounds to the nearest code and clips to the int16 rails, exactly like the
    converter saturates at full scale. Accepts scalars or arrays and always
    returns an ``int32`` array (0-d for a scalar input).

    Raises:
        ValueError: if any input is NaN (a NaN voltage is a modelling bug).
    """
    arr = np.asarray(volts, dtype=np.float64)
    if np.isnan(arr).any():
        raise ValueError("cannot quantise NaN voltages")
    codes = np.clip(np.rint(arr / gain.lsb_v), CODE_MIN, CODE_MAX)
    return codes.astype(np.int32)


def is_saturated(counts: ArrayLike) -> bool:
    """True if any code sits on a rail, i.e. the input exceeded the PGA full scale."""
    arr = np.asarray(counts)
    return bool(np.any((arr >= CODE_MAX) | (arr <= CODE_MIN)))


def conversion_time_s(data_rate: int) -> float:
    """Duration of one conversion at ``data_rate`` samples per second.

    Raises:
        ValueError: if the ADS1115 does not support ``data_rate``.
    """
    if data_rate not in VALID_DATA_RATES:
        raise ValueError(
            f"unsupported ADS1115 data rate {data_rate} SPS; valid: {VALID_DATA_RATES}"
        )
    return 1.0 / data_rate


def select_gain(max_abs_v: float, target_fraction: float = 0.8) -> AdcGain:
    """Most sensitive PGA setting that keeps the signal below ``target_fraction`` of full scale.

    Returns the highest gain whose ``full_scale_v * target_fraction >= |max_abs_v|``,
    keeping headroom for noise and for signals brighter than the one used for
    ranging. Falls back to the least sensitive range (2/3, +/-6.144 V) when
    nothing fits.

    Raises:
        ValueError: for a non-finite signal or a fraction outside (0, 1].
    """
    if not math.isfinite(max_abs_v):
        raise ValueError(f"max_abs_v must be finite, got {max_abs_v}")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError(f"target_fraction must be in (0, 1], got {target_fraction}")
    signal = abs(max_abs_v)
    for gain in GAINS_HIGH_TO_LOW:
        if gain.full_scale_v * target_fraction >= signal:
            return gain
    return AdcGain.G2_3
