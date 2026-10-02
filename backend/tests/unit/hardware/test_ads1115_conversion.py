"""ADS1115 code <-> voltage conversion and PGA selection."""

from __future__ import annotations

import numpy as np
import pytest

from confocal.hardware.ads1115 import (
    CODE_MAX,
    CODE_MIN,
    GAINS_HIGH_TO_LOW,
    VALID_DATA_RATES,
    conversion_time_s,
    counts_to_volts,
    is_saturated,
    select_gain,
    volts_to_counts,
)
from confocal.models.hardware import AdcGain

FULL_SCALE = {
    AdcGain.G2_3: 6.144,
    AdcGain.G1: 4.096,
    AdcGain.G2: 2.048,
    AdcGain.G4: 1.024,
    AdcGain.G8: 0.512,
    AdcGain.G16: 0.256,
}


@pytest.mark.parametrize(("gain", "fsr"), list(FULL_SCALE.items()))
def test_full_scale_and_lsb_per_gain(gain: AdcGain, fsr: float) -> None:
    assert gain.full_scale_v == fsr
    assert gain.lsb_v == fsr / 32768
    assert counts_to_volts(CODE_MIN, gain) == -fsr
    assert counts_to_volts(CODE_MAX, gain) == pytest.approx(fsr - gain.lsb_v)


def test_gain_2_lsb_is_62_5_microvolts() -> None:
    assert AdcGain.G2.lsb_v == 62.5e-6
    assert counts_to_volts(16000, AdcGain.G2) == 1.0


def test_counts_to_volts_scalar_and_array() -> None:
    assert isinstance(counts_to_volts(np.int16(5), AdcGain.G1), float)
    volts = counts_to_volts(np.array([0, 1, -1], dtype=np.int32), AdcGain.G1)
    assert volts.dtype == np.float64
    np.testing.assert_array_equal(volts, [0.0, AdcGain.G1.lsb_v, -AdcGain.G1.lsb_v])


def test_volts_to_counts_rounds_to_nearest_code() -> None:
    lsb = AdcGain.G2.lsb_v
    codes = volts_to_counts([0.4 * lsb, 0.6 * lsb, -0.6 * lsb, 1.0], AdcGain.G2)
    assert codes.dtype == np.int32
    np.testing.assert_array_equal(codes, [0, 1, -1, 16000])


def test_round_trip_is_exact_on_codes() -> None:
    codes = np.arange(CODE_MIN, CODE_MAX + 1, 997, dtype=np.int32)
    for gain in AdcGain:
        np.testing.assert_array_equal(volts_to_counts(counts_to_volts(codes, gain), gain), codes)


def test_volts_to_counts_clips_at_full_scale() -> None:
    codes = volts_to_counts([5.0, -5.0, 2.048], AdcGain.G2)
    np.testing.assert_array_equal(codes, [CODE_MAX, CODE_MIN, CODE_MAX])
    assert volts_to_counts(3.0, AdcGain.G2).shape == ()


def test_volts_to_counts_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        volts_to_counts([0.1, float("nan")], AdcGain.G1)


def test_is_saturated() -> None:
    assert not is_saturated([0, 100, -100])
    assert is_saturated([0, CODE_MAX])
    assert is_saturated([CODE_MIN])


def test_conversion_time() -> None:
    assert VALID_DATA_RATES == (8, 16, 32, 64, 128, 250, 475, 860)
    assert conversion_time_s(860) == pytest.approx(1 / 860)
    assert conversion_time_s(8) == 0.125
    with pytest.raises(ValueError, match="data rate"):
        conversion_time_s(100)


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        (0.0, AdcGain.G16),
        (0.2, AdcGain.G16),
        (0.21, AdcGain.G8),
        (1.0, AdcGain.G2),  # 1.024 * 0.8 < 1.0 <= 2.048 * 0.8
        (1.6, AdcGain.G2),
        (1.7, AdcGain.G1),
        (-1.7, AdcGain.G1),  # magnitude counts
        (4.0, AdcGain.G2_3),
        (10.0, AdcGain.G2_3),  # nothing fits: least sensitive range
    ],
)
def test_select_gain(signal: float, expected: AdcGain) -> None:
    assert select_gain(signal) is expected


def test_select_gain_target_fraction() -> None:
    assert select_gain(1.0, target_fraction=1.0) is AdcGain.G4
    assert GAINS_HIGH_TO_LOW[0] is AdcGain.G16
    with pytest.raises(ValueError, match="target_fraction"):
        select_gain(1.0, target_fraction=0.0)
    with pytest.raises(ValueError, match="finite"):
        select_gain(float("inf"))
