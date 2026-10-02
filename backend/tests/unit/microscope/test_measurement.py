"""Intensity readings (aggregation, dark correction, normalization) and ADC ranging."""

from __future__ import annotations

import numpy as np
import pytest
from tests.unit.microscope.conftest import (
    FOCUS_Z_UM,
    InMemoryCalibrationStore,
    RigFactory,
    ScriptedADC,
)

from confocal.microscope import build_intensity_measurement
from confocal.models.calibration import ADCCalibrateRequest, CalibrationState
from confocal.models.common import Position
from confocal.models.hardware import AdcGain, SamplingMethod
from confocal.models.measurement import AdcSamples

# Gain 1: one LSB = 4.096 V / 32768 = 0.125 mV, so 8000 codes = 1.000 V.
CODES = [8000, 8000, 8000, 8400]  # 1.0, 1.0, 1.0, 1.05 V


def _samples(codes: list[int], gain: AdcGain = AdcGain.G1) -> AdcSamples:
    counts = np.asarray(codes, dtype=np.int32)
    return AdcSamples(
        counts=counts,
        volts=counts.astype(np.float64) * gain.lsb_v,
        timestamps=np.arange(counts.shape[0], dtype=np.float64),
        gain=gain,
        data_rate_sps=860,
    )


async def test_measure_intensity_applies_the_loaded_calibration(make_rig: RigFactory) -> None:
    store = InMemoryCalibrationStore([CalibrationState(dark_v=0.1, reference_v=2.0)])
    rig = await make_rig(adc=ScriptedADC(CODES), store=store)
    await rig.controller.move_to(x_um=3.0)

    median = await rig.controller.measure_intensity(4, SamplingMethod.MEDIAN)
    assert median.voltage_v == pytest.approx(1.0)
    assert median.dark_v == pytest.approx(0.1)
    assert median.corrected_v == pytest.approx(0.9)
    assert median.reference_v == pytest.approx(2.0)
    assert median.normalized == pytest.approx(0.9 / 1.9)
    assert median.calibration_version == 1
    assert median.raw_counts == CODES
    assert median.voltages_v == pytest.approx([1.0, 1.0, 1.0, 1.05])
    assert median.voltage_std_v == pytest.approx(float(np.std([1.0, 1.0, 1.0, 1.05], ddof=1)))
    assert median.position == Position(x_um=3.0, y_um=0.0, z_um=0.0)
    assert median.gain is AdcGain.G1
    assert not median.saturated

    mean = await rig.controller.measure_intensity(4, SamplingMethod.MEAN)
    assert mean.voltage_v == pytest.approx(1.0125)
    assert mean.normalized == pytest.approx((1.0125 - 0.1) / 1.9)


def test_uncalibrated_reading_has_no_normalization() -> None:
    reading = build_intensity_measurement(
        _samples(CODES), SamplingMethod.MEDIAN, CalibrationState(), position=None
    )
    assert reading.dark_v is None
    assert reading.corrected_v == pytest.approx(1.0)
    assert reading.normalized is None
    assert reading.calibration_version is None


def test_inconsistent_calibration_is_not_used_for_normalization() -> None:
    calibration = CalibrationState(dark_v=0.5, reference_v=0.4)
    reading = build_intensity_measurement(
        _samples(CODES), SamplingMethod.MEDIAN, calibration, position=None
    )
    assert reading.corrected_v == pytest.approx(0.5)
    assert reading.normalized is None


def test_single_sample_has_zero_std() -> None:
    reading = build_intensity_measurement(
        _samples([8000]), SamplingMethod.MEAN, CalibrationState(), position=None
    )
    assert reading.voltage_std_v == 0.0


@pytest.mark.parametrize("rail", [32767, -32768])
def test_codes_on_the_rails_flag_saturation(rail: int) -> None:
    reading = build_intensity_measurement(
        _samples([8000, rail, 8000]), SamplingMethod.MEDIAN, CalibrationState(), position=None
    )
    assert reading.saturated


async def test_acquire_returns_raw_samples(make_rig: RigFactory) -> None:
    rig = await make_rig(adc=ScriptedADC(CODES))
    samples = await rig.controller.acquire(6)
    assert samples.n == 6
    assert samples.counts.tolist() == [8000, 8000, 8000, 8400, 8000, 8000]


async def test_explicit_gain(make_rig: RigFactory) -> None:
    rig = await make_rig()
    response = await rig.controller.configure_adc(ADCCalibrateRequest(gain=AdcGain.G4))
    assert response.gain is AdcGain.G4
    assert response.full_scale_v == pytest.approx(1.024)
    assert response.measured_max_v is None
    assert rig.adc.gain is AdcGain.G4


async def test_auto_gain_steps_down_from_saturation(make_rig: RigFactory) -> None:
    rig = await make_rig(gain=AdcGain.G16)  # +/-0.256 V: saturated in focus
    await rig.controller.move_to(z_um=FOCUS_Z_UM)
    response = await rig.controller.configure_adc(
        ADCCalibrateRequest(auto=True, target_fraction=0.8)
    )
    # ~2.6 V in focus: gain 1 (4.096 V x 0.8 = 3.28 V) is the most sensitive that fits.
    assert response.gain is AdcGain.G1
    assert rig.adc.gain is AdcGain.G1
    assert response.measured_max_v is not None
    assert 2.3 < response.measured_max_v < 2.9
    assert "auto-ranged" in response.message


async def test_auto_gain_raises_sensitivity_for_a_weak_signal(make_rig: RigFactory) -> None:
    rig = await make_rig(gain=AdcGain.G1)
    await rig.controller.move_to(z_um=-150.0)  # background + dark only (~58 mV)
    response = await rig.controller.configure_adc(ADCCalibrateRequest(auto=True))
    assert response.gain is AdcGain.G16
    assert rig.adc.gain is AdcGain.G16


async def test_auto_gain_reports_saturation_at_the_widest_range(make_rig: RigFactory) -> None:
    rig = await make_rig(adc=ScriptedADC([32767], gain=AdcGain.G2_3))
    response = await rig.controller.configure_adc(ADCCalibrateRequest(auto=True))
    assert response.gain is AdcGain.G2_3
    assert "saturates even the widest range" in response.message
