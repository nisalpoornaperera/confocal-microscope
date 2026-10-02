"""Single-position intensity readings and ADC range selection.

A reading is a burst of raw ADS1115 conversions at one stage position. It is
reduced exactly like every Z position of an I(Z) profile, through the shared
primitives of :mod:`confocal.processing.normalization`, so a manual reading
and a scan agree on the same sample:

    V          = mean / median of the burst
    corrected  = V - V_dark                        (V_dark = 0 V when uncalibrated)
    normalized = (V - V_dark) / (V_ref - V_dark)   (None without a usable reference)

Saturation is detected on the raw codes: a code on either int16 rail means the
photodiode voltage exceeded the PGA full scale, so the aggregated voltage is
only a lower bound and must not be trusted as an intensity.

These functions do not serialise hardware access themselves; the controller
calls them while holding its hardware lock.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from confocal.errors import ADCError
from confocal.hardware.ads1115.conversion import is_saturated, select_gain
from confocal.hardware.base import ADC
from confocal.models.calibration import (
    ADCCalibrateRequest,
    ADCCalibrateResponse,
    CalibrationState,
)
from confocal.models.common import Position
from confocal.models.hardware import AdcGain, SamplingMethod
from confocal.models.measurement import AdcSamples, IntensityMeasurement
from confocal.processing.normalization import aggregate_samples, normalize, subtract_dark

#: Conversions taken per auto-ranging reading (enough to see the noise peak).
AUTO_GAIN_SAMPLES = 16

#: Least sensitive PGA range (+/-6.144 V): above the OPT101 output rail, so it
#: cannot clip a real photodiode signal.
WIDEST_GAIN = AdcGain.G2_3


def sample_std(values: NDArray[np.float64]) -> float:
    """Sample standard deviation (ddof=1); 0 for a single sample."""
    if values.shape[0] < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def finite_volts(samples: AdcSamples) -> NDArray[np.float64]:
    """The burst's voltages, refusing non-finite values (API models must stay JSON-safe).

    Raises:
        ADCError: if the ADC driver produced NaN or infinite voltages.
    """
    volts = np.asarray(samples.volts, dtype=np.float64)
    if not bool(np.all(np.isfinite(volts))):
        raise ADCError("the ADC returned non-finite voltages")
    return volts


def aggregate_voltage(samples: AdcSamples, method: SamplingMethod) -> float:
    """Mean or median voltage of a burst."""
    return float(aggregate_samples(finite_volts(samples), method))


def build_intensity_measurement(
    samples: AdcSamples,
    method: SamplingMethod,
    calibration: CalibrationState,
    *,
    position: Position | None,
) -> IntensityMeasurement:
    """Aggregate, dark-correct and (when calibrated) normalize one burst.

    ``normalized`` is ``None`` unless the calibration has a reference strictly
    above the dark level (:attr:`CalibrationState.can_normalize`); a reading is
    never normalized against an inconsistent calibration.
    """
    volts = finite_volts(samples)
    voltage = float(aggregate_samples(volts, method))
    dark = calibration.dark_v
    corrected = float(subtract_dark(voltage, dark))
    normalized: float | None = None
    if calibration.can_normalize:
        value = normalize(voltage, dark, calibration.reference_v)
        if value is not None and math.isfinite(float(value)):
            normalized = float(value)
    return IntensityMeasurement(
        position=position,
        n_samples=samples.n,
        method=method,
        gain=samples.gain,
        raw_counts=[int(c) for c in samples.counts],
        voltages_v=[float(v) for v in volts],
        voltage_v=voltage,
        voltage_std_v=sample_std(volts),
        dark_v=dark,
        corrected_v=corrected,
        reference_v=calibration.reference_v,
        normalized=normalized,
        calibration_version=calibration.version,
        saturated=is_saturated(samples.counts),
    )


async def configure_adc(adc: ADC, request: ADCCalibrateRequest) -> ADCCalibrateResponse:
    """Apply an explicit PGA gain, or auto-range on the signal at the current position.

    Auto-ranging reads a burst at the current gain. A saturated burst says only
    that the signal is above the current full scale, so the ADC is switched to
    the widest range (2/3) and read again before choosing. The most sensitive
    gain that keeps the maximum below ``target_fraction`` of full scale is then
    selected (:func:`select_gain`). The operator should range at the brightest
    position (in focus on the most reflective area) so the scan peak fits.
    """
    if request.gain is not None:
        await adc.set_gain(request.gain)
        return ADCCalibrateResponse(
            gain=request.gain,
            full_scale_v=request.gain.full_scale_v,
            message=f"ADC gain set to {request.gain.value} (+/-{request.gain.full_scale_v} V)",
        )

    samples = await adc.read_samples(AUTO_GAIN_SAMPLES)
    if is_saturated(samples.counts) and adc.gain is not WIDEST_GAIN:
        await adc.set_gain(WIDEST_GAIN)
        samples = await adc.read_samples(AUTO_GAIN_SAMPLES)
    measured_max = float(np.max(np.abs(finite_volts(samples))))
    if is_saturated(samples.counts):
        gain = WIDEST_GAIN
        message = (
            f"signal saturates even the widest range (+/-{gain.full_scale_v} V): "
            "reduce the light (ND filter) and check the detector wiring"
        )
    else:
        gain = select_gain(measured_max, request.target_fraction)
        message = (
            f"auto-ranged at the current position: max {measured_max:.4f} V -> gain "
            f"{gain.value} (+/-{gain.full_scale_v} V, target "
            f"{request.target_fraction:.0%} of full scale)"
        )
    if adc.gain is not gain:
        await adc.set_gain(gain)
    return ADCCalibrateResponse(
        gain=gain,
        full_scale_v=gain.full_scale_v,
        measured_max_v=measured_max,
        message=message,
    )
