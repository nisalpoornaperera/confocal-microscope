"""HIL: ADS1115ADC on the real I2C bus (reads only; nothing moves)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
import pytest
from tests.hil.helpers import HIL_I2C, adc_channel, i2c_address, requires_adc

from confocal.hardware.ads1115.adc import ADS1115ADC, ConversionMode
from confocal.models.hardware import AdcGain

pytestmark = requires_adc

RATE = 860


async def _adc(mode: ConversionMode, gain: AdcGain = AdcGain.G2) -> ADS1115ADC:
    adc = ADS1115ADC(
        bus=int(HIL_I2C or "1"),
        address=i2c_address(),
        channel=adc_channel(),
        gain=gain,
        data_rate_sps=RATE,
        mode=mode,
    )
    await adc.connect()
    return adc


@pytest.fixture(params=["single-shot", "continuous"])
async def adc(request: pytest.FixtureRequest) -> AsyncIterator[ADS1115ADC]:
    driver = await _adc(request.param)
    try:
        yield driver
    finally:
        await driver.close()


async def test_burst_of_conversions(adc: ADS1115ADC) -> None:
    samples = await adc.read_samples(64)
    assert samples.n == 64
    # OPT101 on 3.3 V: inside the input window GND - 0.3 V ... VDD + 0.3 V.
    assert float(np.min(samples.volts)) > -0.3
    assert float(np.max(samples.volts)) < 3.6
    gaps = np.diff(samples.timestamps)
    assert np.all(gaps > 0)
    # At least one conversion period between samples (single-shot waits for each one;
    # continuous paces reads 1.2 periods apart), the internal clock is within +/-10 %.
    assert float(np.median(gaps)) >= 0.85 / RATE
    status = await adc.status()
    assert status.connected
    assert status.last_error is None
    assert status.last_voltage_v == pytest.approx(float(np.mean(samples.volts)))


async def test_gain_change(adc: ADS1115ADC) -> None:
    wide = await adc.read_samples(16)
    await adc.set_gain(AdcGain.G1)
    narrow = await adc.read_samples(16)
    assert narrow.gain is AdcGain.G1
    assert wide.gain is AdcGain.G2
    # The same (steady) input reads about the same voltage at both gains.
    assert float(np.median(narrow.volts)) == pytest.approx(float(np.median(wide.volts)), abs=0.05)
