"""SimulationADC: the I(Z) peak through the full simulated detection chain."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from confocal.config import MotionConfig, SimulatedSurfaceConfig, SimulationConfig
from confocal.errors import ADCError, HardwareNotConnectedError
from confocal.hardware.ads1115 import CODE_MAX
from confocal.hardware.simulation import (
    SimulatedConfocalSurface,
    SimulationADC,
    SimulationLaser,
    SimulationStage,
)
from confocal.models.common import Position, default_stage_limits
from confocal.models.hardware import AdcGain

CLEAN_SURFACE = SimulatedSurfaceConfig(
    kind="composite", low_reflectivity_fraction=0.0, spurious_peak_probability=0.0
)


def _config(**overrides: object) -> SimulationConfig:
    return SimulationConfig.model_validate(
        {"time_scale": 0.0, "surface": CLEAN_SURFACE.model_dump(), **overrides}
    )


class Rig:
    """Stage + laser + ADC sharing one surface, as the factory wires them."""

    def __init__(
        self, config: SimulationConfig, *, gain: AdcGain = AdcGain.G2, seed: int = 5
    ) -> None:
        self.surface = SimulatedConfocalSurface.from_config(config)
        self.stage = SimulationStage(
            default_stage_limits(), motion=MotionConfig(), time_scale=config.time_scale
        )
        self.laser = SimulationLaser()
        self.adc = SimulationADC(
            self.surface,
            position_source=lambda: self.stage.current_position,
            laser_source=lambda: self.laser.enabled,
            config=config,
            gain=gain,
            data_rate_sps=860,
            seed=seed,
        )

    async def connect(self) -> Rig:
        for component in (self.stage, self.laser, self.adc):
            await component.connect()
        return self


async def test_iz_peak_is_at_the_true_height_within_a_fraction_of_the_fwhm() -> None:
    config = _config()
    rig = await Rig(config, gain=AdcGain.G1).connect()  # +/-4.096 V: the 2.6 V peak fits
    x, y = 120.0, -75.0
    height = float(rig.surface.height_um(x, y))
    z_values = np.arange(height - 15.0, height + 15.0, 0.5)
    intensities = []
    for z in z_values:
        await rig.stage.move_to(Position(x_um=x, y_um=y, z_um=float(z)))
        samples = await rig.adc.read_samples(8)
        intensities.append(float(np.median(samples.volts)))
    peak_z = float(z_values[int(np.argmax(intensities))])
    assert abs(peak_z - height) <= 0.15 * config.axial_fwhm_um
    expected_peak = config.dark_voltage_v + config.background_v + 0.9 * config.peak_voltage_v
    assert max(intensities) == pytest.approx(expected_peak, rel=0.05)


async def test_dark_level_with_laser_off() -> None:
    config = _config()
    rig = await Rig(config, gain=AdcGain.G16).connect()
    await rig.laser.set_enabled(False)
    surface_z = float(rig.surface.height_um(0.0, 0.0))
    await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=surface_z))  # in focus
    samples = await rig.adc.read_samples(400)
    assert float(np.mean(samples.volts)) == pytest.approx(config.dark_voltage_v, abs=0.0005)
    assert float(np.std(samples.volts)) == pytest.approx(config.read_noise_v, rel=0.2)


async def test_samples_are_quantised_with_the_gain() -> None:
    rig = await Rig(_config(), gain=AdcGain.G2).connect()
    samples = await rig.adc.read_samples(16)
    assert samples.counts.dtype == np.int32
    assert samples.gain is AdcGain.G2
    assert samples.data_rate_sps == 860
    np.testing.assert_array_equal(samples.volts, samples.counts * AdcGain.G2.lsb_v)
    await rig.adc.set_gain(AdcGain.G1)
    assert (await rig.adc.read_samples(1)).gain is AdcGain.G1
    np.testing.assert_allclose(rig.adc.counts_to_volts(samples.counts), samples.counts * 0.000125)


async def test_saturation_at_pga_full_scale() -> None:
    config = _config(peak_voltage_v=3.0, background_v=0.0, opt101_saturation_v=3.7)
    rig = await Rig(config, gain=AdcGain.G2).connect()
    z = float(rig.surface.height_um(0.0, 0.0))
    await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=z))
    samples = await rig.adc.read_samples(10)
    assert np.all(samples.counts == CODE_MAX)
    status = await rig.adc.status()
    assert status.saturated
    assert status.full_scale_v == 2.048


async def test_opt101_output_rail_clips_before_the_adc() -> None:
    config = _config(peak_voltage_v=6.0, opt101_saturation_v=3.7, shot_noise_fraction=0.0)
    rig = await Rig(config, gain=AdcGain.G2_3).connect()
    z = float(rig.surface.height_um(0.0, 0.0))
    await rig.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=z))
    samples = await rig.adc.read_samples(50)
    assert float(np.max(samples.volts)) <= 3.7 + AdcGain.G2_3.lsb_v
    assert float(np.median(samples.volts)) == pytest.approx(3.7, abs=0.003)
    assert not (await rig.adc.status()).saturated


async def test_noise_is_deterministic_per_seed() -> None:
    first = await (await Rig(_config(), seed=11).connect()).adc.read_samples(64)
    again = await (await Rig(_config(), seed=11).connect()).adc.read_samples(64)
    other = await (await Rig(_config(), seed=12).connect()).adc.read_samples(64)
    np.testing.assert_array_equal(first.counts, again.counts)
    assert not np.array_equal(first.counts, other.counts)


async def test_conversion_time_is_scaled() -> None:
    rig = await Rig(_config(time_scale=1.0)).connect()
    loop = asyncio.get_running_loop()
    started = loop.time()
    samples = await rig.adc.read_samples(43)  # 43 / 860 s = 50 ms
    assert loop.time() - started >= 0.04
    np.testing.assert_allclose(np.diff(samples.timestamps), 1.0 / 860, rtol=1e-3)


async def test_fault_injection_fails_the_next_read_once() -> None:
    rig = await Rig(_config(adc_fault_after_reads=1)).connect()
    await rig.adc.read_samples(1)
    with pytest.raises(ADCError, match="injected"):
        await rig.adc.read_samples(1)
    assert (await rig.adc.status()).last_error is not None
    await rig.adc.read_samples(1)
    assert rig.adc.reads == 3
    assert (await rig.adc.status()).last_error is None


async def test_requires_connection_and_valid_arguments() -> None:
    rig = Rig(_config())
    with pytest.raises(HardwareNotConnectedError):
        await rig.adc.read_samples(1)
    await rig.connect()
    with pytest.raises(ValueError, match="n must be"):
        await rig.adc.read_samples(0)
    with pytest.raises(ValueError, match="data rate"):
        SimulationADC(
            rig.surface,
            position_source=lambda: rig.stage.current_position,
            laser_source=lambda: True,
            config=_config(),
            gain=AdcGain.G1,
            data_rate_sps=1000,
            seed=0,
        )


async def test_status_reports_last_voltage() -> None:
    rig = await Rig(_config()).connect()
    samples = await rig.adc.read_samples(4)
    status = await rig.adc.status()
    assert status.connected
    assert status.last_voltage_v == pytest.approx(float(np.mean(samples.volts)))
    assert status.data_rate_sps == rig.adc.data_rate_sps == 860
