"""ADS1115ADC against a register-level model of the chip (no smbus2, no real bus)."""

from __future__ import annotations

import errno

import numpy as np
import pytest
from tests.unit.hardware.fakes import EREMOTEIO, FakeADS1115, FakeClock

from confocal.config import ADS1115Config
from confocal.errors import ADCError, HardwareNotConnectedError
from confocal.hardware.ads1115.adc import (
    ADS1115ADC,
    CONTINUOUS_PACING,
    ConversionMode,
    encode_config,
    open_smbus,
)
from confocal.models.hardware import AdcGain


class Rig:
    def __init__(self, mode: ConversionMode = "single-shot", **adc: object) -> None:
        self.clock = FakeClock()
        self.chip = FakeADS1115(self.clock)
        self.opened: list[int] = []

        def factory(bus: int) -> FakeADS1115:
            self.opened.append(bus)
            return self.chip

        self.adc = ADS1115ADC(
            mode=mode,
            bus_factory=factory,
            sleep=self.clock.sleep,
            monotonic=self.clock.monotonic,
            wall_clock=self.clock.wall,
            **adc,  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- config word


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # AIN0 single-ended, +/-4.096 V, 128 SPS, single-shot, comparator off, start.
        (
            {
                "mux": 0b100,
                "gain": AdcGain.G1,
                "data_rate_sps": 128,
                "continuous": False,
                "start": True,
            },
            0xC383,
        ),
        # AIN0-AIN1, +/-2.048 V, 860 SPS, continuous.
        (
            {"mux": 0b000, "gain": AdcGain.G2, "data_rate_sps": 860, "continuous": True},
            0x04E3,
        ),
        # AIN3, +/-0.256 V, 8 SPS, single-shot idle.
        (
            {"mux": 0b111, "gain": AdcGain.G16, "data_rate_sps": 8, "continuous": False},
            0x7B03,
        ),
    ],
)
def test_config_word_matches_the_datasheet(kwargs: dict[str, object], expected: int) -> None:
    assert encode_config(**kwargs) == expected  # type: ignore[arg-type]


# --------------------------------------------------------------------------- lifecycle


async def test_connect_writes_and_verifies_the_configuration() -> None:
    rig = Rig(gain=AdcGain.G2, data_rate_sps=860, channel=0, bus=1)
    await rig.adc.connect()
    assert rig.adc.connected
    assert rig.opened == [1]
    word = rig.chip.writes[-1]
    assert (word >> 12) & 0b111 == 0b100  # AIN0 vs GND
    assert (word >> 9) & 0b111 == 0b010  # +/-2.048 V
    assert (word >> 5) & 0b111 == 0b111  # 860 SPS
    assert word & 0x0100  # single-shot (idle)
    assert word & 0b11 == 0b11  # comparator disabled
    await rig.adc.connect()  # idempotent
    assert rig.opened == [1]
    await rig.adc.close()
    assert not rig.adc.connected
    assert rig.chip.closed
    assert rig.chip.writes[-1] & 0x0100  # left powered down
    await rig.adc.close()


async def test_connect_without_a_chip_is_an_adc_error() -> None:
    rig = Rig(address=0x49)  # nothing answers at 0x49
    with pytest.raises(ADCError, match="0x49") as excinfo:
        await rig.adc.connect()
    assert "check wiring" in str(excinfo.value)
    assert not rig.adc.connected
    assert rig.chip.closed
    status = await rig.adc.status()
    assert not status.connected
    assert status.last_error is not None


async def test_a_device_that_ignores_the_config_is_not_an_ads1115() -> None:
    rig = Rig()
    rig.chip.write_i2c_block_data = lambda *args: None  # type: ignore[method-assign]
    with pytest.raises(ADCError, match="not an ADS1115"):
        await rig.adc.connect()


async def test_bus_factory_os_error_is_an_adc_error() -> None:
    def broken(bus: int) -> FakeADS1115:
        raise OSError(errno.ENOENT, "No such file or directory: '/dev/i2c-1'")

    adc = ADS1115ADC(bus_factory=broken)
    with pytest.raises(ADCError, match="I2C bus 1"):
        await adc.connect()


def test_smbus2_is_imported_lazily_and_fails_clearly_off_linux() -> None:
    try:
        import smbus2  # noqa: F401
    except ImportError:
        with pytest.raises(ADCError, match="smbus2 cannot be used"):
            open_smbus(1)
    else:  # pragma: no cover - Linux: /dev/i2c-99 does not exist
        with pytest.raises(ADCError):
            open_smbus(99)


async def test_reads_before_connect_raise() -> None:
    rig = Rig()
    with pytest.raises(HardwareNotConnectedError):
        await rig.adc.read_samples(1)
    with pytest.raises(HardwareNotConnectedError):
        await rig.adc.set_gain(AdcGain.G4)


def test_invalid_construction() -> None:
    with pytest.raises(ValueError, match="data rate"):
        ADS1115ADC(data_rate_sps=100)
    with pytest.raises(ValueError, match="channel"):
        ADS1115ADC(channel=4)
    with pytest.raises(ValueError, match="address"):
        ADS1115ADC(address=0x50)


# --------------------------------------------------------------------------- acquisition


@pytest.mark.parametrize("mode", ["single-shot", "continuous"])
@pytest.mark.parametrize("rate", [8, 128, 860])
async def test_samples_are_distinct_conversions_at_the_data_rate(
    mode: ConversionMode, rate: int
) -> None:
    rig = Rig(mode, gain=AdcGain.G1, data_rate_sps=rate)
    await rig.adc.connect()
    rig.clock.sleep(5.0)  # the converter has been running for a while
    called_at = rig.clock.now
    index_at_call = rig.chip.current_index()
    samples = await rig.adc.read_samples(20)
    assert samples.n == 20
    assert samples.gain is AdcGain.G1
    assert samples.data_rate_sps == rate
    # Distinct conversions: the tag (conversion index) differs between samples.
    assert len(np.unique(samples.counts)) == 20
    assert np.all(np.diff(samples.counts) > 0)
    # Fresh: even the first sample comes from a conversion started after the call.
    first_tag = int(samples.counts[0]) - round(0.5 / 4.096 * 32768)
    if mode == "single-shot":
        assert rig.chip.started_at[-20] >= called_at
        assert first_tag == index_at_call + 1
    else:  # conversion index_at_call + 1 was already running when the call came
        assert first_tag >= index_at_call + 2
    # Timing follows the data rate (single-shot ~1 period, continuous ~1.2 periods each).
    elapsed = rig.clock.now - called_at
    per_sample = elapsed / 20
    expected = (1.0 if mode == "single-shot" else CONTINUOUS_PACING) / rate
    assert expected * 0.95 <= per_sample <= expected * 1.4
    # Timestamps are increasing wall-clock times of each conversion.
    assert np.all(np.diff(samples.timestamps) > 0)
    assert samples.timestamps[-1] == pytest.approx(rig.clock.wall())


@pytest.mark.parametrize("oscillator", [0.9, 1.1])
@pytest.mark.parametrize("mode", ["single-shot", "continuous"])
async def test_distinct_even_with_oscillator_tolerance(
    mode: ConversionMode, oscillator: float
) -> None:
    rig = Rig(mode, data_rate_sps=860)
    rig.chip.oscillator = oscillator
    await rig.adc.connect()
    samples = await rig.adc.read_samples(50)
    assert len(np.unique(samples.counts)) == 50


async def test_counts_to_volts_with_the_gain() -> None:
    rig = Rig(gain=AdcGain.G2)
    rig.chip.tag_conversions = False
    rig.chip.inputs["AIN0"] = 1.234
    await rig.adc.connect()
    samples = await rig.adc.read_samples(4)
    assert np.all(samples.counts == round(1.234 / 2.048 * 32768))
    np.testing.assert_allclose(samples.volts, 1.234, atol=AdcGain.G2.lsb_v)
    status = await rig.adc.status()
    assert status.last_voltage_v == pytest.approx(1.234, abs=AdcGain.G2.lsb_v)
    assert not status.saturated
    assert status.channel == 0
    assert status.full_scale_v == pytest.approx(2.048)


@pytest.mark.parametrize(
    ("channel", "differential", "expected_v"),
    [
        (0, False, 0.5),
        (1, False, 0.25),
        (3, False, 0.1),
        (0, True, 0.25),  # AIN0 - AIN1
        (1, True, 0.4),  # AIN0 - AIN3
        (2, True, 0.15),  # AIN1 - AIN3
        (3, True, -0.1),  # AIN2 - AIN3
    ],
)
async def test_input_multiplexer(channel: int, differential: bool, expected_v: float) -> None:
    rig = Rig(channel=channel, differential=differential, gain=AdcGain.G1)
    rig.chip.tag_conversions = False
    await rig.adc.connect()
    samples = await rig.adc.read_samples(2)
    np.testing.assert_allclose(samples.volts, expected_v, atol=AdcGain.G1.lsb_v)


async def test_saturation_is_reported() -> None:
    rig = Rig(gain=AdcGain.G4)  # +/-1.024 V
    rig.chip.tag_conversions = False
    rig.chip.inputs["AIN0"] = 1.9
    await rig.adc.connect()
    samples = await rig.adc.read_samples(3)
    assert np.all(samples.counts == 32767)
    assert (await rig.adc.status()).saturated


@pytest.mark.parametrize("mode", ["single-shot", "continuous"])
async def test_set_gain_applies_to_the_next_conversion(mode: ConversionMode) -> None:
    rig = Rig(mode, gain=AdcGain.G1)
    rig.chip.tag_conversions = False
    await rig.adc.connect()
    await rig.adc.read_samples(2)
    await rig.adc.set_gain(AdcGain.G8)  # +/-0.512 V: 0.5 V is just inside
    assert rig.adc.gain is AdcGain.G8
    samples = await rig.adc.read_samples(2)
    assert samples.gain is AdcGain.G8
    assert np.all(samples.counts == round(0.5 / 0.512 * 32768))
    assert (rig.chip.config >> 9) & 0b111 == 0b100


async def test_i2c_failure_during_a_read_is_an_adc_error() -> None:
    rig = Rig()
    await rig.adc.connect()
    rig.chip.fail_next = OSError(EREMOTEIO, "Remote I/O error")
    with pytest.raises(ADCError, match="Remote I/O error"):
        await rig.adc.read_samples(5)
    assert (await rig.adc.status()).last_error is not None
    samples = await rig.adc.read_samples(5)  # the next read works again
    assert samples.n == 5
    assert (await rig.adc.status()).last_error is None


async def test_conversion_that_never_completes_times_out() -> None:
    rig = Rig(data_rate_sps=860)
    await rig.adc.connect()
    rig.chip.oscillator = 1000.0  # a stuck converter
    with pytest.raises(ADCError, match="did not complete"):
        await rig.adc.read_samples(1)


async def test_continuous_set_gain_failure_keeps_the_old_gain() -> None:
    rig = Rig("continuous", gain=AdcGain.G1)
    await rig.adc.connect()
    rig.chip.fail_next = OSError(errno.EIO, "I/O error")
    with pytest.raises(ADCError):
        await rig.adc.set_gain(AdcGain.G2)
    assert rig.adc.gain is AdcGain.G1


async def test_rejects_zero_samples() -> None:
    rig = Rig()
    await rig.adc.connect()
    with pytest.raises(ValueError, match="n must be"):
        await rig.adc.read_samples(0)


def test_from_config_and_identity() -> None:
    config = ADS1115Config(
        i2c_bus=3, address=0x4A, channel=2, differential=True, gain=AdcGain.G2, data_rate_sps=250
    )
    adc = ADS1115ADC.from_config(config, mode="continuous")
    assert adc.bus_number == 3
    assert adc.address == 0x4A
    assert adc.gain is AdcGain.G2
    assert adc.data_rate_sps == 250
    assert adc.mode == "continuous"
    assert adc.input_description == "AIN1-AIN3"
    assert adc.version() == "ADS1115 i2c-3@0x4A (continuous)"
    assert ADS1115ADC().input_description == "AIN0-GND"
