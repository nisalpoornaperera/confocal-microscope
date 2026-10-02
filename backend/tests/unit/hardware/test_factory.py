"""build_hardware: simulation wiring, limit checks and refusal of unavailable backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.unit.hardware.fakes import FakeADS1115, RealTime, emulated_transport

from confocal.config import (
    ArduinoConfig,
    HardwareSelection,
    MotorConfig,
    Settings,
    load_settings,
)
from confocal.errors import HardwareConfigError, MotionError
from confocal.hardware import HardwareSet, build_hardware
from confocal.hardware.ads1115 import ADS1115ADC
from confocal.hardware.arduino import ArduinoStage
from confocal.hardware.simulation import (
    ManualLaser,
    NullCamera,
    SimulatedConfocalSurface,
    SimulationADC,
    SimulationCamera,
    SimulationLaser,
    SimulationStage,
)
from confocal.models.common import AxisLimits, Position, StageLimits
from confocal.models.hardware import AdcGain

PI_CONFIG = Path(__file__).resolve().parents[4] / "config" / "confocal.pi.toml"


def _with(settings: Settings, **selection: str) -> Settings:
    hardware = HardwareSelection.model_validate(settings.hardware.model_dump() | selection)
    return settings.model_copy(update={"hardware": hardware})


async def _connect(hardware: HardwareSet) -> None:
    for component in (hardware.stage, hardware.adc, hardware.laser, hardware.camera):
        await component.connect()


async def test_simulation_build_shares_one_surface(sim_settings: Settings) -> None:
    hardware = build_hardware(sim_settings)
    assert isinstance(hardware.stage, SimulationStage)
    assert isinstance(hardware.adc, SimulationADC)
    assert isinstance(hardware.laser, SimulationLaser)
    assert isinstance(hardware.camera, NullCamera)
    assert isinstance(hardware.surface, SimulatedConfocalSurface)
    assert hardware.stage.kinematics is not None
    assert hardware.stage.kinematics.name == "delta"
    assert hardware.adc.gain is sim_settings.ads1115.gain

    await _connect(hardware)
    x, y = 30.0, -20.0
    height = float(hardware.surface.height_um(x, y))
    await hardware.stage.move_to(Position(x_um=x, y_um=y, z_um=height))
    in_focus = float(np.mean((await hardware.adc.read_samples(16)).volts))
    await hardware.stage.move_to(Position(x_um=x, y_um=y, z_um=height + 30.0))
    defocused = float(np.mean((await hardware.adc.read_samples(16)).volts))
    assert in_focus > 10 * defocused

    await hardware.laser.set_enabled(False)
    dark = float(np.mean((await hardware.adc.read_samples(64)).volts))
    assert dark == pytest.approx(sim_settings.simulation.dark_voltage_v, abs=0.002)


async def test_simulation_camera_follows_the_stage(sim_settings: Settings) -> None:
    hardware = build_hardware(_with(sim_settings, camera="simulation"))
    assert isinstance(hardware.camera, SimulationCamera)
    await _connect(hardware)
    assert (await hardware.camera.capture()).dtype == np.uint8


async def test_manual_laser(sim_settings: Settings) -> None:
    hardware = build_hardware(_with(sim_settings, laser="manual"))
    assert isinstance(hardware.laser, ManualLaser)
    assert not hardware.laser.controllable
    await _connect(hardware)
    # The simulated detector treats a manual laser as on.
    assert hardware.surface is not None
    height = float(hardware.surface.height_um(0.0, 0.0))
    await hardware.stage.move_to(Position(x_um=0.0, y_um=0.0, z_um=height))
    assert float(np.mean((await hardware.adc.read_samples(8)).volts)) > 0.5  # dark is 7.5 mV


async def test_simulation_faults_and_adc_settings_are_passed_through(
    sim_settings: Settings,
) -> None:
    settings = sim_settings.model_copy(
        update={
            "simulation": sim_settings.simulation.model_copy(update={"fault_after_moves": 3}),
            "ads1115": sim_settings.ads1115.model_copy(
                update={"gain": AdcGain.G2, "data_rate_sps": 250}
            ),
        }
    )
    hardware = build_hardware(settings)
    assert hardware.adc.gain is AdcGain.G2
    assert hardware.adc.data_rate_sps == 250
    await _connect(hardware)
    for x_um in (1.0, 2.0, 3.0):
        await hardware.stage.move_to(Position(x_um=x_um, y_um=0.0, z_um=0.0))
    with pytest.raises(MotionError, match="injected"):
        await hardware.stage.move_to(Position(x_um=4.0, y_um=0.0, z_um=0.0))


def test_refuses_limits_outside_motor_travel(sim_settings: Settings) -> None:
    tight = MotorConfig(min_steps=-1000, max_steps=1000)
    settings = sim_settings.model_copy(update={"arduino": ArduinoConfig(a=tight, b=tight, c=tight)})
    with pytest.raises(HardwareConfigError, match="motor travel") as excinfo:
        build_hardware(settings)
    message = str(excinfo.value)
    assert "motor a" in message
    assert "motor c" in message


def test_refuses_limits_that_only_fail_with_invert(sim_settings: Settings) -> None:
    upward = MotorConfig(min_steps=-100, max_steps=200_000)
    settings = sim_settings.model_copy(
        update={
            "limits": StageLimits(
                x=AxisLimits(min_um=-1.0, max_um=1.0),
                y=AxisLimits(min_um=-1.0, max_um=1.0),
                z=AxisLimits(min_um=0.0, max_um=100.0),
            ),
            "arduino": ArduinoConfig(a=upward, b=upward, c=upward),
        }
    )
    build_hardware(settings)  # safe without inversion
    inverted = ArduinoConfig(a=upward, b=upward, c=MotorConfig(min_steps=-100, invert=True))
    with pytest.raises(HardwareConfigError, match="invert"):
        build_hardware(settings.model_copy(update={"arduino": inverted}))


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ({"stage": "openflexure", "adc": "ads1115"}, "OpenFlexureClient"),
        ({"laser": "gpio"}, "no GPIO laser driver"),
        ({"stage": "arduino", "adc": "simulation"}, "requires stage 'simulation'"),
        ({"stage": "arduino", "adc": "ads1115", "camera": "simulation"}, "camera 'simulation'"),
    ],
)
def test_unsupported_backends_raise(
    sim_settings: Settings, selection: dict[str, str], expected: str
) -> None:
    with pytest.raises(HardwareConfigError, match=expected):
        build_hardware(_with(sim_settings, **selection))


def test_machine_config_builds_the_real_drivers_without_opening_anything() -> None:
    settings = load_settings(PI_CONFIG)
    hardware = build_hardware(settings)  # nothing is opened: no port, no bus needed
    assert isinstance(hardware.stage, ArduinoStage)
    assert isinstance(hardware.adc, ADS1115ADC)
    assert isinstance(hardware.laser, ManualLaser)
    assert isinstance(hardware.camera, NullCamera)
    assert hardware.surface is None
    assert not hardware.stage.connected
    assert not hardware.adc.connected
    assert hardware.stage.transport is None
    assert hardware.stage.mapper.kinematics.name == "delta"
    assert hardware.adc.gain is AdcGain.G2
    assert hardware.adc.data_rate_sps == 860
    assert hardware.adc.address == 0x48
    assert hardware.adc.bus_number == 1
    assert hardware.adc.input_description == "AIN0-GND"


async def test_machine_config_runs_end_to_end_with_injected_fakes() -> None:
    settings = load_settings(PI_CONFIG)
    transport = emulated_transport(settings.arduino)
    chip = FakeADS1115(RealTime(), {"AIN0": 1.2}, tag_conversions=False)
    opened: list[int] = []

    def bus_factory(bus: int) -> FakeADS1115:
        opened.append(bus)
        return chip

    hardware = build_hardware(
        settings, transport_factory=lambda: transport, i2c_bus_factory=bus_factory
    )
    assert isinstance(hardware.adc, ADS1115ADC)
    await _connect(hardware)
    try:
        assert transport.opens == 1
        assert opened == [1]
        target = Position(x_um=25.0, y_um=-10.0, z_um=5.0)
        reported = await hardware.stage.move_to(target)
        assert reported.max_axis_error(target) < settings.motion.position_tolerance_um
        samples = await hardware.adc.read_samples(8)
        assert samples.gain is AdcGain.G2
        assert float(np.min(samples.volts)) >= 1.2 - AdcGain.G2.lsb_v
        status = await hardware.stage.status()
        assert status.connected
        assert status.firmware_version is not None
    finally:
        for component in (hardware.stage, hardware.adc, hardware.laser, hardware.camera):
            await component.close()
    assert not transport.is_open
    assert chip.closed


def test_machine_config_still_checks_travel_limits() -> None:
    settings = load_settings(PI_CONFIG)
    tight = MotorConfig(min_steps=-1000, max_steps=1000)
    with pytest.raises(HardwareConfigError, match="motor travel"):
        build_hardware(
            settings.model_copy(update={"arduino": ArduinoConfig(a=tight, b=tight, c=tight)})
        )


def test_simulated_stage_with_a_real_adc_is_allowed(sim_settings: Settings) -> None:
    hardware = build_hardware(_with(sim_settings, adc="ads1115"))
    assert isinstance(hardware.stage, SimulationStage)
    assert isinstance(hardware.adc, ADS1115ADC)
    assert hardware.surface is not None
