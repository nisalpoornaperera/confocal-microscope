"""Dark / reference calibration procedures and versioned persistence."""

from __future__ import annotations

import pytest
from tests.unit.microscope.conftest import FOCUS_Z_UM, RigFactory, sim_config

from confocal.errors import ADCError, CalibrationError, EmergencyStopActiveError
from confocal.hardware.simulation import SimulationLaser
from confocal.microscope.calibration import search_positions
from confocal.models.calibration import DarkCalibrationRequest, ReferenceCalibrationRequest
from confocal.models.common import AxisLimits
from confocal.models.hardware import AdcGain
from confocal.models.measurement import AdcSamples

DARK_V = 0.0075  # SimulationConfig.dark_voltage_v


async def test_dark_switches_a_controllable_laser_off_and_restores_it(
    make_rig: RigFactory,
) -> None:
    rig = await make_rig()
    laser = rig.laser
    assert isinstance(laser, SimulationLaser)
    await rig.controller.move_to(z_um=FOCUS_Z_UM)  # bright: ~2.6 V with the laser on
    state = await rig.controller.calibrate_dark(DarkCalibrationRequest(n_samples=64))
    assert state.dark_v == pytest.approx(DARK_V, abs=1e-3)
    assert state.dark_std_v is not None
    assert state.dark_n_samples == 64
    assert state.dark_gain is AdcGain.G1
    assert state.updated_field == "dark"
    assert state.version == 1
    assert laser.enabled  # restored
    assert rig.controller.calibration == state


async def test_dark_restores_the_laser_when_the_measurement_fails(make_rig: RigFactory) -> None:
    rig = await make_rig(config=sim_config(adc_fault_after_reads=0))
    assert isinstance(rig.laser, SimulationLaser)
    with pytest.raises(ADCError):
        await rig.controller.calibrate_dark(DarkCalibrationRequest())
    assert rig.laser.enabled
    assert rig.store.saved == []
    assert rig.controller.calibration.version is None


async def test_dark_leaves_a_laser_that_was_off_switched_off(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.set_laser(False)
    await rig.controller.calibrate_dark(DarkCalibrationRequest())
    assert isinstance(rig.laser, SimulationLaser)
    assert not rig.laser.enabled


async def test_dark_refuses_a_saturated_reading(make_rig: RigFactory) -> None:
    rig = await make_rig(manual_laser=True, gain=AdcGain.G16)
    await rig.controller.move_to(z_um=FOCUS_Z_UM)  # manual laser: the light is on
    with pytest.raises(CalibrationError, match="beam really blocked"):
        await rig.controller.calibrate_dark(DarkCalibrationRequest(beam_blocked_confirmed=True))


async def test_manual_laser_dark_needs_confirmation(make_rig: RigFactory) -> None:
    rig = await make_rig(manual_laser=True, config=sim_config(background_v=0.0))
    await rig.controller.move_to(z_um=-150.0)  # far from focus: ~dark level
    with pytest.raises(CalibrationError, match="block the beam"):
        await rig.controller.calibrate_dark(DarkCalibrationRequest())
    state = await rig.controller.calibrate_dark(DarkCalibrationRequest(beam_blocked_confirmed=True))
    assert state.dark_v == pytest.approx(DARK_V, abs=1e-3)
    assert state.version == 1


async def test_reference_z_search_finds_the_simulated_focus(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.calibrate_dark(DarkCalibrationRequest())
    request = ReferenceCalibrationRequest(
        z_search=True, z_search_range_um=40.0, z_search_step_um=1.0
    )
    state = await rig.controller.calibrate_reference(request)
    assert state.reference_position is not None
    assert state.reference_position.z_um == pytest.approx(FOCUS_Z_UM, abs=1.0)
    assert rig.controller.last_position == state.reference_position
    # 0.9 reflectivity x 2.8 V peak + 0.05 V background + dark
    assert state.reference_v == pytest.approx(0.9 * 2.8 + 0.05 + DARK_V, rel=0.05)
    assert state.reference_gain is AdcGain.G1
    assert state.updated_field == "reference"
    assert state.dark_v is not None  # carried over from version 1
    assert state.version == 2


async def test_reference_switches_a_controllable_laser_on(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.set_laser(False)
    await rig.controller.move_to(z_um=FOCUS_Z_UM)
    state = await rig.controller.calibrate_reference(ReferenceCalibrationRequest())
    assert isinstance(rig.laser, SimulationLaser)
    assert rig.laser.enabled
    assert state.reference_v is not None
    assert state.reference_v > 2.0


async def test_reference_search_with_the_maximum_at_the_edge_is_refused(
    make_rig: RigFactory,
) -> None:
    rig = await make_rig()
    await rig.controller.move_to(z_um=-10.0)  # sweep -20..0 um, focus at +5 um
    request = ReferenceCalibrationRequest(z_search=True, z_search_range_um=20.0)
    with pytest.raises(CalibrationError, match="edge of the Z search"):
        await rig.controller.calibrate_reference(request)
    assert rig.store.saved == []


async def test_reference_search_without_a_peak_is_refused(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.move_to(z_um=-100.0)  # only background in the whole sweep
    request = ReferenceCalibrationRequest(z_search=True, z_search_range_um=20.0)
    with pytest.raises(CalibrationError, match="no focus peak"):
        await rig.controller.calibrate_reference(request)


async def test_saturated_reference_is_refused(make_rig: RigFactory) -> None:
    rig = await make_rig(gain=AdcGain.G2)  # +/-2.048 V < ~2.6 V in focus
    await rig.controller.move_to(z_um=FOCUS_Z_UM)
    with pytest.raises(CalibrationError, match="lower gain or add an ND filter"):
        await rig.controller.calibrate_reference(ReferenceCalibrationRequest())
    assert rig.store.saved == []


async def test_reference_not_above_dark_is_refused(make_rig: RigFactory) -> None:
    rig = await make_rig(config=sim_config(background_v=0.0))
    await rig.controller.move_to(z_um=-150.0)  # no reflector in focus
    await rig.controller.calibrate_dark(DarkCalibrationRequest())
    with pytest.raises(CalibrationError, match="not clearly above the dark"):
        await rig.controller.calibrate_reference(ReferenceCalibrationRequest())
    assert len(rig.store.saved) == 1


async def test_calibration_versions_increment_and_carry_values(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.move_to(z_um=FOCUS_Z_UM)
    first = await rig.controller.calibrate_dark(DarkCalibrationRequest(notes="first"))
    second = await rig.controller.calibrate_reference(ReferenceCalibrationRequest())
    third = await rig.controller.calibrate_dark(DarkCalibrationRequest())
    assert [first.version, second.version, third.version] == [1, 2, 3]
    assert [s.updated_field for s in rig.store.saved] == ["dark", "reference", "dark"]
    assert second.dark_v == first.dark_v
    assert third.reference_v == second.reference_v
    assert first.notes == "first"
    assert third.notes is None
    assert rig.controller.calibration == third
    assert third.can_normalize


async def test_estop_during_dark_leaves_the_laser_off(make_rig: RigFactory) -> None:
    rig = await make_rig()
    laser = rig.laser
    assert isinstance(laser, SimulationLaser)
    original = rig.adc.read_samples

    async def read_then_estop(n: int) -> AdcSamples:
        samples = await original(n)
        await rig.controller.emergency_stop("tripped during dark")
        return samples

    rig.adc.read_samples = read_then_estop  # type: ignore[method-assign]
    await rig.controller.calibrate_dark(DarkCalibrationRequest())
    assert not laser.enabled
    with pytest.raises(EmergencyStopActiveError):
        await rig.controller.calibrate_dark(DarkCalibrationRequest())


def test_search_positions_are_increasing_and_clipped() -> None:
    limits = AxisLimits(min_um=-10.0, max_um=10.0)
    zs = search_positions(5.0, 20.0, 2.0, limits)
    assert zs.tolist() == [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0]
    assert search_positions(0.0, 4.0, 1.0, limits).tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]
