"""Lifecycle (connect / close / release), identity and status reporting."""

from __future__ import annotations

import pytest
from tests.unit.microscope.conftest import (
    BrokenStatusStage,
    InMemoryCalibrationStore,
    RigFactory,
    ScriptedADC,
)

from confocal.errors import LaserError
from confocal.hardware.base import MicroscopeController
from confocal.microscope import StandardMicroscopeController
from confocal.models.calibration import CalibrationState
from confocal.models.common import Position
from confocal.models.hardware import LaserStatus, StageState


def test_implements_every_abstract_method() -> None:
    assert issubclass(StandardMicroscopeController, MicroscopeController)
    assert not StandardMicroscopeController.__abstractmethods__


async def test_connect_loads_the_latest_calibration(make_rig: RigFactory) -> None:
    store = InMemoryCalibrationStore(
        [CalibrationState(dark_v=0.01), CalibrationState(dark_v=0.02, reference_v=2.5)]
    )
    rig = await make_rig(store=store)
    assert rig.controller.calibration.version == 2
    assert rig.controller.calibration.dark_v == pytest.approx(0.02)
    assert rig.controller.calibration.can_normalize


async def test_connect_without_calibration_is_uncalibrated(make_rig: RigFactory) -> None:
    rig = await make_rig()
    assert rig.controller.calibration.version is None
    assert not rig.controller.calibration.has_dark


async def test_connect_reads_the_initial_position(make_rig: RigFactory) -> None:
    rig = await make_rig(connect=False)
    await rig.stage.connect()
    await rig.stage.move_to(Position(x_um=1.0, y_um=2.0, z_um=3.0))
    await rig.controller.connect()
    assert rig.controller.last_position == Position(x_um=1.0, y_um=2.0, z_um=3.0)
    assert rig.adc.connected
    assert rig.laser.connected


async def test_close_leaves_the_hardware_safe_and_is_idempotent(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.move_to(z_um=5.0)
    await rig.controller.close()
    assert rig.stage.motors_released is True
    assert await rig.laser.is_enabled() is False
    assert not rig.stage.connected
    assert not rig.adc.connected
    await rig.controller.close()


async def test_release_motors(make_rig: RigFactory) -> None:
    rig = await make_rig()
    await rig.controller.move_to(z_um=5.0)
    await rig.controller.release_motors()
    assert rig.stage.motors_released


async def test_hardware_info(make_rig: RigFactory) -> None:
    rig = await make_rig(manual_laser=True)
    info = rig.controller.hardware_info()
    assert info.controller == "standard"
    assert info.stage_backend == "simulation"
    assert info.adc_backend == "simulation"
    assert info.laser_backend == "manual"
    assert info.camera_backend == rig.controller._camera.backend_name
    assert info.details["laser_controllable"] == "false"


async def test_status_reports_every_component(make_rig: RigFactory) -> None:
    rig = await make_rig()
    status = await rig.controller.status()
    assert status.stage.connected
    assert status.stage.state is StageState.IDLE
    assert status.adc.connected
    assert status.laser.controllable
    assert status.laser.enabled is True
    assert not status.camera.available
    assert not status.estop_engaged


async def test_status_never_raises(make_rig: RigFactory) -> None:
    adc = ScriptedADC([0])
    adc.fail_status = True
    rig = await make_rig(stage_cls=BrokenStatusStage, adc=adc)

    async def broken_laser_status() -> LaserStatus:
        raise LaserError("GPIO busy")

    rig.laser.status = broken_laser_status  # type: ignore[method-assign]
    status = await rig.controller.status()
    assert status.stage.state is StageState.ERROR
    assert status.stage.last_error is not None
    assert "serial port vanished" in status.stage.last_error
    assert status.stage.position == Position(x_um=0.0, y_um=0.0, z_um=0.0)
    assert status.adc.last_error is not None
    assert "I2C NACK" in status.adc.last_error
    assert status.laser.last_error is not None
    assert "GPIO busy" in status.laser.last_error
    status.model_dump_json()  # JSON-safe
