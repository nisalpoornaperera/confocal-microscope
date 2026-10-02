"""SimulationLaser, ManualLaser (the real machine's laser), SimulationCamera, NullCamera."""

from __future__ import annotations

import numpy as np
import pytest

from confocal.errors import CameraError, HardwareNotConnectedError, LaserError
from confocal.hardware.simulation import (
    ManualLaser,
    NullCamera,
    SimulatedConfocalSurface,
    SimulationCamera,
    SimulationLaser,
)
from confocal.models.common import Position


async def test_simulation_laser_is_controllable() -> None:
    laser = SimulationLaser(wavelength_nm=650.0, power_mw=5.0)
    with pytest.raises(HardwareNotConnectedError):
        await laser.set_enabled(False)
    await laser.connect()
    assert laser.controllable
    assert await laser.is_enabled() is True
    await laser.set_enabled(False)
    assert laser.enabled is False
    status = await laser.status()
    assert status.controllable
    assert status.enabled is False
    assert status.wavelength_nm == 650.0
    await laser.set_enabled(True)
    await laser.close()
    assert laser.enabled is False  # closing leaves it in the safe state


async def test_manual_laser_cannot_be_switched_or_read() -> None:
    laser = ManualLaser(wavelength_nm=650.0, power_mw=5.0)
    await laser.connect()
    assert laser.connected
    assert laser.backend_name == "manual"
    assert not laser.controllable
    assert await laser.is_enabled() is None
    for state in (True, False):
        with pytest.raises(LaserError, match="manually"):
            await laser.set_enabled(state)
    status = await laser.status()
    assert not status.controllable
    assert status.enabled is None
    assert status.power_mw == 5.0
    await laser.close()
    assert not laser.connected


async def test_simulation_camera_images_the_sample() -> None:
    surface = SimulatedConfocalSurface(seed=1)
    position = Position(x_um=0.0, y_um=0.0, z_um=float(surface.height_um(0.0, 0.0)))
    camera = SimulationCamera(surface, lambda: position, width=32, height=24, pixel_um=1.0)
    with pytest.raises(HardwareNotConnectedError):
        await camera.capture()
    await camera.connect()
    frame = await camera.capture()
    assert frame.shape == (24, 32)
    assert frame.dtype == np.uint8
    assert int(frame.max()) > 200  # the centre is in focus
    status = await camera.status()
    assert status.available
    assert status.resolution == (32, 24)


def test_simulation_camera_validates_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        SimulationCamera(
            SimulatedConfocalSurface(), lambda: Position(x_um=0, y_um=0, z_um=0), width=0
        )


async def test_null_camera_is_unavailable() -> None:
    camera = NullCamera()
    await camera.connect()
    assert not camera.available
    with pytest.raises(CameraError):
        await camera.capture()
    status = await camera.status()
    assert not status.available
    assert status.connected
