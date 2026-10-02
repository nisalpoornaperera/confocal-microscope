"""Camera backends: a synthetic reflectance camera and the "no camera" placeholder.

The scanner images with the photodiode; a camera is optional (the OpenFlexure
Pi camera may be added later for navigation). :class:`NullCamera` is what a
machine without one uses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from confocal.errors import CameraError, HardwareNotConnectedError
from confocal.hardware.base import Camera
from confocal.hardware.simulation.surface import SimulatedConfocalSurface
from confocal.models.common import Position
from confocal.models.hardware import CameraStatus


class SimulationCamera(Camera):
    """Small synthetic grey-scale image of the sample around the current position.

    Pixel brightness = reflectivity x axial response at the current Z, so the
    image shows the low-reflectivity patches and which parts are in focus.
    """

    backend_name = "simulation"

    def __init__(
        self,
        surface: SimulatedConfocalSurface,
        position_source: Callable[[], Position],
        *,
        width: int = 64,
        height: int = 48,
        pixel_um: float = 2.0,
    ) -> None:
        if width < 1 or height < 1 or not pixel_um > 0:
            raise ValueError("image size and pixel size must be positive")
        self._surface = surface
        self._position_source = position_source
        self._width = int(width)
        self._height = int(height)
        self._pixel_um = float(pixel_um)
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        await asyncio.sleep(0)

    async def close(self) -> None:
        self._connected = False
        await asyncio.sleep(0)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def available(self) -> bool:
        return True

    async def capture(self) -> NDArray[np.uint8]:
        if not self._connected:
            raise HardwareNotConnectedError("simulation camera is not connected")
        await asyncio.sleep(0)
        centre = self._position_source()
        xs = centre.x_um + (np.arange(self._width) - (self._width - 1) / 2.0) * self._pixel_um
        ys = centre.y_um + (np.arange(self._height) - (self._height - 1) / 2.0) * self._pixel_um
        grid_x, grid_y = np.meshgrid(xs, ys)
        focus = self._surface.axial_response(centre.z_um - self._surface.height_um(grid_x, grid_y))
        brightness = self._surface.reflectivity(grid_x, grid_y) * focus
        return np.asarray(np.clip(np.rint(255.0 * brightness), 0, 255), dtype=np.uint8)

    async def status(self) -> CameraStatus:
        return CameraStatus(
            backend=self.backend_name,
            connected=self._connected,
            available=True,
            resolution=(self._width, self._height),
        )


class NullCamera(Camera):
    """No camera fitted: ``available`` is False and :meth:`capture` raises."""

    backend_name = "none"

    def __init__(self) -> None:
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        await asyncio.sleep(0)

    async def close(self) -> None:
        self._connected = False
        await asyncio.sleep(0)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def available(self) -> bool:
        return False

    async def capture(self) -> NDArray[np.uint8]:
        raise CameraError("no camera is configured (hardware.camera = 'none')")

    async def status(self) -> CameraStatus:
        return CameraStatus(backend=self.backend_name, connected=self._connected, available=False)
