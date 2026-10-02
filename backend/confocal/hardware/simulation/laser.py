"""Laser backends: a switchable simulated laser and the manually switched real one.

The real machine's 650 nm module is switched by hand (``laser = "manual"``),
so software can neither switch it nor know its state. :class:`ManualLaser`
reports exactly that: ``controllable = False`` and ``enabled = None``. The
consequences are handled above this layer: dark calibration asks the operator
to block the beam and confirm, and the emergency stop records that the laser
must be switched off by hand.
"""

from __future__ import annotations

import asyncio

from confocal.errors import HardwareNotConnectedError, LaserError
from confocal.hardware.base import Laser
from confocal.models.hardware import LaserStatus


class SimulationLaser(Laser):
    """Software-switchable laser for the simulation.

    It starts *on* by default, like the always-on manual laser of the real
    machine, so a simulated scan sees light without any extra step.
    :meth:`close` switches it off (safe state).
    """

    backend_name = "simulation"

    def __init__(
        self, *, wavelength_nm: float = 650.0, power_mw: float = 5.0, enabled: bool = True
    ) -> None:
        self._wavelength_nm = float(wavelength_nm)
        self._power_mw = float(power_mw)
        self._enabled = bool(enabled)
        self._connected = False

    @property
    def enabled(self) -> bool:
        """Synchronous state accessor (no I/O); the simulated ADC's ``laser_source``."""
        return self._enabled

    async def connect(self) -> None:
        self._connected = True
        await asyncio.sleep(0)

    async def close(self) -> None:
        self._enabled = False
        self._connected = False
        await asyncio.sleep(0)

    @property
    def connected(self) -> bool:
        return self._connected

    def version(self) -> str | None:
        return "simulation"

    @property
    def controllable(self) -> bool:
        return True

    async def set_enabled(self, enabled: bool) -> None:
        if not self._connected:
            raise HardwareNotConnectedError("simulation laser is not connected")
        self._enabled = bool(enabled)
        await asyncio.sleep(0)

    async def is_enabled(self) -> bool | None:
        return self._enabled

    async def status(self) -> LaserStatus:
        return LaserStatus(
            backend=self.backend_name,
            connected=self._connected,
            controllable=True,
            enabled=self._enabled,
            wavelength_nm=self._wavelength_nm,
            power_mw=self._power_mw,
        )


class ManualLaser(Laser):
    """A laser switched by hand: software can neither switch it nor read its state."""

    backend_name = "manual"

    def __init__(self, *, wavelength_nm: float = 650.0, power_mw: float = 5.0) -> None:
        self._wavelength_nm = float(wavelength_nm)
        self._power_mw = float(power_mw)
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
    def controllable(self) -> bool:
        return False

    async def set_enabled(self, enabled: bool) -> None:
        state = "on" if enabled else "off"
        raise LaserError(f"the laser is switched manually: switch it {state} with its own switch")

    async def is_enabled(self) -> bool | None:
        return None

    async def status(self) -> LaserStatus:
        return LaserStatus(
            backend=self.backend_name,
            connected=self._connected,
            controllable=False,
            enabled=None,
            wavelength_nm=self._wavelength_nm,
            power_mw=self._power_mw,
        )
