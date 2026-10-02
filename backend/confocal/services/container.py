"""The service container: the only place concrete classes are wired together.

``ServiceContainer.create(settings)`` builds, without touching any device::

    build_hardware(settings)            -> Stage / ADC / Laser / Camera (+ simulated sample)
    SQLiteHDF5Repository                -> ScanRepository + CalibrationStore
    StandardMicroscopeController        -> safety model, calibration, measurement
    ScanEventBroker                     -> live WebSocket fan-out
    MLService                           -> advisory inference over deployed models
    ScanManager                         -> scan lifecycle (processing / surface / ML injected)

``start()`` creates the data directories, marks scans left active by a previous
run as interrupted, and connects the hardware. ``stop()`` shuts everything down
in the safe order (scan first, then motion, laser, devices, storage); every
step is best effort so one failure never prevents the next safety action.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Self

from confocal import __version__
from confocal.config import Settings
from confocal.hardware.factory import HardwareSet, build_hardware
from confocal.hardware.simulation.stage import SimulationStage
from confocal.hardware.simulation.surface import SimulatedConfocalSurface
from confocal.microscope import StandardMicroscopeController
from confocal.ml import MLService
from confocal.processing import analyse_profile, find_coarse_peak
from confocal.scanning import ScanEventBroker, ScanManager
from confocal.storage import SQLiteHDF5Repository
from confocal.surface import reconstruct_surface

log = logging.getLogger(__name__)


@dataclass(eq=False)
class ServiceContainer:
    """Every long-lived application object, created once per process (per app)."""

    settings: Settings
    hardware: HardwareSet
    repository: SQLiteHDF5Repository
    controller: StandardMicroscopeController
    broker: ScanEventBroker
    ml: MLService
    scan_manager: ScanManager
    created_monotonic: float = field(default_factory=time.monotonic)
    started_monotonic: float | None = None
    recovered_scan_ids: list[str] = field(default_factory=list)
    _stopped: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, settings: Settings) -> Self:
        """Build (but do not connect) the whole application.

        Raises:
            HardwareConfigError: the hardware configuration is invalid or names a
                backend that is not implemented yet (the server refuses to start).
        """
        hardware = build_hardware(settings)
        storage = settings.storage
        repository = SQLiteHDF5Repository(
            storage.database_path, storage.scans_dir, durable=not settings.is_simulation
        )
        try:
            simulated_stage = isinstance(hardware.stage, SimulationStage)
            controller = StandardMicroscopeController(
                hardware.stage,
                hardware.adc,
                hardware.laser,
                hardware.camera,
                limits=settings.limits,
                motion=settings.motion,
                calibration_store=repository,
                laser_warmup_s=settings.laser.warmup_s,
                time_scale=settings.simulation.time_scale if simulated_stage else 1.0,
            )
            broker = ScanEventBroker(settings.server.websocket_queue_size)
            ml = MLService(settings.models_dir, settings.ml.default_model)
            # ScanManager registers its own e-stop listener in __init__.
            scan_manager = ScanManager(
                controller,
                repository,
                broker=broker,
                find_coarse_peak=find_coarse_peak,
                analyse_profile=analyse_profile,
                reconstruct_surface=reconstruct_surface,
                ml_analyser=ml,
                motion=settings.motion,
                software_version=__version__,
                default_saturation_v=settings.processing.saturation_v,
            )
        except BaseException:
            repository.close()
            raise
        return cls(
            settings=settings,
            hardware=hardware,
            repository=repository,
            controller=controller,
            broker=broker,
            ml=ml,
            scan_manager=scan_manager,
        )

    # ------------------------------------------------------------------ properties
    @property
    def surface(self) -> SimulatedConfocalSurface | None:
        """The simulated sample (ground truth), or ``None`` on real hardware."""
        return self.hardware.surface

    @property
    def uptime_s(self) -> float:
        start = self.started_monotonic or self.created_monotonic
        return max(0.0, time.monotonic() - start)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Create data directories, recover interrupted scans, connect the hardware."""
        storage = self.settings.storage
        for directory in (storage.data_dir, storage.scans_dir, self.settings.models_dir):
            await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        self.recovered_scan_ids = await asyncio.to_thread(self.repository.recover_interrupted_scans)
        if self.recovered_scan_ids:
            log.warning(
                "marked %d scan(s) left active by the previous run as interrupted: %s",
                len(self.recovered_scan_ids),
                ", ".join(self.recovered_scan_ids),
            )
        await self.controller.connect()
        self.started_monotonic = time.monotonic()
        log.info(
            "confocal %s started (%s, data dir %s)",
            __version__,
            "simulation" if self.settings.is_simulation else "hardware",
            storage.data_dir,
        )

    async def stop(self) -> None:
        """Shut down in the safe order. Never raises; idempotent."""
        if self._stopped:
            return
        self._stopped = True
        await _best_effort("shut down the scan manager", self.scan_manager.shutdown())
        await _best_effort("stop the stage", self.hardware.stage.stop())
        laser = self.hardware.laser
        try:
            switch_off = laser.controllable and laser.connected
        except Exception:
            log.exception("could not query the laser")
            switch_off = False
        if switch_off:
            await _best_effort("switch the laser off", laser.set_enabled(False))
        await _best_effort("close the microscope controller", self.controller.close())
        await _best_effort("close the repository", asyncio.to_thread(self.repository.close))
        log.info("confocal stopped")


async def _best_effort(action: str, operation: Awaitable[object]) -> None:
    try:
        await operation
    except Exception:
        log.exception("shutdown: could not %s", action)
