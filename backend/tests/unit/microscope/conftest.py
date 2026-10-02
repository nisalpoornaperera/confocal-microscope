"""Fixtures for the microscope-layer tests.

The controller runs on the real simulation backends (SimulationStage,
SimulationADC over a flat simulated mirror at Z = FOCUS_Z_UM, SimulationLaser /
ManualLaser, NullCamera) plus small purpose-built fakes for the failure paths:
a stage that lies about its position, one that never finishes, a broken one, and
an ADC that returns scripted codes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from confocal.config import MotionConfig, SimulatedSurfaceConfig, SimulationConfig
from confocal.errors import CommunicationError, HardwareNotConnectedError
from confocal.hardware.ads1115.conversion import counts_to_volts
from confocal.hardware.base import ADC, Laser
from confocal.hardware.simulation import (
    ManualLaser,
    NullCamera,
    SimulatedConfocalSurface,
    SimulationADC,
    SimulationLaser,
    SimulationStage,
)
from confocal.microscope import StandardMicroscopeController
from confocal.models.calibration import CalibrationState
from confocal.models.common import AxisLimits, Position, StageLimits
from confocal.models.hardware import AdcGain, ADCStatus, StageStatus
from confocal.models.measurement import AdcSamples

FOCUS_Z_UM = 5.0
LIMITS = StageLimits(
    x=AxisLimits(min_um=-1000.0, max_um=1000.0),
    y=AxisLimits(min_um=-1000.0, max_um=1000.0),
    z=AxisLimits(min_um=-200.0, max_um=200.0),
)


class InMemoryCalibrationStore:
    """CalibrationStore fake: monotonic versions, newest-first history."""

    def __init__(self, initial: Sequence[CalibrationState] = ()) -> None:
        self.saved: list[CalibrationState] = []
        for state in initial:
            self.save_calibration(state)

    def save_calibration(self, state: CalibrationState) -> CalibrationState:
        saved = state.model_copy(update={"version": len(self.saved) + 1})
        self.saved.append(saved)
        return saved

    def latest_calibration(self) -> CalibrationState | None:
        return self.saved[-1] if self.saved else None

    def get_calibration(self, version: int) -> CalibrationState | None:
        return next((s for s in self.saved if s.version == version), None)

    def list_calibrations(self, limit: int = 50) -> list[CalibrationState]:
        return list(reversed(self.saved))[:limit]


class RecordingStage(SimulationStage):
    """SimulationStage that records every ``move_to`` target and ``stop`` call."""

    def __init__(
        self,
        limits: StageLimits,
        *,
        motion: MotionConfig,
        time_scale: float,
        fault_after_moves: int | None = None,
    ) -> None:
        super().__init__(
            limits, motion=motion, time_scale=time_scale, fault_after_moves=fault_after_moves
        )
        self.targets: list[Position] = []
        self.stop_calls = 0

    async def move_to(self, target: Position) -> Position:
        self.targets.append(target)
        return await super().move_to(target)

    async def stop(self) -> None:
        self.stop_calls += 1
        await super().stop()


class CountingStage(RecordingStage):
    """Plain recording stage (alias used where only the call log matters)."""


class LyingStage(RecordingStage):
    """Moves correctly but reports a position offset in X (lost steps / bad encoder)."""

    error_um = 5.0

    async def move_to(self, target: Position) -> Position:
        reported = await super().move_to(target)
        return reported.offset(dx_um=self.error_um)


class HangingStage(RecordingStage):
    """``move_to`` never completes and ignores ``stop`` (a stalled firmware)."""

    async def move_to(self, target: Position) -> Position:
        self.targets.append(target)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class FaultyStage(RecordingStage):
    """The second move fails with an injected MotionError."""

    def __init__(self, limits: StageLimits, *, motion: MotionConfig, time_scale: float) -> None:
        super().__init__(limits, motion=motion, time_scale=time_scale, fault_after_moves=1)


class BrokenStatusStage(SimulationStage):
    """Status queries fail (e.g. the serial link dropped)."""

    async def status(self) -> StageStatus:
        raise CommunicationError("serial port vanished")


class ScriptedADC(ADC):
    """Returns pre-programmed raw codes; volts follow from the current gain."""

    backend_name = "scripted"

    def __init__(self, counts: Sequence[int], *, gain: AdcGain = AdcGain.G1) -> None:
        self.counts = list(counts)
        self._gain = gain
        self._connected = False
        self.fail_status = False

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def gain(self) -> AdcGain:
        return self._gain

    async def set_gain(self, gain: AdcGain) -> None:
        self._gain = gain

    @property
    def data_rate_sps(self) -> int:
        return 860

    async def read_samples(self, n: int) -> AdcSamples:
        if not self._connected:
            raise HardwareNotConnectedError("scripted ADC not connected")
        codes = np.resize(np.asarray(self.counts, dtype=np.int32), n)
        return AdcSamples(
            counts=codes,
            volts=counts_to_volts(codes, self._gain),
            timestamps=np.arange(n, dtype=np.float64),
            gain=self._gain,
            data_rate_sps=860,
        )

    async def status(self) -> ADCStatus:
        if self.fail_status:
            raise CommunicationError("I2C NACK")
        return ADCStatus(
            backend=self.backend_name,
            connected=self._connected,
            gain=self._gain,
            full_scale_v=self._gain.full_scale_v,
            data_rate_sps=860,
        )


@dataclass
class Rig:
    controller: StandardMicroscopeController
    stage: SimulationStage
    adc: ADC
    laser: Laser
    store: InMemoryCalibrationStore


RigFactory = Callable[..., Awaitable[Rig]]


def sim_config(**overrides: object) -> SimulationConfig:
    """Flat, uniform, noise-light mirror at FOCUS_Z_UM; instantaneous conversions."""
    surface = SimulatedSurfaceConfig(
        kind="plane",
        base_z_um=FOCUS_Z_UM,
        tilt_x=0.0,
        tilt_y=0.0,
        reflectivity=0.9,
        low_reflectivity_fraction=0.0,
        spurious_peak_probability=0.0,
    )
    # Signal levels are pinned (5 V OPT101 model) so these tests do not depend on defaults.
    values: dict[str, object] = {
        "time_scale": 0.0,
        "seed": 7,
        "surface": surface,
        "peak_voltage_v": 2.8,
        "opt101_saturation_v": 3.7,
    }
    values.update(overrides)
    return SimulationConfig.model_validate(values)


@pytest.fixture
def make_rig() -> RigFactory:
    """Async factory for a connected controller over simulated (or fake) hardware."""

    async def build(
        *,
        stage_cls: type[SimulationStage] = SimulationStage,
        stage_time_scale: float = 0.0,
        motion: MotionConfig | None = None,
        manual_laser: bool = False,
        adc: ADC | None = None,
        gain: AdcGain = AdcGain.G1,
        config: SimulationConfig | None = None,
        store: InMemoryCalibrationStore | None = None,
        laser_warmup_s: float = 0.0,
        connect: bool = True,
    ) -> Rig:
        motion = motion or MotionConfig()
        config = config or sim_config()
        stage = stage_cls(LIMITS, motion=motion, time_scale=stage_time_scale)
        laser: Laser = ManualLaser() if manual_laser else SimulationLaser(enabled=True)
        sim_laser = laser if isinstance(laser, SimulationLaser) else None
        if adc is None:
            adc = SimulationADC(
                SimulatedConfocalSurface.from_config(config),
                position_source=lambda: stage.current_position,
                laser_source=lambda: True if sim_laser is None else sim_laser.enabled,
                config=config,
                gain=gain,
                data_rate_sps=860,
                seed=config.seed,
            )
        store = store if store is not None else InMemoryCalibrationStore()
        controller = StandardMicroscopeController(
            stage,
            adc,
            laser,
            NullCamera(),
            limits=LIMITS,
            motion=motion,
            calibration_store=store,
            laser_warmup_s=laser_warmup_s,
            time_scale=0.0,
        )
        if connect:
            await controller.connect()
        return Rig(controller=controller, stage=stage, adc=adc, laser=laser, store=store)

    return build
