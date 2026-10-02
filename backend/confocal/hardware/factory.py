"""Build the hardware components selected in ``Settings.hardware``.

Rules
-----
* **Never fall back to simulation.** A configuration naming a backend that is
  not available raises :class:`HardwareConfigError` listing *every* problem,
  so the server refuses to start instead of silently scanning a synthetic
  sample on a real machine.
* **Travel limits are checked against the motors at startup.** On the Delta
  Stage every Cartesian move drives all three legs, so the Cartesian limit
  box (``[limits]``) must lie inside every actuator's travel
  (``[arduino.a|b|c] min_steps / max_steps``), evaluated through the
  configured kinematics at the eight corners of the box - both in kinematic
  steps and in firmware steps (each motor's ``invert`` applied). Otherwise a
  move accepted by the Cartesian limit check could drive a leg past its safe
  range. The check runs for every backend, simulation included, so a
  configuration that is unsafe on the machine is never "tested green".
* The simulated stage, ADC and camera share one
  :class:`SimulatedConfocalSurface` (returned as :attr:`HardwareSet.surface`,
  the ground truth for tests); the ADC and camera see the stage's true,
  interpolated position. The simulated ADC models the configured ADS1115
  (``[ads1115] gain`` and ``data_rate_sps``). With a manual laser the
  simulation assumes the beam is on, like the always-on laser of the real
  machine.
* The real machine (``config/confocal.pi.toml``): ``stage = "arduino"`` ->
  :class:`ArduinoStage` on ``[arduino] port``, ``adc = "ads1115"`` ->
  :class:`ADS1115ADC` on ``[ads1115] i2c_bus`` (single-shot conversions),
  ``laser = "manual"`` -> :class:`ManualLaser`. Nothing is opened here: the
  serial port and the I2C bus are opened by ``connect()``. Tests and the
  hardware check inject ``transport_factory`` / ``i2c_bus_factory`` to run
  the real drivers against the firmware emulator and a fake bus.

Backends: ``stage = "simulation" | "arduino"``, ``adc = "simulation" |
"ads1115"``, ``laser = "simulation" | "manual"``, ``camera = "none" |
"simulation"``. Refused: ``stage = "openflexure"`` (no OpenFlexure client
exists yet), ``laser = "gpio"`` (no GPIO laser driver), and a simulated ADC
or camera without the simulated stage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from confocal.config import Settings
from confocal.errors import HardwareConfigError
from confocal.hardware.ads1115.adc import ADS1115ADC, BusFactory
from confocal.hardware.arduino.stage import ArduinoStage, TransportFactory
from confocal.hardware.arduino.steps import StepMapper, motor_configs
from confocal.hardware.base import ADC, Camera, Laser, Stage
from confocal.hardware.kinematics import StageKinematics
from confocal.hardware.simulation.adc import SimulationADC
from confocal.hardware.simulation.camera import NullCamera, SimulationCamera
from confocal.hardware.simulation.laser import ManualLaser, SimulationLaser
from confocal.hardware.simulation.stage import SimulationStage
from confocal.hardware.simulation.surface import SimulatedConfocalSurface
from confocal.models.common import Position


@dataclass(frozen=True, slots=True)
class HardwareSet:
    """The four hardware roles plus the simulated sample (``None`` on real hardware)."""

    stage: Stage
    adc: ADC
    laser: Laser
    camera: Camera
    surface: SimulatedConfocalSurface | None


def limit_problems(settings: Settings, kinematics: StageKinematics) -> list[str]:
    """Every way the Cartesian travel limits exceed the motor travel (empty if safe)."""
    problems = kinematics.config_limit_violations(settings.limits, settings.arduino)
    mapper = StepMapper(kinematics, motor_configs(settings.arduino))
    problems.extend(
        f"{problem} (firmware steps, invert applied)"
        for problem in mapper.limit_violations(settings.limits)
        if problem not in problems
    )
    return problems


def backend_problems(settings: Settings) -> list[str]:
    """Selected backends that are not implemented or cannot be combined."""
    selection = settings.hardware
    problems: list[str] = []
    if selection.stage == "openflexure":
        problems.append(
            "stage 'openflexure' needs an OpenFlexureClient, and none is implemented yet "
            "(use stage 'arduino' or 'simulation')"
        )
    if selection.laser == "gpio":
        problems.append(
            "laser 'gpio': no GPIO laser driver is implemented (use laser 'manual' for a "
            "manually switched laser)"
        )
    if selection.stage != "simulation":
        if selection.adc == "simulation":
            problems.append(
                f"adc 'simulation' requires stage 'simulation' (got {selection.stage!r}): "
                "the simulated detector must see the simulated stage position"
            )
        if selection.camera == "simulation":
            problems.append(
                f"camera 'simulation' requires stage 'simulation' (got {selection.stage!r})"
            )
    return problems


def build_hardware(
    settings: Settings,
    *,
    transport_factory: TransportFactory | None = None,
    i2c_bus_factory: BusFactory | None = None,
) -> HardwareSet:
    """Instantiate (but do not connect) the configured hardware.

    Args:
        settings: the application settings.
        transport_factory: replaces the serial link of an ``arduino`` stage
            (tests and the hardware check use the firmware emulator).
        i2c_bus_factory: replaces smbus2 for an ``ads1115`` ADC (tests).

    Raises:
        HardwareConfigError: the kinematics are invalid, the limits exceed the
            motor travel, a backend is not implemented, or the combination is
            invalid. The message lists every problem found.
    """
    try:
        kinematics = StageKinematics.from_config(settings.kinematics)
    except ValueError as exc:
        raise HardwareConfigError(f"invalid [kinematics]: {exc}") from exc
    problems = [
        f"travel limits exceed the motor travel: {problem}"
        for problem in limit_problems(settings, kinematics)
    ]
    problems.extend(backend_problems(settings))
    if problems:
        raise HardwareConfigError("invalid hardware configuration: " + "; ".join(problems))

    selection = settings.hardware
    laser = _build_laser(settings)
    surface: SimulatedConfocalSurface | None = None
    sim_stage: SimulationStage | None = None
    stage: Stage
    if selection.stage == "simulation":
        sim = settings.simulation
        surface = SimulatedConfocalSurface.from_config(sim)
        sim_stage = SimulationStage(
            settings.limits,
            motion=settings.motion,
            time_scale=sim.time_scale,
            kinematics=kinematics,
            fault_after_moves=sim.fault_after_moves,
        )
        stage = sim_stage
    else:  # "arduino" (backend_problems refused everything else)
        stage = ArduinoStage.from_settings(
            settings, kinematics=kinematics, transport_factory=transport_factory
        )

    adc: ADC
    camera: Camera = NullCamera()
    if sim_stage is not None and surface is not None:
        if selection.camera == "simulation":
            camera = SimulationCamera(surface, _position_of(sim_stage))
        if selection.adc == "simulation":
            sim = settings.simulation
            adc = SimulationADC(
                surface,
                position_source=_position_of(sim_stage),
                laser_source=_laser_source(laser),
                config=sim,
                gain=settings.ads1115.gain,
                data_rate_sps=settings.ads1115.data_rate_sps,
                seed=sim.seed,
            )
            return HardwareSet(stage=stage, adc=adc, laser=laser, camera=camera, surface=surface)
    adc = ADS1115ADC.from_config(settings.ads1115, bus_factory=i2c_bus_factory)
    return HardwareSet(stage=stage, adc=adc, laser=laser, camera=camera, surface=surface)


def _position_of(stage: SimulationStage) -> Callable[[], Position]:
    return lambda: stage.current_position


def _build_laser(settings: Settings) -> Laser:
    cfg = settings.laser
    if settings.hardware.laser == "manual":
        return ManualLaser(wavelength_nm=cfg.wavelength_nm, power_mw=cfg.power_mw)
    return SimulationLaser(wavelength_nm=cfg.wavelength_nm, power_mw=cfg.power_mw)


def _laser_source(laser: Laser) -> Callable[[], bool]:
    """What the simulated detector believes about the beam."""
    if isinstance(laser, SimulationLaser):
        return lambda: laser.enabled
    return lambda: True  # a manual laser is on whenever the machine is in use
