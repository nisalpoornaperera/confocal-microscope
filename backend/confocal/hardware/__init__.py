"""Hardware layer: interfaces, kinematics, simulation, device-protocol code, factory.

Everything above this package works in micrometres through the interfaces in
:mod:`confocal.hardware.base`; motor steps exist only in
:mod:`confocal.hardware.kinematics`, :mod:`confocal.hardware.arduino` and the
OpenFlexure adapter. :func:`build_hardware` turns ``Settings.hardware`` into
concrete components and refuses (``HardwareConfigError``) any configuration it
cannot honour - it never falls back to simulation.
"""

from confocal.hardware.base import (
    ADC,
    Camera,
    EstopListener,
    HardwareComponent,
    Laser,
    MicroscopeController,
    Stage,
)
from confocal.hardware.factory import HardwareSet, build_hardware
from confocal.hardware.kinematics import MOTORS, Motor, MotorSteps, StageKinematics

__all__ = [
    "ADC",
    "MOTORS",
    "Camera",
    "EstopListener",
    "HardwareComponent",
    "HardwareSet",
    "Laser",
    "MicroscopeController",
    "Motor",
    "MotorSteps",
    "Stage",
    "StageKinematics",
    "build_hardware",
]
