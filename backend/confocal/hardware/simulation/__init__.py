"""Simulated hardware: the whole pipeline runs without a physical instrument.

One :class:`SimulatedConfocalSurface` is shared by the stage-position-driven
ADC (and the optional camera), so a simulated Z sweep produces the confocal
I(Z) peak at the true surface height. :class:`ManualLaser` is not a simulation:
it is the backend the real machine uses for its manually switched laser.
"""

from confocal.hardware.simulation.adc import SimulationADC
from confocal.hardware.simulation.camera import NullCamera, SimulationCamera
from confocal.hardware.simulation.laser import ManualLaser, SimulationLaser
from confocal.hardware.simulation.stage import SimulationStage
from confocal.hardware.simulation.surface import (
    LOW_REFLECTIVITY,
    SPURIOUS_AMPLITUDE,
    PsfModel,
    SimulatedConfocalSurface,
    SurfaceGroundTruth,
)

__all__ = [
    "LOW_REFLECTIVITY",
    "SPURIOUS_AMPLITUDE",
    "ManualLaser",
    "NullCamera",
    "PsfModel",
    "SimulatedConfocalSurface",
    "SimulationADC",
    "SimulationCamera",
    "SimulationLaser",
    "SimulationStage",
    "SurfaceGroundTruth",
]
