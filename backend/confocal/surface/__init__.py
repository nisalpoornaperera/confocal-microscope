"""Surface reconstruction from the physics results (X, Y, surface Z, confidence).

``reconstruct_surface`` implements the ``SurfaceReconstructor`` protocol of
``confocal.scanning.protocols``.
"""

from confocal.surface.filtering import (
    ClassifiedPoints,
    classify_points,
    global_mad_outliers,
    local_mad_outliers,
)
from confocal.surface.reconstruction import reconstruct_surface
from confocal.surface.statistics import HeightParameters, fit_plane, height_parameters

__all__ = [
    "ClassifiedPoints",
    "HeightParameters",
    "classify_points",
    "fit_plane",
    "global_mad_outliers",
    "height_parameters",
    "local_mad_outliers",
    "reconstruct_surface",
]
