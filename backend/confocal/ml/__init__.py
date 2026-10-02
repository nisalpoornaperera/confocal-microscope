"""Optional, advisory ML post-processing. Training is offline; the Pi only runs inference.

``MLService`` implements the ``MLAnalyser`` protocol of
``confocal.scanning.protocols``. ``confocal.ml.training`` is deliberately not
exported: it is an offline tool and is never imported by the server.
"""

from confocal.ml.features import FEATURE_NAMES, build_feature_matrix
from confocal.ml.inference import MLService
from confocal.ml.registry import (
    METADATA_FILE,
    MODEL_FILE,
    LoadedModel,
    ModelRegistry,
    compatibility_problem,
)

__all__ = [
    "FEATURE_NAMES",
    "METADATA_FILE",
    "MODEL_FILE",
    "LoadedModel",
    "MLService",
    "ModelRegistry",
    "build_feature_matrix",
    "compatibility_problem",
]
