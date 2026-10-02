"""Stage adapter for an OpenFlexure server (Phase 11; no HTTP client yet).

See :mod:`.stage` for how this lets the scanner run on top of, and later as an
extension of, the OpenFlexure software.
"""

from confocal.hardware.openflexure.client import OpenFlexureClient
from confocal.hardware.openflexure.stage import OpenFlexureStage, StepConverter

__all__ = ["OpenFlexureClient", "OpenFlexureStage", "StepConverter"]
