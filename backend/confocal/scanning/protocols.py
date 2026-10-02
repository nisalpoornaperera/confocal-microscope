"""Callable contracts injected into the scan engine.

The ScanManager / executor never import processing, surface or ML modules
directly; the service container injects implementations matching these
protocols. This keeps scan orchestration testable with fakes and lets any
stage of the physics pipeline be replaced without touching scan logic.

Real implementations:
    ProfileAnalyser      -> confocal.processing.profile.analyse_profile
    CoarsePeakFinder     -> confocal.processing.profile.find_coarse_peak
    SurfaceReconstructor -> confocal.surface.reconstruction.reconstruct_surface
    MLAnalyser           -> confocal.ml.inference.MLService
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from confocal.models.ml import MLAnalysisRequest, MLResult
from confocal.models.processing import CoarsePeak, ProcessingConfig, ProfileProcessingResult
from confocal.models.scan import ScanPoint
from confocal.models.surface import ReconstructionRequest, SurfaceResult


class CoarsePeakFinder(Protocol):
    def __call__(
        self,
        z_um: NDArray[np.float64],
        voltage_v: NDArray[np.float64],
        *,
        dark_v: float | None,
        config: ProcessingConfig,
    ) -> CoarsePeak:
        """Locate the approximate peak of a coarse sweep (per-Z aggregated voltage)."""
        ...


class ProfileAnalyser(Protocol):
    def __call__(
        self,
        z_um: NDArray[np.float64],
        voltage_v: NDArray[np.float64],
        *,
        dark_v: float | None,
        reference_v: float | None,
        config: ProcessingConfig,
    ) -> ProfileProcessingResult:
        """Full physics pipeline on one fine sweep: dark subtraction, normalization,
        filtering, peak detection, parabolic + Gaussian fits, SNR, width, prominence,
        fit error and confidence. Output arrays keep the input order."""
        ...


class SurfaceReconstructor(Protocol):
    def __call__(
        self,
        scan_id: str,
        points: Sequence[ScanPoint],
        request: ReconstructionRequest,
        *,
        xy_step_um: float,
    ) -> SurfaceResult:
        """Raises ReconstructionError when there are too few usable points."""
        ...


@runtime_checkable
class MLAnalyser(Protocol):
    def has_model(self, name: str | None = None) -> bool: ...

    def analyse(
        self, scan_id: str, points: Sequence[ScanPoint], request: MLAnalysisRequest
    ) -> MLResult:
        """Raises ModelNotAvailableError when no model is deployed."""
        ...
