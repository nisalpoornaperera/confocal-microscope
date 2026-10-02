"""Post-acquisition work shared by the scan task and the on-demand API endpoints.

These functions are synchronous on purpose: they load points from the
repository, run CPU-heavy reconstruction / inference and write the result.
Callers on the event loop run them through ``asyncio.to_thread`` so neither the
running scan nor HTTP requests are blocked.
"""

from __future__ import annotations

from confocal.errors import ModelNotAvailableError
from confocal.models.ml import MLAnalysisRequest, MLResult
from confocal.models.surface import ReconstructionRequest, SurfaceResult
from confocal.scanning.protocols import MLAnalyser, SurfaceReconstructor
from confocal.storage.interfaces import ScanRepository


def reconstruct_and_save(
    repository: ScanRepository,
    reconstruct_surface: SurfaceReconstructor,
    scan_id: str,
    request: ReconstructionRequest,
    *,
    xy_step_um: float,
) -> SurfaceResult:
    """Reconstruct the surface from every stored point and persist it.

    Raises:
        ReconstructionError: too few usable points (from the reconstructor).
    """
    points = repository.get_points(scan_id)
    surface = reconstruct_surface(scan_id, points, request, xy_step_um=xy_step_um)
    return repository.save_surface(surface)


def analyse_and_save(
    repository: ScanRepository,
    analyser: MLAnalyser,
    scan_id: str,
    request: MLAnalysisRequest,
) -> MLResult:
    """Run the advisory ML analysis on every stored point and persist the result.

    Raises:
        ModelNotAvailableError: the requested (or default) model is not deployed.
    """
    if not analyser.has_model(request.model_name):
        name = request.model_name or "default"
        raise ModelNotAvailableError(f"ML model '{name}' is not deployed")
    points = repository.get_points(scan_id)
    return repository.save_ml_result(analyser.analyse(scan_id, points, request))
