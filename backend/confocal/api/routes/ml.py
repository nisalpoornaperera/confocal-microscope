"""Advisory ML: analyse a finished scan, read its latest result, list deployed models."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Body

from confocal.api.deps import MLServiceDep, RepositoryDep, ScanManagerDep
from confocal.api.errors import ERROR_RESPONSES, not_found
from confocal.models.ml import MLAnalysisRequest, MLModelInfo, MLResult

router = APIRouter(prefix="/api/v1", tags=["ml"], responses=ERROR_RESPONSES)


@router.post(
    "/scans/{scan_id}/ml/analyse",
    response_model=MLResult,
    summary="Run an advisory ML analysis (never changes the physics results)",
)
async def analyse(
    scan_id: str,
    manager: ScanManagerDep,
    request: Annotated[MLAnalysisRequest | None, Body()] = None,
) -> MLResult:
    return await manager.analyse_ml(scan_id, request or MLAnalysisRequest())


@router.get(
    "/scans/{scan_id}/ml/results",
    response_model=MLResult,
    summary="Latest stored ML analysis of a scan",
)
async def get_latest_result(scan_id: str, repository: RepositoryDep) -> MLResult:
    result = await asyncio.to_thread(repository.get_latest_ml_result, scan_id)
    if result is None:
        raise not_found("MLResultNotFound", f"scan {scan_id} has no ML analysis")
    return result


@router.get("/ml/models", response_model=list[MLModelInfo], summary="Deployed ML models")
async def list_models(ml: MLServiceDep) -> list[MLModelInfo]:
    return await asyncio.to_thread(ml.list_models)
