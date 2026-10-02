"""``/api/v1/scans``: create, estimate, list, control and inspect scans.

``POST /api/v1/scans`` returns 201 as soon as the scan is persisted and its
background task started; progress is observed through ``GET /scans/{id}`` or
the WebSocket ``/ws/scans/{id}``. Every repository read runs in a worker
thread, so these endpoints stay responsive while a scan is running.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Body, Query, status

from confocal.api.deps import RepositoryDep, ScanManagerDep
from confocal.api.errors import ERROR_RESPONSES, not_found
from confocal.models.measurement import ProfileRecord
from confocal.models.scan import ScanConfig, ScanEstimate, ScanPoint, ScanSummary
from confocal.models.surface import ReconstructionRequest, SurfaceResult

router = APIRouter(prefix="/api/v1/scans", tags=["scans"], responses=ERROR_RESPONSES)


@router.post(
    "",
    response_model=ScanSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Validate, persist and start a scan (returns immediately)",
)
async def create_scan(config: ScanConfig, manager: ScanManagerDep) -> ScanSummary:
    return await manager.create_scan(config)


@router.post(
    "/estimate",
    response_model=ScanEstimate,
    summary="Points, measurements, duration, data size and limit check of a scan",
)
async def estimate_scan(config: ScanConfig, manager: ScanManagerDep) -> ScanEstimate:
    return await manager.estimate(config)


@router.get("", response_model=list[ScanSummary], summary="Scans, newest first")
async def list_scans(
    manager: ScanManagerDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ScanSummary]:
    return await manager.list_scans(limit=limit, offset=offset)


@router.get("/{scan_id}", response_model=ScanSummary, summary="One scan (live state if active)")
async def get_scan(scan_id: str, manager: ScanManagerDep) -> ScanSummary:
    return await manager.get_scan(scan_id)


@router.post("/{scan_id}/pause", response_model=ScanSummary, summary="Pause after this point")
async def pause_scan(scan_id: str, manager: ScanManagerDep) -> ScanSummary:
    return await manager.pause(scan_id)


@router.post("/{scan_id}/resume", response_model=ScanSummary, summary="Resume a paused scan")
async def resume_scan(scan_id: str, manager: ScanManagerDep) -> ScanSummary:
    return await manager.resume(scan_id)


@router.post(
    "/{scan_id}/cancel",
    response_model=ScanSummary,
    summary="Cancel at the next Z step (measured data is kept)",
)
async def cancel_scan(scan_id: str, manager: ScanManagerDep) -> ScanSummary:
    return await manager.cancel(scan_id)


@router.get(
    "/{scan_id}/points",
    response_model=list[ScanPoint],
    summary="Per-point results ordered by point_id (since: ids strictly greater)",
)
async def get_points(
    scan_id: str,
    repository: RepositoryDep,
    since: Annotated[int | None, Query(ge=-1)] = None,
    limit: Annotated[int | None, Query(ge=1, le=250_000)] = None,
) -> list[ScanPoint]:
    return await asyncio.to_thread(
        repository.get_points, scan_id, since_point_id=since, limit=limit
    )


@router.get(
    "/{scan_id}/profile/{point_id}",
    response_model=ProfileRecord,
    summary="Every stored value of one point: raw ADC codes, voltages, Z, analysis",
)
async def get_profile(scan_id: str, point_id: int, repository: RepositoryDep) -> ProfileRecord:
    return await asyncio.to_thread(repository.get_profile, scan_id, point_id)


@router.post(
    "/{scan_id}/reconstruct",
    response_model=SurfaceResult,
    summary="(Re)build and store the surface of a finished confocal scan",
)
async def reconstruct(
    scan_id: str,
    manager: ScanManagerDep,
    request: Annotated[ReconstructionRequest | None, Body()] = None,
) -> SurfaceResult:
    return await manager.reconstruct(scan_id, request)


@router.get(
    "/{scan_id}/surface",
    response_model=SurfaceResult,
    summary="Latest stored surface reconstruction",
)
async def get_surface(scan_id: str, repository: RepositoryDep) -> SurfaceResult:
    surface = await asyncio.to_thread(repository.get_latest_surface, scan_id)
    if surface is None:
        raise not_found("SurfaceNotFound", f"scan {scan_id} has no surface reconstruction")
    return surface
