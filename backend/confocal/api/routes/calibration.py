"""``/api/v1/calibration``: dark / reference measurement and versioned history."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Body, Query

from confocal.api.deps import ControllerDep, NoActiveScan, RepositoryDep
from confocal.api.errors import ERROR_RESPONSES
from confocal.models.calibration import (
    CalibrationState,
    DarkCalibrationRequest,
    ReferenceCalibrationRequest,
)

router = APIRouter(prefix="/api/v1/calibration", tags=["calibration"], responses=ERROR_RESPONSES)


@router.get("", response_model=CalibrationState, summary="Current calibration snapshot")
async def get_calibration(controller: ControllerDep) -> CalibrationState:
    return controller.calibration


@router.get(
    "/history",
    response_model=list[CalibrationState],
    summary="Calibration versions, newest first",
)
async def get_history(
    repository: RepositoryDep, limit: Annotated[int, Query(ge=1, le=1000)] = 50
) -> list[CalibrationState]:
    return await asyncio.to_thread(repository.list_calibrations, limit)


@router.post(
    "/dark",
    response_model=CalibrationState,
    dependencies=[NoActiveScan],
    summary="Measure the dark level (manual laser: requires beam_blocked_confirmed)",
)
async def calibrate_dark(
    controller: ControllerDep,
    request: Annotated[DarkCalibrationRequest | None, Body()] = None,
) -> CalibrationState:
    return await controller.calibrate_dark(request or DarkCalibrationRequest())


@router.post(
    "/reference",
    response_model=CalibrationState,
    dependencies=[NoActiveScan],
    summary="Measure the in-focus reference level (optionally searching Z)",
)
async def calibrate_reference(
    controller: ControllerDep,
    request: Annotated[ReferenceCalibrationRequest | None, Body()] = None,
) -> CalibrationState:
    return await controller.calibrate_reference(request or ReferenceCalibrationRequest())
