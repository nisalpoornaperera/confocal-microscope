"""``/api/v1/stage``: position, manual moves, homing, emergency stop and its reset.

Moves and homing are refused while a scan is active (409). ``stop`` is ALWAYS
allowed: it latches the emergency stop (the active scan, if any, ends in
ERROR) and motion stays refused until ``reset``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body

from confocal.api.deps import ControllerDep, NoActiveScan, ScanManagerDep
from confocal.api.errors import ERROR_RESPONSES
from confocal.models.common import Position
from confocal.models.hardware import HardwareStatus
from confocal.models.system import EmergencyStopRequest, StageHomeRequest, StageMoveRequest

router = APIRouter(prefix="/api/v1/stage", tags=["stage"], responses=ERROR_RESPONSES)


@router.get("/position", response_model=Position, summary="Current stage position (um)")
async def get_position(controller: ControllerDep, manager: ScanManagerDep) -> Position:
    if manager.is_active:
        # Never queue behind the scan's hardware lock: report the live status instead.
        status = await controller.status()
        if status.stage.position is not None:
            return status.stage.position
    return await controller.get_position()


@router.post(
    "/move",
    response_model=Position,
    dependencies=[NoActiveScan],
    summary="Absolute or relative move (limit-checked and verified)",
)
async def move(request: StageMoveRequest, controller: ControllerDep) -> Position:
    if request.relative:
        return await controller.move_relative(
            dx_um=request.x_um or 0.0, dy_um=request.y_um or 0.0, dz_um=request.z_um or 0.0
        )
    return await controller.move_to(x_um=request.x_um, y_um=request.y_um, z_um=request.z_um)


@router.post(
    "/home", response_model=Position, dependencies=[NoActiveScan], summary="Home axes to 0"
)
async def home(
    controller: ControllerDep,
    request: Annotated[StageHomeRequest | None, Body()] = None,
) -> Position:
    axes = (request or StageHomeRequest()).axes
    return await controller.home(axes)


@router.post(
    "/stop",
    response_model=HardwareStatus,
    summary="EMERGENCY STOP: always allowed; halts motion and latches until reset",
)
async def stop(
    controller: ControllerDep,
    request: Annotated[EmergencyStopRequest | None, Body()] = None,
) -> HardwareStatus:
    reason = (request or EmergencyStopRequest()).reason
    await controller.emergency_stop(reason)
    return await controller.status()


@router.post(
    "/reset",
    response_model=HardwareStatus,
    dependencies=[NoActiveScan],
    summary="Clear the emergency-stop latch (after inspecting the machine)",
)
async def reset(controller: ControllerDep) -> HardwareStatus:
    await controller.reset_emergency_stop()
    return await controller.status()
