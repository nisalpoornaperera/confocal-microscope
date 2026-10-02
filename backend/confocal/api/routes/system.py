"""``/api/v1/system``: identity and health."""

from __future__ import annotations

from fastapi import APIRouter

from confocal.api.deps import ServicesDep
from confocal.api.errors import ERROR_RESPONSES
from confocal.models.system import SystemInfo, SystemStatus
from confocal.services import build_system_info, build_system_status

router = APIRouter(prefix="/api/v1/system", tags=["system"], responses=ERROR_RESPONSES)


@router.get("", response_model=SystemInfo, summary="Software, platform and hardware identity")
async def get_system_info(services: ServicesDep) -> SystemInfo:
    return build_system_info(services)


@router.get(
    "/status",
    response_model=SystemStatus,
    summary="Health: ok / degraded / estop / error, active scan, calibration, uptime",
)
async def get_system_status(services: ServicesDep) -> SystemStatus:
    return await build_system_status(services)
