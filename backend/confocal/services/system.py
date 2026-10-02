"""System identity and health summaries (``GET /api/v1/system[/status]``)."""

from __future__ import annotations

import platform
from typing import Literal

from confocal import __version__
from confocal.models.hardware import HardwareStatus
from confocal.models.system import SystemInfo, SystemStatus
from confocal.services.container import ServiceContainer

StatusLevel = Literal["ok", "degraded", "estop", "error"]


def build_system_info(services: ServiceContainer) -> SystemInfo:
    settings = services.settings
    return SystemInfo(
        software_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        simulation=settings.is_simulation,
        hardware=services.controller.hardware_info(),
        limits=settings.limits,
        data_dir=str(settings.storage.data_dir),
    )


def component_errors(hardware: HardwareStatus) -> list[str]:
    """Every ``last_error`` reported by a component (empty when all are healthy)."""
    errors: list[str] = []
    for name, error in (
        ("stage", hardware.stage.last_error),
        ("adc", hardware.adc.last_error),
        ("laser", hardware.laser.last_error),
        ("camera", hardware.camera.last_error),
    ):
        if error:
            errors.append(f"{name}: {error}")
    return errors


def overall_status(hardware: HardwareStatus) -> StatusLevel:
    """``estop`` when latched, ``error`` when the stage or ADC is disconnected,
    ``degraded`` when a component reports an error, else ``ok``."""
    if hardware.estop_engaged:
        return "estop"
    if not (hardware.stage.connected and hardware.adc.connected):
        return "error"
    if component_errors(hardware):
        return "degraded"
    return "ok"


async def build_system_status(services: ServiceContainer) -> SystemStatus:
    """Health snapshot. Never takes the hardware lock (stays responsive during a scan)."""
    hardware = await services.controller.status()
    manager = services.scan_manager
    return SystemStatus(
        status=overall_status(hardware),
        hardware=hardware,
        estop_engaged=hardware.estop_engaged,
        active_scan_id=manager.active_scan_id,
        active_scan_state=manager.active_scan_state,
        calibration_version=services.controller.calibration.version,
        uptime_s=services.uptime_s,
    )
