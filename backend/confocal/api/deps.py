"""Typed FastAPI dependencies (the container lives in ``app.state.services``)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from starlette.requests import HTTPConnection

from confocal.errors import ScanConflictError
from confocal.microscope import StandardMicroscopeController
from confocal.ml import MLService
from confocal.models.scan import TERMINAL_SCAN_STATES
from confocal.scanning import ScanManager
from confocal.services import ServiceContainer
from confocal.storage import SQLiteHDF5Repository


def get_services(connection: HTTPConnection) -> ServiceContainer:
    """The application's container (works for HTTP requests and WebSockets)."""
    services = getattr(connection.app.state, "services", None)
    if not isinstance(services, ServiceContainer):
        raise RuntimeError("the application services are not running (lifespan not started)")
    return services


ServicesDep = Annotated[ServiceContainer, Depends(get_services)]


def get_controller(services: ServicesDep) -> StandardMicroscopeController:
    return services.controller


def get_scan_manager(services: ServicesDep) -> ScanManager:
    return services.scan_manager


def get_repository(services: ServicesDep) -> SQLiteHDF5Repository:
    return services.repository


def get_ml_service(services: ServicesDep) -> MLService:
    return services.ml


ControllerDep = Annotated[StandardMicroscopeController, Depends(get_controller)]
ScanManagerDep = Annotated[ScanManager, Depends(get_scan_manager)]
RepositoryDep = Annotated[SQLiteHDF5Repository, Depends(get_repository)]
MLServiceDep = Annotated[MLService, Depends(get_ml_service)]


def require_no_active_scan(manager: ScanManagerDep) -> None:
    """Manual hardware control is refused while a scan owns the instrument (§2.5).

    A scan whose state machine is already terminal is only persisting its final
    state (no hardware access left), so it no longer blocks manual control.
    """
    scan_id = manager.active_scan_id
    state = manager.active_scan_state
    if scan_id is not None and state not in TERMINAL_SCAN_STATES:
        state_text = f" ({state.value})" if state is not None else ""
        raise ScanConflictError(
            f"scan {scan_id} is active{state_text}: manual hardware control is refused "
            "until it finishes (POST /api/v1/stage/stop is always allowed)"
        )


#: Add to a route's ``dependencies`` to refuse it while a scan is active.
NoActiveScan = Depends(require_no_active_scan)
