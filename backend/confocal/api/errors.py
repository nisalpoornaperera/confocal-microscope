"""Mapping of application exceptions to HTTP responses (docs/architecture.md §5.8).

Every non-2xx response produced by the application has an
:class:`~confocal.models.system.ErrorResponse` body ``{error, detail, violations?}``.

| Exception                                                                   | HTTP |
|-----------------------------------------------------------------------------|------|
| ScanNotFoundError, PointNotFoundError, ModelNotAvailableError               | 404  |
| ScanConflictError, ScanStateError, EmergencyStopActiveError, CalibrationError | 409 |
| LimitViolationError, ReconstructionError, request validation                | 422  |
| HardwareError (all subclasses)                                              | 503  |
| HardwareConfigError, any other ConfocalError                                | 500  |
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from confocal.errors import (
    CalibrationError,
    ConfocalError,
    EmergencyStopActiveError,
    HardwareConfigError,
    HardwareError,
    LimitViolationError,
    ModelNotAvailableError,
    PointNotFoundError,
    ReconstructionError,
    ScanConflictError,
    ScanNotFoundError,
    ScanStateError,
)
from confocal.models.system import ErrorResponse

log = logging.getLogger(__name__)

#: Checked in order; the first matching class wins (subclasses before their bases).
ERROR_STATUS: tuple[tuple[type[ConfocalError], int], ...] = (
    (ScanNotFoundError, 404),
    (PointNotFoundError, 404),
    (ModelNotAvailableError, 404),
    (ScanConflictError, 409),
    (ScanStateError, 409),
    (EmergencyStopActiveError, 409),
    (CalibrationError, 409),
    (LimitViolationError, 422),
    (ReconstructionError, 422),
    (HardwareError, 503),
    (HardwareConfigError, 500),
)

#: Documented on every API route (OpenAPI).
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse, "description": description}
    for status, description in (
        (404, "Unknown scan, point, result or ML model"),
        (409, "Conflict: active scan, invalid scan state, e-stop latched or calibration"),
        (422, "Validation error or travel-limit violation"),
        (503, "Hardware failure"),
        (500, "Configuration or internal error"),
    )
}


class APIError(Exception):
    """An HTTP-level error raised by a route (e.g. a scan with no surface yet)."""

    def __init__(self, status_code: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error = error
        self.detail = detail


def not_found(error: str, detail: str) -> APIError:
    return APIError(404, error, detail)


def status_for(exc: ConfocalError) -> int:
    for cls, status in ERROR_STATUS:
        if isinstance(exc, cls):
            return status
    return 500


def error_response(
    status_code: int, error: str, detail: str, violations: list[str] | None = None
) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, violations=violations)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _location(loc: tuple[int | str, ...] | list[int | str]) -> str:
    return ".".join(str(part) for part in loc)


async def _confocal_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ConfocalError)
    status = status_for(exc)
    if status >= 500:
        log.error("%s %s failed: %s", request.method, request.url.path, exc, exc_info=exc)
    violations = exc.violations if isinstance(exc, LimitViolationError) else None
    return error_response(status, type(exc).__name__, str(exc) or type(exc).__name__, violations)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    violations = [f"{_location(err.get('loc', ()))}: {err.get('msg', '')}" for err in exc.errors()]
    detail = f"request validation failed ({len(violations)} error(s))"
    return error_response(422, "RequestValidationError", detail, violations)


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, APIError)
    return error_response(exc.status_code, exc.error, exc.detail)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    try:
        name = HTTPStatus(exc.status_code).phrase.replace(" ", "")
    except ValueError:
        name = "HTTPError"
    response = error_response(exc.status_code, name, str(exc.detail))
    if exc.headers:
        response.headers.update(exc.headers)
    return response


async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    log.error("%s %s: unhandled error", request.method, request.url.path, exc_info=exc)
    return error_response(500, type(exc).__name__, "internal server error")


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ConfocalError, _confocal_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(APIError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected_error)
