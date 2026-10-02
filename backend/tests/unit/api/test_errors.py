"""The exception -> HTTP status table of docs/architecture.md §5.8."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from confocal import errors
from confocal.api.errors import install_exception_handlers, status_for


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (errors.ScanNotFoundError("x"), 404),
        (errors.PointNotFoundError("x"), 404),
        (errors.ModelNotAvailableError("x"), 404),
        (errors.ScanConflictError("x"), 409),
        (errors.ScanStateError("x"), 409),
        (errors.EmergencyStopActiveError("x"), 409),
        (errors.CalibrationError("x"), 409),
        (errors.LimitViolationError("x"), 422),
        (errors.ReconstructionError("x"), 422),
        (errors.HardwareError("x"), 503),
        (errors.HardwareNotConnectedError("x"), 503),
        (errors.CommunicationError("x"), 503),
        (errors.ProtocolError("x"), 503),
        (errors.MotionError("x"), 503),
        (errors.MotionTimeoutError("x"), 503),
        (errors.MotionAbortedError("x"), 503),
        (errors.MotionVerificationError("x"), 503),
        (errors.ADCError("x"), 503),
        (errors.LaserError("x"), 503),
        (errors.CameraError("x"), 503),
        (errors.HardwareConfigError("x"), 500),
        (errors.StorageError("x"), 500),
        (errors.SafetyError("x"), 500),
        (errors.ScanError("x"), 500),
        (errors.ConfocalError("x"), 500),
    ],
)
def test_status_table(exc: errors.ConfocalError, status: int) -> None:
    assert status_for(exc) == status


def _app_raising(exc: Exception) -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise exc

    return app


def test_handler_body_shapes() -> None:
    violations = ["x=9 um outside [-1, 1] um"]
    client = TestClient(_app_raising(errors.LimitViolationError("out", violations=violations)))
    response = client.get("/boom")
    assert response.status_code == 422
    assert response.json() == {
        "error": "LimitViolationError",
        "detail": "out",
        "violations": violations,
    }

    client = TestClient(_app_raising(errors.HardwareConfigError("bad config")))
    response = client.get("/boom")
    assert response.status_code == 500
    assert response.json() == {
        "error": "HardwareConfigError",
        "detail": "bad config",
        "violations": None,
    }


def test_unexpected_exception_is_500_error_body() -> None:
    client = TestClient(_app_raising(RuntimeError("bug")), raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == 500
    assert response.json()["error"] == "RuntimeError"
