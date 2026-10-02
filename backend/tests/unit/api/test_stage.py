"""``/api/v1/stage``: moves, homing, limits, faults, emergency stop latch and reset."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.helpers import (
    assert_error,
    running_client,
    services_of,
    start_scan,
    wait_for_scan,
    with_simulation,
)

from confocal.config import Settings
from confocal.models import HardwareStatus, Position


def test_position_starts_at_origin(client: TestClient) -> None:
    response = client.get("/api/v1/stage/position")
    assert response.status_code == 200
    position = Position.model_validate(response.json())
    assert position.max_axis_error(Position(x_um=0, y_um=0, z_um=0)) < 0.1


def test_absolute_and_relative_moves(client: TestClient) -> None:
    response = client.post("/api/v1/stage/move", json={"x_um": 100.0, "z_um": -20.0})
    assert response.status_code == 200
    moved = Position.model_validate(response.json())
    assert moved.x_um == pytest.approx(100.0, abs=0.5)
    assert moved.y_um == pytest.approx(0.0, abs=0.5)
    assert moved.z_um == pytest.approx(-20.0, abs=0.5)

    response = client.post("/api/v1/stage/move", json={"y_um": 50.0, "relative": True})
    assert response.status_code == 200
    relative = Position.model_validate(response.json())
    assert relative.x_um == pytest.approx(moved.x_um, abs=0.5)
    assert relative.y_um == pytest.approx(50.0, abs=0.5)

    reported = Position.model_validate(client.get("/api/v1/stage/position").json())
    assert reported.max_axis_error(relative) < 0.5


def test_home_with_and_without_body(client: TestClient) -> None:
    client.post("/api/v1/stage/move", json={"x_um": 30.0, "y_um": 40.0, "z_um": 10.0})
    response = client.post("/api/v1/stage/home", json={"axes": ["z"]})
    assert response.status_code == 200
    partial = Position.model_validate(response.json())
    assert partial.z_um == pytest.approx(0.0, abs=0.5)
    assert partial.x_um == pytest.approx(30.0, abs=0.5)

    response = client.post("/api/v1/stage/home")
    assert response.status_code == 200
    assert (
        Position.model_validate(response.json()).max_axis_error(Position(x_um=0, y_um=0, z_um=0))
        < 0.5
    )


def test_limit_violation_is_422_with_violations(client: TestClient) -> None:
    body = assert_error(
        client.post("/api/v1/stage/move", json={"x_um": 1.0e6}), 422, "LimitViolationError"
    )
    assert body["violations"]
    assert body["violations"][0].startswith("x=")
    # nothing moved
    assert Position.model_validate(client.get("/api/v1/stage/position").json()).x_um == (
        pytest.approx(0.0, abs=0.1)
    )


def test_relative_limit_violation(client: TestClient) -> None:
    body = assert_error(
        client.post("/api/v1/stage/move", json={"z_um": 5000.0, "relative": True}),
        422,
        "LimitViolationError",
    )
    assert any(v.startswith("z=") for v in body["violations"])


def test_move_validation_error_body(client: TestClient) -> None:
    body = assert_error(client.post("/api/v1/stage/move", json={}), 422, "RequestValidationError")
    assert body["violations"]
    assert any("at least one of x_um" in v for v in body["violations"])

    body = assert_error(
        client.post("/api/v1/stage/move", json={"x_um": "far", "speed": 3}),
        422,
        "RequestValidationError",
    )
    joined = " ".join(body["violations"])
    assert "body.x_um" in joined
    assert "body.speed" in joined


def test_home_rejects_unknown_axis(client: TestClient) -> None:
    assert_error(
        client.post("/api/v1/stage/home", json={"axes": ["w"]}), 422, "RequestValidationError"
    )


def test_stop_latches_and_reset_clears(client: TestClient) -> None:
    response = client.post("/api/v1/stage/stop", json={"reason": "test stop"})
    assert response.status_code == 200
    status = HardwareStatus.model_validate(response.json())
    assert status.estop_engaged is True
    assert status.estop_reason is not None
    assert status.estop_reason.startswith("test stop")
    assert status.laser.enabled is False  # the simulated laser is switched off

    system = client.get("/api/v1/system/status").json()
    assert system["status"] == "estop"
    assert system["estop_engaged"] is True

    assert_error(
        client.post("/api/v1/stage/move", json={"x_um": 10.0}), 409, "EmergencyStopActiveError"
    )
    assert_error(client.post("/api/v1/stage/home"), 409, "EmergencyStopActiveError")
    assert_error(
        client.post(
            "/api/v1/scans",
            json={
                "x_start_um": 0,
                "x_stop_um": 10,
                "y_start_um": 0,
                "y_stop_um": 10,
                "xy_step_um": 10,
            },
        ),
        409,
        "EmergencyStopActiveError",
    )

    # stop is allowed again (no body) while latched
    assert client.post("/api/v1/stage/stop").status_code == 200

    response = client.post("/api/v1/stage/reset")
    assert response.status_code == 200
    assert HardwareStatus.model_validate(response.json()).estop_engaged is False
    assert client.get("/api/v1/system/status").json()["status"] == "ok"
    assert client.post("/api/v1/stage/move", json={"x_um": 10.0}).status_code == 200


def test_stop_without_body_uses_default_reason(client: TestClient) -> None:
    response = client.post("/api/v1/stage/stop")
    assert response.status_code == 200
    assert response.json()["estop_reason"].startswith("operator emergency stop")


def test_stop_rejects_oversized_reason(client: TestClient) -> None:
    assert_error(
        client.post("/api/v1/stage/stop", json={"reason": "x" * 501}),
        422,
        "RequestValidationError",
    )


def test_hardware_fault_is_503_and_degrades_status(sim_settings: Settings) -> None:
    settings = with_simulation(sim_settings, fault_after_moves=0)
    with running_client(settings) as client:
        body = assert_error(
            client.post("/api/v1/stage/move", json={"x_um": 10.0}), 503, "MotionError"
        )
        assert "injected motion fault" in body["detail"]
        status = client.get("/api/v1/system/status").json()
        assert status["status"] == "degraded"
        assert status["hardware"]["stage"]["last_error"]


def test_manual_motion_refused_during_scan_but_stop_allowed(slow_client: TestClient) -> None:
    scan = start_scan(slow_client)
    try:
        for method, path, payload in (
            ("post", "/api/v1/stage/move", {"x_um": 5.0}),
            ("post", "/api/v1/stage/home", None),
            ("post", "/api/v1/stage/reset", None),
        ):
            response = slow_client.request(method, path, json=payload)
            body = assert_error(response, 409, "ScanConflictError")
            assert scan["id"] in body["detail"]
        # position reads never queue behind the scan
        assert slow_client.get("/api/v1/stage/position").status_code == 200
    finally:
        response = slow_client.post("/api/v1/stage/stop", json={"reason": "end of test"})
    assert response.status_code == 200
    assert response.json()["estop_engaged"] is True
    summary = wait_for_scan(slow_client, scan["id"])
    assert summary["state"] == "error"
    assert summary["interrupted"] is True
    assert services_of(slow_client).scan_manager.is_active is False
    assert slow_client.post("/api/v1/stage/reset").status_code == 200
