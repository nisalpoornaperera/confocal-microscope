"""Application factory, lifespan, system endpoints, OpenAPI, CORS, frontend mount."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.helpers import running_client, services_of

from confocal.api import create_app
from confocal.config import HardwareSelection, Settings
from confocal.errors import HardwareConfigError
from confocal.models import SystemInfo, SystemStatus

EXPECTED_PATHS = {
    "/api/v1/system": {"get"},
    "/api/v1/system/status": {"get"},
    "/api/v1/stage/position": {"get"},
    "/api/v1/stage/move": {"post"},
    "/api/v1/stage/home": {"post"},
    "/api/v1/stage/stop": {"post"},
    "/api/v1/stage/reset": {"post"},
    "/api/v1/adc/status": {"get"},
    "/api/v1/adc/read": {"get"},
    "/api/v1/adc/calibrate": {"post"},
    "/api/v1/calibration": {"get"},
    "/api/v1/calibration/history": {"get"},
    "/api/v1/calibration/dark": {"post"},
    "/api/v1/calibration/reference": {"post"},
    "/api/v1/scans": {"get", "post"},
    "/api/v1/scans/estimate": {"post"},
    "/api/v1/scans/{scan_id}": {"get"},
    "/api/v1/scans/{scan_id}/pause": {"post"},
    "/api/v1/scans/{scan_id}/resume": {"post"},
    "/api/v1/scans/{scan_id}/cancel": {"post"},
    "/api/v1/scans/{scan_id}/points": {"get"},
    "/api/v1/scans/{scan_id}/profile/{point_id}": {"get"},
    "/api/v1/scans/{scan_id}/reconstruct": {"post"},
    "/api/v1/scans/{scan_id}/surface": {"get"},
    "/api/v1/scans/{scan_id}/ml/analyse": {"post"},
    "/api/v1/scans/{scan_id}/ml/results": {"get"},
    "/api/v1/ml/models": {"get"},
}


def test_system_info(client: TestClient, sim_settings: Settings) -> None:
    response = client.get("/api/v1/system")
    assert response.status_code == 200
    info = SystemInfo.model_validate(response.json())
    assert info.simulation is True
    assert info.api_version == "v1"
    assert info.hardware.stage_backend == "simulation"
    assert info.limits == sim_settings.limits
    assert info.data_dir == str(sim_settings.storage.data_dir)


def test_system_status_ok(client: TestClient) -> None:
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    status = SystemStatus.model_validate(response.json())
    assert status.status == "ok"
    assert status.estop_engaged is False
    assert status.active_scan_id is None
    assert status.active_scan_state is None
    assert status.calibration_version is None
    assert status.uptime_s >= 0.0
    assert status.hardware.stage.connected


def test_lifespan_creates_data_dirs_and_stops_cleanly(sim_settings: Settings) -> None:
    app = create_app(sim_settings)
    with TestClient(app):
        services = app.state.services
        assert services is not None
        assert sim_settings.storage.scans_dir.is_dir()
        assert sim_settings.models_dir.is_dir()
        assert sim_settings.storage.database_path.is_file()
        assert services.controller.estop_engaged is False
    assert app.state.services is None
    assert not services.hardware.stage.connected
    # stop() is idempotent and never raises
    asyncio.run(services.stop())


def test_unimplemented_hardware_refuses_to_start(sim_settings: Settings) -> None:
    settings = sim_settings.model_copy(
        update={"hardware": HardwareSelection(stage="simulation", adc="simulation", laser="gpio")}
    )
    app = create_app(settings)
    with pytest.raises(HardwareConfigError, match="GPIO"), TestClient(app):
        pass


def test_default_settings_are_loaded(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    monkeypatch.delenv("CONFOCAL_CONFIG", raising=False)
    monkeypatch.setenv("CONFOCAL_DATA_DIR", str(data_dir / "env"))
    app = create_app()
    assert app.state.settings.storage.data_dir == data_dir / "env"


def test_openapi_schema_lists_every_endpoint(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]
    for path, methods in EXPECTED_PATHS.items():
        assert path in paths, path
        assert methods <= set(paths[path]), (path, set(paths[path]))
    assert "ErrorResponse" in schema["components"]["schemas"]
    move_422 = paths["/api/v1/stage/move"]["post"]["responses"]["422"]
    assert move_422["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")
    assert client.get("/docs").status_code == 200


def test_root_points_to_docs_without_frontend(sim_settings: Settings, tmp_path: Path) -> None:
    with running_client(sim_settings, frontend_dir=tmp_path / "missing") as client:
        body = client.get("/").json()
        assert body["docs"] == "/docs"


def test_built_frontend_is_served_after_the_api(sim_settings: Settings, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>confocal ui</title>", "utf-8")
    with running_client(sim_settings, frontend_dir=dist) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "confocal ui" in page.text
        assert client.get("/api/v1/system/status").json()["status"] == "ok"
        assert client.get("/openapi.json").status_code == 200


def test_cors_preflight_allows_configured_origin(client: TestClient) -> None:
    response = client.options(
        "/api/v1/system",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_unknown_route_uses_error_body(client: TestClient) -> None:
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.json() == {"error": "NotFound", "detail": "Not Found", "violations": None}


def test_container_exposes_the_simulated_surface(client: TestClient) -> None:
    services = services_of(client)
    assert services.surface is services.hardware.surface
    assert services.surface is not None
    assert services.scan_manager.is_active is False
