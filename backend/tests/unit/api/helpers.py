"""Helpers shared by the API unit tests and the integration tests."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient

from confocal.api import create_app
from confocal.config import Settings, SimulationConfig
from confocal.services import ServiceContainer

TERMINAL = frozenset({"complete", "cancelled", "error"})

#: 5 x 4 confocal scan over a few micrometres of simulated relief (about 1 s at time_scale 0).
SMALL_SCAN: dict[str, Any] = {
    "name": "small",
    "x_start_um": 0.0,
    "x_stop_um": 80.0,
    "y_start_um": 0.0,
    "y_stop_um": 60.0,
    "xy_step_um": 20.0,
    "z_center_um": 0.0,
    "z_range_um": 40.0,
    "adaptive_z_range_um": 20.0,
}


def scan_config(**overrides: Any) -> dict[str, Any]:
    return {**SMALL_SCAN, **overrides}


def with_simulation(settings: Settings, **updates: Any) -> Settings:
    """Copy of ``settings`` with simulation fields replaced (re-validated)."""
    simulation = SimulationConfig.model_validate({**settings.simulation.model_dump(), **updates})
    return settings.model_copy(update={"simulation": simulation})


@contextmanager
def running_client(settings: Settings, **app_kwargs: Any) -> Iterator[TestClient]:
    """TestClient with the lifespan running (services started and stopped)."""
    with TestClient(create_app(settings, **app_kwargs)) as client:
        yield client


def services_of(client: TestClient) -> ServiceContainer:
    services = client.app.state.services  # type: ignore[attr-defined]
    assert isinstance(services, ServiceContainer)
    return services


def wait_for(
    predicate: Callable[[], bool], *, timeout_s: float = 20.0, interval_s: float = 0.01
) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout_s} s")
        time.sleep(interval_s)


def wait_for_scan(
    client: TestClient,
    scan_id: str,
    states: frozenset[str] | set[str] = TERMINAL,
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Poll ``GET /scans/{id}`` until its state is in ``states``; returns the summary.

    For a terminal state it also waits until the scan manager has released the
    instrument (the state is persisted just before that).
    """
    latest: dict[str, Any] = {}

    def reached() -> bool:
        nonlocal latest
        response = client.get(f"/api/v1/scans/{scan_id}")
        assert response.status_code == 200, response.text
        latest = response.json()
        if latest["state"] not in states:
            return False
        if latest["state"] in TERMINAL:
            return services_of(client).scan_manager.active_scan_id != scan_id
        return True

    wait_for(reached, timeout_s=timeout_s)
    return latest


def wait_for_points(client: TestClient, scan_id: str, n: int, *, timeout_s: float = 30.0) -> None:
    def enough() -> bool:
        summary = client.get(f"/api/v1/scans/{scan_id}").json()
        return summary["completed_points"] >= n or summary["state"] in TERMINAL

    wait_for(enough, timeout_s=timeout_s)


def start_scan(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post("/api/v1/scans", json=scan_config(**overrides))
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def assert_error(response: Any, status: int, error: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body: dict[str, Any] = response.json()
    assert set(body) <= {"error", "detail", "violations"}
    assert body["error"] == error, body
    assert isinstance(body["detail"], str)
    assert body["detail"]
    return body
