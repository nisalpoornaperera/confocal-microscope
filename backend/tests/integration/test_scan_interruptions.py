"""End to end: cancel, emergency stop, responsiveness during a scan, restart recovery."""

from __future__ import annotations

import time

import h5py
import numpy as np
import pytest
from tests.unit.api.helpers import (
    assert_error,
    running_client,
    scan_config,
    start_scan,
    wait_for_points,
    wait_for_scan,
    with_simulation,
)

from confocal.config import Settings
from confocal.models import PointStatus, ScanPoint
from confocal.scanning import SHUTDOWN_MESSAGE

pytestmark = pytest.mark.integration

#: Max latency of a status request while a scan is running (generous for slow CI).
MAX_STATUS_LATENCY_S = 0.5


@pytest.fixture
def slow_settings(sim_settings: Settings) -> Settings:
    """5 % of real time: the 20-point scan takes several seconds."""
    return with_simulation(sim_settings, time_scale=0.05)


def test_cancel_mid_scan_keeps_data(slow_settings: Settings) -> None:
    with running_client(slow_settings) as client:
        scan = start_scan(client)
        scan_id = scan["id"]
        wait_for_points(client, scan_id, 2)
        response = client.post(f"/api/v1/scans/{scan_id}/cancel")
        assert response.status_code == 200
        summary = wait_for_scan(client, scan_id)
        assert summary["state"] == "cancelled"
        assert summary["interrupted"] is True
        assert summary["error_message"] is None
        completed = summary["completed_points"]
        assert 2 <= completed < summary["total_points"]

        points = [
            ScanPoint.model_validate(p)
            for p in client.get(f"/api/v1/scans/{scan_id}/points").json()
        ]
        measured = [p for p in points if p.status is not PointStatus.ABORTED]
        assert len(measured) == completed
        assert [p.point_id for p in points] == list(range(len(points)))
        profile = client.get(f"/api/v1/scans/{scan_id}/profile/0")
        assert profile.status_code == 200
        assert profile.json()["raw_counts"]
        n_points = len(points)

        # the instrument is free again
        assert client.post("/api/v1/stage/move", json={"x_um": 0.0}).status_code == 200

    with h5py.File(slow_settings.storage.scans_dir / f"{scan_id}.h5", "r") as h5:
        index = np.asarray(h5["profiles/index"])
        assert len(index) >= completed
        assert h5["points"].shape[0] == n_points


def test_emergency_stop_mid_scan(slow_settings: Settings) -> None:
    with running_client(slow_settings) as client:
        scan = start_scan(client)
        scan_id = scan["id"]
        wait_for_points(client, scan_id, 1)

        response = client.post("/api/v1/stage/stop", json={"reason": "operator test"})
        assert response.status_code == 200
        assert response.json()["estop_engaged"] is True

        summary = wait_for_scan(client, scan_id)
        assert summary["state"] == "error"
        assert summary["interrupted"] is True
        assert "emergency stop" in summary["error_message"]
        assert summary["completed_points"] < summary["total_points"]
        assert client.get(f"/api/v1/scans/{scan_id}/points").json()  # data kept

        status = client.get("/api/v1/system/status").json()
        assert status["status"] == "estop"
        assert status["active_scan_id"] is None

        assert_error(
            client.post("/api/v1/stage/move", json={"z_um": 1.0}), 409, "EmergencyStopActiveError"
        )
        assert_error(client.post("/api/v1/stage/home"), 409, "EmergencyStopActiveError")
        assert_error(
            client.post("/api/v1/scans", json=scan_config()), 409, "EmergencyStopActiveError"
        )

        response = client.post("/api/v1/stage/reset")
        assert response.status_code == 200
        assert response.json()["estop_engaged"] is False
        assert client.post("/api/v1/stage/move", json={"z_um": 1.0}).status_code == 200
        assert client.get("/api/v1/system/status").json()["status"] == "ok"

        # scanning works again after the reset (the laser is switched back on)
        again = start_scan(client, mode="fixed_z", x_stop_um=20.0, y_stop_um=0.0)
        final = wait_for_scan(client, again["id"])
        assert final["state"] == "complete", final["error_message"]


def test_hardware_fault_mid_scan_ends_in_error(sim_settings: Settings) -> None:
    settings = with_simulation(sim_settings, fault_after_moves=40)
    with running_client(settings) as client:
        scan = start_scan(client)
        summary = wait_for_scan(client, scan["id"])
        assert summary["state"] == "error"
        assert summary["interrupted"] is True
        assert "hardware failure" in summary["error_message"]
        points = client.get(f"/api/v1/scans/{scan['id']}/points").json()
        assert points[-1]["status"] == "aborted"
        # the failure latched the e-stop: the operator must inspect and reset
        assert client.get("/api/v1/system/status").json()["estop_engaged"] is True


def test_http_stays_responsive_during_a_scan(slow_settings: Settings) -> None:
    with running_client(slow_settings) as client:
        scan = start_scan(client)
        scan_id = scan["id"]
        try:
            wait_for_scan(client, scan_id, {"scanning"})
            seen_scanning = 0
            for _ in range(15):
                started = time.perf_counter()
                response = client.get("/api/v1/system/status")
                latency = time.perf_counter() - started
                assert response.status_code == 200
                assert latency < MAX_STATUS_LATENCY_S, latency
                body = response.json()
                assert body["active_scan_id"] == scan_id
                seen_scanning += body["active_scan_state"] == "scanning"
                assert client.get(f"/api/v1/scans/{scan_id}").status_code == 200
                assert client.get(f"/api/v1/scans/{scan_id}/points").status_code == 200
                assert client.get("/api/v1/stage/position").status_code == 200
                time.sleep(0.02)
            assert seen_scanning > 0
            assert client.get(f"/api/v1/scans/{scan_id}").json()["state"] == "scanning"
        finally:
            client.post(f"/api/v1/scans/{scan_id}/cancel")
        assert wait_for_scan(client, scan_id)["state"] == "cancelled"


def test_restart_lists_old_scans_and_recovers_interrupted(
    sim_settings: Settings, slow_settings: Settings
) -> None:
    with running_client(sim_settings) as client:
        version = client.post("/api/v1/calibration/dark").json()["version"]
        finished = start_scan(client)
        assert wait_for_scan(client, finished["id"])["state"] == "complete"

    # A scan still running at shutdown is stored as interrupted.
    with running_client(slow_settings) as client:
        running = start_scan(client)
        wait_for_points(client, running["id"], 1)

    with running_client(sim_settings) as client:
        scans = {s["id"]: s for s in client.get("/api/v1/scans").json()}
        assert set(scans) == {finished["id"], running["id"]}
        assert scans[finished["id"]]["state"] == "complete"
        assert scans[finished["id"]]["interrupted"] is False
        assert scans[running["id"]]["state"] == "error"
        assert scans[running["id"]]["interrupted"] is True
        assert scans[running["id"]]["error_message"] == SHUTDOWN_MESSAGE

        assert len(client.get(f"/api/v1/scans/{finished['id']}/points").json()) == 20
        assert client.get(f"/api/v1/scans/{finished['id']}/profile/3").status_code == 200
        assert client.get(f"/api/v1/scans/{finished['id']}/surface").status_code == 200
        assert client.get("/api/v1/calibration").json()["version"] == version
        # the old data can still be re-processed
        assert client.post(f"/api/v1/scans/{finished['id']}/reconstruct").status_code == 200
