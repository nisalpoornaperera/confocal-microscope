"""End to end: API -> ScanManager -> simulated hardware -> processing -> SQLite + HDF5."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from tests.unit.api.helpers import (
    assert_error,
    running_client,
    scan_config,
    services_of,
    start_scan,
    wait_for_scan,
    with_simulation,
)

from confocal.config import HardwareSelection, Settings
from confocal.models import (
    PointStatus,
    ProcessingConfig,
    ProfilePhase,
    ProfileRecord,
    ScanEvent,
    ScanEventType,
    ScanPoint,
    ScanState,
    SurfaceResult,
)

pytestmark = pytest.mark.integration

#: Max |measured - true| surface height of a VALID point on the simulated sample.
HEIGHT_TOLERANCE_UM = 0.5


def _reject_constant(value: str) -> Any:
    raise AssertionError(f"non-JSON constant {value} in a response")


def strict_json(text: str) -> Any:
    """Parse JSON refusing NaN / Infinity (API models must never contain them)."""
    return json.loads(text, parse_constant=_reject_constant)


def calibrate(client: TestClient) -> int:
    dark = client.post("/api/v1/calibration/dark", json={"n_samples": 64})
    assert dark.status_code == 200, dark.text
    reference = client.post("/api/v1/calibration/reference", json={"z_search": True})
    assert reference.status_code == 200, reference.text
    version: int = reference.json()["version"]
    assert version == dark.json()["version"] + 1
    return version


def scan_file(settings: Settings, scan_id: str) -> Path:
    return settings.storage.scans_dir / f"{scan_id}.h5"


def test_confocal_scan_end_to_end(sim_settings: Settings) -> None:
    with running_client(sim_settings) as client:
        version = calibrate(client)
        scan = start_scan(client)
        scan_id = scan["id"]
        summary = wait_for_scan(client, scan_id)
        assert summary["state"] == "complete", summary["error_message"]
        assert summary["completed_points"] == summary["total_points"] == 20
        assert summary["interrupted"] is False
        assert summary["calibration_version"] == version
        assert summary["has_surface"] is True
        assert summary["data_file"]

        points = [
            ScanPoint.model_validate(p)
            for p in strict_json(client.get(f"/api/v1/scans/{scan_id}/points").text)
        ]
        assert [p.point_id for p in points] == list(range(20))
        assert {(p.ix, p.iy) for p in points} == {(ix, iy) for ix in range(5) for iy in range(4)}
        allowed = {
            PointStatus.VALID,
            PointStatus.LOW_CONFIDENCE,
            PointStatus.NO_PEAK,
            PointStatus.PEAK_AT_EDGE,
            PointStatus.FIT_FAILED,
        }
        assert all(p.status in allowed for p in points)
        valid = [p for p in points if p.status is PointStatus.VALID]
        assert len(valid) >= 0.6 * len(points)

        surface_model = services_of(client).surface
        assert surface_model is not None
        for point in valid:
            assert point.surface_z_um is not None
            truth = float(surface_model.height_um(point.x_um, point.y_um))
            assert abs(point.surface_z_um - truth) < HEIGHT_TOLERANCE_UM, (point, truth)
            assert 0.0 < point.confidence <= 1.0
        for point in points:
            if point.status is PointStatus.NO_PEAK:
                assert point.surface_z_um is None

        response = client.get(f"/api/v1/scans/{scan_id}/profile/{valid[0].point_id}")
        assert response.status_code == 200
        profile = ProfileRecord.model_validate(strict_json(response.text))
        n = len(profile.z_um)
        assert n == valid[0].n_z_positions > 0
        assert len(profile.z_reported_um) == len(profile.phase) == len(profile.voltage_agg_v) == n
        assert len(profile.raw_counts) == len(profile.voltage_v) == n
        assert all(len(row) == 4 for row in profile.raw_counts)  # samples_per_z
        assert all(isinstance(c, int) for row in profile.raw_counts for c in row)
        assert {ProfilePhase.COARSE, ProfilePhase.FINE} <= set(profile.phase)
        assert profile.calibration_version == version
        assert profile.analysis is not None

        response = client.get(f"/api/v1/scans/{scan_id}/surface")
        assert response.status_code == 200
        surface = SurfaceResult.model_validate(strict_json(response.text))
        assert len(surface.z_um) == len(surface.y_um)
        assert all(len(row) == len(surface.x_um) for row in surface.z_um)
        assert surface.statistics.n_used >= len(valid) - surface.statistics.n_outliers - 1
        finite = [z for row in surface.z_um for z in row if z is not None]
        assert finite
        assert all(math.isfinite(z) for z in finite)
        stored_profile = profile

    # After shutdown the HDF5 file is closed: verify every point's raw data is in it.
    path = scan_file(sim_settings, scan_id)
    assert path.is_file()
    with h5py.File(path, "r") as h5:
        assert h5.attrs["scan_id"] == scan_id
        index = np.asarray(h5["profiles/index"])
        raw_counts = h5["profiles/raw_counts"]
        assert sorted(index[:, 0].tolist()) == list(range(20))
        assert raw_counts.shape[1] == 4
        assert int(index[:, 2].sum()) == raw_counts.shape[0]
        assert (index[:, 2] > 0).all()
        row = next(r for r in index if r[0] == stored_profile.point_id)
        stored = np.asarray(raw_counts[row[1] : row[1] + row[2]])
        assert stored.tolist() == stored_profile.raw_counts
        assert h5["points"].shape[0] == 20


def test_websocket_streams_a_scan(sim_settings: Settings) -> None:
    # time_scale 0: the scan takes about a second, connecting takes milliseconds. (A
    # non-zero time scale is much slower than it suggests on Windows, where every
    # short asyncio.sleep lasts a whole ~15 ms timer tick.)
    with running_client(sim_settings) as client:
        scan = start_scan(client)
        events: list[ScanEvent] = []
        close_code: int | None = None
        with client.websocket_connect(f"/ws/scans/{scan['id']}") as ws:
            while close_code is None:
                try:
                    events.append(ScanEvent.model_validate(ws.receive_json()))
                except WebSocketDisconnect as exc:
                    close_code = exc.code
        assert close_code == 1000

    snapshot, *stream = events
    assert snapshot.type is ScanEventType.SNAPSHOT
    assert snapshot.scan_id == scan["id"]
    assert stream, "no live events after the snapshot"
    types = {event.type for event in stream}
    assert {ScanEventType.STATE, ScanEventType.POINT, ScanEventType.PROGRESS} <= types
    last = stream[-1]
    assert last.type is ScanEventType.STATE
    assert last.progress.state is ScanState.COMPLETE
    point_ids = [e.point.point_id for e in stream if e.type is ScanEventType.POINT and e.point]
    assert point_ids == list(range(snapshot.progress.completed_points, 20))
    states = [e.progress.state for e in stream if e.type is ScanEventType.STATE]
    assert ScanState.PROCESSING in states
    assert states.index(ScanState.PROCESSING) < states.index(ScanState.COMPLETE)


def test_fixed_z_scan(sim_settings: Settings) -> None:
    with running_client(sim_settings) as client:
        calibrate(client)
        scan = start_scan(client, mode="fixed_z", z_center_um=3.0)
        summary = wait_for_scan(client, scan["id"])
        assert summary["state"] == "complete"
        assert summary["has_surface"] is False
        points = [
            ScanPoint.model_validate(p)
            for p in client.get(f"/api/v1/scans/{scan['id']}/points").json()
        ]
        assert len(points) == 20
        assert all(p.status is PointStatus.MEASURED for p in points)
        assert all(p.intensity is not None for p in points)
        assert all(p.surface_z_um is None for p in points)
        profile = ProfileRecord.model_validate(
            client.get(f"/api/v1/scans/{scan['id']}/profile/0").json()
        )
        assert profile.phase == [ProfilePhase.FIXED]
        assert profile.z_um == [pytest.approx(3.0)]
        assert_error(client.get(f"/api/v1/scans/{scan['id']}/surface"), 404, "SurfaceNotFound")


def test_manual_laser_requires_beam_blocked_confirmation(sim_settings: Settings) -> None:
    settings = sim_settings.model_copy(update={"hardware": HardwareSelection(laser="manual")})
    with running_client(settings) as client:
        laser = client.get("/api/v1/system/status").json()["hardware"]["laser"]
        assert laser["controllable"] is False
        assert laser["enabled"] is None

        body = assert_error(
            client.post("/api/v1/calibration/dark", json={}), 409, "CalibrationError"
        )
        assert "beam_blocked_confirmed" in body["detail"]
        assert client.get("/api/v1/calibration").json()["version"] is None

        response = client.post(
            "/api/v1/calibration/dark", json={"beam_blocked_confirmed": True, "notes": "capped"}
        )
        assert response.status_code == 200
        assert response.json()["version"] == 1

        # an automatic dark calibration is impossible with a manual laser: refused up front
        assert_error(
            client.post(
                "/api/v1/scans",
                json=scan_config(calibrate_dark_before_scan=True),
            ),
            409,
            "CalibrationError",
        )
        assert client.get("/api/v1/scans").json() == []

        scan = start_scan(client, mode="fixed_z")
        summary = wait_for_scan(client, scan["id"])
        assert summary["state"] == "complete"
        assert summary["calibration_version"] == 1

        # an e-stop cannot switch a manual laser off: the operator is told
        stop = client.post("/api/v1/stage/stop", json={"reason": "check"}).json()
        assert "switch it off by hand" in stop["estop_reason"]


@pytest.mark.parametrize("rail_v", [1.95, None])
def test_detector_saturation_from_the_settings_reaches_the_analysis(
    sim_settings: Settings, rail_v: float | None
) -> None:
    """The OPT101 clips at 2.0 V, well below the ADC full scale (gain 1: 4.096 V).

    Only the settings-level detector rail (``[processing] saturation_v``) lets the
    analysis see the clipped top; the scan request itself sets no saturation_v.
    """
    settings = with_simulation(sim_settings, peak_voltage_v=3.0)
    settings = settings.model_copy(update={"processing": ProcessingConfig(saturation_v=rail_v)})
    with running_client(settings) as client:
        calibrate(client)
        scan_id = start_scan(client)["id"]
        summary = wait_for_scan(client, scan_id)
        assert summary["state"] == "complete", summary["error_message"]
        assert summary["config"]["processing"]["saturation_v"] == rail_v
        points = [
            ScanPoint.model_validate(p)
            for p in strict_json(client.get(f"/api/v1/scans/{scan_id}/points").text)
        ]
        peaks = [p for p in points if p.peak_intensity is not None]
        assert peaks
        saturated = [p for p in peaks if "saturated" in p.flags]
        if rail_v is None:
            assert saturated == []  # the clipping is invisible without the detector rail
            return
        assert len(saturated) >= len(peaks) // 2
        services = services_of(client)
        assert services.surface is not None
        for point in saturated:
            if point.status is PointStatus.VALID:
                assert point.surface_z_um is not None
                truth = float(services.surface.height_um(point.x_um, point.y_um))
                assert abs(point.surface_z_um - truth) < HEIGHT_TOLERANCE_UM
            record = ProfileRecord.model_validate(
                strict_json(client.get(f"/api/v1/scans/{scan_id}/profile/{point.point_id}").text)
            )
            assert record.analysis is not None
            assert record.analysis.saturated_fraction > 0.0
