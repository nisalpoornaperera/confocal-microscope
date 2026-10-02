"""``/api/v1/scans``: create, estimate, list, control, points, profiles, surfaces."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from tests.unit.api.helpers import (
    assert_error,
    scan_config,
    start_scan,
    wait_for_points,
    wait_for_scan,
)

from confocal.models import ProfileRecord, ScanEstimate, ScanPoint, ScanSummary, SurfaceResult

UNKNOWN = "0" * 32


def test_estimate(client: TestClient) -> None:
    response = client.post("/api/v1/scans/estimate", json=scan_config())
    assert response.status_code == 200
    estimate = ScanEstimate.model_validate(response.json())
    assert (estimate.n_x, estimate.n_y, estimate.total_points) == (5, 4, 20)
    assert estimate.within_limits
    assert estimate.estimated_duration_s > 0
    assert client.get("/api/v1/scans").json() == []  # nothing was created


def test_estimate_reports_limit_violations(client: TestClient) -> None:
    response = client.post("/api/v1/scans/estimate", json=scan_config(x_stop_um=1.0e5))
    assert response.status_code == 200
    estimate = ScanEstimate.model_validate(response.json())
    assert not estimate.within_limits
    assert estimate.limit_violations


def test_create_outside_limits_is_422(client: TestClient) -> None:
    body = assert_error(
        client.post("/api/v1/scans", json=scan_config(x_stop_um=1.0e5)),
        422,
        "LimitViolationError",
    )
    assert body["violations"]
    assert client.get("/api/v1/scans").json() == []


def test_non_finite_values_are_422_not_500(client: TestClient) -> None:
    """inf used to overflow axis_count (OverflowError -> HTTP 500)."""
    for path in ("/api/v1/scans", "/api/v1/scans/estimate"):
        for body in (
            scan_config(z_range_um="inf"),
            scan_config(x_stop_um="Infinity"),
            scan_config(processing={"saturation_v": "nan"}),
            scan_config(reconstruction={"grid_step_um": "inf"}),
        ):
            response = assert_error(client.post(path, json=body), 422, "RequestValidationError")
            assert "finite" in response["detail"] + " ".join(response["violations"])
        raw = json.dumps(scan_config()).replace('"xy_step_um": 20.0', '"xy_step_um": Infinity')
        assert "Infinity" in raw
        response = client.post(path, content=raw, headers={"content-type": "application/json"})
        assert_error(response, 422, "RequestValidationError")
    assert client.get("/api/v1/scans").json() == []


def test_create_validation_error(client: TestClient) -> None:
    body = assert_error(
        client.post("/api/v1/scans", json=scan_config(xy_step_um=0, unknown=1)),
        422,
        "RequestValidationError",
    )
    joined = " ".join(body["violations"])
    assert "body.xy_step_um" in joined
    assert "body.unknown" in joined
    body = assert_error(
        client.post("/api/v1/scans", json=scan_config(x_stop_um=-10.0)),
        422,
        "RequestValidationError",
    )
    assert any("x_stop_um must be >= x_start_um" in v for v in body["violations"])


def test_full_scan_lifecycle(client: TestClient) -> None:
    response = client.post("/api/v1/scans", json=scan_config())
    assert response.status_code == 201
    created = ScanSummary.model_validate(response.json())
    assert created.total_points == 20
    assert created.name == "small"

    final = ScanSummary.model_validate(wait_for_scan(client, created.id))
    assert final.state.value == "complete", final.error_message
    assert final.completed_points == 20
    assert final.interrupted is False
    assert final.has_surface is True
    assert final.progress == 1.0

    listed = [ScanSummary.model_validate(s) for s in client.get("/api/v1/scans").json()]
    assert [s.id for s in listed] == [created.id]
    assert client.get("/api/v1/scans", params={"offset": 1}).json() == []

    points = [
        ScanPoint.model_validate(p) for p in client.get(f"/api/v1/scans/{created.id}/points").json()
    ]
    assert [p.point_id for p in points] == list(range(20))
    since = client.get(f"/api/v1/scans/{created.id}/points", params={"since": 15}).json()
    assert [p["point_id"] for p in since] == [16, 17, 18, 19]
    limited = client.get(f"/api/v1/scans/{created.id}/points", params={"since": 2, "limit": 3})
    assert [p["point_id"] for p in limited.json()] == [3, 4, 5]

    profile = ProfileRecord.model_validate(
        client.get(f"/api/v1/scans/{created.id}/profile/1").json()
    )
    assert profile.point_id == 1
    assert len(profile.raw_counts) == len(profile.z_um) > 0
    assert len(profile.raw_counts[0]) == created.config.samples_per_z

    surface = SurfaceResult.model_validate(client.get(f"/api/v1/scans/{created.id}/surface").json())
    assert surface.scan_id == created.id
    assert surface.surface_id is not None

    response = client.post(
        f"/api/v1/scans/{created.id}/reconstruct", json={"method": "nearest", "min_confidence": 0}
    )
    assert response.status_code == 200
    rebuilt = SurfaceResult.model_validate(response.json())
    assert rebuilt.request.method.value == "nearest"
    assert rebuilt.surface_id is not None
    assert rebuilt.surface_id > surface.surface_id
    latest = client.get(f"/api/v1/scans/{created.id}/surface").json()
    assert latest["surface_id"] == rebuilt.surface_id
    # without a body the scan's own reconstruction settings are used
    assert client.post(f"/api/v1/scans/{created.id}/reconstruct").status_code == 200

    # control endpoints on a finished scan
    for action in ("pause", "resume", "cancel"):
        assert_error(client.post(f"/api/v1/scans/{created.id}/{action}"), 409, "ScanStateError")


def test_unknown_scan_and_point_are_404(client: TestClient) -> None:
    for method, path in (
        ("get", f"/api/v1/scans/{UNKNOWN}"),
        ("post", f"/api/v1/scans/{UNKNOWN}/pause"),
        ("post", f"/api/v1/scans/{UNKNOWN}/resume"),
        ("post", f"/api/v1/scans/{UNKNOWN}/cancel"),
        ("get", f"/api/v1/scans/{UNKNOWN}/points"),
        ("get", f"/api/v1/scans/{UNKNOWN}/profile/0"),
        ("post", f"/api/v1/scans/{UNKNOWN}/reconstruct"),
        ("get", f"/api/v1/scans/{UNKNOWN}/surface"),
    ):
        assert_error(client.request(method, path), 404, "ScanNotFoundError")

    scan = start_scan(client)
    wait_for_scan(client, scan["id"])
    assert_error(client.get(f"/api/v1/scans/{scan['id']}/profile/999"), 404, "PointNotFoundError")
    assert_error(
        client.get(f"/api/v1/scans/{scan['id']}/profile/not-a-number"),
        422,
        "RequestValidationError",
    )


def test_scan_without_surface_is_404(client: TestClient) -> None:
    scan = start_scan(client, reconstruct_on_complete=False)
    summary = wait_for_scan(client, scan["id"])
    assert summary["state"] == "complete"
    assert summary["has_surface"] is False
    assert_error(client.get(f"/api/v1/scans/{scan['id']}/surface"), 404, "SurfaceNotFound")


def test_fixed_z_scan_cannot_be_reconstructed(client: TestClient) -> None:
    scan = start_scan(client, mode="fixed_z")
    assert wait_for_scan(client, scan["id"])["state"] == "complete"
    assert_error(client.post(f"/api/v1/scans/{scan['id']}/reconstruct"), 422, "ReconstructionError")


def test_second_scan_conflicts_and_controls_work(slow_client: TestClient) -> None:
    scan = start_scan(slow_client)
    scan_id = scan["id"]
    try:
        assert_error(
            slow_client.post("/api/v1/scans", json=scan_config()), 409, "ScanConflictError"
        )
        assert_error(
            slow_client.post(f"/api/v1/scans/{scan_id}/reconstruct"), 409, "ScanConflictError"
        )
        assert_error(
            slow_client.post(f"/api/v1/scans/{scan_id}/ml/analyse"), 409, "ScanConflictError"
        )
        assert_error(slow_client.post(f"/api/v1/scans/{scan_id}/resume"), 409, "ScanStateError")

        wait_for_scan(slow_client, scan_id, {"scanning"})
        response = slow_client.post(f"/api/v1/scans/{scan_id}/pause")
        assert response.status_code == 200
        paused = wait_for_scan(slow_client, scan_id, {"paused"})
        completed = paused["completed_points"]
        assert completed < paused["total_points"]
        status = slow_client.get("/api/v1/system/status").json()
        assert status["active_scan_id"] == scan_id
        assert status["active_scan_state"] == "paused"

        response = slow_client.post(f"/api/v1/scans/{scan_id}/resume")
        assert response.status_code == 200
        wait_for_points(slow_client, scan_id, completed + 1)
    finally:
        response = slow_client.post(f"/api/v1/scans/{scan_id}/cancel")
    assert response.status_code == 200
    final = wait_for_scan(slow_client, scan_id)
    assert final["state"] == "cancelled"
    assert final["interrupted"] is True
    assert slow_client.get("/api/v1/system/status").json()["active_scan_id"] is None
