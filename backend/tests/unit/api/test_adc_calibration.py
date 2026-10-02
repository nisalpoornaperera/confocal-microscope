"""``/api/v1/adc`` and ``/api/v1/calibration``."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.unit.api.helpers import (
    assert_error,
    running_client,
    start_scan,
    wait_for_scan,
    with_simulation,
)

from confocal.config import HardwareSelection, Settings
from confocal.models import (
    ADCCalibrateResponse,
    ADCStatus,
    CalibrationState,
    IntensityMeasurement,
)


def test_adc_status(client: TestClient, sim_settings: Settings) -> None:
    response = client.get("/api/v1/adc/status")
    assert response.status_code == 200
    status = ADCStatus.model_validate(response.json())
    assert status.connected
    assert status.gain == sim_settings.ads1115.gain
    assert status.data_rate_sps == sim_settings.ads1115.data_rate_sps


@pytest.mark.parametrize("method", ["mean", "median"])
def test_adc_read(client: TestClient, method: str) -> None:
    response = client.get("/api/v1/adc/read", params={"n_samples": 8, "method": method})
    assert response.status_code == 200
    reading = IntensityMeasurement.model_validate(response.json())
    assert reading.n_samples == 8
    assert reading.method.value == method
    assert len(reading.raw_counts) == 8
    assert len(reading.voltages_v) == 8
    assert reading.dark_v is None  # not calibrated yet
    assert reading.normalized is None


@pytest.mark.parametrize(
    "params",
    [{"n_samples": 0}, {"n_samples": 1025}, {"method": "mode"}, {"n_samples": "many"}],
)
def test_adc_read_validation(client: TestClient, params: dict[str, object]) -> None:
    body = assert_error(
        client.get("/api/v1/adc/read", params=params), 422, "RequestValidationError"
    )
    assert any(v.startswith("query.") for v in body["violations"])


def test_adc_calibrate_explicit_and_auto(client: TestClient) -> None:
    response = client.post("/api/v1/adc/calibrate", json={"gain": "2"})
    assert response.status_code == 200
    explicit = ADCCalibrateResponse.model_validate(response.json())
    assert explicit.gain.value == "2"
    assert client.get("/api/v1/adc/status").json()["gain"] == "2"

    response = client.post("/api/v1/adc/calibrate", json={"auto": True})
    assert response.status_code == 200
    auto = ADCCalibrateResponse.model_validate(response.json())
    assert auto.measured_max_v is not None


def test_adc_calibrate_needs_exactly_one_option(client: TestClient) -> None:
    for payload in ({}, {"gain": "2", "auto": True}):
        assert_error(
            client.post("/api/v1/adc/calibrate", json=payload), 422, "RequestValidationError"
        )


def test_dark_reference_and_history(client: TestClient) -> None:
    assert CalibrationState.model_validate(client.get("/api/v1/calibration").json()).version is None
    assert client.get("/api/v1/calibration/history").json() == []

    response = client.post("/api/v1/calibration/dark", json={"n_samples": 32})
    assert response.status_code == 200
    dark = CalibrationState.model_validate(response.json())
    assert dark.version == 1
    assert dark.updated_field == "dark"
    assert dark.dark_v is not None
    assert dark.dark_n_samples == 32

    response = client.post("/api/v1/calibration/reference", json={"z_search": True})
    assert response.status_code == 200
    reference = CalibrationState.model_validate(response.json())
    assert reference.version == 2
    assert reference.reference_v is not None
    assert reference.dark_v == dark.dark_v
    assert reference.can_normalize

    current = CalibrationState.model_validate(client.get("/api/v1/calibration").json())
    assert current.version == 2
    history = [
        CalibrationState.model_validate(c) for c in client.get("/api/v1/calibration/history").json()
    ]
    assert [c.version for c in history] == [2, 1]
    assert len(client.get("/api/v1/calibration/history", params={"limit": 1}).json()) == 1
    assert client.get("/api/v1/system/status").json()["calibration_version"] == 2

    reading = IntensityMeasurement.model_validate(client.get("/api/v1/adc/read").json())
    assert reading.calibration_version == 2
    assert reading.normalized is not None


def test_dark_without_body_uses_defaults(client: TestClient) -> None:
    response = client.post("/api/v1/calibration/dark")
    assert response.status_code == 200
    assert response.json()["dark_n_samples"] == 64


def test_manual_laser_dark_needs_confirmation(sim_settings: Settings) -> None:
    settings = sim_settings.model_copy(update={"hardware": HardwareSelection(laser="manual")})
    with running_client(settings) as client:
        body = assert_error(
            client.post("/api/v1/calibration/dark", json={}), 409, "CalibrationError"
        )
        assert "beam_blocked_confirmed" in body["detail"]
        assert client.get("/api/v1/calibration/history").json() == []
        response = client.post("/api/v1/calibration/dark", json={"beam_blocked_confirmed": True})
        assert response.status_code == 200
        assert response.json()["version"] == 1
        # The simulated manual laser is always on, so that "dark" level was measured in
        # light: far out of focus the reference is below it and must be refused.
        assert client.post("/api/v1/stage/move", json={"z_um": -1500.0}).status_code == 200
        assert_error(client.post("/api/v1/calibration/reference"), 409, "CalibrationError")


def test_manual_operations_refused_during_scan(slow_settings: Settings) -> None:
    with running_client(slow_settings) as client:
        scan = start_scan(client)
        try:
            for method, path, payload in (
                ("get", "/api/v1/adc/read", None),
                ("post", "/api/v1/adc/calibrate", {"gain": "1"}),
                ("post", "/api/v1/calibration/dark", {}),
                ("post", "/api/v1/calibration/reference", {}),
            ):
                response = client.request(method, path, json=payload)
                assert_error(response, 409, "ScanConflictError")
            # read-only endpoints keep working
            assert client.get("/api/v1/adc/status").status_code == 200
            assert client.get("/api/v1/calibration").status_code == 200
            assert client.get("/api/v1/calibration/history").status_code == 200
        finally:
            client.post(f"/api/v1/scans/{scan['id']}/cancel")
        assert wait_for_scan(client, scan["id"])["state"] == "cancelled"
        assert client.get("/api/v1/adc/read").status_code == 200


def test_adc_fault_is_503(sim_settings: Settings) -> None:
    settings = with_simulation(sim_settings, adc_fault_after_reads=0)
    with running_client(settings) as client:
        assert_error(client.get("/api/v1/adc/read"), 503, "ADCError")
