"""Advisory ML endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from tests.unit.api.helpers import assert_error, running_client, start_scan, wait_for_scan

from confocal.config import Settings
from confocal.ml import FEATURE_NAMES, ModelRegistry
from confocal.ml.training import synthetic_dataset
from confocal.models import MLModelInfo, MLResult, MLTask

UNKNOWN = "0" * 32


def deploy_model(models_dir: Path, name: str = "bad-points") -> None:
    data = synthetic_dataset(120, "bad_point", seed=5)
    estimator = RandomForestClassifier(n_estimators=8, random_state=0)
    estimator.fit(data.features, data.target)
    ModelRegistry(models_dir).save(
        estimator,
        MLModelInfo(
            name=name,
            version="1",
            task=MLTask.BAD_POINT,
            algorithm="RandomForestClassifier",
            feature_names=list(FEATURE_NAMES),
        ),
    )


def test_no_models_deployed(client: TestClient) -> None:
    assert client.get("/api/v1/ml/models").json() == []
    scan = start_scan(client)
    wait_for_scan(client, scan["id"])
    assert_error(
        client.post(f"/api/v1/scans/{scan['id']}/ml/analyse", json={}),
        404,
        "ModelNotAvailableError",
    )
    assert_error(client.get(f"/api/v1/scans/{scan['id']}/ml/results"), 404, "MLResultNotFound")


def test_unknown_scan(client: TestClient) -> None:
    assert_error(client.get(f"/api/v1/scans/{UNKNOWN}/ml/results"), 404, "ScanNotFoundError")


def test_analyse_with_deployed_model(sim_settings: Settings) -> None:
    deploy_model(sim_settings.models_dir)
    with running_client(sim_settings) as client:
        models = [MLModelInfo.model_validate(m) for m in client.get("/api/v1/ml/models").json()]
        assert [m.name for m in models] == ["bad-points"]

        scan = start_scan(client)
        assert wait_for_scan(client, scan["id"])["state"] == "complete"
        points_before = client.get(f"/api/v1/scans/{scan['id']}/points").json()

        response = client.post(f"/api/v1/scans/{scan['id']}/ml/analyse", json={"threshold": 0.4})
        assert response.status_code == 200
        result = MLResult.model_validate(response.json())
        assert result.advisory is True
        assert result.n_points == 20
        assert result.threshold == 0.4
        assert result.result_id is not None

        latest = MLResult.model_validate(
            client.get(f"/api/v1/scans/{scan['id']}/ml/results").json()
        )
        assert latest.result_id == result.result_id
        # ML is advisory: physics results are untouched
        assert client.get(f"/api/v1/scans/{scan['id']}/points").json() == points_before

        assert_error(
            client.post(f"/api/v1/scans/{scan['id']}/ml/analyse", json={"model_name": "missing"}),
            404,
            "ModelNotAvailableError",
        )
        assert_error(
            client.post(f"/api/v1/scans/{scan['id']}/ml/analyse", json={"threshold": 2}),
            422,
            "RequestValidationError",
        )


def test_ml_on_complete_runs_with_a_model(sim_settings: Settings) -> None:
    deploy_model(sim_settings.models_dir)
    with running_client(sim_settings) as client:
        scan = start_scan(client, ml_on_complete=True)
        summary = wait_for_scan(client, scan["id"])
        assert summary["state"] == "complete"
        assert summary["has_ml_result"] is True
        assert client.get(f"/api/v1/scans/{scan['id']}/ml/results").status_code == 200
