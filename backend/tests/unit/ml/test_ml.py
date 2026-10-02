"""Features, registry, inference and the offline training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from confocal.errors import ModelNotAvailableError
from confocal.ml import FEATURE_NAMES, MLService, ModelRegistry, build_feature_matrix
from confocal.ml.registry import METADATA_FILE, MODEL_FILE
from confocal.ml.training import labels_for, load_feature_file, main, synthetic_dataset, train
from confocal.models.ml import MLAnalysisRequest, MLModelInfo, MLTask
from confocal.models.processing import PointStatus
from confocal.models.scan import ScanPoint
from confocal.scanning.protocols import MLAnalyser

COLUMN = {name: i for i, name in enumerate(FEATURE_NAMES)}


def grid_points(n: int = 4) -> list[ScanPoint]:
    points = []
    for iy in range(n):
        for ix in range(n):
            points.append(
                ScanPoint(
                    point_id=iy * n + ix,
                    ix=ix,
                    iy=iy,
                    x_um=2.0 * ix,
                    y_um=2.0 * iy,
                    status=PointStatus.VALID,
                    coarse_peak_z_um=1.0,
                    surface_z_um=1.0,
                    parabolic_z_um=1.1,
                    gaussian_z_um=1.0,
                    peak_intensity=0.9,
                    snr=40.0,
                    peak_width_um=2.8,
                    prominence=0.8,
                    fit_residual=0.01,
                    confidence=0.95,
                    secondary_peak_ratio=0.0,
                    asymmetry=0.01,
                )
            )
    return points


def info(name: str, task: MLTask, **kw: object) -> MLModelInfo:
    return MLModelInfo(
        name=name,
        version="1",
        task=task,
        algorithm="RandomForest",
        feature_names=list(FEATURE_NAMES),
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def dataset() -> tuple[np.ndarray, np.ndarray, list[ScanPoint]]:
    data = synthetic_dataset(240, "bad_point", seed=3)
    return data.features, data.target, data.points


@pytest.fixture
def models_dir(tmp_path: Path, dataset: tuple[np.ndarray, np.ndarray, list[ScanPoint]]) -> Path:
    features, target, _ = dataset
    registry = ModelRegistry(tmp_path / "models")
    clf = RandomForestClassifier(n_estimators=15, random_state=0).fit(features, target)
    registry.save(clf, info("bad", MLTask.BAD_POINT))
    reg = RandomForestRegressor(n_estimators=15, random_state=0).fit(features, 1.0 - target)
    registry.save(reg, info("conf", MLTask.CONFIDENCE))
    return registry.root


# --------------------------------------------------------------------------- features


def test_feature_matrix_shape_and_finiteness() -> None:
    points = grid_points()
    points[5] = points[5].model_copy(
        update={
            "status": PointStatus.NO_PEAK,
            "surface_z_um": None,
            "gaussian_z_um": None,
            "snr": None,
            "peak_width_um": float("nan"),
        }
    )
    points[6] = points[6].model_copy(update={"surface_z_um": 4.0})
    before = [p.model_dump() for p in points]
    x = build_feature_matrix(points)
    assert x.shape == (16, len(FEATURE_NAMES))
    assert np.isfinite(x).all()
    assert [p.model_dump() for p in points] == before
    assert x[5, COLUMN["surface_z_missing"]] == 1.0
    assert x[5, COLUMN["snr_missing"]] == 1.0
    assert x[5, COLUMN["peak_width_missing"]] == 1.0
    assert x[5, COLUMN["fit_disagreement_missing"]] == 1.0
    assert x[0, COLUMN["fit_disagreement_um"]] == pytest.approx(0.1)
    assert x[6, COLUMN["surface_z_deviation_um"]] == pytest.approx(3.0)
    # Corners have 3 grid neighbours, interior points 8; point 5 has no usable height.
    assert x[15, COLUMN["neighbour_count"]] == 3.0
    assert x[0, COLUMN["neighbour_count"]] == 2.0
    assert x[10, COLUMN["neighbour_count"]] == 7.0
    assert x[10, COLUMN["neighbour_z_std_um"]] > 0.0


def test_feature_matrix_of_no_points_and_isolated_point() -> None:
    assert build_feature_matrix([]).shape == (0, len(FEATURE_NAMES))
    x = build_feature_matrix(grid_points(1))
    assert x[0, COLUMN["neighbour_count"]] == 0.0
    assert x[0, COLUMN["neighbour_z_missing"]] == 1.0
    assert np.isfinite(x).all()


# --------------------------------------------------------------------------- registry


def test_registry_round_trip(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    clf = RandomForestClassifier(n_estimators=2, random_state=0).fit(
        np.zeros((4, len(FEATURE_NAMES))), [0, 1, 0, 1]
    )
    meta = info("m1", MLTask.BAD_POINT, metrics={"f1": 0.5})
    directory = registry.save(clf, meta)
    assert (directory / MODEL_FILE).is_file()
    assert json.loads((directory / METADATA_FILE).read_text())["name"] == "m1"
    assert registry.list_models() == [meta]
    loaded = registry.load("m1")
    assert loaded.info == meta
    assert list(loaded.estimator.classes_) == [0, 1]
    with pytest.raises(ValueError, match="exists"):
        registry.save(clf, meta)
    registry.save(clf, meta, overwrite=True)


def test_registry_rejects_unsafe_names_and_broken_entries(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    with pytest.raises(ValueError, match="invalid"):
        registry.model_dir("../evil")
    with pytest.raises(ModelNotAvailableError):
        registry.load_info("../evil")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / METADATA_FILE).write_text("{not json")
    assert registry.list_models() == []
    with pytest.raises(ModelNotAvailableError, match="not found"):
        registry.load("missing")


def test_registry_refuses_incompatible_features(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    meta = info("old", MLTask.BAD_POINT).model_copy(
        update={"feature_names": [*FEATURE_NAMES[:-1], "legacy_feature"]}
    )
    registry.save(object(), meta)
    with pytest.raises(ModelNotAvailableError, match="feature names"):
        registry.load("old")
    assert MLService(tmp_path).has_model("old") is False


# --------------------------------------------------------------------------- inference


def test_service_implements_protocol_and_empty_dir(tmp_path: Path) -> None:
    service = MLService(tmp_path / "nothing")
    assert isinstance(service, MLAnalyser)
    assert service.has_model() is False
    assert service.list_models() == []
    with pytest.raises(ModelNotAvailableError):
        service.analyse("s", grid_points(), MLAnalysisRequest())


def test_bad_point_inference(
    models_dir: Path, dataset: tuple[np.ndarray, np.ndarray, list[ScanPoint]]
) -> None:
    _, target, points = dataset
    service = MLService(models_dir, default_model="bad")
    assert service.has_model()
    assert service.has_model("bad")
    assert not service.has_model("unknown")
    before = [p.model_dump() for p in points]
    result = service.analyse("scan-1", points, MLAnalysisRequest(threshold=0.5))
    assert [p.model_dump() for p in points] == before
    assert result.advisory
    assert result.model.name == "bad"
    assert result.n_points == len(points)
    assert result.n_flagged == sum(p.flagged for p in result.predictions)
    probs = np.array([p.probability_bad for p in result.predictions], dtype=float)
    assert ((probs >= 0) & (probs <= 1)).all()
    flagged = np.array([p.flagged for p in result.predictions])
    np.testing.assert_array_equal(flagged, probs >= 0.5)
    assert np.mean(flagged == target.astype(bool)) > 0.9  # training data: near perfect
    assert {p.label for p in result.predictions} <= {"bad", "good"}
    assert "NaN" not in result.model_dump_json()
    strict = service.analyse("scan-1", points, MLAnalysisRequest(threshold=1.0))
    assert strict.n_flagged <= result.n_flagged


def test_confidence_inference_and_model_selection(
    models_dir: Path, dataset: tuple[np.ndarray, np.ndarray, list[ScanPoint]]
) -> None:
    _, _, points = dataset
    service = MLService(models_dir)
    result = service.analyse("s", points[:20], MLAnalysisRequest(model_name="conf"))
    assert result.model.task is MLTask.CONFIDENCE
    for p in result.predictions:
        assert p.probability_bad is None
        assert p.predicted_confidence is not None
        assert 0.0 <= p.predicted_confidence <= 1.0
        assert p.flagged == (1.0 - p.predicted_confidence >= 0.5)
    empty = service.analyse("s", [], MLAnalysisRequest(model_name="bad"))
    assert empty.n_points == 0
    with pytest.raises(ModelNotAvailableError):
        service.analyse("s", points, MLAnalysisRequest(model_name="nope"))


def test_default_model_is_newest_compatible(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    registry = ModelRegistry(tmp_path)
    x = np.zeros((4, len(FEATURE_NAMES)))
    clf = RandomForestClassifier(n_estimators=2, random_state=0).fit(x, [0, 1, 0, 1])
    registry.save(clf, info("a-old", MLTask.BAD_POINT, trained_at=datetime(2025, 1, 1, tzinfo=UTC)))
    registry.save(clf, info("b-new", MLTask.BAD_POINT, trained_at=datetime(2026, 1, 1, tzinfo=UTC)))
    result = MLService(tmp_path).analyse("s", grid_points(2), MLAnalysisRequest())
    assert result.model.name == "b-new"


# --------------------------------------------------------------------------- training


def test_labels() -> None:
    errors = np.array([0.0, 0.4, 0.6, np.inf])
    np.testing.assert_array_equal(labels_for("bad_point", errors, 0.5), [0, 0, 1, 1])
    conf = labels_for("confidence", errors, 0.5)
    assert conf[0] == 1.0
    assert conf[3] == 0.0
    assert np.all(np.diff(conf) < 0)


def test_synthetic_dataset_has_both_classes(
    dataset: tuple[np.ndarray, np.ndarray, list[ScanPoint]],
) -> None:
    features, target, points = dataset
    assert features.shape == (240, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert 0.05 < float(np.mean(target)) < 0.6
    assert {p.status for p in points} >= {PointStatus.VALID, PointStatus.NO_PEAK}


@pytest.mark.parametrize(
    ("algorithm", "task"),
    [
        ("random_forest", "bad_point"),
        ("extra_trees", "confidence"),
        ("gradient_boosting", "bad_point"),
    ],
)
def test_train_saves_a_usable_model(
    tmp_path: Path,
    dataset: tuple[np.ndarray, np.ndarray, list[ScanPoint]],
    algorithm: str,
    task: str,
) -> None:
    features, target, points = dataset
    y = target if task == "bad_point" else 1.0 - target
    result = train(
        features, y, task=task, algorithm=algorithm, output=tmp_path, name="m", n_estimators=10
    )
    assert result.info.metrics["n_test"] > 0
    assert ("f1" in result.info.metrics) == (task == "bad_point")
    analysis = MLService(tmp_path).analyse("s", points[:10], MLAnalysisRequest())
    assert analysis.n_points == 10


def test_train_rejects_single_class(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both"):
        train(
            np.zeros((20, len(FEATURE_NAMES))),
            np.zeros(20),
            task="bad_point",
            algorithm="random_forest",
            output=tmp_path,
            name="m",
        )


def test_cli_synthetic_and_feature_file(
    tmp_path: Path,
    dataset: tuple[np.ndarray, np.ndarray, list[ScanPoint]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "models"
    argv = ["--synthetic", "120", "--output", str(out), "--n-estimators", "5", "--name", "syn"]
    assert main(argv) == 0
    assert "saved syn" in capsys.readouterr().out
    assert ModelRegistry(out).load_info("syn").task is MLTask.BAD_POINT

    features, target, _ = dataset
    npz = tmp_path / "data.npz"
    np.savez(npz, X=features, y=1.0 - target, feature_names=np.array(FEATURE_NAMES))
    x, y = load_feature_file(npz)
    assert x.shape == features.shape
    assert y.shape == target.shape
    argv = ["--task", "confidence", "--algorithm", "extra_trees", "--features", str(npz)]
    assert main([*argv, "--output", str(out), "--n-estimators", "5"]) == 0
    assert ModelRegistry(out).load_info("confidence-extra_trees").task is MLTask.CONFIDENCE

    bad = tmp_path / "bad.npz"
    np.savez(bad, X=features[:, :3], y=target)
    assert main(["--features", str(bad), "--output", str(out)]) == 2


def test_training_is_not_imported_by_the_package() -> None:
    import subprocess
    import sys

    code = "import sys, confocal.ml; print('confocal.ml.training' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
