"""OFFLINE model training (run on a PC; never imported by the server).

Usage::

    python -m confocal.ml.training --task bad_point --algorithm random_forest \\
        --synthetic 20000 --output models/
    python -m confocal.ml.training --task confidence --features scans.npz --output models/

``--output`` is the models directory; the model is written to
``<output>/<name>/`` (see :mod:`confocal.ml.registry`) and can be copied to
the Pi's models directory as is.

Training data
-------------
``--synthetic N``
    N points on a synthetic scan grid with known true heights. Each point gets
    a fine-sweep I(Z) profile -- a Gaussian axial response plus background,
    read noise and a dark offset -- drawn from a mix of realistic failure
    modes: low reflectivity (weak signal), spurious secondary peaks
    (internal reflections / dust), detector saturation (the OPT101 clips
    easily), a surface near or outside the sweep (coarse-peak error) and
    varying noise. Every profile is analysed with the production
    :func:`~confocal.processing.profile.analyse_profile`, so the model learns
    exactly the failure modes of the physics baseline.
``--features FILE.npz``
    Pre-computed data: ``X`` (n, len(FEATURE_NAMES)), ``y`` (n,) and
    optionally ``feature_names`` (must equal FEATURE_NAMES).

Labels
------
* ``bad_point``: 1 when the physics baseline produced no surface height or
  ``|surface_z - true_z| > --tolerance-um``, else 0.
* ``confidence``: ``exp(-(error / tolerance)^2 / 2)`` -- 1 for a perfect
  height, 0.61 at the tolerance, 0 without a height.

The data are split into train / test sets (stratified for ``bad_point``);
test metrics are stored in ``metadata.json``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from confocal.ml.features import FEATURE_NAMES, build_feature_matrix
from confocal.ml.registry import ModelRegistry
from confocal.models.ml import MLModelInfo, MLTask
from confocal.models.processing import PeakFit, ProcessingConfig
from confocal.models.scan import ScanPoint
from confocal.processing.profile import analyse_profile

ALGORITHMS = ("random_forest", "extra_trees", "gradient_boosting")
TASKS = ("bad_point", "confidence")

_ESTIMATORS: dict[tuple[str, str], Any] = {
    ("random_forest", "bad_point"): RandomForestClassifier,
    ("random_forest", "confidence"): RandomForestRegressor,
    ("extra_trees", "bad_point"): ExtraTreesClassifier,
    ("extra_trees", "confidence"): ExtraTreesRegressor,
    ("gradient_boosting", "bad_point"): GradientBoostingClassifier,
    ("gradient_boosting", "confidence"): GradientBoostingRegressor,
}


# --------------------------------------------------------------------------- synthetic data


@dataclass(frozen=True, slots=True)
class SyntheticProfileConfig:
    """Physical parameters of the synthetic profiles (volts, micrometres)."""

    fine_range_um: float = 12.0
    fine_step_um: float = 0.25
    axial_fwhm_um: float = 2.8
    dark_v: float = 0.02
    background_v: float = 0.03
    peak_v: float = 1.0
    rail_v: float = 2.0  # OPT101 output saturation
    noise_v: tuple[float, float] = (0.003, 0.03)
    centring_error_um: float = 1.0  # std of the coarse-peak error
    p_low_signal: float = 0.15
    p_secondary: float = 0.2
    p_saturated: float = 0.15
    p_off_centre: float = 0.08


@dataclass(frozen=True, slots=True)
class LabelledData:
    features: NDArray[np.float64]
    target: NDArray[np.float64]
    points: list[ScanPoint]
    true_z_um: NDArray[np.float64]


def _true_height(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """A tilted, gently corrugated sample surface (micrometres)."""
    return np.asarray(
        0.02 * x - 0.015 * y + 1.5 * np.sin(x / 25.0) * np.cos(y / 35.0), dtype=np.float64
    )


def _fit_center(fit: PeakFit | None) -> float | None:
    return fit.center_um if fit is not None and fit.success else None


def _profile_voltage(
    z: NDArray[np.float64],
    true_z: float,
    rng: np.random.Generator,
    cfg: SyntheticProfileConfig,
) -> NDArray[np.float64]:
    sigma = cfg.axial_fwhm_um / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    amplitude = cfg.peak_v * rng.uniform(0.6, 1.4)
    if rng.random() < cfg.p_low_signal:
        amplitude *= rng.uniform(0.005, 0.08)
    if rng.random() < cfg.p_saturated:
        amplitude *= rng.uniform(2.2, 4.0)
    signal = amplitude * np.exp(-0.5 * ((z - true_z) / sigma) ** 2)
    if rng.random() < cfg.p_secondary:
        offset = rng.choice((-1.0, 1.0)) * rng.uniform(2.5, 5.5)
        ratio = rng.uniform(0.3, 1.3)
        signal += ratio * amplitude * np.exp(-0.5 * ((z - true_z - offset) / sigma) ** 2)
    noise = rng.uniform(*cfg.noise_v)
    v = cfg.dark_v + cfg.background_v + signal + rng.normal(0.0, noise, z.shape[0])
    return np.asarray(np.clip(v, 0.0, cfg.rail_v), dtype=np.float64)


def synthetic_points(
    n_points: int,
    *,
    seed: int = 0,
    xy_step_um: float = 2.0,
    profile: SyntheticProfileConfig | None = None,
    processing: ProcessingConfig | None = None,
) -> tuple[list[ScanPoint], NDArray[np.float64]]:
    """Analysed synthetic scan points (row-major grid) and their true heights."""
    cfg = profile or SyntheticProfileConfig()
    proc = processing or ProcessingConfig(
        saturation_v=0.999 * cfg.rail_v, expected_fwhm_um=cfg.axial_fwhm_um
    )
    rng = np.random.default_rng(seed)
    nx = max(1, math.ceil(math.sqrt(n_points)))
    ix = np.arange(n_points) % nx
    iy = np.arange(n_points) // nx
    x, y = ix * xy_step_um, iy * xy_step_um
    true_z = _true_height(x.astype(np.float64), y.astype(np.float64))
    half = 0.5 * cfg.fine_range_um
    offsets = np.arange(-half, half + 0.5 * cfg.fine_step_um, cfg.fine_step_um)

    points: list[ScanPoint] = []
    for i in range(n_points):
        error = rng.normal(0.0, cfg.centring_error_um)
        if rng.random() < cfg.p_off_centre:
            error = rng.choice((-1.0, 1.0)) * rng.uniform(0.4, 0.8) * cfg.fine_range_um
        centre = float(true_z[i]) + error
        z = centre + offsets
        a = analyse_profile(
            z,
            _profile_voltage(z, float(true_z[i]), rng, cfg),
            dark_v=cfg.dark_v,
            reference_v=None,
            config=proc,
        ).analysis
        points.append(
            ScanPoint(
                point_id=i,
                ix=int(ix[i]),
                iy=int(iy[i]),
                x_um=float(x[i]),
                y_um=float(y[i]),
                status=a.status,
                z_estimate_um=centre,
                coarse_peak_z_um=centre,
                surface_z_um=a.surface_z_um,
                parabolic_z_um=_fit_center(a.parabolic),
                gaussian_z_um=_fit_center(a.gaussian),
                peak_intensity=a.peak_intensity,
                snr=a.snr,
                peak_width_um=a.fwhm_um,
                prominence=a.prominence,
                fit_residual=a.fit_residual,
                confidence=a.confidence,
                secondary_peak_ratio=a.secondary_peak_ratio,
                asymmetry=a.asymmetry,
                n_z_positions=int(z.shape[0]),
                flags=list(a.flags),
            )
        )
    return points, true_z


def height_errors(
    points: Sequence[ScanPoint], true_z_um: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``|surface_z - true_z|`` per point; +inf where there is no surface height."""
    measured = np.array(
        [math.inf if p.surface_z_um is None else p.surface_z_um for p in points], dtype=np.float64
    )
    return np.asarray(np.abs(measured - true_z_um), dtype=np.float64)


def labels_for(task: str, errors: NDArray[np.float64], tolerance_um: float) -> NDArray[np.float64]:
    """Training target for ``task`` from the absolute height errors (see module docstring)."""
    if task == "bad_point":
        return np.asarray(~(errors <= tolerance_um), dtype=np.float64)
    with np.errstate(over="ignore"):
        quality = np.exp(-0.5 * (errors / tolerance_um) ** 2)
    return np.asarray(np.where(np.isfinite(errors), quality, 0.0), dtype=np.float64)


def synthetic_dataset(
    n_points: int, task: str, *, seed: int = 0, tolerance_um: float = 0.5
) -> LabelledData:
    points, true_z = synthetic_points(n_points, seed=seed)
    target = labels_for(task, height_errors(points, true_z), tolerance_um)
    return LabelledData(build_feature_matrix(points), target, points, true_z)


def load_feature_file(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(X, y)`` from an ``.npz`` file (see the module docstring).

    Raises:
        ValueError: missing arrays, wrong shapes, non-finite values or different features.
    """
    with np.load(path, allow_pickle=False) as data:
        if "X" not in data or "y" not in data:
            raise ValueError(f"{path} must contain arrays 'X' and 'y'")
        features = np.asarray(data["X"], dtype=np.float64)
        target = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        names = tuple(str(n) for n in data["feature_names"]) if "feature_names" in data else None
    if names is not None and names != FEATURE_NAMES:
        raise ValueError(f"{path}: feature_names differ from FEATURE_NAMES")
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"{path}: X must have shape (n, {len(FEATURE_NAMES)})")
    if features.shape[0] != target.shape[0]:
        raise ValueError(f"{path}: X and y have different lengths")
    if not (np.isfinite(features).all() and np.isfinite(target).all()):
        raise ValueError(f"{path}: X and y must be finite")
    return features, target


# --------------------------------------------------------------------------- training


def make_estimator(algorithm: str, task: str, *, n_estimators: int, seed: int) -> Any:
    """Unfitted scikit-learn estimator for ``algorithm`` x ``task``."""
    try:
        cls = _ESTIMATORS[(algorithm, task)]
    except KeyError as exc:
        raise ValueError(f"unknown algorithm/task {algorithm!r}/{task!r}") from exc
    if algorithm == "gradient_boosting":
        return cls(n_estimators=n_estimators, random_state=seed)
    extra = {"class_weight": "balanced"} if task == "bad_point" else {}
    return cls(n_estimators=n_estimators, random_state=seed, n_jobs=-1, min_samples_leaf=2, **extra)


def evaluate(
    task: str, estimator: Any, x_test: NDArray[np.float64], y_test: NDArray[np.float64]
) -> dict[str, float]:
    """Test-set metrics stored in the model metadata."""
    metrics: dict[str, float] = {"n_test": float(y_test.shape[0])}
    if task == "bad_point":
        predicted = np.asarray(estimator.predict(x_test), dtype=np.float64)
        metrics["accuracy"] = float(accuracy_score(y_test, predicted))
        metrics["precision"] = float(precision_score(y_test, predicted, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, predicted, zero_division=0))
        metrics["f1"] = float(f1_score(y_test, predicted, zero_division=0))
        metrics["bad_fraction"] = float(np.mean(y_test))
        if np.unique(y_test).size == 2:
            proba = np.asarray(estimator.predict_proba(x_test), dtype=np.float64)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    else:
        predicted = np.clip(np.asarray(estimator.predict(x_test), dtype=np.float64), 0.0, 1.0)
        metrics["mae"] = float(mean_absolute_error(y_test, predicted))
        if y_test.shape[0] > 1 and float(np.var(y_test)) > 0.0:
            metrics["r2"] = float(r2_score(y_test, predicted))
    return {k: v for k, v in metrics.items() if math.isfinite(v)}


@dataclass(frozen=True, slots=True)
class TrainingResult:
    info: MLModelInfo
    directory: Path
    estimator: Any


def train(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    task: str,
    algorithm: str,
    output: Path,
    name: str,
    version: str = "1",
    n_estimators: int = 200,
    test_fraction: float = 0.25,
    seed: int = 0,
    description: str | None = None,
    overwrite: bool = False,
) -> TrainingResult:
    """Fit, evaluate on a held-out split and save the model to the registry at ``output``.

    Raises:
        ValueError: unusable data (too few samples, a single class) or bad arguments.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    if features.shape[0] < 10:
        raise ValueError("need at least 10 samples to train and test")
    y = target.astype(np.int64) if task == "bad_point" else target
    stratify = None
    if task == "bad_point":
        classes, counts = np.unique(y, return_counts=True)
        if classes.size < 2:
            raise ValueError("bad_point training needs both good and bad samples")
        stratify = y if int(counts.min()) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=test_fraction, random_state=seed, stratify=stratify
    )
    estimator = make_estimator(algorithm, task, n_estimators=n_estimators, seed=seed)
    estimator.fit(x_train, y_train)
    metrics = evaluate(task, estimator, x_test, np.asarray(y_test, dtype=np.float64))
    metrics["n_train"] = float(x_train.shape[0])
    info = MLModelInfo(
        name=name,
        version=version,
        task=MLTask(task),
        algorithm=type(estimator).__name__,
        feature_names=list(FEATURE_NAMES),
        trained_at=datetime.now(UTC),
        sklearn_version=sklearn.__version__,
        metrics=metrics,
        description=description,
    )
    directory = ModelRegistry(output).save(estimator, info, overwrite=overwrite)
    return TrainingResult(info=info, directory=directory, estimator=estimator)


# --------------------------------------------------------------------------- CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m confocal.ml.training",
        description="Train an advisory ML model offline (bad-point classifier or confidence "
        "regressor) and save it to a models directory.",
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="random_forest")
    parser.add_argument("--task", choices=TASKS, default="bad_point")
    parser.add_argument("--output", type=Path, required=True, help="Models directory.")
    parser.add_argument("--name", help="Model name (default: <task>-<algorithm>).")
    parser.add_argument("--version", default="1")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", type=int, metavar="N", help="Generate N synthetic points.")
    source.add_argument("--features", type=Path, metavar="FILE.npz", help="Load X / y arrays.")
    parser.add_argument("--tolerance-um", type=float, default=0.5)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.synthetic is not None:
        if args.synthetic < 10:
            print("--synthetic needs at least 10 points", file=sys.stderr)
            return 2
        data = synthetic_dataset(
            args.synthetic, args.task, seed=args.seed, tolerance_um=args.tolerance_um
        )
        features, target = data.features, data.target
        source = f"{args.synthetic} synthetic points (tolerance {args.tolerance_um} um)"
    else:
        try:
            features, target = load_feature_file(args.features)
        except (OSError, ValueError) as exc:
            print(f"cannot read {args.features}: {exc}", file=sys.stderr)
            return 2
        source = str(args.features)
    try:
        result = train(
            features,
            target,
            task=args.task,
            algorithm=args.algorithm,
            output=args.output,
            name=args.name or f"{args.task}-{args.algorithm}",
            version=args.version,
            n_estimators=args.n_estimators,
            test_fraction=args.test_fraction,
            seed=args.seed,
            description=f"trained on {source}",
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1
    metrics = ", ".join(f"{k}={v:.4g}" for k, v in sorted(result.info.metrics.items()))
    print(f"saved {result.info.name} ({result.info.algorithm}) to {result.directory}: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
