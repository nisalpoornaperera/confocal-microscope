"""Advisory ML inference over a finished scan (implements ``MLAnalyser``).

ML never changes the physics baseline: predictions are returned as a
separate :class:`~confocal.models.ml.MLResult` (``advisory=True``) and the
input points are not modified. Only inference runs here; training is the
offline :mod:`confocal.ml.training`.

Tasks:

``BAD_POINT``
    A binary classifier whose positive class (1 / True) means "this point's
    surface height is unreliable". ``probability_bad`` = P(positive);
    ``flagged`` when it is at or above ``request.threshold``.
``CONFIDENCE``
    A regressor predicting a confidence in [0, 1] (clipped).
    ``1 - predicted_confidence`` plays the role of the probability of being
    bad, so ``flagged`` when it is at or above ``request.threshold``.

Model selection: ``request.model_name``, else the service's
``default_model``, else the most recently trained compatible model.

All methods do blocking file I/O and CPU work (``analyse`` unpickles the
model on first use): async callers run them via ``asyncio.to_thread``.
Loaded models are cached per name and reloaded when ``model.joblib`` changes.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from confocal.errors import ModelNotAvailableError
from confocal.ml.features import build_feature_matrix
from confocal.ml.registry import (
    MODEL_FILE,
    LoadedModel,
    ModelRegistry,
    compatibility_problem,
    is_valid_model_name,
)
from confocal.models.ml import (
    MLAnalysisRequest,
    MLModelInfo,
    MLPointPrediction,
    MLResult,
    MLTask,
)
from confocal.models.scan import ScanPoint

LABEL_BAD = "bad"
LABEL_GOOD = "good"
LABEL_UNKNOWN = "unknown"  # the model produced a non-finite output for the point

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class MLService:
    """Model registry access plus inference; safe to call from worker threads."""

    def __init__(self, models_dir: Path, default_model: str | None = None) -> None:
        if default_model is not None and not is_valid_model_name(default_model):
            raise ValueError(f"invalid default model name {default_model!r}")
        self._registry = ModelRegistry(models_dir)
        self._default = default_model
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[int, LoadedModel]] = {}

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def list_models(self) -> list[MLModelInfo]:
        """Every model in the models directory (compatible or not), sorted by name."""
        return self._registry.list_models()

    def _compatible(self) -> list[MLModelInfo]:
        return [m for m in self._registry.list_models() if compatibility_problem(m) is None]

    def _resolve(self, name: str | None) -> MLModelInfo:
        """Metadata of the requested, default or newest compatible model."""
        chosen = name if name is not None else self._default
        if chosen is not None:
            info = self._registry.load_info(chosen)
            problem = compatibility_problem(info)
            if problem is not None:
                raise ModelNotAvailableError(f"model {chosen!r} is not usable: {problem}")
            return info
        candidates = self._compatible()
        if not candidates:
            raise ModelNotAvailableError(
                f"no compatible ML model is deployed in {self._registry.root}"
            )
        return max(candidates, key=lambda m: (m.trained_at or _EPOCH, m.name))

    def has_model(self, name: str | None = None) -> bool:
        """True when ``analyse`` would find a compatible model (metadata check only)."""
        try:
            self._resolve(name)
        except ModelNotAvailableError:
            return False
        return True

    def _load(self, name: str) -> LoadedModel:
        path = self._registry.model_dir(name) / MODEL_FILE
        try:
            stamp = path.stat().st_mtime_ns
        except OSError as exc:
            raise ModelNotAvailableError(f"model {name!r} has no {MODEL_FILE}") from exc
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None and cached[0] == stamp:
                return cached[1]
            loaded = self._registry.load(name)
            self._cache[name] = (stamp, loaded)
            return loaded

    def analyse(
        self, scan_id: str, points: Sequence[ScanPoint], request: MLAnalysisRequest
    ) -> MLResult:
        """Predict per-point reliability for one scan (advisory).

        Raises:
            ModelNotAvailableError: no (compatible, loadable) model.
        """
        info = self._resolve(request.model_name)
        model = self._load(info.name)
        features = build_feature_matrix(points)
        if model.info.task is MLTask.BAD_POINT:
            predictions = _bad_point_predictions(model, points, features, request.threshold)
        else:
            predictions = _confidence_predictions(model, points, features, request.threshold)
        return MLResult(
            scan_id=scan_id,
            model=model.info,
            threshold=request.threshold,
            n_points=len(predictions),
            n_flagged=sum(1 for p in predictions if p.flagged),
            predictions=predictions,
        )


def _positive_column(estimator: object) -> int:
    """Column of ``predict_proba`` that belongs to the "bad" class (1 / True)."""
    classes = list(getattr(estimator, "classes_", []))
    for i, cls in enumerate(classes):
        if cls in (1, True) or str(cls).lower() in {"1", "true", LABEL_BAD}:
            return i
    raise ModelNotAvailableError(
        f"bad-point classifier has no positive class among its classes {classes}"
    )


def _prediction(
    point: ScanPoint,
    probability_bad: float,
    threshold: float,
    *,
    confidence: float | None,
    report_probability: bool,
) -> MLPointPrediction:
    if not math.isfinite(probability_bad):
        return MLPointPrediction(point_id=point.point_id, label=LABEL_UNKNOWN, flagged=False)
    p = min(1.0, max(0.0, probability_bad))
    flagged = p >= threshold
    return MLPointPrediction(
        point_id=point.point_id,
        label=LABEL_BAD if flagged else LABEL_GOOD,
        flagged=flagged,
        probability_bad=p if report_probability else None,
        predicted_confidence=confidence,
    )


def _bad_point_predictions(
    model: LoadedModel,
    points: Sequence[ScanPoint],
    features: NDArray[np.float64],
    threshold: float,
) -> list[MLPointPrediction]:
    if not points:
        return []
    column = _positive_column(model.estimator)
    proba = np.asarray(model.estimator.predict_proba(features), dtype=np.float64)[:, column]
    return [
        _prediction(p, float(q), threshold, confidence=None, report_probability=True)
        for p, q in zip(points, proba, strict=True)
    ]


def _confidence_predictions(
    model: LoadedModel,
    points: Sequence[ScanPoint],
    features: NDArray[np.float64],
    threshold: float,
) -> list[MLPointPrediction]:
    if not points:
        return []
    raw = np.asarray(model.estimator.predict(features), dtype=np.float64).reshape(-1)
    out: list[MLPointPrediction] = []
    for p, value in zip(points, raw, strict=True):
        conf = min(1.0, max(0.0, float(value))) if math.isfinite(value) else None
        bad = math.nan if conf is None else 1.0 - conf
        out.append(_prediction(p, bad, threshold, confidence=conf, report_probability=False))
    return out
