"""ML inference models.

ML is advisory only: it never modifies the physics baseline (peak-derived
surface heights). Models are trained offline on a PC and deployed to the Pi
for inference.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from confocal.models.common import utc_now


class MLTask(StrEnum):
    BAD_POINT = "bad_point"  # binary: is this measured point unreliable?
    CONFIDENCE = "confidence"  # regression: predicted confidence 0..1
    OUTLIER = "outlier"  # multiclass noise/outlier classification


class MLModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    version: str
    task: MLTask
    algorithm: str = Field(description="e.g. RandomForestClassifier, ExtraTreesRegressor.")
    feature_names: list[str]
    trained_at: datetime | None = None
    sklearn_version: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    description: str | None = None


class MLAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str | None = Field(default=None, description="None selects the default model.")
    threshold: float = Field(0.5, ge=0, le=1, description="Flag points above this probability.")


class MLPointPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point_id: int
    label: str
    flagged: bool
    probability_bad: float | None = None
    predicted_confidence: float | None = None


class MLResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: int | None = None
    scan_id: str
    model: MLModelInfo
    created_at: datetime = Field(default_factory=utc_now)
    threshold: float
    n_points: int
    n_flagged: int
    predictions: list[MLPointPrediction]
    advisory: bool = Field(
        default=True, description="Always true: ML never replaces the physics baseline."
    )
