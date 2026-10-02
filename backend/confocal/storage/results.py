"""HDF5 encoding of surface reconstructions and ML results.

::

    /surfaces/<id>/   x_um (nx,), y_um (ny,), z_um (ny, nx), confidence (ny, nx) float64
                      (NaN = no value), gap_mask (ny, nx) bool
                      attrs: surface_json = the SurfaceResult without those five grids
    /ml/<id>/         predictions: table (point_id, label, flagged, probability_bad,
                      predicted_confidence; NaN = no value)
                      attrs: ml_json = the MLResult without its predictions

Grids and predictions are datasets (compact, readable by any HDF5 tool); the
rest (request, point classification, mesh, cross-sections, gaps, statistics,
model identity) is the model's own JSON. Decoding is the exact inverse, so a
stored result compares equal to the one that was saved.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from confocal.errors import StorageError
from confocal.models.measurement import nan_to_none
from confocal.models.ml import MLPointPrediction, MLResult
from confocal.models.surface import SurfaceResult
from confocal.storage.layout import STRING_DTYPE

SURFACE_JSON_ATTR: Final = "surface_json"
ML_JSON_ATTR: Final = "ml_json"
_SURFACE_GRIDS: Final = frozenset({"x_um", "y_um", "z_um", "confidence", "gap_mask"})
_PREDICTIONS: Final = "predictions"

PREDICTIONS_DTYPE: Final = np.dtype(
    [
        ("point_id", np.int64),
        ("label", STRING_DTYPE),
        ("flagged", np.bool_),
        ("probability_bad", np.float64),
        ("predicted_confidence", np.float64),
    ]
)


def write_surface(group: Any, surface: SurfaceResult) -> None:
    """Store ``surface`` in the (new, empty) HDF5 ``group``.

    Raises:
        StorageError: the grids are not ``[ny][nx]`` with ny = len(y_um), nx = len(x_um).
    """
    shape = (len(surface.y_um), len(surface.x_um))
    z = _float_grid("z_um", surface.z_um, shape)
    confidence = _float_grid("confidence", surface.confidence, shape)
    _check_shape("gap_mask", surface.gap_mask, shape)
    gap_mask = np.asarray(surface.gap_mask, dtype=np.bool_).reshape(shape)

    for name, data, units in (
        ("x_um", np.asarray(surface.x_um, dtype=np.float64), "um (grid columns)"),
        ("y_um", np.asarray(surface.y_um, dtype=np.float64), "um (grid rows)"),
        ("z_um", z, "um, [iy][ix] (NaN: no value)"),
        ("confidence", confidence, "0..1, [iy][ix] (NaN: no value)"),
        ("gap_mask", gap_mask, "True where the cell is a gap, [iy][ix]"),
    ):
        group.create_dataset(name, data=data).attrs["units"] = units
    group.attrs[SURFACE_JSON_ATTR] = surface.model_dump_json(exclude=set(_SURFACE_GRIDS))


def read_surface(group: Any) -> SurfaceResult:
    """Inverse of :func:`write_surface`."""
    document = json.loads(group.attrs[SURFACE_JSON_ATTR])
    z = np.asarray(group["z_um"][()], dtype=np.float64)
    confidence = np.asarray(group["confidence"][()], dtype=np.float64)
    gap_mask = np.asarray(group["gap_mask"][()], dtype=np.bool_)
    document.update(
        x_um=np.asarray(group["x_um"][()], dtype=np.float64).tolist(),
        y_um=np.asarray(group["y_um"][()], dtype=np.float64).tolist(),
        z_um=[nan_to_none(row) for row in z],
        confidence=[nan_to_none(row) for row in confidence],
        gap_mask=[[bool(cell) for cell in row] for row in gap_mask],
    )
    return SurfaceResult.model_validate(document)


def write_ml_result(group: Any, result: MLResult) -> None:
    """Store ``result`` in the (new, empty) HDF5 ``group``."""
    table = np.zeros(len(result.predictions), dtype=PREDICTIONS_DTYPE)
    for i, prediction in enumerate(result.predictions):
        table[i] = (
            prediction.point_id,
            prediction.label,
            prediction.flagged,
            _nan_if_none(prediction.probability_bad),
            _nan_if_none(prediction.predicted_confidence),
        )
    predictions = group.create_dataset(_PREDICTIONS, data=table)
    predictions.attrs["units"] = "probability_bad, predicted_confidence: 0..1 (NaN: no value)"
    group.attrs[ML_JSON_ATTR] = result.model_dump_json(exclude={"predictions"})


def read_ml_result(group: Any) -> MLResult:
    """Inverse of :func:`write_ml_result`."""
    document = json.loads(group.attrs[ML_JSON_ATTR])
    table = group[_PREDICTIONS][()]
    document["predictions"] = [
        MLPointPrediction(
            point_id=int(row["point_id"]),
            label=_text(row["label"]),
            flagged=bool(row["flagged"]),
            probability_bad=_none_if_nan(float(row["probability_bad"])),
            predicted_confidence=_none_if_nan(float(row["predicted_confidence"])),
        )
        for row in table
    ]
    return MLResult.model_validate(document)


def _float_grid(
    name: str, rows: Sequence[Sequence[float | None]], shape: tuple[int, int]
) -> NDArray[np.float64]:
    _check_shape(name, rows, shape)
    values = [[math.nan if v is None else float(v) for v in row] for row in rows]
    return np.asarray(values, dtype=np.float64).reshape(shape)


def _check_shape(name: str, rows: Sequence[Sequence[object]], shape: tuple[int, int]) -> None:
    ny, nx = shape
    if len(rows) != ny or any(len(row) != nx for row in rows):
        raise StorageError(f"surface {name} must have shape [{ny}][{nx}] (len(y_um), len(x_um))")


def _nan_if_none(value: float | None) -> float:
    return math.nan if value is None else float(value)


def _none_if_nan(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _text(value: object) -> str:
    """h5py returns variable-length strings inside compound rows as UTF-8 bytes."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
