"""``ScanPoint`` <-> stored representation.

The same field groups drive the SQLite row (:class:`ScanPointRow`) and the
HDF5 ``/points`` table (:mod:`confocal.storage.layout`), so both always
describe exactly the fields of :class:`ScanPoint`.

Non-finite values: API models must never contain NaN/inf, and "no value" is
``None``. :func:`sanitize_point` therefore maps non-finite *optional* scalars to
``None`` before anything is written (SQLite would silently turn NaN into NULL
anyway). *Required* scalars (coordinates, duration, confidence) have no "no
value" representation; a non-finite one is a bug upstream and is rejected
before anything is written.
"""

from __future__ import annotations

import json
import math

from confocal.errors import StorageError
from confocal.models.processing import PointStatus, ProfileAnalysis
from confocal.models.scan import ScanPoint
from confocal.storage.tables import ScanPointRow, ensure_utc

POINT_INT_FIELDS: tuple[str, ...] = ("point_id", "ix", "iy", "n_z_positions")
POINT_REQUIRED_FLOAT_FIELDS: tuple[str, ...] = ("x_um", "y_um", "confidence", "duration_s")
POINT_OPTIONAL_FLOAT_FIELDS: tuple[str, ...] = (
    "z_estimate_um",
    "coarse_peak_z_um",
    "surface_z_um",
    "parabolic_z_um",
    "gaussian_z_um",
    "peak_intensity",
    "snr",
    "peak_width_um",
    "prominence",
    "fit_residual",
    "intensity",
    "secondary_peak_ratio",
    "asymmetry",
)
#: Fields with a dedicated encoding: status (enum value), flags (JSON list), acquired_at (UTC).
POINT_SPECIAL_FIELDS: tuple[str, ...] = ("status", "flags", "acquired_at")


def finite_or_none(value: float | None) -> float | None:
    """``float(value)`` if finite, else ``None``."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def sanitize_point(point: ScanPoint) -> ScanPoint:
    """Validated copy fit for storage: optional NaN/inf -> None, acquired_at in UTC.

    Raises:
        StorageError: a required scalar (coordinate, duration, confidence) is not finite.
    """
    bad = [
        name
        for name in POINT_REQUIRED_FLOAT_FIELDS
        if not math.isfinite(float(getattr(point, name)))
    ]
    if bad:
        raise StorageError(
            f"point {point.point_id}: non-finite required value(s) {', '.join(bad)}; not stored"
        )
    update: dict[str, object] = {
        name: finite_or_none(getattr(point, name)) for name in POINT_OPTIONAL_FLOAT_FIELDS
    }
    update["acquired_at"] = ensure_utc(point.acquired_at)
    update["flags"] = list(point.flags)
    return point.model_copy(update=update)


def point_to_row(
    scan_id: str, point: ScanPoint, *, profile_row: int | None, analysis: ProfileAnalysis | None
) -> ScanPointRow:
    """SQLite row for a point already passed through :func:`sanitize_point`."""
    values: dict[str, object] = {
        name: getattr(point, name)
        for name in POINT_INT_FIELDS + POINT_REQUIRED_FLOAT_FIELDS + POINT_OPTIONAL_FLOAT_FIELDS
    }
    return ScanPointRow.model_validate(
        {
            **values,
            "scan_id": scan_id,
            "status": point.status.value,
            "flags_json": json.dumps(point.flags),
            "acquired_at": point.acquired_at,
            "profile_row": profile_row,
            "analysis_json": None if analysis is None else analysis.model_dump_json(),
        }
    )


def row_to_point(row: ScanPointRow) -> ScanPoint:
    """Rebuild the ``ScanPoint`` stored in ``row`` (timestamps are UTC-aware)."""
    values: dict[str, object] = {
        name: getattr(row, name)
        for name in POINT_INT_FIELDS + POINT_REQUIRED_FLOAT_FIELDS + POINT_OPTIONAL_FLOAT_FIELDS
    }
    flags = json.loads(row.flags_json)
    if not isinstance(flags, list):
        raise StorageError(f"point {row.point_id} of scan {row.scan_id}: corrupt flags")
    return ScanPoint.model_validate(
        {
            **values,
            "status": PointStatus(row.status),
            "flags": [str(flag) for flag in flags],
            "acquired_at": ensure_utc(row.acquired_at),
        }
    )
