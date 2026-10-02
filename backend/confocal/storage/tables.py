"""SQLite schema (SQLModel tables).

SQLite holds metadata and per-point *scalar* results so they can be queried
(scan lists, point clouds for reconstruction, calibration history, audit log).
Every raw array lives in the scan's HDF5 file (:mod:`confocal.storage.hdf5`).

Conventions

* JSON documents (configs, calibration snapshots, analyses, hardware identity)
  are TEXT produced by Pydantic's ``model_dump_json`` and read back with
  ``model_validate_json``: an exact round trip, with NaN/inf written as ``null``.
* SQLite has no timezone-aware type. :class:`UTCDateTime` stores naive UTC and
  returns timezone-aware UTC datetimes; naive inputs are taken to be UTC.
* Rows are never deleted (there is no delete API anywhere in storage).
* Integer surrogate keys define insertion order, which is what "newest first"
  means. The wall clock is deliberately not used for ordering: a Raspberry Pi
  without network time can boot with its clock in the past.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Table, Text, UniqueConstraint
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, SQLModel


def ensure_utc(value: datetime) -> datetime:
    """Timezone-aware UTC copy of ``value`` (a naive datetime is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """DateTime column that always round-trips as timezone-aware UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value).replace(tzinfo=None)

    def process_result_value(self, value: Any | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"expected a datetime from the database, got {type(value).__name__}")
        return ensure_utc(value)


class ScanRow(SQLModel, table=True):
    """One scan: configuration and calibration snapshots plus lifecycle state."""

    __tablename__ = "scans"
    __table_args__ = {"sqlite_autoincrement": True}

    seq: int | None = Field(default=None, primary_key=True, description="Insertion order.")
    id: str = Field(unique=True, index=True, max_length=128)
    name: str | None = Field(default=None, max_length=200)
    mode: str = Field(max_length=32)
    state: str = Field(index=True, max_length=32)
    created_at: datetime = Field(sa_type=UTCDateTime)
    started_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    finished_at: datetime | None = Field(default=None, sa_type=UTCDateTime)
    total_points: int
    completed_points: int = 0
    interrupted: bool = False
    error_message: str | None = Field(default=None, sa_type=Text)
    config_json: str = Field(sa_type=Text)
    calibration_version: int | None = None
    calibration_json: str = Field(sa_type=Text)
    software_version: str = Field(max_length=64)
    hardware_json: str = Field(sa_type=Text)
    data_file: str = Field(max_length=255, description="HDF5 file name inside the scans dir.")


class ScanPointRow(SQLModel, table=True):
    """Scalar results of one XY point (mirrors ``ScanPoint``; NaN is stored as NULL).

    The unique ``(scan_id, point_id)`` constraint is also the index used by every
    per-scan point query (ordered by ``point_id``).
    """

    __tablename__ = "scan_points"
    __table_args__ = (UniqueConstraint("scan_id", "point_id", name="uq_scan_points_scan_point"),)

    id: int | None = Field(default=None, primary_key=True)
    scan_id: str = Field(foreign_key="scans.id", max_length=128)
    point_id: int
    ix: int
    iy: int
    x_um: float
    y_um: float
    status: str = Field(max_length=32)
    z_estimate_um: float | None = None
    coarse_peak_z_um: float | None = None
    surface_z_um: float | None = None
    parabolic_z_um: float | None = None
    gaussian_z_um: float | None = None
    peak_intensity: float | None = None
    snr: float | None = None
    peak_width_um: float | None = None
    prominence: float | None = None
    fit_residual: float | None = None
    confidence: float
    intensity: float | None = None
    secondary_peak_ratio: float | None = None
    asymmetry: float | None = None
    n_z_positions: int
    flags_json: str = Field(sa_type=Text)
    acquired_at: datetime = Field(sa_type=UTCDateTime)
    duration_s: float
    profile_row: int | None = Field(
        default=None,
        description="Row of /profiles/index and /profiles/meta in the HDF5 file (NULL: none).",
    )
    analysis_json: str | None = Field(
        default=None, sa_type=Text, description="Full ProfileAnalysis (queryable via JSON1)."
    )


class CalibrationRow(SQLModel, table=True):
    """Immutable calibration snapshot; the primary key *is* the calibration version.

    ``AUTOINCREMENT`` guarantees a version number is never reused, even after the
    newest row of an aborted transaction was rolled back.
    """

    __tablename__ = "calibrations"
    __table_args__ = {"sqlite_autoincrement": True}

    version: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(sa_type=UTCDateTime)
    updated_field: str | None = Field(default=None, max_length=32)
    dark_v: float | None = None
    reference_v: float | None = None
    state_json: str = Field(sa_type=Text)


class SurfaceRow(SQLModel, table=True):
    """Index entry for a reconstruction whose grids live in ``/surfaces/<id>`` of the HDF5 file.

    ``hdf5_group`` is NULL while the HDF5 group is being written (or if writing it
    failed); such rows are ignored by readers and keep their id reserved.
    """

    __tablename__ = "surfaces"
    __table_args__ = {"sqlite_autoincrement": True}

    id: int | None = Field(default=None, primary_key=True)
    scan_id: str = Field(foreign_key="scans.id", index=True, max_length=128)
    created_at: datetime = Field(sa_type=UTCDateTime)
    method: str = Field(max_length=32)
    n_used: int
    coverage_fraction: float | None = None
    request_json: str = Field(sa_type=Text)
    statistics_json: str = Field(sa_type=Text)
    hdf5_group: str | None = Field(default=None, max_length=255)


class MLResultRow(SQLModel, table=True):
    """Index entry for an advisory ML analysis stored in ``/ml/<id>`` of the HDF5 file.

    ``hdf5_group`` follows the same two-phase rule as :class:`SurfaceRow`.
    """

    __tablename__ = "ml_results"
    __table_args__ = {"sqlite_autoincrement": True}

    id: int | None = Field(default=None, primary_key=True)
    scan_id: str = Field(foreign_key="scans.id", index=True, max_length=128)
    created_at: datetime = Field(sa_type=UTCDateTime)
    model_name: str = Field(max_length=200)
    model_version: str = Field(max_length=64)
    task: str = Field(max_length=32)
    threshold: float
    n_points: int
    n_flagged: int
    model_json: str = Field(sa_type=Text)
    hdf5_group: str | None = Field(default=None, max_length=255)


class EventRow(SQLModel, table=True):
    """Audit log entry (e-stops, hardware failures, state changes, recoveries).

    ``scan_id`` is intentionally not a foreign key: the audit log must accept an
    entry even for a scan whose creation failed.
    """

    __tablename__ = "events"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(sa_type=UTCDateTime, index=True)
    kind: str = Field(index=True, max_length=64)
    message: str = Field(sa_type=Text)
    scan_id: str | None = Field(default=None, index=True, max_length=128)


#: Every table owned by the storage layer, in dependency order.
STORAGE_TABLE_NAMES: tuple[str, ...] = (
    "scans",
    "scan_points",
    "calibrations",
    "surfaces",
    "ml_results",
    "events",
)


def storage_tables() -> list[Table]:
    """SQLAlchemy ``Table`` objects of the storage schema (for ``create_all``)."""
    return [SQLModel.metadata.tables[name] for name in STORAGE_TABLE_NAMES]
