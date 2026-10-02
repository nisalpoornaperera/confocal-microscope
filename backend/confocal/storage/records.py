"""SQLite rows -> API models (scan summaries, calibration snapshots, audit events)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from confocal.models.calibration import CalibrationState
from confocal.models.hardware import HardwareInfo
from confocal.models.scan import ScanConfig, ScanMode, ScanState, ScanSummary
from confocal.storage.tables import CalibrationRow, EventRow, ScanRow, ensure_utc


class AuditEvent(BaseModel):
    """One entry of the audit log (e-stops, hardware failures, state changes, recoveries)."""

    model_config = ConfigDict(extra="forbid")

    id: int
    created_at: datetime
    kind: str
    message: str
    scan_id: str | None = None


def progress_fraction(completed_points: int, total_points: int) -> float:
    """Completed fraction clamped to [0, 1] (0 for an empty scan)."""
    if total_points <= 0:
        return 0.0
    return min(1.0, max(0.0, completed_points / total_points))


def scan_summary(
    row: ScanRow, *, data_file: Path, has_surface: bool, has_ml_result: bool
) -> ScanSummary:
    return ScanSummary(
        id=row.id,
        name=row.name,
        mode=ScanMode(row.mode),
        state=ScanState(row.state),
        created_at=ensure_utc(row.created_at),
        started_at=None if row.started_at is None else ensure_utc(row.started_at),
        finished_at=None if row.finished_at is None else ensure_utc(row.finished_at),
        total_points=row.total_points,
        completed_points=row.completed_points,
        progress=progress_fraction(row.completed_points, row.total_points),
        config=ScanConfig.model_validate_json(row.config_json),
        calibration_version=row.calibration_version,
        calibration=CalibrationState.model_validate_json(row.calibration_json),
        software_version=row.software_version,
        hardware=HardwareInfo.model_validate_json(row.hardware_json),
        interrupted=row.interrupted,
        error_message=row.error_message,
        data_file=str(data_file),
        has_surface=has_surface,
        has_ml_result=has_ml_result,
    )


def calibration_state(row: CalibrationRow) -> CalibrationState:
    return CalibrationState.model_validate_json(row.state_json)


def audit_event(row: EventRow) -> AuditEvent:
    if row.id is None:
        raise ValueError("event row has not been persisted")
    return AuditEvent(
        id=row.id,
        created_at=ensure_utc(row.created_at),
        kind=row.kind,
        message=row.message,
        scan_id=row.scan_id,
    )
