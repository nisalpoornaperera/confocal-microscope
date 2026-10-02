"""Storage protocols used by the microscope and scanning layers.

Implementations (``confocal.storage.repository``) are **synchronous and
thread-safe**. Async callers on the event loop must invoke them through
``await asyncio.to_thread(store.method, ...)`` because HDF5 flushes on an SD
card can take tens of milliseconds.

Persistence rules:
* SQLite holds metadata and per-point scalar results (queryable).
* HDF5 (one file per scan) holds every raw array; it is self-describing
  (config, calibration, software and hardware versions as attributes).
* Nothing is ever deleted. Interrupted scans are kept and marked interrupted.
* ``append_point`` is durable when it returns (flushed to disk).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from confocal.models.calibration import CalibrationState
from confocal.models.hardware import HardwareInfo
from confocal.models.measurement import ProfileData, ProfileRecord
from confocal.models.ml import MLResult
from confocal.models.processing import ProfileAnalysis
from confocal.models.scan import ScanConfig, ScanPoint, ScanState, ScanSummary
from confocal.models.surface import SurfaceResult


@runtime_checkable
class CalibrationStore(Protocol):
    def save_calibration(self, state: CalibrationState) -> CalibrationState:
        """Persist a snapshot; returns it with ``version`` assigned (monotonic)."""
        ...

    def latest_calibration(self) -> CalibrationState | None: ...

    def get_calibration(self, version: int) -> CalibrationState | None: ...

    def list_calibrations(self, limit: int = 50) -> list[CalibrationState]:
        """Newest first."""
        ...


@runtime_checkable
class ScanRepository(Protocol):
    def create_scan(
        self,
        *,
        scan_id: str,
        config: ScanConfig,
        total_points: int,
        calibration: CalibrationState,
        software_version: str,
        hardware: HardwareInfo,
    ) -> ScanSummary:
        """Create the SQLite row (state IDLE) and the scan's HDF5 file."""
        ...

    def update_scan(
        self,
        scan_id: str,
        *,
        state: ScanState | None = None,
        completed_points: int | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error_message: str | None = None,
        interrupted: bool | None = None,
        calibration: CalibrationState | None = None,
    ) -> ScanSummary:
        """Update only the given fields. Raises ScanNotFoundError."""
        ...

    def get_scan(self, scan_id: str) -> ScanSummary:
        """Raises ScanNotFoundError."""
        ...

    def list_scans(self, *, limit: int = 100, offset: int = 0) -> list[ScanSummary]:
        """Newest first."""
        ...

    def append_point(
        self,
        scan_id: str,
        point: ScanPoint,
        profile: ProfileData | None,
        analysis: ProfileAnalysis | None,
    ) -> None:
        """Persist one point: scalars to SQLite + HDF5 table, raw arrays to HDF5. Durable."""
        ...

    def get_points(
        self, scan_id: str, *, since_point_id: int | None = None, limit: int | None = None
    ) -> list[ScanPoint]:
        """Points ordered by point_id; ``since_point_id`` returns ids strictly greater."""
        ...

    def get_profile(self, scan_id: str, point_id: int) -> ProfileRecord:
        """Raises ScanNotFoundError / PointNotFoundError."""
        ...

    def save_surface(self, surface: SurfaceResult) -> SurfaceResult:
        """Persist a reconstruction; returns it with ``surface_id`` assigned."""
        ...

    def get_latest_surface(self, scan_id: str) -> SurfaceResult | None: ...

    def save_ml_result(self, result: MLResult) -> MLResult:
        """Persist an ML analysis; returns it with ``result_id`` assigned."""
        ...

    def get_latest_ml_result(self, scan_id: str) -> MLResult | None: ...

    def record_event(self, kind: str, message: str, *, scan_id: str | None = None) -> None:
        """Append to the audit log (e-stops, hardware failures, state changes)."""
        ...

    def recover_interrupted_scans(self) -> list[str]:
        """At startup: mark scans left in an active state as ERROR + interrupted."""
        ...

    def close(self) -> None: ...
