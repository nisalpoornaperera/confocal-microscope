"""``SQLiteHDF5Repository``: the persistence layer behind ScanRepository and CalibrationStore.

Every method is synchronous and thread-safe; async callers use
``asyncio.to_thread``. Writes are serialised by one lock (SQLite allows a single
writer anyway, and HDF5 appends must not interleave); reads run concurrently
with them - SQLite in WAL mode never blocks readers, and HDF5 reads go through
the store's single handle per active file.

Consistency between the two stores
    A scan's HDF5 file is created before its SQLite row, and a point's arrays
    are written, flushed and fsync'ed before its SQLite row is committed, so
    everything listed in SQLite is durably in HDF5. SQLite is the record of
    which points a scan has. If the commit fails (or the power fails between
    the two writes) the raw arrays - which exist nowhere else - stay in the
    file, their ``/points`` row listed in ``/points_uncommitted``; they are not
    reported as points of the scan. When the scan ends (:meth:`update_scan`
    to a terminal state, or :meth:`recover_interrupted_scans` at startup) the
    file is reconciled with SQLite and a ``points_uncommitted`` audit event
    lists the ids of such points.

Nothing is ever deleted: there is no delete method, interrupted scans are
marked ``interrupted=True``, and a surface / ML result whose HDF5 write failed
keeps its reserved id with no data (readers skip it).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Final, Self

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlmodel import Session, col, func, select, update

from confocal.errors import PointNotFoundError, ScanNotFoundError, StorageError
from confocal.models.calibration import CalibrationState
from confocal.models.common import utc_now
from confocal.models.hardware import HardwareInfo
from confocal.models.measurement import ProfileData, ProfileRecord, nan_to_none
from confocal.models.ml import MLResult
from confocal.models.processing import ProfileAnalysis
from confocal.models.scan import (
    ACTIVE_SCAN_STATES,
    TERMINAL_SCAN_STATES,
    ScanConfig,
    ScanPoint,
    ScanState,
    ScanSummary,
)
from confocal.models.surface import SurfaceResult
from confocal.storage.database import Database
from confocal.storage.hdf5 import FilePoints, HDF5ScanStore, StoredProfile
from confocal.storage.points import finite_or_none, point_to_row, row_to_point, sanitize_point
from confocal.storage.records import AuditEvent, audit_event, calibration_state, scan_summary
from confocal.storage.tables import (
    CalibrationRow,
    EventRow,
    MLResultRow,
    ScanPointRow,
    ScanRow,
    SurfaceRow,
)

logger = logging.getLogger(__name__)

#: Audit-log kind recorded for every scan found in an active state at startup.
EVENT_SCAN_INTERRUPTED: Final = "scan_interrupted"
#: Audit-log kind listing points whose raw data is in the HDF5 file but whose
#: database commit did not complete (recorded when the scan ends or is recovered).
EVENT_POINTS_UNCOMMITTED: Final = "points_uncommitted"


class SQLiteHDF5Repository:
    """SQLite metadata at ``database_path`` + one HDF5 file per scan in ``scans_dir``.

    ``durable=False`` skips fsync (SQLite ``synchronous=NORMAL``, no HDF5
    fsync): data then survives a crash of the process but not a power cut.
    Use it only for tests and throwaway simulations.
    """

    def __init__(self, database_path: Path, scans_dir: Path, *, durable: bool = True) -> None:
        self._db = Database(Path(database_path), synchronous="FULL" if durable else "NORMAL")
        try:
            self._db.create_all()
            self._hdf5 = HDF5ScanStore(Path(scans_dir), fsync=durable)
        except BaseException:
            self._db.close()  # release the file (Windows keeps open files locked)
            raise
        self._write_lock = threading.Lock()
        # Ids of scans known to exist. Sound because scans are never deleted; it
        # saves the per-point existence query on the append path.
        self._known_scans: set[str] = set()

    @property
    def database_path(self) -> Path:
        return self._db.path

    @property
    def scans_dir(self) -> Path:
        return self._hdf5.scans_dir

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close every HDF5 file and database connection (idempotent)."""
        with self._write_lock:
            self._hdf5.close()
            self._db.close()

    # ------------------------------------------------------------ CalibrationStore

    def save_calibration(self, state: CalibrationState) -> CalibrationState:
        """Persist a snapshot under a new version (any incoming ``version`` is replaced)."""
        with self._write_lock, self._db.session() as session:
            row = CalibrationRow(
                created_at=state.created_at,
                updated_field=state.updated_field,
                dark_v=finite_or_none(state.dark_v),
                reference_v=finite_or_none(state.reference_v),
                state_json="",
            )
            session.add(row)
            session.flush()  # assigns the AUTOINCREMENT version
            stored = state.model_copy(update={"version": row.version})
            row.state_json = stored.model_dump_json()
            session.add(row)
        return stored

    def latest_calibration(self) -> CalibrationState | None:
        with self._db.session() as session:
            row = session.exec(
                select(CalibrationRow).order_by(col(CalibrationRow.version).desc()).limit(1)
            ).first()
        return None if row is None else calibration_state(row)

    def get_calibration(self, version: int) -> CalibrationState | None:
        with self._db.session() as session:
            row = session.get(CalibrationRow, version)
        return None if row is None else calibration_state(row)

    def list_calibrations(self, limit: int = 50) -> list[CalibrationState]:
        """Newest (highest version) first."""
        _require_non_negative("limit", limit)
        with self._db.session() as session:
            rows = session.exec(
                select(CalibrationRow).order_by(col(CalibrationRow.version).desc()).limit(limit)
            ).all()
        return [calibration_state(row) for row in rows]

    # ------------------------------------------------------------ scans

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
        """Create the scan's HDF5 file and SQLite row (state IDLE).

        Raises:
            StorageError: the id is not a safe file name, or the scan (or its
                file) already exists.
        """
        _require_non_negative("total_points", total_points)
        created_at = utc_now()
        with self._write_lock:
            with self._db.session() as session:
                if self._find_scan(session, scan_id) is not None:
                    raise StorageError(f"scan {scan_id!r} already exists")
            path = self._hdf5.create_scan(
                scan_id=scan_id,
                created_at=created_at,
                config=config,
                calibration=calibration,
                software_version=software_version,
                hardware=hardware,
            )
            row = ScanRow(
                id=scan_id,
                name=config.name,
                mode=config.mode.value,
                state=ScanState.IDLE.value,
                created_at=created_at,
                total_points=total_points,
                config_json=config.model_dump_json(),
                calibration_version=calibration.version,
                calibration_json=calibration.model_dump_json(),
                software_version=software_version,
                hardware_json=hardware.model_dump_json(),
                data_file=path.name,
            )
            try:
                with self._db.session() as session:
                    session.add(row)
            except BaseException:
                self._hdf5.finish_scan(scan_id)  # the file stays, as every file does
                raise
            self._known_scans.add(scan_id)
        return scan_summary(row, data_file=path, has_surface=False, has_ml_result=False)

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
        """Update only the given fields; a terminal ``state`` closes the scan's HDF5 file.

        ``completed_points`` overrides the count that :meth:`append_point`
        maintains (one per stored point). When the scan first reaches a
        terminal state its file is reconciled with the database, in the same
        transaction: uncommitted points are marked in the file and listed in a
        ``points_uncommitted`` audit event.
        """
        if completed_points is not None:
            _require_non_negative("completed_points", completed_points)
        with self._write_lock:
            with self._db.session() as session:
                row = self._scan_row(session, scan_id)
                if (
                    state in TERMINAL_SCAN_STATES
                    and ScanState(row.state) not in TERMINAL_SCAN_STATES
                ):
                    self._audit_uncommitted(session, scan_id)
                if state is not None:
                    row.state = state.value
                if completed_points is not None:
                    row.completed_points = completed_points
                if started_at is not None:
                    row.started_at = started_at
                if finished_at is not None:
                    row.finished_at = finished_at
                if error_message is not None:
                    row.error_message = error_message
                if interrupted is not None:
                    row.interrupted = interrupted
                if calibration is not None:
                    row.calibration_version = calibration.version
                    row.calibration_json = calibration.model_dump_json()
                    self._hdf5.update_calibration(scan_id, calibration)
                session.add(row)
            if state is not None and state in TERMINAL_SCAN_STATES:
                self._hdf5.finish_scan(scan_id)
        return self.get_scan(scan_id)

    def get_scan(self, scan_id: str) -> ScanSummary:
        with self._db.session() as session:
            summaries = self._summaries(session, scan_id=scan_id)
        if not summaries:
            raise ScanNotFoundError(f"scan {scan_id!r} does not exist")
        return summaries[0]

    def list_scans(self, *, limit: int = 100, offset: int = 0) -> list[ScanSummary]:
        """Newest (most recently created) first."""
        _require_non_negative("limit", limit)
        _require_non_negative("offset", offset)
        with self._db.session() as session:
            return self._summaries(session, limit=limit, offset=offset)

    # ------------------------------------------------------------ points

    def append_point(
        self,
        scan_id: str,
        point: ScanPoint,
        profile: ProfileData | None,
        analysis: ProfileAnalysis | None,
    ) -> None:
        """Persist one point durably: arrays to HDF5 (flushed + fsync'ed), then the SQLite row.

        Optional non-finite scalars are stored as ``None``. Also increments the
        scan's ``completed_points``.

        Raises:
            ScanNotFoundError: unknown scan.
            StorageError: duplicate ``point_id``, invalid profile (wrong samples
                per Z, non-finite positions/voltages) or an I/O failure. The
                point is not stored in any of these cases. Only if the database
                commit fails after the HDF5 write completed are its raw arrays
                kept in the file, marked uncommitted (see the module docstring).
        """
        stored_point = sanitize_point(point)
        stored_analysis = _storable_analysis(analysis)
        point_id = stored_point.point_id
        with self._write_lock, self._db.session() as session:
            self._require_scan(session, scan_id)
            if self._find_point(session, scan_id, point_id) is not None:
                raise StorageError(f"point {point_id} of scan {scan_id!r} is already stored")
            with self._hdf5.append_point(
                scan_id, stored_point, profile, stored_analysis
            ) as profile_row:
                session.exec(
                    update(ScanRow)
                    .where(col(ScanRow.id) == scan_id)
                    .values(completed_points=col(ScanRow.completed_points) + 1)
                )
                session.add(
                    point_to_row(
                        scan_id, stored_point, profile_row=profile_row, analysis=stored_analysis
                    )
                )
                try:
                    session.commit()  # inside the HDF5 block: a failure marks it uncommitted
                except IntegrityError as exc:  # the unique (scan_id, point_id) constraint
                    raise StorageError(
                        f"point {point_id} of scan {scan_id!r} is already stored ({exc.orig}); "
                        "the new raw data is kept in the HDF5 file, marked uncommitted"
                    ) from exc
                except SQLAlchemyError as exc:
                    raise StorageError(
                        f"database commit of point {point_id} of scan {scan_id!r} failed "
                        f"({exc}); its raw data is kept in the HDF5 file, marked uncommitted"
                    ) from exc

    def get_points(
        self, scan_id: str, *, since_point_id: int | None = None, limit: int | None = None
    ) -> list[ScanPoint]:
        """Points ordered by ``point_id``; ``since_point_id`` returns ids strictly greater."""
        if limit is not None:
            _require_non_negative("limit", limit)
        with self._db.session() as session:
            self._require_scan(session, scan_id)
            statement = select(ScanPointRow).where(ScanPointRow.scan_id == scan_id)
            if since_point_id is not None:
                statement = statement.where(ScanPointRow.point_id > since_point_id)
            statement = statement.order_by(col(ScanPointRow.point_id))
            if limit is not None:
                statement = statement.limit(limit)
            rows = session.exec(statement).all()
        return [row_to_point(row) for row in rows]

    def get_profile(self, scan_id: str, point_id: int) -> ProfileRecord:
        """Every stored value of one point (raw codes, voltages, processed arrays, analysis).

        Raises:
            ScanNotFoundError: unknown scan.
            PointNotFoundError: unknown point, or a point stored without a profile.
        """
        with self._db.session() as session:
            self._require_scan(session, scan_id)
            row = self._find_point(session, scan_id, point_id)
        if row is None:
            raise PointNotFoundError(f"scan {scan_id!r} has no point {point_id}")
        if row.profile_row is None:
            raise PointNotFoundError(f"point {point_id} of scan {scan_id!r} has no stored profile")
        stored = self._hdf5.read_profile(scan_id, row.profile_row, point_id)
        return _profile_record(scan_id, row, stored)

    # ------------------------------------------------------------ surfaces and ML

    def save_surface(self, surface: SurfaceResult) -> SurfaceResult:
        """Store a reconstruction (grids in HDF5); returns it with ``surface_id`` assigned."""
        with self._write_lock:
            with self._db.session() as session:  # phase 1: reserve the id
                self._require_scan(session, surface.scan_id)
                row = SurfaceRow(
                    scan_id=surface.scan_id,
                    created_at=surface.created_at,
                    method=surface.request.method.value,
                    n_used=surface.statistics.n_used,
                    coverage_fraction=finite_or_none(surface.statistics.coverage_fraction),
                    request_json=surface.request.model_dump_json(),
                    statistics_json=surface.statistics.model_dump_json(),
                )
                session.add(row)
            stored = surface.model_copy(update={"surface_id": row.id})
            group = self._hdf5.write_surface(stored)
            with self._db.session() as session:  # phase 2: publish it
                reserved = session.get(SurfaceRow, row.id)
                if reserved is None:
                    raise StorageError(f"surface row {row.id} vanished")
                reserved.hdf5_group = group
                session.add(reserved)
        return stored

    def get_latest_surface(self, scan_id: str) -> SurfaceResult | None:
        """Most recently saved reconstruction of the scan, or ``None``."""
        with self._db.session() as session:
            self._require_scan(session, scan_id)
            group = session.exec(
                select(SurfaceRow.hdf5_group)
                .where(SurfaceRow.scan_id == scan_id, col(SurfaceRow.hdf5_group).is_not(None))
                .order_by(col(SurfaceRow.id).desc())
                .limit(1)
            ).first()
        return None if group is None else self._hdf5.read_surface(scan_id, group)

    def save_ml_result(self, result: MLResult) -> MLResult:
        """Store an advisory ML analysis (predictions in HDF5); returns it with ``result_id``."""
        with self._write_lock:
            with self._db.session() as session:  # phase 1: reserve the id
                self._require_scan(session, result.scan_id)
                row = MLResultRow(
                    scan_id=result.scan_id,
                    created_at=result.created_at,
                    model_name=result.model.name,
                    model_version=result.model.version,
                    task=result.model.task.value,
                    threshold=result.threshold,
                    n_points=result.n_points,
                    n_flagged=result.n_flagged,
                    model_json=result.model.model_dump_json(),
                )
                session.add(row)
            stored = result.model_copy(update={"result_id": row.id})
            group = self._hdf5.write_ml_result(stored)
            with self._db.session() as session:  # phase 2: publish it
                reserved = session.get(MLResultRow, row.id)
                if reserved is None:
                    raise StorageError(f"ML result row {row.id} vanished")
                reserved.hdf5_group = group
                session.add(reserved)
        return stored

    def get_latest_ml_result(self, scan_id: str) -> MLResult | None:
        """Most recently saved ML analysis of the scan, or ``None``."""
        with self._db.session() as session:
            self._require_scan(session, scan_id)
            group = session.exec(
                select(MLResultRow.hdf5_group)
                .where(MLResultRow.scan_id == scan_id, col(MLResultRow.hdf5_group).is_not(None))
                .order_by(col(MLResultRow.id).desc())
                .limit(1)
            ).first()
        return None if group is None else self._hdf5.read_ml_result(scan_id, group)

    # ------------------------------------------------------------ audit log and recovery

    def record_event(self, kind: str, message: str, *, scan_id: str | None = None) -> None:
        """Append to the audit log (e-stops, hardware failures, state changes)."""
        with self._write_lock, self._db.session() as session:
            session.add(EventRow(created_at=utc_now(), kind=kind, message=message, scan_id=scan_id))

    def list_events(
        self, *, scan_id: str | None = None, kind: str | None = None, limit: int = 100
    ) -> list[AuditEvent]:
        """Audit-log entries, newest first, optionally filtered by scan and kind."""
        _require_non_negative("limit", limit)
        statement = select(EventRow)
        if scan_id is not None:
            statement = statement.where(EventRow.scan_id == scan_id)
        if kind is not None:
            statement = statement.where(EventRow.kind == kind)
        with self._db.session() as session:
            rows = session.exec(statement.order_by(col(EventRow.id).desc()).limit(limit)).all()
        return [audit_event(row) for row in rows]

    def recover_interrupted_scans(self) -> list[str]:
        """At startup: mark scans left in an active state as ERROR + interrupted.

        Each one gets an explanatory ``error_message`` (state at the time, points
        stored, data-file condition) and a ``scan_interrupted`` audit event. Its
        file is reconciled with the database: points whose commit did not
        complete are marked in the file and listed in a ``points_uncommitted``
        audit event. Returns their ids, oldest first.
        """
        recovered: list[str] = []
        with self._write_lock:
            with self._db.session() as session:
                rows = session.exec(
                    select(ScanRow)
                    .where(col(ScanRow.state).in_([s.value for s in ACTIVE_SCAN_STATES]))
                    .order_by(col(ScanRow.seq))
                ).all()
                now = utc_now()
                for row in rows:
                    message = self._interruption_message(session, row)
                    row.state = ScanState.ERROR.value
                    row.interrupted = True
                    row.error_message = message
                    session.add(row)
                    session.add(
                        EventRow(
                            created_at=now,
                            kind=EVENT_SCAN_INTERRUPTED,
                            message=message,
                            scan_id=row.id,
                        )
                    )
                    recovered.append(row.id)
            for scan_id in recovered:
                self._hdf5.finish_scan(scan_id)
        return recovered

    # ------------------------------------------------------------ internals

    def _summaries(
        self,
        session: Session,
        *,
        scan_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ScanSummary]:
        """Scan summaries with ``has_surface`` / ``has_ml_result`` computed in one query."""
        has_surface = (
            select(SurfaceRow.id)
            .where(col(SurfaceRow.scan_id) == col(ScanRow.id))
            .where(col(SurfaceRow.hdf5_group).is_not(None))
            .exists()
        )
        has_ml_result = (
            select(MLResultRow.id)
            .where(col(MLResultRow.scan_id) == col(ScanRow.id))
            .where(col(MLResultRow.hdf5_group).is_not(None))
            .exists()
        )
        statement = select(ScanRow, has_surface, has_ml_result)
        if scan_id is not None:
            statement = statement.where(ScanRow.id == scan_id)
        statement = statement.order_by(col(ScanRow.seq).desc()).offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
        return [
            scan_summary(
                row,
                data_file=self._hdf5.scans_dir / row.data_file,
                has_surface=bool(surface),
                has_ml_result=bool(ml_result),
            )
            for row, surface, ml_result in session.exec(statement).all()
        ]

    @staticmethod
    def _find_scan(session: Session, scan_id: str) -> ScanRow | None:
        return session.exec(select(ScanRow).where(ScanRow.id == scan_id)).first()

    def _scan_row(self, session: Session, scan_id: str) -> ScanRow:
        row = self._find_scan(session, scan_id)
        if row is None:
            raise ScanNotFoundError(f"scan {scan_id!r} does not exist")
        self._known_scans.add(scan_id)
        return row

    def _require_scan(self, session: Session, scan_id: str) -> None:
        """Raise ScanNotFoundError for an unknown scan (cached: scans are never deleted)."""
        if scan_id not in self._known_scans:
            self._scan_row(session, scan_id)

    @staticmethod
    def _find_point(session: Session, scan_id: str, point_id: int) -> ScanPointRow | None:
        return session.exec(
            select(ScanPointRow).where(
                ScanPointRow.scan_id == scan_id, ScanPointRow.point_id == point_id
            )
        ).first()

    def _interruption_message(self, session: Session, row: ScanRow) -> str:
        n_stored = session.exec(
            select(func.count()).select_from(ScanPointRow).where(ScanPointRow.scan_id == row.id)
        ).one()
        message = (
            f"Interrupted: the server stopped while the scan was {row.state} "
            f"({n_stored} of {row.total_points} points stored); marked at startup"
        )
        try:
            uncommitted = self._reconcile(session, row.id)
        except StorageError as exc:
            return f"{message}. Data file problem: {exc}"
        if uncommitted:
            session.add(self._uncommitted_event(row.id, uncommitted))
            return (
                f"{message}. The data file holds {len(uncommitted)} further point(s) whose "
                f"database commit did not complete (point ids {uncommitted}); their raw data "
                "is kept in the file"
            )
        return message

    def _reconcile(self, session: Session, scan_id: str) -> list[int]:
        """Mark the file's points that SQLite does not hold; returns their point ids.

        Raises:
            StorageError: the file cannot be read or marked.
        """
        file_points = self._hdf5.file_points(scan_id)
        stored = session.exec(
            select(ScanPointRow.point_id, ScanPointRow.profile_row).where(
                ScanPointRow.scan_id == scan_id
            )
        ).all()
        committed = _committed_rows(file_points, [(int(p), r) for p, r in stored])
        rows = [row for row in range(len(file_points.keys)) if row not in committed]
        self._hdf5.set_uncommitted(scan_id, rows)
        return [file_points.keys[row][0] for row in rows]

    def _audit_uncommitted(self, session: Session, scan_id: str) -> None:
        """Reconcile a scan that is ending; an unreadable file is logged, never fatal."""
        try:
            uncommitted = self._reconcile(session, scan_id)
        except StorageError:
            logger.exception("could not reconcile the data file of scan %r", scan_id)
            return
        if uncommitted:
            session.add(self._uncommitted_event(scan_id, uncommitted))

    def _uncommitted_event(self, scan_id: str, point_ids: list[int]) -> EventRow:
        path = self._hdf5.path_for(scan_id)
        return EventRow(
            created_at=utc_now(),
            kind=EVENT_POINTS_UNCOMMITTED,
            message=(
                f"{len(point_ids)} point(s) were written to {path.name} but their database "
                f"commit did not complete: point ids {point_ids}. Their raw data is kept in "
                "the file (rows listed in /points_uncommitted); they are not stored points "
                "of the scan"
            ),
            scan_id=scan_id,
        )


def _committed_rows(file_points: FilePoints, stored: list[tuple[int, int | None]]) -> set[int]:
    """Rows of ``/points`` that the database holds, matched by ``(point_id, profile_row)``.

    A key the file holds more than once (a point re-acquired after a failed
    commit) is matched to its last row not already marked uncommitted.
    """
    rows_by_key: dict[tuple[int, int | None], list[int]] = {}
    for row, key in enumerate(file_points.keys):
        rows_by_key.setdefault(key, []).append(row)
    committed: set[int] = set()
    for key in stored:
        rows = rows_by_key.get(key)
        if rows:
            unmarked = [row for row in rows if row not in file_points.uncommitted]
            committed.add((unmarked or rows)[-1])
    return committed


def _storable_analysis(analysis: ProfileAnalysis | None) -> ProfileAnalysis | None:
    """The analysis exactly as it will read back (NaN/inf -> None).

    Raises:
        StorageError: a required field is non-finite, so it could never be read back.
    """
    if analysis is None:
        return None
    try:
        return ProfileAnalysis.model_validate_json(analysis.model_dump_json())
    except ValueError as exc:
        raise StorageError(f"profile analysis cannot be stored: {exc}") from exc


def _profile_record(scan_id: str, row: ScanPointRow, stored: StoredProfile) -> ProfileRecord:
    data = stored.data
    return ProfileRecord(
        scan_id=scan_id,
        point_id=row.point_id,
        x_um=row.x_um,
        y_um=row.y_um,
        z_um=data.z_um.tolist(),
        z_reported_um=data.z_reported_um.tolist(),
        phase=[int(phase) for phase in data.phase],
        raw_counts=data.raw_counts.tolist(),
        voltage_v=data.voltage_v.tolist(),
        voltage_agg_v=data.voltage_agg_v.tolist(),
        timestamps=data.timestamps.tolist(),
        gain=data.gain,
        sampling_method=data.sampling_method,
        dark_v=data.dark_v,
        reference_v=data.reference_v,
        calibration_version=data.calibration_version,
        normalized=None if data.normalized is None else nan_to_none(data.normalized),
        filtered=None if data.filtered is None else nan_to_none(data.filtered),
        analysis=stored.analysis,
    )


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
