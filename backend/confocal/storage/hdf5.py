"""Per-scan HDF5 files: ``<scans_dir>/<scan_id>.h5`` (layout: :mod:`confocal.storage.layout`).

Thread safety and file handles
    h5py serialises HDF5 library calls, but a point is many calls (resize and
    write a dozen datasets), so every operation of the store holds one
    re-entrant lock. A file is never opened twice: on Windows HDF5 refuses to
    open a file a second time with a different mode, and two handles on one
    file can corrupt it anywhere. The store therefore keeps ONE writable handle
    per scan being acquired (opened by :meth:`HDF5ScanStore.create_scan`,
    reused for every read of that scan while it runs, closed by
    :meth:`HDF5ScanStore.finish_scan` / :meth:`HDF5ScanStore.close`). A finished
    scan's file is opened only for the duration of one call: read-only to read,
    read-write to add a surface / ML result.

File locking
    HDF5 locks the files it opens (``flock`` on Linux, ``LockFileEx`` on
    Windows), which stops a *second process* - an accidentally started second
    server, an analysis script - from opening a file that is being written
    (without SWMR a concurrent reader can see half-written metadata). Files are
    opened with ``locking="best-effort"``: the lock is used wherever the
    filesystem supports it (ext4 on the Pi's SD card, NTFS) and silently skipped
    where it does not (some USB-stick and network filesystems) instead of
    making every open fail. ``HDF5_USE_FILE_LOCKING=FALSE`` overrides it. SWMR
    is deliberately not used: it forbids creating objects (surfaces, ML results)
    while active and is not crash-proof either.

Durability
    After every point the file is flushed (HDF5 buffers reach the OS: survives
    a crash of this process) and, unless ``fsync=False``, ``fsync``'ed (data
    reaches the SD card: survives a power cut). The repository commits the
    point's SQLite row only after that, so every point listed in SQLite has its
    arrays durably in HDF5. HDF5 is not journaled: a power cut *during* the few
    milliseconds of a write can damage the file; the already-durable SQLite
    scalars are unaffected.

Never deleting flushed raw data
    An append whose own HDF5 write did not complete is truncated away (the
    data is partial). Once the write is flushed and synced it is never removed:
    if the database commit that follows fails, the point's ``/points`` row is
    listed in ``/points_uncommitted`` instead (:meth:`HDF5ScanStore.set_uncommitted`
    lets the repository reconcile that list with SQLite later, e.g. after a
    power cut between the two writes).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TypeVar

import h5py
import numpy as np
from numpy.typing import NDArray

from confocal.errors import StorageError
from confocal.models.calibration import CalibrationState
from confocal.models.hardware import HardwareInfo
from confocal.models.measurement import ProfileData
from confocal.models.ml import MLResult
from confocal.models.processing import ProfileAnalysis
from confocal.models.scan import ScanConfig, ScanPoint
from confocal.models.surface import SurfaceResult
from confocal.storage import layout, results

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Scan ids become file names: no separators, no leading dot, at most 128 characters.
_SCAN_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_LOCKING: Final = "best-effort"
_ARRAYS: Final = (*layout.POSITION_DATASETS, *layout.SAMPLE_DATASETS)


@dataclass(frozen=True, slots=True)
class StoredProfile:
    """The complete stored I(Z) data of one point."""

    data: ProfileData
    analysis: ProfileAnalysis | None


@dataclass(frozen=True, slots=True)
class FilePoints:
    """What a scan file holds in ``/points``, for reconciliation with the database.

    Attributes:
        keys: ``(point_id, profile_row)`` of every row, in file order.
        uncommitted: rows currently marked in ``/points_uncommitted``.
    """

    keys: list[tuple[int, int | None]]
    uncommitted: frozenset[int]


@dataclass(frozen=True, slots=True)
class _Lengths:
    """Dataset lengths before an append: the rollback target if it does not complete."""

    points: int
    profiles: int
    positions: int


class _OpenScan:
    """A scan file open for appending, its growing datasets looked up once.

    (A path lookup costs as much as a small write; a point touches 12 datasets.)
    """

    def __init__(self, path: Path, f: Any) -> None:
        self.path = path
        self.file = f
        self.points = f[layout.POINTS]
        self.index = f[layout.INDEX]
        self.meta = f[layout.META]
        self.arrays = {name: f[f"{layout.PROFILES}/{name}"] for name in _ARRAYS}
        self.samples_per_z = int(f[layout.PROFILES].attrs["samples_per_z"])

    def lengths(self) -> _Lengths:
        return _Lengths(
            points=int(self.points.shape[0]),
            profiles=int(self.index.shape[0]),
            positions=int(self.arrays["z_um"].shape[0]),
        )

    def write_profile(
        self, before: _Lengths, arrays: dict[str, NDArray[Any]], meta: layout.ProfileMeta
    ) -> int:
        """Append one profile's arrays, then its metadata, then (last) its index row.

        The index row is written last so that, even without a rollback, a reader
        never finds an index entry pointing at data that was not written.
        """
        row, start = before.profiles, before.positions
        count = int(arrays["z_um"].shape[0])
        stop = start + count
        for name, values in arrays.items():
            ds = self.arrays[name]
            ds.resize(stop, axis=0)
            if count:
                ds[start:stop] = values
        self.meta.resize((row + 1,))
        self.meta[row] = meta.model_dump_json()
        self.index.resize((row + 1, 3))
        self.index[row] = (meta.point_id, start, count)
        return row

    def write_point(self, before: _Lengths, point: ScanPoint, profile_row: int | None) -> None:
        self.points.resize((before.points + 1,))
        self.points[before.points : before.points + 1] = layout.point_record(point, profile_row)

    def mark_uncommitted(self, row: int) -> None:
        """Add ``/points`` row ``row`` to ``/points_uncommitted`` (its data stays)."""
        rows = layout.uncommitted_rows(self.file)
        if row not in rows:
            layout.write_uncommitted_rows(self.file, [*rows, row])

    def truncate(self, before: _Lengths) -> None:
        """Shrink every growing dataset back to ``before`` (undo an incomplete append)."""
        self.points.resize((before.points,))
        self.index.resize(before.profiles, axis=0)
        self.meta.resize((before.profiles,))
        for ds in self.arrays.values():
            ds.resize(before.positions, axis=0)


def fsync_file(path: Path) -> None:
    """Force the file's data to stable storage.

    Uses a separate descriptor: ``fsync`` (Linux) and ``FlushFileBuffers``
    (Windows, which needs write access) act on the file, not on the handle that
    wrote it.
    """
    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_directory(path: Path) -> None:
    """Make a new directory entry durable (POSIX; NTFS journals it, Windows cannot fsync dirs)."""
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _hdf5_errors(action: str) -> Iterator[None]:
    """Report HDF5 / filesystem failures as :class:`StorageError`."""
    try:
        yield
    except (OSError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        raise StorageError(f"HDF5 {action} failed: {exc}") from exc


class HDF5ScanStore:
    """Thread-safe owner of every scan file in ``scans_dir`` (created if missing)."""

    def __init__(self, scans_dir: Path, *, fsync: bool = True) -> None:
        self._dir = Path(scans_dir).resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync
        self._lock = threading.RLock()
        self._writers: dict[str, _OpenScan] = {}
        self._closed = False

    @property
    def scans_dir(self) -> Path:
        return self._dir

    def path_for(self, scan_id: str) -> Path:
        """``<scans_dir>/<scan_id>.h5``; rejects ids that are not safe file names."""
        if not _SCAN_ID_PATTERN.fullmatch(scan_id):
            raise StorageError(
                f"invalid scan id {scan_id!r}: use 1-128 letters, digits, '.', '_' or '-'"
            )
        return self._dir / f"{scan_id}.h5"

    def is_open(self, scan_id: str) -> bool:
        """True while the scan's writable handle is cached (the scan is being acquired)."""
        with self._lock:
            return scan_id in self._writers

    # ------------------------------------------------------------------ writing

    def create_scan(
        self,
        *,
        scan_id: str,
        created_at: datetime,
        config: ScanConfig,
        calibration: CalibrationState,
        software_version: str,
        hardware: HardwareInfo,
    ) -> Path:
        """Create the scan's file (never overwriting one) and keep it open for writing."""
        path = self.path_for(scan_id)
        with self._lock:
            self._require_open()
            if path.exists():
                raise StorageError(f"HDF5 file {path} already exists; it is never overwritten")
            with _hdf5_errors(f"create {path.name}"):
                f = h5py.File(path, "w-", libver=layout.LIBVER, locking=_LOCKING)
                try:
                    layout.initialise_file(
                        f,
                        scan_id=scan_id,
                        created_at=created_at,
                        config=config,
                        calibration=calibration,
                        software_version=software_version,
                        hardware=hardware,
                    )
                    f.flush()
                    self._writers[scan_id] = _OpenScan(path, f)
                except BaseException:
                    f.close()
                    raise
            self._sync(path)
            if self._fsync:
                fsync_directory(self._dir)
        return path

    @contextmanager
    def append_point(
        self,
        scan_id: str,
        point: ScanPoint,
        profile: ProfileData | None,
        analysis: ProfileAnalysis | None,
    ) -> Iterator[int | None]:
        """Transactional append of one (sanitised) point.

        On entry the point and its arrays are written, flushed and fsync'ed; the
        context yields the profile's row in ``/profiles/index`` (``None`` without
        a profile). If that write does not complete, the partial data is
        truncated away. If the ``with`` body raises - e.g. the SQLite commit
        fails - the durable data is kept and the point's ``/points`` row is
        marked in ``/points_uncommitted`` (best effort: if even that fails, the
        repository's reconciliation marks it later). The store lock is held for
        the whole block.

        Raises:
            StorageError: invalid profile (nothing written) or HDF5 failure
                (partial writes rolled back).
        """
        with self._lock:
            self._require_open()
            scan = self._writer(scan_id)
            action = f"append point {point.point_id} to {scan.path.name}"
            encoded: tuple[dict[str, NDArray[Any]], layout.ProfileMeta] | None = None
            if profile is not None:  # validated before anything is written
                encoded = (
                    layout.encode_profile(profile, scan.samples_per_z),
                    layout.profile_meta(point.point_id, profile, analysis),
                )
            with _hdf5_errors(action):
                before = scan.lengths()
            try:
                with _hdf5_errors(action):
                    profile_row = None
                    if encoded is not None:
                        profile_row = scan.write_profile(before, *encoded)
                    scan.write_point(before, point, profile_row)
                    scan.file.flush()
                self._sync(scan.path)
            except BaseException:
                self._rollback(scan, before)
                raise
            try:
                yield profile_row
            except BaseException:
                self._mark_uncommitted(scan, before.points)
                raise

    def update_calibration(self, scan_id: str, calibration: CalibrationState) -> None:
        """Replace the root ``calibration_json`` attribute (the scan re-calibrated)."""
        path = self.path_for(scan_id)
        with self._lock, self._handle(scan_id, writable=True) as f:
            with _hdf5_errors(f"update calibration of {path.name}"):
                f.attrs["calibration_json"] = calibration.model_dump_json()
                f.flush()
            self._sync(path)

    def write_surface(self, surface: SurfaceResult) -> str:
        """Store a surface whose ``surface_id`` is assigned; returns its group path."""
        if surface.surface_id is None:
            raise StorageError("surface_id must be assigned before the surface is stored")
        group = f"/{layout.SURFACES}/{surface.surface_id}"
        self._write_group(surface.scan_id, group, lambda g: results.write_surface(g, surface))
        return group

    def write_ml_result(self, result: MLResult) -> str:
        """Store an ML result whose ``result_id`` is assigned; returns its group path."""
        if result.result_id is None:
            raise StorageError("result_id must be assigned before the ML result is stored")
        group = f"/{layout.ML}/{result.result_id}"
        self._write_group(result.scan_id, group, lambda g: results.write_ml_result(g, result))
        return group

    def finish_scan(self, scan_id: str) -> None:
        """Flush, sync and close the scan's writable handle (no-op if it is not open)."""
        with self._lock:
            scan = self._writers.pop(scan_id, None)
            if scan is not None:
                self._close_writer(scan)

    def close(self) -> None:
        """Close every open file (idempotent). The store refuses further use."""
        with self._lock:
            self._closed = True
            while self._writers:
                _scan_id, scan = self._writers.popitem()
                self._close_writer(scan)

    # ------------------------------------------------------------------ reading

    def read_profile(self, scan_id: str, profile_row: int, point_id: int) -> StoredProfile:
        """Arrays and metadata of index row ``profile_row`` (which must belong to ``point_id``)."""
        path = self.path_for(scan_id)
        with (
            self._lock,
            self._handle(scan_id, writable=False) as f,
            _hdf5_errors(f"read point {point_id} from {path.name}"),
        ):
            index = f[layout.INDEX]
            if not 0 <= profile_row < index.shape[0]:
                raise StorageError(f"{path.name}: profile row {profile_row} does not exist")
            stored_id, start, count = (int(v) for v in index[profile_row])
            if stored_id != point_id:
                raise StorageError(
                    f"{path.name}: profile row {profile_row} belongs to point {stored_id}, "
                    f"not {point_id}"
                )
            arrays = {
                name: np.asarray(f[f"{layout.PROFILES}/{name}"][start : start + count])
                for name in _ARRAYS
            }
            meta = layout.ProfileMeta.model_validate_json(f[layout.META][profile_row])
        return StoredProfile(data=layout.decode_profile(arrays, meta), analysis=meta.analysis)

    def read_surface(self, scan_id: str, group: str) -> SurfaceResult:
        return self._read_group(scan_id, group, results.read_surface)

    def read_ml_result(self, scan_id: str, group: str) -> MLResult:
        return self._read_group(scan_id, group, results.read_ml_result)

    def file_points(self, scan_id: str) -> FilePoints:
        """Every ``/points`` row's key and the uncommitted marks (StorageError if unreadable)."""
        path = self.path_for(scan_id)
        with (
            self._lock,
            self._handle(scan_id, writable=False) as f,
            _hdf5_errors(f"read {path.name}"),
        ):
            return FilePoints(
                keys=layout.point_keys(f),
                uncommitted=frozenset(layout.uncommitted_rows(f)),
            )

    def set_uncommitted(self, scan_id: str, rows: list[int]) -> None:
        """Make ``/points_uncommitted`` list exactly ``rows`` (no write if it already does).

        Only marks change: no point data is ever removed.
        """
        path = self.path_for(scan_id)
        wanted = sorted(set(rows))
        with self._lock:
            with self._handle(scan_id, writable=False) as f, _hdf5_errors(f"read {path.name}"):
                n_rows = int(f[layout.POINTS].shape[0])
                if layout.uncommitted_rows(f) == wanted:
                    return
            if any(not 0 <= row < n_rows for row in wanted):
                raise StorageError(f"{path.name}: uncommitted rows {wanted} exceed /points")
            with self._handle(scan_id, writable=True) as f:
                with _hdf5_errors(f"mark uncommitted points in {path.name}"):
                    layout.write_uncommitted_rows(f, wanted)
                    f.flush()
                self._sync(path)

    # ------------------------------------------------------------------ internals

    def _require_open(self) -> None:
        if self._closed:
            raise StorageError(f"HDF5 store {self._dir} is closed")

    def _open(self, path: Path, *, writable: bool) -> Any:
        if not path.exists():
            raise StorageError(f"HDF5 file {path} is missing")
        with _hdf5_errors(f"open {path.name}"):
            if writable:
                return h5py.File(path, "r+", libver=layout.LIBVER, locking=_LOCKING)
            return h5py.File(path, "r", locking=_LOCKING)

    def _writer(self, scan_id: str) -> _OpenScan:
        """The scan's cached writable handle, opened (and cached) if needed. Lock held."""
        scan = self._writers.get(scan_id)
        if scan is None:
            path = self.path_for(scan_id)
            f = self._open(path, writable=True)
            try:
                with _hdf5_errors(f"open {path.name}"):
                    scan = _OpenScan(path, f)
            except BaseException:
                f.close()
                raise
            self._writers[scan_id] = scan
        return scan

    @contextmanager
    def _handle(self, scan_id: str, *, writable: bool) -> Iterator[Any]:
        """The cached writer's file if the scan is open, else a handle closed after the block."""
        self._require_open()
        cached = self._writers.get(scan_id)
        if cached is not None:
            yield cached.file
            return
        f = self._open(self.path_for(scan_id), writable=writable)
        try:
            yield f
        finally:
            f.close()

    def _write_group(self, scan_id: str, group: str, write: Callable[[Any], None]) -> None:
        path = self.path_for(scan_id)
        with self._lock, self._handle(scan_id, writable=True) as f:
            with _hdf5_errors(f"write {group} to {path.name}"):
                if group in f:
                    raise StorageError(f"{path.name}: {group} already exists")
                write(f.create_group(group))
                f.flush()
            self._sync(path)

    def _read_group(self, scan_id: str, group: str, read: Callable[[Any], _T]) -> _T:
        path = self.path_for(scan_id)
        with (
            self._lock,
            self._handle(scan_id, writable=False) as f,
            _hdf5_errors(f"read {group} from {path.name}"),
        ):
            if group not in f:
                raise StorageError(f"{path.name}: {group} is missing")
            return read(f[group])

    def _mark_uncommitted(self, scan: _OpenScan, row: int) -> None:
        """Keep a durable point whose commit failed, marked; a failure here is only logged."""
        try:
            with _hdf5_errors(f"mark point row {row} of {scan.path.name} uncommitted"):
                scan.mark_uncommitted(row)
                scan.file.flush()
            self._sync(scan.path)
        except Exception:
            logger.exception(
                "could not mark /points row %d of %s uncommitted (its data is kept; "
                "reconciliation will mark it)",
                row,
                scan.path,
            )

    def _rollback(self, scan: _OpenScan, before: _Lengths) -> None:
        """Undo an incomplete append; a failure here is logged, the original error re-raised."""
        try:
            scan.truncate(before)
            scan.file.flush()
            self._sync(scan.path)
        except Exception:
            logger.exception("could not roll back a failed append in %s", scan.path)

    def _sync(self, path: Path) -> None:
        if self._fsync:
            with _hdf5_errors(f"fsync {path.name}"):
                fsync_file(path)

    def _close_writer(self, scan: _OpenScan) -> None:
        try:
            with _hdf5_errors(f"close {scan.path.name}"):
                scan.file.flush()
                scan.file.close()
            self._sync(scan.path)
        except StorageError:
            logger.exception("closing %s failed", scan.path)
