"""Persistence: SQLite metadata (SQLModel) and HDF5 measurement arrays.

:class:`SQLiteHDF5Repository` implements both storage protocols of
:mod:`confocal.storage.interfaces` (:class:`ScanRepository` and
:class:`CalibrationStore`). It is synchronous and thread-safe; call it from the
event loop through ``asyncio.to_thread``.
"""

from confocal.storage.database import SCHEMA_VERSION, Database
from confocal.storage.hdf5 import FilePoints, HDF5ScanStore, StoredProfile
from confocal.storage.interfaces import CalibrationStore, ScanRepository
from confocal.storage.layout import FORMAT_VERSION
from confocal.storage.records import AuditEvent
from confocal.storage.repository import (
    EVENT_POINTS_UNCOMMITTED,
    EVENT_SCAN_INTERRUPTED,
    SQLiteHDF5Repository,
)

__all__ = [
    "EVENT_POINTS_UNCOMMITTED",
    "EVENT_SCAN_INTERRUPTED",
    "FORMAT_VERSION",
    "SCHEMA_VERSION",
    "AuditEvent",
    "CalibrationStore",
    "Database",
    "FilePoints",
    "HDF5ScanStore",
    "SQLiteHDF5Repository",
    "ScanRepository",
    "StoredProfile",
]
