"""SQLite engine and session management.

Connection settings and why they matter on the Pi:

* ``journal_mode=WAL`` - readers (HTTP requests listing scans and points) never
  block the writer (the scan appending points) and vice versa.
* ``synchronous=FULL`` (default) - in WAL mode every commit is fsync'ed, so a
  committed point survives a power cut. ``NORMAL`` only survives a process
  crash; it is offered for tests and throwaway simulations.
* ``foreign_keys=ON`` - SQLite enforces FOREIGN KEY constraints only when asked,
  per connection.
* ``busy_timeout`` - another process (e.g. an offline analysis script) holding a
  write lock makes us wait instead of failing immediately with SQLITE_BUSY.
* ``check_same_thread=False`` - callers run in ``asyncio.to_thread`` worker
  threads; the pool hands each session its own connection.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from sqlalchemy import URL, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, SQLModel

from confocal.errors import StorageError
from confocal.storage.tables import storage_tables

#: Bumped whenever the schema changes incompatibly (stored in ``PRAGMA user_version``).
SCHEMA_VERSION = 1
DEFAULT_BUSY_TIMEOUT_MS = 10_000

SynchronousMode = Literal["FULL", "NORMAL"]


class Database:
    """SQLite database at ``path`` (parent directories are created)."""

    def __init__(
        self,
        path: Path,
        *,
        synchronous: SynchronousMode = "FULL",
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if synchronous not in ("FULL", "NORMAL"):
            raise ValueError(f"unsupported synchronous mode {synchronous!r}")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be >= 0")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._synchronous: SynchronousMode = synchronous
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._closed = False
        self._engine = create_engine(
            URL.create("sqlite", database=str(self.path)),
            connect_args={"check_same_thread": False, "timeout": busy_timeout_ms / 1000.0},
        )
        event.listen(self._engine, "connect", self._configure_connection)

    @property
    def engine(self) -> Engine:
        return self._engine

    def _configure_connection(self, dbapi_connection: sqlite3.Connection, _record: object) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            cursor.execute(f"PRAGMA synchronous={self._synchronous}")
        finally:
            cursor.close()

    def create_all(self) -> None:
        """Create missing tables and indexes; refuse a database from a newer schema.

        Raises:
            StorageError: the file was written by a newer, incompatible version,
                or it cannot be opened / written.
        """
        self._require_open()
        try:
            with self._engine.begin() as connection:
                version = int(connection.exec_driver_sql("PRAGMA user_version").scalar_one())
                if version > SCHEMA_VERSION:
                    raise StorageError(
                        f"database {self.path} has schema version {version}; this software "
                        f"supports up to {SCHEMA_VERSION}"
                    )
                SQLModel.metadata.create_all(connection, tables=storage_tables())
                if version == 0:
                    connection.exec_driver_sql(f"PRAGMA user_version={SCHEMA_VERSION}")
        except SQLAlchemyError as exc:
            raise StorageError(f"cannot initialise database {self.path}: {exc}") from exc

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Unit of work: commits when the block succeeds, rolls back on any exception.

        Objects stay usable after the block (``expire_on_commit=False``). Database
        failures surface as :class:`StorageError`; every other exception raised in
        the block (e.g. ``ScanNotFoundError``) propagates unchanged.
        """
        self._require_open()
        with Session(self._engine, expire_on_commit=False) as session:
            try:
                yield session
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                raise StorageError(f"database {self.path.name}: {exc}") from exc
            except BaseException:
                session.rollback()
                raise

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Close every pooled connection (SQLite checkpoints the WAL on the last close)."""
        self._closed = True
        self._engine.dispose()

    def _require_open(self) -> None:
        if self._closed:
            raise StorageError(f"database {self.path} is closed")
