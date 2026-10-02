"""Flushed raw data is never deleted: a failed database commit leaves it marked uncommitted."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, NoReturn

import h5py
import numpy as np
import pytest
from sqlalchemy.exc import OperationalError
from sqlmodel import Session
from tests.unit.storage.conftest import (
    create_scan,
    make_point,
    make_profile,
    open_repository,
)

from confocal.errors import PointNotFoundError, StorageError
from confocal.models import ProfileData, ScanState
from confocal.storage import EVENT_POINTS_UNCOMMITTED, EVENT_SCAN_INTERRUPTED
from confocal.storage import hdf5 as hdf5_module
from confocal.storage.repository import SQLiteHDF5Repository


def _failing_commit(_session: Session) -> NoReturn:
    raise OperationalError("COMMIT", {}, sqlite3.OperationalError("disk I/O error"))


def _append_with_failed_commit(
    repo: SQLiteHDF5Repository,
    monkeypatch: pytest.MonkeyPatch,
    point_id: int,
    profile: ProfileData | None,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", _failing_commit)
        with pytest.raises(StorageError, match="raw data is kept"):
            repo.append_point("scan-001", make_point(point_id), profile, None)


def _file(repo: SQLiteHDF5Repository) -> Path:
    return repo.scans_dir / "scan-001.h5"


def test_failed_commit_keeps_flushed_raw_data_marked_uncommitted(
    repo: SQLiteHDF5Repository, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_scan(repo)
    profiles = [make_profile(rng, n_coarse=8 + i) for i in range(3)]
    repo.append_point("scan-001", make_point(0), profiles[0], None)
    _append_with_failed_commit(repo, monkeypatch, 1, profiles[1])
    repo.append_point("scan-001", make_point(2), profiles[2], None)

    # Readers follow SQLite: point 1 is not a stored point of the scan.
    assert [p.point_id for p in repo.get_points("scan-001")] == [0, 2]
    assert repo.get_scan("scan-001").completed_points == 2
    with pytest.raises(PointNotFoundError):
        repo.get_profile("scan-001", 1)
    assert repo.get_profile("scan-001", 2).raw_counts == profiles[2].raw_counts.tolist()
    assert repo.list_events(kind=EVENT_POINTS_UNCOMMITTED) == []  # recorded when the scan ends

    repo.update_scan("scan-001", state=ScanState.ERROR)

    with h5py.File(_file(repo), "r") as f:
        assert f["points"]["point_id"].tolist() == [0, 1, 2]
        assert f["points_uncommitted"][()].tolist() == [1]  # /points row of point 1
        profile_row = int(f["points"]["profile_row"][1])
        point_id, start, count = (int(v) for v in f["profiles/index"][profile_row])
        assert (point_id, count) == (1, profiles[1].n_positions)
        np.testing.assert_array_equal(
            f["profiles/raw_counts"][start : start + count], profiles[1].raw_counts
        )
        np.testing.assert_array_equal(f["profiles/z_um"][start : start + count], profiles[1].z_um)
    (event,) = repo.list_events(kind=EVENT_POINTS_UNCOMMITTED)
    assert event.scan_id == "scan-001"
    assert "point ids [1]" in event.message
    repo.update_scan("scan-001", error_message="again")  # no second reconciliation event
    assert len(repo.list_events(kind=EVENT_POINTS_UNCOMMITTED)) == 1


def test_clean_scan_records_no_uncommitted_event(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    repo.append_point("scan-001", make_point(0), None, None)
    repo.update_scan("scan-001", state=ScanState.COMPLETE)
    assert repo.list_events(kind=EVENT_POINTS_UNCOMMITTED) == []
    with h5py.File(_file(repo), "r") as f:
        assert f["points_uncommitted"].shape == (0,)


def test_point_reacquired_after_failed_commit(
    repo: SQLiteHDF5Repository, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_scan(repo)
    first, second = make_profile(rng), make_profile(rng)
    _append_with_failed_commit(repo, monkeypatch, 0, first)
    _append_with_failed_commit(repo, monkeypatch, 1, None)
    repo.append_point("scan-001", make_point(0), second, None)  # retried: stored
    repo.append_point("scan-001", make_point(1), None, None)

    assert [p.point_id for p in repo.get_points("scan-001")] == [0, 1]
    assert repo.get_profile("scan-001", 0).raw_counts == second.raw_counts.tolist()
    repo.update_scan("scan-001", state=ScanState.COMPLETE)
    with h5py.File(_file(repo), "r") as f:
        assert f["points"]["point_id"].tolist() == [0, 1, 0, 1]
        assert f["points_uncommitted"][()].tolist() == [0, 1]
    (event,) = repo.list_events(kind=EVENT_POINTS_UNCOMMITTED)
    assert "point ids [0, 1]" in event.message


def test_unmarked_point_is_found_and_audited_by_recovery(
    tmp_path: Path, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Commit and marking both fail (as after a power cut between the two writes)."""

    def failing_mark(_self: Any, _row: int) -> NoReturn:
        raise OSError("disk full")

    repo = open_repository(tmp_path)
    create_scan(repo)
    repo.update_scan("scan-001", state=ScanState.SCANNING)
    repo.append_point("scan-001", make_point(0), None, None)
    profile = make_profile(rng)
    with monkeypatch.context() as patch:
        patch.setattr(hdf5_module._OpenScan, "mark_uncommitted", failing_mark)
        _append_with_failed_commit(repo, monkeypatch, 1, profile)
    repo.close()  # the server stops with the scan still active
    with h5py.File(tmp_path / "scans" / "scan-001.h5", "r") as f:
        assert f["points"].shape == (2,)  # data kept even though it could not be marked
        assert f["points_uncommitted"].shape == (0,)

    with open_repository(tmp_path) as reopened:
        assert reopened.recover_interrupted_scans() == ["scan-001"]
        message = reopened.get_scan("scan-001").error_message or ""
        assert "1 of 12 points" in message
        assert "1 further point(s)" in message
        assert "point ids [1]" in message
        (event,) = reopened.list_events(kind=EVENT_POINTS_UNCOMMITTED)
        assert "point ids [1]" in event.message
        assert len(reopened.list_events(kind=EVENT_SCAN_INTERRUPTED)) == 1
        with pytest.raises(PointNotFoundError):
            reopened.get_profile("scan-001", 1)
    with h5py.File(tmp_path / "scans" / "scan-001.h5", "r") as f:
        assert f["points_uncommitted"][()].tolist() == [1]
        np.testing.assert_array_equal(f["profiles/raw_counts"][()], profile.raw_counts)


def test_marks_are_added_to_files_without_the_dataset(
    tmp_path: Path, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    with open_repository(tmp_path) as repo:
        create_scan(repo)
        repo.update_scan("scan-001", state=ScanState.SCANNING)
    with h5py.File(tmp_path / "scans" / "scan-001.h5", "r+") as f:
        del f["points_uncommitted"]  # a file written before the dataset existed

    with open_repository(tmp_path) as repo:
        repo.append_point("scan-001", make_point(0), None, None)
        _append_with_failed_commit(repo, monkeypatch, 1, make_profile(rng))
        repo.update_scan("scan-001", state=ScanState.CANCELLED)
        assert len(repo.list_events(kind=EVENT_POINTS_UNCOMMITTED)) == 1
    with h5py.File(tmp_path / "scans" / "scan-001.h5", "r") as f:
        assert f["points_uncommitted"][()].tolist() == [1]


def test_incomplete_hdf5_write_is_still_rolled_back(
    repo: SQLiteHDF5Repository, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_write(_self: Any, *_args: Any) -> NoReturn:
        raise OSError("write failed")

    create_scan(repo)
    repo.append_point("scan-001", make_point(0), make_profile(rng), None)
    with monkeypatch.context() as patch:
        patch.setattr(hdf5_module._OpenScan, "write_point", failing_write)
        with pytest.raises(StorageError, match="write failed"):
            repo.append_point("scan-001", make_point(1), make_profile(rng), None)
    repo.update_scan("scan-001", state=ScanState.COMPLETE)
    with h5py.File(_file(repo), "r") as f:
        assert f["points"].shape == (1,)
        assert f["profiles/index"].shape == (1, 3)
        assert f["profiles/z_um"].shape == (make_profile(rng).n_positions,)
        assert f["points_uncommitted"].shape == (0,)
    assert repo.list_events(kind=EVENT_POINTS_UNCOMMITTED) == []
