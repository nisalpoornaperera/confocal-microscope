"""Scan lifecycle: create / update / get / list, recovery, audit log, close and reopen."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.unit.storage.conftest import (
    SOFTWARE_VERSION,
    create_scan,
    make_calibration,
    make_config,
    make_hardware,
    make_point,
    open_repository,
)

from confocal.errors import ScanNotFoundError, StorageError
from confocal.models import ACTIVE_SCAN_STATES, ScanMode, ScanState
from confocal.storage import (
    EVENT_SCAN_INTERRUPTED,
    CalibrationStore,
    ScanRepository,
    SQLiteHDF5Repository,
)


def test_implements_both_protocols(repo: SQLiteHDF5Repository) -> None:
    assert isinstance(repo, ScanRepository)
    assert isinstance(repo, CalibrationStore)


def test_create_scan_returns_idle_summary_with_snapshots(repo: SQLiteHDF5Repository) -> None:
    config = make_config(name="coin")
    calibration = make_calibration(version=7)
    summary = repo.create_scan(
        scan_id="scan-a",
        config=config,
        total_points=config.total_points,
        calibration=calibration,
        software_version=SOFTWARE_VERSION,
        hardware=make_hardware(),
    )
    assert summary.id == "scan-a"
    assert summary.name == "coin"
    assert summary.mode is ScanMode.CONFOCAL
    assert summary.state is ScanState.IDLE
    assert summary.total_points == 12
    assert summary.completed_points == 0
    assert summary.progress == 0.0
    assert summary.config == config
    assert summary.calibration == calibration
    assert summary.calibration_version == 7
    assert summary.hardware == make_hardware()
    assert summary.software_version == SOFTWARE_VERSION
    assert summary.created_at.tzinfo is not None
    assert not summary.interrupted
    assert not summary.has_surface
    assert not summary.has_ml_result
    assert summary.data_file is not None
    assert Path(summary.data_file) == repo.scans_dir / "scan-a.h5"
    assert Path(summary.data_file).is_file()
    assert repo.get_scan("scan-a") == summary


def test_duplicate_scan_id_is_rejected_and_nothing_is_overwritten(
    repo: SQLiteHDF5Repository,
) -> None:
    create_scan(repo, "scan-a")
    before = (repo.scans_dir / "scan-a.h5").stat().st_size
    with pytest.raises(StorageError, match="already exists"):
        create_scan(repo, "scan-a")
    assert (repo.scans_dir / "scan-a.h5").stat().st_size == before
    assert len(repo.list_scans()) == 1


@pytest.mark.parametrize("scan_id", ["../escape", "a/b", "", ".hidden", "x" * 129, "a b"])
def test_unsafe_scan_ids_are_rejected(repo: SQLiteHDF5Repository, scan_id: str) -> None:
    with pytest.raises(StorageError, match="invalid scan id"):
        create_scan(repo, scan_id)
    assert repo.list_scans() == []


def test_update_scan_changes_only_given_fields(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo, "scan-a")
    started = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
    updated = repo.update_scan("scan-a", state=ScanState.SCANNING, started_at=started)
    assert updated.state is ScanState.SCANNING
    assert updated.started_at == started
    assert updated.finished_at is None
    assert updated.error_message is None

    updated = repo.update_scan("scan-a", completed_points=6)
    assert updated.state is ScanState.SCANNING  # untouched
    assert updated.started_at == started  # untouched
    assert updated.completed_points == 6
    assert updated.progress == pytest.approx(0.5)

    finished = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    updated = repo.update_scan(
        "scan-a",
        state=ScanState.ERROR,
        finished_at=finished,
        error_message="stage stalled",
        interrupted=True,
    )
    assert updated.state is ScanState.ERROR
    assert updated.finished_at == finished
    assert updated.error_message == "stage stalled"
    assert updated.interrupted
    assert updated.completed_points == 6
    assert repo.get_scan("scan-a") == updated


def test_naive_datetimes_are_taken_as_utc(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo, "scan-a")
    updated = repo.update_scan("scan-a", started_at=datetime(2026, 10, 1, 8, 0))
    assert updated.started_at == datetime(2026, 10, 1, 8, 0, tzinfo=UTC)


def test_update_scan_calibration_updates_summary_and_hdf5(repo: SQLiteHDF5Repository) -> None:
    import h5py

    create_scan(repo, "scan-a")
    new_calibration = make_calibration(version=9, dark_v=0.02)
    updated = repo.update_scan("scan-a", calibration=new_calibration)
    assert updated.calibration == new_calibration
    assert updated.calibration_version == 9
    repo.update_scan("scan-a", state=ScanState.COMPLETE)  # closes the writable handle
    with h5py.File(repo.scans_dir / "scan-a.h5", "r") as f:
        assert f.attrs["calibration_json"] == new_calibration.model_dump_json()


def test_unknown_scan_raises_scan_not_found(repo: SQLiteHDF5Repository) -> None:
    with pytest.raises(ScanNotFoundError):
        repo.get_scan("nope")
    with pytest.raises(ScanNotFoundError):
        repo.update_scan("nope", state=ScanState.SCANNING)
    with pytest.raises(ScanNotFoundError):
        repo.get_points("nope")
    with pytest.raises(ScanNotFoundError):
        repo.get_profile("nope", 0)
    with pytest.raises(ScanNotFoundError):
        repo.append_point("nope", make_point(0), None, None)
    with pytest.raises(ScanNotFoundError):
        repo.get_latest_surface("nope")
    with pytest.raises(ScanNotFoundError):
        repo.get_latest_ml_result("nope")


def test_list_scans_newest_first_with_pagination(repo: SQLiteHDF5Repository) -> None:
    ids = [f"scan-{i:02d}" for i in range(7)]
    for scan_id in ids:
        create_scan(repo, scan_id)
    newest_first = list(reversed(ids))
    assert [s.id for s in repo.list_scans()] == newest_first
    assert [s.id for s in repo.list_scans(limit=3)] == newest_first[:3]
    assert [s.id for s in repo.list_scans(limit=3, offset=3)] == newest_first[3:6]
    assert [s.id for s in repo.list_scans(limit=3, offset=6)] == newest_first[6:]
    assert repo.list_scans(offset=10) == []
    assert repo.list_scans(limit=0) == []
    with pytest.raises(ValueError, match="limit must be >= 0"):
        repo.list_scans(limit=-1)


def test_recover_interrupted_scans_marks_active_scans(repo: SQLiteHDF5Repository) -> None:
    active = sorted(ACTIVE_SCAN_STATES)
    for i, state in enumerate(active):
        create_scan(repo, f"active-{i}")
        repo.update_scan(f"active-{i}", state=state)
    create_scan(repo, "idle")
    create_scan(repo, "done")
    repo.update_scan("done", state=ScanState.COMPLETE)
    create_scan(repo, "cancelled")
    repo.update_scan("cancelled", state=ScanState.CANCELLED)
    repo.append_point("active-0", make_point(0), None, None)

    recovered = repo.recover_interrupted_scans()

    assert recovered == [f"active-{i}" for i in range(len(active))]  # oldest first
    for i, state in enumerate(active):
        summary = repo.get_scan(f"active-{i}")
        assert summary.state is ScanState.ERROR
        assert summary.interrupted
        assert summary.error_message is not None
        assert state.value in summary.error_message
    assert "1 of 12 points" in (repo.get_scan("active-0").error_message or "")
    assert repo.get_scan("idle").state is ScanState.IDLE
    assert repo.get_scan("done").state is ScanState.COMPLETE
    assert repo.get_scan("cancelled").state is ScanState.CANCELLED
    assert not repo.get_scan("done").interrupted
    events = repo.list_events(kind=EVENT_SCAN_INTERRUPTED)
    assert sorted(e.scan_id or "" for e in events) == sorted(recovered)
    assert repo.recover_interrupted_scans() == []  # idempotent


def test_record_event_and_list_events(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo, "scan-a")
    repo.record_event("estop", "emergency stop pressed")
    repo.record_event("state", "idle -> scanning", scan_id="scan-a")
    repo.record_event("hardware", "ADC timeout", scan_id="scan-a")
    repo.record_event("state", "for a scan that failed to be created", scan_id="ghost")

    events = repo.list_events()
    assert [e.kind for e in events] == ["state", "hardware", "state", "estop"]  # newest first
    assert all(e.created_at.tzinfo is not None for e in events)
    assert [e.message for e in repo.list_events(scan_id="scan-a")] == [
        "ADC timeout",
        "idle -> scanning",
    ]
    assert [e.scan_id for e in repo.list_events(kind="state")] == ["ghost", "scan-a"]
    assert len(repo.list_events(limit=2)) == 2


def test_close_then_reopen_sees_everything(tmp_path: Path) -> None:
    repo = open_repository(tmp_path, durable=True)
    create_scan(repo, "scan-a")
    repo.update_scan("scan-a", state=ScanState.SCANNING)
    for point_id in range(3):
        repo.append_point("scan-a", make_point(point_id), None, None)
    repo.record_event("state", "scanning", scan_id="scan-a")
    calibration = repo.save_calibration(make_calibration())
    before = repo.get_scan("scan-a")
    points = repo.get_points("scan-a")
    repo.close()
    repo.close()  # idempotent
    with pytest.raises(StorageError):
        repo.get_scan("scan-a")

    with open_repository(tmp_path) as reopened:
        assert reopened.get_scan("scan-a") == before
        assert reopened.get_points("scan-a") == points
        assert reopened.latest_calibration() == calibration
        assert [e.message for e in reopened.list_events()] == ["scanning"]
        # The scan was still SCANNING when the process "stopped": recovery marks it.
        assert reopened.recover_interrupted_scans() == ["scan-a"]
        # Appending to a reopened file works (the writable handle is re-created).
        reopened.append_point("scan-a", make_point(3), None, None)
        assert reopened.get_scan("scan-a").completed_points == 4


def test_database_from_newer_schema_is_refused(tmp_path: Path) -> None:
    import sqlite3

    open_repository(tmp_path).close()
    connection = sqlite3.connect(tmp_path / "confocal.db")
    connection.execute("PRAGMA user_version=99")
    connection.close()
    with pytest.raises(StorageError, match="schema version 99"):
        open_repository(tmp_path)
