"""Surface / ML round trips and calibration versioning."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.storage.conftest import (
    create_scan,
    make_calibration,
    make_ml_result,
    make_surface,
    open_repository,
)

from confocal.errors import ScanNotFoundError, StorageError
from confocal.models import ScanState
from confocal.storage import SQLiteHDF5Repository


def test_surface_round_trip_is_exact(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    assert repo.get_latest_surface("scan-001") is None
    surface = make_surface("scan-001")
    stored = repo.save_surface(surface)
    assert stored.surface_id is not None
    assert stored == surface.model_copy(update={"surface_id": stored.surface_id})
    assert repo.get_latest_surface("scan-001") == stored
    assert repo.get_scan("scan-001").has_surface
    assert not repo.get_scan("scan-001").has_ml_result


def test_latest_surface_is_the_most_recent(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    first = repo.save_surface(make_surface("scan-001"))
    second_input = make_surface("scan-001").model_copy(update={"x_um": [5.0, 15.0, 25.0]})
    second = repo.save_surface(second_input)
    assert first.surface_id is not None
    assert second.surface_id is not None
    assert second.surface_id > first.surface_id
    assert repo.get_latest_surface("scan-001") == second


def test_surface_of_finished_scan_can_be_added_and_survives_reopen(tmp_path: Path) -> None:
    repo = open_repository(tmp_path)
    create_scan(repo)
    repo.update_scan("scan-001", state=ScanState.COMPLETE)  # writable handle closed
    stored = repo.save_surface(make_surface("scan-001"))
    repo.close()
    with open_repository(tmp_path) as reopened:
        assert reopened.get_latest_surface("scan-001") == stored
        assert reopened.get_scan("scan-001").has_surface


def test_surface_with_mismatched_grid_is_rejected(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    bad = make_surface("scan-001").model_copy(update={"x_um": [0.0, 10.0]})
    with pytest.raises(StorageError, match="shape"):
        repo.save_surface(bad)
    assert repo.get_latest_surface("scan-001") is None
    assert not repo.get_scan("scan-001").has_surface
    good = repo.save_surface(make_surface("scan-001"))  # the failed id stays reserved
    assert repo.get_latest_surface("scan-001") == good


def test_surface_for_unknown_scan(repo: SQLiteHDF5Repository) -> None:
    with pytest.raises(ScanNotFoundError):
        repo.save_surface(make_surface("nope"))


def test_ml_result_round_trip_is_exact(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    assert repo.get_latest_ml_result("scan-001") is None
    result = make_ml_result("scan-001")
    stored = repo.save_ml_result(result)
    assert stored.result_id is not None
    assert stored == result.model_copy(update={"result_id": stored.result_id})
    assert repo.get_latest_ml_result("scan-001") == stored
    assert repo.get_scan("scan-001").has_ml_result
    second = repo.save_ml_result(result.model_copy(update={"threshold": 0.7}))
    assert repo.get_latest_ml_result("scan-001") == second


def test_ml_result_with_no_predictions(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    empty = make_ml_result("scan-001").model_copy(
        update={"predictions": [], "n_points": 0, "n_flagged": 0}
    )
    stored = repo.save_ml_result(empty)
    assert repo.get_latest_ml_result("scan-001") == stored


def test_ml_result_for_unknown_scan(repo: SQLiteHDF5Repository) -> None:
    with pytest.raises(ScanNotFoundError):
        repo.save_ml_result(make_ml_result("nope"))


def test_calibration_versions_are_monotonic(repo: SQLiteHDF5Repository) -> None:
    assert repo.latest_calibration() is None
    assert repo.list_calibrations() == []
    saved = [repo.save_calibration(make_calibration(dark_v=0.01 * i)) for i in range(1, 5)]
    versions = [s.version for s in saved if s.version is not None]
    assert len(versions) == 4
    assert versions == sorted(set(versions))  # strictly increasing
    assert repo.latest_calibration() == saved[-1]
    assert [c.version for c in repo.list_calibrations()] == list(reversed(versions))
    assert repo.list_calibrations(limit=2) == [saved[3], saved[2]]
    for state in saved:
        assert state.version is not None
        assert repo.get_calibration(state.version) == state
    assert repo.get_calibration(9999) is None


def test_incoming_calibration_version_is_replaced(repo: SQLiteHDF5Repository) -> None:
    first = repo.save_calibration(make_calibration(version=42))
    assert first.version == 1
    second = repo.save_calibration(make_calibration(version=first.version))
    assert first.version is not None
    assert second.version is not None
    assert second.version > first.version
    assert second.dark_v == first.dark_v


def test_calibration_versions_continue_after_reopen(tmp_path: Path) -> None:
    repo = open_repository(tmp_path)
    first = repo.save_calibration(make_calibration())
    repo.close()
    with open_repository(tmp_path) as reopened:
        second = reopened.save_calibration(make_calibration())
        assert first.version is not None
        assert second.version is not None
        assert second.version > first.version
        assert reopened.list_calibrations() == [second, first]
