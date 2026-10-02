"""Points and raw I(Z) profiles: exact round trips, validation, ordering, durability."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from tests.unit.storage.conftest import (
    create_scan,
    make_analysis,
    make_config,
    make_fixed_z_profile,
    make_point,
    make_profile,
    open_repository,
)

from confocal.errors import PointNotFoundError, StorageError
from confocal.models import (
    PointStatus,
    ProfileData,
    ProfileRecord,
    ScanMode,
    ScanState,
    nan_to_none,
)
from confocal.storage import SQLiteHDF5Repository


def assert_record_matches(record: ProfileRecord, profile: ProfileData) -> None:
    """Every stored value equals the acquired one, bit for bit."""
    assert record.z_um == profile.z_um.tolist()
    assert record.z_reported_um == profile.z_reported_um.tolist()
    assert record.phase == [int(p) for p in profile.phase]
    assert record.raw_counts == profile.raw_counts.tolist()
    assert record.voltage_v == profile.voltage_v.tolist()
    assert record.voltage_agg_v == profile.voltage_agg_v.tolist()
    assert record.timestamps == profile.timestamps.tolist()
    assert record.gain is profile.gain
    assert record.sampling_method is profile.sampling_method
    assert record.dark_v == profile.dark_v
    assert record.reference_v == profile.reference_v
    assert record.calibration_version == profile.calibration_version
    expected_normalized = None if profile.normalized is None else nan_to_none(profile.normalized)
    expected_filtered = None if profile.filtered is None else nan_to_none(profile.filtered)
    assert record.normalized == expected_normalized
    assert record.filtered == expected_filtered


def test_append_and_read_back_exact_profiles(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    create_scan(repo)
    profiles = [make_profile(rng, n_coarse=10 + i, n_fine=5 + 2 * i) for i in range(5)]
    for point_id, profile in enumerate(profiles):
        point = make_point(point_id, n_z_positions=profile.n_positions)
        repo.append_point("scan-001", point, profile, make_analysis(point.surface_z_um or 0.0))

    for point_id, profile in enumerate(profiles):
        record = repo.get_profile("scan-001", point_id)
        assert record.scan_id == "scan-001"
        assert record.point_id == point_id
        point = make_point(point_id)
        assert (record.x_um, record.y_um) == (point.x_um, point.y_um)
        assert_record_matches(record, profile)
        assert record.analysis == make_analysis(point.surface_z_um or 0.0)
        assert record.filtered is not None
        assert record.filtered[:2] == [None, None]  # NaN -> None
    assert repo.get_scan("scan-001").completed_points == 5


def test_points_round_trip_exactly(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    points = [make_point(i) for i in range(12)]
    for point in points:
        repo.append_point("scan-001", point, None, None)
    assert repo.get_points("scan-001") == points


def test_get_points_ordered_since_and_limit(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    for point_id in (3, 0, 2, 1, 5, 4):  # stored out of order
        repo.append_point("scan-001", make_point(point_id), None, None)
    assert [p.point_id for p in repo.get_points("scan-001")] == [0, 1, 2, 3, 4, 5]
    assert [p.point_id for p in repo.get_points("scan-001", since_point_id=2)] == [3, 4, 5]
    assert [p.point_id for p in repo.get_points("scan-001", since_point_id=-1, limit=2)] == [0, 1]
    assert [p.point_id for p in repo.get_points("scan-001", since_point_id=3, limit=1)] == [4]
    assert repo.get_points("scan-001", since_point_id=5) == []
    assert repo.get_points("scan-001", limit=0) == []


def test_fixed_z_single_position_profile(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    config = make_config(mode=ScanMode.FIXED_Z, samples_per_z=8)
    create_scan(repo, "fixed", config=config)
    profile = make_fixed_z_profile(rng, samples_per_z=8)
    point = make_point(0, status=PointStatus.MEASURED, n_z_positions=1).model_copy(
        update={"surface_z_um": None, "intensity": 0.4375}
    )
    repo.append_point("fixed", point, profile, None)

    record = repo.get_profile("fixed", 0)
    assert_record_matches(record, profile)
    assert len(record.raw_counts) == 1
    assert len(record.raw_counts[0]) == 8
    assert record.normalized is None
    assert record.filtered is None
    assert record.analysis is None
    assert record.dark_v is None
    stored = repo.get_points("fixed")[0]
    assert stored.intensity == 0.4375
    assert stored.surface_z_um is None


def test_point_without_profile(repo: SQLiteHDF5Repository, rng: np.random.Generator) -> None:
    create_scan(repo)
    repo.append_point("scan-001", make_point(0, status=PointStatus.ABORTED), None, None)
    repo.append_point("scan-001", make_point(1), make_profile(rng), None)
    with pytest.raises(PointNotFoundError, match="no stored profile"):
        repo.get_profile("scan-001", 0)
    assert repo.get_profile("scan-001", 1).point_id == 1
    assert repo.get_points("scan-001")[0].status is PointStatus.ABORTED


def test_unknown_point_raises_point_not_found(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    with pytest.raises(PointNotFoundError):
        repo.get_profile("scan-001", 0)


def test_non_finite_optional_scalars_become_none(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    point = make_point(0).model_copy(
        update={"snr": math.nan, "fit_residual": math.inf, "asymmetry": -math.inf}
    )
    repo.append_point("scan-001", point, None, None)
    stored = repo.get_points("scan-001")[0]
    assert stored.snr is None
    assert stored.fit_residual is None
    assert stored.asymmetry is None
    assert stored.surface_z_um == point.surface_z_um


def test_non_finite_analysis_values_are_stored_as_none(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    create_scan(repo)
    analysis = make_analysis().model_copy(update={"snr": math.nan, "baseline": math.inf})
    repo.append_point("scan-001", make_point(0), make_profile(rng), analysis)
    stored = repo.get_profile("scan-001", 0).analysis
    assert stored is not None
    assert stored.snr is None
    assert stored.baseline is None
    assert stored.surface_z_um == analysis.surface_z_um


def test_non_finite_required_scalar_is_rejected(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    with pytest.raises(StorageError, match="x_um"):
        repo.append_point(
            "scan-001", make_point(0).model_copy(update={"x_um": math.nan}), None, None
        )
    assert repo.get_points("scan-001") == []


def test_inconsistent_samples_per_z_is_rejected_and_nothing_stored(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    create_scan(repo)  # samples_per_z = 4
    with pytest.raises(StorageError, match="samples_per_z must be constant"):
        repo.append_point("scan-001", make_point(0), make_profile(rng, samples_per_z=5), None)
    assert repo.get_points("scan-001") == []
    assert repo.get_scan("scan-001").completed_points == 0
    profile = make_profile(rng)
    repo.append_point("scan-001", make_point(0), profile, None)
    assert_record_matches(repo.get_profile("scan-001", 0), profile)


def test_non_finite_raw_voltage_is_rejected(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    create_scan(repo)
    profile = make_profile(rng)
    profile.voltage_v[3, 1] = math.nan
    with pytest.raises(StorageError, match="voltage_v"):
        repo.append_point("scan-001", make_point(0), profile, None)
    assert repo.get_points("scan-001") == []


def test_raw_counts_that_do_not_fit_int32_are_rejected(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    create_scan(repo)
    profile = make_profile(rng)
    profile.raw_counts = profile.raw_counts.astype(np.int64)
    profile.raw_counts[0, 0] = 2**40
    with pytest.raises(StorageError, match="raw_counts"):
        repo.append_point("scan-001", make_point(0), profile, None)


def test_duplicate_point_id_is_rejected_and_file_rolled_back(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    create_scan(repo)
    first = make_profile(rng)
    repo.append_point("scan-001", make_point(0), first, None)
    with pytest.raises(StorageError, match="already stored"):
        repo.append_point("scan-001", make_point(0), make_profile(rng), None)
    repo.append_point("scan-001", make_point(1), make_profile(rng), None)
    assert_record_matches(repo.get_profile("scan-001", 0), first)
    assert [p.point_id for p in repo.get_points("scan-001")] == [0, 1]
    assert repo.get_scan("scan-001").completed_points == 2
    repo.update_scan("scan-001", state=ScanState.COMPLETE)
    import h5py

    with h5py.File(repo.scans_dir / "scan-001.h5", "r") as f:
        assert f["points"].shape == (2,)
        assert f["profiles/index"].shape == (2, 3)


def test_profiles_survive_close_and_reopen(tmp_path: Path, rng: np.random.Generator) -> None:
    profiles = [make_profile(rng) for _ in range(3)]
    repo = open_repository(tmp_path, durable=True)
    create_scan(repo)
    for point_id, profile in enumerate(profiles):
        repo.append_point("scan-001", make_point(point_id), profile, make_analysis())
    repo.close()

    with open_repository(tmp_path) as reopened:
        for point_id, profile in enumerate(profiles):
            assert_record_matches(reopened.get_profile("scan-001", point_id), profile)


def test_reads_of_finished_scan_reopen_file_read_only(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    create_scan(repo)
    profile = make_profile(rng)
    repo.append_point("scan-001", make_point(0), profile, None)
    repo.update_scan("scan-001", state=ScanState.COMPLETE)
    assert_record_matches(repo.get_profile("scan-001", 0), profile)
    assert_record_matches(repo.get_profile("scan-001", 0), profile)  # handle was closed again
