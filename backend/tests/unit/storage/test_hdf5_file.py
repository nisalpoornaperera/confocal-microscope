"""The per-scan HDF5 file is self-describing: readable with plain h5py, no confocal code."""

from __future__ import annotations

import json

import h5py
import numpy as np
from tests.unit.storage.conftest import (
    SOFTWARE_VERSION,
    create_scan,
    make_analysis,
    make_calibration,
    make_config,
    make_hardware,
    make_ml_result,
    make_point,
    make_profile,
    make_surface,
)

from confocal.models import ScanState
from confocal.storage import FORMAT_VERSION, SQLiteHDF5Repository

POSITION_ARRAYS = ("z_um", "z_reported_um", "phase", "voltage_agg_v", "timestamps")


def test_root_attributes_describe_the_scan(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    repo.update_scan("scan-001", state=ScanState.COMPLETE)
    with h5py.File(repo.scans_dir / "scan-001.h5", "r") as f:
        assert f.attrs["scan_id"] == "scan-001"
        assert int(f.attrs["format_version"]) == FORMAT_VERSION
        assert f.attrs["software_version"] == SOFTWARE_VERSION
        assert f.attrs["mode"] == "confocal"
        assert json.loads(f.attrs["config_json"]) == json.loads(make_config().model_dump_json())
        assert json.loads(f.attrs["hardware_json"]) == json.loads(make_hardware().model_dump_json())
        assert json.loads(f.attrs["calibration_json"])["version"] == 1
        assert isinstance(f.attrs["created_at"], str)
        assert int(f["profiles"].attrs["samples_per_z"]) == 4
        for name in ("points", "profiles/index", "profiles/meta", "profiles/raw_counts"):
            assert "units" in f[name].attrs


def test_datasets_hold_every_raw_value(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    create_scan(repo)
    profiles = [make_profile(rng, n_coarse=8 + i, n_fine=4) for i in range(3)]
    repo.append_point("scan-001", make_point(0), None, None)  # no profile
    for point_id, profile in enumerate(profiles, start=1):
        repo.append_point("scan-001", make_point(point_id), profile, make_analysis())
    repo.update_scan("scan-001", state=ScanState.COMPLETE)

    with h5py.File(repo.scans_dir / "scan-001.h5", "r") as f:
        points = f["points"][()]
        assert points["point_id"].tolist() == [0, 1, 2, 3]
        assert points["profile_row"].tolist() == [-1, 0, 1, 2]
        assert points["status"][0] == b"valid"
        assert points["x_um"].tolist() == [make_point(i).x_um for i in range(4)]

        index = f["profiles/index"][()]
        assert index.dtype == np.int64
        assert f["profiles/raw_counts"].dtype == np.int32
        assert f["profiles/voltage_v"].dtype == np.float64
        assert f["profiles/phase"].dtype == np.uint8
        start = 0
        for row, profile in enumerate(profiles):
            point_id, first, count = (int(v) for v in index[row])
            assert (point_id, first, count) == (row + 1, start, profile.n_positions)
            stop = first + count
            np.testing.assert_array_equal(f["profiles/raw_counts"][first:stop], profile.raw_counts)
            np.testing.assert_array_equal(f["profiles/voltage_v"][first:stop], profile.voltage_v)
            for name in POSITION_ARRAYS:
                np.testing.assert_array_equal(
                    f[f"profiles/{name}"][first:stop], getattr(profile, name)
                )
            assert profile.filtered is not None
            np.testing.assert_array_equal(f["profiles/filtered"][first:stop], profile.filtered)
            meta = json.loads(f["profiles/meta"][row])
            assert meta["point_id"] == row + 1
            assert meta["gain"] == "2"
            assert meta["calibration_version"] == 3
            assert meta["analysis"]["status"] == "valid"
            start = stop


def test_file_is_readable_by_another_handle_after_each_flush(
    repo: SQLiteHDF5Repository, rng: np.random.Generator
) -> None:
    """After a scan finishes, its file is complete on disk without any further step."""
    create_scan(repo)
    repo.append_point("scan-001", make_point(0), make_profile(rng), None)
    repo.update_scan("scan-001", state=ScanState.CANCELLED)
    with h5py.File(repo.scans_dir / "scan-001.h5", "r") as f:
        assert f["points"].shape == (1,)


def test_surface_and_ml_groups_are_plain_datasets(repo: SQLiteHDF5Repository) -> None:
    create_scan(repo)
    surface = repo.save_surface(make_surface("scan-001"))
    ml = repo.save_ml_result(make_ml_result("scan-001"))
    repo.update_scan("scan-001", state=ScanState.COMPLETE)
    with h5py.File(repo.scans_dir / "scan-001.h5", "r") as f:
        group = f[f"surfaces/{surface.surface_id}"]
        z = group["z_um"][()]
        assert z.shape == (2, 3)
        assert np.isnan(z[0, 2])
        assert group["gap_mask"][()].tolist() == [[False, False, True], [False, False, False]]
        assert json.loads(group.attrs["surface_json"])["statistics"]["n_used"] == 1
        predictions = f[f"ml/{ml.result_id}/predictions"][()]
        assert predictions["point_id"].tolist() == [0, 1, 2]
        assert predictions["flagged"].tolist() == [False, True, False]
        assert json.loads(f[f"ml/{ml.result_id}"].attrs["ml_json"])["advisory"] is True


def test_calibration_snapshot_in_file_matches_the_scan(repo: SQLiteHDF5Repository) -> None:
    calibration = make_calibration(version=5)
    create_scan(repo, calibration=calibration)
    repo.update_scan("scan-001", state=ScanState.COMPLETE)
    with h5py.File(repo.scans_dir / "scan-001.h5", "r") as f:
        assert f.attrs["calibration_json"] == calibration.model_dump_json()
