"""HDF5 file layout (format version 1): one self-describing file per scan.

::

    /                     attrs: scan_id, format_version, created_at, software_version,
                                 hardware_json, config_json, calibration_json, mode
    /points               structured, resizable table of every ScanPoint scalar
    /points_uncommitted   (M,) int64: rows of /points whose database commit did not complete
    /profiles             attrs: samples_per_z
    /profiles/index       (N, 3) int64: point_id, start, count  (into the arrays below)
    /profiles/z_um, z_reported_um, phase, voltage_agg_v, timestamps, normalized, filtered
                          (total_positions,) resizable, chunked
    /profiles/raw_counts  (total_positions, samples_per_z) int32
    /profiles/voltage_v   (total_positions, samples_per_z) float64
    /profiles/meta        (N,) JSON strings; row k describes index row k (dark_v,
                          reference_v, gain, sampling method, calibration version,
                          ProfileAnalysis)
    /surfaces/<id>/       see confocal.storage.results
    /ml/<id>/             see confocal.storage.results

Encoding rules (so the file can be read without this code, e.g. with h5py,
HDFView or MATLAB):

* Missing floats are NaN (``None`` in the API models).
* Times are float64 seconds since the Unix epoch (UTC); ``created_at`` is ISO 8601.
* Strings are UTF-8. JSON documents are Pydantic ``model_dump_json`` output.
* Every dataset carries a ``units`` attribute.
* The positions of all points are concatenated in acquisition order; a profile
  is ``arrays[start:start + count]`` of its index row. Data beyond the last
  index row (possible only after an interrupted write) belongs to no point.
* Raw data is never deleted once it is flushed. A ``/points`` row listed in
  ``/points_uncommitted`` (with the ``/profiles/index`` row it references) was
  written durably but its database commit failed: it is kept for inspection
  but is not a stored point of the scan. Files written before this dataset
  existed lack it, which means "none".

Chunks are small (8-32 KiB) on purpose: the file is flushed after every point,
and each flush rewrites the partially filled chunk of every growing dataset.
Compression is off for the same reason (and because raw ADC codes compress
poorly); a typical scan is tens of MB.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Final

import h5py
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from confocal.errors import StorageError
from confocal.models.calibration import CalibrationState
from confocal.models.hardware import AdcGain, HardwareInfo, SamplingMethod
from confocal.models.measurement import ProfileData, ProfilePhase
from confocal.models.processing import ProfileAnalysis
from confocal.models.scan import ScanConfig, ScanPoint
from confocal.storage.points import (
    POINT_INT_FIELDS,
    POINT_OPTIONAL_FLOAT_FIELDS,
    POINT_REQUIRED_FLOAT_FIELDS,
)

FORMAT_VERSION: Final = 1

#: Library-version bounds for files we write: the HDF5 1.8 object format (readable by
#: every HDF5 >= 1.8) with dense attribute storage, so JSON attributes may exceed 64 KiB.
LIBVER: Final = ("v108", "latest")

POINTS: Final = "points"
UNCOMMITTED: Final = "points_uncommitted"
PROFILES: Final = "profiles"
INDEX: Final = "profiles/index"
META: Final = "profiles/meta"
SURFACES: Final = "surfaces"
ML: Final = "ml"

STRING_DTYPE: Final[Any] = h5py.string_dtype("utf-8")
STATUS_DTYPE: Final = np.dtype("S32")

#: (total_positions,) datasets: name (= ProfileData attribute) -> (dtype, units).
POSITION_DATASETS: Final[dict[str, tuple[np.dtype[Any], str]]] = {
    "z_um": (np.dtype(np.float64), "um (commanded)"),
    "z_reported_um": (np.dtype(np.float64), "um (reported by the stage after the move)"),
    "phase": (
        np.dtype(np.uint8),
        "ProfilePhase: " + ", ".join(f"{p.value}={p.name.lower()}" for p in ProfilePhase),
    ),
    "voltage_agg_v": (np.dtype(np.float64), "V (per-position aggregate of voltage_v)"),
    "timestamps": (np.dtype(np.float64), "s since the Unix epoch (UTC), first sample"),
    "normalized": (np.dtype(np.float64), "normalized intensity (NaN: not available)"),
    "filtered": (np.dtype(np.float64), "filtered signal in analysis.signal_units (NaN: none)"),
}
#: (total_positions, samples_per_z) datasets: name -> (dtype, units).
SAMPLE_DATASETS: Final[dict[str, tuple[np.dtype[Any], str]]] = {
    "raw_counts": (np.dtype(np.int32), "raw signed ADC codes"),
    "voltage_v": (np.dtype(np.float64), "V (raw_counts converted with the gain in effect)"),
}
#: Position arrays that must be finite: the API model has no "missing" value for them.
_FINITE_DATASETS: Final = ("z_um", "z_reported_um", "voltage_agg_v", "timestamps", "voltage_v")
#: Optional ProfileData arrays, stored as all-NaN when absent (see ProfileMeta).
_OPTIONAL_DATASETS: Final = ("normalized", "filtered")

_POSITION_CHUNK: Final = 1024
_TABLE_CHUNK: Final = 256
_SAMPLE_CHUNK_VALUES: Final = 2048

POINTS_DTYPE: Final = np.dtype(
    [(name, np.int64) for name in POINT_INT_FIELDS]
    + [(name, np.float64) for name in POINT_REQUIRED_FLOAT_FIELDS + POINT_OPTIONAL_FLOAT_FIELDS]
    + [
        ("status", STATUS_DTYPE),
        ("acquired_at", np.float64),
        ("flags", STRING_DTYPE),
        ("profile_row", np.int64),
    ]
)


class ProfileMeta(BaseModel):
    """Per-profile metadata stored as one JSON string in ``/profiles/meta``."""

    model_config = ConfigDict(extra="forbid")

    point_id: int
    gain: AdcGain
    sampling_method: SamplingMethod
    dark_v: float | None
    reference_v: float | None
    calibration_version: int | None
    has_normalized: bool
    has_filtered: bool
    analysis: ProfileAnalysis | None


def initialise_file(
    f: Any,
    *,
    scan_id: str,
    created_at: datetime,
    config: ScanConfig,
    calibration: CalibrationState,
    software_version: str,
    hardware: HardwareInfo,
) -> None:
    """Write the root attributes and create every (empty) dataset of a new scan file."""
    f.attrs["scan_id"] = scan_id
    f.attrs["format_version"] = FORMAT_VERSION
    f.attrs["created_at"] = created_at.isoformat()
    f.attrs["software_version"] = software_version
    f.attrs["hardware_json"] = hardware.model_dump_json()
    f.attrs["config_json"] = config.model_dump_json()
    f.attrs["calibration_json"] = calibration.model_dump_json()
    f.attrs["mode"] = config.mode.value

    points = f.create_dataset(
        POINTS, shape=(0,), maxshape=(None,), chunks=(_TABLE_CHUNK,), dtype=POINTS_DTYPE
    )
    points.attrs["units"] = (
        "um for *_um fields; NaN = no value; acquired_at in s since the Unix epoch (UTC); "
        "flags is a JSON list; profile_row indexes /profiles/index (-1 = no profile)"
    )
    _create_uncommitted(f)

    samples_per_z = config.samples_per_z
    profiles = f.create_group(PROFILES)
    profiles.attrs["samples_per_z"] = samples_per_z
    index = profiles.create_dataset(
        "index", shape=(0, 3), maxshape=(None, 3), chunks=(_TABLE_CHUNK, 3), dtype=np.int64
    )
    index.attrs["units"] = "columns: point_id, start, count (rows of the position arrays)"
    for name, (dtype, units) in POSITION_DATASETS.items():
        ds = profiles.create_dataset(
            name, shape=(0,), maxshape=(None,), chunks=(_POSITION_CHUNK,), dtype=dtype
        )
        ds.attrs["units"] = units
    rows = max(16, _SAMPLE_CHUNK_VALUES // samples_per_z)
    for name, (dtype, units) in SAMPLE_DATASETS.items():
        ds = profiles.create_dataset(
            name,
            shape=(0, samples_per_z),
            maxshape=(None, samples_per_z),
            chunks=(rows, samples_per_z),
            dtype=dtype,
        )
        ds.attrs["units"] = units
    meta = profiles.create_dataset(
        "meta", shape=(0,), maxshape=(None,), chunks=(_TABLE_CHUNK,), dtype=STRING_DTYPE
    )
    meta.attrs["units"] = "JSON (ProfileMeta); row k describes /profiles/index row k"
    f.create_group(SURFACES)
    f.create_group(ML)


def _create_uncommitted(f: Any) -> Any:
    ds = f.create_dataset(
        UNCOMMITTED, shape=(0,), maxshape=(None,), chunks=(_TABLE_CHUNK,), dtype=np.int64
    )
    ds.attrs["units"] = (
        "rows of /points (and the /profiles/index rows they reference) whose database "
        "commit did not complete: raw data kept, not a stored point of the scan"
    )
    return ds


def uncommitted_rows(f: Any) -> list[int]:
    """Rows of ``/points`` marked uncommitted, ascending (none if the dataset is absent)."""
    if UNCOMMITTED not in f:
        return []
    return sorted(int(v) for v in f[UNCOMMITTED][()])


def write_uncommitted_rows(f: Any, rows: list[int]) -> None:
    """Replace the uncommitted-row list (creating the dataset in files that predate it)."""
    ds = f[UNCOMMITTED] if UNCOMMITTED in f else _create_uncommitted(f)
    values = np.asarray(sorted(set(rows)), dtype=np.int64)
    ds.resize((values.shape[0],))
    if values.shape[0]:
        ds[:] = values


def point_keys(f: Any) -> list[tuple[int, int | None]]:
    """``(point_id, profile_row)`` of every ``/points`` row, in file order (row -1 -> None)."""
    points = f[POINTS]
    if points.shape[0] == 0:
        return []
    table = points.fields(["point_id", "profile_row"])[()]
    return [
        (int(point_id), None if int(row) < 0 else int(row))
        for point_id, row in zip(table["point_id"], table["profile_row"], strict=True)
    ]


def point_record(point: ScanPoint, profile_row: int | None) -> NDArray[Any]:
    """One-element ``/points`` record of a sanitised point."""
    record = np.zeros(1, dtype=POINTS_DTYPE)
    for name in POINT_INT_FIELDS + POINT_REQUIRED_FLOAT_FIELDS:
        record[name] = getattr(point, name)
    for name in POINT_OPTIONAL_FLOAT_FIELDS:
        value = getattr(point, name)
        record[name] = math.nan if value is None else value
    record["status"] = point.status.value.encode("ascii")
    record["acquired_at"] = point.acquired_at.timestamp()
    record["flags"] = json.dumps(point.flags)
    record["profile_row"] = -1 if profile_row is None else profile_row
    return record


def encode_profile(profile: ProfileData, samples_per_z: int) -> dict[str, NDArray[Any]]:
    """Arrays to append for ``profile``, in the file's dtypes, validated before any write.

    Raises:
        StorageError: wrong samples per position, raw codes that do not fit int32
            exactly, invalid phase values, or non-finite values in arrays the API
            cannot represent as missing.
    """
    n = profile.n_positions
    if n > 0 and profile.samples_per_position != samples_per_z:
        raise StorageError(
            f"profile has {profile.samples_per_position} samples per Z position, but this "
            f"scan stores {samples_per_z} (samples_per_z must be constant within a scan)"
        )
    arrays: dict[str, NDArray[Any]] = {}
    for name, (dtype, _units) in POSITION_DATASETS.items():
        value: NDArray[Any] | None = getattr(profile, name)
        if value is None:
            arrays[name] = np.full(n, np.nan, dtype=np.float64)
        else:
            arrays[name] = _exact(name, value, dtype)
    for name, (dtype, _units) in SAMPLE_DATASETS.items():
        arrays[name] = _exact(name, getattr(profile, name), dtype).reshape(n, samples_per_z)
    for name in _FINITE_DATASETS:
        if not bool(np.all(np.isfinite(arrays[name]))):
            raise StorageError(f"profile {name} contains NaN/inf; not stored")
    if not bool(np.all(np.isin(arrays["phase"], [p.value for p in ProfilePhase]))):
        raise StorageError("profile phase contains values that are not ProfilePhase members")
    return arrays


def profile_meta(
    point_id: int, profile: ProfileData, analysis: ProfileAnalysis | None
) -> ProfileMeta:
    return ProfileMeta(
        point_id=point_id,
        gain=profile.gain,
        sampling_method=profile.sampling_method,
        dark_v=profile.dark_v,
        reference_v=profile.reference_v,
        calibration_version=profile.calibration_version,
        has_normalized=profile.normalized is not None,
        has_filtered=profile.filtered is not None,
        analysis=analysis,
    )


def decode_profile(arrays: dict[str, NDArray[Any]], meta: ProfileMeta) -> ProfileData:
    """Rebuild the ``ProfileData`` of one point from its array slices and metadata."""
    return ProfileData(
        z_um=np.asarray(arrays["z_um"], dtype=np.float64),
        z_reported_um=np.asarray(arrays["z_reported_um"], dtype=np.float64),
        phase=np.asarray(arrays["phase"], dtype=np.uint8),
        raw_counts=np.asarray(arrays["raw_counts"], dtype=np.int32),
        voltage_v=np.asarray(arrays["voltage_v"], dtype=np.float64),
        voltage_agg_v=np.asarray(arrays["voltage_agg_v"], dtype=np.float64),
        timestamps=np.asarray(arrays["timestamps"], dtype=np.float64),
        gain=meta.gain,
        sampling_method=meta.sampling_method,
        dark_v=meta.dark_v,
        reference_v=meta.reference_v,
        calibration_version=meta.calibration_version,
        normalized=np.asarray(arrays["normalized"], dtype=np.float64)
        if meta.has_normalized
        else None,
        filtered=np.asarray(arrays["filtered"], dtype=np.float64) if meta.has_filtered else None,
    )


def _exact(name: str, value: NDArray[Any], dtype: np.dtype[Any]) -> NDArray[Any]:
    """``value`` converted to ``dtype``, refusing any conversion that changes a value."""
    source = np.asarray(value)
    converted = source.astype(dtype, copy=False)
    if source.dtype != dtype and not np.array_equal(converted, source, equal_nan=True):
        raise StorageError(f"profile {name} ({source.dtype}) does not convert exactly to {dtype}")
    return converted
