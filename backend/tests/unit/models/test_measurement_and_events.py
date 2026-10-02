"""Measurement containers, JSON safety and round trips of API-facing result models."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from confocal.models import (
    AdcGain,
    AdcSamples,
    CoarsePeak,
    CrossSection,
    LiveProfile,
    MeshData,
    PointClassification,
    PointStatus,
    ProfileData,
    ReconstructionRequest,
    SamplingMethod,
    ScanEvent,
    ScanEventType,
    ScanPoint,
    ScanProgress,
    ScanState,
    SurfacePoint,
    SurfaceResult,
    SurfaceStatistics,
    nan_to_none,
)


def _profile_arrays(n: int = 3, s: int = 2) -> dict[str, object]:
    return {
        "z_um": np.linspace(0.0, 1.0, n),
        "z_reported_um": np.linspace(0.0, 1.0, n),
        "phase": np.zeros(n, dtype=np.uint8),
        "raw_counts": np.zeros((n, s), dtype=np.int32),
        "voltage_v": np.zeros((n, s), dtype=np.float64),
        "voltage_agg_v": np.zeros(n),
        "timestamps": np.zeros(n),
    }


def _profile(**overrides: object) -> ProfileData:
    values = _profile_arrays()
    values.update(overrides)
    return ProfileData(
        **values,  # type: ignore[arg-type]
        gain=AdcGain.G2,
        sampling_method=SamplingMethod.MEAN,
        dark_v=None,
        reference_v=None,
        calibration_version=None,
    )


def _progress() -> ScanProgress:
    return ScanProgress(
        scan_id="abc",
        state=ScanState.SCANNING,
        progress=0.5,
        completed_points=3,
        total_points=6,
        current_z_um=1.25,
        estimated_remaining_s=12.0,
    )


# --------------------------------------------------------------------------- containers
def test_nan_to_none() -> None:
    assert nan_to_none([1.0, math.nan, math.inf, -2.5]) == [1.0, None, None, -2.5]


def test_adc_samples_shape_validation() -> None:
    counts = np.array([1, 2, 3], dtype=np.int32)
    samples = AdcSamples(counts, counts * 0.1, np.arange(3.0), AdcGain.G2, 860)
    assert samples.n == 3
    with pytest.raises(ValueError, match="equal length"):
        AdcSamples(counts, np.zeros(2), np.arange(3.0), AdcGain.G2, 860)
    with pytest.raises(ValueError, match="1-D"):
        AdcSamples(counts.reshape(3, 1), np.zeros((3, 1)), np.zeros((3, 1)), AdcGain.G2, 860)
    empty = np.zeros(0, dtype=np.int32)
    with pytest.raises(ValueError, match="at least one"):
        AdcSamples(empty, np.zeros(0), np.zeros(0), AdcGain.G2, 860)


def test_profile_data_shape_validation() -> None:
    profile = _profile(normalized=np.zeros(3), filtered=np.full(3, np.nan))
    assert profile.n_positions == 3
    assert profile.samples_per_position == 2
    with pytest.raises(ValueError, match="z_reported_um"):
        _profile(z_reported_um=np.zeros(2))
    with pytest.raises(ValueError, match="raw_counts"):
        _profile(raw_counts=np.zeros(3, dtype=np.int32))
    with pytest.raises(ValueError, match="voltage_v"):
        _profile(voltage_v=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="normalized"):
        _profile(normalized=np.zeros(4))
    with pytest.raises(ValueError, match="filtered"):
        _profile(filtered=np.zeros((3, 1)))


# --------------------------------------------------------------------------- API models
def test_scan_point_validation() -> None:
    point = ScanPoint(point_id=0, ix=0, iy=0, x_um=0.0, y_um=0.0, status=PointStatus.VALID)
    assert point.confidence == 0.0
    assert point.acquired_at.utcoffset() is not None
    for bad in ({"point_id": -1}, {"confidence": 1.5}, {"extra_field": 1}):
        values = {"point_id": 0, "ix": 0, "iy": 0, "x_um": 0, "y_um": 0, "status": "valid"}
        values.update(bad)
        with pytest.raises(ValidationError):
            ScanPoint.model_validate(values)


def test_scan_event_json_round_trip() -> None:
    event = ScanEvent(
        type=ScanEventType.POINT,
        scan_id="abc",
        progress=_progress(),
        point=ScanPoint(
            point_id=2,
            ix=2,
            iy=0,
            x_um=20.0,
            y_um=0.0,
            status=PointStatus.LOW_CONFIDENCE,
            surface_z_um=3.1,
            confidence=0.4,
            flags=["fit_disagreement"],
        ),
        profile=LiveProfile(
            point_id=2, x_um=20.0, y_um=0.0, phase=[0, 1], z_um=[1.0, 2.0], intensity=[0.1, None]
        ),
        message="point done",
    )
    restored = ScanEvent.model_validate_json(event.model_dump_json())
    assert restored == event
    assert restored.profile is not None
    assert restored.profile.intensity == [0.1, None]
    with pytest.raises(ValidationError):
        ScanEvent.model_validate({**event.model_dump(), "unexpected": True})


def test_progress_is_bounded() -> None:
    with pytest.raises(ValidationError):
        ScanProgress(
            scan_id="abc",
            state=ScanState.SCANNING,
            progress=1.5,
            completed_points=0,
            total_points=1,
        )


def test_surface_result_json_round_trip() -> None:
    surface = SurfaceResult(
        surface_id=1,
        scan_id="abc",
        request=ReconstructionRequest(),
        x_um=[0.0, 10.0],
        y_um=[0.0],
        z_um=[[1.0, None]],
        confidence=[[0.9, None]],
        gap_mask=[[False, True]],
        points=[
            SurfacePoint(
                point_id=0,
                x_um=0.0,
                y_um=0.0,
                z_um=1.0,
                confidence=0.9,
                classification=PointClassification.USED,
            ),
            SurfacePoint(
                point_id=1,
                x_um=10.0,
                y_um=0.0,
                z_um=None,
                confidence=0.0,
                classification=PointClassification.INVALID,
            ),
        ],
        mesh=MeshData(vertices=[(0.0, 0.0, 1.0)], faces=[]),
        cross_sections=[
            CrossSection(along="x", position_um=0.0, coordinate_um=[0.0, 10.0], z_um=[1.0, None])
        ],
        statistics=SurfaceStatistics(
            n_input=2,
            n_invalid=1,
            n_low_confidence=0,
            n_outliers=0,
            n_used=1,
            coverage_fraction=0.5,
            gap_fraction=0.5,
            n_gaps=1,
            plane_coefficients=(0.0, 0.0, 1.0),
        ),
    )
    restored = SurfaceResult.model_validate_json(surface.model_dump_json())
    assert restored == surface
    assert restored.mesh is not None
    assert restored.mesh.vertices == [(0.0, 0.0, 1.0)]


def test_result_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CoarsePeak.model_validate({"found": True, "z_um": 1.0, "peak": 3})
    with pytest.raises(ValidationError):
        ReconstructionRequest.model_validate({"method": "linear", "smoothness": 1})
    with pytest.raises(ValidationError):
        ReconstructionRequest(outlier_neighbours=2)
