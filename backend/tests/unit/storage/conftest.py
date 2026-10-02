"""Builders of foundation-model instances and repository fixtures for the storage tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from confocal.models import (
    AdcGain,
    CalibrationState,
    FitMethod,
    HardwareInfo,
    MLModelInfo,
    MLPointPrediction,
    MLResult,
    MLTask,
    PeakFit,
    PointClassification,
    PointStatus,
    ProfileAnalysis,
    ProfileData,
    ProfilePhase,
    ReconstructionRequest,
    SamplingMethod,
    ScanConfig,
    ScanMode,
    ScanPoint,
    SurfacePoint,
    SurfaceResult,
    SurfaceStatistics,
)
from confocal.storage import SQLiteHDF5Repository

SOFTWARE_VERSION = "0.1.0-test"
#: Commanded-to-reported Z offset of the fake stage (makes the two arrays differ).
REPORTED_Z_OFFSET_UM = 0.03125


def make_config(
    *, samples_per_z: int = 4, mode: ScanMode = ScanMode.CONFOCAL, name: str | None = "test"
) -> ScanConfig:
    """A 4 x 3 grid (12 points)."""
    return ScanConfig(
        name=name,
        mode=mode,
        x_start_um=0.0,
        x_stop_um=30.0,
        y_start_um=-10.0,
        y_stop_um=10.0,
        xy_step_um=10.0,
        samples_per_z=samples_per_z,
    )


def make_hardware() -> HardwareInfo:
    return HardwareInfo(
        controller="simulated",
        stage_backend="sim-delta",
        stage_version="1.2",
        adc_backend="sim-ads1115",
        laser_backend="manual",
        camera_backend="none",
        details={"firmware": "delta-uno 0.3"},
    )


def make_calibration(*, version: int | None = None, dark_v: float = 0.012) -> CalibrationState:
    return CalibrationState(
        version=version,
        created_at=datetime(2026, 9, 30, 12, 0, tzinfo=UTC),
        updated_field="reference",
        dark_v=dark_v,
        dark_std_v=0.0004,
        dark_n_samples=64,
        dark_gain=AdcGain.G2,
        reference_v=1.85,
        reference_std_v=0.003,
        reference_n_samples=64,
        reference_gain=AdcGain.G2,
        notes="plane mirror",
    )


def make_point(
    point_id: int,
    *,
    status: PointStatus = PointStatus.VALID,
    n_z_positions: int = 0,
    surface_z_um: float | None = None,
) -> ScanPoint:
    """Point ``point_id`` of the 4 x 3 grid of :func:`make_config` (serpentine order)."""
    iy, col = divmod(point_id, 4)
    ix = col if iy % 2 == 0 else 3 - col
    surface = 5.0 + 0.25 * point_id if surface_z_um is None else surface_z_um
    return ScanPoint(
        point_id=point_id,
        ix=ix,
        iy=iy,
        x_um=10.0 * ix,
        y_um=-10.0 + 10.0 * iy,
        status=status,
        z_estimate_um=surface - 1.0,
        coarse_peak_z_um=surface + 0.5,
        surface_z_um=surface,
        parabolic_z_um=surface + 0.01,
        gaussian_z_um=surface - 0.01,
        peak_intensity=0.875,
        snr=42.5,
        peak_width_um=3.25,
        prominence=0.75,
        fit_residual=0.0125,
        confidence=0.9,
        secondary_peak_ratio=0.1,
        asymmetry=-0.05,
        n_z_positions=n_z_positions,
        flags=["fits_agree"],
        acquired_at=datetime(2026, 10, 1, 9, 30, point_id % 60, 123456, tzinfo=UTC),
        duration_s=1.5,
    )


def make_profile(
    rng: np.random.Generator,
    *,
    n_coarse: int = 20,
    n_fine: int = 12,
    samples_per_z: int = 4,
    with_processed: bool = True,
) -> ProfileData:
    """A coarse + fine I(Z) sweep with random raw ADC codes (full int16 range)."""
    z = np.concatenate([np.linspace(-50.0, 50.0, n_coarse), np.linspace(-3.0, 3.0, n_fine)]).astype(
        np.float64
    )
    n = z.shape[0]
    phase = np.concatenate(
        [np.full(n_coarse, ProfilePhase.COARSE), np.full(n_fine, ProfilePhase.FINE)]
    ).astype(np.uint8)
    counts = rng.integers(-32768, 32768, size=(n, samples_per_z), dtype=np.int32)
    volts = counts.astype(np.float64) * (2.048 / 32768.0)
    filtered = np.asarray(rng.normal(size=n), dtype=np.float64)
    filtered[:2] = np.nan
    return ProfileData(
        z_um=z,
        z_reported_um=z + REPORTED_Z_OFFSET_UM,
        phase=phase,
        raw_counts=counts,
        voltage_v=volts,
        voltage_agg_v=volts.mean(axis=1),
        timestamps=1_790_000_000.0 + np.arange(n, dtype=np.float64) * 0.0123,
        gain=AdcGain.G2,
        sampling_method=SamplingMethod.MEDIAN,
        dark_v=0.012,
        reference_v=1.85,
        calibration_version=3,
        normalized=np.asarray(rng.uniform(0.0, 1.0, size=n), dtype=np.float64)
        if with_processed
        else None,
        filtered=filtered if with_processed else None,
    )


def make_fixed_z_profile(rng: np.random.Generator, *, samples_per_z: int = 4) -> ProfileData:
    """Fixed-Z mode: exactly one position, no reference calibration."""
    counts = rng.integers(0, 20_000, size=(1, samples_per_z), dtype=np.int32)
    volts = counts.astype(np.float64) * (2.048 / 32768.0)
    return ProfileData(
        z_um=np.array([12.5]),
        z_reported_um=np.array([12.5]),
        phase=np.array([ProfilePhase.FIXED], dtype=np.uint8),
        raw_counts=counts,
        voltage_v=volts,
        voltage_agg_v=np.median(volts, axis=1),
        timestamps=np.array([1_790_000_000.5]),
        gain=AdcGain.G2,
        sampling_method=SamplingMethod.MEDIAN,
        dark_v=None,
        reference_v=None,
        calibration_version=None,
    )


def make_analysis(surface_z_um: float = 5.0) -> ProfileAnalysis:
    fit = PeakFit(
        method=FitMethod.GAUSSIAN,
        success=True,
        center_um=surface_z_um,
        amplitude=0.8,
        fwhm_um=3.2,
        offset=0.02,
        residual_rms=0.01,
        r_squared=0.995,
        n_points=9,
    )
    return ProfileAnalysis(
        status=PointStatus.VALID,
        signal_units="normalized",
        peak_found=True,
        peak_index=25,
        peak_z_um=surface_z_um,
        peak_intensity=0.82,
        baseline=0.02,
        noise_std=0.004,
        gaussian=fit,
        surface_z_um=surface_z_um,
        surface_method=FitMethod.GAUSSIAN,
        snr=200.0,
        fwhm_um=3.2,
        confidence=0.95,
        n_peaks=1,
        flags=["ok"],
    )


def make_surface(scan_id: str) -> SurfaceResult:
    """A 3 x 2 height map with one gap cell (None in z and confidence)."""
    return SurfaceResult(
        scan_id=scan_id,
        created_at=datetime(2026, 10, 1, 10, 0, tzinfo=UTC),
        request=ReconstructionRequest(),
        x_um=[0.0, 10.0, 20.0],
        y_um=[0.0, 10.0],
        z_um=[[1.0, 1.5, None], [2.0, 2.125, 2.25]],
        confidence=[[0.9, 0.8, None], [0.7, 0.75, 1.0]],
        gap_mask=[[False, False, True], [False, False, False]],
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
                x_um=20.0,
                y_um=0.0,
                z_um=None,
                confidence=0.0,
                classification=PointClassification.INVALID,
            ),
        ],
        statistics=SurfaceStatistics(
            n_input=2,
            n_invalid=1,
            n_low_confidence=0,
            n_outliers=0,
            n_used=1,
            coverage_fraction=5 / 6,
            gap_fraction=1 / 6,
            n_gaps=1,
            z_min_um=1.0,
            z_max_um=2.25,
            plane_coefficients=(0.01, 0.1, 1.0),
        ),
    )


def make_ml_result(scan_id: str) -> MLResult:
    return MLResult(
        scan_id=scan_id,
        model=MLModelInfo(
            name="bad-point-rf",
            version="2026.09",
            task=MLTask.BAD_POINT,
            algorithm="RandomForestClassifier",
            feature_names=["snr", "fwhm_um", "asymmetry"],
            trained_at=datetime(2026, 9, 1, tzinfo=UTC),
            metrics={"f1": 0.93},
        ),
        created_at=datetime(2026, 10, 1, 11, 0, tzinfo=UTC),
        threshold=0.5,
        n_points=3,
        n_flagged=1,
        predictions=[
            MLPointPrediction(point_id=0, label="good", flagged=False, probability_bad=0.1),
            MLPointPrediction(point_id=1, label="bad", flagged=True, probability_bad=0.9),
            MLPointPrediction(point_id=2, label="good", flagged=False, predicted_confidence=0.66),
        ],
    )


def open_repository(root: Path, *, durable: bool = False) -> SQLiteHDF5Repository:
    return SQLiteHDF5Repository(root / "confocal.db", root / "scans", durable=durable)


def create_scan(
    repo: SQLiteHDF5Repository,
    scan_id: str = "scan-001",
    *,
    config: ScanConfig | None = None,
    calibration: CalibrationState | None = None,
) -> None:
    cfg = make_config() if config is None else config
    repo.create_scan(
        scan_id=scan_id,
        config=cfg,
        total_points=cfg.total_points,
        calibration=make_calibration(version=1) if calibration is None else calibration,
        software_version=SOFTWARE_VERSION,
        hardware=make_hardware(),
    )


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[SQLiteHDF5Repository]:
    repository = open_repository(tmp_path)
    try:
        yield repository
    finally:
        repository.close()
