"""ScanConfig validation and grid counting."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from confocal.models import (
    MAX_SCAN_POINTS,
    MAX_Z_POSITIONS_PER_POINT,
    ProcessingConfig,
    ReconstructionRequest,
    ScanConfig,
    ScanMode,
    ScanOrder,
    axis_count,
)


def _config(**overrides: Any) -> ScanConfig:
    values: dict[str, Any] = {
        "x_start_um": 0.0,
        "x_stop_um": 100.0,
        "y_start_um": 0.0,
        "y_stop_um": 50.0,
        "xy_step_um": 10.0,
    }
    values.update(overrides)
    return ScanConfig.model_validate(values)


def test_defaults_describe_an_adaptive_serpentine_confocal_scan() -> None:
    config = _config()
    assert config.mode is ScanMode.CONFOCAL
    assert config.order is ScanOrder.SERPENTINE
    assert config.adaptive_z
    assert (config.n_x, config.n_y, config.total_points) == (11, 6, 66)


def test_axis_count() -> None:
    assert axis_count(0.0, 1.0, 0.1) == 11  # robust to floating point
    assert axis_count(0.0, 10.5, 2.0) == 6  # never beyond stop
    assert axis_count(5.0, 5.0, 1.0) == 1
    with pytest.raises(ValueError, match="step"):
        axis_count(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="stop"):
        axis_count(1.0, 0.0, 0.1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"x_start_um": 10.0, "x_stop_um": 0.0}, "x_stop_um must be >= x_start_um"),
        ({"y_start_um": 10.0, "y_stop_um": 0.0}, "y_stop_um must be >= y_start_um"),
        ({"fine_z_step_um": 3.0, "coarse_z_step_um": 2.0}, "fine_z_step_um must be <="),
        ({"fine_z_range_um": 120.0, "z_range_um": 100.0}, "fine_z_range_um must be <= z_range"),
        ({"fine_z_range_um": 3.0, "coarse_z_step_um": 2.0}, "2 x coarse_z_step_um"),
        ({"adaptive_z_range_um": 150.0, "z_range_um": 100.0}, "adaptive_z_range_um must be"),
    ],
)
def test_confocal_step_and_range_relations(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _config(**overrides)


def test_fixed_z_mode_does_not_check_the_z_sweep_relations() -> None:
    config = _config(mode=ScanMode.FIXED_Z, fine_z_step_um=3.0, coarse_z_step_um=2.0)
    assert config.mode is ScanMode.FIXED_Z


def test_too_many_points_are_refused() -> None:
    with pytest.raises(ValidationError, match=f"max {MAX_SCAN_POINTS}"):
        _config(x_stop_um=1000.0, y_stop_um=1000.0, xy_step_um=1.0)


def test_too_many_z_positions_per_point_are_refused() -> None:
    with pytest.raises(ValidationError, match=f"max {MAX_Z_POSITIONS_PER_POINT}"):
        _config(
            z_range_um=2000.0,
            coarse_z_step_um=0.4,  # 5001 coarse + 121 fine positions
            fine_z_range_um=12.0,
            fine_z_step_um=0.1,
            adaptive_z_range_um=30.0,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"xy_step_um": 0.0},
        {"xy_step_um": -1.0},
        {"z_range_um": 0.0},
        {"samples_per_z": 0},
        {"samples_per_z": 257},
        {"settle_time_ms": -1.0},
        {"x_start_um": float("nan")},
        {"name": "x" * 201},
        {"unknown_field": 1},
    ],
)
def test_field_constraints(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _config(**overrides)


def test_config_round_trips_through_json() -> None:
    config = _config(name="wafer", order=ScanOrder.RASTER, samples_per_z=16)
    assert ScanConfig.model_validate_json(config.model_dump_json()) == config


NON_FINITE = [float("inf"), float("-inf"), float("nan"), "inf", "Infinity", "NaN"]


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize(
    "field",
    [
        "x_start_um",
        "x_stop_um",
        "xy_step_um",
        "z_center_um",
        "z_range_um",
        "coarse_z_step_um",
        "fine_z_step_um",
        "fine_z_range_um",
        "adaptive_z_range_um",
        "settle_time_ms",
    ],
)
def test_non_finite_scan_values_are_refused(field: str, value: object) -> None:
    """inf used to pass validation and overflow axis_count (HTTP 500)."""
    with pytest.raises(ValidationError, match="finite"):
        _config(**{field: value})


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize(
    "field",
    [
        "gaussian_sigma_samples",
        "min_snr",
        "max_fit_disagreement_um",
        "expected_fwhm_um",
        "saturation_v",
    ],
)
def test_non_finite_processing_values_are_refused(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="finite"):
        ProcessingConfig.model_validate({field: value})
    with pytest.raises(ValidationError, match="finite"):
        _config(processing={field: value})


@pytest.mark.parametrize("value", NON_FINITE)
@pytest.mark.parametrize(
    "field",
    ["outlier_threshold", "grid_step_um", "max_gap_distance_um", "rbf_smoothing"],
)
def test_non_finite_reconstruction_values_are_refused(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="finite"):
        ReconstructionRequest.model_validate({field: value})
    with pytest.raises(ValidationError, match="finite"):
        _config(reconstruction={field: value})


def test_non_finite_values_are_refused_from_json() -> None:
    text = '{"x_start_um": 0, "x_stop_um": 10, "y_start_um": 0, "y_stop_um": 10, '
    with pytest.raises(ValidationError, match="finite"):
        ScanConfig.model_validate_json(text + '"xy_step_um": 1, "z_range_um": Infinity}')
    with pytest.raises(ValidationError, match="finite"):
        ScanConfig.model_validate_json(text + '"xy_step_um": NaN}')
