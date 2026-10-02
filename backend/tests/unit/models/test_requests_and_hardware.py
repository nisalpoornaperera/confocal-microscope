"""API request models, hardware models, calibration state and configuration sections."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from confocal.config import ArduinoConfig, KinematicsConfig, MotorConfig, Settings
from confocal.models import (
    ADCCalibrateRequest,
    AdcGain,
    Axis,
    CalibrationState,
    DarkCalibrationRequest,
    EmergencyStopRequest,
    ErrorResponse,
    ProcessingConfig,
    StageHomeRequest,
    StageMoveRequest,
)


# --------------------------------------------------------------------------- stage requests
def test_stage_move_request_needs_at_least_one_axis() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        StageMoveRequest()
    with pytest.raises(ValidationError, match="at least one"):
        StageMoveRequest(relative=True)
    move = StageMoveRequest(z_um=-2.5, relative=True)
    assert (move.x_um, move.y_um, move.z_um) == (None, None, -2.5)
    with pytest.raises(ValidationError):
        StageMoveRequest(x_um=float("inf"))
    with pytest.raises(ValidationError):
        StageMoveRequest.model_validate({"x_um": 1.0, "speed": 3})


def test_stage_home_and_estop_requests() -> None:
    assert StageHomeRequest().axes == [Axis.X, Axis.Y, Axis.Z]
    with pytest.raises(ValidationError):
        StageHomeRequest(axes=[])
    assert EmergencyStopRequest().reason == "operator emergency stop"
    with pytest.raises(ValidationError):
        EmergencyStopRequest(reason="x" * 501)


def test_adc_calibrate_request_needs_exactly_one_of_gain_or_auto() -> None:
    assert ADCCalibrateRequest(gain=AdcGain.G2).gain is AdcGain.G2
    assert ADCCalibrateRequest(auto=True).auto
    with pytest.raises(ValidationError, match="exactly one"):
        ADCCalibrateRequest()
    with pytest.raises(ValidationError, match="exactly one"):
        ADCCalibrateRequest(gain=AdcGain.G1, auto=True)
    with pytest.raises(ValidationError):
        ADCCalibrateRequest(auto=True, target_fraction=0.99)


def test_dark_calibration_request_defaults_need_operator_confirmation() -> None:
    request = DarkCalibrationRequest()
    assert not request.beam_blocked_confirmed
    with pytest.raises(ValidationError):
        DarkCalibrationRequest(n_samples=0)


def test_error_response_shape() -> None:
    body = ErrorResponse(error="LimitViolationError", detail="outside", violations=["x"])
    assert body.model_dump() == {
        "error": "LimitViolationError",
        "detail": "outside",
        "violations": ["x"],
    }


# --------------------------------------------------------------------------- hardware models
@pytest.mark.parametrize(
    ("gain", "full_scale"),
    [
        (AdcGain.G2_3, 6.144),
        (AdcGain.G1, 4.096),
        (AdcGain.G2, 2.048),
        (AdcGain.G4, 1.024),
        (AdcGain.G8, 0.512),
        (AdcGain.G16, 0.256),
    ],
)
def test_adc_gain_full_scale_and_lsb(gain: AdcGain, full_scale: float) -> None:
    assert gain.full_scale_v == full_scale
    assert gain.lsb_v == pytest.approx(full_scale / 32768.0)


def test_adc_gain_values_match_the_config_file_strings() -> None:
    assert AdcGain("2") is AdcGain.G2  # config/confocal.pi.toml uses gain = "2"
    assert AdcGain("2/3") is AdcGain.G2_3


# --------------------------------------------------------------------------- calibration
def test_calibration_state_flags() -> None:
    empty = CalibrationState()
    assert not empty.has_dark
    assert not empty.has_reference
    assert not empty.can_normalize
    assert CalibrationState(reference_v=1.0).can_normalize  # dark treated as 0 V
    assert CalibrationState(dark_v=0.01, reference_v=1.0).can_normalize
    assert not CalibrationState(dark_v=1.0, reference_v=0.5).can_normalize


def test_processing_config_requires_an_odd_filter_window() -> None:
    assert ProcessingConfig(filter_window=7).filter_window == 7
    with pytest.raises(ValidationError, match="odd"):
        ProcessingConfig(filter_window=6)


# --------------------------------------------------------------------------- config sections
def test_matrix_geometry_requires_a_3x3_matrix() -> None:
    with pytest.raises(ValidationError, match="3 x 3"):
        KinematicsConfig(geometry="matrix")
    with pytest.raises(ValidationError, match="3 x 3"):
        KinematicsConfig(geometry="matrix", um_per_step_matrix=[[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValidationError, match="3 x 3"):
        KinematicsConfig(
            geometry="matrix", um_per_step_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]]
        )
    matrix = [[0.06, 0.0, 0.0], [0.0, 0.06, 0.0], [0.0, 0.0, 0.05]]
    assert KinematicsConfig(geometry="matrix", um_per_step_matrix=matrix).um_per_step_matrix


def test_kinematics_scale_factors_must_be_positive() -> None:
    assert KinematicsConfig().geometry == "delta"
    with pytest.raises(ValidationError, match="positive"):
        KinematicsConfig(um_per_step=(0.06, 0.0, 0.06))
    with pytest.raises(ValidationError, match="positive"):
        KinematicsConfig(geometry="cartesian", steps_per_um=(14.3, -1.0, 20.0))


def test_motor_travel_must_include_the_origin() -> None:
    MotorConfig(min_steps=-1, max_steps=1)
    for low, high in ((0, 100), (-100, 0), (10, 100), (-100, -10)):
        with pytest.raises(ValidationError, match="include the origin"):
            MotorConfig(min_steps=low, max_steps=high)
    with pytest.raises(ValidationError):
        MotorConfig(backlash_steps=-1)
    with pytest.raises(ValidationError):
        MotorConfig(max_speed_steps_s=0.0)


def test_arduino_config_has_three_motors_and_rejects_unknown_keys() -> None:
    arduino = ArduinoConfig()
    assert {arduino.a.max_steps, arduino.b.max_steps, arduino.c.max_steps} == {200_000}
    with pytest.raises(ValidationError):
        ArduinoConfig.model_validate({"d": {}})
    with pytest.raises(ValidationError):
        Settings.model_validate({"hardware": {"stage": "teleporter"}})
