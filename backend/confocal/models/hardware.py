"""Hardware status / identity models (API-facing)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from confocal.models.common import Position, StageLimits


class AdcGain(StrEnum):
    """Programmable-gain-amplifier setting (ADS1x15 convention).

    The value is the PGA gain; :attr:`full_scale_v` is the matching +/- full-scale
    input range. The ADS1115 returns signed 16-bit counts, so one LSB is
    ``full_scale_v / 32768``.
    """

    G2_3 = "2/3"
    G1 = "1"
    G2 = "2"
    G4 = "4"
    G8 = "8"
    G16 = "16"

    @property
    def full_scale_v(self) -> float:
        return _ADC_FULL_SCALE_V[self]

    @property
    def lsb_v(self) -> float:
        return _ADC_FULL_SCALE_V[self] / 32768.0


_ADC_FULL_SCALE_V: dict[AdcGain, float] = {
    AdcGain.G2_3: 6.144,
    AdcGain.G1: 4.096,
    AdcGain.G2: 2.048,
    AdcGain.G4: 1.024,
    AdcGain.G8: 0.512,
    AdcGain.G16: 0.256,
}


class SamplingMethod(StrEnum):
    """How repeated ADC samples at one position are reduced to one value."""

    MEAN = "mean"
    MEDIAN = "median"


class StageState(StrEnum):
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    MOVING = "moving"
    HOMING = "homing"
    STOPPED = "stopped"  # halted by stop/emergency stop
    ERROR = "error"


class _Status(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageStatus(_Status):
    backend: str
    connected: bool
    state: StageState
    position: Position | None = None
    homed: bool = False
    estop_engaged: bool = False
    limits: StageLimits
    firmware_version: str | None = None
    last_error: str | None = None


class ADCStatus(_Status):
    backend: str
    connected: bool
    gain: AdcGain
    full_scale_v: float
    data_rate_sps: int
    channel: int | None = None
    last_voltage_v: float | None = None
    saturated: bool = False
    last_error: str | None = None


class LaserStatus(_Status):
    backend: str
    connected: bool
    controllable: bool
    enabled: bool | None = Field(
        default=None, description="None when the laser state cannot be known (manual laser)."
    )
    wavelength_nm: float
    power_mw: float | None = None
    last_error: str | None = None


class CameraStatus(_Status):
    backend: str
    connected: bool
    available: bool
    resolution: tuple[int, int] | None = None
    last_error: str | None = None


class HardwareInfo(_Status):
    """Identity of the hardware stack; stored with every scan as its hardware version."""

    controller: str
    stage_backend: str
    stage_version: str | None = None
    adc_backend: str
    adc_version: str | None = None
    laser_backend: str
    camera_backend: str
    details: dict[str, str] = Field(default_factory=dict)


class HardwareStatus(_Status):
    stage: StageStatus
    adc: ADCStatus
    laser: LaserStatus
    camera: CameraStatus
    estop_engaged: bool = False
    estop_reason: str | None = None
