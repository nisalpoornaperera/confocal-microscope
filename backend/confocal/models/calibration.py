"""Calibration models.

Calibration values are *measured*, never configured. Every new dark or
reference measurement produces a new immutable :class:`CalibrationState`
snapshot with an incremented ``version`` (assigned by the CalibrationStore).
Each scan records the calibration version it used.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from confocal.models.common import Position, utc_now
from confocal.models.hardware import AdcGain, SamplingMethod


class CalibrationState(BaseModel):
    """Snapshot of the complete calibration after the most recent update."""

    model_config = ConfigDict(extra="forbid")

    version: int | None = Field(
        default=None, description="Assigned on persistence; None means never persisted."
    )
    created_at: datetime = Field(default_factory=utc_now)
    updated_field: str | None = Field(
        default=None, description="Which part changed in this version: 'dark' or 'reference'."
    )

    dark_v: float | None = None
    dark_std_v: float | None = None
    dark_n_samples: int = 0
    dark_measured_at: datetime | None = None
    dark_gain: AdcGain | None = None

    reference_v: float | None = None
    reference_std_v: float | None = None
    reference_n_samples: int = 0
    reference_measured_at: datetime | None = None
    reference_gain: AdcGain | None = None
    reference_position: Position | None = None

    notes: str | None = None

    @property
    def has_dark(self) -> bool:
        return self.dark_v is not None

    @property
    def has_reference(self) -> bool:
        return self.reference_v is not None

    @property
    def can_normalize(self) -> bool:
        """True when a normalized intensity (V - dark) / (ref - dark) can be computed."""
        if self.reference_v is None:
            return False
        return self.reference_v - (self.dark_v or 0.0) > 0.0


class DarkCalibrationRequest(BaseModel):
    """Measure the detector signal with no laser light reaching it."""

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(64, ge=1, le=4096)
    method: SamplingMethod = SamplingMethod.MEDIAN
    beam_blocked_confirmed: bool = Field(
        default=False,
        description=(
            "Required when the laser is not software-controllable: the operator confirms "
            "the beam is blocked. Ignored when the laser can be switched off automatically."
        ),
    )
    notes: str | None = Field(default=None, max_length=1000)


class ReferenceCalibrationRequest(BaseModel):
    """Measure the in-focus signal of a reference reflector (e.g. a plane mirror).

    With ``z_search`` the controller sweeps Z around the current position and
    uses the maximum; otherwise the operator must already be in focus.
    """

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(64, ge=1, le=4096)
    method: SamplingMethod = SamplingMethod.MEDIAN
    z_search: bool = False
    z_search_range_um: float = Field(40.0, gt=0, le=2000)
    z_search_step_um: float = Field(1.0, gt=0, le=100)
    notes: str | None = Field(default=None, max_length=1000)


class ADCCalibrateRequest(BaseModel):
    """Configure the ADC gain explicitly or let the controller auto-range it."""

    model_config = ConfigDict(extra="forbid")

    gain: AdcGain | None = None
    auto: bool = False
    target_fraction: float = Field(
        0.8, gt=0.1, le=0.95, description="Auto-gain: max signal as a fraction of full scale."
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> ADCCalibrateRequest:
        if (self.gain is None) == (not self.auto):
            raise ValueError("specify exactly one of 'gain' or 'auto=true'")
        return self


class ADCCalibrateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gain: AdcGain
    full_scale_v: float
    measured_max_v: float | None = None
    message: str
