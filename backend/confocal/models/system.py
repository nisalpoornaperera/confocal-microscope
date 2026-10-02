"""System-level API models (info, status, stage requests, errors)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from confocal.models.common import Axis, StageLimits
from confocal.models.hardware import HardwareInfo, HardwareStatus
from confocal.models.scan import ScanState


class SystemInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "confocal-surface-scanner"
    software_version: str
    api_version: str = "v1"
    python_version: str
    platform: str
    simulation: bool
    hardware: HardwareInfo
    limits: StageLimits
    data_dir: str


class SystemStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "estop", "error"]
    hardware: HardwareStatus
    estop_engaged: bool
    active_scan_id: str | None = None
    active_scan_state: ScanState | None = None
    calibration_version: int | None = None
    uptime_s: float


class StageMoveRequest(BaseModel):
    """Manual stage move. Refused while a scan is active or the e-stop is latched."""

    model_config = ConfigDict(extra="forbid")

    x_um: float | None = Field(default=None, allow_inf_nan=False)
    y_um: float | None = Field(default=None, allow_inf_nan=False)
    z_um: float | None = Field(default=None, allow_inf_nan=False)
    relative: bool = False

    @model_validator(mode="after")
    def _at_least_one_axis(self) -> StageMoveRequest:
        if self.x_um is None and self.y_um is None and self.z_um is None:
            raise ValueError("at least one of x_um, y_um, z_um is required")
        return self


class StageHomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axes: list[Axis] = Field(default_factory=lambda: [Axis.X, Axis.Y, Axis.Z], min_length=1)


class EmergencyStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field("operator emergency stop", max_length=500)


class ErrorResponse(BaseModel):
    """Body of every non-2xx response produced by the application's error handlers."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(description="Exception class name, e.g. LimitViolationError.")
    detail: str
    violations: list[str] | None = None
