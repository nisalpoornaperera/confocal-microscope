"""Scan configuration, state, per-point results and live-progress events."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from confocal.models.calibration import CalibrationState
from confocal.models.common import utc_now
from confocal.models.hardware import HardwareInfo, SamplingMethod
from confocal.models.processing import PointStatus, ProcessingConfig
from confocal.models.surface import ReconstructionRequest

MAX_SCAN_POINTS = 250_000
MAX_Z_POSITIONS_PER_POINT = 5_000
_GRID_EPS = 1e-9


def axis_count(start_um: float, stop_um: float, step_um: float) -> int:
    """Number of grid positions from ``start`` to ``stop`` (inclusive) at ``step``.

    The last position never exceeds ``stop``; a zero-length range gives 1 position.
    """
    if step_um <= 0:
        raise ValueError("step must be > 0")
    span = stop_um - start_um
    if span < 0:
        raise ValueError("stop must be >= start")
    return math.floor(span / step_um + _GRID_EPS) + 1


class ScanMode(StrEnum):
    FIXED_Z = "fixed_z"  # intensity map at a constant Z
    CONFOCAL = "confocal"  # surface scan: Z sweep + peak detection at every XY point


class ScanOrder(StrEnum):
    SERPENTINE = "serpentine"  # boustrophedon: alternate X direction on every row
    RASTER = "raster"  # every row in the same X direction


class ScanState(StrEnum):
    IDLE = "idle"
    PREPARING = "preparing"
    CALIBRATING = "calibrating"
    HOMING = "homing"
    SCANNING = "scanning"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    ERROR = "error"
    PROCESSING = "processing"
    SURFACE_RECONSTRUCTION = "surface_reconstruction"
    ML_PROCESSING = "ml_processing"
    COMPLETE = "complete"


#: States in which a scan owns the instrument (manual hardware control is refused).
ACTIVE_SCAN_STATES: frozenset[ScanState] = frozenset(
    {
        ScanState.PREPARING,
        ScanState.CALIBRATING,
        ScanState.HOMING,
        ScanState.SCANNING,
        ScanState.PAUSED,
        ScanState.PROCESSING,
        ScanState.SURFACE_RECONSTRUCTION,
        ScanState.ML_PROCESSING,
    }
)

#: States a scan never leaves.
TERMINAL_SCAN_STATES: frozenset[ScanState] = frozenset(
    {ScanState.CANCELLED, ScanState.ERROR, ScanState.COMPLETE}
)


class ScanConfig(BaseModel):
    """Everything needed to reproduce a scan. Stored verbatim with the scan."""

    # Every float must be finite: inf / NaN would overflow sweep and grid sizes.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str | None = Field(default=None, max_length=200)
    mode: ScanMode = ScanMode.CONFOCAL

    x_start_um: float
    x_stop_um: float
    y_start_um: float
    y_stop_um: float
    xy_step_um: float = Field(gt=0)

    z_center_um: float = Field(0.0)
    z_range_um: float = Field(100.0, gt=0, description="Full width of the coarse Z sweep.")
    coarse_z_step_um: float = Field(2.0, gt=0)
    fine_z_step_um: float = Field(0.25, gt=0)
    fine_z_range_um: float = Field(12.0, gt=0, description="Full width of the fine Z sweep.")

    order: ScanOrder = ScanOrder.SERPENTINE
    adaptive_z: bool = Field(
        default=True,
        description="Centre each coarse sweep on the previous valid point's surface Z.",
    )
    adaptive_z_range_um: float = Field(
        30.0, gt=0, description="Coarse sweep width when an adaptive estimate is available."
    )

    samples_per_z: int = Field(4, ge=1, le=256)
    sampling_method: SamplingMethod = SamplingMethod.MEAN
    settle_time_ms: float = Field(10.0, ge=0, le=10_000)

    home_before_scan: bool = False
    calibrate_dark_before_scan: bool = False

    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    reconstruct_on_complete: bool = True
    reconstruction: ReconstructionRequest = Field(default_factory=ReconstructionRequest)
    ml_on_complete: bool = False

    @property
    def n_x(self) -> int:
        return axis_count(self.x_start_um, self.x_stop_um, self.xy_step_um)

    @property
    def n_y(self) -> int:
        return axis_count(self.y_start_um, self.y_stop_um, self.xy_step_um)

    @property
    def total_points(self) -> int:
        return self.n_x * self.n_y

    @model_validator(mode="after")
    def _validate(self) -> ScanConfig:
        if self.x_stop_um < self.x_start_um:
            raise ValueError("x_stop_um must be >= x_start_um")
        if self.y_stop_um < self.y_start_um:
            raise ValueError("y_stop_um must be >= y_start_um")
        if self.total_points > MAX_SCAN_POINTS:
            raise ValueError(f"scan has {self.total_points} points (max {MAX_SCAN_POINTS})")
        if self.mode is ScanMode.CONFOCAL:
            if self.fine_z_step_um > self.coarse_z_step_um:
                raise ValueError("fine_z_step_um must be <= coarse_z_step_um")
            if self.fine_z_range_um > self.z_range_um:
                raise ValueError("fine_z_range_um must be <= z_range_um")
            if self.fine_z_range_um < 2 * self.coarse_z_step_um:
                raise ValueError(
                    "fine_z_range_um must be >= 2 x coarse_z_step_um so the fine sweep "
                    "covers the coarse quantisation"
                )
            if self.adaptive_z_range_um > self.z_range_um:
                raise ValueError("adaptive_z_range_um must be <= z_range_um")
            per_point = axis_count(0.0, self.z_range_um, self.coarse_z_step_um) + axis_count(
                0.0, self.fine_z_range_um, self.fine_z_step_um
            )
            if per_point > MAX_Z_POSITIONS_PER_POINT:
                raise ValueError(
                    f"{per_point} Z positions per point (max {MAX_Z_POSITIONS_PER_POINT})"
                )
        return self


class ScanEstimate(BaseModel):
    """Pre-scan estimate shown in Scan Setup."""

    model_config = ConfigDict(extra="forbid")

    n_x: int
    n_y: int
    total_points: int
    z_positions_per_point: int
    total_measurements: int = Field(description="Z positions visited over the whole scan.")
    total_adc_samples: int
    estimated_duration_s: float
    estimated_data_bytes: int
    within_limits: bool
    limit_violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ScanPoint(BaseModel):
    """Scalar result of one XY point. Raw arrays live in HDF5 (see ProfileRecord)."""

    model_config = ConfigDict(extra="forbid")

    point_id: int = Field(ge=0, description="Index in acquisition order.")
    ix: int = Field(ge=0)
    iy: int = Field(ge=0)
    x_um: float
    y_um: float
    status: PointStatus

    z_estimate_um: float | None = None
    coarse_peak_z_um: float | None = None
    surface_z_um: float | None = None
    parabolic_z_um: float | None = None
    gaussian_z_um: float | None = None
    peak_intensity: float | None = None
    snr: float | None = None
    peak_width_um: float | None = Field(default=None, description="FWHM of the axial peak.")
    prominence: float | None = None
    fit_residual: float | None = None
    confidence: float = Field(0.0, ge=0, le=1)
    intensity: float | None = Field(
        default=None, description="Fixed-Z mode: calibrated intensity at z_center."
    )
    secondary_peak_ratio: float | None = None
    asymmetry: float | None = None
    n_z_positions: int = 0
    flags: list[str] = Field(default_factory=list)
    acquired_at: datetime = Field(default_factory=utc_now)
    duration_s: float = 0.0


class ScanSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    mode: ScanMode
    state: ScanState
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    total_points: int
    completed_points: int = 0
    progress: float = Field(0.0, ge=0, le=1)
    config: ScanConfig
    calibration_version: int | None = None
    calibration: CalibrationState | None = None
    software_version: str
    hardware: HardwareInfo | None = None
    interrupted: bool = Field(
        default=False, description="True when the scan ended before all points were measured."
    )
    error_message: str | None = None
    data_file: str | None = None
    has_surface: bool = False
    has_ml_result: bool = False


class ScanProgress(BaseModel):
    """Live progress; included in every WebSocket event."""

    model_config = ConfigDict(extra="forbid")

    scan_id: str
    state: ScanState
    progress: float = Field(ge=0, le=1)
    completed_points: int
    total_points: int
    current_point_id: int | None = None
    current_x_um: float | None = None
    current_y_um: float | None = None
    current_z_um: float | None = None
    current_intensity: float | None = None
    elapsed_s: float = 0.0
    estimated_remaining_s: float | None = None
    message: str | None = None


class ScanEventType(StrEnum):
    SNAPSHOT = "snapshot"  # first message after a client connects
    STATE = "state"  # state transition
    PROGRESS = "progress"  # periodic position / intensity update
    POINT = "point"  # an XY point finished
    PROFILE = "profile"  # live I(Z) data of the point being measured
    ERROR = "error"


class LiveProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point_id: int
    x_um: float
    y_um: float
    phase: list[int]
    z_um: list[float]
    intensity: list[float | None]


class ScanEvent(BaseModel):
    """Message published on ``/ws/scans/{scan_id}``."""

    model_config = ConfigDict(extra="forbid")

    type: ScanEventType
    scan_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    progress: ScanProgress
    point: ScanPoint | None = None
    profile: LiveProfile | None = None
    message: str | None = None
