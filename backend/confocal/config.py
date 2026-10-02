"""Application configuration.

Loaded from a TOML file (explicit path, or the ``CONFOCAL_CONFIG`` environment
variable). Every value has a safe default, so the application runs fully in
simulation mode out of the box. ``CONFOCAL_DATA_DIR`` overrides the data
directory.

Only *machine constants* live here (travel limits, steps per micrometre, bus
addresses). Calibration *values* (dark level, reference level) are measured at
runtime and versioned in the database; they are never configured.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from confocal.models.common import Position, StageLimits, default_stage_limits
from confocal.models.hardware import AdcGain
from confocal.models.processing import ProcessingConfig

CONFIG_ENV_VAR = "CONFOCAL_CONFIG"
DATA_DIR_ENV_VAR = "CONFOCAL_DATA_DIR"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HardwareSelection(_Section):
    """Which backend implements each hardware role.

    Roles are independent so that, e.g., an OpenFlexure-server stage can be
    combined with an ADS1115 photodiode ADC on the same Pi.
    """

    stage: Literal["simulation", "arduino", "openflexure"] = "simulation"
    adc: Literal["simulation", "ads1115"] = "simulation"
    laser: Literal["simulation", "manual", "gpio"] = "simulation"
    camera: Literal["none", "simulation"] = "none"


class MotionConfig(_Section):
    """Application-level motion parameters (micrometres)."""

    position_tolerance_um: float = Field(
        0.5, gt=0, description="Max |reported - commanded| per axis before a move is a failure."
    )
    move_timeout_s: float = Field(
        120.0,
        gt=0,
        description=(
            "Fixed margin (s) of every move's timeout, added to move_timeout_factor x the "
            "expected duration (distance / speed); see move_timeout_for()."
        ),
    )
    move_timeout_factor: float = Field(
        2.0,
        ge=1.0,
        description="Safety factor applied to a move's expected duration in its timeout.",
    )
    xy_speed_um_s: float = Field(
        40.0, gt=0, description="Used for scan-time estimates and move timeouts."
    )
    z_speed_um_s: float = Field(
        30.0, gt=0, description="Used for scan-time estimates and move timeouts."
    )
    per_move_overhead_s: float = Field(0.01, ge=0, description="Command round-trip estimate.")
    release_motors_when_idle: bool = Field(
        default=True, description="De-energise 28BYJ-48 coils when idle to limit heating."
    )

    def expected_move_s(self, start: Position, end: Position) -> float:
        """Nominal duration of a straight move (the three motors are interpolated).

        The slowest axis sets it: ``max(|dx| / v_xy, |dy| / v_xy, |dz| / v_z)``.
        """
        return max(
            abs(end.x_um - start.x_um) / self.xy_speed_um_s,
            abs(end.y_um - start.y_um) / self.xy_speed_um_s,
            abs(end.z_um - start.z_um) / self.z_speed_um_s,
        )

    def move_timeout_for(self, start: Position, end: Position) -> float:
        """Timeout of one move: ``move_timeout_factor * expected + move_timeout_s``.

        Distance-aware, so a full-range move is not mistaken for a stall, while
        a stage that stops reporting progress still times out.
        """
        return self.move_timeout_factor * self.expected_move_s(start, end) + self.move_timeout_s


class KinematicsConfig(_Section):
    """How Cartesian micrometres map onto the three motors a, b, c.

    Used only by the hardware layer (``confocal.hardware.kinematics``). The
    defaults describe an OpenFlexure Delta Stage with nominal geometry; the
    scale factors are approximate and MUST be calibrated per machine (best:
    measure the full matrix and use ``geometry = "matrix"``).
    """

    geometry: Literal["delta", "cartesian", "matrix"] = "delta"

    # delta: OpenFlexure Delta Stage (lever dimensions; only ratios matter)
    delta_flex_h: float = Field(80.0, gt=0)
    delta_flex_a: float = Field(50.0, gt=0)
    delta_flex_b: float = Field(50.0, gt=0)
    um_per_step: tuple[float, float, float] = Field(
        (0.06, 0.06, 0.06),
        description="delta: micrometres per stage step along x, y, z (calibrate).",
    )
    xy_rotation_deg: float = Field(
        0.0, description="Rotation of the X/Y frame about Z; 0 without a camera."
    )

    # cartesian: one motor per axis (a -> X, b -> Y, c -> Z)
    steps_per_um: tuple[float, float, float] = Field(
        (14.3, 14.3, 20.0), description="cartesian: motor steps per micrometre (x, y, z)."
    )

    # matrix: measured M[i][j] = micrometres along axis i (x, y, z) per step of motor j (a, b, c)
    um_per_step_matrix: list[list[float]] | None = None

    @model_validator(mode="after")
    def _check_geometry(self) -> KinematicsConfig:
        if any(not v > 0 for v in (*self.um_per_step, *self.steps_per_um)):
            raise ValueError("um_per_step and steps_per_um values must be positive")
        if self.geometry == "matrix":
            m = self.um_per_step_matrix
            if m is None or len(m) != 3 or any(len(row) != 3 for row in m):
                raise ValueError("geometry 'matrix' requires a 3 x 3 um_per_step_matrix")
        return self


class MotorConfig(_Section):
    """One 28BYJ-48 + ULN2003 channel as driven by the Arduino firmware.

    Steps are counted from the power-up origin (the stage has no end-stops).
    ``min_steps`` / ``max_steps`` are the actuator's safe travel; at startup the
    Cartesian travel limits are checked to lie entirely inside them.
    """

    invert: bool = False
    backlash_steps: int = Field(
        0, ge=0, le=5000, description="Per-motor backlash; every move finishes approaching +."
    )
    max_speed_steps_s: float = Field(600.0, gt=0, le=2000)
    min_steps: int = -200_000
    max_steps: int = 200_000

    @model_validator(mode="after")
    def _check_travel(self) -> MotorConfig:
        if not self.min_steps < 0 < self.max_steps:
            raise ValueError("motor travel must include the origin: min_steps < 0 < max_steps")
        return self


class ArduinoConfig(_Section):
    """Serial link to the Arduino (Uno or Nano, ATmega328P) driving three ULN2003 boards."""

    port: str = Field(
        default="/dev/ttyACM0",
        description=(
            "Genuine Uno R3/R4 boards enumerate as /dev/ttyACM*, CH340 clones as "
            "/dev/ttyUSB*. Prefer the stable /dev/serial/by-id/... path."
        ),
    )
    baudrate: int = 115200
    timeout_s: float = Field(2.0, gt=0)
    handshake_timeout_s: float = Field(
        5.0,
        gt=0,
        description="Opening the port resets the Arduino (DTR auto-reset); allow it to boot.",
    )
    a: MotorConfig = Field(default_factory=MotorConfig)
    b: MotorConfig = Field(default_factory=MotorConfig)
    c: MotorConfig = Field(default_factory=MotorConfig)


class ADS1115Config(_Section):
    i2c_bus: int = 1
    address: int = Field(0x48, ge=0x48, le=0x4B)
    channel: int = Field(0, ge=0, le=3)
    differential: bool = False
    gain: AdcGain = AdcGain.G1
    data_rate_sps: Literal[8, 16, 32, 64, 128, 250, 475, 860] = 860


class LaserConfig(_Section):
    wavelength_nm: float = 650.0
    power_mw: float = 5.0
    gpio_pin: int | None = None
    warmup_s: float = Field(0.5, ge=0, description="Settling time after switching the laser on.")


class SimulatedSurfaceConfig(_Section):
    """Synthetic sample used by the simulation."""

    kind: Literal["plane", "sinusoid", "steps", "sphere", "composite"] = "composite"
    base_z_um: float = 0.0
    tilt_x: float = Field(0.002, description="dZ/dX (dimensionless).")
    tilt_y: float = Field(-0.001, description="dZ/dY (dimensionless).")
    feature_amplitude_um: float = 5.0
    feature_period_um: float = Field(400.0, gt=0)
    step_height_um: float = 3.0
    sphere_radius_um: float = Field(150.0, gt=0)
    sphere_height_um: float = 8.0
    reflectivity: float = Field(0.9, ge=0, le=1)
    low_reflectivity_fraction: float = Field(
        0.03, ge=0, le=1, description="Fraction of the area with too little signal for a peak."
    )
    spurious_peak_probability: float = Field(
        0.01, ge=0, le=1, description="Probability that a point shows a false secondary peak."
    )


class SimulationConfig(_Section):
    seed: int = 1234
    time_scale: float = Field(
        0.05, ge=0, description="0 = instantaneous, 1 = real time (moves and conversions)."
    )
    surface: SimulatedSurfaceConfig = Field(default_factory=SimulatedSurfaceConfig)
    psf_model: Literal["gaussian", "sinc2"] = "gaussian"
    axial_fwhm_um: float = Field(6.0, gt=0)
    peak_voltage_v: float = Field(
        1.6, gt=0, description="OPT101 output at focus, reflectivity 1 (3.3 V supply)."
    )
    background_v: float = Field(0.05, ge=0, description="Stray light with the laser on.")
    dark_voltage_v: float = Field(0.0075, ge=0, description="OPT101 output with no light.")
    read_noise_v: float = Field(0.002, ge=0)
    shot_noise_fraction: float = Field(0.01, ge=0)
    opt101_saturation_v: float = Field(
        2.0, gt=0, description="Output rail on a 3.3 V supply (~2 V)."
    )
    fault_after_moves: int | None = Field(
        default=None, ge=0, description="Fault injection: fail the Nth move."
    )
    adc_fault_after_reads: int | None = Field(
        default=None, ge=0, description="Fault injection: fail the Nth ADC read."
    )


class StorageConfig(_Section):
    data_dir: Path = Path("data")

    @property
    def database_path(self) -> Path:
        return self.data_dir / "confocal.db"

    @property
    def scans_dir(self) -> Path:
        return self.data_dir / "scans"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


class ServerConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    websocket_queue_size: int = Field(256, ge=8, le=10_000)
    log_level: Literal["critical", "error", "warning", "info", "debug"] = "info"


class MLConfig(_Section):
    models_dir: Path | None = Field(
        default=None, description="Deployed model directory; defaults to <data_dir>/models."
    )
    default_model: str | None = None


class Settings(_Section):
    hardware: HardwareSelection = Field(default_factory=HardwareSelection)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    limits: StageLimits = Field(default_factory=default_stage_limits)
    kinematics: KinematicsConfig = Field(default_factory=KinematicsConfig)
    arduino: ArduinoConfig = Field(default_factory=ArduinoConfig)
    ads1115: ADS1115Config = Field(default_factory=ADS1115Config)
    laser: LaserConfig = Field(default_factory=LaserConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)

    @property
    def is_simulation(self) -> bool:
        return self.hardware.stage == "simulation" and self.hardware.adc == "simulation"

    @property
    def models_dir(self) -> Path:
        return self.ml.models_dir or self.storage.models_dir


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from TOML (``path`` or ``$CONFOCAL_CONFIG``), else defaults."""
    source = path if path is not None else os.environ.get(CONFIG_ENV_VAR)
    data: dict[str, Any] = {}
    if source:
        with Path(source).open("rb") as fh:
            data = tomllib.load(fh)
    settings = Settings.model_validate(data)
    data_dir = os.environ.get(DATA_DIR_ENV_VAR)
    if data_dir:
        settings = settings.model_copy(
            update={"storage": settings.storage.model_copy(update={"data_dir": Path(data_dir)})}
        )
    return settings
