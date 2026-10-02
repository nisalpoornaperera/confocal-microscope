"""Measurement containers.

``AdcSamples`` and ``ProfileData`` are plain dataclasses holding NumPy arrays
(internal, never serialised directly). ``IntensityMeasurement`` and
``ProfileRecord`` are their API-facing Pydantic counterparts.

API models must never contain NaN/inf: Starlette's JSON encoder rejects them.
Use ``None`` for missing values (see :func:`nan_to_none`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from confocal.models.common import Position, utc_now
from confocal.models.hardware import AdcGain, SamplingMethod
from confocal.models.processing import ProfileAnalysis


def nan_to_none(values: Iterable[float]) -> list[float | None]:
    """Convert an iterable of floats to a JSON-safe list (NaN/inf -> None)."""
    return [float(v) if math.isfinite(v) else None for v in values]


class ProfilePhase(IntEnum):
    """Which sweep a Z sample belongs to."""

    COARSE = 0
    FINE = 1
    FIXED = 2


@dataclass(frozen=True, slots=True)
class AdcSamples:
    """A burst of consecutive ADC conversions taken at one stage position."""

    counts: NDArray[np.int32]  # (n,) raw signed ADC codes
    volts: NDArray[np.float64]  # (n,) counts converted with the gain in effect
    timestamps: NDArray[np.float64]  # (n,) unix time of each conversion
    gain: AdcGain
    data_rate_sps: int

    def __post_init__(self) -> None:
        n = self.counts.shape
        if len(n) != 1 or self.volts.shape != n or self.timestamps.shape != n:
            raise ValueError("counts, volts and timestamps must be 1-D arrays of equal length")
        if n[0] == 0:
            raise ValueError("AdcSamples must contain at least one sample")

    @property
    def n(self) -> int:
        return int(self.counts.shape[0])


@dataclass(slots=True)
class ProfileData:
    """Every value recorded at one XY point, in acquisition order. Never discarded.

    ``n`` = number of Z positions visited, ``s`` = ADC samples per position.
    """

    z_um: NDArray[np.float64]  # (n,) commanded Z
    z_reported_um: NDArray[np.float64]  # (n,) Z reported by the stage after the move
    phase: NDArray[np.uint8]  # (n,) ProfilePhase values
    raw_counts: NDArray[np.int32]  # (n, s) every raw ADC code
    voltage_v: NDArray[np.float64]  # (n, s) every raw voltage
    voltage_agg_v: NDArray[np.float64]  # (n,) per-position mean or median
    timestamps: NDArray[np.float64]  # (n,) unix time of the first sample at each position
    gain: AdcGain
    sampling_method: SamplingMethod
    dark_v: float | None
    reference_v: float | None
    calibration_version: int | None
    normalized: NDArray[np.float64] | None = None  # (n,) None without reference calibration
    filtered: NDArray[np.float64] | None = None  # (n,) NaN where not analysed

    def __post_init__(self) -> None:
        n = self.z_um.shape[0]
        for name in ("z_reported_um", "phase", "voltage_agg_v", "timestamps"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        if self.raw_counts.ndim != 2 or self.raw_counts.shape[0] != n:
            raise ValueError(f"raw_counts must have shape ({n}, s)")
        if self.voltage_v.shape != self.raw_counts.shape:
            raise ValueError("voltage_v must have the same shape as raw_counts")
        for name in ("normalized", "filtered"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")

    @property
    def n_positions(self) -> int:
        return int(self.z_um.shape[0])

    @property
    def samples_per_position(self) -> int:
        return int(self.raw_counts.shape[1])


class IntensityMeasurement(BaseModel):
    """One calibrated intensity reading at the current stage position."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=utc_now)
    position: Position | None = None
    n_samples: int
    method: SamplingMethod
    gain: AdcGain
    raw_counts: list[int]
    voltages_v: list[float]
    voltage_v: float = Field(description="Aggregated (mean/median) raw voltage.")
    voltage_std_v: float
    dark_v: float | None = None
    corrected_v: float = Field(description="voltage_v - dark_v (dark treated as 0 if unknown).")
    reference_v: float | None = None
    normalized: float | None = Field(
        default=None, description="(V - dark) / (reference - dark); None without reference."
    )
    calibration_version: int | None = None
    saturated: bool = False


class ProfileRecord(BaseModel):
    """API representation of the complete stored I(Z) data of one point."""

    model_config = ConfigDict(extra="forbid")

    scan_id: str
    point_id: int
    x_um: float
    y_um: float
    z_um: list[float]
    z_reported_um: list[float]
    phase: list[int]
    raw_counts: list[list[int]]
    voltage_v: list[list[float]]
    voltage_agg_v: list[float]
    timestamps: list[float]
    gain: AdcGain
    sampling_method: SamplingMethod
    dark_v: float | None = None
    reference_v: float | None = None
    calibration_version: int | None = None
    normalized: list[float | None] | None = None
    filtered: list[float | None] | None = None
    analysis: ProfileAnalysis | None = None
