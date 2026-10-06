"""Signal-processing configuration and results for I(Z) profiles.

Thresholds are deliberately unit-free (SNR, relative prominence) so they work
both with normalized intensity (when a reference calibration exists) and with
dark-corrected volts (when it does not).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FilterMethod(StrEnum):
    NONE = "none"
    SAVGOL = "savgol"
    MEDIAN = "median"
    GAUSSIAN = "gaussian"


class FitMethod(StrEnum):
    PARABOLIC = "parabolic"
    GAUSSIAN = "gaussian"


class PointStatus(StrEnum):
    """Outcome of measuring one XY point. Only VALID points define the surface."""

    VALID = "valid"
    LOW_CONFIDENCE = "low_confidence"
    NO_PEAK = "no_peak"
    PEAK_AT_EDGE = "peak_at_edge"
    FIT_FAILED = "fit_failed"
    MEASURED = "measured"  # fixed-Z intensity map: an intensity, no surface height
    ABORTED = "aborted"  # acquisition interrupted (cancel / e-stop / hardware failure)
    ERROR = "error"


class ProcessingConfig(BaseModel):
    # Every float must be finite: inf / NaN would overflow sweep and grid sizes.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    filter_method: FilterMethod = FilterMethod.SAVGOL
    filter_window: int = Field(5, ge=3, le=101, description="Odd window length in samples.")
    savgol_polyorder: int = Field(2, ge=1, le=5)
    gaussian_sigma_samples: float = Field(1.0, gt=0, le=20)

    min_snr: float = Field(5.0, ge=0, description="Peak height above baseline / noise std.")
    min_relative_prominence: float = Field(
        0.3, ge=0, le=1, description="Prominence / peak height above baseline."
    )
    fit_window_fwhm: float = Field(
        1.5, gt=0, le=10, description="Fit samples within +/- this many FWHM of the peak."
    )
    max_fit_disagreement_um: float = Field(
        1.0, gt=0, description="Max |parabolic - gaussian| centre before flagging."
    )
    expected_fwhm_um: float | None = Field(
        default=None, gt=0, description="Expected axial FWHM; used for confidence when set."
    )
    edge_margin_samples: int = Field(
        2, ge=0, le=50, description="A peak this close to the sweep end is 'at edge'."
    )
    accept_weak_peaks: bool = Field(
        default=False,
        description=(
            "Low-contrast mode: keep a peak that is not significant against the noise "
            "(the fixed 4-sigma rule) as long as it passes min_snr and "
            "min_relative_prominence. Such points get the 'weak_peak' flag and are at "
            "most LOW_CONFIDENCE, never VALID."
        ),
    )
    min_confidence: float = Field(
        0.5, ge=0, le=1, description="Below this the point is LOW_CONFIDENCE, not VALID."
    )
    saturation_v: float | None = Field(
        default=None, gt=0, description="Voltage at/above which samples count as saturated."
    )

    @field_validator("filter_window")
    @classmethod
    def _odd_window(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("filter_window must be odd")
        return value


class PeakFit(BaseModel):
    """Result of fitting one peak model around the detected maximum."""

    model_config = ConfigDict(extra="forbid")

    method: FitMethod
    success: bool
    center_um: float | None = None
    amplitude: float | None = None
    fwhm_um: float | None = None
    offset: float | None = None
    residual_rms: float | None = None
    r_squared: float | None = None
    n_points: int = 0
    message: str | None = None


class CoarsePeak(BaseModel):
    """Approximate peak location from the coarse Z sweep."""

    model_config = ConfigDict(extra="forbid")

    found: bool
    z_um: float | None = None
    index: int | None = None
    at_edge: bool = False
    global_max_not_selected: bool = Field(
        default=False,
        description="The sweep's brightest signal is not the selected peak (ambiguous).",
    )
    snr: float | None = None
    relative_prominence: float | None = None
    message: str | None = None


class ProfileAnalysis(BaseModel):
    """Every scalar derived from one I(Z) profile (the physics baseline)."""

    model_config = ConfigDict(extra="forbid")

    status: PointStatus
    signal_units: Literal["normalized", "volts"]
    peak_found: bool = False
    peak_index: int | None = None
    peak_z_um: float | None = Field(default=None, description="Z of the filtered maximum.")
    peak_intensity: float | None = None
    baseline: float | None = None
    noise_std: float | None = None

    parabolic: PeakFit | None = None
    gaussian: PeakFit | None = None
    surface_z_um: float | None = None
    surface_method: FitMethod | None = None

    snr: float | None = None
    fwhm_um: float | None = None
    prominence: float | None = None
    relative_prominence: float | None = None
    fit_residual: float | None = None
    confidence: float = Field(0.0, ge=0, le=1)

    n_peaks: int = 0
    secondary_peak_ratio: float | None = Field(
        default=None, description="Second-highest peak prominence / main peak prominence."
    )
    asymmetry: float | None = Field(
        default=None, description="Signed asymmetry of the peak (0 = symmetric)."
    )
    saturated_fraction: float = 0.0
    flags: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ProfileProcessingResult:
    """Scalar analysis plus the processed arrays that are stored next to the raw data."""

    analysis: ProfileAnalysis
    corrected_v: NDArray[np.float64]  # dark-subtracted voltage, same order as the input
    normalized: NDArray[np.float64] | None  # None when no reference calibration
    filtered: NDArray[np.float64]  # filtered signal in ``analysis.signal_units``
