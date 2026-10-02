"""Surface-reconstruction request and result models.

Grids are row-major ``[iy][ix]`` with ``None`` for cells that have no value
(gaps / outside the convex hull). Never NaN in these API models.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from confocal.models.common import utc_now


class InterpolationMethod(StrEnum):
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    RBF = "rbf"


class OutlierMethod(StrEnum):
    NONE = "none"
    LOCAL_MAD = "local_mad"  # robust z-score against the k nearest neighbours
    GLOBAL_MAD = "global_mad"  # robust z-score against a fitted plane over all points


RbfKernel = Literal[
    "thin_plate_spline",
    "linear",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
]


class ReconstructionRequest(BaseModel):
    # Every float must be finite: inf / NaN would overflow sweep and grid sizes.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    method: InterpolationMethod = InterpolationMethod.LINEAR
    min_confidence: float = Field(0.5, ge=0, le=1)
    outlier_method: OutlierMethod = OutlierMethod.LOCAL_MAD
    outlier_threshold: float = Field(3.5, gt=0, description="Robust z-score threshold.")
    outlier_neighbours: int = Field(8, ge=3, le=64)
    grid_step_um: float | None = Field(
        default=None, gt=0, description="Output grid spacing; defaults to the scan XY step."
    )
    max_gap_distance_um: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Grid cells farther than this from any used point are gaps. "
            "Defaults to 1.5 x the scan XY step."
        ),
    )
    fill_gaps: bool = Field(default=False, description="If false, gap cells are returned as None.")
    rbf_kernel: RbfKernel = "thin_plate_spline"
    rbf_smoothing: float = Field(0.0, ge=0)
    rbf_neighbors: int | None = Field(
        default=None, ge=4, description="Local RBF neighbourhood size (None = global)."
    )
    build_mesh: bool = True
    cross_section_count: int = Field(
        1, ge=0, le=16, description="Evenly spaced X and Y cross-sections to include."
    )


class PointClassification(StrEnum):
    USED = "used"
    INVALID = "invalid"  # no surface height (no peak / fit failed / aborted / non-finite)
    LOW_CONFIDENCE = "low_confidence"
    OUTLIER = "outlier"


class SurfacePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point_id: int
    x_um: float
    y_um: float
    z_um: float | None
    confidence: float
    classification: PointClassification


class CrossSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    along: Literal["x", "y"] = Field(description="Axis the section runs along.")
    position_um: float = Field(description="Fixed coordinate of the other axis.")
    coordinate_um: list[float]
    z_um: list[float | None]


class MeshData(BaseModel):
    """Triangle mesh of the used points (Delaunay in XY)."""

    model_config = ConfigDict(extra="forbid")

    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]


class GapRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: int
    n_cells: int
    area_um2: float
    centroid_x_um: float
    centroid_y_um: float
    bbox_um: tuple[float, float, float, float] = Field(description="(x_min, y_min, x_max, y_max)")


class SurfaceStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_input: int
    n_invalid: int
    n_low_confidence: int
    n_outliers: int
    n_used: int
    coverage_fraction: float = Field(description="Fraction of grid cells with a value.")
    gap_fraction: float
    n_gaps: int
    z_min_um: float | None = None
    z_max_um: float | None = None
    z_mean_um: float | None = None
    z_std_um: float | None = None
    sa_um: float | None = Field(default=None, description="Arithmetic mean height (ISO 25178).")
    sq_um: float | None = Field(default=None, description="Root-mean-square height.")
    sz_um: float | None = Field(default=None, description="Maximum height (peak to valley).")
    ssk: float | None = Field(default=None, description="Skewness of the height distribution.")
    sku: float | None = Field(default=None, description="Kurtosis of the height distribution.")
    plane_coefficients: tuple[float, float, float] | None = Field(
        default=None, description="Least-squares plane z = a*x + b*y + c (a, b, c)."
    )
    mean_confidence: float | None = None


class SurfaceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    surface_id: int | None = None
    scan_id: str
    created_at: datetime = Field(default_factory=utc_now)
    request: ReconstructionRequest
    x_um: list[float] = Field(description="Grid X coordinates (columns).")
    y_um: list[float] = Field(description="Grid Y coordinates (rows).")
    z_um: list[list[float | None]] = Field(description="Height map Z(X, Y), shape [ny][nx].")
    confidence: list[list[float | None]] = Field(description="Interpolated confidence map.")
    gap_mask: list[list[bool]] = Field(description="True where the cell is a gap.")
    points: list[SurfacePoint] = Field(description="Point cloud with per-point classification.")
    mesh: MeshData | None = None
    cross_sections: list[CrossSection] = Field(default_factory=list)
    gaps: list[GapRegion] = Field(default_factory=list)
    statistics: SurfaceStatistics
