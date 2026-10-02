"""The regular output grid: geometry, JSON conversion and cross-sections.

The grid spans the bounding box of the *used* points exactly (first and last
nodes on the extreme points), with a spacing that is the requested step or
marginally less: the node count is ``ceil(span / step) + 1``, with a
tolerance of a thousandth of a step so that a bounding box that is a whole
number of steps (up to the delta stage's position quantisation) does not gain
an extra column. Arrays are row-major ``[iy, ix]``, matching the API models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from confocal.errors import ReconstructionError
from confocal.models.surface import CrossSection

#: Refuse grids larger than this (memory, CPU time and JSON size on the Pi).
MAX_GRID_CELLS = 1_000_000

_STEP_TOLERANCE = 1e-3


def _axis(lo: float, hi: float, step_um: float) -> NDArray[np.float64]:
    span = hi - lo
    n = max(1, math.ceil(span / step_um - _STEP_TOLERANCE) + 1)
    if n == 1:
        return np.array([0.5 * (lo + hi)], dtype=np.float64)
    return np.linspace(lo, hi, n, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class OutputGrid:
    x_um: NDArray[np.float64]  # (nx,) column coordinates, increasing
    y_um: NDArray[np.float64]  # (ny,) row coordinates, increasing
    step_um: float  # requested spacing (used for single-node axes)

    @classmethod
    def covering(
        cls, x_um: NDArray[np.float64], y_um: NDArray[np.float64], step_um: float
    ) -> OutputGrid:
        """Grid over the bounding box of the given coordinates.

        Raises:
            ReconstructionError: if the grid would exceed :data:`MAX_GRID_CELLS`.
        """
        x_lo, x_hi = float(np.min(x_um)), float(np.max(x_um))
        y_lo, y_hi = float(np.min(y_um)), float(np.max(y_um))
        nx = math.ceil((x_hi - x_lo) / step_um - _STEP_TOLERANCE) + 1
        ny = math.ceil((y_hi - y_lo) / step_um - _STEP_TOLERANCE) + 1
        if nx * ny > MAX_GRID_CELLS:
            raise ReconstructionError(
                f"output grid of {nx} x {ny} cells exceeds the maximum of {MAX_GRID_CELLS}; "
                "increase grid_step_um"
            )
        return cls(
            x_um=_axis(x_lo, x_hi, step_um), y_um=_axis(y_lo, y_hi, step_um), step_um=step_um
        )

    @property
    def nx(self) -> int:
        return int(self.x_um.shape[0])

    @property
    def ny(self) -> int:
        return int(self.y_um.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def dx_um(self) -> float:
        return float(self.x_um[1] - self.x_um[0]) if self.nx > 1 else self.step_um

    @property
    def dy_um(self) -> float:
        return float(self.y_um[1] - self.y_um[0]) if self.ny > 1 else self.step_um

    @property
    def cell_area_um2(self) -> float:
        return self.dx_um * self.dy_um

    def mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(X, Y)`` coordinate arrays of shape ``(ny, nx)``."""
        gx, gy = np.meshgrid(self.x_um, self.y_um)
        return np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)

    def points(self) -> NDArray[np.float64]:
        """Cell centres as ``(ny * nx, 2)`` XY pairs in row-major order."""
        gx, gy = self.mesh()
        return np.column_stack((gx.ravel(), gy.ravel()))


def to_json_grid(values: NDArray[np.float64]) -> list[list[float | None]]:
    """Nested ``[iy][ix]`` lists with every non-finite value replaced by None."""
    out = np.asarray(values, dtype=np.float64).astype(object)
    out[~np.isfinite(values)] = None
    rows: list[list[float | None]] = out.tolist()
    return rows


def _section_indices(n: int, count: int) -> list[int]:
    """``count`` evenly spaced interior indices of an axis of ``n`` nodes (deduplicated)."""
    picks = (round((i + 1) / (count + 1) * (n - 1)) for i in range(count))
    return sorted(set(picks))


def cross_sections(
    grid: OutputGrid, heights: NDArray[np.float64], count: int
) -> list[CrossSection]:
    """``count`` evenly spaced sections along X (fixed rows) and along Y (fixed columns).

    Sections are cut from the final height map, so gaps appear as None.
    """
    if count <= 0:
        return []
    sections: list[CrossSection] = []
    x_coords = [float(v) for v in grid.x_um]
    y_coords = [float(v) for v in grid.y_um]
    for row in _section_indices(grid.ny, count):
        sections.append(
            CrossSection(
                along="x",
                position_um=y_coords[row],
                coordinate_um=x_coords,
                z_um=to_json_grid(heights[row : row + 1, :])[0],
            )
        )
    for col in _section_indices(grid.nx, count):
        sections.append(
            CrossSection(
                along="y",
                position_um=x_coords[col],
                coordinate_um=y_coords,
                z_um=to_json_grid(heights[:, col][None, :])[0],
            )
        )
    return sections
