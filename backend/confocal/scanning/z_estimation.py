"""Adaptive estimate of the expected surface Z at the next XY point (§4 step 2).

Real surfaces are continuous on the scale of one XY step, so the surface Z of
an adjacent, already measured point predicts where the confocal peak of the
next point lies. Centring the coarse sweep on that prediction lets it use the
narrow ``adaptive_z_range_um`` instead of the full ``z_range_um``, which is
most of the scan time on a slow stepper stage.

Only *adjacent* points are used: the predecessor in acquisition order when it
is a grid neighbour (always true for serpentine order, not at the start of a
raster row), else the nearest valid neighbour in the previous row. Without one
the full default range is swept, which is always safe. A wrong narrow guess is
caught by the executor's full-range retry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from confocal.models.scan import ScanConfig
from confocal.scanning.plan import GridPoint

ZEstimateSource = Literal["previous", "neighbour", "default"]


@dataclass(frozen=True, slots=True)
class ZEstimate:
    """Where to centre the coarse sweep of one point and how wide to make it."""

    center_um: float
    width_um: float
    source: ZEstimateSource


class AdaptiveZEstimator:
    """Tracks the surface Z of measured points on the scan grid."""

    def __init__(self, config: ScanConfig, *, n_x: int, n_y: int) -> None:
        if n_x < 1 or n_y < 1:
            raise ValueError("the grid needs at least one point in each direction")
        self._config = config
        self._surface = np.full((n_y, n_x), np.nan, dtype=np.float64)
        self._last: GridPoint | None = None

    def estimate(self, point: GridPoint) -> ZEstimate:
        """Coarse-sweep centre and width for ``point``."""
        cfg = self._config
        if cfg.adaptive_z:
            previous = self._previous_z(point)
            if previous is not None:
                return ZEstimate(previous, cfg.adaptive_z_range_um, "previous")
            neighbour = self._previous_row_z(point)
            if neighbour is not None:
                return ZEstimate(neighbour, cfg.adaptive_z_range_um, "neighbour")
        return ZEstimate(cfg.z_center_um, cfg.z_range_um, "default")

    def record(self, point: GridPoint, surface_z_um: float | None) -> None:
        """Remember the outcome of ``point``: its surface Z, or ``None`` if it is not valid."""
        value = surface_z_um if surface_z_um is not None and math.isfinite(surface_z_um) else None
        self._surface[point.iy, point.ix] = np.nan if value is None else value
        self._last = point

    def surface_z(self, ix: int, iy: int) -> float | None:
        """Recorded surface Z of a grid cell, or ``None``."""
        value = float(self._surface[iy, ix])
        return value if math.isfinite(value) else None

    def _previous_z(self, point: GridPoint) -> float | None:
        last = self._last
        if last is None or max(abs(last.ix - point.ix), abs(last.iy - point.iy)) > 1:
            return None
        return self.surface_z(last.ix, last.iy)

    def _previous_row_z(self, point: GridPoint) -> float | None:
        """Nearest valid value among the (up to three) neighbours in the row before."""
        if point.iy == 0:
            return None
        n_x = self._surface.shape[1]
        for ix in (point.ix, point.ix - 1, point.ix + 1):
            if 0 <= ix < n_x:
                value = self.surface_z(ix, point.iy - 1)
                if value is not None:
                    return value
        return None
