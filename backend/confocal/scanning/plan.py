"""Scan plan: XY grid, acquisition order, Z sweeps, envelope validation and estimates.

Everything here is pure (no I/O), so the complete scan envelope can be checked
against the travel limits and the duration estimated before anything moves
(safety model, docs/architecture.md §2.1).

Grid positions are always computed as ``start + i * step`` (never by repeated
addition, which accumulates rounding error over hundreds of points) and the
last position never lies beyond ``stop``.

The plan works in Cartesian micrometres only. How the stage turns a Cartesian
move into motor steps (e.g. all three legs of a delta stage) is a hardware
concern; for estimates every move is a straight line at the configured speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from confocal.config import MotionConfig
from confocal.errors import LimitViolationError
from confocal.models.common import AxisLimits, StageLimits
from confocal.models.scan import ScanConfig, ScanEstimate, ScanMode, ScanOrder, axis_count

#: Positions closer than this (um) are treated as equal; far below one motor step.
_EPS_UM = 1e-9

#: Approximate smallest displacement of the 28BYJ-48 driven flexure stage (one
#: motor step moves the platform by roughly 0.05-0.06 um). Requested steps finer
#: than this cannot be resolved: consecutive commanded positions coincide.
MOTOR_RESOLUTION_UM = 0.05

#: Thresholds for estimate warnings.
LONG_SCAN_WARNING_S = 8 * 3600.0
LARGE_DATA_WARNING_BYTES = 2 * 1024**3

# Stored bytes: every ADC sample is kept as an int32 code and a float64 voltage;
# every Z position adds z, z_reported, voltage_agg, timestamp, normalized and
# filtered (float64) plus the uint8 phase; every point adds one (3,) int64 index row.
_BYTES_PER_SAMPLE = 4 + 8
_BYTES_PER_POSITION = 6 * 8 + 1
_BYTES_PER_POINT = 3 * 8


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One XY point of the scan grid. ``point_id`` is its index in acquisition order."""

    point_id: int
    ix: int
    iy: int
    x_um: float
    y_um: float


def axis_positions(start_um: float, stop_um: float, step_um: float) -> NDArray[np.float64]:
    """Grid positions ``start + i * step`` from ``start`` to ``stop`` (inclusive)."""
    n = axis_count(start_um, stop_um, step_um)
    positions = start_um + np.arange(n, dtype=np.float64) * step_um
    # The floor(span / step + eps) count can admit a last position that exceeds
    # ``stop`` by a rounding error; never command beyond the requested range.
    return np.asarray(np.minimum(positions, stop_um), dtype=np.float64)


def grid_axes(config: ScanConfig) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """X (columns) and Y (rows) grid coordinates of a scan, in micrometres."""
    xs = axis_positions(config.x_start_um, config.x_stop_um, config.xy_step_um)
    ys = axis_positions(config.y_start_um, config.y_stop_um, config.xy_step_um)
    return xs, ys


def order_points(xs: ArrayLike, ys: ArrayLike, order: ScanOrder) -> list[GridPoint]:
    """Grid points in acquisition order, row by row (increasing Y).

    SERPENTINE reverses the X direction on every odd row, so consecutive points
    are always grid neighbours: no long fly-back moves, and the previous point's
    surface height is a good estimate for the next one. RASTER scans every row
    in the same (+X) direction.
    """
    x_values = np.asarray(xs, dtype=np.float64)
    y_values = np.asarray(ys, dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.size == 0 or y_values.size == 0:
        raise ValueError("xs and ys must be non-empty 1-D arrays")
    x_list = [float(x) for x in x_values]
    forward = range(len(x_list))
    backward = range(len(x_list) - 1, -1, -1)
    points: list[GridPoint] = []
    for iy, y in enumerate(float(v) for v in y_values):
        reverse = order is ScanOrder.SERPENTINE and iy % 2 == 1
        for ix in backward if reverse else forward:
            points.append(GridPoint(len(points), ix, iy, x_list[ix], y))
    return points


def _check_sweep(width_um: float, step_um: float) -> None:
    if not (math.isfinite(width_um) and width_um > 0):
        raise ValueError(f"sweep width must be finite and > 0 (got {width_um})")
    if not (math.isfinite(step_um) and step_um > 0):
        raise ValueError(f"sweep step must be finite and > 0 (got {step_um})")


def _intervals(span_um: float, step_um: float) -> int:
    """Fewest equal intervals covering ``span`` that are no longer than ``step``."""
    return max(1, math.ceil(span_um / step_um - _EPS_UM))


def sweep_length(width_um: float, step_um: float) -> int:
    """Number of positions :func:`z_sweep` visits for an unclipped sweep."""
    _check_sweep(width_um, step_um)
    return _intervals(width_um, step_um) + 1


def z_sweep(
    center_um: float, width_um: float, step_um: float, limits: AxisLimits
) -> NDArray[np.float64]:
    """Strictly increasing Z positions covering ``center +/- width / 2``, clipped to ``limits``.

    The (clipped) interval ``[lo, hi]`` is divided into the fewest equal
    intervals that are not longer than ``step_um``. Both ends are therefore
    always visited, the spacing is uniform (the profile filters and fits assume
    it) and never coarser than requested. Sweeps always run upwards so stage
    backlash is taken up the same way at every point.

    Degenerate cases:

    * the interval lies entirely outside ``limits``: an empty array is returned
      (nothing can be measured safely; callers treat it as "no data");
    * the clipped interval collapses to a single point (the sweep only touches
      a limit): that one position is returned.

    Raises:
        ValueError: non-finite ``center_um`` or non-positive width / step.
    """
    if not math.isfinite(center_um):
        raise ValueError(f"sweep centre must be finite (got {center_um})")
    _check_sweep(width_um, step_um)
    lo = max(center_um - width_um / 2.0, limits.min_um)
    hi = min(center_um + width_um / 2.0, limits.max_um)
    span = hi - lo
    if span < -_EPS_UM:
        return np.empty(0, dtype=np.float64)
    if span <= _EPS_UM:
        return np.array([limits.clamp(lo)], dtype=np.float64)
    n = _intervals(span, step_um)
    z = lo + np.arange(n + 1, dtype=np.float64) * (span / n)
    z[-1] = hi  # exact end, free of rounding
    return z


def sweep_is_clipped(center_um: float, width_um: float, limits: AxisLimits) -> bool:
    """True when ``center +/- width / 2`` extends beyond ``limits`` (the sweep is shortened)."""
    return (
        center_um - width_um / 2.0 < limits.min_um - _EPS_UM
        or center_um + width_um / 2.0 > limits.max_um + _EPS_UM
    )


def _range_violation(axis: str, lo: float, hi: float, limits: AxisLimits) -> str | None:
    if limits.contains(lo) and limits.contains(hi):
        return None
    return (
        f"{axis} range [{lo:.3f}, {hi:.3f}] um outside travel limits "
        f"[{limits.min_um:.3f}, {limits.max_um:.3f}] um"
    )


@dataclass(frozen=True, slots=True, eq=False)
class ScanPlan:
    """The validated, ordered list of XY points plus everything needed to estimate a scan."""

    config: ScanConfig
    limits: StageLimits
    motion: MotionConfig
    data_rate_sps: int
    x_positions: NDArray[np.float64]
    y_positions: NDArray[np.float64]
    points: tuple[GridPoint, ...]

    @classmethod
    def from_config(
        cls,
        config: ScanConfig,
        limits: StageLimits,
        motion: MotionConfig,
        data_rate_sps: int,
    ) -> ScanPlan:
        """Build the grid and acquisition order.

        Limits are not checked here so that an estimate can still be shown for
        an out-of-range scan; call :meth:`validate_limits` before scanning.
        """
        if data_rate_sps <= 0:
            raise ValueError(f"data_rate_sps must be > 0 (got {data_rate_sps})")
        xs, ys = grid_axes(config)
        points = tuple(order_points(xs, ys, config.order))
        return cls(config, limits, motion, int(data_rate_sps), xs, ys, points)

    @property
    def n_x(self) -> int:
        return int(self.x_positions.size)

    @property
    def n_y(self) -> int:
        return int(self.y_positions.size)

    @property
    def total_points(self) -> int:
        return len(self.points)

    @property
    def is_confocal(self) -> bool:
        return self.config.mode is ScanMode.CONFOCAL

    def z_envelope(self) -> tuple[float, float]:
        """Lowest and highest Z the scan may command.

        Confocal: the full coarse range ``z_center +/- z_range / 2``; every
        sweep (adaptive, retry and fine) is clipped to it at run time, see
        :meth:`sweep_limits`. Fixed-Z: ``z_center`` only.
        """
        cfg = self.config
        if not self.is_confocal:
            return cfg.z_center_um, cfg.z_center_um
        half = cfg.z_range_um / 2.0
        return cfg.z_center_um - half, cfg.z_center_um + half

    def sweep_limits(self) -> AxisLimits:
        """Interval every confocal Z sweep is clipped to: the envelope within the travel limits.

        Clipping to the validated envelope (and not only to the stage limits)
        guarantees that a scan never commands a Z the operator did not approve,
        even when an adaptive estimate lies near the edge of the range.

        Raises:
            ValueError: for fixed-Z scans, which never sweep Z.
            LimitViolationError: the envelope does not overlap the Z limits.
        """
        if not self.is_confocal:
            raise ValueError("fixed-Z scans do not sweep Z")
        z_lo, z_hi = self.z_envelope()
        lo = max(z_lo, self.limits.z.min_um)
        hi = min(z_hi, self.limits.z.max_um)
        if not lo < hi:
            message = _range_violation("z", z_lo, z_hi, self.limits.z) or "empty Z range"
            raise LimitViolationError(message, violations=[message])
        return AxisLimits(min_um=lo, max_um=hi)

    def limit_violations(self) -> list[str]:
        """Every axis whose scan envelope leaves the travel limits (empty when safe)."""
        z_lo, z_hi = self.z_envelope()
        checks = (
            _range_violation(
                "x", float(self.x_positions[0]), float(self.x_positions[-1]), self.limits.x
            ),
            _range_violation(
                "y", float(self.y_positions[0]), float(self.y_positions[-1]), self.limits.y
            ),
            _range_violation("z", z_lo, z_hi, self.limits.z),
        )
        return [message for message in checks if message is not None]

    def validate_limits(self) -> None:
        """Raise :class:`LimitViolationError` unless the whole scan envelope is reachable."""
        violations = self.limit_violations()
        if violations:
            raise LimitViolationError(
                "scan envelope outside travel limits: " + "; ".join(violations),
                violations=violations,
            )

    # ------------------------------------------------------------------ estimate
    def _positions_per_point(self) -> tuple[int, int]:
        """Z positions of the first point and of a typical later point.

        The first point has no adaptive estimate and sweeps the full coarse
        range; with ``adaptive_z`` later points sweep ``adaptive_z_range_um``.
        Coarse retries (missed peaks) are not predictable and not included.
        """
        cfg = self.config
        if not self.is_confocal:
            return 1, 1
        fine = sweep_length(cfg.fine_z_range_um, cfg.fine_z_step_um) if cfg.fine_scan else 0
        first = sweep_length(cfg.z_range_um, cfg.coarse_z_step_um) + fine
        if not cfg.adaptive_z:
            return first, first
        return first, sweep_length(cfg.adaptive_z_range_um, cfg.coarse_z_step_um) + fine

    def _z_travel_um(self) -> float:
        """Total Z travel, assuming the surface lies near each sweep centre.

        Per point: from the previous fine-sweep end down to the coarse start
        (about half the coarse plus half the fine width), up through the coarse
        sweep, back down to the fine start (again about half of each) and up
        through the fine sweep, i.e. 2 x (coarse width + fine width).
        """
        cfg = self.config
        if not self.is_confocal:
            return 0.0
        typical = cfg.adaptive_z_range_um if cfg.adaptive_z else cfg.z_range_um
        fine = cfg.fine_z_range_um if cfg.fine_scan else 0.0
        per_point_first = 2.0 * (cfg.z_range_um + fine)
        per_point = 2.0 * (typical + fine)
        return per_point_first + (self.total_points - 1) * per_point

    def _xy_travel_time_s(self) -> float:
        """Straight-line XY moves between consecutive points at ``xy_speed_um_s``."""
        if self.total_points < 2:
            return 0.0
        x = np.fromiter((p.x_um for p in self.points), dtype=np.float64, count=self.total_points)
        y = np.fromiter((p.y_um for p in self.points), dtype=np.float64, count=self.total_points)
        distance = np.hypot(np.diff(x), np.diff(y))
        return float(np.sum(distance)) / self.motion.xy_speed_um_s

    def estimate(self) -> ScanEstimate:
        """Duration, data volume and sanity warnings for this plan (never raises on limits).

        Duration = XY travel + Z travel at the configured speeds + one command
        overhead per move + (settle time + ``samples_per_z / data_rate``) per Z
        position. The move to the first point and coarse retries are unknown
        in advance and not included.
        """
        cfg = self.config
        motion = self.motion
        n = self.total_points
        first, typical = self._positions_per_point()
        measurements = first + (n - 1) * typical
        # Fixed-Z: one combined XYZ move per point. Confocal: one XY move per point
        # plus one Z move per position.
        moves = n if not self.is_confocal else n + measurements
        dwell_s = cfg.settle_time_ms / 1000.0 + cfg.samples_per_z / self.data_rate_sps
        duration_s = (
            self._xy_travel_time_s()
            + self._z_travel_um() / motion.z_speed_um_s
            + moves * motion.per_move_overhead_s
            + measurements * dwell_s
        )
        data_bytes = (
            measurements * (cfg.samples_per_z * _BYTES_PER_SAMPLE + _BYTES_PER_POSITION)
            + n * _BYTES_PER_POINT
        )
        violations = self.limit_violations()
        return ScanEstimate(
            n_x=self.n_x,
            n_y=self.n_y,
            total_points=n,
            z_positions_per_point=typical,
            total_measurements=measurements,
            total_adc_samples=measurements * cfg.samples_per_z,
            estimated_duration_s=duration_s,
            estimated_data_bytes=data_bytes,
            within_limits=not violations,
            limit_violations=violations,
            warnings=self._warnings(duration_s, data_bytes),
        )

    def _warnings(self, duration_s: float, data_bytes: int) -> list[str]:
        cfg = self.config
        warnings: list[str] = []
        if duration_s > LONG_SCAN_WARNING_S:
            warnings.append(
                f"very long scan: about {duration_s / 3600.0:.1f} h; consider a coarser XY "
                "step, a smaller area or narrower Z ranges"
            )
        if data_bytes > LARGE_DATA_WARNING_BYTES:
            warnings.append(f"large data volume: about {data_bytes / 1024**3:.1f} GiB of raw data")
        if cfg.xy_step_um < MOTOR_RESOLUTION_UM:
            warnings.append(
                f"xy_step_um {cfg.xy_step_um:g} um is below the ~{MOTOR_RESOLUTION_UM} um "
                "motor resolution; neighbouring points will coincide"
            )
        if self.is_confocal:
            if cfg.fine_scan and cfg.fine_z_step_um < MOTOR_RESOLUTION_UM:
                warnings.append(
                    f"fine_z_step_um {cfg.fine_z_step_um:g} um is below the "
                    f"~{MOTOR_RESOLUTION_UM} um motor resolution; Z positions will repeat"
                )
            fwhm = cfg.processing.expected_fwhm_um
            if fwhm is not None and cfg.coarse_z_step_um > fwhm / 2.0:
                warnings.append(
                    f"coarse_z_step_um {cfg.coarse_z_step_um:g} um exceeds half the expected "
                    f"axial FWHM ({fwhm:g} um); the coarse sweep may step over the peak"
                )
        return warnings
