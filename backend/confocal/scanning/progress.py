"""Live progress of the running scan: counters, current position, ETA and event throttling."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from confocal.models.common import Position
from confocal.models.scan import ScanProgress, ScanState
from confocal.scanning.plan import GridPoint

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class LiveEventConfig:
    """Rate of the periodic (droppable) live events and the ETA averaging window."""

    progress_interval_s: float = 0.25
    profile_interval_s: float = 0.5
    eta_window_points: int = 20

    def __post_init__(self) -> None:
        if self.progress_interval_s < 0 or self.profile_interval_s < 0:
            raise ValueError("event intervals must be >= 0")
        if self.eta_window_points < 1:
            raise ValueError("eta_window_points must be >= 1")


class EventThrottle:
    """Lets an event through at most once per ``interval_s`` (0 = always)."""

    def __init__(self, interval_s: float, clock: Clock = time.monotonic) -> None:
        self._interval = interval_s
        self._clock = clock
        self._last = -math.inf

    def ready(self) -> bool:
        now = self._clock()
        if now - self._last < self._interval:
            return False
        self._last = now
        return True


class ProgressTracker:
    """Mutable live state of one scan, turned into :class:`ScanProgress` snapshots.

    The ETA is the mean duration of the last ``eta_window_points`` points times
    the number of points left: a moving average follows changes of the
    per-point time (adaptive Z ranges, retries, surface regions without signal)
    much better than the overall average.
    """

    def __init__(
        self,
        scan_id: str,
        total_points: int,
        *,
        eta_window_points: int = 20,
        clock: Clock = time.monotonic,
    ) -> None:
        self.scan_id = scan_id
        self.total_points = total_points
        self.state = ScanState.IDLE
        self.completed_points = 0
        self._clock = clock
        self._started: float | None = None
        self._durations: deque[float] = deque(maxlen=eta_window_points)
        self._point: GridPoint | None = None
        self._position: Position | None = None
        self._intensity: float | None = None

    @property
    def progress(self) -> float:
        if self.total_points <= 0:
            return 0.0
        return min(1.0, self.completed_points / self.total_points)

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._started is None else max(0.0, self._clock() - self._started)

    def start(self) -> None:
        """Start the elapsed-time clock (when the scan leaves IDLE)."""
        self._started = self._clock()

    def begin_point(self, point: GridPoint) -> None:
        self._point = point
        self._intensity = None

    def update(self, position: Position, intensity: float | None) -> None:
        """Latest stage-reported position and aggregated intensity."""
        self._position = position
        self._intensity = intensity if intensity is not None and math.isfinite(intensity) else None

    def finish_point(self, duration_s: float) -> None:
        """Count a measured point and feed its duration into the ETA average."""
        self.completed_points += 1
        if math.isfinite(duration_s) and duration_s >= 0:
            self._durations.append(duration_s)

    def eta_s(self) -> float | None:
        if not self._durations:
            return None
        remaining = max(0, self.total_points - self.completed_points)
        return remaining * (sum(self._durations) / len(self._durations))

    def snapshot(self, message: str | None = None) -> ScanProgress:
        point = self._point
        position = self._position
        return ScanProgress(
            scan_id=self.scan_id,
            state=self.state,
            progress=self.progress,
            completed_points=self.completed_points,
            total_points=self.total_points,
            current_point_id=None if point is None else point.point_id,
            current_x_um=None if position is None else position.x_um,
            current_y_um=None if position is None else position.y_um,
            current_z_um=None if position is None else position.z_um,
            current_intensity=self._intensity,
            elapsed_s=self.elapsed_s,
            estimated_remaining_s=self.eta_s(),
            message=message,
        )
