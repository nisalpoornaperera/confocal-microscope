"""Simulated three-axis stage with realistic timing, interruption and quantisation.

The simulation behaves like the real Arduino-driven Delta Stage in every way
the upper layers can observe:

* a move takes ``max(|dx| / v_xy, |dy| / v_xy, |dz| / v_z) * time_scale``
  seconds (``MotionConfig`` speeds) and always yields to the event loop, even
  at ``time_scale = 0``, so concurrent tasks (HTTP, e-stop) keep running;
* :meth:`stop` from another task interrupts an in-flight move: that move raises
  :class:`MotionAbortedError` and the stage stays where it was, on the straight
  line between start and target (the firmware interpolates the three motors).
  A move registers itself as in flight synchronously, before its first
  ``await``, so a stop that arrives before the move's first sleep aborts it
  at its start;
* with ``kinematics`` every position is quantised to the motor step grid,
  which on the delta stage is not aligned with X/Y/Z: a commanded position is
  reached only to within ``kinematics.max_quantization_error_um()``;
* travel limits are enforced (defence in depth below the controller);
* fault injection: with ``fault_after_moves = N`` the first ``N`` moves
  succeed and move ``N + 1`` raises :class:`MotionError` (once).

There are no end-stops: :meth:`home` drives back to the power-up origin.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from confocal.config import MotionConfig
from confocal.errors import HardwareNotConnectedError, MotionAbortedError, MotionError
from confocal.hardware.base import Stage
from confocal.hardware.kinematics import StageKinematics
from confocal.models.common import Axis, Position, StageLimits
from confocal.models.hardware import StageState, StageStatus

_ORIGIN = Position(x_um=0.0, y_um=0.0, z_um=0.0)


def _format(position: Position) -> str:
    return f"({position.x_um:.3f}, {position.y_um:.3f}, {position.z_um:.3f}) um"


@dataclass(slots=True)
class _Segment:
    """An in-flight straight-line move."""

    start: Position
    end: Position
    started_at: float  # time.monotonic()
    duration_s: float
    abort: asyncio.Event = field(default_factory=asyncio.Event)

    def fraction_done(self) -> float:
        if self.duration_s <= 0.0:
            return 0.0
        return min(1.0, max(0.0, (time.monotonic() - self.started_at) / self.duration_s))

    def interpolate(self, fraction: float) -> Position:
        return Position(
            x_um=self.start.x_um + fraction * (self.end.x_um - self.start.x_um),
            y_um=self.start.y_um + fraction * (self.end.y_um - self.start.y_um),
            z_um=self.start.z_um + fraction * (self.end.z_um - self.start.z_um),
        )


class SimulationStage(Stage):
    """In-memory stage; see the module docstring for its behaviour."""

    backend_name = "simulation"

    def __init__(
        self,
        limits: StageLimits,
        *,
        motion: MotionConfig,
        time_scale: float,
        kinematics: StageKinematics | None = None,
        fault_after_moves: int | None = None,
    ) -> None:
        if not (time_scale >= 0.0 and math.isfinite(time_scale)):
            raise ValueError("time_scale must be finite and >= 0")
        if fault_after_moves is not None and fault_after_moves < 0:
            raise ValueError("fault_after_moves must be >= 0")
        self._limits = limits
        self._motion = motion
        self._time_scale = float(time_scale)
        self._kinematics = kinematics
        self._fault_after_moves = fault_after_moves
        self._connected = False
        self._state = StageState.DISCONNECTED
        self._position = self._quantize(_ORIGIN)
        self._segment: _Segment | None = None
        self._moves_started = 0
        self._homed = False
        self._released = True
        self._last_error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> None:
        if not self._connected:
            self._connected = True
            self._state = StageState.IDLE
        await asyncio.sleep(0)

    async def close(self) -> None:
        await self.stop()
        self._connected = False
        self._state = StageState.DISCONNECTED

    @property
    def connected(self) -> bool:
        return self._connected

    def version(self) -> str | None:
        return "simulation"

    # ------------------------------------------------------------------ state
    @property
    def limits(self) -> StageLimits:
        return self._limits

    @property
    def kinematics(self) -> StageKinematics | None:
        return self._kinematics

    @property
    def current_position(self) -> Position:
        """Where the platform is right now (interpolated during a move). No I/O.

        Used by the simulated ADC and camera, which must see the true position.
        """
        segment = self._segment
        if segment is None:
            return self._position
        return self._quantize(segment.interpolate(segment.fraction_done()))

    @property
    def moving(self) -> bool:
        return self._segment is not None

    @property
    def motors_released(self) -> bool:
        return self._released

    @property
    def moves_started(self) -> int:
        return self._moves_started

    async def get_position(self) -> Position:
        self._require_connected()
        await asyncio.sleep(0)
        return self.current_position

    async def status(self) -> StageStatus:
        return StageStatus(
            backend=self.backend_name,
            connected=self._connected,
            state=self._state,
            position=self.current_position,
            homed=self._homed,
            limits=self._limits,
            firmware_version=self.version(),
            last_error=self._last_error,
        )

    # ------------------------------------------------------------------ motion
    async def move_to(self, target: Position) -> Position:
        self._require_connected()
        self.check_limits(target)
        return await self._travel(target, StageState.MOVING)

    async def home(self, axes: Sequence[Axis] | None = None) -> Position:
        self._require_connected()
        selected = set(Axis) if axes is None else set(axes)
        target = self._position
        for axis in selected:
            target = target.with_axis(axis, 0.0)
        self.check_limits(target)
        position = await self._travel(target, StageState.HOMING)
        if selected == set(Axis):
            self._homed = True
        return position

    async def stop(self) -> None:
        segment = self._segment
        if segment is not None:
            segment.abort.set()
        await asyncio.sleep(0)

    async def release(self) -> None:
        self._require_connected()
        if self._segment is not None:
            raise MotionError("cannot release the motors during a move")
        self._released = True
        await asyncio.sleep(0)

    # ------------------------------------------------------------------ internals
    def _require_connected(self) -> None:
        if not self._connected:
            raise HardwareNotConnectedError("simulation stage is not connected")

    def _quantize(self, position: Position) -> Position:
        if self._kinematics is None:
            return position
        return self._kinematics.quantize(position)

    def _duration_s(self, start: Position, end: Position) -> float:
        return self._motion.expected_move_s(start, end) * self._time_scale

    async def _travel(self, target: Position, state: StageState) -> Position:
        if self._segment is not None:
            raise MotionError("a move is already in progress")
        self._moves_started += 1
        if (
            self._fault_after_moves is not None
            and self._moves_started == self._fault_after_moves + 1
        ):
            self._state = StageState.ERROR
            self._last_error = f"injected motion fault on move {self._moves_started}"
            raise MotionError(self._last_error)

        start = self._position
        end = self._quantize(target)
        segment = _Segment(
            start=start,
            end=end,
            started_at=time.monotonic(),
            duration_s=self._duration_s(start, end),
        )
        self._segment = segment
        self._state = state
        self._released = False
        try:
            if segment.duration_s > 0.0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(segment.abort.wait(), timeout=segment.duration_s)
            else:
                await asyncio.sleep(0)
            if segment.abort.is_set():
                self._position = self._quantize(segment.interpolate(segment.fraction_done()))
                self._state = StageState.STOPPED
                raise MotionAbortedError(
                    f"move to {_format(target)} stopped at {_format(self._position)}"
                )
            self._position = end
            self._state = StageState.IDLE
            return end
        except asyncio.CancelledError:
            # The caller gave up (e.g. a controller timeout): freeze where we are.
            self._position = self._quantize(segment.interpolate(segment.fraction_done()))
            self._state = StageState.STOPPED
            raise
        finally:
            self._segment = None
