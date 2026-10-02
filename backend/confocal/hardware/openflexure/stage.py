"""``Stage`` implementation on top of an OpenFlexure server.

Why this exists
---------------
The OpenFlexure software already drives Delta Stages (Sangaboard firmware,
delta transform, camera). With this adapter the confocal application can use
an OpenFlexure server as its stage backend instead of the Arduino, without any
change above the hardware layer: scanning, processing and the API only see
:class:`~confocal.hardware.base.Stage` in micrometres. It is also the first
step towards packaging the scanner as an *OpenFlexure extension*: an extension
runs inside the server process and would provide an in-process
:class:`~confocal.hardware.openflexure.client.OpenFlexureClient` that calls the
server's stage object directly; everything else (this adapter, the
``MicroscopeController``, the scan engine) stays as it is. A future
``OpenFlexureMicroscopeController`` can additionally reuse the server's camera.

Coordinates
-----------
The server reports and accepts integer x/y/z steps; the delta transform is
the server's business. :class:`StepConverter` maps them to micrometres with
one scale per axis (``um_per_step``); the origin is the server's step origin.

Safety, as for every stage backend (``docs/architecture.md`` section 2):

* targets are checked against :attr:`limits` before anything is sent;
* after every move the position is *read back from the server* and compared
  with the quantised target; a per-axis error above ``position_tolerance_um``
  raises :class:`MotionVerificationError` (a move is never assumed done);
* :meth:`stop` may be called from another task during a move: it forwards
  the stop to the server and the move raises :class:`MotionAbortedError`. A
  move registers itself as in flight before its first ``await``, so a stop
  that arrives while it is still preparing (e.g. :meth:`home` reading the
  position) aborts it before anything is sent to the server;
* a move that does not finish within ``MotionConfig.move_timeout_for`` (the
  expected duration from distance and speed, times a safety factor, plus the
  fixed ``move_timeout_s`` margin) is stopped and raises
  :class:`MotionTimeoutError`;
* any failure of the client that is not already a :class:`HardwareError`
  becomes a :class:`CommunicationError`.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TypeVar

from confocal.config import MotionConfig
from confocal.errors import (
    CommunicationError,
    ConfocalError,
    HardwareNotConnectedError,
    MotionAbortedError,
    MotionError,
    MotionTimeoutError,
    MotionVerificationError,
)
from confocal.hardware.base import Stage
from confocal.hardware.openflexure.client import OpenFlexureClient
from confocal.models.common import Axis, Position, StageLimits
from confocal.models.hardware import StageState, StageStatus

_T = TypeVar("_T")

_AXES: tuple[Axis, Axis, Axis] = (Axis.X, Axis.Y, Axis.Z)


class StepConverter:
    """Micrometres <-> OpenFlexure server step coordinates, one scale per axis."""

    def __init__(self, um_per_step: Mapping[Axis, float] | Sequence[float]) -> None:
        if isinstance(um_per_step, Mapping):
            missing = [axis.value for axis in _AXES if axis not in um_per_step]
            if missing:
                raise ValueError(f"um_per_step missing for axes: {', '.join(missing)}")
            scales = tuple(float(um_per_step[axis]) for axis in _AXES)
        else:
            if len(um_per_step) != 3:
                raise ValueError("um_per_step needs three values (x, y, z)")
            scales = tuple(float(value) for value in um_per_step)
        if any(not (scale > 0 and math.isfinite(scale)) for scale in scales):
            raise ValueError("um_per_step values must be positive and finite")
        self._scales: dict[Axis, float] = dict(zip(_AXES, scales, strict=True))

    @property
    def um_per_step(self) -> dict[Axis, float]:
        return dict(self._scales)

    def to_steps(self, position: Position) -> dict[Axis, int]:
        """Nearest server steps (round half away from zero) for every axis."""
        return {axis: _round_half_away(position.get(axis) / self._scales[axis]) for axis in _AXES}

    def to_position(self, steps: Mapping[Axis, int]) -> Position:
        """Position of the given server step coordinates.

        Raises:
            CommunicationError: if the server did not report all three axes.
        """
        missing = [axis.value for axis in _AXES if axis not in steps]
        if missing:
            raise CommunicationError(f"OpenFlexure server did not report axes {missing}")
        return Position(
            x_um=int(steps[Axis.X]) * self._scales[Axis.X],
            y_um=int(steps[Axis.Y]) * self._scales[Axis.Y],
            z_um=int(steps[Axis.Z]) * self._scales[Axis.Z],
        )

    def quantize(self, position: Position) -> Position:
        """The position the server can actually reach for ``position``."""
        return self.to_position(self.to_steps(position))

    def resolution_um(self) -> float:
        """Largest per-axis rounding error (half the coarsest step)."""
        return 0.5 * max(self._scales.values())


def _round_half_away(value: float) -> int:
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _format(position: Position) -> str:
    return f"({position.x_um:.3f}, {position.y_um:.3f}, {position.z_um:.3f}) um"


class OpenFlexureStage(Stage):
    """A :class:`Stage` in micrometres driven through an :class:`OpenFlexureClient`."""

    backend_name = "openflexure"

    def __init__(
        self,
        client: OpenFlexureClient,
        limits: StageLimits,
        *,
        converter: StepConverter,
        motion: MotionConfig,
    ) -> None:
        self._client = client
        self._limits = limits
        self._converter = converter
        self._tolerance_um = motion.position_tolerance_um
        self._motion = motion
        self._connected = False
        self._version: str | None = None
        self._state = StageState.DISCONNECTED
        self._position: Position | None = None
        self._moving = False
        self._abort = asyncio.Event()
        self._homed = False
        self._last_error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> None:
        if self._connected:
            return
        self._version = await self._call(self._client.server_version())
        self._position = await self._read_position()
        self._connected = True
        self._state = StageState.IDLE

    async def close(self) -> None:
        if self._moving:
            await self.stop()
        self._connected = False
        self._state = StageState.DISCONNECTED

    @property
    def connected(self) -> bool:
        return self._connected

    def version(self) -> str | None:
        return self._version

    @property
    def converter(self) -> StepConverter:
        return self._converter

    # ------------------------------------------------------------------ state
    @property
    def limits(self) -> StageLimits:
        return self._limits

    async def get_position(self) -> Position:
        self._require_connected()
        self._position = await self._read_position()
        return self._position

    async def status(self) -> StageStatus:
        return StageStatus(
            backend=self.backend_name,
            connected=self._connected,
            state=self._state,
            position=self._position,
            homed=self._homed,
            limits=self._limits,
            firmware_version=self._version,
            last_error=self._last_error,
        )

    # ------------------------------------------------------------------ motion
    async def move_to(self, target: Position) -> Position:
        self._require_connected()
        self.check_limits(target)

        async def fixed_target() -> Position:
            return target

        return await self._travel(fixed_target, StageState.MOVING)

    async def home(self, axes: Sequence[Axis] | None = None) -> Position:
        """Return the selected axes to the server's step origin (there are no end-stops)."""
        self._require_connected()
        selected = set(Axis) if axes is None else set(axes)

        async def origin_target() -> Position:
            target = await self.get_position()
            for axis in selected:
                target = target.with_axis(axis, 0.0)
            self.check_limits(target)
            return target

        position = await self._travel(origin_target, StageState.HOMING)
        if selected == set(Axis):
            self._homed = True
        return position

    async def stop(self) -> None:
        """Forward a stop to the server; a running :meth:`move_to` then raises."""
        if self._moving:
            self._abort.set()
        await self._call(self._client.stop())

    # ------------------------------------------------------------------ internals
    def _require_connected(self) -> None:
        if not self._connected:
            raise HardwareNotConnectedError("OpenFlexure stage is not connected")

    async def _call(self, awaitable: Awaitable[_T]) -> _T:
        """Await a client call, turning unexpected failures into CommunicationError."""
        try:
            return await awaitable
        except ConfocalError:
            raise
        except Exception as exc:  # device boundary: anything else is a link failure
            self._last_error = f"OpenFlexure server request failed: {exc!r}"
            raise CommunicationError(self._last_error) from exc

    async def _read_position(self) -> Position:
        steps = await self._call(self._client.get_position_steps())
        return self._converter.to_position(steps)

    async def _travel(
        self, target_of: Callable[[], Awaitable[Position]], state: StageState
    ) -> Position:
        """Run one move. ``target_of`` computes the target once the move is in flight."""
        if self._moving:
            raise MotionError("a move is already in progress")
        # In flight from here on, before the first await: a stop() is never lost.
        self._moving = True
        self._abort.clear()
        try:
            target = await target_of()  # may raise (e.g. limits): nothing was sent
            if self._abort.is_set():
                self._state = StageState.STOPPED
                raise MotionAbortedError(
                    f"move to {_format(target)} stopped before it was sent to the server"
                )
            return await self._execute(target, state)
        finally:
            self._moving = False

    async def _execute(self, target: Position, state: StageState) -> Position:
        """Send the move, wait for it (time-bounded) and verify the read-back position."""
        self._state = state
        try:
            expected = self._converter.quantize(target)
            start = self._position if self._position is not None else target
            timeout_s = self._motion.move_timeout_for(start, target)
            try:
                async with asyncio.timeout(timeout_s):
                    await self._call(
                        self._client.move_steps(self._converter.to_steps(target), absolute=True)
                    )
            except TimeoutError:
                await self._call(self._client.stop())
                self._position = await self._read_position()
                raise MotionTimeoutError(
                    f"move to {_format(target)} did not finish within {timeout_s:.3g} s; "
                    f"stopped at {_format(self._position)}"
                ) from None
            reported = await self._read_position()
            self._position = reported
            if self._abort.is_set():
                self._state = StageState.STOPPED
                raise MotionAbortedError(
                    f"move to {_format(target)} stopped at {_format(reported)}"
                )
            error = reported.max_axis_error(expected)
            if error > self._tolerance_um:
                raise MotionVerificationError(
                    f"OpenFlexure stage reports {_format(reported)} after a move to "
                    f"{_format(expected)} (error {error:.3f} um > {self._tolerance_um:g} um)"
                )
            self._state = StageState.IDLE
            self._last_error = None
            return reported
        except MotionAbortedError:
            raise
        except ConfocalError as exc:
            self._state = StageState.ERROR
            self._last_error = str(exc)
            raise
