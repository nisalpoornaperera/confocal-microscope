"""In-memory reference implementation of the Arduino Uno motor firmware.

``FirmwareEmulator`` is the *executable specification* of the Uno firmware
(``firmware/README.md``, ``docs/serial-protocol.md``): the C++ sketch must
produce exactly the replies and events this class produces for the same input
lines (``tests/hil`` compares them byte for byte on a flashed board), and
``ArduinoStage`` is tested against it (through ``EmulatorTransport``) before it
ever drives real motors. It works in firmware motor steps only - it knows nothing
about micrometres or the delta geometry.

Coordinated moves
-----------------
On the Delta Stage every X/Y move drives at least two legs in opposite
directions, so the three motors must start and finish together or the
platform wanders sideways. A move of ``d = (da, db, dc)`` steps is executed as
``N = max(|da|, |db|, |dc|)`` *ticks* of a multi-axis Bresenham interpolation.
At each tick every motor ``i`` adds ``|d_i|`` to an error accumulator that
starts at ``N / 2``; when the accumulator reaches ``N`` the motor takes one
step towards its target and ``N`` is subtracted. After ``k`` ticks motor ``i``
has therefore taken ``floor((k |d_i| + N // 2) / N)`` steps (the closed form
used here): never more than one step per tick, all motors arrive at tick
``N``, and every intermediate position is within half a step of the straight
line from start to target.

Tick rate: the move lasts ``T = max_i |d_i| / max_speed_steps_s_i``, so the
motor that has the furthest to go relative to its own speed limit runs at its
maximum rate and the others proportionally slower (with equal speed limits:
the motor with the most steps runs at full speed).

Stepping is non-blocking: the firmware's main loop polls the serial port
between steps, so ``STOP`` pre-empts a running move within one step period.
Here time only advances through :meth:`FirmwareEmulator.advance`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from confocal.config import ArduinoConfig, MotorConfig
from confocal.hardware.arduino.protocol import (
    STEP_LIMIT,
    Command,
    ErrorCode,
    Event,
    FirmwareState,
    FirmwareStatus,
    MalformedMessageError,
    Request,
    Response,
    parse_request,
)
from confocal.hardware.kinematics import MOTORS, Motor, MotorSteps

#: Version reported in the BOOT banner and STATUS reply.
FIRMWARE_VERSION = "1.0.0"

#: Fixed, short error texts (the firmware never echoes received bytes).
_ERROR_TEXT: dict[ErrorCode, str] = {
    ErrorCode.E_CRC: "bad checksum",
    ErrorCode.E_SYNTAX: "malformed line",
    ErrorCode.E_LENGTH: "line too long",
    ErrorCode.E_UNKNOWN: "unknown command",
    ErrorCode.E_ARGS: "bad arguments",
    ErrorCode.E_BUSY: "moving",
    ErrorCode.E_RANGE: "outside travel",
}

#: Tolerance (ticks) for floating-point time accumulation in :meth:`advance`.
_TICK_EPSILON = 1e-6


@dataclass(slots=True)
class _Move:
    """A coordinated move in progress."""

    seq: int
    start: MotorSteps
    delta: MotorSteps
    ticks_total: int
    ticks_per_s: float
    elapsed_s: float = 0.0

    def ticks_done(self) -> int:
        return min(self.ticks_total, math.floor(self.elapsed_s * self.ticks_per_s + _TICK_EPSILON))

    def position_after(self, ticks: int) -> MotorSteps:
        half = self.ticks_total // 2
        a, b, c = (
            s + (1 if d > 0 else -1) * ((ticks * abs(d) + half) // self.ticks_total)
            for s, d in zip(self.start, self.delta, strict=True)
        )
        return (a, b, c)


class FirmwareEmulator:
    """Deterministic model of the Uno firmware: bytes in, bytes out.

    Call :meth:`reset` to emulate power-up / the DTR auto-reset that opening
    the serial port causes (returns the BOOT banner), :meth:`handle_line` for
    every received line (returns the reply and any immediate events) and
    :meth:`advance` to let time pass (returns DONE events of finished moves).
    """

    def __init__(
        self, motors: Mapping[Motor, MotorConfig], *, version: str = FIRMWARE_VERSION
    ) -> None:
        missing = [motor.value for motor in MOTORS if motor not in motors]
        if missing:
            raise ValueError(f"motor configuration missing for motors: {', '.join(missing)}")
        self._motors: dict[Motor, MotorConfig] = {motor: motors[motor] for motor in MOTORS}
        for motor, cfg in self._motors.items():
            if max(abs(cfg.min_steps), abs(cfg.max_steps)) > STEP_LIMIT:
                raise ValueError(
                    f"travel of motor {motor.value} exceeds the protocol range +/-{STEP_LIMIT}"
                )
        Event.boot(version)  # validates the version string
        self._version = version
        self._handlers: dict[Command, Callable[[Request], list[bytes]]] = {
            Command.MOVE: self._on_move,
            Command.GETPOS: self._on_getpos,
            Command.HOME: self._on_home,
            Command.STOP: self._on_stop,
            Command.STATUS: self._on_status,
            Command.PING: self._on_ping,
            Command.ZERO: self._on_zero,
            Command.RELEASE: self._on_release,
        }
        self._position: MotorSteps = (0, 0, 0)
        self._state = FirmwareState.IDLE
        self._coils_enabled = False
        self._move: _Move | None = None
        self._time_s = 0.0

    @classmethod
    def from_config(
        cls, arduino: ArduinoConfig, *, version: str = FIRMWARE_VERSION
    ) -> FirmwareEmulator:
        return cls({Motor.A: arduino.a, Motor.B: arduino.b, Motor.C: arduino.c}, version=version)

    # ------------------------------------------------------------------ observation
    @property
    def position(self) -> MotorSteps:
        return self._position

    @property
    def state(self) -> FirmwareState:
        return self._state

    @property
    def moving(self) -> bool:
        return self._move is not None

    @property
    def coils_enabled(self) -> bool:
        return self._coils_enabled

    @property
    def version(self) -> str:
        return self._version

    @property
    def time_s(self) -> float:
        """Emulated time since construction (advanced only by :meth:`advance`)."""
        return self._time_s

    def move_duration_s(self, target: MotorSteps) -> float:
        """How long a coordinated move from the current position to ``target`` takes."""
        return max(
            abs(goal - now) / self._motors[motor].max_speed_steps_s
            for motor, now, goal in zip(MOTORS, self._position, target, strict=True)
        )

    # ------------------------------------------------------------------ inputs
    def reset(self) -> list[bytes]:
        """Power-up / reset: counters 0 (the new origin), coils off, BOOT banner."""
        self._position = (0, 0, 0)
        self._state = FirmwareState.IDLE
        self._coils_enabled = False
        self._move = None
        return [Event.boot(self._version).encode()]

    def handle_line(self, line: bytes) -> list[bytes]:
        """Process one received line; returns the reply followed by any immediate events."""
        try:
            request = parse_request(line)
        except MalformedMessageError as exc:
            return [Response.failure(exc.seq, exc.code, _ERROR_TEXT[exc.code]).encode()]
        return self._handlers[request.command](request)

    def advance(self, dt_s: float) -> list[bytes]:
        """Let ``dt_s`` seconds pass; returns ``! DONE`` if the running move finished."""
        if not (dt_s >= 0.0 and math.isfinite(dt_s)):
            raise ValueError("dt_s must be finite and >= 0")
        self._time_s += dt_s
        move = self._move
        if move is None:
            return []
        move.elapsed_s += dt_s
        ticks = move.ticks_done()
        self._position = move.position_after(ticks)
        if ticks < move.ticks_total:
            return []
        self._move = None
        self._state = FirmwareState.IDLE
        return [Event.done(move.seq, self._position).encode()]

    # ------------------------------------------------------------------ commands
    @staticmethod
    def _error(seq: int, code: ErrorCode) -> list[bytes]:
        return [Response.failure(seq, code, _ERROR_TEXT[code]).encode()]

    def _start_move(
        self, request: Request, target: MotorSteps, state: FirmwareState
    ) -> list[bytes]:
        if self._move is not None:
            return self._error(request.seq, ErrorCode.E_BUSY)
        for motor, goal in zip(MOTORS, target, strict=True):
            cfg = self._motors[motor]
            if not cfg.min_steps <= goal <= cfg.max_steps:
                return self._error(request.seq, ErrorCode.E_RANGE)
        self._coils_enabled = True
        delta = _as_steps([goal - now for goal, now in zip(target, self._position, strict=True)])
        ticks_total = max(abs(d) for d in delta)
        ack = Response.success(request.seq).encode()
        if ticks_total == 0:
            self._state = FirmwareState.IDLE
            return [ack, Event.done(request.seq, self._position).encode()]
        self._move = _Move(
            seq=request.seq,
            start=self._position,
            delta=delta,
            ticks_total=ticks_total,
            ticks_per_s=ticks_total / self.move_duration_s(target),
        )
        self._state = state
        return [ack]

    def _on_move(self, request: Request) -> list[bytes]:
        return self._start_move(request, _as_steps(request.args), FirmwareState.MOVING)

    def _on_home(self, request: Request) -> list[bytes]:
        return self._start_move(request, (0, 0, 0), FirmwareState.HOMING)

    def _on_stop(self, request: Request) -> list[bytes]:
        lines: list[bytes] = []
        move = self._move
        if move is not None:
            self._move = None
            self._state = FirmwareState.STOPPED
            lines.append(Event.aborted(move.seq, self._position).encode())
        lines.append(Response.success(request.seq, *self._position).encode())
        return lines

    def _on_getpos(self, request: Request) -> list[bytes]:
        return [Response.success(request.seq, *self._position).encode()]

    def _on_status(self, request: Request) -> list[bytes]:
        status = FirmwareStatus(
            state=self._state,
            steps=self._position,
            moving=self._move is not None,
            coils_enabled=self._coils_enabled,
            version=self._version,
        )
        return [Response.success(request.seq, *status.payload()).encode()]

    def _on_ping(self, request: Request) -> list[bytes]:
        return [Response.success(request.seq).encode()]

    def _on_zero(self, request: Request) -> list[bytes]:
        if self._move is not None:
            return self._error(request.seq, ErrorCode.E_BUSY)
        self._position = (0, 0, 0)
        return [Response.success(request.seq, *self._position).encode()]

    def _on_release(self, request: Request) -> list[bytes]:
        if self._move is not None:
            return self._error(request.seq, ErrorCode.E_BUSY)
        self._coils_enabled = False
        return [Response.success(request.seq).encode()]


def _as_steps(values: list[int] | tuple[int, ...]) -> MotorSteps:
    a, b, c = values
    return (a, b, c)
