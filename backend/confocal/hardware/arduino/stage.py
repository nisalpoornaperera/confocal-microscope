"""``ArduinoStage``: the Delta Stage driven by the Arduino Uno firmware over USB serial.

Thin layer over :mod:`.steps` (micrometres <-> firmware steps, backlash
planning), :mod:`.protocol` (CRC-framed lines) and a :mod:`.transport`.
Behaviour (``docs/serial-protocol.md``, ``docs/architecture.md`` section 2):

Connection
    :meth:`ArduinoStage.connect` opens the transport - which resets the Uno -
    waits up to ``handshake_timeout_s`` for the ``! BOOT`` banner, then sends
    ``PING``, ``STATUS`` and ``GETPOS``. The protocol has no command that sets
    travel limits or speeds: the firmware's motor travel and speeds are
    compiled in and must match ``[arduino.a|b|c]`` (the firmware then refuses
    an out-of-travel target with ``E_RANGE`` as the last line of defence).
    After a reset the step counters are 0, so the power-up position is the
    origin and :attr:`homed` is False.

One reader task
    A background task reads every line and dispatches it: replies resolve the
    single outstanding request (``seq`` matched with
    :func:`~.protocol.match_reply`), ``! DONE`` / ``! ABORTED`` resolve the
    move started by that ``seq``. Requests are serialised by a *writer lock*
    held only for one request / reply exchange, never for a whole move, so
    :meth:`stop`, :meth:`get_position` and :meth:`status` work during a move.

Moves
    :meth:`move_to` checks the Cartesian limits and the motor travel, plans
    backlash waypoints (:meth:`~.steps.StepMapper.plan_moves`) and sends one
    ``MOVE`` per waypoint, waiting for its ``DONE`` with a distance-aware
    timeout. It returns the position the firmware *reported* in ``DONE``. A
    move registers itself as in flight synchronously, before its first
    ``await``, and every ``MOVE`` is sent only after re-checking - under the
    writer lock - that no stop was requested, so a :meth:`stop` can never be
    overtaken by a move that was just starting. :meth:`home` uses the
    firmware's ``HOME`` for the final leg when all axes are homed.

Failures (never assume a move succeeded)
    * timeout of a reply, corrupt line (CRC), reply out of sequence: the
      outstanding request and any running move fail with
      :class:`~confocal.errors.CommunicationError` /
      :class:`~confocal.errors.ProtocolError`, the stage goes to ``ERROR``
      and the next request first resynchronises (``PING`` + ``GETPOS``);
      if that fails the stage disconnects;
    * no ``DONE`` within the move timeout: ``STOP`` is sent and
      :class:`~confocal.errors.MotionTimeoutError` raised;
    * ``! ABORTED``: :class:`~confocal.errors.MotionAbortedError` at the
      partial position;
    * the link breaks (USB unplugged) or the Uno resets unexpectedly (an
      unsolicited ``BOOT``: the position is lost): the stage disconnects and
      every operation raises :class:`~confocal.errors.HardwareNotConnectedError`
      until :meth:`connect` is called again (which re-references the origin);
    * ``E_RANGE`` -> :class:`~confocal.errors.LimitViolationError`,
      ``E_BUSY`` -> :class:`~confocal.errors.MotionError`, other ``ERR`` codes
      -> :class:`~confocal.errors.ProtocolError`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

from confocal.config import ArduinoConfig, MotionConfig, Settings
from confocal.errors import (
    CommunicationError,
    HardwareError,
    HardwareNotConnectedError,
    LimitViolationError,
    MotionAbortedError,
    MotionError,
    MotionTimeoutError,
    MotionVerificationError,
    ProtocolError,
)
from confocal.hardware.arduino.protocol import (
    Command,
    ErrorCode,
    Event,
    EventKind,
    FirmwareState,
    FirmwareStatus,
    MalformedMessageError,
    Response,
    SequenceCounter,
    encode_command,
    match_reply,
    parse_line,
)
from confocal.hardware.arduino.steps import StepMapper, motor_configs
from confocal.hardware.arduino.transport import (
    SerialTransport,
    Transport,
    TransportClosedError,
)
from confocal.hardware.base import Stage
from confocal.hardware.kinematics import MOTORS, MotorSteps, StageKinematics
from confocal.models.common import Axis, Position, StageLimits
from confocal.models.hardware import StageState, StageStatus

log = logging.getLogger(__name__)

TransportFactory = Callable[[], Transport]

_ORIGIN_STEPS: MotorSteps = (0, 0, 0)

#: ERR codes meaning the device could not read our line: the link is noisy.
_LINK_ERROR_CODES = frozenset({ErrorCode.E_CRC, ErrorCode.E_LENGTH, ErrorCode.E_SYNTAX})

_FIRMWARE_STATES: dict[FirmwareState, StageState] = {
    FirmwareState.IDLE: StageState.IDLE,
    FirmwareState.MOVING: StageState.MOVING,
    FirmwareState.HOMING: StageState.HOMING,
    FirmwareState.STOPPED: StageState.STOPPED,
}


def _format(position: Position) -> str:
    return f"({position.x_um:.3f}, {position.y_um:.3f}, {position.z_um:.3f}) um"


def _quiet_fail(future: asyncio.Future[Any], error: BaseException) -> None:
    """Fail ``future`` without an "exception never retrieved" warning if nobody awaits it."""
    if not future.done():
        future.set_exception(error)
        future.exception()


class ArduinoStage(Stage):
    """:class:`Stage` in micrometres on top of the Uno firmware; see the module docstring."""

    backend_name = "arduino"

    def __init__(
        self,
        *,
        mapper: StepMapper,
        limits: StageLimits,
        motion: MotionConfig,
        transport_factory: TransportFactory,
        timeout_s: float = 2.0,
        handshake_timeout_s: float = 5.0,
    ) -> None:
        if not timeout_s > 0 or not handshake_timeout_s > 0:
            raise ValueError("timeouts must be positive")
        self._mapper = mapper
        self._limits = limits
        self._motion = motion
        self._transport_factory = transport_factory
        self._timeout_s = float(timeout_s)
        self._handshake_timeout_s = float(handshake_timeout_s)

        self._transport: Transport | None = None
        self._reader: asyncio.Task[None] | None = None
        self._background: set[asyncio.Task[None]] = set()
        self._writer_lock = asyncio.Lock()
        self._seq = SequenceCounter()
        self._pending: tuple[int, asyncio.Future[Response]] | None = None
        self._completions: dict[int, asyncio.Future[Event]] = {}

        self._connected = False
        self._needs_resync = False
        self._state = StageState.DISCONNECTED
        self._steps: MotorSteps | None = None
        self._version: str | None = None
        self._homed = False
        self._moving = False
        self._stop_requested = False
        self._last_error: str | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        kinematics: StageKinematics | None = None,
        transport_factory: TransportFactory | None = None,
    ) -> ArduinoStage:
        """Build from ``[arduino]``, ``[kinematics]``, ``[limits]`` and ``[motion]``.

        Nothing is opened here. ``transport_factory`` (tests, the hardware
        check) replaces the default :class:`SerialTransport` on ``arduino.port``.
        """
        arduino = settings.arduino
        kin = (
            kinematics
            if kinematics is not None
            else StageKinematics.from_config(settings.kinematics)
        )
        return cls(
            mapper=StepMapper(kin, motor_configs(arduino)),
            limits=settings.limits,
            motion=settings.motion,
            transport_factory=transport_factory or serial_transport_factory(arduino),
            timeout_s=arduino.timeout_s,
            handshake_timeout_s=arduino.handshake_timeout_s,
        )

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> None:
        """Open the link, wait for BOOT, PING, STATUS, GETPOS (see the module docstring)."""
        if self._connected:
            return
        await self._shutdown_link("reconnecting")
        transport = self._transport_factory()
        self._transport = transport
        self._needs_resync = False
        try:
            await transport.open()
            boot_version = await self._await_boot(transport)
            self._reader = asyncio.create_task(
                self._reader_loop(transport), name=f"arduino reader ({transport.description})"
            )
            self._connected = True
            await self._request(Command.PING, resync=False)
            status = FirmwareStatus.from_response(await self._request(Command.STATUS, resync=False))
            if status.moving:
                raise ProtocolError("the firmware reports a running move right after its reset")
            if status.version != boot_version:
                log.warning(
                    "firmware BOOT banner says %s but STATUS says %s", boot_version, status.version
                )
            steps = (await self._request(Command.GETPOS, resync=False)).steps()
        except BaseException as exc:
            await self._shutdown_link("connect failed")
            self._state = StageState.ERROR
            if isinstance(exc, HardwareError) or not isinstance(exc, Exception):
                self._last_error = f"connect failed: {exc}"
                raise  # hardware errors as they are; cancellation / interrupts untouched
            self._last_error = f"connect failed: {exc!r}"
            raise CommunicationError(
                f"cannot connect to the Arduino on {transport.description}: {exc}"
            ) from exc
        self._version = status.version
        self._steps = steps
        self._homed = False
        self._state = StageState.IDLE
        self._last_error = None
        log.info(
            "Arduino stage connected on %s (firmware %s), steps %s",
            transport.description,
            self._version,
            steps,
        )

    async def close(self) -> None:
        if self._moving and self._connected:
            with contextlib.suppress(HardwareError):
                await self.stop()
        await self._shutdown_link("closed")
        self._connected = False
        self._state = StageState.DISCONNECTED

    @property
    def connected(self) -> bool:
        return self._connected

    def version(self) -> str | None:
        return self._version

    # ------------------------------------------------------------------ state
    @property
    def limits(self) -> StageLimits:
        return self._limits

    @property
    def mapper(self) -> StepMapper:
        return self._mapper

    @property
    def homed(self) -> bool:
        return self._homed

    @property
    def moving(self) -> bool:
        """True while a :meth:`move_to` / :meth:`home` is in flight."""
        return self._moving

    @property
    def last_steps(self) -> MotorSteps | None:
        """Firmware step counters last reported by the device (no I/O)."""
        return self._steps

    @property
    def transport(self) -> Transport | None:
        return self._transport

    async def get_position(self) -> Position:
        """GETPOS (allowed during a move: the interpolated position reached so far)."""
        self._require_connected()
        steps = (await self._request(Command.GETPOS)).steps()
        self._steps = steps
        return self._mapper.to_position(steps)

    async def get_steps(self) -> MotorSteps:
        """GETPOS in firmware steps (hardware check and diagnostics)."""
        await self.get_position()
        assert self._steps is not None
        return self._steps

    async def ping(self) -> float:
        """PING round trip in seconds."""
        self._require_connected()
        started = time.perf_counter()
        await self._request(Command.PING)
        return time.perf_counter() - started

    async def firmware_status(self) -> FirmwareStatus:
        """STATUS, parsed. Raises on any failure (unlike :meth:`status`)."""
        self._require_connected()
        status = FirmwareStatus.from_response(await self._request(Command.STATUS))
        self._steps = status.steps
        self._version = status.version
        return status

    async def status(self) -> StageStatus:
        """Stage status from the firmware's STATUS. Never raises.

        Without a usable link (disconnected, or out of step until the next
        request resynchronises) the last known values are reported.
        """
        state = self._state
        error = self._last_error
        if self._connected and not self._needs_resync:
            try:
                async with asyncio.timeout(2.0 * self._timeout_s):
                    firmware = await self.firmware_status()
                if state is not StageState.ERROR:
                    state = _FIRMWARE_STATES[firmware.state]
            except Exception as exc:
                error = f"stage status unavailable: {exc}"
                state = StageState.ERROR if self._connected else StageState.DISCONNECTED
        elif not self._connected:
            state = StageState.DISCONNECTED if state is not StageState.ERROR else state
        steps = self._steps
        return StageStatus(
            backend=self.backend_name,
            connected=self._connected,
            state=state,
            position=self._mapper.to_position(steps) if steps is not None else None,
            homed=self._homed,
            limits=self._limits,
            firmware_version=self._version,
            last_error=error,
        )

    # ------------------------------------------------------------------ motion
    async def move_to(self, target: Position) -> Position:
        self._require_connected()
        self.check_limits(target)
        goal = self._mapper.to_motor_steps(target)
        self._mapper.check_travel(goal)
        self._begin_motion()
        try:
            final = await self._drive(goal, StageState.MOVING, f"move to {_format(target)}")
        finally:
            self._moving = False
        return self._mapper.to_position(final)

    async def home(self, axes: Sequence[Axis] | None = None) -> Position:
        """Move the selected axes (default: all) to the power-up origin (there are no end-stops)."""
        self._require_connected()
        selected = set(Axis) if axes is None else set(axes)
        everything = selected == set(Axis)
        self._begin_motion()
        try:
            current = await self._fresh_steps()
            target = self._mapper.to_position(current)
            for axis in selected:
                target = target.with_axis(axis, 0.0)
            self.check_limits(target)
            goal = _ORIGIN_STEPS if everything else self._mapper.to_motor_steps(target)
            self._mapper.check_travel(goal)
            final = await self._drive(
                goal,
                StageState.HOMING,
                f"homing {sorted(axis.value for axis in selected)}",
                final_command=Command.HOME if everything else Command.MOVE,
            )
        finally:
            self._moving = False
        if everything:
            self._homed = True
        return self._mapper.to_position(final)

    async def stop(self) -> None:
        """Send STOP at once (from any task, during a move). Never raises because idle.

        A move in flight then fails with :class:`MotionAbortedError`; a move
        that has not sent its ``MOVE`` yet is not sent at all. Without a link
        there is nothing to stop and this returns quietly; a failing STOP
        exchange raises :class:`CommunicationError`.
        """
        self._stop_requested = True
        if not self._connected or self._transport is None:
            return
        response = await self._request(Command.STOP, resync=False)
        self._steps = response.steps()

    async def release(self) -> None:
        """RELEASE: de-energise the coils (refused during a move)."""
        self._require_connected()
        if self._moving:
            raise MotionError("cannot release the motors during a move")
        await self._request(Command.RELEASE)

    # ------------------------------------------------------------------ motion internals
    def _begin_motion(self) -> None:
        """Register a move as in flight. Synchronous: call before the first ``await``."""
        if self._moving:
            raise MotionError("a move is already in progress")
        self._moving = True
        self._stop_requested = False

    def _abort_if_stopped(self, what: str) -> None:
        if self._stop_requested:
            self._state = StageState.STOPPED
            raise MotionAbortedError(f"{what} stopped before it was sent to the Arduino")

    async def _fresh_steps(self) -> MotorSteps:
        """The current counters; re-read when unknown or the link was out of step."""
        if self._steps is None or self._needs_resync:
            return (await self._request(Command.GETPOS)).steps()
        return self._steps

    def _segment_timeout_s(self, start: MotorSteps, end: MotorSteps) -> float:
        """Distance-aware completion timeout of one waypoint.

        The larger of ``MotionConfig.move_timeout_for`` (Cartesian distance at
        the configured speeds) and the same formula on the firmware's own
        duration (the slowest motor at its ``max_speed_steps_s``), so a
        mis-set speed in ``[motion]`` cannot turn a slow move into a timeout.
        """
        motion = self._motion
        cartesian = motion.move_timeout_for(
            self._mapper.to_position(start), self._mapper.to_position(end)
        )
        firmware_s = max(
            abs(b - a) / self._mapper.motor_config(motor).max_speed_steps_s
            for motor, a, b in zip(MOTORS, start, end, strict=True)
        )
        return max(cartesian, motion.move_timeout_factor * firmware_s + motion.move_timeout_s)

    async def _drive(
        self,
        goal: MotorSteps,
        state: StageState,
        description: str,
        *,
        final_command: Command = Command.MOVE,
    ) -> MotorSteps:
        """Run the backlash-planned waypoints to ``goal``; return the reported final steps."""
        try:
            current = await self._fresh_steps()
            waypoints = self._mapper.plan_moves(current, goal, limits=self._limits)
            self._state = state
            for index, waypoint in enumerate(waypoints):
                last = index == len(waypoints) - 1
                command = final_command if last else Command.MOVE
                reported = await self._run_segment(command, current, waypoint, description)
                if reported != waypoint:
                    raise MotionVerificationError(
                        f"{description}: the firmware finished at steps {reported} "
                        f"instead of {waypoint}"
                    )
                current = reported
            if self._stop_requested:
                self._state = StageState.STOPPED
                raise MotionAbortedError(
                    f"{description} was stopped (it finished at "
                    f"{_format(self._mapper.to_position(current))})"
                )
        except MotionAbortedError:
            self._state = StageState.STOPPED
            raise
        except LimitViolationError:
            self._state = StageState.IDLE  # E_RANGE: the firmware refused, nothing moved
            raise
        except HardwareError as exc:
            self._state = StageState.ERROR
            self._last_error = f"{description} failed: {exc}"
            raise
        except asyncio.CancelledError:
            # The caller gave up (e.g. the controller's own timeout): never leave the motors
            # running. The STOP is sent from a separate task because this one is cancelled.
            self._state = StageState.STOPPED
            self._spawn(self._stop_quietly("move cancelled"))
            raise
        except Exception:
            self._state = StageState.ERROR
            raise
        self._state = StageState.IDLE
        self._last_error = None
        return current

    async def _run_segment(
        self, command: Command, start: MotorSteps, waypoint: MotorSteps, description: str
    ) -> MotorSteps:
        """One MOVE / HOME: OK ack, then the DONE / ABORTED event with its sequence number."""
        args = waypoint if command is Command.MOVE else ()
        response, completion = await self._request_motion(command, args, description)
        timeout_s = self._segment_timeout_s(start, waypoint)
        try:
            # asyncio.timeout, not wait_for: on Python 3.11 wait_for can swallow a
            # cancellation that races with the future completing.
            async with asyncio.timeout(timeout_s):
                event = await completion
        except TimeoutError:
            await self._stop_quietly(f"{description} timed out")
            where = self._steps
            raise MotionTimeoutError(
                f"{description}: no DONE from the Arduino within {timeout_s:.3g} s "
                f"(stopped at steps {where})"
            ) from None
        finally:
            self._completions.pop(response.seq, None)
        assert event.steps is not None
        if event.kind is EventKind.ABORTED:
            self._state = StageState.STOPPED
            raise MotionAbortedError(
                f"{description} stopped at {_format(self._mapper.to_position(event.steps))}"
            )
        return event.steps

    async def _stop_quietly(self, reason: str) -> None:
        try:
            await self.stop()
        except Exception:
            log.exception("could not stop the Arduino stage (%s)", reason)

    def _spawn(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # ------------------------------------------------------------------ requests
    def _require_connected(self) -> None:
        if not self._connected or self._transport is None:
            detail = f": {self._last_error}" if self._last_error else ""
            raise HardwareNotConnectedError(f"Arduino stage is not connected{detail}")

    async def _request(self, command: Command, *, resync: bool = True) -> Response:
        """A non-motion request; returns its OK reply (ERR replies raise)."""
        response, _ = await self._locked_exchange(command, (), resync=resync)
        return response

    async def _request_motion(
        self, command: Command, args: Sequence[int], description: str
    ) -> tuple[Response, asyncio.Future[Event]]:
        response, completion = await self._locked_exchange(
            command, args, resync=True, completion=True, guard=description
        )
        assert completion is not None
        return response, completion

    async def _locked_exchange(
        self,
        command: Command,
        args: Sequence[int],
        *,
        resync: bool,
        completion: bool = False,
        guard: str | None = None,
    ) -> tuple[Response, asyncio.Future[Event] | None]:
        self._require_connected()
        async with self._writer_lock:
            self._require_connected()
            transport = self._transport
            assert transport is not None
            if resync and self._needs_resync:
                await self._resync(transport)
            if guard is not None:
                # Checked under the writer lock right before writing: a stop() that got here
                # first has already been sent, so this move must not start.
                self._abort_if_stopped(guard)
            return await self._exchange(transport, command, args, completion=completion)

    async def _exchange(
        self,
        transport: Transport,
        command: Command,
        args: Sequence[int],
        *,
        completion: bool,
    ) -> tuple[Response, asyncio.Future[Event] | None]:
        """Write one request and wait for its reply. Writer lock held."""
        loop = asyncio.get_running_loop()
        seq = self._seq.next()
        line = encode_command(seq, command, args)
        reply: asyncio.Future[Response] = loop.create_future()
        done: asyncio.Future[Event] | None = loop.create_future() if completion else None
        self._pending = (seq, reply)
        if done is not None:
            self._completions[seq] = done
        try:
            try:
                await transport.write_line(line)
            except CommunicationError as exc:
                self._link_lost(f"writing {command.value} failed: {exc}")
                raise
            try:
                async with asyncio.timeout(self._timeout_s):
                    response = await reply
            except TimeoutError:
                error = CommunicationError(
                    f"no reply to {command.value} (seq {seq}) from the Arduino within "
                    f"{self._timeout_s:.3g} s"
                )
                self._link_fault(error)
                raise error from None
        except BaseException:
            self._completions.pop(seq, None)
            raise
        finally:
            self._pending = None
        if not response.ok:
            self._completions.pop(seq, None)
            if response.error in _LINK_ERROR_CODES:
                self._needs_resync = True
            self._last_error = (
                f"{command.value} refused: {response.error.value if response.error else ''} "
                f"{response.message}".rstrip()
            )
            response.raise_for_error()
        return response, done

    async def _resync(self, transport: Transport) -> None:
        """PING + GETPOS after the link was out of step. Writer lock held.

        While ``_needs_resync`` is set the dispatcher drops replies to other
        sequence numbers (late replies to requests that timed out).
        """
        log.info("resynchronising with the Arduino")
        try:
            await self._exchange(transport, Command.PING, (), completion=False)
            response, _ = await self._exchange(transport, Command.GETPOS, (), completion=False)
            steps = response.steps()
        except HardwareError as exc:
            error = CommunicationError(f"cannot resynchronise with the Arduino: {exc}")
            self._link_lost(str(error))
            raise error from exc
        self._steps = steps
        self._needs_resync = False

    # ------------------------------------------------------------------ reader / dispatch
    async def _await_boot(self, transport: Transport) -> str:
        """Wait for ``! BOOT <version>``, ignoring bootloader noise and stale lines."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._handshake_timeout_s
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                line = await transport.read_line(remaining)
            except TransportClosedError:
                raise
            except CommunicationError:  # timeout
                break
            try:
                message = parse_line(line)
            except MalformedMessageError:
                log.debug("ignoring %r while waiting for BOOT", line)
                continue
            if isinstance(message, Event) and message.kind is EventKind.BOOT:
                assert message.version is not None
                return message.version
            log.debug("ignoring %r while waiting for BOOT", line)
        raise CommunicationError(
            f"no BOOT banner from {transport.description} within "
            f"{self._handshake_timeout_s:.3g} s: is the confocal firmware flashed, is this the "
            "Arduino's port, and does the board reset when the port opens (DTR)?"
        )

    async def _reader_loop(self, transport: Transport) -> None:
        try:
            while True:
                line = await transport.read_line(None)
                self._dispatch(line)
        except CommunicationError as exc:
            if transport is self._transport:
                self._link_lost(f"serial link lost: {exc}")

    def _dispatch(self, line: bytes) -> None:
        try:
            message = parse_line(line)
        except MalformedMessageError as exc:
            self._link_fault(ProtocolError(f"corrupt line from the Arduino ({exc}): {line!r}"))
            return
        if isinstance(message, Event):
            self._on_event(message)
            return
        pending = self._pending
        if pending is None:
            log.warning("unsolicited reply from the Arduino: %r", line)
            self._needs_resync = True
            return
        seq, reply = pending
        try:
            response = match_reply(message, seq)
        except ProtocolError as exc:
            if self._needs_resync:
                log.info("dropping late reply %r while resynchronising", line)
                return
            self._link_fault(exc)
            return
        if response is not None and not reply.done():
            reply.set_result(response)

    def _on_event(self, event: Event) -> None:
        if event.kind is EventKind.BOOT:
            self._link_lost(
                f"the Arduino reset unexpectedly (BOOT {event.version}): the stage position is "
                "lost; reconnect the stage and re-reference it",
                error_type=MotionError,
            )
            return
        assert event.seq is not None
        assert event.steps is not None
        self._steps = event.steps
        waiter = self._completions.pop(event.seq, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(event)
        else:
            log.debug("%s for request %d that nobody waits for", event.kind.value, event.seq)

    # ------------------------------------------------------------------ failures
    def _fail_waiters(self, error: BaseException) -> None:
        pending = self._pending
        if pending is not None:
            _quiet_fail(pending[1], error)
        for waiter in list(self._completions.values()):
            _quiet_fail(waiter, error)
        self._completions.clear()

    def _link_fault(self, error: HardwareError) -> None:
        """The link is out of step (timeout, CRC, sequence): fail everything, resync later."""
        log.error("Arduino link fault: %s", error)
        self._needs_resync = True
        self._state = StageState.ERROR
        self._last_error = str(error)
        self._fail_waiters(error)

    def _link_lost(
        self, reason: str, *, error_type: type[HardwareError] = CommunicationError
    ) -> None:
        """The link is unusable (unplugged, reset): fail everything and disconnect."""
        log.error("Arduino stage disconnected: %s", reason)
        error = error_type(reason)
        self._connected = False
        self._state = StageState.ERROR
        self._last_error = reason
        self._fail_waiters(error)
        transport = self._transport
        self._transport = None
        if transport is not None:
            self._spawn(self._close_transport(transport))

    async def _close_transport(self, transport: Transport) -> None:
        try:
            await transport.close()
        except Exception:
            log.exception("closing the Arduino transport failed")

    async def _shutdown_link(self, reason: str) -> None:
        transport, reader = self._transport, self._reader
        self._transport = None
        self._reader = None
        self._connected = False
        self._fail_waiters(HardwareNotConnectedError(f"Arduino stage {reason}"))
        if transport is not None:
            await self._close_transport(transport)
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
        for task in list(self._background):
            if task is not asyncio.current_task():
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await task


def serial_transport_factory(arduino: ArduinoConfig) -> TransportFactory:
    """Factory of the real link on ``arduino.port`` (nothing is opened until ``open()``)."""

    def factory() -> Transport:
        return SerialTransport(arduino.port, arduino.baudrate, write_timeout_s=arduino.timeout_s)

    return factory
