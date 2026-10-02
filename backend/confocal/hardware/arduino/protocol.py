"""Serial line protocol between the Raspberry Pi and the Arduino Uno firmware (pure codec).

The complete specification is ``docs/serial-protocol.md``; in short, one
ASCII message per line, at most :data:`MAX_LINE_BYTES` bytes including the
``\\n`` terminator::

    host -> device   <seq> <CMD> [args...]*<CRC>
    device -> host   <seq> OK [payload...]*<CRC>
                     <seq> ERR <CODE> [message]*<CRC>
                     ! <EVENT> [payload...]*<CRC>          (asynchronous event)

``<CRC>`` is CRC-8 (polynomial 0x07, initial value 0x00, no reflection) of
every byte before the ``*``, as two upper-case hex digits. ``<seq>`` (1..65535,
rolling) ties every reply to its request; the device replies with seq 0 when
it could not read the request's own number (corrupt line). Tokens are
separated by single spaces.

``MOVE <a> <b> <c>`` carries absolute firmware steps of the three motors. It is
acknowledged at once with ``OK``; completion arrives later as the event
``! DONE <seq> <a> <b> <c>`` (or ``! ABORTED <seq> <a> <b> <c>`` when ``STOP``
pre-empts it), so the host can always send ``STOP`` during a move.

This module performs no I/O: it turns typed messages into bytes and bytes into
typed messages, raising :class:`MalformedMessageError` (a
:class:`~confocal.errors.ProtocolError`) for anything malformed, corrupt or
oversized. The serial transport and ``ArduinoStage`` (``.stage``) build on it;
``emulator.FirmwareEmulator`` uses it from the device side.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from confocal.errors import LimitViolationError, MotionError, ProtocolError
from confocal.hardware.kinematics import MotorSteps

#: Longest line in either direction, including the ``\n`` terminator.
MAX_LINE_BYTES = 96

#: Sequence numbers used by the host, rolling over from SEQ_MAX to SEQ_MIN.
SEQ_MIN = 1
SEQ_MAX = 65535

#: Sequence number of a device reply to a request whose own number was unreadable.
UNATTRIBUTED_SEQ = 0

#: Largest absolute step value the protocol carries (7 digits; fits the Uno's ``long``).
STEP_LIMIT = 9_999_999

#: Longest firmware version string (keeps the STATUS reply within MAX_LINE_BYTES).
MAX_VERSION_LENGTH = 8

EVENT_MARKER = "!"

_SEQ_RE = re.compile(r"[0-9]{1,5}")
_INT_RE = re.compile(r"-?[0-9]{1,7}")
_VERSION_RE = re.compile(rf"[0-9A-Za-z._+-]{{1,{MAX_VERSION_LENGTH}}}")
_HEX_DIGITS = frozenset(b"0123456789ABCDEFabcdef")


class Command(StrEnum):
    """Host -> device commands."""

    MOVE = "MOVE"  # MOVE <a> <b> <c>: coordinated move to absolute firmware steps
    GETPOS = "GETPOS"  # current steps (interpolated during a move)
    HOME = "HOME"  # coordinated move of every motor back to step 0
    STOP = "STOP"  # halt immediately; replies with the position where it stopped
    STATUS = "STATUS"  # key=value state summary
    PING = "PING"  # liveness check
    ZERO = "ZERO"  # define the current position as the origin (all counters 0)
    RELEASE = "RELEASE"  # de-energise the coils (28BYJ-48 coils heat up when held)


#: Number of integer arguments each command takes.
COMMAND_ARITY: dict[Command, int] = dict.fromkeys(Command, 0) | {Command.MOVE: 3}


class ErrorCode(StrEnum):
    """Codes of ``ERR`` replies."""

    E_CRC = "E_CRC"  # checksum missing or wrong: the line cannot be trusted at all
    E_SYNTAX = "E_SYNTAX"  # malformed line: sequence number, characters or spacing
    E_LENGTH = "E_LENGTH"  # line longer than MAX_LINE_BYTES
    E_UNKNOWN = "E_UNKNOWN"  # unknown command
    E_ARGS = "E_ARGS"  # wrong number or format of arguments
    E_BUSY = "E_BUSY"  # MOVE / HOME / ZERO / RELEASE while a move is running
    E_RANGE = "E_RANGE"  # target outside the firmware's motor travel; nothing moved


class EventKind(StrEnum):
    """Asynchronous device -> host events (lines starting with ``!``)."""

    DONE = "DONE"  # ! DONE <seq> <a> <b> <c>: the move started by <seq> finished
    ABORTED = "ABORTED"  # ! ABORTED <seq> <a> <b> <c>: STOP pre-empted that move
    BOOT = "BOOT"  # ! BOOT <version>: the firmware (re)started; counters are 0


class FirmwareState(StrEnum):
    """``state`` field of the STATUS reply."""

    IDLE = "idle"
    MOVING = "moving"
    HOMING = "homing"
    STOPPED = "stopped"  # the last move was pre-empted by STOP


class MalformedMessageError(ProtocolError):
    """A line or message violates the protocol.

    ``code`` is the error code the device sends back for it, ``seq`` the
    sequence number it is attributed to (:data:`UNATTRIBUTED_SEQ` if unknown).
    """

    def __init__(self, code: ErrorCode, reason: str, *, seq: int = UNATTRIBUTED_SEQ) -> None:
        super().__init__(f"{code.value}: {reason}")
        self.code = code
        self.reason = reason
        self.seq = seq


# --------------------------------------------------------------------------- framing


def _crc8_table() -> tuple[int, ...]:
    table: list[int] = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return tuple(table)


_CRC8_TABLE = _crc8_table()


def crc8(data: bytes) -> int:
    """CRC-8, polynomial 0x07, initial value 0x00 (check value of b"123456789": 0xF4).

    Detects every error burst of up to 8 bits, in particular any single
    corrupted byte.
    """
    crc = 0
    for byte in data:
        crc = _CRC8_TABLE[crc ^ byte]
    return crc


def _check_text(text: str, *, what: str, code: ErrorCode = ErrorCode.E_SYNTAX) -> None:
    """Printable ASCII, no ``*``, single spaces between non-empty tokens."""
    if any(not " " <= ch <= "~" for ch in text) or "*" in text:
        raise MalformedMessageError(code, f"{what} contains a forbidden character")
    if text and any(token == "" for token in text.split(" ")):
        raise MalformedMessageError(code, f"{what} has leading, trailing or double spaces")


def frame(body: str) -> bytes:
    """``body`` + ``*`` + CRC-8 + ``\\n``.

    Raises:
        MalformedMessageError: forbidden characters (E_SYNTAX) or too long (E_LENGTH).
    """
    if not body:
        raise MalformedMessageError(ErrorCode.E_SYNTAX, "empty message")
    _check_text(body, what="message")
    data = body.encode("ascii")
    line = data + b"*%02X\n" % crc8(data)
    if len(line) > MAX_LINE_BYTES:
        raise MalformedMessageError(
            ErrorCode.E_LENGTH, f"line of {len(line)} bytes exceeds {MAX_LINE_BYTES}"
        )
    return line


def unframe(line: bytes) -> str:
    """Validate length, checksum and characters of one received line; return its body.

    A trailing ``\\n`` (optionally preceded by ``\\r``) is accepted and removed.
    The checksum is verified on the raw bytes *before* anything is decoded, so
    any corruption is reported as E_CRC (the receiver may simply retry).
    """
    if len(line) > MAX_LINE_BYTES:
        raise MalformedMessageError(
            ErrorCode.E_LENGTH, f"line of {len(line)} bytes exceeds {MAX_LINE_BYTES}"
        )
    data = line.removesuffix(b"\n").removesuffix(b"\r")
    star = data.rfind(b"*")
    if star < 0:
        raise MalformedMessageError(ErrorCode.E_CRC, "missing '*<crc>' checksum")
    raw_body, crc_field = data[:star], data[star + 1 :]
    if len(crc_field) != 2 or not all(byte in _HEX_DIGITS for byte in crc_field):
        raise MalformedMessageError(ErrorCode.E_CRC, "checksum field is not two hex digits")
    if crc8(raw_body) != int(crc_field, 16):
        raise MalformedMessageError(ErrorCode.E_CRC, "checksum mismatch")
    try:
        body = raw_body.decode("ascii")
    except UnicodeDecodeError:
        raise MalformedMessageError(ErrorCode.E_SYNTAX, "non-ASCII bytes") from None
    if not body:
        raise MalformedMessageError(ErrorCode.E_SYNTAX, "empty message")
    _check_text(body, what="message")
    return body


# --------------------------------------------------------------------------- tokens


def _check_seq(seq: int, *, allow_unattributed: bool) -> None:
    low = UNATTRIBUTED_SEQ if allow_unattributed else SEQ_MIN
    if not low <= seq <= SEQ_MAX:
        raise MalformedMessageError(
            ErrorCode.E_SYNTAX, f"sequence number {seq} outside [{low}, {SEQ_MAX}]"
        )


def _parse_seq(token: str, *, allow_unattributed: bool) -> int:
    if not _SEQ_RE.fullmatch(token):
        raise MalformedMessageError(ErrorCode.E_SYNTAX, f"bad sequence number {token!r}")
    seq = int(token)
    _check_seq(seq, allow_unattributed=allow_unattributed)
    return seq


def _parse_int(token: str, code: ErrorCode, seq: int) -> int:
    if not _INT_RE.fullmatch(token):
        raise MalformedMessageError(code, f"bad integer {token!r}", seq=seq)
    return int(token)


def _check_steps(values: Sequence[int], code: ErrorCode, seq: int) -> None:
    for value in values:
        if not -STEP_LIMIT <= value <= STEP_LIMIT:
            raise MalformedMessageError(
                code, f"step value {value} exceeds +/-{STEP_LIMIT}", seq=seq
            )


def _check_payload(tokens: Sequence[str], seq: int) -> None:
    for token in tokens:
        if not token or " " in token:
            raise MalformedMessageError(
                ErrorCode.E_SYNTAX, "empty or spaced payload token", seq=seq
            )
        _check_text(token, what="payload")


def _as_steps(values: Sequence[int]) -> MotorSteps:
    a, b, c = (int(v) for v in values)
    return (a, b, c)


# --------------------------------------------------------------------------- messages


@dataclass(frozen=True, slots=True)
class Request:
    """One host -> device command."""

    seq: int
    command: Command
    args: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _check_seq(self.seq, allow_unattributed=False)
        expected = COMMAND_ARITY[self.command]
        if len(self.args) != expected:
            raise MalformedMessageError(
                ErrorCode.E_ARGS,
                f"{self.command.value} takes {expected} arguments, got {len(self.args)}",
                seq=self.seq,
            )
        _check_steps(self.args, ErrorCode.E_ARGS, self.seq)

    def encode(self) -> bytes:
        return frame(" ".join([str(self.seq), self.command.value, *map(str, self.args)]))


@dataclass(frozen=True, slots=True)
class FirmwareStatus:
    """Typed form of the STATUS reply payload."""

    state: FirmwareState
    steps: MotorSteps
    moving: bool
    coils_enabled: bool
    version: str

    def payload(self) -> tuple[str, ...]:
        a, b, c = self.steps
        return (
            f"state={self.state.value}",
            f"a={a}",
            f"b={b}",
            f"c={c}",
            f"moving={int(self.moving)}",
            f"enabled={int(self.coils_enabled)}",
            f"version={self.version}",
        )

    @classmethod
    def from_response(cls, response: Response) -> FirmwareStatus:
        """Parse a STATUS ``OK`` reply. Raises ProtocolError if a field is missing or bad."""
        fields = response.fields()
        try:
            state = FirmwareState(fields["state"])
            steps = _as_steps(
                [_parse_int(fields[key], ErrorCode.E_SYNTAX, response.seq) for key in "abc"]
            )
            flags = {key: fields[key] for key in ("moving", "enabled")}
            version = fields["version"]
        except (KeyError, ValueError) as exc:
            raise ProtocolError(f"incomplete or invalid STATUS reply: {exc}") from None
        if any(value not in ("0", "1") for value in flags.values()):
            raise ProtocolError("STATUS flags must be 0 or 1")
        return cls(
            state=state,
            steps=steps,
            moving=flags["moving"] == "1",
            coils_enabled=flags["enabled"] == "1",
            version=version,
        )


@dataclass(frozen=True, slots=True)
class Response:
    """A device reply (``OK`` or ``ERR``) to the request with the same ``seq``."""

    seq: int
    ok: bool
    payload: tuple[str, ...] = ()
    error: ErrorCode | None = None
    message: str = ""

    def __post_init__(self) -> None:
        _check_seq(self.seq, allow_unattributed=True)
        if self.ok and (self.error is not None or self.message):
            raise MalformedMessageError(
                ErrorCode.E_SYNTAX, "an OK reply has no error", seq=self.seq
            )
        if not self.ok and (self.error is None or self.payload):
            raise MalformedMessageError(
                ErrorCode.E_SYNTAX, "an ERR reply needs a code and no payload", seq=self.seq
            )
        _check_payload(self.payload, self.seq)
        _check_text(self.message, what="error message")

    @classmethod
    def success(cls, seq: int, *payload: object) -> Response:
        return cls(seq=seq, ok=True, payload=tuple(str(item) for item in payload))

    @classmethod
    def failure(cls, seq: int, code: ErrorCode, message: str = "") -> Response:
        return cls(seq=seq, ok=False, error=code, message=message)

    def encode(self) -> bytes:
        if self.ok:
            parts = [str(self.seq), "OK", *self.payload]
        else:
            assert self.error is not None  # guaranteed by __post_init__
            parts = [str(self.seq), "ERR", self.error.value]
            if self.message:
                parts.append(self.message)
        return frame(" ".join(parts))

    def steps(self) -> MotorSteps:
        """The ``<a> <b> <c>`` payload of GETPOS / STOP / ZERO replies."""
        if not self.ok or len(self.payload) != 3:
            raise ProtocolError(f"reply {self.seq} does not carry three step values")
        return _as_steps([_parse_int(t, ErrorCode.E_SYNTAX, self.seq) for t in self.payload])

    def fields(self) -> dict[str, str]:
        """``key=value`` payload tokens as a dict (STATUS reply)."""
        result: dict[str, str] = {}
        for token in self.payload:
            key, sep, value = token.partition("=")
            if not sep or not key:
                raise ProtocolError(f"payload token {token!r} is not key=value")
            result[key] = value
        return result

    def raise_for_error(self) -> None:
        """Map an ``ERR`` reply to the application's exception; no-op for ``OK``.

        E_RANGE -> LimitViolationError (nothing moved), E_BUSY -> MotionError,
        every framing / syntax code -> ProtocolError.
        """
        if self.ok:
            return
        assert self.error is not None
        text = f"firmware rejected request {self.seq}: {self.error.value}"
        if self.message:
            text += f" ({self.message})"
        if self.error is ErrorCode.E_RANGE:
            raise LimitViolationError(text, violations=[self.message or self.error.value])
        if self.error is ErrorCode.E_BUSY:
            raise MotionError(text)
        raise ProtocolError(text)


@dataclass(frozen=True, slots=True)
class Event:
    """An asynchronous device event (``! <EVENT> ...``)."""

    kind: EventKind
    seq: int | None = None
    steps: MotorSteps | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        if self.kind is EventKind.BOOT:
            if self.version is None or not _VERSION_RE.fullmatch(self.version):
                raise MalformedMessageError(ErrorCode.E_SYNTAX, f"bad version {self.version!r}")
            if self.seq is not None or self.steps is not None:
                raise MalformedMessageError(ErrorCode.E_SYNTAX, "BOOT carries only a version")
            return
        if self.seq is None or self.steps is None or self.version is not None:
            raise MalformedMessageError(
                ErrorCode.E_SYNTAX, f"{self.kind.value} needs a sequence number and three steps"
            )
        _check_seq(self.seq, allow_unattributed=False)
        if len(self.steps) != 3:
            raise MalformedMessageError(ErrorCode.E_SYNTAX, "expected three step values")
        _check_steps(self.steps, ErrorCode.E_SYNTAX, self.seq)

    @classmethod
    def done(cls, seq: int, steps: MotorSteps) -> Event:
        return cls(kind=EventKind.DONE, seq=seq, steps=steps)

    @classmethod
    def aborted(cls, seq: int, steps: MotorSteps) -> Event:
        return cls(kind=EventKind.ABORTED, seq=seq, steps=steps)

    @classmethod
    def boot(cls, version: str) -> Event:
        return cls(kind=EventKind.BOOT, version=version)

    def completes(self, seq: int) -> bool:
        """True if this event ends the move started by request ``seq``."""
        return self.kind in (EventKind.DONE, EventKind.ABORTED) and self.seq == seq

    def encode(self) -> bytes:
        if self.kind is EventKind.BOOT:
            return frame(f"{EVENT_MARKER} {self.kind.value} {self.version}")
        assert self.steps is not None
        a, b, c = self.steps
        return frame(f"{EVENT_MARKER} {self.kind.value} {self.seq} {a} {b} {c}")


# --------------------------------------------------------------------------- codec


def encode_command(seq: int, command: Command, args: Sequence[int] = ()) -> bytes:
    """Encode one host -> device command line (raises MalformedMessageError if invalid)."""
    return Request(seq=seq, command=command, args=tuple(int(a) for a in args)).encode()


def parse_request(line: bytes) -> Request:
    """Device side: decode one host command line.

    The raised :class:`MalformedMessageError` carries the code and sequence
    number for the ``ERR`` reply (seq 0 when the line's own number is unreadable).
    """
    tokens = unframe(line).split(" ")
    seq = _parse_seq(tokens[0], allow_unattributed=False)
    if len(tokens) < 2:
        raise MalformedMessageError(ErrorCode.E_SYNTAX, "missing command", seq=seq)
    try:
        command = Command(tokens[1])
    except ValueError:
        raise MalformedMessageError(
            ErrorCode.E_UNKNOWN, f"unknown command {tokens[1]!r}", seq=seq
        ) from None
    values = tokens[2:]
    if len(values) != COMMAND_ARITY[command]:
        raise MalformedMessageError(
            ErrorCode.E_ARGS,
            f"{command.value} takes {COMMAND_ARITY[command]} arguments, got {len(values)}",
            seq=seq,
        )
    return Request(
        seq=seq, command=command, args=tuple(_parse_int(v, ErrorCode.E_ARGS, seq) for v in values)
    )


def _parse_event(tokens: list[str]) -> Event:
    if not tokens:
        raise MalformedMessageError(ErrorCode.E_SYNTAX, "empty event")
    try:
        kind = EventKind(tokens[0])
    except ValueError:
        raise MalformedMessageError(ErrorCode.E_SYNTAX, f"unknown event {tokens[0]!r}") from None
    rest = tokens[1:]
    if kind is EventKind.BOOT:
        if len(rest) != 1:
            raise MalformedMessageError(ErrorCode.E_SYNTAX, "BOOT takes exactly a version")
        return Event.boot(rest[0])
    if len(rest) != 4:
        raise MalformedMessageError(ErrorCode.E_SYNTAX, f"{kind.value} takes <seq> <a> <b> <c>")
    seq = _parse_seq(rest[0], allow_unattributed=False)
    steps = _as_steps([_parse_int(t, ErrorCode.E_SYNTAX, seq) for t in rest[1:]])
    return Event(kind=kind, seq=seq, steps=steps)


def parse_line(line: bytes) -> Response | Event:
    """Host side: decode one device line into a :class:`Response` or an :class:`Event`."""
    tokens = unframe(line).split(" ")
    if tokens[0] == EVENT_MARKER:
        return _parse_event(tokens[1:])
    seq = _parse_seq(tokens[0], allow_unattributed=True)
    if len(tokens) >= 2 and tokens[1] == "OK":
        return Response(seq=seq, ok=True, payload=tuple(tokens[2:]))
    if len(tokens) >= 3 and tokens[1] == "ERR":
        try:
            code = ErrorCode(tokens[2])
        except ValueError:
            raise MalformedMessageError(
                ErrorCode.E_SYNTAX, f"unknown error code {tokens[2]!r}", seq=seq
            ) from None
        return Response(seq=seq, ok=False, error=code, message=" ".join(tokens[3:]))
    raise MalformedMessageError(ErrorCode.E_SYNTAX, "expected '<seq> OK' or '<seq> ERR <code>'")


# --------------------------------------------------------------------------- sequencing


def next_seq(seq: int) -> int:
    """The sequence number after ``seq``: rolls over from 65535 to 1 (0 is never sent)."""
    return SEQ_MIN if seq >= SEQ_MAX else seq + 1


class SequenceCounter:
    """Host-side source of rolling sequence numbers."""

    def __init__(self, start: int = SEQ_MIN) -> None:
        _check_seq(start, allow_unattributed=False)
        self._next = start
        self._last: int | None = None

    @property
    def last(self) -> int | None:
        """The most recently issued number (None before the first)."""
        return self._last

    def next(self) -> int:
        seq = self._next
        self._next = next_seq(seq)
        self._last = seq
        return seq


def match_reply(message: Response | Event, seq: int) -> Response | None:
    """Classify a received message while request ``seq`` awaits its reply.

    Returns the :class:`Response` if it answers ``seq`` - including an ``ERR``
    with seq 0, which the device sends when it could not read the request's
    number (the host has only one request outstanding, so it is ours).
    Returns None for an asynchronous :class:`Event` (handled separately).

    Raises:
        ProtocolError: a reply to a different request (out of sequence, e.g. a
            late reply after a timeout): the link is out of step.
    """
    if isinstance(message, Event):
        return None
    if message.seq == seq or (message.seq == UNATTRIBUTED_SEQ and not message.ok):
        return message
    raise ProtocolError(f"reply to request {message.seq} while waiting for request {seq}")
