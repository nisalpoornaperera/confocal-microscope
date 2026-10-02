"""Serial line protocol codec (docs/serial-protocol.md)."""

from __future__ import annotations

import pytest

from confocal.errors import LimitViolationError, MotionError, ProtocolError
from confocal.hardware.arduino.protocol import (
    MAX_LINE_BYTES,
    SEQ_MAX,
    UNATTRIBUTED_SEQ,
    Command,
    ErrorCode,
    Event,
    EventKind,
    FirmwareState,
    FirmwareStatus,
    MalformedMessageError,
    Request,
    Response,
    SequenceCounter,
    crc8,
    encode_command,
    frame,
    match_reply,
    next_seq,
    parse_line,
    parse_request,
    unframe,
)

# --------------------------------------------------------------------------- framing


def test_crc8_check_value() -> None:
    assert crc8(b"123456789") == 0xF4
    assert crc8(b"") == 0


def test_frame_layout() -> None:
    line = encode_command(7, Command.MOVE, (10, -20, 30))
    body = b"7 MOVE 10 -20 30"
    assert line == body + b"*%02X\n" % crc8(body)
    assert unframe(line) == body.decode()


def test_unframe_accepts_crlf() -> None:
    line = frame("1 PING").replace(b"\n", b"\r\n")
    assert unframe(line) == "1 PING"


def test_crc_detects_every_single_byte_corruption() -> None:
    line = encode_command(42, Command.MOVE, (1234, -5678, 9))
    content = line[:-1]  # keep the terminator
    for index in range(len(content)):
        for replacement in (0x30, 0x41, 0x2D, 0x20, 0x7E):
            if content[index] == replacement:
                continue
            corrupt = content[:index] + bytes([replacement]) + content[index + 1 :] + b"\n"
            with pytest.raises(ProtocolError):
                parse_request(corrupt)


@pytest.mark.parametrize(
    "line",
    [
        b"1 PING\n",  # no checksum
        b"1 PING*G1\n",  # not hex
        b"1 PING*1\n",  # one digit
        b"*00\n",  # empty body
    ],
)
def test_bad_checksum_field_is_e_crc_or_syntax(line: bytes) -> None:
    with pytest.raises(MalformedMessageError) as excinfo:
        unframe(line)
    assert excinfo.value.code in (ErrorCode.E_CRC, ErrorCode.E_SYNTAX)


def test_oversized_line_is_rejected_both_ways() -> None:
    with pytest.raises(MalformedMessageError) as excinfo:
        unframe(b"1 PING " + b"x" * MAX_LINE_BYTES + b"*00\n")
    assert excinfo.value.code is ErrorCode.E_LENGTH
    with pytest.raises(MalformedMessageError):
        frame("1 OK " + "y" * MAX_LINE_BYTES)


@pytest.mark.parametrize("body", ["1  PING", " 1 PING", "1 PING ", "1 PI*NG", "1 PING\t"])
def test_bad_spacing_or_characters_are_e_syntax(body: str) -> None:
    with pytest.raises(MalformedMessageError) as excinfo:
        frame(body)
    assert excinfo.value.code is ErrorCode.E_SYNTAX


def test_non_ascii_bytes_with_valid_crc_are_rejected() -> None:
    raw = "1 PÏNG".encode("latin-1")
    with pytest.raises(MalformedMessageError) as excinfo:
        unframe(raw + b"*%02X\n" % crc8(raw))
    assert excinfo.value.code is ErrorCode.E_SYNTAX


def _framed(body: str) -> bytes:
    data = body.encode()
    return data + b"*%02X\n" % crc8(data)


# --------------------------------------------------------------------------- requests


@pytest.mark.parametrize(
    ("command", "args"),
    [
        (Command.MOVE, (1, -2, 3)),
        (Command.GETPOS, ()),
        (Command.HOME, ()),
        (Command.STOP, ()),
        (Command.STATUS, ()),
        (Command.PING, ()),
        (Command.ZERO, ()),
        (Command.RELEASE, ()),
    ],
)
def test_request_round_trip(command: Command, args: tuple[int, ...]) -> None:
    request = parse_request(encode_command(SEQ_MAX, command, args))
    assert request == Request(seq=SEQ_MAX, command=command, args=args)


@pytest.mark.parametrize(
    ("body", "code", "seq"),
    [
        ("x PING", ErrorCode.E_SYNTAX, UNATTRIBUTED_SEQ),
        ("0 PING", ErrorCode.E_SYNTAX, UNATTRIBUTED_SEQ),
        ("70000 PING", ErrorCode.E_SYNTAX, UNATTRIBUTED_SEQ),
        ("5", ErrorCode.E_SYNTAX, 5),
        ("5 JUMP", ErrorCode.E_UNKNOWN, 5),
        ("5 MOVE 1 2", ErrorCode.E_ARGS, 5),
        ("5 MOVE 1 2 x", ErrorCode.E_ARGS, 5),
        ("5 MOVE 1 2 99999999", ErrorCode.E_ARGS, 5),
        ("5 PING 1", ErrorCode.E_ARGS, 5),
    ],
)
def test_malformed_requests_carry_code_and_seq(body: str, code: ErrorCode, seq: int) -> None:
    with pytest.raises(MalformedMessageError) as excinfo:
        parse_request(_framed(body))
    assert excinfo.value.code is code
    assert excinfo.value.seq == seq


def test_invalid_requests_cannot_be_encoded() -> None:
    with pytest.raises(MalformedMessageError):
        encode_command(0, Command.PING)
    with pytest.raises(MalformedMessageError):
        encode_command(1, Command.MOVE, (1, 2))
    with pytest.raises(MalformedMessageError):
        encode_command(1, Command.MOVE, (1, 2, 10_000_000))


# --------------------------------------------------------------------------- replies


def test_ok_reply_round_trip() -> None:
    reply = Response.success(9, 10, -20, 30)
    parsed = parse_line(reply.encode())
    assert parsed == reply
    assert isinstance(parsed, Response)
    assert parsed.steps() == (10, -20, 30)
    parsed.raise_for_error()


@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_error_code_round_trips(code: ErrorCode) -> None:
    reply = Response.failure(3, code, "some text")
    parsed = parse_line(reply.encode())
    assert parsed == reply


@pytest.mark.parametrize(
    ("code", "exception"),
    [
        (ErrorCode.E_RANGE, LimitViolationError),
        (ErrorCode.E_BUSY, MotionError),
        (ErrorCode.E_CRC, ProtocolError),
        (ErrorCode.E_SYNTAX, ProtocolError),
        (ErrorCode.E_LENGTH, ProtocolError),
        (ErrorCode.E_UNKNOWN, ProtocolError),
        (ErrorCode.E_ARGS, ProtocolError),
    ],
)
def test_error_replies_map_to_application_errors(
    code: ErrorCode, exception: type[Exception]
) -> None:
    with pytest.raises(exception, match=code.value):
        Response.failure(3, code).raise_for_error()


@pytest.mark.parametrize(
    "body",
    ["3 MAYBE", "3 ERR", "3 ERR E_NOPE", "! WHAT 1 2 3 4", "! DONE 1 2 3", "! BOOT", "x OK"],
)
def test_malformed_device_lines_raise_protocol_error(body: str) -> None:
    with pytest.raises(ProtocolError):
        parse_line(_framed(body))


def test_steps_of_a_reply_without_three_values_is_an_error() -> None:
    with pytest.raises(ProtocolError):
        Response.success(1).steps()
    with pytest.raises(ProtocolError):
        Response.success(1, "a", "b", "c").steps()


def test_inconsistent_responses_are_rejected() -> None:
    with pytest.raises(MalformedMessageError):
        Response(seq=1, ok=True, error=ErrorCode.E_CRC)
    with pytest.raises(MalformedMessageError):
        Response(seq=1, ok=False)
    with pytest.raises(MalformedMessageError):
        Response(seq=1, ok=True, payload=("a b",))


def test_status_reply_round_trip() -> None:
    status = FirmwareStatus(
        state=FirmwareState.MOVING,
        steps=(1, -2, 3),
        moving=True,
        coils_enabled=True,
        version="1.0.0",
    )
    parsed = parse_line(Response.success(4, *status.payload()).encode())
    assert isinstance(parsed, Response)
    assert FirmwareStatus.from_response(parsed) == status


@pytest.mark.parametrize(
    "payload",
    [
        ("state=idle", "a=0", "b=0", "c=0", "moving=0", "enabled=0"),  # no version
        ("state=dancing", "a=0", "b=0", "c=0", "moving=0", "enabled=0", "version=1"),
        ("state=idle", "a=0", "b=0", "c=0", "moving=2", "enabled=0", "version=1"),
        ("state=idle", "a=x", "b=0", "c=0", "moving=0", "enabled=0", "version=1"),
        ("garbage",),
    ],
)
def test_bad_status_reply_is_a_protocol_error(payload: tuple[str, ...]) -> None:
    with pytest.raises(ProtocolError):
        FirmwareStatus.from_response(Response.success(4, *payload))


# --------------------------------------------------------------------------- events


@pytest.mark.parametrize(
    "event",
    [Event.done(12, (1, 2, -3)), Event.aborted(SEQ_MAX, (0, -7, 7)), Event.boot("1.0.0")],
)
def test_event_round_trip(event: Event) -> None:
    assert parse_line(event.encode()) == event


def test_event_wire_format() -> None:
    assert unframe(Event.done(5, (1, 2, 3)).encode()) == "! DONE 5 1 2 3"
    assert unframe(Event.aborted(5, (1, 2, 3)).encode()) == "! ABORTED 5 1 2 3"
    assert unframe(Event.boot("1.0.0").encode()) == "! BOOT 1.0.0"


def test_event_completes_only_its_move() -> None:
    assert Event.done(5, (0, 0, 0)).completes(5)
    assert Event.aborted(5, (0, 0, 0)).completes(5)
    assert not Event.done(5, (0, 0, 0)).completes(6)
    assert not Event.boot("1").completes(5)


def test_invalid_events_are_rejected() -> None:
    with pytest.raises(MalformedMessageError):
        Event(kind=EventKind.BOOT, version="far too long version")
    with pytest.raises(MalformedMessageError):
        Event(kind=EventKind.DONE, seq=1)
    with pytest.raises(MalformedMessageError):
        Event(kind=EventKind.DONE, seq=0, steps=(0, 0, 0))


# --------------------------------------------------------------------------- sequencing


def test_sequence_numbers_roll_over_and_skip_zero() -> None:
    assert next_seq(1) == 2
    assert next_seq(SEQ_MAX) == 1
    counter = SequenceCounter(start=SEQ_MAX)
    assert counter.last is None
    assert [counter.next(), counter.next()] == [SEQ_MAX, 1]
    assert counter.last == 1


def test_match_reply() -> None:
    assert match_reply(Response.success(5), 5) == Response.success(5)
    assert match_reply(Event.done(5, (0, 0, 0)), 5) is None
    unattributed = Response.failure(UNATTRIBUTED_SEQ, ErrorCode.E_CRC)
    assert match_reply(unattributed, 5) is unattributed
    with pytest.raises(ProtocolError, match="while waiting"):
        match_reply(Response.success(4), 5)
