"""HIL: protocol conformance of a flashed Uno - every command and every error code.

Error replies and idle commands are compared byte for byte with the
FirmwareEmulator, the executable specification of the firmware. Motion stays
within a few hundred steps of where the stage is when the test starts.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from tests.hil.helpers import RawLink, hil_settings, requires_arduino

from confocal.hardware.arduino.protocol import (
    MAX_LINE_BYTES,
    STEP_LIMIT,
    UNATTRIBUTED_SEQ,
    Command,
    ErrorCode,
    Event,
    EventKind,
    FirmwareState,
    FirmwareStatus,
    Response,
    crc8,
    encode_command,
    parse_line,
)

pytestmark = requires_arduino


@pytest.fixture
async def link() -> AsyncIterator[RawLink]:
    raw = RawLink(hil_settings())
    await raw.open()
    try:
        yield raw
    finally:
        # Leave the stage stopped and the coils off whatever happened.
        try:
            await raw.send(Command.STOP)
            await raw.send(Command.RELEASE)
        finally:
            await raw.close()


def _framed(body: bytes, *, lower: bool = False, crlf: bool = False) -> bytes:
    crc = b"%02x" % crc8(body) if lower else b"%02X" % crc8(body)
    return body + b"*" + crc + (b"\r\n" if crlf else b"\n")


async def _assert_like_emulator(link: RawLink, line: bytes) -> Response:
    """The device's reply to ``line`` equals the emulator's, byte for byte."""
    expected = link.emulator().handle_line(line)
    raw, events = await link.exchange_raw(line)
    assert events == []
    assert [raw] == expected, f"device {raw!r} != spec {expected!r} for {line!r}"
    reply = parse_line(raw)
    assert isinstance(reply, Response)
    return reply


async def test_boot_banner_ping_and_status(link: RawLink) -> None:
    assert link.boot is not None
    assert link.boot.kind is EventKind.BOOT
    reply, events = await link.send(Command.PING)
    assert reply.ok
    assert reply.payload == ()
    assert events == []
    reply, _ = await link.send(Command.STATUS)
    status = FirmwareStatus.from_response(reply)
    assert status.state is FirmwareState.IDLE
    assert status.steps == (0, 0, 0)  # counters restart at 0 after the reset
    assert not status.moving
    assert not status.coils_enabled  # coils off from the first instant after reset
    assert status.version == link.boot.version


async def test_getpos_zero_release(link: RawLink) -> None:
    reply, _ = await link.send(Command.GETPOS)
    assert reply.steps() == (0, 0, 0)
    reply, _ = await link.send(Command.ZERO)
    assert reply.steps() == (0, 0, 0)
    reply, _ = await link.send(Command.RELEASE)
    assert reply.ok


async def test_move_done_getpos_and_home(link: RawLink) -> None:
    target = (120, -80, 40)
    reply, _ = await link.send(Command.MOVE, *target)
    assert reply.ok
    move_seq = reply.seq
    event = await link.wait_event(link.move_duration_s(120) + 2.0)
    assert event == Event.done(move_seq, target)
    reply, _ = await link.send(Command.GETPOS)
    assert reply.steps() == target
    status = FirmwareStatus.from_response((await link.send(Command.STATUS))[0])
    assert status.coils_enabled
    assert status.state is FirmwareState.IDLE

    reply, events = await link.send(Command.MOVE, *target)  # zero-length: DONE at once
    assert reply.ok
    done = events or [await link.wait_event(1.0)]
    assert done == [Event.done(reply.seq, target)]

    reply, _ = await link.send(Command.HOME)
    assert reply.ok
    event = await link.wait_event(link.move_duration_s(120) + 2.0)
    assert event == Event.done(reply.seq, (0, 0, 0))


async def test_only_queries_and_stop_are_accepted_while_moving(link: RawLink) -> None:
    reply, _ = await link.send(Command.MOVE, 400, -400, 200)
    move_seq = reply.seq
    for command, args in (
        (Command.MOVE, (1, 1, 1)),
        (Command.HOME, ()),
        (Command.ZERO, ()),
        (Command.RELEASE, ()),
    ):
        busy, _ = await link.send(command, *args)
        assert not busy.ok
        assert busy.error is ErrorCode.E_BUSY, command
    status = FirmwareStatus.from_response((await link.send(Command.STATUS))[0])
    assert status.moving
    assert status.state is FirmwareState.MOVING
    assert (await link.send(Command.PING))[0].ok
    a, b, _ = (await link.send(Command.GETPOS))[0].steps()
    assert 0 <= a <= 400
    assert -400 <= b <= 0
    event = await link.wait_event(link.move_duration_s(400) + 2.0)
    assert event == Event.done(move_seq, (400, -400, 200))
    await link.send(Command.HOME)
    await link.wait_event(link.move_duration_s(400) + 2.0)


async def test_stop_pre_empts_a_move(link: RawLink) -> None:
    reply, _ = await link.send(Command.MOVE, 600, 600, 600)
    move_seq = reply.seq
    await asyncio.sleep(0.4 * link.move_duration_s(600))
    stop, events = await link.send(Command.STOP)
    assert stop.ok
    assert len(events) == 1
    aborted = events[0]
    assert aborted.kind is EventKind.ABORTED
    assert aborted.seq == move_seq
    assert aborted.steps == stop.steps()
    a, b, c = stop.steps()
    assert a == b == c  # a pure Z-like move stays on the line
    assert 0 < a < 600
    status = FirmwareStatus.from_response((await link.send(Command.STATUS))[0])
    assert status.state is FirmwareState.STOPPED
    assert status.coils_enabled  # STOP keeps the coils energised
    idle_stop, events = await link.send(Command.STOP)  # never fails when idle
    assert idle_stop.ok
    assert events == []
    assert idle_stop.steps() == stop.steps()
    await link.send(Command.HOME)
    assert (await link.wait_event(link.move_duration_s(600) + 2.0)).steps == (0, 0, 0)


async def test_range_error_moves_nothing(link: RawLink) -> None:
    beyond = link.settings.arduino.a.max_steps + 1
    reply, events = await link.send(Command.MOVE, beyond, 0, 0)
    assert reply.error is ErrorCode.E_RANGE
    assert events == []
    far = encode_command(link.seq.next(), Command.MOVE, (STEP_LIMIT, 0, 0))
    assert (await _assert_like_emulator(link, far)).error is ErrorCode.E_RANGE
    assert (await link.send(Command.GETPOS))[0].steps() == (0, 0, 0)


async def test_framing_and_syntax_error_codes(link: RawLink) -> None:
    seq = link.seq.next()
    # E_CRC: wrong checksum, missing checksum, non-hex checksum -> seq 0.
    good = encode_command(seq, Command.PING)
    bad_crc = good[:-3] + (b"00" if good[-3:-1] != b"00" else b"01") + b"\n"
    for line in (bad_crc, b"%d PING\n" % seq, b"%d PING*ZZ\n" % seq):
        reply = await _assert_like_emulator(link, line)
        assert reply.error is ErrorCode.E_CRC
        assert reply.seq == UNATTRIBUTED_SEQ
    # E_LENGTH: longer than 96 bytes -> seq 0.
    long_body = b"%d PING " % seq + b"X" * MAX_LINE_BYTES
    reply = await _assert_like_emulator(link, _framed(long_body))
    assert reply.error is ErrorCode.E_LENGTH
    # E_SYNTAX: unreadable sequence number (seq 0), double space, empty line.
    for body in (b"x1 PING", b"%d  PING" % seq, b"%d" % seq):
        reply = await _assert_like_emulator(link, _framed(body))
        assert reply.error is ErrorCode.E_SYNTAX
    # E_UNKNOWN and E_ARGS carry the request's own sequence number.
    reply = await _assert_like_emulator(link, _framed(b"%d FOO" % seq))
    assert (reply.error, reply.seq) == (ErrorCode.E_UNKNOWN, seq)
    for body in (b"%d MOVE 1 2" % seq, b"%d MOVE 1 2 99999999" % seq, b"%d PING 1" % seq):
        reply = await _assert_like_emulator(link, _framed(body))
        assert (reply.error, reply.seq) == (ErrorCode.E_ARGS, seq)


async def test_tolerated_variants(link: RawLink) -> None:
    seq = link.seq.next()
    reply = await _assert_like_emulator(link, _framed(b"%d PING" % seq, lower=True))
    assert reply.ok
    seq = link.seq.next()
    reply = await _assert_like_emulator(link, _framed(b"%d PING" % seq, crlf=True))
    assert reply.ok
    # Sequence numbers roll over at 65535.
    reply = await _assert_like_emulator(link, encode_command(65535, Command.PING))
    assert reply.seq == 65535
