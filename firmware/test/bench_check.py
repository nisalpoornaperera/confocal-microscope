"""First power-on check of the flashed Uno: compare the real board with the emulator.

Opens the serial port (which resets the Uno), waits for the BOOT banner and
then sends a fixed list of requests - valid ones, every kind of malformed
line, out-of-travel moves - comparing every reply **byte for byte** with what
``FirmwareEmulator`` (the executable specification) answers for the same
lines. With ``--moves`` it also runs a few small real moves (DONE, STOP ->
ABORTED, HOME, RELEASE) and checks them semantically, because their exact
positions depend on timing.

Run on the Pi (or the PC) with the confocal server stopped, from the backend
environment so ``confocal`` and ``pyserial`` are available::

    uv run --directory backend python ../firmware/test/bench_check.py --port /dev/ttyACM0
    uv run --directory backend python ../firmware/test/bench_check.py --port COM5 --moves

Keep the motor switch OFF for the first run (the ULN2003 boards then have no
power, so nothing can move even if the firmware's travel limits were wrong).
The travel limits expected from the firmware come from the Pi configuration
(``--config``, else ``$CONFOCAL_CONFIG``, else the built-in defaults): an
``E_RANGE`` mismatch means ``firmware/confocal_stage/config.h`` and
``[arduino.a|b|c]`` disagree. Any unexpected ``OK`` to an out-of-range move
is followed by an immediate ``STOP``.

``--self-test`` runs the whole check against an in-process emulator instead of
a serial port (used to test this script; no hardware involved).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from confocal.config import ArduinoConfig, load_settings
from confocal.hardware.arduino.emulator import FirmwareEmulator
from confocal.hardware.arduino.protocol import (
    MAX_LINE_BYTES,
    Command,
    Event,
    EventKind,
    FirmwareStatus,
    Response,
    crc8,
    encode_command,
    parse_line,
)

REPLY_TIMEOUT_S = 1.0
QUIET_S = 0.15


class Link(Protocol):
    def write(self, data: bytes) -> None: ...
    def readline(self, timeout_s: float) -> bytes | None: ...
    def close(self) -> None: ...


class SerialLink:
    """The real board over pyserial (opening the port resets the Uno)."""

    def __init__(self, port: str, baudrate: int) -> None:
        import serial

        self._port = serial.Serial(port, baudrate=baudrate, timeout=0.05)

    def write(self, data: bytes) -> None:
        self._port.write(data)
        self._port.flush()

    def readline(self, timeout_s: float) -> bytes | None:
        deadline = time.monotonic() + timeout_s
        buffer = b""
        while time.monotonic() < deadline:
            buffer += self._port.readline()
            if buffer.endswith(b"\n"):
                return buffer
        return buffer or None

    def close(self) -> None:
        self._port.close()


class EmulatorLink:
    """Stand-in for the board (``--self-test``): a second emulator in real time."""

    def __init__(self, arduino: ArduinoConfig) -> None:
        self._emulator = FirmwareEmulator.from_config(arduino)
        self._pending: deque[bytes] = deque(self._emulator.reset())
        self._last = time.monotonic()

    def _tick(self) -> None:
        now = time.monotonic()
        self._pending.extend(self._emulator.advance(now - self._last))
        self._last = now

    def write(self, data: bytes) -> None:
        self._tick()
        self._pending.extend(self._emulator.handle_line(data))

    def readline(self, timeout_s: float) -> bytes | None:
        deadline = time.monotonic() + timeout_s
        while True:
            self._tick()
            if self._pending:
                return self._pending.popleft()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)

    def close(self) -> None:
        pass


class Checker:
    def __init__(self, link: Link, reference: FirmwareEmulator) -> None:
        self.link = link
        self.reference = reference
        self.failures = 0
        self.passed = 0
        self.seq = 0

    def result(self, ok: bool, what: str, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  ok    {what}")
        else:
            self.failures += 1
            print(f"  FAIL  {what}{': ' + detail if detail else ''}")
        return ok

    def read_lines(self, count: int, timeout_s: float = REPLY_TIMEOUT_S) -> list[bytes]:
        lines: list[bytes] = []
        while len(lines) < count:
            line = self.link.readline(timeout_s)
            if line is None:
                break
            lines.append(line)
        return lines

    def drain(self) -> list[bytes]:
        extra: list[bytes] = []
        while (line := self.link.readline(QUIET_S)) is not None:
            extra.append(line)
        return extra

    def next_seq(self) -> int:
        self.seq = 1 if self.seq >= 65535 else self.seq + 1
        return self.seq

    # ------------------------------------------------------------------ checks
    def boot(self, timeout_s: float) -> bool:
        expected = self.reference.reset()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self.link.readline(max(0.05, deadline - time.monotonic()))
            if line is not None and line.startswith(b"! BOOT"):
                return self.result(
                    line == expected[0], f"BOOT banner {line!r}", f"expected {expected[0]!r}"
                )
        return self.result(
            False, "BOOT banner", f"none within {timeout_s} s (port, cable, firmware?)"
        )

    def exact(self, line: bytes, what: str) -> None:
        """Send ``line``; the board's replies must equal the emulator's, byte for byte."""
        expected = self.reference.handle_line(line)
        self.link.write(line)
        actual = self.read_lines(len(expected))
        actual += self.drain()
        ok = actual == expected
        if not ok and any(a.split(b" ")[1:2] == [b"OK"] for a in actual) and b" MOVE " in line:
            self.link.write(encode_command(self.next_seq(), Command.STOP))  # never let it run
            self.drain()
        self.result(ok, what, f"expected {expected!r}, got {actual!r}")

    def request(self, command: Command, *args: int) -> tuple[int, list[Response | Event]]:
        seq = self.next_seq()
        self.link.write(encode_command(seq, command, args))
        messages: list[Response | Event] = []
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        while time.monotonic() < deadline:
            line = self.link.readline(max(0.05, deadline - time.monotonic()))
            if line is None:
                break
            message = parse_line(line)
            messages.append(message)
            if isinstance(message, Response):
                break
        return seq, messages

    def wait_event(self, seq: int, timeout_s: float) -> Event | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self.link.readline(max(0.05, deadline - time.monotonic()))
            if line is None:
                continue
            message = parse_line(line)
            if isinstance(message, Event) and message.completes(seq):
                return message
        return None


def protocol_checks(c: Checker, arduino: ArduinoConfig) -> None:
    w: Callable[[bytes], bytes] = lambda body: body + f"*{crc8(body):02X}\n".encode()  # noqa: E731
    print("protocol (exact bytes):")
    c.exact(encode_command(c.next_seq(), Command.PING), "PING")
    c.exact(encode_command(c.next_seq(), Command.STATUS), "STATUS after boot")
    c.exact(encode_command(c.next_seq(), Command.GETPOS), "GETPOS")
    c.exact(encode_command(c.next_seq(), Command.STOP), "STOP while idle")
    c.exact(encode_command(c.next_seq(), Command.ZERO), "ZERO")
    c.exact(encode_command(c.next_seq(), Command.HOME), "HOME at the origin (OK + DONE at once)")
    c.exact(encode_command(c.next_seq(), Command.MOVE, (0, 0, 0)), "zero-length MOVE")
    c.exact(encode_command(c.next_seq(), Command.RELEASE), "RELEASE")
    c.exact(encode_command(c.next_seq(), Command.STATUS), "STATUS after RELEASE")
    for index, motor in enumerate((arduino.a, arduino.b, arduino.c)):
        for value, side in ((motor.max_steps + 1, "max"), (motor.min_steps - 1, "min")):
            target = [0, 0, 0]
            target[index] = value
            name = "abc"[index]
            c.exact(
                encode_command(c.next_seq(), Command.MOVE, target),
                f"E_RANGE just beyond {side}_steps of motor {name} ({value})",
            )
    malformed = [
        (b"\n", "blank line"),
        (b"1 PING\n", "no checksum"),
        (b"1 PING*00\n", "wrong checksum"),
        (w(b"1 PING")[:-1] + b"\r\n", "CRLF terminator"),
        (b"1 PING*" + f"{crc8(b'1 PING'):02x}".encode() + b"\n", "lower-case checksum"),
        (b"*00\n", "empty message"),
        (w(b"1  PING"), "double space"),
        (w(b"0 PING"), "sequence number 0"),
        (w(b"65536 PING"), "sequence number 65536"),
        (w(b"7 ping"), "lower-case command"),
        (w(b"7 FOO"), "unknown command"),
        (w(b"7 PING 1"), "extra argument"),
        (w(b"7 MOVE 1 2"), "missing argument"),
        (w(b"7 MOVE 1 2 12345678"), "8-digit argument"),
        (w(b"7 MOVE 1 2 x"), "non-numeric argument"),
        (w(b"7 PI\x80NG"), "non-ASCII byte"),
        (b"A" * MAX_LINE_BYTES + b"\n", f"line over {MAX_LINE_BYTES} bytes"),
        (b"\x00" * 200 + b"\n", "200 NUL bytes"),
    ]
    for line, what in malformed:
        c.exact(line, what)
    c.exact(encode_command(c.next_seq(), Command.PING), "still in sync afterwards")


def move_checks(c: Checker, arduino: ArduinoConfig) -> None:
    speed = min(
        arduino.a.max_speed_steps_s, arduino.b.max_speed_steps_s, arduino.c.max_speed_steps_s
    )
    print("moves (motors turn if powered):")
    seq, messages = c.request(Command.MOVE, 64, -64, 32)
    ok = any(isinstance(m, Response) and m.ok for m in messages)
    c.result(ok, "MOVE 64 -64 32 acknowledged", repr(messages))
    done = c.wait_event(seq, 64 / speed * 2 + 2.0)
    c.result(
        done is not None and done.kind is EventKind.DONE and done.steps == (64, -64, 32),
        "DONE at 64 -64 32",
        repr(done),
    )
    _, messages = c.request(Command.GETPOS)
    reply = next((m for m in messages if isinstance(m, Response)), None)
    c.result(reply is not None and reply.steps() == (64, -64, 32), "GETPOS 64 -64 32", repr(reply))

    distance = int(speed * 4)  # ~4 s
    seq, _ = c.request(Command.MOVE, distance, distance, distance)
    time.sleep(1.0)
    _, messages = c.request(Command.STOP)
    aborted = next((m for m in messages if isinstance(m, Event)), None)
    reply = next((m for m in messages if isinstance(m, Response)), None)
    ok = (
        aborted is not None
        and aborted.kind is EventKind.ABORTED
        and aborted.seq == seq
        and reply is not None
        and aborted.steps == reply.steps()
        and 0 < reply.steps()[0] < distance
    )
    c.result(
        ok, "STOP pre-empts a move: ABORTED then OK at the same partial position", repr(messages)
    )
    _, messages = c.request(Command.STATUS)
    status = next((m for m in messages if isinstance(m, Response)), None)
    c.result(
        status is not None and FirmwareStatus.from_response(status).state.value == "stopped",
        "STATUS state=stopped",
        repr(status),
    )
    seq, _ = c.request(Command.HOME)
    done = c.wait_event(seq, distance / speed * 2 + 2.0)
    c.result(done is not None and done.steps == (0, 0, 0), "HOME: DONE 0 0 0", repr(done))
    _, messages = c.request(Command.RELEASE)
    _, messages = c.request(Command.STATUS)
    status = next((m for m in messages if isinstance(m, Response)), None)
    c.result(
        status is not None and not FirmwareStatus.from_response(status).coils_enabled,
        "RELEASE: enabled=0",
        repr(status),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--port", help="serial port of the Uno, e.g. /dev/ttyACM0 or COM5")
    target.add_argument(
        "--self-test", action="store_true", help="check against an in-process emulator"
    )
    parser.add_argument(
        "--config", type=Path, help="Pi configuration (default: $CONFOCAL_CONFIG or built-in)"
    )
    parser.add_argument("--moves", action="store_true", help="also run small real moves")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    arduino = settings.arduino
    link: Link = (
        EmulatorLink(arduino) if args.self_test else SerialLink(args.port, arduino.baudrate)
    )
    checker = Checker(link, FirmwareEmulator.from_config(arduino))
    try:
        print("boot:")
        if checker.boot(arduino.handshake_timeout_s):
            protocol_checks(checker, arduino)
            if args.moves:
                move_checks(checker, arduino)
    finally:
        link.close()
    print(f"{checker.passed} passed, {checker.failures} failed")
    return 0 if checker.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
