"""Generate firmware test vectors from the Python reference implementation.

The firmware must be byte-for-byte compatible with
``backend/confocal/hardware/arduino/protocol.py`` and behave exactly like
``backend/confocal/hardware/arduino/emulator.py`` (``FirmwareEmulator``). This
script records what the reference produces - CRC values, Bresenham positions,
tick intervals and complete serial sessions (input lines, elapsed time and
every reply / event byte) - into ``firmware/test/vectors.txt``, which the
native C++ test (``test_firmware.cpp``) replays against the firmware sources.

Run from the backend environment (it imports ``confocal``)::

    uv run --directory backend python ../firmware/test/gen_vectors.py

The output is deterministic (fixed random seeds), so it only changes when the
reference implementation changes.

File format (one record per line, ``#`` starts a comment)::

    CRC <data hex|-> <crc hex>
    BRES <sa> <sb> <sc> <ta> <tb> <tc> <k> <a> <b> <c>   position after k ticks
    PERIOD <da> <db> <dc> <va> <vb> <vc> <ideal interval us>
    SCENARIO <name>
    LIMITS <min> <max> <speed>  x3 (motors a, b, c)
    BOOT | IN <hex> | ADV <microseconds>                  actions, each followed by
    OUT <hex>                                            ... the lines it emits
    END
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Iterator
from pathlib import Path

from confocal.config import MotorConfig
from confocal.hardware.arduino.emulator import FirmwareEmulator
from confocal.hardware.arduino.protocol import (
    MAX_LINE_BYTES,
    STEP_LIMIT,
    Command,
    crc8,
    encode_command,
    frame,
)
from confocal.hardware.kinematics import MOTORS

OUTPUT = Path(__file__).resolve().parent / "vectors.txt"

#: The C++ test polls the firmware every POLL_US microseconds of simulated time.
POLL_US = 20
#: Keep every observation at least this far (us) from a tick boundary, plus a
#: per-tick allowance for the firmware's 1/256 us interval rounding and float math.
BOUNDARY_MARGIN_US = 40.0
PER_TICK_MARGIN_US = 0.01


Triple = tuple[int, int, int]


def _random_triple(rng: random.Random, low: int, high: int) -> Triple:
    return (rng.randint(low, high), rng.randint(low, high), rng.randint(low, high))


def _hex(data: bytes) -> str:
    return data.hex().upper() if data else "-"


def _with_crc(body: bytes) -> bytes:
    """``body*CRC\\n`` for an arbitrary byte body (bypasses the codec's own checks)."""
    return body + b"*%02X\n" % crc8(body)


class Session:
    """Drives one FirmwareEmulator and records the scenario."""

    def __init__(self, name: str, limits: list[tuple[int, int, float]]) -> None:
        self.motors = {
            motor: MotorConfig(min_steps=lo, max_steps=hi, max_speed_steps_s=speed)
            for motor, (lo, hi, speed) in zip(MOTORS, limits, strict=True)
        }
        self.limits = limits
        self.emulator = FirmwareEmulator(self.motors)
        self.lines = [
            f"SCENARIO {name}",
            "LIMITS " + " ".join(f"{lo} {hi} {sp!r}" for lo, hi, sp in limits),
        ]
        self.seq = 0

    def _record(self, action: str, outputs: list[bytes]) -> list[bytes]:
        self.lines.append(action)
        self.lines.extend(f"OUT {_hex(line)}" for line in outputs)
        return outputs

    def boot(self) -> list[bytes]:
        return self._record("BOOT", self.emulator.reset())

    def raw(self, line: bytes) -> list[bytes]:
        assert line.endswith(b"\n"), line
        assert b"\n" not in line[:-1], line
        return self._record(f"IN {_hex(line)}", self.emulator.handle_line(line))

    def next_seq(self) -> int:
        self.seq = 1 if self.seq >= 65535 else self.seq + 1
        return self.seq

    def send(self, command: Command, *args: int) -> list[bytes]:
        return self.raw(encode_command(self.next_seq(), command, args))

    def _safe_dt_us(self, dt_us: int) -> int:
        """Nudge ``dt_us`` so the observation is not ambiguous near a tick boundary."""
        move = self.emulator._move
        if move is None:
            return dt_us
        period_us = 1e6 / move.ticks_per_s
        for _ in range(100):
            ticks = (move.elapsed_s + dt_us / 1e6) * move.ticks_per_s
            if ticks >= move.ticks_total:  # only the end of the move matters
                distance_us = (ticks - move.ticks_total) * period_us
            else:
                frac = ticks - math.floor(ticks)
                distance_us = min(frac, 1.0 - frac) * period_us
            margin = BOUNDARY_MARGIN_US + PER_TICK_MARGIN_US * min(ticks, move.ticks_total)
            if distance_us >= margin:
                return dt_us
            dt_us += int(2 * margin) + 1
        raise AssertionError("could not find a safe observation time")

    def advance(self, dt_us: int) -> list[bytes]:
        dt_us = self._safe_dt_us(max(0, int(dt_us)))
        return self._record(f"ADV {dt_us}", self.emulator.advance(dt_us / 1e6))

    def finish_move(self) -> list[bytes]:
        """Advance until the running move (if any) has finished."""
        move = self.emulator._move
        if move is None:
            return []
        remaining_s = move.ticks_total / move.ticks_per_s - move.elapsed_s
        return self.advance(int(remaining_s * 1e6) + 1000)  # _safe_dt_us adds drift margin

    def end(self) -> list[str]:
        return [*self.lines, "END"]


# --------------------------------------------------------------------------- unit vectors


def crc_vectors(rng: random.Random) -> Iterator[str]:
    fixed = [b"", b"123456789", b"1 PING", b"! BOOT 1.0.0", b"\x00", b"\xff" * 10]
    for data in fixed:
        yield f"CRC {_hex(data)} {crc8(data):02X}"
    for _ in range(200):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 95)))
        yield f"CRC {_hex(data)} {crc8(data):02X}"


def bresenham_vectors(rng: random.Random) -> Iterator[str]:
    from confocal.hardware.arduino.emulator import _Move

    def record(
        start: tuple[int, int, int], target: tuple[int, int, int], ks: list[int]
    ) -> Iterator[str]:
        delta = (target[0] - start[0], target[1] - start[1], target[2] - start[2])
        n = max(abs(d) for d in delta)
        move = _Move(seq=1, start=start, delta=delta, ticks_total=n, ticks_per_s=1.0)
        ends = " ".join(map(str, (*start, *target)))
        for k in sorted({min(n, max(0, k)) for k in ks}):
            a, b, c = move.position_after(k)
            yield f"BRES {ends} {k} {a} {b} {c}"

    cases: list[tuple[Triple, Triple]] = [
        ((0, 0, 0), (1200, -600, 300)),
        ((0, 0, 0), (7, 3, -2)),
        ((5, 5, 5), (5, 6, 5)),
        ((0, 0, 0), (1, 1, 1)),
        ((10, -10, 0), (-10, 10, 1)),
        ((-STEP_LIMIT, STEP_LIMIT, 0), (STEP_LIMIT, -STEP_LIMIT, 1)),
    ]
    for _ in range(60):
        span = rng.choice([3, 10, 100, 1000, 20000])
        cases.append((_random_triple(rng, -span, span), _random_triple(rng, -span, span)))
    for start, target in cases:
        n = max(abs(t - s) for s, t in zip(start, target, strict=True))
        if n == 0:
            continue
        ks = [0, 1, 2, n // 3, n // 2, n - 1, n, *(rng.randint(0, n) for _ in range(5))]
        yield from record(start, target, ks)


def period_vectors(rng: random.Random) -> Iterator[str]:
    cases: list[tuple[Triple, tuple[float, float, float]]] = [
        ((1200, 600, 300), (600.0, 600.0, 600.0)),
        ((100, 100, 100), (600.0, 450.0, 1000.0)),
    ]
    choices = [1.0, 37.5, 300.0, 500.0, 600.0, 999.9, 2000.0]
    for _ in range(60):
        distance = _random_triple(rng, 0, 50000)
        if max(distance) == 0:
            continue
        cases.append((distance, (rng.choice(choices), rng.choice(choices), rng.choice(choices))))
    for distance, speeds in cases:
        n = max(distance)
        duration = max(d / v for d, v in zip(distance, speeds, strict=True))
        ideal_us = duration * 1e6 / n
        yield (
            "PERIOD "
            + " ".join(map(str, distance))
            + " "
            + " ".join(map(repr, speeds))
            + f" {ideal_us!r}"
        )


# --------------------------------------------------------------------------- scenarios


def docs_example() -> list[str]:
    """The example session of docs/serial-protocol.md §8."""
    s = Session("docs_example", [(-200000, 200000, 600.0)] * 3)
    s.boot()
    s.send(Command.PING)
    s.send(Command.STATUS)
    s.send(Command.MOVE, 1200, -600, 300)
    s.advance(1_000_000)
    s.send(Command.GETPOS)
    s.send(Command.STOP)
    s.send(Command.MOVE, 0, 0, 0)
    s.send(Command.MOVE, 1, 1, 1)
    s.finish_move()
    s.send(Command.MOVE, 300000, 0, 0)
    s.send(Command.RELEASE)
    good = encode_command(s.next_seq(), Command.MOVE, (1, 2, 3))
    s.raw(good[:-3] + b"00\n")  # corrupt checksum
    return s.end()


def emulator_unit_cases() -> list[str]:
    """Mirrors backend/tests/unit/hardware/test_emulator.py (speed 500, travel +/-1000)."""
    s = Session("emulator_unit_cases", [(-1000, 1000, 500.0)] * 3)
    s.boot()
    s.send(Command.MOVE, 10, 10, 10)
    s.advance(1_000_000)
    s.boot()  # reset zeroes everything
    s.send(Command.STATUS)
    s.send(Command.MOVE, 0, 0, 0)  # zero-length: OK + DONE at once
    s.send(Command.MOVE, 500, -250, 100)
    for _ in range(10):
        s.advance(97_000)
        s.send(Command.GETPOS)
    s.send(Command.STATUS)
    for command, args in [
        (Command.MOVE, (1, 2, 3)),
        (Command.HOME, ()),
        (Command.ZERO, ()),
        (Command.RELEASE, ()),
    ]:
        s.send(command, *args)  # E_BUSY
    for command in (Command.GETPOS, Command.STATUS, Command.PING):
        s.send(command)
    s.send(Command.STOP)  # ABORTED, then OK
    s.send(Command.STOP)  # idle: just the position
    s.send(Command.STATUS)  # state=stopped
    s.send(Command.MOVE, 1001, 0, 0)  # E_RANGE
    s.send(Command.MOVE, 0, -1001, 0)
    s.send(Command.MOVE, 0, 0, 1001)
    s.send(Command.MOVE, 1000, -1000, 1000)  # edges are inside
    s.finish_move()
    s.send(Command.HOME)
    s.advance(500_000)
    s.send(Command.STATUS)  # homing
    s.finish_move()
    s.send(Command.STATUS)
    s.send(Command.MOVE, 7, -3, 2)
    s.finish_move()
    s.send(Command.ZERO)
    s.send(Command.GETPOS)
    s.send(Command.MOVE, 3, 3, 3)
    s.finish_move()
    s.send(Command.RELEASE)
    s.send(Command.STATUS)
    s.send(Command.MOVE, 4, 4, 4)  # re-energises
    s.send(Command.STATUS)
    s.finish_move()
    return s.end()


def slow_motor_case() -> list[str]:
    """Per-motor speed limits: the slowest motor relative to its distance sets the pace."""
    s = Session(
        "per_motor_speeds", [(-5000, 5000, 600.0), (-3000, 4000, 150.0), (-5000, 100, 2000.0)]
    )
    s.boot()
    s.send(Command.MOVE, 600, 300, -1)
    for _ in range(12):
        s.advance(170_000)
        s.send(Command.GETPOS)
    s.finish_move()
    s.send(Command.MOVE, -600, 4000, 100)
    s.advance(3_333_333)
    s.send(Command.STOP)
    s.send(Command.MOVE, 0, 4001, 0)
    s.send(Command.MOVE, -5001, 0, 0)
    s.send(Command.MOVE, 0, 0, 101)
    s.send(Command.HOME)
    s.finish_move()
    return s.end()


def large_values_case() -> list[str]:
    """Full protocol range: 7-digit positions and the longest STATUS line."""
    lim = STEP_LIMIT
    s = Session("large_values", [(-lim, lim, 2000.0)] * 3)
    s.boot()
    s.send(Command.MOVE, -1000, 1000, -999)
    s.finish_move()
    s.send(Command.ZERO)
    s.send(Command.MOVE, -2000, 2000, -1999)
    s.advance(400_000)
    s.send(Command.STOP)
    s.send(Command.STATUS)
    s.send(Command.HOME)
    s.finish_move()
    s.raw(_with_crc(b"8 MOVE 9999999 -9999999 0000000"))  # the very edge of the range
    s.finish_move()
    s.raw(_with_crc(b"9 STATUS"))
    s.raw(_with_crc(b"10 MOVE -9999999 9999999 -9999999"))
    s.advance(1_000_000)
    s.seq = 65533
    s.send(Command.STOP)
    s.send(Command.STATUS)  # "stopped" with three 7-digit values: the longest reply
    for _ in range(4):  # sequence numbers roll over 65535 -> 1
        s.send(Command.PING)
    s.raw(_with_crc(b"65535 STATUS"))
    s.raw(_with_crc(b"65536 STATUS"))
    s.raw(_with_crc(b"99999 PING"))
    s.raw(_with_crc(b"100000 PING"))
    s.raw(_with_crc(b"0 PING"))
    s.raw(_with_crc(b"00000 PING"))
    s.raw(_with_crc(b"00007 PING"))
    s.raw(_with_crc(b"11 MOVE 10000000 0 0"))  # 8 digits: E_ARGS
    s.raw(_with_crc(b"12 MOVE -10000000 0 0"))
    s.raw(_with_crc(b"13 MOVE -0 -0000000 0000000"))
    s.finish_move()
    s.raw(_with_crc(b"14 STATUS"))
    return s.end()


def malformed_lines_case() -> list[str]:
    """Every framing / syntax error path, with a valid checksum where it matters."""
    s = Session("malformed", [(-1000, 1000, 600.0)] * 3)
    s.boot()
    w = _with_crc
    good = encode_command(1, Command.PING)
    lines = [
        b"\n",  # blank line: no checksum
        b"\r\n",
        b"*00\n",  # empty message with a valid checksum
        b"*00\r\n",
        good,
        good[:-1] + b"\r\n",  # CRLF accepted
        good[:-1] + b"\r\r\n",
        b"1 PING\n",  # no checksum
        b"1 PING*\n",
        b"1 PING*9\n",
        b"1 PING*927\n",
        b"1 PING*G2\n",
        good.lower(),  # lower-case command: the checksum no longer matches
        b"1 PING*" + f"{crc8(b'1 PING'):02x}".encode() + b"\n",  # lower-case hex accepted
        b"1 PIN*" + f"{crc8(b'1 PING'):02X}".encode() + b"\n",  # corrupt body
        b"2 PING*" + f"{crc8(b'1 PING'):02X}".encode() + b"\n",
        w(b" 1 PING"),
        w(b"1 PING "),
        w(b"1  PING"),
        w(b"1\tPING"),
        w(b"1 PI\x00NG"),
        w(b"1 P\xc3\xa9NG"),  # non-ASCII
        w(b"1 PI*NG"),  # '*' inside the body
        w(b"1 PI\rNG"),
        w(b"1 PI\x7fNG"),
        w(b"1"),  # missing command
        w(b"12345"),
        w(b"x PING"),
        w(b"-1 PING"),
        w(b"+1 PING"),
        w(b"1 ping"),
        w(b"1 PINGS"),
        w(b"1 PIN"),
        w(b"1 MOVEX 1 2 3"),
        w(b"1 FOO"),
        w(b"1 PING 1"),
        w(b"1 STATUS x"),
        w(b"1 MOVE"),
        w(b"1 MOVE 1 2"),
        w(b"1 MOVE 1 2 3 4"),
        w(b"1 MOVE 1 2 x"),
        w(b"1 MOVE 1 2 -"),
        w(b"1 MOVE 1 2 --3"),
        w(b"1 MOVE 1 2 +3"),
        w(b"1 MOVE 1 2 3-"),
        w(b"1 MOVE 1 2 0x3"),
        w(b"1 MOVE 1 2 3.0"),
        w(b"1 MOVE 1 2 12345678"),
        w(b"1 MOVE 1 2 1234567"),  # E_RANGE (7 digits, outside travel)
        w(b"1 MOVE 0001 -0002 0003"),  # leading zeros accepted
        w(b"1 HOME"),
        w(b"77 GETPOS"),
        w(b"78 STOP"),
        w(b"79 ZERO"),
        w(b"80 RELEASE"),
        w(b"81 STATUS"),
    ]
    for line in lines:
        s.raw(line)
        s.advance(1000)
    s.finish_move()
    # Length limit: 96 bytes including '\n' is the longest accepted line.
    for total in (95, 96, 97, 98, 150, 400):
        body_len = total - 4  # "*XX\n"
        body = b"5 PING" + b" 1" * ((body_len - 6) // 2)
        body = body + b"9" * (body_len - len(body))
        line = w(body)
        assert len(line) == total, (total, len(line))
        s.raw(line)
    s.raw(b"A" * 95 + b"\n")
    s.raw(b"A" * 96 + b"\n")
    s.raw(b"\x00" * 300 + b"\n")
    s.raw(bytes(range(1, 10)) + bytes(range(11, 256)) + b"\n")  # every byte except '\n'
    s.raw(w(b"6 PING"))  # still in sync afterwards
    return s.end()


def random_case(seed: int) -> list[str]:
    """Randomised sessions: valid and broken requests interleaved with time."""
    rng = random.Random(seed)
    lo = -rng.choice([50, 400, 3000])
    hi = rng.choice([60, 500, 2500])
    speeds = [rng.choice([300.0, 450.0, 600.0, 1000.0, 2000.0]) for _ in range(3)]
    s = Session(f"random_{seed}", [(lo, hi, sp) for sp in speeds])
    s.boot()
    commands = list(Command)
    for _ in range(250):
        roll = rng.random()
        if roll < 0.35:
            target = [
                rng.randint(lo - 10, hi + 10) if rng.random() < 0.1 else rng.randint(lo, hi)
                for _ in range(3)
            ]
            if rng.random() < 0.2:  # pure Z-like move: all motors together
                target = [target[0]] * 3
            s.send(Command.MOVE, *target)
        elif roll < 0.6:
            command = rng.choice(commands)
            if command is Command.MOVE:
                continue
            s.send(command)
        elif roll < 0.7:
            text = f"{s.next_seq()} {rng.choice(commands).value}"
            if rng.random() < 0.5:
                text += " " + " ".join(str(rng.randint(lo, hi)) for _ in range(3))
            body = _with_crc(text.encode())[:-1]  # framed line minus its terminator, then mutated
            mutation = rng.random()
            data = bytearray(body)
            if mutation < 0.4:
                data[rng.randrange(len(data))] = rng.randrange(256)
                data = bytearray(b if b != 0x0A else 0x0B for b in data)
            elif mutation < 0.7:
                del data[rng.randrange(len(data))]
            else:
                data.insert(rng.randrange(len(data) + 1), rng.choice(b" *\r0-A\x80"))
            s.raw(bytes(data) + b"\n")
        else:
            s.advance(rng.choice([0, 300, 5_000, 50_000, 250_000, 1_000_000, 3_000_000]))
    s.send(Command.STOP)
    s.send(Command.STATUS)
    return s.end()


def main() -> int:
    rng = random.Random(20261002)
    out: list[str] = [
        "# Generated by firmware/test/gen_vectors.py from the Python reference",
        "# (confocal.hardware.arduino.protocol / emulator). Do not edit by hand.",
        f"# MAX_LINE_BYTES={MAX_LINE_BYTES} STEP_LIMIT={STEP_LIMIT} POLL_US={POLL_US}",
    ]
    out += crc_vectors(rng)
    out += bresenham_vectors(rng)
    out += period_vectors(rng)
    out += docs_example()
    out += emulator_unit_cases()
    out += slow_motor_case()
    out += large_values_case()
    out += malformed_lines_case()
    for seed in range(1, 13):
        out += random_case(seed)
    # Sanity: the codec's own framing agrees with the helper used above.
    assert frame("1 PING") == _with_crc(b"1 PING")
    OUTPUT.write_text("\n".join(out) + "\n", encoding="ascii", newline="\n")
    scenarios = sum(1 for line in out if line.startswith("SCENARIO"))
    print(f"wrote {OUTPUT} ({len(out)} records, {scenarios} scenarios)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
