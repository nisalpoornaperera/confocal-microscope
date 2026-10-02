"""SerialTransport (with a fake pyserial port) and EmulatorTransport."""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

import pytest

from confocal.config import MotorConfig
from confocal.errors import CommunicationError, ProtocolError
from confocal.hardware.arduino.emulator import FIRMWARE_VERSION, FirmwareEmulator
from confocal.hardware.arduino.protocol import (
    MAX_LINE_BYTES,
    Command,
    Event,
    Response,
    encode_command,
    parse_line,
)
from confocal.hardware.arduino.transport import (
    MAX_BUFFERED_BYTES,
    EmulatorTransport,
    ManualClock,
    ScaledClock,
    SerialTransport,
    Transport,
    TransportClosedError,
    TransportTimeoutError,
    _LineAssembler,
)
from confocal.hardware.kinematics import Motor

# --------------------------------------------------------------------------- line assembly


def test_assembler_splits_lines_across_chunks() -> None:
    assembler = _LineAssembler()
    assert assembler.feed(b"1 OK*92\n2 O") == [b"1 OK*92\n"]
    assert assembler.feed(b"K*") == []
    assert assembler.feed(b"AA\r\n! BOOT 1*00\n") == [b"2 OK*AA\r\n", b"! BOOT 1*00\n"]


def test_assembler_bounds_an_unterminated_line() -> None:
    assembler = _LineAssembler()
    lines = assembler.feed(b"x" * (MAX_BUFFERED_BYTES + 50) + b"\n1 OK*92\n")
    assert len(lines) == 2
    assert len(lines[0]) == MAX_BUFFERED_BYTES + 1
    assert lines[1] == b"1 OK*92\n"  # the rest of the noise was discarded
    with pytest.raises(ProtocolError):
        parse_line(lines[0])


# --------------------------------------------------------------------------- serial


class FakeSerial:
    """Enough of ``serial.Serial`` for SerialTransport, fed by the test."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.timeout = float(kwargs["timeout"])
        self.incoming: queue.Queue[bytes | Exception] = queue.Queue()
        self.written = bytearray()
        self.dtr_history: list[bool] = []
        self.input_resets = 0
        self.flushes = 0
        self.closed = False
        self.read_threads: set[str] = set()

    @property
    def dtr(self) -> bool:
        return self.dtr_history[-1] if self.dtr_history else True

    @dtr.setter
    def dtr(self, value: bool) -> None:
        self.dtr_history.append(value)

    @property
    def in_waiting(self) -> int:
        return 0

    def read(self, size: int = 1) -> bytes:
        del size
        self.read_threads.add(threading.current_thread().name)
        try:
            item = self.incoming.get(timeout=self.timeout)
        except queue.Empty:
            return b""
        if isinstance(item, Exception):
            raise item
        return item

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def flush(self) -> None:
        self.flushes += 1

    def reset_input_buffer(self) -> None:
        self.input_resets += 1

    def cancel_read(self) -> None:
        self.incoming.put(b"")

    def close(self) -> None:
        self.closed = True


class SerialRig:
    def __init__(self, *, reset_on_open: bool = True) -> None:
        self.port: FakeSerial | None = None
        self.transport = SerialTransport(
            "/dev/ttyACM0", 115200, reset_on_open=reset_on_open, serial_factory=self.factory
        )

    def factory(self, **kwargs: Any) -> Any:
        self.port = FakeSerial(**kwargs)
        return self.port

    @property
    def fake(self) -> FakeSerial:
        assert self.port is not None
        return self.port


async def test_serial_open_configures_8n1_exclusive_and_pulses_dtr() -> None:
    rig = SerialRig()
    assert isinstance(rig.transport, Transport)
    await rig.transport.open()
    try:
        kwargs = rig.fake.kwargs
        assert kwargs["port"] == "/dev/ttyACM0"
        assert kwargs["baudrate"] == 115200
        assert (kwargs["bytesize"], kwargs["parity"], kwargs["stopbits"]) == (8, "N", 1)
        assert kwargs["exclusive"] is True
        assert not kwargs["rtscts"]
        assert not kwargs["xonxoff"]
        assert rig.fake.dtr_history == [False, True]  # the reset pulse
        assert rig.fake.input_resets == 1
        assert rig.transport.is_open
        assert "/dev/ttyACM0" in rig.transport.description
    finally:
        await rig.transport.close()


async def test_serial_open_without_reset() -> None:
    rig = SerialRig(reset_on_open=False)
    await rig.transport.open()
    await rig.transport.close()
    assert rig.fake.dtr_history == []


async def test_serial_reads_lines_from_the_reader_thread() -> None:
    rig = SerialRig()
    await rig.transport.open()
    try:
        rig.fake.incoming.put(b"! BOOT 1.0.0*D2\n1 O")
        rig.fake.incoming.put(b"K*92\n")
        assert await rig.transport.read_line(1.0) == b"! BOOT 1.0.0*D2\n"
        assert await rig.transport.read_line(1.0) == b"1 OK*92\n"
        assert rig.fake.read_threads == {"serial-reader /dev/ttyACM0"}
        with pytest.raises(TransportTimeoutError):
            await rig.transport.read_line(0.05)
    finally:
        await rig.transport.close()


async def test_serial_write_line() -> None:
    rig = SerialRig()
    await rig.transport.open()
    line = encode_command(1, Command.PING)
    await rig.transport.write_line(line)
    await rig.transport.close()
    assert bytes(rig.fake.written) == line
    assert rig.fake.flushes == 1
    with pytest.raises(TransportClosedError):
        await rig.transport.write_line(line)


async def test_serial_read_error_closes_the_link() -> None:
    rig = SerialRig()
    await rig.transport.open()
    try:
        rig.fake.incoming.put(OSError("device reports readiness to read but returned no data"))
        with pytest.raises(TransportClosedError, match="failed"):
            await rig.transport.read_line(2.0)
    finally:
        await rig.transport.close()


async def test_serial_close_wakes_a_pending_read_and_is_idempotent() -> None:
    rig = SerialRig()
    await rig.transport.open()
    pending = asyncio.create_task(rig.transport.read_line(None))
    await asyncio.sleep(0.01)
    await rig.transport.close()
    with pytest.raises(TransportClosedError):
        await pending
    assert rig.fake.closed
    assert not rig.transport.is_open
    await rig.transport.close()
    with pytest.raises(TransportClosedError):
        await rig.transport.read_line(0.1)


@pytest.mark.parametrize(
    ("error", "hint"),
    [
        (OSError("could not open port /dev/ttyACM0: No such file or directory"), "plugged in"),
        (PermissionError("Permission denied: '/dev/ttyACM0'"), "dialout"),
        (OSError("Could not exclusively lock port: Resource temporarily unavailable"), "stop it"),
    ],
)
async def test_serial_open_failures_carry_a_hint(error: Exception, hint: str) -> None:
    def failing(**kwargs: Any) -> Any:
        raise error

    transport = SerialTransport("/dev/ttyACM0", serial_factory=failing)
    with pytest.raises(CommunicationError, match=hint):
        await transport.open()
    assert not transport.is_open


# --------------------------------------------------------------------------- emulator


@pytest.fixture
def emulator() -> FirmwareEmulator:
    return FirmwareEmulator(dict.fromkeys(Motor, MotorConfig(max_speed_steps_s=100.0)))


async def test_emulator_open_resets_and_sends_boot(emulator: FirmwareEmulator) -> None:
    transport = EmulatorTransport(emulator, ManualClock())
    await transport.open()
    assert parse_line(await transport.read_line(0.1)) == Event.boot(FIRMWARE_VERSION)
    assert transport.opens == 1


async def test_manual_clock_controls_when_done_arrives(emulator: FirmwareEmulator) -> None:
    clock = ManualClock()
    transport = EmulatorTransport(emulator, clock)
    await transport.open()
    await transport.read_line(0.1)  # BOOT
    await transport.write_line(encode_command(1, Command.MOVE, (50, 0, 0)))
    assert parse_line(await transport.read_line(0.1)) == Response.success(1)
    with pytest.raises(TransportTimeoutError):
        await transport.read_line(0.02)  # no emulated time has passed
    clock.advance(0.25)
    assert emulator.position == (0, 0, 0)  # the emulator catches up lazily ...
    pending = asyncio.create_task(transport.read_line(0.5))
    clock.advance(0.25)
    assert parse_line(await pending) == Event.done(1, (50, 0, 0))  # ... and exactly
    with pytest.raises(ValueError, match="dt_s"):
        clock.advance(-1.0)


async def test_scaled_clock_runs_moves_in_real_time(emulator: FirmwareEmulator) -> None:
    transport = EmulatorTransport(emulator, ScaledClock(speed=1000.0))
    await transport.open()
    await transport.read_line(0.1)
    await transport.write_line(encode_command(2, Command.MOVE, (0, 100, 0)))
    await transport.read_line(0.1)  # OK
    assert parse_line(await transport.read_line(1.0)) == Event.done(2, (0, 100, 0))
    with pytest.raises(ValueError, match="speed"):
        ScaledClock(speed=0.0)


async def test_emulator_fault_injection(emulator: FirmwareEmulator) -> None:
    transport = EmulatorTransport(emulator, ManualClock())
    await transport.open()
    await transport.read_line(0.1)

    transport.corrupt_next()
    await transport.write_line(encode_command(1, Command.PING))
    with pytest.raises(ProtocolError):
        parse_line(await transport.read_line(0.1))

    transport.mute = True
    await transport.write_line(encode_command(2, Command.PING))
    with pytest.raises(TransportTimeoutError):
        await transport.read_line(0.02)
    transport.mute = False

    transport.inject(b"raw\n")
    assert await transport.read_line(0.1) == b"raw\n"

    transport.reset_device()
    assert parse_line(await transport.read_line(0.1)) == Event.boot(FIRMWARE_VERSION)

    transport.fail_link("cable pulled")
    assert not transport.is_open
    with pytest.raises(TransportClosedError, match="cable pulled"):
        await transport.read_line(0.1)
    with pytest.raises(TransportClosedError, match="cable pulled"):
        await transport.write_line(encode_command(3, Command.PING))
    await transport.close()
    await transport.open()  # a new open brings the link back
    assert transport.is_open
    assert transport.opens == 2
    assert len(transport.sent) == 2
    assert all(len(line) <= MAX_LINE_BYTES for line in transport.received)


async def test_emulator_open_failure_and_closed_reads(emulator: FirmwareEmulator) -> None:
    transport = EmulatorTransport(emulator)
    transport.fail_open = CommunicationError("no such port")
    with pytest.raises(CommunicationError, match="no such port"):
        await transport.open()
    transport.fail_open = None
    with pytest.raises(TransportClosedError):
        await transport.read_line(0.01)
    with pytest.raises(TransportClosedError):
        await transport.write_line(encode_command(1, Command.PING))
    await transport.open()
    await transport.read_line(0.1)  # BOOT
    pending = asyncio.create_task(transport.read_line(None))
    await asyncio.sleep(0.01)
    await transport.close()
    with pytest.raises(TransportClosedError):
        await pending
