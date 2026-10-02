"""Byte transports between ``ArduinoStage`` and the Uno firmware.

A :class:`Transport` moves whole protocol lines (``docs/serial-protocol.md``)
and nothing else: it neither frames nor parses them. Two implementations:

:class:`SerialTransport`
    The real USB-serial link (pyserial). All blocking calls run off the event
    loop: a dedicated daemon *reader thread* assembles incoming bytes into
    lines and hands them to the loop with ``call_soon_threadsafe``; writes,
    opening and closing run in ``asyncio.to_thread``. Opening the port of a
    Uno resets the board (DTR auto-reset). :meth:`SerialTransport.open`
    additionally pulses DTR explicitly (``reset_on_open``), so the reset - and
    therefore the ``BOOT`` banner the handshake waits for - happens even when
    the port was left with DTR asserted, and discards whatever was buffered
    before the reset. The port is opened ``exclusive`` (POSIX ``flock``), so a
    second process (the server while ``confocal-hwcheck`` runs, or vice
    versa) fails with a clear error instead of stealing replies.

:class:`EmulatorTransport`
    An in-memory link to a :class:`~.emulator.FirmwareEmulator` for tests and
    the hardware check's self-test. Emulated time comes from a controllable
    clock: :class:`ManualClock` (time passes only when the test calls
    ``advance``) or :class:`ScaledClock` (follows real time, optionally sped
    up). Faults can be injected: corrupted replies, a silent device, raw
    lines, an unexpected reset.

Errors: every failure is a :class:`~confocal.errors.CommunicationError`;
``read_line`` raises :class:`TransportTimeoutError` (a subclass) when no line
arrived in time, and :class:`TransportClosedError` once the link is gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from confocal.errors import CommunicationError
from confocal.hardware.arduino.emulator import FirmwareEmulator
from confocal.hardware.arduino.protocol import MAX_LINE_BYTES

if TYPE_CHECKING:
    import serial

log = logging.getLogger(__name__)

#: Longest run of bytes without a newline that is still buffered. Anything longer is
#: delivered as one oversized line (the codec rejects it with E_LENGTH) and the rest
#: of it discarded up to the next newline, so line noise can never grow memory.
MAX_BUFFERED_BYTES = 4 * MAX_LINE_BYTES

#: How long the DTR line is held de-asserted to reset the Uno.
DTR_PULSE_S = 0.1

#: Poll interval of the serial reader thread (bounded wait inside ``Serial.read``).
READER_POLL_S = 0.05


class TransportTimeoutError(CommunicationError):
    """No complete line arrived within the requested time."""


class TransportClosedError(CommunicationError):
    """The link is closed or failed (port unplugged, I/O error)."""


@runtime_checkable
class Transport(Protocol):
    """Line-oriented, asynchronous byte link to the firmware."""

    @property
    def is_open(self) -> bool: ...

    @property
    def description(self) -> str:
        """Human-readable name of the link (port path, "emulator")."""
        ...

    async def open(self) -> None:
        """Open the link. Opening a Uno resets it; it then prints ``! BOOT``."""

    async def write_line(self, line: bytes) -> None:
        """Send one complete line (terminator included)."""

    async def read_line(self, timeout_s: float | None) -> bytes:
        """Next received line including its ``\\n`` (an oversized fragment may lack it).

        Raises:
            TransportTimeoutError: nothing arrived within ``timeout_s``
                (``None`` waits forever).
            TransportClosedError: the link is closed or failed.
        """

    async def close(self) -> None:
        """Close the link. Idempotent; a pending ``read_line`` fails with TransportClosedError."""


class _LineAssembler:
    """Splits a byte stream into lines, bounding the size of an unterminated line."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._discarding = False

    def feed(self, data: bytes) -> list[bytes]:
        lines: list[bytes] = []
        for byte in data:
            if self._discarding:
                if byte == 0x0A:
                    self._discarding = False
                continue
            self._buffer.append(byte)
            if byte == 0x0A:
                lines.append(bytes(self._buffer))
                self._buffer.clear()
            elif len(self._buffer) > MAX_BUFFERED_BYTES:
                lines.append(bytes(self._buffer))
                self._buffer.clear()
                self._discarding = True
        return lines


class _LineQueue:
    """Loop-side inbox shared by both transports: lines, or a terminal failure."""

    def __init__(self) -> None:
        self._lines: deque[bytes] = deque()
        self._failure: CommunicationError | None = None
        self._ready = asyncio.Event()

    def put(self, line: bytes) -> None:
        self._lines.append(line)
        self._ready.set()

    def fail(self, error: CommunicationError) -> None:
        if self._failure is None:
            self._failure = error
        self._ready.set()

    def pop_nowait(self) -> bytes | None:
        """Next line, None if empty; raises the terminal failure once drained."""
        if self._lines:
            line = self._lines.popleft()
            if not self._lines and self._failure is None:
                self._ready.clear()
            return line
        if self._failure is not None:
            raise self._failure
        return None

    async def wait(self, timeout_s: float | None) -> None:
        """Wait until a line or failure is available (raises TimeoutError)."""
        if self._lines or self._failure is not None:
            return
        self._ready.clear()
        if timeout_s is None:
            await self._ready.wait()
        else:
            async with asyncio.timeout(max(0.0, timeout_s)):
                await self._ready.wait()


# --------------------------------------------------------------------------- serial


SerialFactory = Callable[..., "serial.Serial"]


def _default_serial_factory(**kwargs: object) -> serial.Serial:
    import serial  # imported lazily: only the real link needs pyserial at runtime

    return serial.Serial(**kwargs)  # type: ignore[arg-type]


class SerialTransport:
    """USB-serial link to the Uno (pyserial), never blocking the event loop."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        *,
        write_timeout_s: float = 2.0,
        reset_on_open: bool = True,
        serial_factory: SerialFactory | None = None,
    ) -> None:
        self._port = port
        self._baudrate = int(baudrate)
        self._write_timeout_s = float(write_timeout_s)
        self._reset_on_open = reset_on_open
        self._serial_factory = serial_factory or _default_serial_factory
        self._serial: serial.Serial | None = None
        self._inbox: _LineQueue | None = None
        self._reader: threading.Thread | None = None
        self._stop_reader = threading.Event()

    @property
    def port(self) -> str:
        return self._port

    @property
    def description(self) -> str:
        return f"{self._port} @ {self._baudrate} baud"

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._inbox is not None

    async def open(self) -> None:
        if self.is_open:
            return
        try:
            port = await asyncio.to_thread(self._open_blocking)
        except CommunicationError:
            raise
        except Exception as exc:
            raise CommunicationError(_open_error_message(self._port, exc)) from exc
        loop = asyncio.get_running_loop()
        inbox = _LineQueue()
        self._serial = port
        self._inbox = inbox
        self._stop_reader.clear()
        self._reader = threading.Thread(
            target=self._reader_main,
            args=(port, inbox, loop),
            name=f"serial-reader {self._port}",
            daemon=True,
        )
        self._reader.start()

    def _open_blocking(self) -> serial.Serial:
        port = self._serial_factory(
            port=self._port,
            baudrate=self._baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=READER_POLL_S,
            write_timeout=self._write_timeout_s,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            exclusive=True,
        )
        try:
            if self._reset_on_open:
                # Falling then rising DTR edge: the Uno's auto-reset capacitor turns it
                # into a reset pulse even if the port was already open with DTR asserted.
                port.dtr = False
                time.sleep(DTR_PULSE_S)
                port.reset_input_buffer()
                port.dtr = True
            else:
                port.reset_input_buffer()
        except Exception:
            port.close()
            raise
        return port

    def _reader_main(
        self, port: serial.Serial, inbox: _LineQueue, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Reader thread: bytes -> lines -> event loop. Ends on close or I/O error."""
        assembler = _LineAssembler()
        try:
            while not self._stop_reader.is_set():
                data = port.read(max(1, port.in_waiting))
                if not data:
                    continue
                for line in assembler.feed(data):
                    loop.call_soon_threadsafe(inbox.put, line)
        except Exception as exc:
            if not self._stop_reader.is_set():
                error = TransportClosedError(f"serial link {self._port} failed: {exc}")
                with contextlib.suppress(RuntimeError):  # loop already closed
                    loop.call_soon_threadsafe(inbox.fail, error)
                return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(
                inbox.fail, TransportClosedError(f"serial link {self._port} is closed")
            )

    async def write_line(self, line: bytes) -> None:
        port = self._serial
        if port is None or self._inbox is None:
            raise TransportClosedError(f"serial link {self._port} is not open")
        try:
            await asyncio.to_thread(self._write_blocking, port, line)
        except Exception as exc:
            raise CommunicationError(f"writing to {self._port} failed: {exc}") from exc

    @staticmethod
    def _write_blocking(port: serial.Serial, line: bytes) -> None:
        written = port.write(line)
        port.flush()
        if written is not None and written != len(line):
            raise OSError(f"short write ({written} of {len(line)} bytes)")

    async def read_line(self, timeout_s: float | None) -> bytes:
        inbox = self._inbox
        if inbox is None:
            raise TransportClosedError(f"serial link {self._port} is not open")
        return await _read_from(inbox, timeout_s, self._port)

    async def close(self) -> None:
        port, reader, inbox = self._serial, self._reader, self._inbox
        self._serial = None
        self._reader = None
        self._inbox = None
        if inbox is not None:
            inbox.fail(TransportClosedError(f"serial link {self._port} is closed"))
        if port is None:
            return
        self._stop_reader.set()
        await asyncio.to_thread(self._close_blocking, port, reader)

    @staticmethod
    def _close_blocking(port: serial.Serial, reader: threading.Thread | None) -> None:
        with contextlib.suppress(Exception):
            port.cancel_read()
        if reader is not None:
            reader.join(timeout=1.0)
        with contextlib.suppress(Exception):
            port.close()


def _open_error_message(port: str, exc: BaseException) -> str:
    text = f"cannot open serial port {port}: {exc}"
    lowered = str(exc).lower()
    if isinstance(exc, PermissionError) or "permission denied" in lowered:
        text += " (add the user to the 'dialout' group)"
    elif "no such file" in lowered or "could not open port" in lowered:
        text += " (is the Arduino plugged in? check `ls /dev/serial/by-id/`)"
    elif "resource temporarily unavailable" in lowered or "busy" in lowered:
        text += " (another program uses it, e.g. the confocal service: stop it first)"
    return text


async def _read_from(inbox: _LineQueue, timeout_s: float | None, name: str) -> bytes:
    line = inbox.pop_nowait()
    if line is not None:
        return line
    try:
        await inbox.wait(timeout_s)
    except TimeoutError:
        raise TransportTimeoutError(
            f"no reply from {name} within {timeout_s:.3g} s" if timeout_s is not None else name
        ) from None
    line = inbox.pop_nowait()
    if line is None:  # pragma: no cover - wait() only returns with a line or a failure
        raise TransportTimeoutError(f"no reply from {name}")
    return line


# --------------------------------------------------------------------------- emulator


class EmulatorClock(Protocol):
    """Source of emulated time (seconds, monotonic)."""

    def now(self) -> float: ...


class ManualClock:
    """Emulated time that passes only when :meth:`advance` is called (deterministic tests)."""

    def __init__(self, start_s: float = 0.0) -> None:
        self._now = float(start_s)

    def now(self) -> float:
        return self._now

    def advance(self, dt_s: float) -> None:
        if not dt_s >= 0.0:
            raise ValueError("dt_s must be >= 0")
        self._now += dt_s


class ScaledClock:
    """Emulated time following ``time.monotonic()``, ``speed`` times faster."""

    def __init__(self, speed: float = 1.0) -> None:
        if not speed > 0.0:
            raise ValueError("speed must be > 0")
        self._speed = float(speed)
        self._origin = time.monotonic()

    def now(self) -> float:
        return (time.monotonic() - self._origin) * self._speed


class EmulatorTransport:
    """In-memory link to a :class:`FirmwareEmulator`.

    Every operation first lets the emulator catch up with the clock, so events
    (``! DONE``) appear exactly when emulated time says the move finished; a
    pending :meth:`read_line` re-checks the clock every ``poll_s`` seconds.
    ``open()`` resets the emulator (like the DTR auto-reset of a real Uno) and
    queues its ``BOOT`` banner, unless ``reset_on_open`` is False.

    Fault injection for tests (each takes effect on device -> host lines):

    * :meth:`corrupt_next` - flip a byte of the next ``count`` lines (bad CRC);
    * :attr:`mute` - the device swallows every line silently (timeouts);
    * :meth:`inject` - deliver raw bytes as if the device had sent them;
    * :meth:`reset_device` - an unexpected reset: counters lost, ``BOOT`` sent;
    * :meth:`fail_link` - the link breaks (unplugged cable).
    """

    def __init__(
        self,
        emulator: FirmwareEmulator,
        clock: EmulatorClock | None = None,
        *,
        reset_on_open: bool = True,
        boot_banner: bool = True,
        poll_s: float = 0.001,
    ) -> None:
        self.emulator = emulator
        self.clock: EmulatorClock = clock if clock is not None else ManualClock()
        self._reset_on_open = reset_on_open
        self._boot_banner = boot_banner
        self._poll_s = float(poll_s)
        self._inbox: _LineQueue | None = None
        self._synced_s = self.clock.now()
        self._corrupt = 0
        self.mute = False
        self.fail_open: CommunicationError | None = None
        self._failure: TransportClosedError | None = None
        #: Every line written by the host, in order.
        self.sent: list[bytes] = []
        #: Every line delivered to the host, in order (after fault injection).
        self.received: list[bytes] = []
        self.opens = 0

    @property
    def description(self) -> str:
        return "firmware emulator"

    @property
    def is_open(self) -> bool:
        return self._inbox is not None and self._failure is None

    # ------------------------------------------------------------------ fault injection
    def corrupt_next(self, count: int = 1) -> None:
        """Corrupt the next ``count`` device -> host lines (checksum then fails)."""
        self._corrupt += int(count)

    def inject(self, line: bytes) -> None:
        """Deliver ``line`` to the host as if the device had sent it."""
        self._deliver([line])

    def reset_device(self) -> None:
        """Unexpected reset (brown-out, watchdog): counters lost, BOOT banner sent."""
        self._sync()
        self._deliver(self.emulator.reset())

    def fail_link(self, reason: str = "USB link lost") -> None:
        """Break the link: pending and future reads and writes raise TransportClosedError."""
        self._failure = TransportClosedError(reason)
        if self._inbox is not None:
            self._inbox.fail(self._failure)

    # ------------------------------------------------------------------ Transport
    async def open(self) -> None:
        await asyncio.sleep(0)
        if self.fail_open is not None:
            raise self.fail_open
        if self.is_open:
            return
        self._failure = None
        self._inbox = _LineQueue()
        self.opens += 1
        self._sync()
        if self._reset_on_open:
            banner = self.emulator.reset()
            if self._boot_banner:
                self._deliver(banner)

    async def write_line(self, line: bytes) -> None:
        await asyncio.sleep(0)
        if self._failure is not None:
            raise self._failure
        if self._inbox is None:
            raise TransportClosedError("emulator link is not open")
        self.sent.append(line)
        self._sync()
        replies = self.emulator.handle_line(line)
        if not self.mute:
            self._deliver(replies)

    async def read_line(self, timeout_s: float | None) -> bytes:
        loop = asyncio.get_running_loop()
        deadline = None if timeout_s is None else loop.time() + timeout_s
        while True:
            inbox = self._inbox
            if inbox is None:
                raise TransportClosedError("emulator link is closed")
            self._sync()
            line = inbox.pop_nowait()
            if line is not None:
                return line
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                raise TransportTimeoutError(f"no reply from the emulator within {timeout_s:.3g} s")
            wait = self._poll_s if remaining is None else min(self._poll_s, remaining)
            with contextlib.suppress(TimeoutError):
                await inbox.wait(wait)

    async def close(self) -> None:
        inbox = self._inbox
        self._inbox = None
        if inbox is not None:
            inbox.fail(TransportClosedError("emulator link is closed"))
        await asyncio.sleep(0)

    # ------------------------------------------------------------------ internals
    def _sync(self) -> None:
        """Advance the emulator to the clock's time; deliver the events that produces."""
        now = self.clock.now()
        dt = now - self._synced_s
        if dt <= 0.0:
            return
        self._synced_s = now
        events = self.emulator.advance(dt)
        if not self.mute:
            self._deliver(events)

    def _deliver(self, lines: Iterable[bytes]) -> None:
        inbox = self._inbox
        if inbox is None or self._failure is not None:
            return
        for line in lines:
            if self._corrupt > 0:
                self._corrupt -= 1
                line = _corrupt(line)
            self.received.append(line)
            inbox.put(line)


def _corrupt(line: bytes) -> bytes:
    """Flip one bit of the first byte: the CRC-8 check is guaranteed to fail."""
    if not line:
        return line
    return bytes([line[0] ^ 0x01]) + line[1:]
