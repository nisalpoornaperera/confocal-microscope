"""Shared helpers of the hardware-in-the-loop tests (see ``tests/hil/README.md``).

Every test here talks to *real* devices and is skipped unless the matching
environment variable is set:

* ``CONFOCAL_HIL_PORT``  - serial port of a Uno flashed with the confocal firmware,
  e.g. ``/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_...-if00``;
* ``CONFOCAL_HIL_I2C``   - I2C bus number of the ADS1115 (``1`` on the Pi);
  optional ``CONFOCAL_HIL_I2C_ADDRESS`` (default ``0x48``) and
  ``CONFOCAL_HIL_ADC_CHANNEL`` (default ``0``).

``CONFOCAL_CONFIG`` (optional) supplies the machine configuration (motor
travel and speeds, kinematics, limits); otherwise the built-in defaults are
used. The firmware's compiled-in motor travel must match ``[arduino.a|b|c]``.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from confocal.config import Settings, load_settings
from confocal.hardware.arduino.emulator import FirmwareEmulator
from confocal.hardware.arduino.protocol import (
    Command,
    Event,
    EventKind,
    MalformedMessageError,
    Response,
    SequenceCounter,
    encode_command,
    parse_line,
)
from confocal.hardware.arduino.transport import SerialTransport, TransportTimeoutError

HIL_PORT = os.environ.get("CONFOCAL_HIL_PORT", "")
HIL_I2C = os.environ.get("CONFOCAL_HIL_I2C", "")

requires_arduino = pytest.mark.skipif(
    not HIL_PORT, reason="hardware-in-the-loop: set CONFOCAL_HIL_PORT to a flashed Uno's port"
)
requires_adc = pytest.mark.skipif(
    not HIL_I2C, reason="hardware-in-the-loop: set CONFOCAL_HIL_I2C to the ADS1115's I2C bus"
)

#: Upper bound of how long the Uno's bootloader + setup() may take after the DTR reset.
BOOT_TIMEOUT_S = 6.0
REPLY_TIMEOUT_S = 2.0


def hil_settings() -> Settings:
    """``$CONFOCAL_CONFIG`` (if set) with the port taken from ``CONFOCAL_HIL_PORT``."""
    settings = load_settings()
    if HIL_PORT:
        settings = settings.model_copy(
            update={"arduino": settings.arduino.model_copy(update={"port": HIL_PORT})}
        )
    return settings


def i2c_address() -> int:
    return int(os.environ.get("CONFOCAL_HIL_I2C_ADDRESS", "0x48"), 0)


def adc_channel() -> int:
    return int(os.environ.get("CONFOCAL_HIL_ADC_CHANNEL", "0"))


class RawLink:
    """Protocol-level access to the Uno: numbered requests, raw lines, events."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.transport = SerialTransport(settings.arduino.port, settings.arduino.baudrate)
        self.seq = SequenceCounter()
        self.boot: Event | None = None
        self.events: list[Event] = []

    async def open(self) -> Event:
        """Open (resets the Uno) and wait for the BOOT banner."""
        await self.transport.open()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + BOOT_TIMEOUT_S
        while loop.time() < deadline:
            try:
                line = await self.transport.read_line(deadline - loop.time())
            except TransportTimeoutError:
                break
            try:
                message = parse_line(line)
            except MalformedMessageError:
                continue  # bootloader noise
            if isinstance(message, Event) and message.kind is EventKind.BOOT:
                self.boot = message
                return message
        raise AssertionError(f"no BOOT banner from {self.transport.port} in {BOOT_TIMEOUT_S} s")

    async def close(self) -> None:
        await self.transport.close()

    async def exchange_raw(self, line: bytes) -> tuple[bytes, list[Event]]:
        """Send ``line``; return the raw reply line and the events that came before it."""
        await self.transport.write_line(line)
        events: list[Event] = []
        while True:
            raw = await self.transport.read_line(REPLY_TIMEOUT_S)
            message = parse_line(raw)
            if isinstance(message, Event):
                events.append(message)
                self.events.append(message)
                continue
            return raw, events

    async def send(self, command: Command, *args: int) -> tuple[Response, list[Event]]:
        seq = self.seq.next()
        raw, events = await self.exchange_raw(encode_command(seq, command, args))
        reply = parse_line(raw)
        assert isinstance(reply, Response)
        assert reply.seq == seq, f"reply {reply.seq} to request {seq}"
        return reply, events

    async def wait_event(self, timeout_s: float) -> Event:
        raw = await self.transport.read_line(timeout_s)
        message = parse_line(raw)
        assert isinstance(message, Event), f"expected an event, got {raw!r}"
        self.events.append(message)
        return message

    def emulator(self) -> FirmwareEmulator:
        """The reference model of the firmware, configured like the machine."""
        version = self.boot.version if self.boot is not None and self.boot.version else "1.0.0"
        return FirmwareEmulator.from_config(self.settings.arduino, version=version)

    def move_duration_s(self, steps: int) -> float:
        speed = min(
            motor.max_speed_steps_s
            for motor in (self.settings.arduino.a, self.settings.arduino.b, self.settings.arduino.c)
        )
        return abs(steps) / speed
