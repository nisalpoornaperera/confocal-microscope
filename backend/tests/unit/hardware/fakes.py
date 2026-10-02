"""Shared fakes for the hardware driver tests (no real serial port or I2C bus).

* :class:`FakeClock` - simulated time whose ``sleep`` advances instantly;
* :class:`RealTime` - the same interface on the real monotonic clock;
* :class:`FakeADS1115` - register-level model of an ADS1115 on a fake SMBus,
  written from the datasheet independently of the driver's constants;
* :func:`emulated_transport` - an EmulatorTransport whose moves finish at once.
"""

from __future__ import annotations

import errno
import math
import time
from collections.abc import Sequence
from typing import Protocol

from confocal.config import ArduinoConfig
from confocal.hardware.arduino.emulator import FirmwareEmulator
from confocal.hardware.arduino.transport import EmulatorTransport, ScaledClock

ADDRESS = 0x48
EREMOTEIO = 121  # Linux errno of an unacknowledged I2C transfer (absent on Windows)


class TimeSource(Protocol):
    @property
    def now(self) -> float: ...


class RealTime:
    """``now`` on the real clock the ADS1115 driver uses by default (``time.perf_counter``)."""

    @property
    def now(self) -> float:
        return time.perf_counter()


# Datasheet tables, written out independently of the driver's own constants.
FULL_SCALE = {0: 6.144, 1: 4.096, 2: 2.048, 3: 1.024, 4: 0.512, 5: 0.256, 6: 0.256, 7: 0.256}
RATES = (8, 16, 32, 64, 128, 250, 475, 860)
MUX_INPUTS = {
    0: ("AIN0", "AIN1"),
    1: ("AIN0", "AIN3"),
    2: ("AIN1", "AIN3"),
    3: ("AIN2", "AIN3"),
    4: ("AIN0", "GND"),
    5: ("AIN1", "GND"),
    6: ("AIN2", "GND"),
    7: ("AIN3", "GND"),
}


class FakeClock:
    """Shared simulated time: the driver's sleep() advances it instantly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> float:
        return 1_700_000_000.0 + self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


class FakeADS1115:
    """Register map of one ADS1115 on a fake SMBus.

    Each conversion's code is ``round(V / lsb) + index`` where ``index`` counts
    the conversions since power-up, so tests can tell distinct conversions
    apart and check that a sample was taken after a given moment.
    """

    def __init__(
        self,
        clock: TimeSource,
        inputs: dict[str, float] | None = None,
        *,
        oscillator: float = 1.0,
        address: int = ADDRESS,
        tag_conversions: bool = True,
    ) -> None:
        self.clock = clock
        self.inputs = {"AIN0": 0.5, "AIN1": 0.25, "AIN2": 0.0, "AIN3": 0.1, "GND": 0.0} | (
            inputs or {}
        )
        self.oscillator = oscillator  # >1: slower than nominal
        self.address = address
        self.tag_conversions = tag_conversions
        self.config = 0x8583  # power-on default
        self.conversion = 0
        self.index = 0
        self.busy_until: float | None = None
        self.continuous_since: float | None = None
        self.writes: list[int] = []
        self.fail_next: OSError | None = None
        self.closed = False
        self.started_at: list[float] = []
        self.present: set[int] = {address}

    # -- decoding (datasheet) --------------------------------------------------
    def field(self, shift: int, mask: int) -> int:
        return (self.config >> shift) & mask

    def period(self) -> float:
        return self.oscillator / RATES[self.field(5, 0b111)]

    def code(self) -> int:
        positive, negative = MUX_INPUTS[self.field(12, 0b111)]
        volts = self.inputs[positive] - self.inputs[negative]
        fs = FULL_SCALE[self.field(9, 0b111)]
        tag = self.index if self.tag_conversions else 0
        return max(-32768, min(32767, round(volts / fs * 32768) + tag))

    def _update(self) -> None:
        now = self.clock.now
        if self.busy_until is not None and now >= self.busy_until:
            self.busy_until = None
            self.index += 1
            self.conversion = self.code()
        if self.continuous_since is not None:
            done = math.floor((now - self.continuous_since) / self.period())
            while self.index < done:
                self.index += 1
                self.conversion = self.code()

    # -- SMBus -----------------------------------------------------------------
    def _check(self, address: int) -> None:
        if self.closed:
            raise OSError(errno.EBADF, "bus closed")
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        if address != self.address:
            raise OSError(EREMOTEIO, "Remote I/O error")

    def write_i2c_block_data(self, i2c_addr: int, register: int, data: Sequence[int]) -> None:
        self._check(i2c_addr)
        self._update()
        assert len(data) == 2
        word = (data[0] << 8) | data[1]
        if register != 0x01:
            return
        self.writes.append(word)
        start = bool(word & 0x8000)
        self.config = word & 0x7FFF
        single_shot = bool(word & 0x0100)
        if single_shot:
            self.continuous_since = None
            if start and self.busy_until is None:
                self.busy_until = self.clock.now + self.period()
                self.started_at.append(self.clock.now)
        else:
            self.busy_until = None
            self.continuous_since = self.clock.now
            self.index = 0

    def read_i2c_block_data(self, i2c_addr: int, register: int, length: int) -> list[int]:
        self._check(i2c_addr)
        self._update()
        assert length == 2
        if register == 0x01:
            value = self.config | (0 if self.busy_until is not None else 0x8000)
        elif register == 0x00:
            value = self.conversion & 0xFFFF
        else:
            value = 0
        return [value >> 8, value & 0xFF]

    def close(self) -> None:
        self.closed = True

    def reopen(self) -> FakeADS1115:
        """A new handle on the same chip (each ``SMBus(bus)`` open is independent)."""
        self.closed = False
        return self

    def current_index(self) -> int:
        """Conversions completed so far (brought up to the current time)."""
        self._update()
        return self.index

    # -- bus scan (i2cdetect-style probes) -------------------------------------
    def write_quick(self, i2c_addr: int) -> None:
        if i2c_addr not in self.present:
            raise OSError(EREMOTEIO, "Remote I/O error")

    def read_byte(self, i2c_addr: int) -> int:
        if i2c_addr not in self.present:
            raise OSError(EREMOTEIO, "Remote I/O error")
        return 0


def emulated_transport(arduino: ArduinoConfig, *, speed: float = 10_000.0) -> EmulatorTransport:
    """A link to a fresh firmware emulator; emulated time runs ``speed`` x real time."""
    return EmulatorTransport(FirmwareEmulator.from_config(arduino), ScaledClock(speed=speed))
