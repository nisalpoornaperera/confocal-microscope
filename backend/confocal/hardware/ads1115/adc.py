"""``ADS1115ADC``: the ADS1115 reading the OPT101, by direct I2C register access (smbus2).

Register map (TI SBAS444, section 9.6)
--------------------------------------
``0x00`` conversion (signed 16 bit), ``0x01`` config, ``0x02`` / ``0x03``
comparator thresholds (unused). 16-bit registers are big-endian on the bus,
so they are accessed with I2C *block* transfers of two bytes (SMBus word
transfers are little-endian and would swap them). Config bits::

    15     OS         write 1: start a single-shot conversion; read 0: converting
    14:12  MUX        100..111: AIN0..AIN3 against GND;
                      000: AIN0-AIN1, 001: AIN0-AIN3, 010: AIN1-AIN3, 011: AIN2-AIN3
    11:9   PGA        000 +/-6.144 V, 001 4.096, 010 2.048, 011 1.024, 100 0.512, 101 0.256
    8      MODE       0: continuous conversion, 1: single-shot / power-down
    7:5    DR         8, 16, 32, 64, 128, 250, 475, 860 SPS
    4:0    comparator 00011: disabled (ALERT/RDY high impedance)

``[ads1115] channel`` selects AIN0..AIN3 single-ended; with
``differential = true`` it selects the MUX code of the same number, i.e.
0: AIN0-AIN1, 1: AIN0-AIN3, 2: AIN1-AIN3, 3: AIN2-AIN3.

Acquisition modes
-----------------
``single-shot`` (default)
    For every sample: write the config with OS = 1, wait one nominal
    conversion time, then poll the OS bit until the conversion is done, then
    read it. Every sample is therefore a distinct conversion that *started
    after* the call - after the stage has settled - whatever the converter's
    oscillator tolerance.
``continuous``
    The converter runs freely at the data rate; reads are paced
    :data:`CONTINUOUS_PACING` x the nominal period apart (the internal
    oscillator is specified to +/-10 %), so each read returns a newer
    conversion than the previous one, and the first read waits for a
    conversion that started after the call (or after the last gain change).

Pacing and timeouts use ``time.perf_counter`` (monotonic and high-resolution
on every platform; ``time.monotonic`` ticks in ~16 ms steps on Windows).
All bus I/O of a burst runs in one worker thread (``asyncio.to_thread``), so
the event loop never blocks. ``smbus2`` is imported only in :meth:`connect`
(it needs Linux's ``fcntl``; the driver module itself imports anywhere).
Every I2C failure becomes :class:`~confocal.errors.ADCError`.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import threading
import time
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import numpy as np

from confocal.config import ADS1115Config
from confocal.errors import ADCError, HardwareNotConnectedError
from confocal.hardware.ads1115.conversion import (
    VALID_DATA_RATES,
    conversion_time_s,
    counts_to_volts,
    is_saturated,
)
from confocal.hardware.base import ADC
from confocal.models.hardware import AdcGain, ADCStatus
from confocal.models.measurement import AdcSamples

REG_CONVERSION = 0x00
REG_CONFIG = 0x01

OS_BIT = 0x8000
MODE_SINGLE_SHOT = 0x0100
COMP_DISABLED = 0x0003

_MUX_SHIFT = 12
_PGA_SHIFT = 9
_DR_SHIFT = 5

PGA_CODES: dict[AdcGain, int] = {
    AdcGain.G2_3: 0b000,
    AdcGain.G1: 0b001,
    AdcGain.G2: 0b010,
    AdcGain.G4: 0b011,
    AdcGain.G8: 0b100,
    AdcGain.G16: 0b101,
}
DATA_RATE_CODES: dict[int, int] = {rate: code for code, rate in enumerate(VALID_DATA_RATES)}
#: MUX codes for ``channel`` 0..3, single-ended (against GND) and differential.
SINGLE_ENDED_MUX = (0b100, 0b101, 0b110, 0b111)
DIFFERENTIAL_MUX = (0b000, 0b001, 0b010, 0b011)
DIFFERENTIAL_PAIRS = ("AIN0-AIN1", "AIN0-AIN3", "AIN1-AIN3", "AIN2-AIN3")

#: Continuous mode: spacing of reads in nominal conversion periods (oscillator +/-10 %
#: plus margin), so consecutive reads always return different conversions.
CONTINUOUS_PACING = 1.2
#: Single-shot: wait this fraction of the nominal conversion time before polling OS.
_FIRST_POLL_FRACTION = 0.9
#: Single-shot: give up on a conversion after this many nominal periods (+ 10 ms).
_CONVERSION_TIMEOUT_PERIODS = 2.0

#: errno values of an I2C transfer nobody acknowledged: EIO, ENXIO, EREMOTEIO (Linux 121).
_NO_ACK_ERRNOS = (errno.EIO, errno.ENXIO, 121)

ConversionMode = Literal["single-shot", "continuous"]


class SMBusLike(Protocol):
    """The part of ``smbus2.SMBus`` this driver uses."""

    def read_i2c_block_data(self, i2c_addr: int, register: int, length: int) -> list[int]: ...

    def write_i2c_block_data(self, i2c_addr: int, register: int, data: Sequence[int]) -> None: ...

    def close(self) -> None: ...


BusFactory = Callable[[int], SMBusLike]


def open_smbus(bus: int) -> SMBusLike:
    """Open ``/dev/i2c-<bus>`` with smbus2 (imported lazily; Linux only).

    Raises:
        ADCError: smbus2 is unusable here, or the bus cannot be opened.
    """
    try:
        from smbus2 import SMBus
    except ImportError as exc:  # smbus2 needs fcntl: absent on Windows
        raise ADCError(
            f"smbus2 cannot be used on this system ({exc}); the ADS1115 driver needs "
            "Linux /dev/i2c-* (Raspberry Pi OS)"
        ) from exc
    try:
        handle: SMBusLike = SMBus(bus)
    except FileNotFoundError as exc:
        raise ADCError(
            f"I2C bus {bus} does not exist (/dev/i2c-{bus}): enable I2C with "
            "`sudo raspi-config nonint do_i2c 0` and reboot"
        ) from exc
    except PermissionError as exc:
        raise ADCError(
            f"no permission to open /dev/i2c-{bus}: add the user to the 'i2c' group"
        ) from exc
    except OSError as exc:
        raise ADCError(f"cannot open I2C bus {bus}: {exc}") from exc
    return handle


def mux_code(channel: int, differential: bool) -> int:
    if not 0 <= channel <= 3:
        raise ValueError(f"ADS1115 channel must be 0..3, got {channel}")
    return (DIFFERENTIAL_MUX if differential else SINGLE_ENDED_MUX)[channel]


def encode_config(
    *, mux: int, gain: AdcGain, data_rate_sps: int, continuous: bool, start: bool = False
) -> int:
    """The 16-bit config register value (comparator disabled)."""
    if data_rate_sps not in DATA_RATE_CODES:
        raise ValueError(f"unsupported ADS1115 data rate {data_rate_sps} SPS")
    word = (
        (mux & 0b111) << _MUX_SHIFT
        | PGA_CODES[gain] << _PGA_SHIFT
        | DATA_RATE_CODES[data_rate_sps] << _DR_SHIFT
        | COMP_DISABLED
    )
    if not continuous:
        word |= MODE_SINGLE_SHOT
    if start:
        word |= OS_BIT
    return word


def _to_signed(msb: int, lsb: int) -> int:
    raw = ((msb & 0xFF) << 8) | (lsb & 0xFF)
    return raw - 0x10000 if raw & 0x8000 else raw


class ADS1115ADC(ADC):
    """ADS1115 over I2C; see the module docstring for the register-level behaviour."""

    backend_name = "ads1115"

    def __init__(
        self,
        *,
        bus: int = 1,
        address: int = 0x48,
        channel: int = 0,
        differential: bool = False,
        gain: AdcGain = AdcGain.G1,
        data_rate_sps: int = 860,
        mode: ConversionMode = "single-shot",
        bus_factory: BusFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if not 0x48 <= address <= 0x4B:
            raise ValueError(f"ADS1115 address must be 0x48..0x4B, got {address:#x}")
        if mode not in ("single-shot", "continuous"):
            raise ValueError(f"unknown conversion mode {mode!r}")
        self._conversion_s = conversion_time_s(data_rate_sps)  # validates the rate
        self._mux = mux_code(channel, differential)
        self._bus_number = int(bus)
        self._address = int(address)
        self._channel = int(channel)
        self._differential = bool(differential)
        self._gain = gain
        self._data_rate = int(data_rate_sps)
        self._mode: ConversionMode = mode
        self._bus_factory: BusFactory = bus_factory or open_smbus
        self._sleep = sleep
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._io_lock = threading.Lock()
        self._bus: SMBusLike | None = None
        self._config_written_at = 0.0
        self._last_voltage_v: float | None = None
        self._saturated = False
        self._last_error: str | None = None

    @classmethod
    def from_config(
        cls,
        config: ADS1115Config,
        *,
        mode: ConversionMode = "single-shot",
        bus_factory: BusFactory | None = None,
    ) -> ADS1115ADC:
        return cls(
            bus=config.i2c_bus,
            address=config.address,
            channel=config.channel,
            differential=config.differential,
            gain=config.gain,
            data_rate_sps=config.data_rate_sps,
            mode=mode,
            bus_factory=bus_factory,
        )

    # ------------------------------------------------------------------ identity
    @property
    def address(self) -> int:
        return self._address

    @property
    def bus_number(self) -> int:
        return self._bus_number

    @property
    def mode(self) -> ConversionMode:
        return self._mode

    @property
    def input_description(self) -> str:
        if self._differential:
            return DIFFERENTIAL_PAIRS[self._channel]
        return f"AIN{self._channel}-GND"

    def version(self) -> str | None:
        return f"ADS1115 i2c-{self._bus_number}@0x{self._address:02X} ({self._mode})"

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> None:
        """Open the bus, write the configuration and read it back (detects a missing chip)."""
        if self._bus is not None:
            return
        try:
            await asyncio.to_thread(self._connect_blocking)
        except ADCError as exc:
            self._last_error = str(exc)
            raise
        self._last_error = None

    def _connect_blocking(self) -> None:
        with self._io_lock:
            try:
                bus = self._bus_factory(self._bus_number)
            except OSError as exc:
                raise ADCError(f"cannot open I2C bus {self._bus_number}: {exc}") from exc
            try:
                written = self._write_config(bus, start=False)
                readback = self._read_register(bus, REG_CONFIG)
            except ADCError:
                with contextlib.suppress(Exception):
                    bus.close()
                raise
            if (readback & ~OS_BIT) != (written & ~OS_BIT):
                with contextlib.suppress(Exception):
                    bus.close()
                raise ADCError(
                    f"the device at 0x{self._address:02X} on I2C bus {self._bus_number} is not "
                    f"an ADS1115 (wrote config 0x{written:04X}, read back 0x{readback:04X})"
                )
            self._bus = bus

    async def close(self) -> None:
        """Power the converter down (single-shot idle) and release the bus. Idempotent."""
        if self._bus is None:
            return
        await asyncio.to_thread(self._close_blocking)

    def _close_blocking(self) -> None:
        with self._io_lock:
            bus, self._bus = self._bus, None
            if bus is None:
                return
            with contextlib.suppress(Exception):
                word = encode_config(
                    mux=self._mux, gain=self._gain, data_rate_sps=self._data_rate, continuous=False
                )
                bus.write_i2c_block_data(self._address, REG_CONFIG, [word >> 8, word & 0xFF])
            with contextlib.suppress(Exception):
                bus.close()

    @property
    def connected(self) -> bool:
        return self._bus is not None

    # ------------------------------------------------------------------ configuration
    @property
    def gain(self) -> AdcGain:
        return self._gain

    async def set_gain(self, gain: AdcGain) -> None:
        self._require_connected()
        previous = self._gain
        self._gain = gain
        if self._mode == "continuous":
            try:
                await asyncio.to_thread(self._rewrite_config_blocking)
            except ADCError as exc:
                self._gain = previous
                self._last_error = str(exc)
                raise

    def _rewrite_config_blocking(self) -> None:
        with self._io_lock:
            self._write_config(self._live_bus(), start=False)

    @property
    def data_rate_sps(self) -> int:
        return self._data_rate

    # ------------------------------------------------------------------ acquisition
    async def read_samples(self, n: int) -> AdcSamples:
        """``n`` distinct conversions, each timestamped when it completed. Never averages."""
        self._require_connected()
        if n < 1:
            raise ValueError("n must be >= 1")
        gain = self._gain
        try:
            counts_list, times = await asyncio.to_thread(self._read_burst_blocking, n)
        except ADCError as exc:
            self._last_error = str(exc)
            raise
        counts = np.asarray(counts_list, dtype=np.int32)
        volts = counts_to_volts(counts, gain)
        self._last_voltage_v = float(np.mean(volts))
        self._saturated = is_saturated(counts)
        self._last_error = None
        return AdcSamples(
            counts=counts,
            volts=volts,
            timestamps=np.asarray(times, dtype=np.float64),
            gain=gain,
            data_rate_sps=self._data_rate,
        )

    def _read_burst_blocking(self, n: int) -> tuple[list[int], list[float]]:
        with self._io_lock:
            bus = self._live_bus()
            wall0, mono0 = self._wall_clock(), self._monotonic()
            counts: list[int] = []
            times: list[float] = []
            if self._mode == "single-shot":
                for _ in range(n):
                    counts.append(self._single_shot(bus))
                    times.append(wall0 + (self._monotonic() - mono0))
            else:
                pace = CONTINUOUS_PACING * self._conversion_s
                # The first conversion must start after both this call and the last config write.
                due = max(mono0, self._config_written_at) + 2.0 * pace
                for _ in range(n):
                    self._sleep_until(due)
                    counts.append(_to_signed(*self._read_bytes(bus, REG_CONVERSION)))
                    read_at = self._monotonic()
                    times.append(wall0 + (read_at - mono0))
                    due = read_at + pace
            return counts, times

    def _single_shot(self, bus: SMBusLike) -> int:
        started = self._monotonic()
        self._write_config(bus, start=True)
        self._sleep(_FIRST_POLL_FRACTION * self._conversion_s)
        deadline = started + _CONVERSION_TIMEOUT_PERIODS * self._conversion_s + 0.01
        poll_s = min(self._conversion_s / 20.0, 0.0002)
        while True:
            # Look at the clock *before* reading OS, so the conversion gets one more read
            # after the deadline has passed (coarse clocks, a delayed thread).
            expired = self._monotonic() > deadline
            if self._read_register(bus, REG_CONFIG) & OS_BIT:
                break
            if expired:
                raise ADCError(
                    f"ADS1115 at 0x{self._address:02X}: conversion did not complete within "
                    f"{deadline - started:.3g} s"
                )
            self._sleep(poll_s)
        return _to_signed(*self._read_bytes(bus, REG_CONVERSION))

    def _sleep_until(self, due: float) -> None:
        remaining = due - self._monotonic()
        if remaining > 0:
            self._sleep(remaining)

    # ------------------------------------------------------------------ register I/O
    def _live_bus(self) -> SMBusLike:
        bus = self._bus
        if bus is None:
            raise HardwareNotConnectedError("ADS1115 is not connected")
        return bus

    def _write_config(self, bus: SMBusLike, *, start: bool) -> int:
        word = encode_config(
            mux=self._mux,
            gain=self._gain,
            data_rate_sps=self._data_rate,
            continuous=self._mode == "continuous",
            start=start,
        )
        try:
            bus.write_i2c_block_data(self._address, REG_CONFIG, [word >> 8, word & 0xFF])
        except OSError as exc:
            raise self._i2c_error("write to", exc) from exc
        if not start:
            self._config_written_at = self._monotonic()
        return word

    def _read_bytes(self, bus: SMBusLike, register: int) -> tuple[int, int]:
        try:
            data = bus.read_i2c_block_data(self._address, register, 2)
        except OSError as exc:
            raise self._i2c_error("read from", exc) from exc
        if len(data) != 2:
            raise ADCError(f"ADS1115 returned {len(data)} bytes for register {register:#04x}")
        return int(data[0]), int(data[1])

    def _read_register(self, bus: SMBusLike, register: int) -> int:
        msb, lsb = self._read_bytes(bus, register)
        return (msb << 8) | lsb

    def _i2c_error(self, action: str, exc: OSError) -> ADCError:
        hint = " (no device answered: check wiring, power and address)"
        return ADCError(
            f"I2C {action} the ADS1115 at 0x{self._address:02X} on bus {self._bus_number} "
            f"failed: {exc}{hint if exc.errno in _NO_ACK_ERRNOS else ''}"
        )

    # ------------------------------------------------------------------ status
    async def status(self) -> ADCStatus:
        return ADCStatus(
            backend=self.backend_name,
            connected=self.connected,
            gain=self._gain,
            full_scale_v=self._gain.full_scale_v,
            data_rate_sps=self._data_rate,
            channel=self._channel,
            last_voltage_v=self._last_voltage_v,
            saturated=self._saturated,
            last_error=self._last_error,
        )

    def _require_connected(self) -> None:
        if self._bus is None:
            detail = f": {self._last_error}" if self._last_error else ""
            raise HardwareNotConnectedError(f"ADS1115 is not connected{detail}")
