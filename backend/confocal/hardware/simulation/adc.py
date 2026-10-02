"""Simulated ADS1115 reading a simulated OPT101 that looks at a simulated sample.

Per-sample detector voltage::

    V = dark_voltage_v + [laser on] * (background_v + signal_v(x, y, z))
        + read noise   N(0, read_noise_v)
        + shot noise   N(0, shot_noise_fraction * sqrt(V_optical * peak_voltage_v))

``V_optical`` is the noise-free optical part (background + confocal signal):
photon shot noise grows with the square root of the detected light, and the
dark offset carries none. The result is clipped to the OPT101's output range
``[0, opt101_saturation_v]`` (a single-supply amplifier cannot go below ground
or above its rail) and then converted exactly like the real ADS1115 does:
rounded to signed 16-bit codes with the PGA gain, clipping at full scale. The
returned voltages are ``counts * lsb``, so they carry the real quantisation.

A burst of ``n`` conversions takes ``n / data_rate * time_scale`` seconds.
Noise comes from a seeded NumPy generator (deterministic per seed and call
sequence). Fault injection: with ``adc_fault_after_reads = N`` the first ``N``
reads succeed and read ``N + 1`` raises :class:`ADCError` (once).
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable

import numpy as np

from confocal.config import SimulationConfig
from confocal.errors import ADCError, HardwareNotConnectedError
from confocal.hardware.ads1115.conversion import (
    conversion_time_s,
    counts_to_volts,
    is_saturated,
    volts_to_counts,
)
from confocal.hardware.base import ADC
from confocal.hardware.simulation.surface import SimulatedConfocalSurface
from confocal.models.common import Position
from confocal.models.hardware import AdcGain, ADCStatus
from confocal.models.measurement import AdcSamples


class SimulationADC(ADC):
    """ADS1115 + OPT101 model driven by the true (simulated) stage position."""

    backend_name = "simulation"

    def __init__(
        self,
        surface: SimulatedConfocalSurface,
        *,
        position_source: Callable[[], Position],
        laser_source: Callable[[], bool],
        config: SimulationConfig,
        gain: AdcGain,
        data_rate_sps: int,
        seed: int,
    ) -> None:
        self._conversion_s = conversion_time_s(data_rate_sps)  # validates the rate
        self._surface = surface
        self._position_source = position_source
        self._laser_source = laser_source
        self._config = config
        self._gain = gain
        self._data_rate = int(data_rate_sps)
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._reads = 0
        self._last_voltage_v: float | None = None
        self._saturated = False
        self._last_error: str | None = None

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> None:
        self._connected = True
        await asyncio.sleep(0)

    async def close(self) -> None:
        self._connected = False
        await asyncio.sleep(0)

    @property
    def connected(self) -> bool:
        return self._connected

    def version(self) -> str | None:
        return "simulation"

    # ------------------------------------------------------------------ configuration
    @property
    def gain(self) -> AdcGain:
        return self._gain

    async def set_gain(self, gain: AdcGain) -> None:
        self._require_connected()
        self._gain = gain
        await asyncio.sleep(0)

    @property
    def data_rate_sps(self) -> int:
        return self._data_rate

    @property
    def reads(self) -> int:
        """Number of :meth:`read_samples` calls so far."""
        return self._reads

    # ------------------------------------------------------------------ acquisition
    def expected_voltage_v(self, position: Position, *, laser_on: bool) -> float:
        """Noise-free OPT101 output at ``position`` (before clipping and quantisation)."""
        cfg = self._config
        volts = cfg.dark_voltage_v
        if laser_on:
            signal = self._surface.signal_v(position.x_um, position.y_um, position.z_um)
            volts += cfg.background_v + float(signal)
        return volts

    async def read_samples(self, n: int) -> AdcSamples:
        self._require_connected()
        if n < 1:
            raise ValueError("n must be >= 1")
        self._reads += 1
        fault_after = self._config.adc_fault_after_reads
        if fault_after is not None and self._reads == fault_after + 1:
            self._last_error = f"injected ADC fault on read {self._reads}"
            raise ADCError(self._last_error)

        started = time.time()
        sample_period_s = self._conversion_s * self._config.time_scale
        await asyncio.sleep(n * sample_period_s)

        cfg = self._config
        ideal = self.expected_voltage_v(self._position_source(), laser_on=self._laser_source())
        optical = max(ideal - cfg.dark_voltage_v, 0.0)
        shot_sigma = cfg.shot_noise_fraction * math.sqrt(optical * cfg.peak_voltage_v)
        volts = (
            ideal
            + cfg.read_noise_v * self._rng.standard_normal(n)
            + shot_sigma * self._rng.standard_normal(n)
        )
        volts = np.clip(volts, 0.0, cfg.opt101_saturation_v)
        counts = volts_to_counts(volts, self._gain)
        quantised = counts_to_volts(counts, self._gain)
        timestamps = started + np.arange(1, n + 1, dtype=np.float64) * sample_period_s

        self._last_voltage_v = float(np.mean(quantised))
        self._saturated = is_saturated(counts)
        self._last_error = None
        return AdcSamples(
            counts=counts,
            volts=quantised,
            timestamps=timestamps,
            gain=self._gain,
            data_rate_sps=self._data_rate,
        )

    async def status(self) -> ADCStatus:
        return ADCStatus(
            backend=self.backend_name,
            connected=self._connected,
            gain=self._gain,
            full_scale_v=self._gain.full_scale_v,
            data_rate_sps=self._data_rate,
            channel=0,
            last_voltage_v=self._last_voltage_v,
            saturated=self._saturated,
            last_error=self._last_error,
        )

    def _require_connected(self) -> None:
        if not self._connected:
            raise HardwareNotConnectedError("simulation ADC is not connected")
