"""Dark and reference calibration procedures and their versioned persistence.

Physics: the OPT101 output is ``V = V_dark + k * P_light``. The *dark* level
``V_dark`` (amplifier offset + dark current) is measured with no laser light
on the detector; the *reference* level ``V_ref`` is the in-focus signal of a
reference reflector (a plane mirror). Readings are then reported as
``(V - V_dark) / (V_ref - V_dark)``.

Safety and validity rules implemented here:

* Dark with a software-switchable laser: remember its state, switch it off,
  wait the (scaled) settling time, measure, and ALWAYS restore the previous
  state - also when the measurement fails - unless an emergency stop has been
  latched in the meantime (the laser is never switched back on then).
* Dark with a manually switched laser (the real machine): software cannot
  block the beam, so the operator must confirm it (``beam_blocked_confirmed``).
* A saturated reading is refused: its value is only a lower bound.
* A reference not clearly above the dark level is refused, since it would make
  every normalized value meaningless (or divide by ~0).
* Every result is a new immutable snapshot (previous values carried over,
  ``version=None``) persisted through the :class:`CalibrationStore`, which
  assigns the monotonic version. The store is synchronous, so it is called in
  ``asyncio.to_thread``.

The functions here do not serialise hardware access; the controller calls
them while holding its hardware lock and provides verified Z moves.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from confocal.errors import CalibrationError
from confocal.hardware.ads1115.conversion import is_saturated
from confocal.hardware.base import ADC, Laser
from confocal.microscope.measurement import aggregate_voltage, finite_volts, sample_std
from confocal.models.calibration import (
    CalibrationState,
    DarkCalibrationRequest,
    ReferenceCalibrationRequest,
)
from confocal.models.common import AxisLimits, Position, utc_now
from confocal.models.hardware import AdcGain, SamplingMethod
from confocal.models.measurement import AdcSamples
from confocal.storage.interfaces import CalibrationStore

log = logging.getLogger(__name__)

#: Minimum reference-above-dark margin, independent of the dark noise (volts).
MIN_REFERENCE_MARGIN_V = 1e-3
#: Reference must exceed dark by this many dark standard deviations.
REFERENCE_DARK_SIGMAS = 5.0
#: Scale factor from the median absolute deviation to a Gaussian standard deviation.
MAD_TO_SIGMA = 1.4826
#: Conversions per Z position during the reference focus search (speed vs noise).
SEARCH_SAMPLES = 16
#: Mechanical settling after each Z step of the focus search (seconds, before scaling).
SEARCH_SETTLE_S = 0.05

Sleep = Callable[[float], Awaitable[None]]
MoveZ = Callable[[float], Awaitable[Position]]


@dataclass(frozen=True, slots=True)
class CalibrationReading:
    """One aggregated calibration measurement."""

    value_v: float
    std_v: float
    n_samples: int
    gain: AdcGain
    measured_at: datetime


async def read_level(adc: ADC, n_samples: int, method: SamplingMethod) -> CalibrationReading:
    """Measure one burst and refuse it if any code is on the int16 rails."""
    samples = await adc.read_samples(n_samples)
    if is_saturated(samples.counts):
        raise CalibrationError(
            f"the detector signal saturates the ADC at gain {samples.gain.value} "
            f"(+/-{samples.gain.full_scale_v} V): select a lower gain or add an ND filter"
        )
    return _reading(samples, method)


def _reading(samples: AdcSamples, method: SamplingMethod) -> CalibrationReading:
    volts = finite_volts(samples)
    return CalibrationReading(
        value_v=aggregate_voltage(samples, method),
        std_v=sample_std(volts),
        n_samples=samples.n,
        gain=samples.gain,
        measured_at=utc_now(),
    )


# --------------------------------------------------------------------------- dark


async def measure_dark(
    adc: ADC,
    laser: Laser,
    request: DarkCalibrationRequest,
    *,
    laser_settle_s: float,
    sleep: Sleep,
    laser_on_allowed: Callable[[], bool],
) -> CalibrationReading:
    """Measure the detector level with no laser light on it.

    Args:
        laser_settle_s: wait after switching the laser (unscaled; ``sleep`` scales it).
        sleep: the controller's (time-scaled) settling wait.
        laser_on_allowed: False while an emergency stop is latched; the laser is
            then left off instead of being restored.

    Raises:
        CalibrationError: manual laser without ``beam_blocked_confirmed``, or a
            saturated dark reading (light is reaching the detector).
    """
    if not laser.controllable:
        if not request.beam_blocked_confirmed:
            raise CalibrationError(
                "the laser is switched manually: block the beam (or switch the laser off) "
                "and confirm with beam_blocked_confirmed=true"
            )
        return await _read_dark(adc, request)

    previous = await laser.is_enabled()
    await laser.set_enabled(False)
    try:
        await sleep(laser_settle_s)
        reading = await _read_dark(adc, request)
    except BaseException:
        await _restore_laser(
            laser, previous, laser_settle_s, sleep, laser_on_allowed, raise_errors=False
        )
        raise
    await _restore_laser(
        laser, previous, laser_settle_s, sleep, laser_on_allowed, raise_errors=True
    )
    return reading


async def _read_dark(adc: ADC, request: DarkCalibrationRequest) -> CalibrationReading:
    samples = await adc.read_samples(request.n_samples)
    if is_saturated(samples.counts):
        raise CalibrationError(
            "the dark reading saturates the ADC: light is reaching the detector "
            "(is the beam really blocked?)"
        )
    return _reading(samples, request.method)


async def _restore_laser(
    laser: Laser,
    previous: bool | None,
    settle_s: float,
    sleep: Sleep,
    laser_on_allowed: Callable[[], bool],
    *,
    raise_errors: bool,
) -> None:
    """Return the laser to the state it had before the dark measurement.

    An unknown previous state (``None``) leaves it off: off is the safe state.
    When a failure is already propagating (``raise_errors=False``) a restore
    error is logged so it does not mask the original exception.
    """
    if previous is not True:
        return
    if not laser_on_allowed():
        log.warning("emergency stop latched: laser left off after the dark calibration")
        return
    try:
        await laser.set_enabled(True)
        await sleep(settle_s)
    except Exception:
        if raise_errors:
            raise
        log.exception("could not switch the laser back on after a failed dark calibration")


# --------------------------------------------------------------------------- reference


def search_positions(
    center_um: float, width_um: float, step_um: float, limits: AxisLimits
) -> NDArray[np.float64]:
    """Increasing Z positions of the focus search, centred on ``center_um``, clipped to limits."""
    low = limits.clamp(center_um - width_um / 2.0)
    high = limits.clamp(center_um + width_um / 2.0)
    count = int(np.floor((high - low) / step_um + 1e-9)) + 1
    return low + step_um * np.arange(count, dtype=np.float64)


async def find_reference_focus(
    adc: ADC,
    request: ReferenceCalibrationRequest,
    *,
    center_um: float,
    z_limits: AxisLimits,
    move_z: MoveZ,
    sleep: Sleep,
) -> Position:
    """Sweep Z upwards around ``center_um`` and move to the maximum of the aggregated signal.

    Every move goes through ``move_z`` (the controller's limit-checked, verified
    move). A maximum on the first or last position means the focus may lie
    outside the searched range, so it is refused rather than used.

    Returns:
        The verified stage position at the maximum.
    """
    zs = search_positions(center_um, request.z_search_range_um, request.z_search_step_um, z_limits)
    if zs.shape[0] < 3:
        raise CalibrationError(
            "the Z search range has fewer than 3 positions inside the travel limits: "
            "move away from the Z limit or reduce the search step"
        )
    n = min(request.n_samples, SEARCH_SAMPLES)
    signal = np.empty(zs.shape[0], dtype=np.float64)
    for index, z in enumerate(zs):
        await move_z(float(z))
        await sleep(SEARCH_SETTLE_S)
        samples = await adc.read_samples(n)
        signal[index] = aggregate_voltage(samples, request.method)
    best = int(np.argmax(signal))
    if not has_focus_peak(signal):
        raise CalibrationError(
            "no focus peak found in the Z search (the signal is flat): place the reference "
            "reflector closer to focus or widen z_search_range_um"
        )
    if best in (0, zs.shape[0] - 1):
        raise CalibrationError(
            f"the signal maximum is at the edge of the Z search (z={zs[best]:.2f} um): "
            "focus on the reference reflector more closely or widen z_search_range_um"
        )
    return await move_z(float(zs[best]))


def has_focus_peak(signal: NDArray[np.float64]) -> bool:
    """True when the maximum stands clearly out of the search's noise floor.

    Out of focus the confocal signal is a flat background, so the maximum of a
    flat sweep is only noise. The peak must exceed the median by
    ``max(5 * sigma, 1 mV)`` with ``sigma`` the robust (MAD) spread of the sweep;
    a focus peak covers few positions of the sweep and barely affects the MAD.
    """
    median = float(np.median(signal))
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(signal - median)))
    margin = max(REFERENCE_DARK_SIGMAS * sigma, MIN_REFERENCE_MARGIN_V)
    return float(np.max(signal)) - median > margin


def check_reference(reading: CalibrationReading, calibration: CalibrationState) -> None:
    """Refuse a reference that is not clearly above the dark level.

    Threshold: ``dark + max(5 * dark_std, 1 mV)``; an uncalibrated dark counts as 0 V.
    """
    dark = calibration.dark_v or 0.0
    margin = max(REFERENCE_DARK_SIGMAS * (calibration.dark_std_v or 0.0), MIN_REFERENCE_MARGIN_V)
    if reading.value_v < dark + margin:
        raise CalibrationError(
            f"the reference signal ({reading.value_v:.4f} V) is not clearly above the dark "
            f"level ({dark:.4f} V, margin {margin * 1e3:.1f} mV): is the laser on and the "
            "reference reflector in focus?"
        )


# --------------------------------------------------------------------------- snapshots


def with_dark(
    previous: CalibrationState, reading: CalibrationReading, notes: str | None
) -> CalibrationState:
    """New unpersisted snapshot: ``previous`` with a new dark level."""
    return CalibrationState.model_validate(
        {
            **previous.model_dump(),
            "version": None,
            "created_at": utc_now(),
            "updated_field": "dark",
            "dark_v": reading.value_v,
            "dark_std_v": reading.std_v,
            "dark_n_samples": reading.n_samples,
            "dark_measured_at": reading.measured_at,
            "dark_gain": reading.gain,
            "notes": notes,
        }
    )


def with_reference(
    previous: CalibrationState,
    reading: CalibrationReading,
    position: Position | None,
    notes: str | None,
) -> CalibrationState:
    """New unpersisted snapshot: ``previous`` with a new reference level."""
    return CalibrationState.model_validate(
        {
            **previous.model_dump(),
            "version": None,
            "created_at": utc_now(),
            "updated_field": "reference",
            "reference_v": reading.value_v,
            "reference_std_v": reading.std_v,
            "reference_n_samples": reading.n_samples,
            "reference_measured_at": reading.measured_at,
            "reference_gain": reading.gain,
            "reference_position": None if position is None else position.model_dump(),
            "notes": notes,
        }
    )


async def persist(store: CalibrationStore, state: CalibrationState) -> CalibrationState:
    """Save a snapshot (blocking store, run in a thread); returns it with its version."""
    saved = await asyncio.to_thread(store.save_calibration, state)
    if saved.version is None:
        raise CalibrationError("the calibration store did not assign a version")
    return saved


async def load_latest(store: CalibrationStore) -> CalibrationState:
    """Latest persisted snapshot, or an empty (uncalibrated) state."""
    latest = await asyncio.to_thread(store.latest_calibration)
    return CalibrationState() if latest is None else latest
