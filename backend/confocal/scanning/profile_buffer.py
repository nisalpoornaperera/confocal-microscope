"""Accumulation of the raw data of one XY point and its conversion to ``ProfileData``.

Every Z position visited at a point (coarse sweep, coarse retry, fine sweep)
is appended in acquisition order with its phase label, the commanded and the
stage-reported Z, every raw ADC code and voltage, and the aggregated voltage.
Nothing is ever discarded: the stored profile is the complete measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from confocal.errors import ADCError
from confocal.models.calibration import CalibrationState
from confocal.models.hardware import AdcGain, SamplingMethod
from confocal.models.measurement import AdcSamples, ProfileData, ProfilePhase, nan_to_none
from confocal.models.scan import LiveProfile
from confocal.processing.normalization import aggregate_samples, normalize, subtract_dark
from confocal.scanning.plan import GridPoint

SignalUnits = Literal["normalized", "volts"]

#: Signed 16-bit code range of the converter (``AdcGain.lsb_v`` = full scale / 32768).
#: An input beyond the PGA full scale returns a rail code instead of wrapping.
ADC_CODE_MIN = -32768
ADC_CODE_MAX = 32767


def hits_adc_rails(counts: NDArray[np.int32]) -> bool:
    """True when any raw code sits on a converter rail (the reading is saturated)."""
    return bool(np.any((counts >= ADC_CODE_MAX) | (counts <= ADC_CODE_MIN)))


@dataclass(frozen=True, slots=True)
class CalibrationSnapshot:
    """Calibration values frozen at scan start, applied identically to every point.

    ``reference_v`` is kept only when it lies above the dark level, i.e. when a
    normalized intensity can actually be computed; otherwise the scan works in
    dark-corrected volts throughout instead of failing at the first point.
    """

    dark_v: float | None
    reference_v: float | None
    version: int | None

    @classmethod
    def from_state(cls, state: CalibrationState) -> CalibrationSnapshot:
        reference = state.reference_v if state.can_normalize else None
        return cls(dark_v=state.dark_v, reference_v=reference, version=state.version)

    @property
    def signal_units(self) -> SignalUnits:
        return "normalized" if self.reference_v is not None else "volts"

    def normalized(self, voltage_v: NDArray[np.float64]) -> NDArray[np.float64] | None:
        return normalize(voltage_v, self.dark_v, self.reference_v)

    def intensity(self, voltage_v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalized intensity when possible, else dark-corrected volts."""
        normalized = self.normalized(voltage_v)
        return normalized if normalized is not None else subtract_dark(voltage_v, self.dark_v)

    def intensity_of(self, voltage_v: float) -> float:
        return float(self.intensity(np.array([voltage_v], dtype=np.float64))[0])


class ProfileBuffer:
    """Growing per-position record of one point, in acquisition order."""

    def __init__(self, samples_per_z: int, sampling_method: SamplingMethod) -> None:
        if samples_per_z < 1:
            raise ValueError("samples_per_z must be >= 1")
        self._samples_per_z = samples_per_z
        self._method = sampling_method
        self._z: list[float] = []
        self._z_reported: list[float] = []
        self._phase: list[int] = []
        self._counts: list[NDArray[np.int32]] = []
        self._volts: list[NDArray[np.float64]] = []
        self._aggregated: list[float] = []
        self._timestamps: list[float] = []
        self._gain: AdcGain | None = None

    @property
    def n(self) -> int:
        return len(self._z)

    @property
    def gain(self) -> AdcGain | None:
        """PGA gain of the recorded samples (``None`` before the first position)."""
        return self._gain

    def add(
        self, *, z_um: float, z_reported_um: float, phase: ProfilePhase, samples: AdcSamples
    ) -> float:
        """Append one position; returns its aggregated voltage.

        Raises:
            ADCError: the burst does not contain ``samples_per_z`` conversions.
        """
        if samples.n != self._samples_per_z:
            raise ADCError(f"ADC returned {samples.n} samples, expected {self._samples_per_z}")
        volts = np.asarray(samples.volts, dtype=np.float64)
        value = float(aggregate_samples(volts, self._method))
        if self._gain is None:
            self._gain = samples.gain
        self._z.append(float(z_um))
        self._z_reported.append(float(z_reported_um))
        self._phase.append(int(phase))
        self._counts.append(np.asarray(samples.counts, dtype=np.int32))
        self._volts.append(volts)
        self._aggregated.append(value)
        self._timestamps.append(float(samples.timestamps[0]))
        return value

    def reported_z(self, part: slice) -> NDArray[np.float64]:
        return np.asarray(self._z_reported[part], dtype=np.float64)

    def aggregated(self, part: slice) -> NDArray[np.float64]:
        return np.asarray(self._aggregated[part], dtype=np.float64)

    def build(
        self,
        calibration: CalibrationSnapshot,
        *,
        filtered: tuple[slice, NDArray[np.float64]] | None = None,
    ) -> ProfileData:
        """The complete profile. ``filtered`` = (positions analysed, analyser output).

        ``filtered`` is NaN at every position that was not analysed (the coarse
        sweeps); without an analysis it is ``None``.
        """
        if self._gain is None:
            raise ValueError("cannot build a profile without any position")
        aggregated = self.aggregated(slice(None))
        filtered_full: NDArray[np.float64] | None = None
        if filtered is not None:
            part, values = filtered
            filtered_full = np.full(self.n, np.nan, dtype=np.float64)
            filtered_full[part] = np.asarray(values, dtype=np.float64)
        return ProfileData(
            z_um=np.asarray(self._z, dtype=np.float64),
            z_reported_um=np.asarray(self._z_reported, dtype=np.float64),
            phase=np.asarray(self._phase, dtype=np.uint8),
            raw_counts=np.stack(self._counts).astype(np.int32, copy=False),
            voltage_v=np.stack(self._volts).astype(np.float64, copy=False),
            voltage_agg_v=aggregated,
            timestamps=np.asarray(self._timestamps, dtype=np.float64),
            gain=self._gain,
            sampling_method=self._method,
            dark_v=calibration.dark_v,
            reference_v=calibration.reference_v,
            calibration_version=calibration.version,
            normalized=calibration.normalized(aggregated),
            filtered=filtered_full,
        )

    def live(self, point: GridPoint, calibration: CalibrationSnapshot) -> LiveProfile:
        """JSON-safe live I(Z) of the point so far (stage-reported Z)."""
        intensity = calibration.intensity(self.aggregated(slice(None)))
        return LiveProfile(
            point_id=point.point_id,
            x_um=point.x_um,
            y_um=point.y_um,
            phase=list(self._phase),
            z_um=list(self._z_reported),
            intensity=nan_to_none(intensity.tolist()),
        )
