"""The physics baseline of one XY point: I(Z) profile -> confocal peak -> surface Z.

In a confocal microscope the detected intensity is maximal when the surface
lies in the focal plane, so the surface height at (x, y) is the Z of the
maximum of the axial response I(Z). Two functions implement the scan
engine's protocols (``scanning/protocols.py``):

``find_coarse_peak``
    Locates the approximate maximum of the coarse sweep (dark-corrected
    volts) so that the fine sweep can be centred on it.

``analyse_profile``
    The full analysis of the fine sweep::

        dark subtraction -> normalization -> sort / de-duplicate -> filter
        -> baseline + noise -> peak detection -> parabolic + Gaussian fits
        -> surface Z -> SNR, FWHM, prominence, fit residual -> confidence -> status

Status precedence (first match wins)::

    NO_PEAK         no significant peak, SNR < min_snr or relative prominence
                    < min_relative_prominence
    PEAK_AT_EDGE    the maximum lies within edge_margin_samples of a sweep end
    (with ``accept_weak_peaks`` a peak below the fixed significance rule is kept
    as a *weak peak* if it passes min_snr / min_relative_prominence; it is
    flagged ``weak_peak`` and is at most LOW_CONFIDENCE)
    FIT_FAILED      neither fit gives a usable surface height
    LOW_CONFIDENCE  confidence < min_confidence, or the selected peak is not the
                    brightest signal of the sweep (``global_max_not_selected``:
                    which reflection is the surface is ambiguous; the confidence
                    also takes the edge penalty)
    VALID

A peak is never assumed valid: every point must earn VALID by passing all
checks. The surface height is the Gaussian centre when that fit succeeds,
lies inside the sweep and agrees with the parabolic vertex within
``max_fit_disagreement_um``; otherwise the parabolic vertex; otherwise none.

Saturation: the OPT101 clips at a few microwatts, giving a flat top. Samples
at or above ``saturation_v`` (raw volts), plus short unsaturated notches
between them, form the *clipped top*: peak detection treats it as one flat
plateau, and baseline, noise and both fits ignore it -- the flanks still
locate the centre. Saturation is only recognised when ``saturation_v`` is
configured.

The fine sweep should span at least ~2 axial FWHM: the relative prominence
(and with it the NO_PEAK decision) measures how far the signal falls on both
sides of the maximum *within the sweep*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from confocal.models.processing import (
    CoarsePeak,
    FitMethod,
    PeakFit,
    PointStatus,
    ProcessingConfig,
    ProfileAnalysis,
    ProfileProcessingResult,
)
from confocal.processing.fitting import (
    GAUSSIAN_FWHM_PER_SIGMA,
    MIN_GAUSSIAN_POINTS,
    fit_gaussian,
    fit_parabolic,
)
from confocal.processing.metrics import (
    PeakQuality,
    compute_snr,
    confidence_score,
    relative_prominence,
)
from confocal.processing.normalization import normalize, subtract_dark
from confocal.processing.peaks import NO_DETECTION, PeakDetection, PeakInfo, detect_peaks
from confocal.processing.preparation import (
    PreparedProfile,
    as_profile_arrays,
    fill_short_gaps,
    prepare_profile,
    saturation_mask,
)
from confocal.processing.signal import (
    MIN_OFF_PEAK_SAMPLES,
    estimate_baseline_and_noise,
    filter_signal,
)

#: A peak (main or secondary) must stand this many noise standard deviations
#: above its surroundings to count. Peaks are detected on the smoothed signal,
#: whose noise is lower than the raw noise this is measured in, so noise
#: bumps of a ~50-sample profile essentially never reach it.
MIN_PEAK_PROMINENCE_NOISE = 4.0

#: Samples closer to the peak than this many widths are excluded from the
#: refined (second-pass) baseline and noise estimate.
BASELINE_EXCLUSION_WIDTHS = 1.5

#: Unsaturated runs of at most this many samples between saturated samples
#: belong to the clipped top.
SATURATION_GAP_SAMPLES = 2

SignalUnits = Literal["normalized", "volts"]


class ProfileFlag(StrEnum):
    """Short machine-readable reasons recorded in ``ProfileAnalysis.flags``."""

    TOO_FEW_SAMPLES = "too_few_samples"
    NON_FINITE_SAMPLES = "non_finite_samples"
    DUPLICATE_Z = "duplicate_z"
    SATURATED = "saturated"
    ALL_SATURATED = "all_saturated"
    NO_PEAK = "no_peak"
    LOW_SNR = "low_snr"
    LOW_RELATIVE_PROMINENCE = "low_relative_prominence"
    PEAK_AT_EDGE = "peak_at_edge"
    MULTIPLE_PEAKS = "multiple_peaks"
    GLOBAL_MAX_NOT_SELECTED = "global_max_not_selected"
    PARABOLIC_FAILED = "parabolic_failed"
    GAUSSIAN_FAILED = "gaussian_failed"
    GAUSSIAN_OUTSIDE_SWEEP = "gaussian_outside_sweep"
    FIT_DISAGREEMENT = "fit_disagreement"
    WEAK_PEAK = "weak_peak"


# --------------------------------------------------------------------------- detection


@dataclass(frozen=True, slots=True)
class _Located:
    """Peak search result shared by the coarse and the fine analysis."""

    prepared: PreparedProfile
    filtered: NDArray[np.float64]  # prepared order
    clipped: NDArray[np.bool_]  # clipped top (prepared order)
    baseline: float | None
    noise_std: float | None
    peaks: PeakDetection
    peak_value: float | None  # smoothed signal at the main peak (clip level if clipped)
    snr: float | None
    relative_prominence: float | None
    flags: tuple[ProfileFlag, ...]

    @property
    def found(self) -> bool:
        """A significant peak that passes the SNR and relative-prominence thresholds."""
        return not {
            ProfileFlag.NO_PEAK,
            ProfileFlag.LOW_SNR,
            ProfileFlag.LOW_RELATIVE_PROMINENCE,
        } & set(self.flags)

    def input_index(self, info: PeakInfo) -> int:
        """Index of the peak sample in the caller's (unsorted) arrays."""
        return int(self.prepared.first_input_index[info.index])


def _detection_signal(
    filtered: NDArray[np.float64], signal: NDArray[np.float64], clipped: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Smoothed signal with the clipped top replaced by one exactly flat plateau.

    Smoothing rings at the corners of a flat top; capping everything at the
    clip level and flattening the clipped samples leaves a single plateau
    whose middle is the peak.
    """
    if not clipped.any():
        return filtered
    level = float(np.max(signal[clipped]))
    out = np.minimum(filtered, level)
    out[clipped] = level
    return out


def _detect(
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    baseline: float,
    noise_std: float,
    config: ProcessingConfig,
) -> PeakDetection:
    return detect_peaks(
        z,
        y,
        baseline=baseline,
        min_prominence=MIN_PEAK_PROMINENCE_NOISE * noise_std,
        edge_margin_samples=config.edge_margin_samples,
    )


def _locate(
    z: NDArray[np.float64],
    signal: NDArray[np.float64],
    saturated: NDArray[np.bool_],
    config: ProcessingConfig,
) -> _Located:
    prepared = prepare_profile(z, signal, saturated)
    flags: list[ProfileFlag] = []
    if prepared.n_dropped:
        flags.append(ProfileFlag.NON_FINITE_SAMPLES)
    if prepared.n_merged:
        flags.append(ProfileFlag.DUPLICATE_Z)
    if prepared.n < 3:
        flags.append(ProfileFlag.TOO_FEW_SAMPLES)

    filtered = filter_signal(prepared.signal, config)
    clipped = fill_short_gaps(prepared.saturated, SATURATION_GAP_SAMPLES)
    if clipped.any():
        flags.append(ProfileFlag.SATURATED)
    unclipped = np.where(clipped, np.nan, prepared.signal)
    if not np.isfinite(unclipped).any():
        if prepared.n:
            flags.append(ProfileFlag.ALL_SATURATED)
        flags.append(ProfileFlag.NO_PEAK)
        return _Located(
            prepared, filtered, clipped, None, None, NO_DETECTION, None, None, None, tuple(flags)
        )

    zs = prepared.z_um
    detect_y = _detection_signal(filtered, prepared.signal, clipped)
    baseline, noise = estimate_baseline_and_noise(unclipped)
    peaks = _detect(zs, detect_y, baseline, noise, config)
    width = None if peaks.main is None else peaks.main.width_estimate_um
    if peaks.main is not None and width is not None and width > 0.0:
        # Second pass: baseline and noise from the samples clearly off the peak.
        near_peak = np.abs(zs - peaks.main.z_refined_um) < BASELINE_EXCLUSION_WIDTHS * width
        if int(np.count_nonzero(~near_peak & ~clipped)) >= MIN_OFF_PEAK_SAMPLES:
            baseline, noise = estimate_baseline_and_noise(unclipped, exclude=near_peak)
            peaks = _detect(zs, detect_y, baseline, noise, config)

    main = peaks.main
    peak_value = None if main is None else float(detect_y[main.index])
    snr = compute_snr(peak_value, baseline, noise)
    rel = relative_prominence(None if main is None else main.prominence, peak_value, baseline)
    if main is None or not peaks.significant:
        if config.accept_weak_peaks and main is not None and main.prominence > 0.0:
            # Low-contrast mode: keep the best candidate; SNR / prominence below.
            flags.append(ProfileFlag.WEAK_PEAK)
        else:
            flags.append(ProfileFlag.NO_PEAK)
    if snr is None or snr < config.min_snr:
        flags.append(ProfileFlag.LOW_SNR)
    if rel is None or rel < config.min_relative_prominence:
        flags.append(ProfileFlag.LOW_RELATIVE_PROMINENCE)
    return _Located(
        prepared, filtered, clipped, baseline, noise, peaks, peak_value, snr, rel, tuple(flags)
    )


# --------------------------------------------------------------------------- coarse sweep


def find_coarse_peak(
    z_um: NDArray[np.float64],
    voltage_v: NDArray[np.float64],
    *,
    dark_v: float | None,
    config: ProcessingConfig,
) -> CoarsePeak:
    """Approximate peak of a coarse sweep (per-Z aggregated raw voltage).

    ``z_um`` is the sub-sample (three-point parabola) maximum, so the fine
    sweep is centred better than one coarse step; ``index`` refers to the
    caller's arrays. A peak within ``edge_margin_samples`` of a sweep end --
    including a signal still rising towards the end -- is reported with
    ``found=True, at_edge=True`` so the caller can widen the sweep;
    ``global_max_not_selected`` marks a selection the caller must not trust
    blindly. The thresholds are the unit-free ``min_snr`` and
    ``min_relative_prominence``.
    """
    z, v = as_profile_arrays(z_um, voltage_v)
    located = _locate(z, subtract_dark(v, dark_v), saturation_mask(v, config.saturation_v), config)
    main = located.peaks.main
    if main is None or not located.found:
        reasons = ", ".join(flag.value for flag in located.flags)
        return CoarsePeak(
            found=False,
            snr=located.snr,
            relative_prominence=located.relative_prominence,
            message=f"no usable peak in the coarse sweep ({reasons})",
        )
    ambiguous = located.peaks.global_max_not_selected
    messages = []
    if main.at_edge:
        messages.append("peak at the edge of the sweep")
    if ambiguous:
        messages.append("the brightest signal of the sweep is not the selected peak")
    return CoarsePeak(
        found=True,
        z_um=main.z_refined_um,
        index=located.input_index(main),
        at_edge=main.at_edge,
        global_max_not_selected=ambiguous,
        snr=located.snr,
        relative_prominence=located.relative_prominence,
        message="; ".join(messages) or None,
    )


# --------------------------------------------------------------------------- fine sweep


@dataclass(frozen=True, slots=True)
class _Selection:
    fit: PeakFit | None
    disagreement_um: float | None
    flags: tuple[ProfileFlag, ...]


def _fit_mask(
    z: NDArray[np.float64],
    main: PeakInfo,
    clipped: NDArray[np.bool_],
    fit_window_fwhm: float,
) -> NDArray[np.bool_]:
    """Unclipped samples within +/- ``fit_window_fwhm`` widths of the peak.

    Grown to the :data:`MIN_GAUSSIAN_POINTS` unclipped samples nearest to the
    peak when the window is too small; the whole profile when the width is
    unknown.
    """
    usable = ~clipped
    width = main.width_estimate_um
    if width is None or not width > 0.0:
        return usable
    distance = np.abs(z - main.z_refined_um)
    mask = usable & (distance <= fit_window_fwhm * width)
    if int(np.count_nonzero(mask)) < MIN_GAUSSIAN_POINTS:
        candidates = np.flatnonzero(usable)
        nearest = candidates[np.argsort(distance[candidates], kind="stable")]
        mask[nearest[:MIN_GAUSSIAN_POINTS]] = True
    return mask


def _select_surface(
    parabolic: PeakFit,
    gaussian: PeakFit,
    z_range: tuple[float, float],
    max_disagreement_um: float,
) -> _Selection:
    flags: list[ProfileFlag] = []
    g = gaussian.center_um if gaussian.success else None
    p = parabolic.center_um if parabolic.success else None
    if g is None:
        flags.append(ProfileFlag.GAUSSIAN_FAILED)
    if p is None:
        flags.append(ProfileFlag.PARABOLIC_FAILED)
    inside = g is not None and z_range[0] <= g <= z_range[1]
    if g is not None and not inside:
        flags.append(ProfileFlag.GAUSSIAN_OUTSIDE_SWEEP)
    disagreement = abs(g - p) if g is not None and p is not None else None
    agrees = disagreement is not None and disagreement <= max_disagreement_um
    if disagreement is not None and not agrees:
        flags.append(ProfileFlag.FIT_DISAGREEMENT)
    if inside and agrees:
        fit: PeakFit | None = gaussian
    elif p is not None:
        fit = parabolic
    else:
        fit = None
    return _Selection(fit=fit, disagreement_um=disagreement, flags=tuple(flags))


def _residual_ratio(fit: PeakFit) -> float | None:
    if fit.residual_rms is None or fit.amplitude is None or not fit.amplitude > 0.0:
        return None
    return fit.residual_rms / fit.amplitude


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _status(
    located: _Located, main: PeakInfo, surface_z: float | None, confidence: float, minimum: float
) -> PointStatus:
    if not located.found:
        return PointStatus.NO_PEAK
    if main.at_edge:
        return PointStatus.PEAK_AT_EDGE
    if surface_z is None:
        return PointStatus.FIT_FAILED
    weak = ProfileFlag.WEAK_PEAK in located.flags
    if confidence < minimum or located.peaks.global_max_not_selected or weak:
        return PointStatus.LOW_CONFIDENCE
    return PointStatus.VALID


def _analyse(
    located: _Located, config: ProcessingConfig, signal_units: SignalUnits
) -> ProfileAnalysis:
    prepared = located.prepared
    main = located.peaks.main
    flags = list(located.flags)
    if main is None or not located.found:
        return ProfileAnalysis(
            status=PointStatus.NO_PEAK,
            signal_units=signal_units,
            peak_index=None if main is None else located.input_index(main),
            peak_z_um=None if main is None else main.z_um,
            peak_intensity=_finite_or_none(located.peak_value),
            baseline=located.baseline,
            noise_std=located.noise_std,
            snr=_finite_or_none(located.snr),
            fwhm_um=None if main is None else _finite_or_none(main.fwhm_um),
            prominence=None if main is None else main.prominence,
            relative_prominence=located.relative_prominence,
            confidence=0.0,
            saturated_fraction=prepared.saturated_fraction,
            flags=[flag.value for flag in flags],
        )

    z, y = prepared.z_um, prepared.signal
    mask = _fit_mask(z, main, located.clipped, config.fit_window_fwhm)
    width = main.width_estimate_um
    parabolic = fit_parabolic(
        z[mask], y[mask], baseline=located.baseline, select_by=located.filtered[mask]
    )
    gaussian = fit_gaussian(
        z[mask],
        y[mask],
        center_guess=main.z_refined_um,
        sigma_guess=None if width is None else width / GAUSSIAN_FWHM_PER_SIGMA,
        offset_guess=located.baseline,
    )
    selection = _select_surface(
        parabolic, gaussian, (float(z[0]), float(z[-1])), config.max_fit_disagreement_um
    )
    flags.extend(selection.flags)
    if main.at_edge:
        flags.append(ProfileFlag.PEAK_AT_EDGE)
    if located.peaks.n_peaks > 1:
        flags.append(ProfileFlag.MULTIPLE_PEAKS)
    if located.peaks.global_max_not_selected:
        flags.append(ProfileFlag.GLOBAL_MAX_NOT_SELECTED)

    fit = selection.fit
    surface_z = None if fit is None else fit.center_um
    # Model FWHM when the Gaussian defines the height: unlike the half-maximum
    # crossings it does not depend on the (conservative) baseline estimate.
    fwhm = (
        gaussian.fwhm_um
        if fit is not None and fit.method is FitMethod.GAUSSIAN
        else _finite_or_none(main.fwhm_um)
    )
    confidence = 0.0
    if fit is not None:
        confidence = confidence_score(
            PeakQuality(
                snr=located.snr,
                relative_prominence=located.relative_prominence,
                r_squared=fit.r_squared,
                residual_ratio=_residual_ratio(fit),
                fit_disagreement_um=selection.disagreement_um,
                max_fit_disagreement_um=config.max_fit_disagreement_um,
                fwhm_um=fwhm,
                expected_fwhm_um=config.expected_fwhm_um,
                # An ambiguous selection is penalised like a truncated peak.
                at_edge=main.at_edge or located.peaks.global_max_not_selected,
                saturated_fraction=prepared.saturated_fraction,
                secondary_peak_ratio=located.peaks.secondary_peak_ratio,
            )
        )
    return ProfileAnalysis(
        status=_status(located, main, surface_z, confidence, config.min_confidence),
        signal_units=signal_units,
        peak_found=True,
        peak_index=located.input_index(main),
        peak_z_um=main.z_um,
        peak_intensity=_finite_or_none(located.peak_value),
        baseline=located.baseline,
        noise_std=located.noise_std,
        parabolic=parabolic,
        gaussian=gaussian,
        surface_z_um=surface_z,
        surface_method=None if fit is None else fit.method,
        snr=_finite_or_none(located.snr),
        fwhm_um=fwhm,
        prominence=main.prominence,
        relative_prominence=located.relative_prominence,
        fit_residual=None if fit is None else fit.residual_rms,
        confidence=confidence,
        n_peaks=located.peaks.n_peaks,
        secondary_peak_ratio=located.peaks.secondary_peak_ratio,
        asymmetry=main.asymmetry,
        saturated_fraction=prepared.saturated_fraction,
        flags=[flag.value for flag in flags],
    )


def analyse_profile(
    z_um: NDArray[np.float64],
    voltage_v: NDArray[np.float64],
    *,
    dark_v: float | None,
    reference_v: float | None,
    config: ProcessingConfig,
) -> ProfileProcessingResult:
    """Full physics analysis of one fine sweep (see the module docstring).

    ``z_um`` are the stage-*reported* positions and ``voltage_v`` the per-Z
    aggregated raw voltages, in acquisition order; they may be unsorted and
    contain repeated Z values or NaN. The returned arrays keep the input
    order: ``corrected_v`` = V - dark, ``normalized`` = (V - dark) /
    (reference - dark) or None without a reference, ``filtered`` = the
    smoothed analysis signal (NaN for dropped samples). Bad data never
    raises; it yields a NO_PEAK / FIT_FAILED / LOW_CONFIDENCE analysis.

    Raises:
        ValueError: the arrays are not 1-D or differ in length (caller bug).
        CalibrationError: ``reference_v`` is not above ``dark_v``.
    """
    z, v = as_profile_arrays(z_um, voltage_v)
    corrected = subtract_dark(v, dark_v)
    normalized = normalize(v, dark_v, reference_v)
    signal = corrected if normalized is None else normalized
    units: SignalUnits = "volts" if normalized is None else "normalized"
    located = _locate(z, signal, saturation_mask(v, config.saturation_v), config)
    return ProfileProcessingResult(
        analysis=_analyse(located, config, units),
        corrected_v=corrected,
        normalized=normalized,
        filtered=located.prepared.to_input_order(located.filtered),
    )
