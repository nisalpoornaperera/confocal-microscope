"""Figures of merit of one confocal peak and the confidence score.

``confidence_score`` turns the independent quality indicators of a profile
analysis into one number in [0, 1]. It is the *product* of one factor per
indicator, each in [0, 1], so every failure mode lowers the score on its own
and no good indicator can compensate for a bad one::

    confidence = q_snr * q_prominence * q_fit * q_agreement * q_width
                 * p_edge * p_saturation * p_secondary

    q_snr        = 1 - 2 ** (-SNR / SNR_HALF)            0.5 at SNR = 4, 0.97 at 20
    q_prominence = sqrt(relative prominence)              topographic, clipped to [0, 1]
    q_fit        = R^2 / (1 + (rho / RESIDUAL_HALF)^2)     rho = residual RMS / amplitude
    q_agreement  = 1 / (1 + (d / max_fit_disagreement)^2) d = |parabolic - Gaussian| centre;
                   UNVERIFIED_AGREEMENT when only one fit succeeded
    q_width      = exp(-(ln(FWHM / expected) / ln WIDTH_TOLERANCE)^2 / 2)
                   (1 when no expected FWHM is configured, UNKNOWN_WIDTH without a width)
    p_edge       = EDGE_PENALTY at the sweep edge, else 1
    p_saturation = SATURATION_PENALTY * (1 - saturated fraction) if any sample saturated
    p_secondary  = 1 - SECONDARY_PEAK_WEIGHT * secondary-peak ratio

Every factor is non-decreasing in the quality it measures, so the score is
monotonic in SNR (strictly, while the other factors are non-zero), in
relative prominence and in fit quality, and decreasing in the disagreement,
the saturated fraction and the secondary-peak ratio. A missing indicator
(``None``) scores as a failure (0) except where noted, because a quantity
that could not be measured cannot vouch for the point.

The constants are deliberately round: they encode judgement ("a single
unverified estimate is at most half as trustworthy"), not a fitted model.
The ML layer can learn a calibrated confidence from ground truth; this score
is the transparent physics baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: SNR at which the SNR factor is 0.5.
SNR_HALF = 4.0
#: Residual RMS / amplitude at which the residual part of the fit factor is 0.5.
RESIDUAL_HALF = 0.2
#: Agreement factor when only one of the two fits succeeded (nothing to compare).
UNVERIFIED_AGREEMENT = 0.5
#: Width ratio (either way) at which the width factor is exp(-1/2) = 0.61.
WIDTH_TOLERANCE = 1.5
#: Width factor when an expected FWHM is configured but no width could be measured.
UNKNOWN_WIDTH = 0.5
#: Factor for a peak within ``edge_margin_samples`` of the sweep end.
EDGE_PENALTY = 0.3
#: Factor for any saturation (further scaled by the unsaturated fraction).
SATURATION_PENALTY = 0.7
#: Weight of the secondary-peak ratio (a secondary peak as strong as the main: 0.2).
SECONDARY_PEAK_WEIGHT = 0.8


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def compute_snr(
    peak: float | None, baseline: float | None, noise_std: float | None
) -> float | None:
    """``(peak - baseline) / noise_std``; None when undefined (no noise estimate)."""
    peak, baseline, noise_std = _finite(peak), _finite(baseline), _finite(noise_std)
    if peak is None or baseline is None or noise_std is None or not noise_std > 0.0:
        return None
    return (peak - baseline) / noise_std


def relative_prominence(
    prominence: float | None, peak: float | None, baseline: float | None
) -> float | None:
    """Prominence / peak height above baseline, clipped to [0, 1].

    1 means the signal falls all the way to the baseline on both sides of the
    peak; a value near 0 is a bump on a shoulder or a plateau. The ratio can
    exceed 1 when the off-peak baseline is over-estimated (a fine sweep that
    hardly leaves the peak); it is clipped because "falls to the baseline" is
    the best a peak can do. None when the peak is not above the baseline.
    """
    prominence, peak, baseline = _finite(prominence), _finite(peak), _finite(baseline)
    if prominence is None or peak is None or baseline is None:
        return None
    height = peak - baseline
    if not height > 0.0:
        return None
    return min(1.0, max(0.0, prominence / height))


@dataclass(frozen=True, slots=True)
class PeakQuality:
    """The indicators that ``confidence_score`` combines (see the module docstring)."""

    snr: float | None
    relative_prominence: float | None
    r_squared: float | None  # of the fit that defines the surface height
    residual_ratio: float | None  # residual RMS / amplitude of that fit
    fit_disagreement_um: float | None  # |parabolic - Gaussian|; None unless both succeeded
    max_fit_disagreement_um: float = 1.0
    fwhm_um: float | None = None
    expected_fwhm_um: float | None = None
    at_edge: bool = False
    saturated_fraction: float = 0.0
    secondary_peak_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class ConfidenceFactors:
    """Individual factors of the confidence score, each in [0, 1]."""

    snr: float
    prominence: float
    fit_quality: float
    agreement: float
    width: float
    edge: float
    saturation: float
    secondary_peak: float

    @property
    def score(self) -> float:
        product = (
            self.snr
            * self.prominence
            * self.fit_quality
            * self.agreement
            * self.width
            * self.edge
            * self.saturation
            * self.secondary_peak
        )
        return min(1.0, max(0.0, product))


def _snr_factor(snr: float | None) -> float:
    snr = _finite(snr)
    if snr is None or snr <= 0.0:
        return 0.0
    return 1.0 - math.pow(2.0, -snr / SNR_HALF)


def _prominence_factor(rel: float | None) -> float:
    rel = _finite(rel)
    return 0.0 if rel is None else math.sqrt(min(1.0, max(0.0, rel)))


def _fit_factor(r_squared: float | None, residual_ratio: float | None) -> float:
    r_squared, residual_ratio = _finite(r_squared), _finite(residual_ratio)
    if r_squared is None or residual_ratio is None or residual_ratio < 0.0:
        return 0.0
    return min(1.0, max(0.0, r_squared)) / (1.0 + (residual_ratio / RESIDUAL_HALF) ** 2)


def _agreement_factor(disagreement: float | None, tolerance: float) -> float:
    disagreement = _finite(disagreement)
    if disagreement is None:
        return UNVERIFIED_AGREEMENT
    return 1.0 / (1.0 + (abs(disagreement) / tolerance) ** 2)


def _width_factor(fwhm: float | None, expected: float | None) -> float:
    if expected is None:
        return 1.0
    fwhm = _finite(fwhm)
    if fwhm is None or fwhm <= 0.0:
        return UNKNOWN_WIDTH
    log_ratio = math.log(fwhm / expected) / math.log(WIDTH_TOLERANCE)
    return math.exp(-0.5 * log_ratio * log_ratio)


def _saturation_factor(fraction: float) -> float:
    fraction = min(1.0, max(0.0, fraction)) if math.isfinite(fraction) else 1.0
    return 1.0 if fraction <= 0.0 else SATURATION_PENALTY * (1.0 - fraction)


def _secondary_factor(ratio: float | None) -> float:
    ratio = _finite(ratio)
    if ratio is None:
        return 1.0
    return 1.0 - SECONDARY_PEAK_WEIGHT * min(1.0, max(0.0, ratio))


def confidence_factors(quality: PeakQuality) -> ConfidenceFactors:
    """Evaluate every factor of the confidence formula for ``quality``."""
    if not quality.max_fit_disagreement_um > 0.0:
        raise ValueError("max_fit_disagreement_um must be positive")
    if quality.expected_fwhm_um is not None and not quality.expected_fwhm_um > 0.0:
        raise ValueError("expected_fwhm_um must be positive")
    return ConfidenceFactors(
        snr=_snr_factor(quality.snr),
        prominence=_prominence_factor(quality.relative_prominence),
        fit_quality=_fit_factor(quality.r_squared, quality.residual_ratio),
        agreement=_agreement_factor(quality.fit_disagreement_um, quality.max_fit_disagreement_um),
        width=_width_factor(quality.fwhm_um, quality.expected_fwhm_um),
        edge=EDGE_PENALTY if quality.at_edge else 1.0,
        saturation=_saturation_factor(quality.saturated_fraction),
        secondary_peak=_secondary_factor(quality.secondary_peak_ratio),
    )


def confidence_score(quality: PeakQuality) -> float:
    """Confidence in [0, 1] that the peak defines the surface height (see module docstring)."""
    return confidence_factors(quality).score
