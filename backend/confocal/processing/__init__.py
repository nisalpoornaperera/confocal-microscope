"""Pure signal processing of intensity-vs-Z profiles (the physics baseline).

``analyse_profile`` and ``find_coarse_peak`` implement the ``ProfileAnalyser``
and ``CoarsePeakFinder`` protocols of ``confocal.scanning.protocols``; the
other names are the building blocks they are made of.
"""

from confocal.processing.fitting import GAUSSIAN_FWHM_PER_SIGMA, fit_gaussian, fit_parabolic
from confocal.processing.metrics import (
    ConfidenceFactors,
    PeakQuality,
    compute_snr,
    confidence_factors,
    confidence_score,
    relative_prominence,
)
from confocal.processing.normalization import aggregate_samples, normalize, subtract_dark
from confocal.processing.peaks import PeakDetection, PeakInfo, detect_peaks
from confocal.processing.profile import ProfileFlag, analyse_profile, find_coarse_peak
from confocal.processing.signal import estimate_baseline_and_noise, filter_signal

__all__ = [
    "GAUSSIAN_FWHM_PER_SIGMA",
    "ConfidenceFactors",
    "PeakDetection",
    "PeakInfo",
    "PeakQuality",
    "ProfileFlag",
    "aggregate_samples",
    "analyse_profile",
    "compute_snr",
    "confidence_factors",
    "confidence_score",
    "detect_peaks",
    "estimate_baseline_and_noise",
    "filter_signal",
    "find_coarse_peak",
    "fit_gaussian",
    "fit_parabolic",
    "normalize",
    "relative_prominence",
    "subtract_dark",
]
