"""Construction of the per-point ``ScanPoint`` results from what was measured and analysed.

``ScanPoint`` is API-facing, so every float copied from an analysis is passed
through :func:`finite_or_none`: a NaN produced by a numerical corner case
becomes ``None`` instead of breaking JSON serialisation later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from confocal.models.processing import CoarsePeak, PeakFit, PointStatus, ProfileAnalysis
from confocal.models.scan import ScanPoint
from confocal.scanning.plan import GridPoint
from confocal.scanning.profile_buffer import ProfileBuffer, SignalUnits

#: ScanPoint flags set by the executor (the analyser adds its own).
FLAG_COARSE_RETRY = "coarse_retry_full_range"
FLAG_COARSE_CLIPPED = "coarse_sweep_clipped"
FLAG_FINE_CLIPPED = "fine_sweep_clipped"
FLAG_NO_COARSE_PEAK = "no_coarse_peak"
FLAG_ABORTED = "aborted"
FLAG_ANALYSIS_FAILED = "analysis_failed"
#: At least one raw ADC code of the point sits on a converter rail: the true
#: signal exceeded the PGA full scale there (same name as the analyser's flag
#: for a clipped top, so a point never carries both spellings).
FLAG_SATURATED = "saturated"


def finite_or_none(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


@dataclass(slots=True)
class PointContext:
    """Working state of the point being measured (also what an aborted point keeps)."""

    grid_point: GridPoint
    buffer: ProfileBuffer
    started_s: float
    z_estimate_um: float | None = None
    coarse_peak: CoarsePeak | None = None
    flags: list[str] = field(default_factory=list)

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)

    @property
    def coarse_peak_z_um(self) -> float | None:
        peak = self.coarse_peak
        return finite_or_none(peak.z_um) if peak is not None and peak.found else None


def _fit_center(fit: PeakFit | None) -> float | None:
    return finite_or_none(fit.center_um) if fit is not None and fit.success else None


def no_peak_analysis(signal_units: SignalUnits) -> ProfileAnalysis:
    """Analysis of a point whose coarse sweep(s) found no peak: no fine sweep was made."""
    return ProfileAnalysis(
        status=PointStatus.NO_PEAK, signal_units=signal_units, flags=[FLAG_NO_COARSE_PEAK]
    )


def confocal_point(ctx: PointContext, analysis: ProfileAnalysis, *, duration_s: float) -> ScanPoint:
    """Result of a confocal point: every scalar of the profile analysis."""
    gp = ctx.grid_point
    return ScanPoint(
        point_id=gp.point_id,
        ix=gp.ix,
        iy=gp.iy,
        x_um=gp.x_um,
        y_um=gp.y_um,
        status=analysis.status,
        z_estimate_um=finite_or_none(ctx.z_estimate_um),
        coarse_peak_z_um=ctx.coarse_peak_z_um,
        surface_z_um=finite_or_none(analysis.surface_z_um),
        parabolic_z_um=_fit_center(analysis.parabolic),
        gaussian_z_um=_fit_center(analysis.gaussian),
        peak_intensity=finite_or_none(analysis.peak_intensity),
        snr=finite_or_none(analysis.snr),
        peak_width_um=finite_or_none(analysis.fwhm_um),
        prominence=finite_or_none(analysis.prominence),
        fit_residual=finite_or_none(analysis.fit_residual),
        confidence=analysis.confidence,
        secondary_peak_ratio=finite_or_none(analysis.secondary_peak_ratio),
        asymmetry=finite_or_none(analysis.asymmetry),
        n_z_positions=ctx.buffer.n,
        flags=[*ctx.flags, *(f for f in analysis.flags if f not in ctx.flags)],
        duration_s=duration_s,
    )


def fixed_z_point(ctx: PointContext, intensity: float, *, duration_s: float) -> ScanPoint:
    """Result of a fixed-Z point: one calibrated intensity, no surface height."""
    gp = ctx.grid_point
    return ScanPoint(
        point_id=gp.point_id,
        ix=gp.ix,
        iy=gp.iy,
        x_um=gp.x_um,
        y_um=gp.y_um,
        status=PointStatus.MEASURED,
        intensity=finite_or_none(intensity),
        n_z_positions=ctx.buffer.n,
        flags=list(ctx.flags),
        duration_s=duration_s,
    )


def failed_point(
    ctx: PointContext, status: PointStatus, flag: str, *, duration_s: float
) -> ScanPoint:
    """Result of a point that was interrupted (ABORTED) or could not be analysed (ERROR)."""
    gp = ctx.grid_point
    return ScanPoint(
        point_id=gp.point_id,
        ix=gp.ix,
        iy=gp.iy,
        x_um=gp.x_um,
        y_um=gp.y_um,
        status=status,
        z_estimate_um=finite_or_none(ctx.z_estimate_um),
        coarse_peak_z_um=ctx.coarse_peak_z_um,
        n_z_positions=ctx.buffer.n,
        flags=[*ctx.flags, flag],
        duration_s=duration_s,
    )
