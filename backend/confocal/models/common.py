"""Coordinate and limit types. All application coordinates are micrometres."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Timezone-aware current UTC time. Use this everywhere instead of ``datetime.now()``."""
    return datetime.now(UTC)


class Axis(StrEnum):
    X = "x"
    Y = "y"
    Z = "z"


class Position(BaseModel):
    """A stage position in micrometres.

    This is the only coordinate type used above the hardware layer. Motor
    steps never appear outside ``confocal.hardware.arduino``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x_um: float = Field(allow_inf_nan=False)
    y_um: float = Field(allow_inf_nan=False)
    z_um: float = Field(allow_inf_nan=False)

    def get(self, axis: Axis) -> float:
        if axis is Axis.X:
            return self.x_um
        if axis is Axis.Y:
            return self.y_um
        return self.z_um

    def with_axis(self, axis: Axis, value: float) -> Position:
        return self.model_copy(update={f"{axis.value}_um": float(value)})

    def with_updates(
        self,
        *,
        x_um: float | None = None,
        y_um: float | None = None,
        z_um: float | None = None,
    ) -> Position:
        """Return a copy with the given axes replaced (``None`` keeps the current value)."""
        return Position(
            x_um=self.x_um if x_um is None else float(x_um),
            y_um=self.y_um if y_um is None else float(y_um),
            z_um=self.z_um if z_um is None else float(z_um),
        )

    def offset(self, dx_um: float = 0.0, dy_um: float = 0.0, dz_um: float = 0.0) -> Position:
        return Position(x_um=self.x_um + dx_um, y_um=self.y_um + dy_um, z_um=self.z_um + dz_um)

    def distance_to(self, other: Position) -> float:
        return math.dist(self.as_tuple(), other.as_tuple())

    def max_axis_error(self, other: Position) -> float:
        """Largest per-axis absolute difference, used for move verification."""
        return max(
            abs(self.x_um - other.x_um),
            abs(self.y_um - other.y_um),
            abs(self.z_um - other.z_um),
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x_um, self.y_um, self.z_um)


class AxisLimits(BaseModel):
    """Inclusive software travel limits for one axis, in micrometres."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_um: float = Field(allow_inf_nan=False)
    max_um: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_order(self) -> AxisLimits:
        if not self.min_um < self.max_um:
            raise ValueError(f"min_um ({self.min_um}) must be < max_um ({self.max_um})")
        return self

    @property
    def span_um(self) -> float:
        return self.max_um - self.min_um

    def contains(self, value: float) -> bool:
        return math.isfinite(value) and self.min_um <= value <= self.max_um

    def clamp(self, value: float) -> float:
        return min(max(value, self.min_um), self.max_um)


class StageLimits(BaseModel):
    """Software travel limits for all three axes.

    The OpenFlexure stage driven by 28BYJ-48 motors has no end-stops, so these
    limits are relative to the origin defined at power-up / zeroing and must be
    set conservatively for each machine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: AxisLimits
    y: AxisLimits
    z: AxisLimits

    def for_axis(self, axis: Axis) -> AxisLimits:
        if axis is Axis.X:
            return self.x
        if axis is Axis.Y:
            return self.y
        return self.z

    def violations(self, position: Position) -> list[str]:
        """Human-readable description of every axis that is out of range (empty if safe)."""
        problems: list[str] = []
        for axis in Axis:
            limits = self.for_axis(axis)
            value = position.get(axis)
            if not limits.contains(value):
                problems.append(
                    f"{axis.value}={value:.3f} um outside "
                    f"[{limits.min_um:.3f}, {limits.max_um:.3f}] um"
                )
        return problems

    def contains(self, position: Position) -> bool:
        return not self.violations(position)


def default_stage_limits() -> StageLimits:
    """Conservative defaults for an OpenFlexure high-resolution stage (set per machine)."""
    return StageLimits(
        x=AxisLimits(min_um=-5000.0, max_um=5000.0),
        y=AxisLimits(min_um=-5000.0, max_um=5000.0),
        z=AxisLimits(min_um=-2000.0, max_um=2000.0),
    )
