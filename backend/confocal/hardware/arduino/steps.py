"""Cartesian micrometres <-> Arduino motor steps, travel checks and backlash planning.

This is the **only** place in the application where micrometres become motor
steps. Everything above the hardware layer works in micrometres; the Arduino
firmware works only in motor steps of motors ``a``, ``b`` and ``c`` (the three
legs of the Delta Stage).

Conversion pipeline::

    Position (um) --kinematics.to_steps--> kinematic steps --invert--> firmware steps

``kinematics`` (``StageKinematics``) is the linear delta / cartesian / measured
map; a motor configured with ``invert = true`` turns the other way round (its
coil order or mounting is mirrored), so its sign is flipped. The resulting
*firmware steps* are what the Arduino counts, what ``MOVE`` carries and what
``MotorConfig.min_steps`` / ``max_steps`` (each actuator's safe travel) refer to.

Rounding to whole steps is done once, by the kinematics, half away from zero.
On the delta stage the step grid is not aligned with X/Y/Z, so a commanded
position is reached only to within :meth:`StepMapper.resolution_um`
(about 0.04 um with the default geometry) - far below the move-verification
tolerance, but the reason the stage reports the *quantised* position.

Backlash: the 28BYJ-48 gearbox has tens of steps of play. Approaching every
target from the same side makes positioning repeatable, so every motor
finishes its move travelling in its *logical* + direction, i.e. the
kinematic direction before ``invert`` is applied (see
:func:`plan_backlash_moves`). A Z sweep in increasing Z drives all three legs
logically upwards and therefore needs no compensation at all, whichever motors
are inverted. The overshoot waypoint is itself a platform position: it is kept
inside the motor travel *and* inside the Cartesian travel limits (shrunk, or
skipped with a warning, when there is no room for it).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

from confocal.config import ArduinoConfig, MotorConfig, Settings
from confocal.errors import LimitViolationError
from confocal.hardware.kinematics import MOTORS, Motor, MotorSteps, StageKinematics
from confocal.models.common import Axis, Position, StageLimits

log = logging.getLogger(__name__)

#: Bisection steps used to shrink an overshoot that would leave the limits (2**-30 resolution).
_SHRINK_ITERATIONS = 30


def motor_configs(arduino: ArduinoConfig) -> dict[Motor, MotorConfig]:
    """The per-motor sections ``[arduino.a]``, ``[arduino.b]``, ``[arduino.c]``."""
    return {Motor.A: arduino.a, Motor.B: arduino.b, Motor.C: arduino.c}


def plan_backlash_moves(
    current: MotorSteps,
    target: MotorSteps,
    backlash: Mapping[Motor, int],
    travel: Mapping[Motor, tuple[int, int]] | None = None,
    *,
    signs: Sequence[int] = (1, 1, 1),
    admissible: Callable[[MotorSteps], bool] | None = None,
) -> list[MotorSteps]:
    """Waypoints (absolute firmware steps) such that every motor finishes moving logically up.

    ``signs`` are the per-motor firmware signs (``-1`` for an ``invert``-ed
    motor): ``sign * firmware_steps`` is the motor's logical (kinematic) step
    count. A motor that has to move logically down first overshoots to
    ``backlash`` steps logically below its target (clamped to its ``travel``)
    and then comes back up to the target in the final coordinated move, taking
    up the gear play in the logical + direction. Motors moving logically up go
    straight to their target in the first waypoint. When no motor moves
    logically down (e.g. every step of an increasing Z sweep, whatever the
    inversion) or no such motor has backlash, the plan is the single waypoint
    ``[target]``. The last waypoint is always ``target``.

    ``admissible`` (if given) tells whether a waypoint is a safe platform
    position (the Cartesian limit check). An overshoot that is not admissible
    is shrunk towards the target to the largest admissible fraction; if no
    overshoot fits, it is skipped (the plan is ``[target]``). Either case is
    logged as a warning, because positioning is less repeatable there.

    Raises:
        ValueError: for a negative backlash or a sign other than +1 / -1.
    """
    if len(signs) != len(MOTORS) or any(sign not in (1, -1) for sign in signs):
        raise ValueError(f"motor signs must be +1 or -1, got {tuple(signs)}")
    overshoot: list[int] = []
    for motor, sign, now, goal in zip(MOTORS, signs, current, target, strict=True):
        play = int(backlash.get(motor, 0))
        if play < 0:
            raise ValueError(f"backlash of motor {motor.value} must be >= 0")
        if sign * (goal - now) < 0 and play > 0:
            dip = goal - sign * play  # `play` steps logically below the target
            if travel is not None:
                # Never overshoot past the end of the travel, and never "overshoot" the
                # wrong way for a target already beyond it (check_travel rejects that).
                low, high = travel[motor]
                dip = min(max(dip, low), high)
                dip = min(goal, dip) if sign > 0 else max(goal, dip)
            overshoot.append(dip)
        else:
            overshoot.append(goal)
    first: MotorSteps = (overshoot[0], overshoot[1], overshoot[2])
    if first == target:
        return [target]
    if admissible is not None and not admissible(first):
        shrunk = _shrink_overshoot(target, first, admissible)
        if shrunk is None:
            log.warning(
                "backlash overshoot %s for target %s skipped: it would leave the travel "
                "limits (positioning there is less repeatable)",
                first,
                target,
            )
            return [target]
        log.warning(
            "backlash overshoot %s for target %s shrunk to %s to stay inside the travel limits",
            first,
            target,
            shrunk,
        )
        first = shrunk
    return [first, target]


def _shrink_overshoot(
    target: MotorSteps, overshoot: MotorSteps, admissible: Callable[[MotorSteps], bool]
) -> MotorSteps | None:
    """Largest admissible waypoint on the segment target -> overshoot (None if none).

    The step map is linear and the limit box convex, so the admissible part of
    the segment is an interval starting at the target: bisect its end.
    """

    def scaled(fraction: float) -> MotorSteps:
        # int() truncates towards zero, i.e. towards the target: never beyond `fraction`.
        a, b, c = (
            goal + int(fraction * (dip - goal)) for goal, dip in zip(target, overshoot, strict=True)
        )
        return (a, b, c)

    low, high = 0.0, 1.0  # scaled(high) is not admissible
    for _ in range(_SHRINK_ITERATIONS):
        middle = 0.5 * (low + high)
        if admissible(scaled(middle)):
            low = middle
        else:
            high = middle
    best = scaled(low)
    if best == target or not admissible(best):
        return None
    return best


def _excess_um(limits: StageLimits, position: Position) -> dict[Axis, float]:
    """How far ``position`` lies outside ``limits`` on each axis (0 when inside)."""
    excess: dict[Axis, float] = {}
    for axis in Axis:
        axis_limits = limits.for_axis(axis)
        value = position.get(axis)
        excess[axis] = max(0.0, axis_limits.min_um - value, value - axis_limits.max_um)
    return excess


def _within(limits: StageLimits, position: Position, margin_um: Mapping[Axis, float]) -> bool:
    """``position`` inside ``limits`` widened by ``margin_um[axis]`` on each axis."""
    excess = _excess_um(limits, position)
    return all(excess[axis] <= margin_um[axis] for axis in Axis)


class StepMapper:
    """Cartesian micrometres <-> firmware motor steps for the Arduino stage."""

    def __init__(self, kinematics: StageKinematics, motors: Mapping[Motor, MotorConfig]) -> None:
        missing = [motor.value for motor in MOTORS if motor not in motors]
        if missing:
            raise ValueError(f"motor configuration missing for motors: {', '.join(missing)}")
        self._kinematics = kinematics
        self._motors: dict[Motor, MotorConfig] = {motor: motors[motor] for motor in MOTORS}
        self._signs = tuple(-1 if self._motors[m].invert else 1 for m in MOTORS)

    @classmethod
    def from_settings(cls, settings: Settings) -> StepMapper:
        return cls(
            StageKinematics.from_config(settings.kinematics), motor_configs(settings.arduino)
        )

    # ------------------------------------------------------------------ configuration
    @property
    def kinematics(self) -> StageKinematics:
        return self._kinematics

    def motor_config(self, motor: Motor) -> MotorConfig:
        return self._motors[motor]

    @property
    def travel(self) -> dict[Motor, tuple[int, int]]:
        """Each motor's safe travel ``(min_steps, max_steps)`` in firmware steps."""
        return {m: (cfg.min_steps, cfg.max_steps) for m, cfg in self._motors.items()}

    @property
    def backlash(self) -> dict[Motor, int]:
        return {m: cfg.backlash_steps for m, cfg in self._motors.items()}

    def resolution_um(self) -> float:
        """Upper bound of the per-axis error introduced by rounding to whole steps."""
        return self._kinematics.max_quantization_error_um()

    # ------------------------------------------------------------------ conversion
    def _flip(self, steps: MotorSteps) -> MotorSteps:
        a, b, c = (sign * int(value) for sign, value in zip(self._signs, steps, strict=True))
        return (a, b, c)

    def to_motor_steps(self, position: Position) -> MotorSteps:
        """Firmware steps (inversion applied) nearest to ``position``."""
        return self._flip(self._kinematics.to_steps(position))

    def to_position(self, steps: MotorSteps) -> Position:
        """Cartesian position of the given firmware step counters."""
        return self._kinematics.to_position(self._flip(steps))  # the flip is its own inverse

    def quantize(self, position: Position) -> Position:
        """The position the stage actually reaches when commanded to ``position``."""
        return self.to_position(self.to_motor_steps(position))

    # ------------------------------------------------------------------ safety
    def check_travel(self, steps: MotorSteps) -> None:
        """Host-side defence: refuse firmware steps outside a motor's travel.

        Raises:
            LimitViolationError: listing every motor out of range (nothing moves).
        """
        travel = self.travel
        violations = [
            f"motor {motor.value} step {value} outside travel "
            f"[{travel[motor][0]}, {travel[motor][1]}]"
            for motor, value in zip(MOTORS, steps, strict=True)
            if not travel[motor][0] <= value <= travel[motor][1]
        ]
        if violations:
            raise LimitViolationError(
                "target outside motor travel: " + "; ".join(violations), violations=violations
            )

    def limit_violations(self, limits: StageLimits) -> list[str]:
        """Motors whose travel cannot cover the Cartesian limit box (empty if safe).

        Same check as ``StageKinematics.config_limit_violations`` but in
        firmware steps, i.e. honouring ``invert``: an inverted motor needs the
        mirrored range. When this is empty, every target accepted by the
        Cartesian limits is reachable by every motor.
        """
        problems: list[str] = []
        ranges = self._kinematics.motor_ranges(limits)
        for motor, sign in zip(MOTORS, self._signs, strict=True):
            low_k, high_k = ranges[motor]
            needed_min, needed_max = (low_k, high_k) if sign > 0 else (-high_k, -low_k)
            low, high = self.travel[motor]
            if needed_min < low or needed_max > high:
                problems.append(
                    f"motor {motor.value} needs steps [{needed_min:.0f}, {needed_max:.0f}] "
                    f"but its travel is [{low}, {high}]"
                )
        return problems

    def plan_moves(
        self, current: MotorSteps, target: MotorSteps, *, limits: StageLimits
    ) -> list[MotorSteps]:
        """:func:`plan_backlash_moves` with this machine's backlash, travel and inversion.

        The approach direction is defined per motor in logical (kinematic)
        terms, so ``invert`` never changes which moves need compensation. Every
        overshoot waypoint is converted back to Cartesian and must lie inside
        ``limits``. The only tolerance is for a target *on* a face, whose step
        grid point may lie a few nanometres outside (at most
        :meth:`resolution_um`): a waypoint may then be no further outside than
        the target itself.
        """
        resolution = self.resolution_um()
        target_excess = _excess_um(limits, self.to_position(target))
        margin = {axis: min(excess, resolution) for axis, excess in target_excess.items()}

        def admissible(steps: MotorSteps) -> bool:
            return _within(limits, self.to_position(steps), margin)

        return plan_backlash_moves(
            current,
            target,
            self.backlash,
            self.travel,
            signs=self._signs,
            admissible=admissible,
        )
