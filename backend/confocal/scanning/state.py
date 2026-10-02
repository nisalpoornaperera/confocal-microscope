"""Scan state machine (docs/architecture.md §3).

```
IDLE ─► PREPARING ─► [CALIBRATING] ─► [HOMING] ─► SCANNING ◄─► PAUSED
                                                     │
                                                     ▼
                     PROCESSING ─► [SURFACE_RECONSTRUCTION] ─► [ML_PROCESSING] ─► COMPLETE

any non-terminal state ─► CANCELLED | ERROR
```

The machine only validates and records transitions; persisting them and
publishing STATE events is the job of :class:`confocal.scanning.manager.ScanManager`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

from confocal.errors import ScanStateError
from confocal.models.common import utc_now
from confocal.models.scan import ACTIVE_SCAN_STATES, TERMINAL_SCAN_STATES, ScanState

_S = ScanState

#: Forward edges of the diagram (CANCELLED / ERROR are added for every non-terminal state).
_FORWARD: dict[ScanState, frozenset[ScanState]] = {
    _S.IDLE: frozenset({_S.PREPARING}),
    _S.PREPARING: frozenset({_S.CALIBRATING, _S.HOMING, _S.SCANNING}),
    _S.CALIBRATING: frozenset({_S.HOMING, _S.SCANNING}),
    _S.HOMING: frozenset({_S.SCANNING}),
    _S.SCANNING: frozenset({_S.PAUSED, _S.PROCESSING}),
    _S.PAUSED: frozenset({_S.SCANNING}),
    _S.PROCESSING: frozenset({_S.SURFACE_RECONSTRUCTION, _S.ML_PROCESSING, _S.COMPLETE}),
    _S.SURFACE_RECONSTRUCTION: frozenset({_S.ML_PROCESSING, _S.COMPLETE}),
    _S.ML_PROCESSING: frozenset({_S.COMPLETE}),
}


def _build_transitions() -> Mapping[ScanState, frozenset[ScanState]]:
    table: dict[ScanState, frozenset[ScanState]] = {}
    for state in ScanState:
        if state in TERMINAL_SCAN_STATES:
            table[state] = frozenset()
        else:
            table[state] = _FORWARD.get(state, frozenset()) | {_S.CANCELLED, _S.ERROR}
    return MappingProxyType(table)


#: Every legal ``from -> {to}`` edge. Terminal states have no outgoing edges.
ALLOWED_TRANSITIONS: Mapping[ScanState, frozenset[ScanState]] = _build_transitions()


@dataclass(frozen=True, slots=True)
class StateTransition:
    """One entry of the state history. The first entry has ``previous=None``."""

    previous: ScanState | None
    state: ScanState
    timestamp: datetime
    reason: str | None = None


class ScanStateMachine:
    """Validated scan state with a timestamped history."""

    def __init__(
        self,
        initial: ScanState = ScanState.IDLE,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._clock = clock
        self._history: list[StateTransition] = [StateTransition(None, initial, clock())]

    @property
    def state(self) -> ScanState:
        return self._history[-1].state

    @property
    def entered_at(self) -> datetime:
        """When the current state was entered."""
        return self._history[-1].timestamp

    @property
    def history(self) -> tuple[StateTransition, ...]:
        return tuple(self._history)

    @property
    def is_active(self) -> bool:
        """True while the scan owns the instrument (manual control is refused)."""
        return self.state in ACTIVE_SCAN_STATES

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_SCAN_STATES

    def can_transition(self, to: ScanState) -> bool:
        return to in ALLOWED_TRANSITIONS[self.state]

    def transition(self, to: ScanState, reason: str | None = None) -> StateTransition:
        """Move to ``to``; raises :class:`ScanStateError` for an edge not in the diagram."""
        current = self.state
        if not self.can_transition(to):
            raise ScanStateError(f"invalid scan state transition {current.value} -> {to.value}")
        record = StateTransition(current, to, self._clock(), reason)
        self._history.append(record)
        return record
