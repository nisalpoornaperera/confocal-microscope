"""Scan state machine: every edge of the §3 diagram and nothing else."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from confocal.errors import ScanStateError
from confocal.models import ACTIVE_SCAN_STATES, TERMINAL_SCAN_STATES, ScanState
from confocal.scanning.state import ALLOWED_TRANSITIONS, ScanStateMachine

S = ScanState

EXPECTED_FORWARD: dict[ScanState, set[ScanState]] = {
    S.IDLE: {S.PREPARING},
    S.PREPARING: {S.CALIBRATING, S.HOMING, S.SCANNING},
    S.CALIBRATING: {S.HOMING, S.SCANNING},
    S.HOMING: {S.SCANNING},
    S.SCANNING: {S.PAUSED, S.PROCESSING},
    S.PAUSED: {S.SCANNING},
    S.PROCESSING: {S.SURFACE_RECONSTRUCTION, S.ML_PROCESSING, S.COMPLETE},
    S.SURFACE_RECONSTRUCTION: {S.ML_PROCESSING, S.COMPLETE},
    S.ML_PROCESSING: {S.COMPLETE},
}

ALL_EDGES = [(a, b) for a, targets in ALLOWED_TRANSITIONS.items() for b in sorted(targets)]

FORBIDDEN = [
    (S.IDLE, S.SCANNING),
    (S.IDLE, S.COMPLETE),
    (S.PREPARING, S.PAUSED),
    (S.PREPARING, S.PROCESSING),
    (S.CALIBRATING, S.PREPARING),
    (S.HOMING, S.CALIBRATING),
    (S.SCANNING, S.COMPLETE),
    (S.SCANNING, S.SURFACE_RECONSTRUCTION),
    (S.PAUSED, S.PROCESSING),
    (S.PROCESSING, S.SCANNING),
    (S.SURFACE_RECONSTRUCTION, S.PROCESSING),
    (S.ML_PROCESSING, S.SURFACE_RECONSTRUCTION),
    (S.SCANNING, S.IDLE),
    (S.SCANNING, S.SCANNING),
]


def test_table_matches_the_diagram() -> None:
    for state in ScanState:
        targets = set(ALLOWED_TRANSITIONS[state])
        if state in TERMINAL_SCAN_STATES:
            assert targets == set()
        else:
            assert targets == EXPECTED_FORWARD[state] | {S.CANCELLED, S.ERROR}


@pytest.mark.parametrize(("source", "target"), ALL_EDGES)
def test_every_allowed_transition_succeeds(source: ScanState, target: ScanState) -> None:
    machine = ScanStateMachine(source)
    assert machine.can_transition(target)
    record = machine.transition(target, reason="test")
    assert machine.state is target
    assert record.previous is source
    assert record.reason == "test"


@pytest.mark.parametrize(("source", "target"), FORBIDDEN)
def test_forbidden_transitions_raise(source: ScanState, target: ScanState) -> None:
    machine = ScanStateMachine(source)
    assert not machine.can_transition(target)
    with pytest.raises(ScanStateError, match=f"{source.value} -> {target.value}"):
        machine.transition(target)
    assert machine.state is source


@pytest.mark.parametrize("terminal", sorted(TERMINAL_SCAN_STATES))
def test_terminal_states_never_leave(terminal: ScanState) -> None:
    machine = ScanStateMachine(terminal)
    assert machine.is_terminal
    assert not machine.is_active
    for target in ScanState:
        with pytest.raises(ScanStateError):
            machine.transition(target)


@pytest.mark.parametrize("state", sorted(set(ScanState) - TERMINAL_SCAN_STATES))
def test_every_non_terminal_state_can_be_cancelled_or_fail(state: ScanState) -> None:
    for target in (S.CANCELLED, S.ERROR):
        machine = ScanStateMachine(state)
        machine.transition(target)
        assert machine.is_terminal


def test_is_active_follows_the_shared_definition() -> None:
    for state in ScanState:
        assert ScanStateMachine(state).is_active is (state in ACTIVE_SCAN_STATES)
    assert not ScanStateMachine().is_active  # IDLE: created, not started


def test_history_records_every_transition_with_timestamps() -> None:
    start = datetime(2026, 10, 1, tzinfo=UTC)
    ticks = iter(start + timedelta(seconds=i) for i in range(10))
    machine = ScanStateMachine(clock=lambda: next(ticks))
    for state in (S.PREPARING, S.SCANNING, S.PAUSED, S.SCANNING, S.PROCESSING, S.COMPLETE):
        machine.transition(state)
    history = machine.history
    assert [h.state for h in history] == [
        S.IDLE,
        S.PREPARING,
        S.SCANNING,
        S.PAUSED,
        S.SCANNING,
        S.PROCESSING,
        S.COMPLETE,
    ]
    assert history[0].previous is None
    assert [h.timestamp for h in history] == [start + timedelta(seconds=i) for i in range(7)]
    assert machine.entered_at == start + timedelta(seconds=6)
