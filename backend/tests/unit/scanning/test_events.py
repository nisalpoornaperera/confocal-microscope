"""Event broker: non-blocking fan-out, drop policy, subscription lifecycle."""

from __future__ import annotations

import asyncio
import time

import pytest

from confocal.models import ScanEvent, ScanEventType, ScanProgress, ScanState
from confocal.scanning.events import ScanEventBroker, Subscription

T = ScanEventType


def _event(event_type: ScanEventType, n: int = 0, scan_id: str = "scan") -> ScanEvent:
    progress = ScanProgress(
        scan_id=scan_id,
        state=ScanState.SCANNING,
        progress=0.0,
        completed_points=n,
        total_points=1000,
    )
    return ScanEvent(type=event_type, scan_id=scan_id, progress=progress, message=str(n))


def _drain(subscription: Subscription) -> list[ScanEvent]:
    events: list[ScanEvent] = []
    while (event := subscription.get_nowait()) is not None:
        events.append(event)
    return events


async def _collect(subscription: Subscription, timeout_s: float = 1.0) -> list[ScanEvent]:
    async def run() -> list[ScanEvent]:
        return [event async for event in subscription]

    return await asyncio.wait_for(run(), timeout_s)


def test_publish_never_blocks_with_a_stalled_subscriber() -> None:
    broker = ScanEventBroker(queue_size=8)
    stalled = broker.subscribe("scan")
    started = time.perf_counter()
    for i in range(10_000):
        broker.publish(_event(T.PROGRESS, i))
    assert time.perf_counter() - started < 2.0
    assert stalled.pending == 8  # bounded
    assert stalled.dropped == 10_000 - 8
    assert [int(e.message or 0) for e in _drain(stalled)] == list(range(9992, 10_000))


def test_full_queue_drops_the_oldest_periodic_event_first() -> None:
    broker = ScanEventBroker(queue_size=4)
    sub = broker.subscribe("scan")
    for event in (
        _event(T.STATE, 0),
        _event(T.PROGRESS, 1),
        _event(T.POINT, 2),
        _event(T.PROFILE, 3),
    ):
        broker.publish(event)
    broker.publish(_event(T.POINT, 4))  # evicts PROGRESS 1
    broker.publish(_event(T.STATE, 5))  # evicts PROFILE 3
    assert [(e.type, e.message) for e in _drain(sub)] == [
        (T.STATE, "0"),
        (T.POINT, "2"),
        (T.POINT, "4"),
        (T.STATE, "5"),
    ]
    assert sub.dropped == 2


def test_queue_of_critical_events_drops_new_periodic_events() -> None:
    broker = ScanEventBroker(queue_size=2)
    sub = broker.subscribe("scan")
    broker.publish(_event(T.STATE, 0))
    broker.publish(_event(T.POINT, 1))
    broker.publish(_event(T.PROGRESS, 2))  # nothing droppable queued: the new one goes
    assert [e.message for e in _drain(sub)] == ["0", "1"]
    broker.publish(_event(T.STATE, 3))
    broker.publish(_event(T.POINT, 4))
    broker.publish(_event(T.ERROR, 5))  # only critical events: the oldest one goes
    assert [e.message for e in _drain(sub)] == ["4", "5"]
    assert sub.dropped == 2


async def test_every_subscriber_receives_every_event() -> None:
    broker = ScanEventBroker()
    first = broker.subscribe("scan")
    second = broker.subscribe("scan")
    other = broker.subscribe("other")
    assert broker.subscriber_count("scan") == 2
    for i in range(3):
        broker.publish(_event(T.POINT, i))
    broker.close_scan("scan")
    for sub in (first, second):
        assert [e.message for e in await _collect(sub)] == ["0", "1", "2"]
    assert other.pending == 0
    assert not other.finished


async def test_consumer_wakes_up_on_publish() -> None:
    broker = ScanEventBroker()
    sub = broker.subscribe("scan")
    consumer = asyncio.create_task(_collect(sub))
    await asyncio.sleep(0.01)
    broker.publish(_event(T.STATE, 1))
    await asyncio.sleep(0.01)
    broker.publish(_event(T.STATE, 2))
    broker.close_scan("scan")
    assert [e.message for e in await consumer] == ["1", "2"]


async def test_close_ends_iteration_and_unsubscribes() -> None:
    broker = ScanEventBroker()
    sub = broker.subscribe("scan")
    broker.publish(_event(T.POINT, 1))
    sub.close()
    assert sub.finished
    assert await _collect(sub) == []
    assert broker.subscriber_count("scan") == 0
    broker.publish(_event(T.POINT, 2))  # no error after unsubscribing
    assert sub.pending == 0


async def test_async_context_manager_closes_the_subscription() -> None:
    broker = ScanEventBroker()
    async with broker.subscribe("scan") as sub:
        assert broker.subscriber_count("scan") == 1
    assert broker.subscriber_count("scan") == 0
    assert sub.finished


async def test_subscribing_after_close_scan_ends_immediately() -> None:
    broker = ScanEventBroker()
    broker.publish(_event(T.STATE, 1))
    broker.close_scan("scan")
    late = broker.subscribe("scan")
    assert late.finished
    assert await _collect(late) == []


def test_last_event_is_kept_per_scan() -> None:
    broker = ScanEventBroker()
    assert broker.last_event("scan") is None
    broker.publish(_event(T.STATE, 1))
    broker.publish(_event(T.PROGRESS, 2))
    broker.publish(_event(T.POINT, 3, scan_id="other"))
    last = broker.last_event("scan")
    assert last is not None
    assert last.message == "2"
    broker.close_scan("scan")
    assert broker.last_event("scan") is last


def test_queue_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="queue_size"):
        ScanEventBroker(queue_size=0)
