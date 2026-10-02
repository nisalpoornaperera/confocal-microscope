"""In-process publish/subscribe of live scan events (WebSocket fan-out).

The scan task publishes synchronously from the event loop; every subscriber
(one per WebSocket client) owns a bounded queue. A slow or stalled client must
never slow the scan down or exhaust memory, so :meth:`ScanEventBroker.publish`
never blocks and never raises because of a consumer. When a queue is full the
oldest PROGRESS / PROFILE event is discarded first: those are periodic and the
next one supersedes them. STATE, POINT and ERROR events are dropped only when
the queue holds nothing else (then the oldest one goes); the client can always
re-synchronise from the REST API, and the ``dropped`` counter tells it that it
lost events.

All methods must be called from the event loop thread.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from types import TracebackType
from typing import TypeVar

from confocal.models.scan import ScanEvent, ScanEventType

#: Superseded by the next event of the same type, so they are dropped first.
DROPPABLE_EVENT_TYPES: frozenset[ScanEventType] = frozenset(
    {ScanEventType.PROGRESS, ScanEventType.PROFILE}
)

#: Last events / closed markers kept for this many scans (one scan runs at a time).
_RETAINED_SCANS = 64


class Subscription:
    """Bounded stream of the events of one scan; an async iterator and context manager.

    Iteration ends after the scan's terminal event has been delivered
    (:meth:`ScanEventBroker.close_scan`) or immediately after :meth:`close`.
    """

    def __init__(self, broker: ScanEventBroker, scan_id: str, maxsize: int) -> None:
        self._broker = broker
        self._scan_id = scan_id
        self._maxsize = maxsize
        self._items: deque[ScanEvent] = deque()
        self._ready = asyncio.Event()
        self._finished = False  # no more events will arrive; pending ones are delivered
        self._dropped = 0

    @property
    def scan_id(self) -> str:
        return self._scan_id

    @property
    def dropped(self) -> int:
        """Number of events discarded because this subscriber fell behind."""
        return self._dropped

    @property
    def pending(self) -> int:
        return len(self._items)

    @property
    def finished(self) -> bool:
        """True when no further events will be added (pending ones may remain)."""
        return self._finished

    def close(self) -> None:
        """Unsubscribe now; pending events are discarded and iteration stops."""
        self._items.clear()
        self._end()
        self._broker._discard(self)

    def get_nowait(self) -> ScanEvent | None:
        """Next pending event without waiting, or ``None``."""
        return self._items.popleft() if self._items else None

    def __aiter__(self) -> Subscription:
        return self

    async def __anext__(self) -> ScanEvent:
        while not self._items:
            if self._finished:
                raise StopAsyncIteration
            self._ready.clear()
            await self._ready.wait()
        return self._items.popleft()

    async def __aenter__(self) -> Subscription:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ broker side
    def _offer(self, event: ScanEvent) -> None:
        if self._finished:
            return
        if len(self._items) >= self._maxsize and not self._make_room(event):
            return
        self._items.append(event)
        self._ready.set()

    def _make_room(self, event: ScanEvent) -> bool:
        """Free one slot for ``event``; False when ``event`` itself is the one to drop."""
        self._dropped += 1
        for index, queued in enumerate(self._items):
            if queued.type in DROPPABLE_EVENT_TYPES:
                del self._items[index]
                return True
        if event.type in DROPPABLE_EVENT_TYPES:
            return False
        self._items.popleft()
        return True

    def _end(self) -> None:
        self._finished = True
        self._ready.set()


class ScanEventBroker:
    """Fan-out of :class:`ScanEvent` objects to per-scan subscribers."""

    def __init__(self, queue_size: int = 256) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self._queue_size = queue_size
        self._subscribers: dict[str, set[Subscription]] = {}
        self._last: OrderedDict[str, ScanEvent] = OrderedDict()
        self._closed: OrderedDict[str, None] = OrderedDict()

    @property
    def queue_size(self) -> int:
        return self._queue_size

    def publish(self, event: ScanEvent) -> None:
        """Deliver ``event`` to every subscriber of its scan. Never blocks."""
        scan_id = event.scan_id
        self._last[scan_id] = event
        self._last.move_to_end(scan_id)
        _trim(self._last)
        for subscription in tuple(self._subscribers.get(scan_id, ())):
            subscription._offer(event)

    def subscribe(self, scan_id: str) -> Subscription:
        """New subscription to future events of ``scan_id``.

        A subscription to a scan that was already closed ends immediately.
        """
        subscription = Subscription(self, scan_id, self._queue_size)
        if scan_id in self._closed:
            subscription._end()
        else:
            self._subscribers.setdefault(scan_id, set()).add(subscription)
        return subscription

    def last_event(self, scan_id: str) -> ScanEvent | None:
        """Most recent event published for ``scan_id`` (for connection snapshots)."""
        return self._last.get(scan_id)

    def subscriber_count(self, scan_id: str) -> int:
        return len(self._subscribers.get(scan_id, ()))

    def close_scan(self, scan_id: str) -> None:
        """No more events for ``scan_id``: subscriptions end once their queues are drained."""
        for subscription in self._subscribers.pop(scan_id, set()):
            subscription._end()
        self._closed[scan_id] = None
        self._closed.move_to_end(scan_id)
        _trim(self._closed)

    def _discard(self, subscription: Subscription) -> None:
        subscribers = self._subscribers.get(subscription.scan_id)
        if subscribers is None:
            return
        subscribers.discard(subscription)
        if not subscribers:
            del self._subscribers[subscription.scan_id]


_V = TypeVar("_V")


def _trim(mapping: OrderedDict[str, _V]) -> None:
    while len(mapping) > _RETAINED_SCANS:
        mapping.popitem(last=False)
