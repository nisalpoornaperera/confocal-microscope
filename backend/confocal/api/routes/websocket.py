"""``/ws/scans/{scan_id}``: live scan events.

Protocol (server -> client JSON, ``ScanEvent.model_dump(mode="json")``):

1. On connect: one ``snapshot`` event with the current progress.
2. Then every broker event of the scan (``state``, ``progress``, ``point``,
   ``profile``, ``error``) in publication order. A slow client loses
   ``progress`` / ``profile`` events first (see ``ScanEventBroker``); it can
   always re-synchronise from the REST API.
3. After the terminal ``state`` event (complete / cancelled / error) the server
   closes with code 1000. A finished scan gets its snapshot and is closed at once.

An unknown scan is accepted and immediately closed with code 4404. Messages
sent by the client are ignored; a client disconnect ends the stream cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from confocal.api.deps import get_services
from confocal.errors import ScanNotFoundError
from confocal.models.scan import TERMINAL_SCAN_STATES, ScanEvent, ScanEventType
from confocal.scanning import ScanEventBroker, ScanManager, Subscription

log = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

#: Close code for an unknown scan (4000-4999 are application-defined).
CLOSE_UNKNOWN_SCAN = 4404
CLOSE_NORMAL = 1000


async def build_snapshot(manager: ScanManager, broker: ScanEventBroker, scan_id: str) -> ScanEvent:
    """SNAPSHOT event: live progress of an active scan, else built from the stored summary.

    Raises:
        ScanNotFoundError: unknown scan.
    """
    progress = await manager.get_progress(scan_id)
    last = broker.last_event(scan_id)
    return ScanEvent(
        type=ScanEventType.SNAPSHOT,
        scan_id=scan_id,
        progress=progress,
        point=last.point if last is not None else None,
        message=progress.message if progress.message is not None else _message(last),
    )


def _message(event: ScanEvent | None) -> str | None:
    return None if event is None else event.message


def is_terminal_event(event: ScanEvent) -> bool:
    return event.type in (ScanEventType.STATE, ScanEventType.SNAPSHOT) and (
        event.progress.state in TERMINAL_SCAN_STATES
    )


@router.websocket("/ws/scans/{scan_id}")
async def scan_events(websocket: WebSocket, scan_id: str) -> None:
    services = get_services(websocket)
    manager = services.scan_manager
    await websocket.accept()
    try:
        # No await yields between subscribing and building the snapshot of an
        # active scan, so no event can fall between the two.
        subscription = await manager.subscribe(scan_id)
        snapshot = await build_snapshot(manager, services.broker, scan_id)
    except ScanNotFoundError:
        await websocket.close(code=CLOSE_UNKNOWN_SCAN, reason=f"unknown scan {scan_id}")
        return
    try:
        await _stream(websocket, subscription, snapshot)
    except WebSocketDisconnect:
        log.debug("websocket client of scan %s disconnected", scan_id)
    finally:
        subscription.close()


async def _stream(websocket: WebSocket, subscription: Subscription, snapshot: ScanEvent) -> None:
    await websocket.send_json(snapshot.model_dump(mode="json"))
    if is_terminal_event(snapshot):
        await _close(websocket)
        return
    sender = asyncio.create_task(_forward(websocket, subscription))
    receiver = asyncio.create_task(_wait_for_disconnect(websocket))
    try:
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (sender, receiver):
            if not task.done():
                task.cancel()
        for task in (sender, receiver):
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                await task
    if sender in done:
        sender.result()  # re-raise a send failure
        await _close(websocket)


async def _forward(websocket: WebSocket, subscription: Subscription) -> None:
    async for event in subscription:
        await websocket.send_json(event.model_dump(mode="json"))
        if is_terminal_event(event):
            return


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    """Drain (and ignore) client messages until the client goes away."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def _close(websocket: WebSocket) -> None:
    if websocket.client_state is WebSocketState.CONNECTED:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=CLOSE_NORMAL)
