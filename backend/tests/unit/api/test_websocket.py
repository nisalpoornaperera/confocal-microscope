"""``/ws/scans/{scan_id}``: unknown scans, finished scans, client disconnects."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from tests.unit.api.helpers import services_of, start_scan, wait_for, wait_for_scan

from confocal.models import ScanEvent, ScanEventType, ScanState


def test_unknown_scan_closes_with_4404(client: TestClient) -> None:
    with (
        client.websocket_connect("/ws/scans/does-not-exist") as ws,
        pytest.raises(WebSocketDisconnect) as excinfo,
    ):
        ws.receive_json()
    assert excinfo.value.code == 4404


def test_finished_scan_gets_snapshot_then_close(client: TestClient) -> None:
    scan = start_scan(client)
    wait_for_scan(client, scan["id"])
    with client.websocket_connect(f"/ws/scans/{scan['id']}") as ws:
        snapshot = ScanEvent.model_validate(ws.receive_json())
        assert snapshot.type is ScanEventType.SNAPSHOT
        assert snapshot.scan_id == scan["id"]
        assert snapshot.progress.state is ScanState.COMPLETE
        assert snapshot.progress.completed_points == 20
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
    assert excinfo.value.code == 1000


def test_client_disconnect_mid_scan_unsubscribes(slow_client: TestClient) -> None:
    scan = start_scan(slow_client)
    broker = services_of(slow_client).broker
    try:
        with slow_client.websocket_connect(f"/ws/scans/{scan['id']}") as ws:
            snapshot = ScanEvent.model_validate(ws.receive_json())
            assert snapshot.type is ScanEventType.SNAPSHOT
            assert snapshot.progress.state not in (ScanState.COMPLETE, ScanState.ERROR)
            assert broker.subscriber_count(scan["id"]) == 1
            ws.send_text("ignored client message")
            ScanEvent.model_validate(ws.receive_json())
        wait_for(lambda: broker.subscriber_count(scan["id"]) == 0, timeout_s=5.0)
        # the scan is unaffected by the client leaving
        assert slow_client.get(f"/api/v1/scans/{scan['id']}").json()["state"] in {
            "preparing",
            "calibrating",
            "scanning",
        }
    finally:
        slow_client.post(f"/api/v1/scans/{scan['id']}/cancel")
    assert wait_for_scan(slow_client, scan["id"])["state"] == "cancelled"
