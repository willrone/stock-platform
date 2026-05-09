"""Backtest WebSocket endpoint tests.

These tests run against an in-process FastAPI app instead of requiring a live
server on localhost:8000.  They verify the maintained HTTP/WebSocket contracts
for the backtest progress router.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api.v1 import backtest_websocket


@pytest.fixture
def client() -> TestClient:
    """Create an isolated ASGI test client for the backtest WebSocket router."""
    app = FastAPI()
    app.include_router(backtest_websocket.router, prefix="/api/v1")

    def override_db():
        yield object()

    app.dependency_overrides[backtest_websocket.get_db] = override_db
    backtest_websocket.backtest_ws_manager.active_connections.clear()
    backtest_websocket.backtest_ws_manager.task_subscriptions.clear()
    backtest_websocket.backtest_ws_manager.connection_tasks.clear()

    with TestClient(app) as test_client:
        yield test_client

    backtest_websocket.backtest_ws_manager.active_connections.clear()
    backtest_websocket.backtest_ws_manager.task_subscriptions.clear()
    backtest_websocket.backtest_ws_manager.connection_tasks.clear()


def _task(task_type: str = "backtest") -> SimpleNamespace:
    return SimpleNamespace(task_id="test_task_001", task_type=task_type)


def test_http_endpoints(client: TestClient) -> None:
    """HTTP stats and progress endpoints should work without a live server."""
    stats_response = client.get("/api/v1/backtest/ws/stats")
    assert stats_response.status_code == 200
    stats_payload = stats_response.json()
    assert stats_payload["success"] is True
    assert set(stats_payload["data"]).issuperset(
        {"total_connections", "task_subscriptions", "active_backtests", "timestamp"}
    )

    repo = Mock()
    repo.get_task_by_id.return_value = None
    with patch.object(backtest_websocket, "TaskRepository", return_value=repo):
        missing_response = client.get("/api/v1/backtest/progress/test_task_001")

    assert missing_response.status_code == 404
    assert missing_response.json()["detail"] == "任务不存在"

    repo.get_task_by_id.return_value = _task()
    with patch.object(backtest_websocket, "TaskRepository", return_value=repo):
        no_progress_response = client.get("/api/v1/backtest/progress/test_task_001")

    assert no_progress_response.status_code == 200
    assert no_progress_response.json() == {
        "success": True,
        "message": "当前没有进度数据",
        "data": None,
    }


def test_websocket_connection(client: TestClient) -> None:
    """WebSocket should accept a valid backtest task and answer control messages."""
    repo = Mock()
    repo.get_task_by_id.return_value = _task()

    with patch.object(backtest_websocket, "TaskRepository", return_value=repo):
        with client.websocket_connect("/api/v1/backtest/ws/test_task_001") as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connection_established"
            assert connected["task_id"] == "test_task_001"
            assert connected["message"] == "回测进度WebSocket连接建立成功"

            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"

            websocket.send_json({"type": "get_current_progress"})
            no_progress = websocket.receive_json()
            assert no_progress["type"] == "no_progress_data"
            assert no_progress["message"] == "当前没有进度数据"

            websocket.send_json({"type": "unknown"})
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert "未知的消息类型" in error["message"]


def test_websocket_rejects_missing_task(client: TestClient) -> None:
    """Missing tasks should close the WebSocket with the endpoint contract code."""
    repo = Mock()
    repo.get_task_by_id.return_value = None

    with patch.object(backtest_websocket, "TaskRepository", return_value=repo):
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/v1/backtest/ws/missing-task"):
                pass

    assert exc_info.value.code == 4004


def test_websocket_rejects_non_backtest_task(client: TestClient) -> None:
    """Non-backtest tasks should be rejected before connection registration."""
    repo = Mock()
    repo.get_task_by_id.return_value = _task(task_type="prediction")

    with patch.object(backtest_websocket, "TaskRepository", return_value=repo):
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/v1/backtest/ws/prediction-task"):
                pass

    assert exc_info.value.code == 4005
