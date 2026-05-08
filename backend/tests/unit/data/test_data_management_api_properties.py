"""当前 data 路由契约测试。"""

import sys
from datetime import datetime
from types import ModuleType, SimpleNamespace

fake_paramiko = ModuleType("paramiko")
fake_paramiko.SSHClient = type("SSHClient", (), {})
fake_paramiko.SFTPClient = type("SFTPClient", (), {})
fake_paramiko.SSHException = Exception
fake_paramiko.socket = SimpleNamespace(error=OSError)
sys.modules.setdefault("paramiko", fake_paramiko)
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.data import (
    get_data_service,
    get_sftp_sync_service,
)
from app.api.v1.data import router as data_router
from app.services.data.sftp_sync_service import SyncResult


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(data_router, prefix="/api/v1")
    return TestClient(app)


def test_status_route_uses_current_data_service_contract() -> None:
    client = _build_client()
    status = SimpleNamespace(
        service_url="http://example.test",
        is_available=True,
        last_check=datetime(2026, 1, 1, 12, 0, 0),
        response_time_ms=12.5,
        error_message=None,
    )
    mock_service = MagicMock()
    mock_service.check_remote_service_status = AsyncMock(return_value=status)
    client.app.dependency_overrides[get_data_service] = lambda: mock_service

    try:
        response = client.get("/api/v1/data/status")
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["service_url"] == "http://example.test"
    assert payload["data"]["is_connected"] is True
    mock_service.check_remote_service_status.assert_awaited_once()


def test_remote_stocks_route_maps_ts_codes_from_service() -> None:
    client = _build_client()
    stocks = [
        {"ts_code": "000001.SZ", "name": "平安银行"},
        {"ts_code": "000002.SZ", "name": "万科A"},
    ]
    mock_service = MagicMock()
    mock_service.get_remote_stock_list = AsyncMock(return_value=stocks)
    client.app.dependency_overrides[get_data_service] = lambda: mock_service

    try:
        response = client.get("/api/v1/data/remote/stocks")
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["total_stocks"] == 2
    assert payload["data"]["stock_codes"] == ["000001.SZ", "000002.SZ"]
    mock_service.get_remote_stock_list.assert_awaited_once()


def test_local_stocks_route_reads_current_parquet_layout(tmp_path, monkeypatch) -> None:
    client = _build_client()
    monkeypatch.setattr("app.api.v1.data.settings.DATA_ROOT_PATH", str(tmp_path))

    stock_data_dir = tmp_path / "parquet" / "stock_data"
    stock_data_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [10.0, 10.5, 20.0],
        }
    )
    df.to_parquet(stock_data_dir / "sample.parquet", index=False)

    response = client.get("/api/v1/data/local/stocks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["stock_codes"] == ["000001.SZ", "000002.SZ"]
    assert payload["data"]["total_stocks"] == 2
    assert payload["data"]["stocks"][0]["record_count"] == 2


def test_local_stocks_simple_route_uses_current_filename_contract(
    tmp_path, monkeypatch
) -> None:
    client = _build_client()
    monkeypatch.setattr("app.api.v1.data.settings.DATA_ROOT_PATH", str(tmp_path))

    stock_data_dir = tmp_path / "parquet" / "stock_data"
    stock_data_dir.mkdir(parents=True)
    (stock_data_dir / "000001_SZ.parquet").touch()
    (stock_data_dir / "600000_SH.parquet").touch()
    (stock_data_dir / "430001_BJ.parquet").touch()

    response = client.get("/api/v1/data/local/stocks/simple")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["stock_codes"] == ["000001.SZ", "430001.BJ", "600000.SH"]
    assert payload["data"]["total_stocks"] == 3


def test_sync_remote_route_uses_current_sftp_service_contract() -> None:
    client = _build_client()
    sync_result = SyncResult(
        success=True,
        total_files=2,
        synced_files=2,
        failed_files=[],
        total_size=4096,
        message="同步完成: 2/2 成功",
    )
    mock_sftp = MagicMock()
    mock_sftp.enabled = True
    mock_sftp.sync_selected_stocks = MagicMock(return_value=sync_result)
    client.app.dependency_overrides[get_sftp_sync_service] = lambda: mock_sftp

    try:
        response = client.post(
            "/api/v1/data/sync/remote",
            json={"stock_codes": ["000001.SZ", "000002.SZ"]},
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["total_files"] == 2
    assert payload["data"]["synced_files"] == 2
    mock_sftp.sync_selected_stocks.assert_called_once_with(["000001.SZ", "000002.SZ"])


def test_sync_remote_route_reports_disabled_sftp() -> None:
    client = _build_client()
    mock_sftp = MagicMock()
    mock_sftp.enabled = False
    client.app.dependency_overrides[get_sftp_sync_service] = lambda: mock_sftp

    try:
        response = client.post("/api/v1/data/sync/remote", json={})
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert "SFTP同步未启用" in payload["message"]


def test_event_history_route_uses_current_event_manager_contract(monkeypatch) -> None:
    client = _build_client()
    event = SimpleNamespace(
        to_dict=lambda: {
            "event_type": "sync_completed",
            "stock_code": "000001.SZ",
            "sync_type": "sftp_sync",
        }
    )
    mock_event_manager = MagicMock()
    mock_event_manager.get_event_history.return_value = [event]
    monkeypatch.setattr(
        "app.api.v1.data.get_data_sync_event_manager",
        lambda: mock_event_manager,
    )

    response = client.get(
        "/api/v1/data/events/history",
        params={"stock_code": "000001.SZ", "event_type": "sync_completed", "limit": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["total_events"] == 1
    assert payload["data"]["events"][0]["stock_code"] == "000001.SZ"


def test_event_history_route_rejects_invalid_event_type() -> None:
    client = _build_client()

    response = client.get(
        "/api/v1/data/events/history", params={"event_type": "bad_type"}
    )

    assert response.status_code == 400
    assert "无效的事件类型" in response.json()["detail"]


def test_event_stats_and_clear_history_use_current_event_manager(monkeypatch) -> None:
    client = _build_client()
    mock_event_manager = MagicMock()
    mock_event_manager.get_stats.return_value = {"total_events": 3, "history_size": 3}
    monkeypatch.setattr(
        "app.api.v1.data.get_data_sync_event_manager",
        lambda: mock_event_manager,
    )

    stats_response = client.get("/api/v1/data/events/stats")
    clear_response = client.delete("/api/v1/data/events/history")

    assert stats_response.status_code == 200
    assert stats_response.json()["data"]["total_events"] == 3
    assert clear_response.status_code == 200
    mock_event_manager.get_stats.assert_called_once()
    mock_event_manager.clear_history.assert_called_once()
