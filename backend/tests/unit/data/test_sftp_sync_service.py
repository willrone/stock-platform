"""
SFTP sync service characterization tests.

These tests lock the current sync orchestration behavior before we
refactor `sync_all_stocks` to reduce complexity.
"""

import asyncio
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

fake_paramiko = ModuleType("paramiko")
fake_paramiko.SSHClient = type("SSHClient", (), {})
fake_paramiko.SFTPClient = type("SFTPClient", (), {})
fake_paramiko.SSHException = Exception
fake_paramiko.socket = SimpleNamespace(error=OSError)

sys.modules.setdefault("paramiko", fake_paramiko)

from app.services.data import sftp_sync_service
from app.services.data.sftp_sync_service import SFTPSyncService


class _DummyEventManager:
    async def emit_sync_started(self, **kwargs):
        return None

    async def emit_sync_completed(self, **kwargs):
        return None

    async def emit_sync_failed(self, **kwargs):
        return None


def _build_service(tmp_path: Path) -> SFTPSyncService:
    service = SFTPSyncService(
        host="127.0.0.1",
        username="tester",
        password="secret",
        remote_list_path="/remote/list.txt",
        remote_data_dir="/remote/data",
        local_data_dir=str(tmp_path),
        port=22,
    )
    service.enabled = True
    service.event_manager = _DummyEventManager()
    service._disconnect_sftp = MagicMock()
    return service


def test_data_sync_events_module_works_without_loguru(monkeypatch):
    real_import = builtins.__import__
    temp_module_name = "temp_data_sync_events_without_loguru"

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "loguru":
            raise ModuleNotFoundError("No module named 'loguru'", name="loguru")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module_path = BACKEND_ROOT / "app/services/events/data_sync_events.py"
    spec = importlib.util.spec_from_file_location(temp_module_name, module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[temp_module_name] = module

    try:
        spec.loader.exec_module(module)
        event_manager = module.DataSyncEventManager()
        assert event_manager.__class__.__name__ == "DataSyncEventManager"
    finally:
        sys.modules.pop(temp_module_name, None)


def test_get_data_sync_event_manager_reraises_internal_import_error(
    monkeypatch,
):
    error = ImportError(
        "cannot import name 'BrokenThing' from 'app.services.events.helpers'",
        name="app.services.events.helpers",
    )
    monkeypatch.setattr(
        sftp_sync_service,
        "import_module",
        MagicMock(side_effect=error),
    )

    with pytest.raises(ImportError, match="BrokenThing"):
        sftp_sync_service._get_data_sync_event_manager()


def test_emit_sync_event_runs_immediately_without_running_loop(tmp_path):
    service = _build_service(tmp_path)
    observed = []

    async def record_event():
        observed.append("completed")

    service._emit_sync_event(record_event(), "emit failed")

    assert observed == ["completed"]


def test_emit_sync_event_schedules_on_existing_running_loop(tmp_path):
    service = _build_service(tmp_path)
    observed_loops = []

    async def record_event():
        observed_loops.append(asyncio.get_running_loop())

    async def run_scenario():
        running_loop = asyncio.get_running_loop()
        service._emit_sync_event(record_event(), "emit failed")
        await asyncio.sleep(0)
        assert observed_loops == [running_loop]

    asyncio.run(run_scenario())


def test_sync_all_stocks_skips_missing_remote_files(tmp_path):
    service = _build_service(tmp_path)
    service._connect_sftp = MagicMock(return_value=("ssh", MagicMock()))
    service._build_remote_files_cache = MagicMock(
        return_value={"000001.SZ": "/remote/000001.SZ.parquet"}
    )
    service.sync_stock_file = MagicMock(return_value=(True, 128, ""))

    result = service.sync_all_stocks(["000001.SZ", "000002.SZ"])

    assert result.success is True
    assert result.total_files == 1
    assert result.synced_files == 1
    assert result.failed_files == []
    assert "1 个文件在远端不存在（已跳过）" in result.message
    service.sync_stock_file.assert_called_once()


def test_sync_all_stocks_fails_when_no_remote_files_available(tmp_path):
    service = _build_service(tmp_path)
    service._connect_sftp = MagicMock(return_value=("ssh", MagicMock()))
    service._build_remote_files_cache = MagicMock(return_value={})

    result = service.sync_all_stocks(["000001.SZ", "000002.SZ"])

    assert result.success is False
    assert result.total_files == 2
    assert result.synced_files == 0
    assert result.failed_files == ["000001.SZ", "000002.SZ"]
    assert "远端服务器上没有可用的股票文件" in result.message


def test_sync_all_stocks_reconnects_and_retries_connection_failures(tmp_path):
    service = _build_service(tmp_path)
    first_sftp = MagicMock()
    second_sftp = MagicMock()
    service._connect_sftp = MagicMock(
        side_effect=[("ssh-1", first_sftp), ("ssh-2", second_sftp)]
    )
    service._build_remote_files_cache = MagicMock(
        return_value={"000001.SZ": "/remote/000001.SZ.parquet"}
    )
    service.sync_stock_file = MagicMock(
        side_effect=[
            (False, 0, "连接错误: timeout"),
            (True, 256, ""),
        ]
    )

    result = service.sync_all_stocks(["000001.SZ"])

    assert result.success is True
    assert result.synced_files == 1
    assert service._connect_sftp.call_count == 2
    assert service._disconnect_sftp.call_count >= 1
    assert service.sync_stock_file.call_count == 2
