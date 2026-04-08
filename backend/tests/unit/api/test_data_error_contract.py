"""
数据路由错误处理 contract tests
"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

api_package = ModuleType("app.api")
api_package.__path__ = [str(BACKEND_ROOT / "app" / "api")]
v1_package = ModuleType("app.api.v1")
v1_package.__path__ = [str(BACKEND_ROOT / "app" / "api" / "v1")]
sys.modules.setdefault("app.api", api_package)
sys.modules.setdefault("app.api.v1", v1_package)

fake_dependencies_module = ModuleType("app.api.v1.dependencies")
fake_dependencies_module.get_current_user = MagicMock()

fake_container_module = ModuleType("app.core.container")
fake_container_module.get_data_service = MagicMock()
fake_container_module.get_sftp_sync_service = MagicMock()

fake_services_data_module = ModuleType("app.services.data")
fake_services_data_module.SimpleDataService = object

fake_parquet_manager_module = ModuleType("app.services.data.parquet_manager")
fake_parquet_manager_module.ParquetManager = object

fake_sftp_sync_module = ModuleType("app.services.data.sftp_sync_service")
fake_sftp_sync_module.SFTPSyncService = object

fake_events_module = ModuleType("app.services.events.data_sync_events")
fake_events_module.DataSyncEventType = object
fake_events_module.get_data_sync_event_manager = MagicMock()

with patch.dict(
    "sys.modules",
    {
        "app.api.v1.dependencies": fake_dependencies_module,
        "app.core.container": fake_container_module,
        "app.services.data": fake_services_data_module,
        "app.services.data.parquet_manager": fake_parquet_manager_module,
        "app.services.data.sftp_sync_service": fake_sftp_sync_module,
        "app.services.events.data_sync_events": fake_events_module,
        "paramiko": MagicMock(),
    },
):
    from app.api.v1.data import _mark_task_failed_after_submit_error
    from app.core.error_handler import ErrorContext


def test_mark_task_failed_after_submit_error_logs_compensation_failure():
    """数据任务回写 FAILED 再失败时不能静默吞错。"""

    repository = MagicMock()
    repository.update_task_status.side_effect = RuntimeError("db write failed")
    submit_error = RuntimeError("pool down")
    context = ErrorContext(
        task_id="qlib-task-1",
        additional_data={"route": "trigger_qlib_precompute"},
    )

    with patch("app.api.v1.data.log_best_effort_failure") as mock_log_best_effort:
        _mark_task_failed_after_submit_error(
            repository,
            task_id="qlib-task-1",
            submit_error=submit_error,
            context=context,
        )

    repository.update_task_status.assert_called_once()
    mock_log_best_effort.assert_called_once()
    assert mock_log_best_effort.call_args.kwargs["context"]["task_id"] == "qlib-task-1"
    assert mock_log_best_effort.call_args.kwargs["context"]["submit_error"] == "pool down"
