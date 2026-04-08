"""TaskNotifier 回归与事件载荷 contract tests。"""

from datetime import datetime
from types import SimpleNamespace

import pytest

import app.services.tasks.task_notifier as task_notifier_module
from app.models.task_models import TaskStatus
from app.services.tasks.task_notifier import TaskNotifier


class FakeStage:
    """提供最小可变阶段对象，模拟 monitor 内部状态。"""

    def __init__(self) -> None:
        self.stage_name = "backtest_execution"
        self.start_time = datetime.utcnow()
        self.end_time = datetime.utcnow()
        self.progress = 50.0
        self.status = "completed"
        self.details = {"processed_days": 999, "current_date": "1999-01-01"}


class FakeBacktestProgressData:
    """提供最小回测进度对象，便于验证 reset 行为。"""

    def __init__(self) -> None:
        self.task_id = "t-1"
        self.backtest_id = "bt_x"
        self.start_time = datetime.utcnow()
        self.overall_progress = 100.0
        self.current_stage = "data_storage"
        self.stages = [FakeStage()]
        self.total_trading_days = 500
        self.processed_trading_days = 999
        self.current_date = "1999-01-01"
        self.processing_speed = 1.0
        self.estimated_completion = None
        self.elapsed_time = None
        self.total_signals_generated = 10
        self.total_trades_executed = 20
        self.current_portfolio_value = 123.0
        self.error_message = "old_error"
        self.warnings = ["old_warning"]


class FakeBacktestProgressMonitor:
    """模拟 backtest_progress_monitor。"""

    def __init__(self, initial_progress: FakeBacktestProgressData | None) -> None:
        self.active = {}
        if initial_progress is not None:
            self.active[initial_progress.task_id] = initial_progress
        self.start_called = False
        self.update_stage_calls: list[dict[str, object]] = []

    def get_progress_data(self, task_id: str):
        """返回指定任务的进度对象。"""
        return self.active.get(task_id)

    async def start_backtest_monitoring(
        self, task_id: str, backtest_id: str, total_trading_days: int = 0
    ) -> None:
        """模拟初始化 monitor。"""
        self.start_called = True
        progress_data = FakeBacktestProgressData()
        progress_data.task_id = task_id
        progress_data.backtest_id = backtest_id
        self.active[task_id] = progress_data

    async def update_stage(
        self,
        task_id: str,
        stage_name: str,
        progress: float | None = None,
        status: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """记录阶段同步调用。"""
        self.update_stage_calls.append(
            {
                "task_id": task_id,
                "stage_name": stage_name,
                "progress": progress,
                "status": status,
                "details": details,
            }
        )


class FakeBacktestWebSocketManager:
    """记录回测 websocket 下发的消息。"""

    def __init__(self) -> None:
        self.progress_updates: list[tuple[str, object]] = []
        self.messages: list[tuple[str, dict[str, object]]] = []

    async def send_progress_update(self, task_id: str, progress_data: object) -> None:
        """记录详细进度推送。"""
        self.progress_updates.append((task_id, progress_data))

    async def send_to_task_subscribers(
        self, task_id: str, message: dict[str, object]
    ) -> None:
        """记录基础消息推送。"""
        self.messages.append((task_id, message))


class FakeGeneralWebSocketManager:
    """记录通用 websocket 下发的消息。"""

    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, object]]] = []

    async def send_to_task_subscribers(
        self, task_id: str, message: dict[str, object]
    ) -> None:
        """记录基础消息推送。"""
        self.messages.append((task_id, message))


def build_task(**overrides: object) -> SimpleNamespace:
    """构造最小 task 样本。"""
    now = datetime.utcnow()
    payload = {
        "task_id": "t-1",
        "task_name": "示例任务",
        "task_type": "backtest",
        "status": TaskStatus.RUNNING.value,
        "progress": 10.0,
        "started_at": now,
        "completed_at": now,
        "result": None,
        "error_message": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


@pytest.mark.asyncio
async def test_running_backtest_does_not_backfill_stale_progress_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """running 状态下不得把旧 progress_data 回灌给新运行。"""
    fake_monitor = FakeBacktestProgressMonitor(FakeBacktestProgressData())
    monkeypatch.setattr(task_notifier_module, "backtest_progress_monitor", fake_monitor)

    notifier = TaskNotifier(poll_interval=0.1)
    task = build_task(
        result={
            "progress_data": {
                "processed_days": 45,
                "current_date": "2024-01-15",
            }
        }
    )

    await notifier._sync_backtest_progress_from_task(task)

    progress_data = fake_monitor.get_progress_data("t-1")
    assert progress_data is not None
    assert progress_data.processed_trading_days == 0
    assert progress_data.current_date is None
    assert progress_data.total_signals_generated == 0
    assert progress_data.total_trades_executed == 0


@pytest.mark.asyncio
async def test_completed_backtest_without_monitor_data_emits_completion_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """completed 回测在没有 monitor 数据时应下发结果事件。"""
    fake_monitor = FakeBacktestProgressMonitor(initial_progress=None)
    fake_ws_manager = FakeBacktestWebSocketManager()
    monkeypatch.setattr(task_notifier_module, "backtest_progress_monitor", fake_monitor)
    monkeypatch.setattr(task_notifier_module, "backtest_ws_manager", fake_ws_manager)

    notifier = TaskNotifier(poll_interval=0.1)
    task = build_task(
        status=TaskStatus.COMPLETED.value,
        progress=100.0,
        result={"summary": {"annual_return": 0.12}},
    )

    await notifier._notify_backtest_update(task)

    assert fake_ws_manager.progress_updates == []
    assert len(fake_ws_manager.messages) == 1
    _, message = fake_ws_manager.messages[0]
    assert message["type"] == "backtest_completed"
    assert message["results"] == task.result
    assert message["task_id"] == task.task_id


@pytest.mark.asyncio
async def test_failed_backtest_without_monitor_data_emits_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """failed 回测在没有 monitor 数据时应下发错误事件。"""
    fake_monitor = FakeBacktestProgressMonitor(initial_progress=None)
    fake_ws_manager = FakeBacktestWebSocketManager()
    monkeypatch.setattr(task_notifier_module, "backtest_progress_monitor", fake_monitor)
    monkeypatch.setattr(task_notifier_module, "backtest_ws_manager", fake_ws_manager)

    notifier = TaskNotifier(poll_interval=0.1)
    task = build_task(
        status=TaskStatus.FAILED.value,
        progress=35.0,
        error_message="boom",
    )

    await notifier._notify_backtest_update(task)

    assert fake_ws_manager.progress_updates == []
    assert len(fake_ws_manager.messages) == 1
    _, message = fake_ws_manager.messages[0]
    assert message["type"] == "backtest_error"
    assert message["error_message"] == "boom"
    assert message["task_id"] == task.task_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_type", "expected_extra_key"),
    [
        (TaskStatus.RUNNING.value, "task:progress", "progress"),
        (TaskStatus.COMPLETED.value, "task:completed", "results"),
        (TaskStatus.FAILED.value, "task:failed", "error_message"),
    ],
)
async def test_general_task_payload_contracts(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_type: str,
    expected_extra_key: str,
) -> None:
    """普通任务 websocket 事件类型与关键字段应保持稳定。"""
    fake_manager = FakeGeneralWebSocketManager()
    monkeypatch.setattr(task_notifier_module, "manager", fake_manager)

    notifier = TaskNotifier(poll_interval=0.1)
    task = build_task(
        task_type="prediction",
        status=status,
        progress=50.0,
        result={"predictions": []},
        error_message="boom",
    )

    await notifier._notify_general_task_update(task)

    assert len(fake_manager.messages) == 1
    _, message = fake_manager.messages[0]
    assert message["type"] == expected_type
    assert message["task_id"] == task.task_id
    assert expected_extra_key in message
