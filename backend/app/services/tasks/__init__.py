"""
任务管理模块。

该包对外保留历史聚合导出，但通过懒加载避免导入某个轻量子模块时
连带加载 WebSocket、资源监控等可选/重依赖模块。
"""

from importlib import import_module
from typing import Any

_TASK_EXPORTS = {
    # 任务管理器
    "TaskManager": "task_manager",
    "TaskCreateRequest": "task_manager",
    "TaskUpdateRequest": "task_manager",
    "TaskQuery": "task_manager",
    "TaskSummary": "task_manager",
    # 任务队列
    "TaskQueueManager": "task_queue",
    "TaskScheduler": "task_queue",
    "TaskExecutor": "task_queue",
    "TaskPriority": "task_queue",
    "QueuedTask": "task_queue",
    "TaskExecutionContext": "task_queue",
    # 任务执行引擎
    "TaskExecutionEngine": "task_execution_engine",
    "PredictionTaskExecutor": "task_execution_engine",
    "BacktestTaskExecutor": "task_execution_engine",
    "TrainingTaskExecutor": "task_execution_engine",
    "QlibPrecomputeTaskExecutor": "task_execution_engine",
    "ProgressTracker": "task_execution_engine",
    "TaskProgress": "task_execution_engine",
    # 任务通知服务
    "TaskNotificationService": "task_notification_service",
    "TaskStatusNotification": "task_notification_service",
    "TaskProgressNotification": "task_notification_service",
}


def __getattr__(name: str) -> Any:
    module_name = _TASK_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    exported = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = exported
    return exported


__all__ = list(_TASK_EXPORTS)
