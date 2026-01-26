"""
任务管理模块

该模块包含所有与异步任务调度、执行和通知相关的服务，包括：
- 任务管理和生命周期控制
- 任务队列和调度机制
- 任务执行引擎和执行器
- 任务通知和进度跟踪

主要组件：
- TaskManager: 任务管理器
- TaskQueueManager: 任务队列管理器
- TaskExecutionEngine: 任务执行引擎
- TaskNotificationService: 任务通知服务
"""

# 任务执行引擎
from .task_execution_engine import (
    BacktestTaskExecutor,
    PredictionTaskExecutor,
    ProgressTracker,
    QlibPrecomputeTaskExecutor,
    TaskExecutionEngine,
    TaskProgress,
    TrainingTaskExecutor,
)

# 任务管理器
from .task_manager import (
    TaskCreateRequest,
    TaskManager,
    TaskQuery,
    TaskSummary,
    TaskUpdateRequest,
)

# 任务通知服务
from .task_notification_service import (
    TaskNotificationService,
    TaskProgressNotification,
    TaskStatusNotification,
)

# 任务队列
from .task_queue import (
    QueuedTask,
    TaskExecutionContext,
    TaskExecutor,
    TaskPriority,
    TaskQueueManager,
    TaskScheduler,
)

__all__ = [
    # 任务管理器
    "TaskManager",
    "TaskCreateRequest",
    "TaskUpdateRequest",
    "TaskQuery",
    "TaskSummary",
    # 任务队列
    "TaskQueueManager",
    "TaskScheduler",
    "TaskExecutor",
    "TaskPriority",
    "QueuedTask",
    "TaskExecutionContext",
    # 任务执行引擎
    "TaskExecutionEngine",
    "PredictionTaskExecutor",
    "BacktestTaskExecutor",
    "TrainingTaskExecutor",
    "QlibPrecomputeTaskExecutor",
    "ProgressTracker",
    "TaskProgress",
    # 任务通知服务
    "TaskNotificationService",
    "TaskStatusNotification",
    "TaskProgressNotification",
]
