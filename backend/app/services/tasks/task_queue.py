"""
任务队列和调度器 - 处理任务的排队、调度和执行管理
"""

import asyncio
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from queue import Empty, PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from app.core.error_handler import ErrorContext, ErrorSeverity, TaskError
from app.core.logging_config import PerformanceLogger
from app.models.task_models import Task, TaskStatus, TaskType


class TaskPriority(Enum):
    """任务优先级"""

    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class QueuedTask:
    """队列中的任务"""

    task_id: str
    task_type: TaskType
    priority: TaskPriority
    config: Dict[str, Any]
    user_id: str
    created_at: datetime
    estimated_duration: Optional[int] = None  # 预估执行时间（秒）
    retry_count: int = 0
    max_retries: int = 3

    def __lt__(self, other):
        """优先级比较，用于优先队列排序"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # 相同优先级按创建时间排序
        return self.created_at < other.created_at


@dataclass
class TaskExecutionContext:
    """任务执行上下文"""

    task_id: str
    executor_id: str
    start_time: datetime
    estimated_end_time: Optional[datetime] = None
    progress_callback: Optional[Callable] = None
    cancel_event: Optional[threading.Event] = None


class TaskExecutor:
    """任务执行器"""

    def __init__(self, executor_id: str, max_concurrent_tasks: int = 2):
        self.executor_id = executor_id
        self.max_concurrent_tasks = max_concurrent_tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.running_tasks: Dict[str, TaskExecutionContext] = {}
        self.task_handlers: Dict[TaskType, Callable] = {}
        self.is_running = True

    def register_handler(self, task_type: TaskType, handler: Callable):
        """注册任务处理器"""
        self.task_handlers[task_type] = handler
        logger.info(f"注册任务处理器: {task_type.value} -> {handler.__name__}")

    def can_accept_task(self) -> bool:
        """检查是否可以接受新任务"""
        return len(self.running_tasks) < self.max_concurrent_tasks and self.is_running

    def execute_task(
        self, queued_task: QueuedTask, progress_callback: Optional[Callable] = None
    ) -> Future:
        """执行任务"""
        if not self.can_accept_task():
            raise TaskError(
                message="执行器已满，无法接受新任务",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(task_id=queued_task.task_id),
            )

        handler = self.task_handlers.get(queued_task.task_type)
        if not handler:
            raise TaskError(
                message=f"未找到任务类型处理器: {queued_task.task_type.value}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(task_id=queued_task.task_id),
            )

        # 创建执行上下文
        cancel_event = threading.Event()
        context = TaskExecutionContext(
            task_id=queued_task.task_id,
            executor_id=self.executor_id,
            start_time=datetime.utcnow(),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        if queued_task.estimated_duration:
            context.estimated_end_time = context.start_time + timedelta(
                seconds=queued_task.estimated_duration
            )

        self.running_tasks[queued_task.task_id] = context

        # 提交任务到线程池
        future = self.thread_pool.submit(
            self._execute_task_with_context, handler, queued_task, context
        )

        logger.info(f"任务开始执行: {queued_task.task_id}, 执行器: {self.executor_id}")
        return future

    def _execute_task_with_context(
        self, handler: Callable, queued_task: QueuedTask, context: TaskExecutionContext
    ) -> Dict[str, Any]:
        """在上下文中执行任务"""
        try:
            # 执行任务
            result = handler(queued_task, context)

            # 记录性能指标
            duration = (datetime.utcnow() - context.start_time).total_seconds()
            PerformanceLogger.log_task_performance(
                task_id=queued_task.task_id,
                task_type=queued_task.task_type.value,
                duration_seconds=duration,
                success=True,
                details={"executor_id": self.executor_id},
            )

            logger.info(f"任务执行完成: {queued_task.task_id}, 耗时: {duration:.2f}秒")
            return result

        except Exception as e:
            # 记录失败的性能指标
            duration = (datetime.utcnow() - context.start_time).total_seconds()
            PerformanceLogger.log_task_performance(
                task_id=queued_task.task_id,
                task_type=queued_task.task_type.value,
                duration_seconds=duration,
                success=False,
                details={"executor_id": self.executor_id, "error": str(e)},
            )

            logger.error(f"任务执行失败: {queued_task.task_id}, 错误: {e}")
            raise
        finally:
            # 清理执行上下文
            self.running_tasks.pop(queued_task.task_id, None)

    def cancel_task(self, task_id: str) -> bool:
        """取消任务执行"""
        context = self.running_tasks.get(task_id)
        if not context:
            return False

        # 设置取消事件
        if context.cancel_event:
            context.cancel_event.set()

        logger.info(f"任务取消请求: {task_id}")
        return True

    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """获取正在运行的任务"""
        running_tasks = []
        for task_id, context in self.running_tasks.items():
            task_info = {
                "task_id": task_id,
                "executor_id": context.executor_id,
                "start_time": context.start_time.isoformat(),
                "estimated_end_time": context.estimated_end_time.isoformat()
                if context.estimated_end_time
                else None,
                "running_duration": (
                    datetime.utcnow() - context.start_time
                ).total_seconds(),
            }
            running_tasks.append(task_info)

        return running_tasks

    def shutdown(self):
        """关闭执行器"""
        self.is_running = False
        self.thread_pool.shutdown(wait=True)
        logger.info(f"任务执行器关闭: {self.executor_id}")


class TaskScheduler:
    """任务调度器"""

    def __init__(self, max_executors: int = 3):
        self.task_queue = PriorityQueue()
        self.executors: List[TaskExecutor] = []
        self.max_executors = max_executors
        self.scheduler_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.task_timeout_seconds = 3600  # 默认任务超时1小时

        # 统计信息
        self.stats = {
            "total_queued": 0,
            "total_executed": 0,
            "total_failed": 0,
            "total_timeout": 0,
            "queue_size": 0,
        }

        # 创建执行器
        for i in range(max_executors):
            executor = TaskExecutor(f"executor_{i}", max_concurrent_tasks=2)
            self.executors.append(executor)

    def register_task_handler(self, task_type: TaskType, handler: Callable):
        """为所有执行器注册任务处理器"""
        for executor in self.executors:
            executor.register_handler(task_type, handler)

    def start(self):
        """启动调度器"""
        if self.is_running:
            return

        self.is_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self.scheduler_thread.start()
        logger.info("任务调度器启动")

    def stop(self):
        """停止调度器"""
        self.is_running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        # 关闭所有执行器
        for executor in self.executors:
            executor.shutdown()

        logger.info("任务调度器停止")

    def enqueue_task(
        self,
        task_id: str,
        task_type: TaskType,
        config: Dict[str, Any],
        user_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        estimated_duration: Optional[int] = None,
    ) -> bool:
        """将任务加入队列"""
        try:
            queued_task = QueuedTask(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                config=config,
                user_id=user_id,
                created_at=datetime.utcnow(),
                estimated_duration=estimated_duration,
            )

            self.task_queue.put(queued_task)
            self.stats["total_queued"] += 1
            self.stats["queue_size"] = self.task_queue.qsize()

            logger.info(
                f"任务加入队列: {task_id}, 优先级: {priority.name}, 队列大小: {self.stats['queue_size']}"
            )
            return True

        except Exception as e:
            logger.error(f"任务入队失败: {task_id}, 错误: {e}")
            return False

    def _scheduler_loop(self):
        """调度器主循环"""
        while self.is_running:
            try:
                # 检查是否有可用的执行器
                available_executor = self._get_available_executor()
                if not available_executor:
                    time.sleep(1)  # 没有可用执行器，等待1秒
                    continue

                # 从队列获取任务
                try:
                    queued_task = self.task_queue.get(timeout=1)
                    self.stats["queue_size"] = self.task_queue.qsize()
                except Empty:
                    continue  # 队列为空，继续循环

                # 执行任务
                try:
                    future = available_executor.execute_task(
                        queued_task,
                        progress_callback=self._create_progress_callback(
                            queued_task.task_id
                        ),
                    )

                    # 监控任务执行
                    self._monitor_task_execution(queued_task, future)

                except Exception as e:
                    logger.error(f"任务执行启动失败: {queued_task.task_id}, 错误: {e}")
                    self._handle_task_failure(queued_task, str(e))

            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                time.sleep(1)

    def _get_available_executor(self) -> Optional[TaskExecutor]:
        """获取可用的执行器"""
        for executor in self.executors:
            if executor.can_accept_task():
                return executor
        return None

    def _create_progress_callback(self, task_id: str) -> Callable:
        """创建进度回调函数"""

        def progress_callback(progress: float, message: str = ""):
            # 这里可以通过WebSocket发送进度更新
            logger.debug(f"任务进度更新: {task_id}, 进度: {progress:.2f}, 消息: {message}")

        return progress_callback

    def _monitor_task_execution(self, queued_task: QueuedTask, future: Future):
        """监控任务执行"""

        def monitor():
            try:
                # 等待任务完成或超时
                result = future.result(timeout=self.task_timeout_seconds)
                self.stats["total_executed"] += 1
                logger.info(f"任务执行成功: {queued_task.task_id}")

            except TimeoutError:
                self.stats["total_timeout"] += 1
                logger.error(f"任务执行超时: {queued_task.task_id}")
                self._handle_task_timeout(queued_task)

            except Exception as e:
                self.stats["total_failed"] += 1
                logger.error(f"任务执行异常: {queued_task.task_id}, 错误: {e}")
                self._handle_task_failure(queued_task, str(e))

        # 在单独线程中监控
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def _handle_task_failure(self, queued_task: QueuedTask, error_message: str):
        """处理任务失败"""
        # 检查是否需要重试
        if queued_task.retry_count < queued_task.max_retries:
            queued_task.retry_count += 1
            logger.info(f"任务重试: {queued_task.task_id}, 重试次数: {queued_task.retry_count}")

            # 延迟重新入队
            def retry_task():
                time.sleep(min(2**queued_task.retry_count, 60))  # 指数退避
                self.task_queue.put(queued_task)

            retry_thread = threading.Thread(target=retry_task, daemon=True)
            retry_thread.start()
        else:
            logger.error(f"任务最终失败: {queued_task.task_id}, 已达到最大重试次数")

    def _handle_task_timeout(self, queued_task: QueuedTask):
        """处理任务超时"""
        # 尝试取消任务
        for executor in self.executors:
            if executor.cancel_task(queued_task.task_id):
                break

        logger.warning(f"任务超时处理: {queued_task.task_id}")

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        running_tasks = []
        for executor in self.executors:
            running_tasks.extend(executor.get_running_tasks())

        return {
            "queue_size": self.task_queue.qsize(),
            "running_tasks": len(running_tasks),
            "available_executors": len(
                [e for e in self.executors if e.can_accept_task()]
            ),
            "total_executors": len(self.executors),
            "statistics": self.stats.copy(),
            "running_task_details": running_tasks,
        }

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 尝试从执行器中取消
        for executor in self.executors:
            if executor.cancel_task(task_id):
                return True

        # 如果任务还在队列中，需要从队列中移除（这比较复杂，暂时跳过）
        logger.warning(f"无法取消任务: {task_id}，任务可能已完成或不在执行中")
        return False

    def set_task_timeout(self, timeout_seconds: int):
        """设置任务超时时间"""
        self.task_timeout_seconds = timeout_seconds
        logger.info(f"任务超时时间设置为: {timeout_seconds}秒")


class TaskQueueManager:
    """任务队列管理器 - 统一管理多个调度器"""

    def __init__(self):
        self.schedulers: Dict[str, TaskScheduler] = {}
        self.default_scheduler_name = "default"

        # 创建默认调度器
        self.create_scheduler(self.default_scheduler_name, max_executors=3)

    def create_scheduler(self, name: str, max_executors: int = 3) -> TaskScheduler:
        """创建新的调度器"""
        if name in self.schedulers:
            raise TaskError(message=f"调度器已存在: {name}", severity=ErrorSeverity.MEDIUM)

        scheduler = TaskScheduler(max_executors=max_executors)
        self.schedulers[name] = scheduler
        logger.info(f"创建调度器: {name}, 最大执行器数: {max_executors}")
        return scheduler

    def get_scheduler(self, name: str = None) -> TaskScheduler:
        """获取调度器"""
        scheduler_name = name or self.default_scheduler_name
        scheduler = self.schedulers.get(scheduler_name)

        if not scheduler:
            raise TaskError(
                message=f"调度器不存在: {scheduler_name}", severity=ErrorSeverity.MEDIUM
            )

        return scheduler

    def start_all_schedulers(self):
        """启动所有调度器"""
        for name, scheduler in self.schedulers.items():
            scheduler.start()
            logger.info(f"启动调度器: {name}")

    def stop_all_schedulers(self):
        """停止所有调度器"""
        for name, scheduler in self.schedulers.items():
            scheduler.stop()
            logger.info(f"停止调度器: {name}")

    def enqueue_task(
        self,
        task_id: str,
        task_type: TaskType,
        config: Dict[str, Any],
        user_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        scheduler_name: str = None,
    ) -> bool:
        """将任务加入指定调度器的队列"""
        scheduler = self.get_scheduler(scheduler_name)
        return scheduler.enqueue_task(task_id, task_type, config, user_id, priority)

    def get_overall_status(self) -> Dict[str, Any]:
        """获取所有调度器的整体状态"""
        overall_stats = {"total_schedulers": len(self.schedulers), "schedulers": {}}

        for name, scheduler in self.schedulers.items():
            overall_stats["schedulers"][name] = scheduler.get_queue_status()

        return overall_stats


# 全局任务队列管理器实例
task_queue_manager = TaskQueueManager()
