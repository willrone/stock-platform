"""
任务状态通知服务

监控数据库中的任务状态变化，并通过WebSocket推送给前端。
由于任务在独立进程中执行，无法直接访问主进程的WebSocket连接，
因此通过数据库同步状态，主进程监控并推送。
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Dict, Optional

from loguru import logger

from app.api.v1.backtest_websocket import backtest_ws_manager
from app.core.database import SessionLocal
from app.models.task_models import Task, TaskStatus
from app.repositories.task_repository import TaskRepository
from app.services.backtest.execution.backtest_progress_monitor import (
    BacktestProgressData,
    backtest_progress_monitor,
)
from app.websocket import manager


def utcnow() -> datetime:
    """Return naive UTC datetime for consistent task timestamp handling."""
    return datetime.now(UTC).replace(tzinfo=None)


class TaskNotifier:
    """任务状态通知器"""

    def __init__(self, poll_interval: float = 1.0):
        """
        初始化任务通知器

        Args:
            poll_interval: 轮询数据库的间隔（秒），默认1秒
        """
        self.poll_interval = poll_interval
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_check_time: Dict[str, datetime] = {}  # 记录每个任务的最后检查时间
        self._last_progress: Dict[str, float] = {}  # 记录每个任务的上次进度

    async def start(self):
        """启动任务状态监控"""
        if self.is_running:
            logger.warning("任务通知器已经在运行")
            return

        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("任务状态通知器已启动")

    async def stop(self):
        """停止任务状态监控"""
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("任务状态通知器已停止")

    async def _monitor_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                await self._check_and_notify()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"任务状态监控出错: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

    async def _check_and_notify(self):
        """检查任务状态变化并通知"""
        with SessionLocal() as session:
            try:
                task_repository = TaskRepository(session)

                # 获取最近更新的任务（运行中或刚完成的任务）
                # 只检查最近1分钟内更新的任务
                cutoff_time = utcnow() - timedelta(minutes=1)

                # 获取所有运行中的任务
                running_tasks = task_repository.get_tasks_by_status(TaskStatus.RUNNING)

                # 获取最近完成或失败的任务
                recent_tasks = task_repository.get_recently_updated_tasks(cutoff_time)

                # 合并任务列表
                all_tasks = set(running_tasks) | set(recent_tasks)

                for task in all_tasks:
                    last_check = self._last_check_time.get(task.task_id)
                    last_progress = self._last_progress.get(task.task_id, -1)

                    progress_changed = task.progress != last_progress
                    task_update_time = (
                        task.completed_at or task.started_at or task.created_at
                    )
                    time_changed = not last_check or (
                        task_update_time and task_update_time > last_check
                    )

                    if not progress_changed and not time_changed:
                        continue

                    self._last_check_time[task.task_id] = utcnow()
                    self._last_progress[task.task_id] = task.progress

                    await self._notify_task_update(task)

            except Exception as e:
                logger.error(f"检查任务状态失败: {e}", exc_info=True)

    async def _notify_task_update(self, task):
        """通知任务状态更新"""
        try:
            # 如果是回测任务，需要特殊处理
            if task.task_type == "backtest":
                await self._notify_backtest_update(task)
            else:
                # 普通任务通知（包括qlib_precompute等）
                await self._notify_general_task_update(task)

        except Exception as e:
            logger.error(f"发送任务状态通知失败: {task.task_id}, 错误: {e}", exc_info=True)

    async def _notify_backtest_update(self, task):
        """通知回测任务更新"""
        try:
            # 总是先同步数据库中的最新数据到进度监控器
            await self._sync_backtest_progress_from_task(task)

            # 获取同步后的进度数据
            progress_data = backtest_progress_monitor.get_progress_data(task.task_id)

            if progress_data:
                # 如果有进度监控数据，使用回测WebSocket管理器发送详细进度
                await backtest_ws_manager.send_progress_update(
                    task.task_id, progress_data
                )
            else:
                # 如果仍然没有，发送基本进度消息
                message = {
                    "type": "progress_update",
                    "task_id": task.task_id,
                    "overall_progress": task.progress,
                    "status": task.status,
                    "timestamp": utcnow().isoformat(),
                }

                # 根据任务状态添加信息
                if task.status == TaskStatus.COMPLETED.value:
                    message["type"] = "backtest_completed"
                    if task.result:
                        message["results"] = task.result
                elif task.status == TaskStatus.FAILED.value:
                    message["type"] = "backtest_error"
                    message["error_message"] = task.error_message

                await backtest_ws_manager.send_to_task_subscribers(
                    task.task_id, message
                )

            logger.debug(
                f"已发送回测任务状态更新: {task.task_id}, 状态: {task.status}, 进度: {task.progress}%"
            )

        except Exception as e:
            logger.error(f"发送回测任务状态通知失败: {task.task_id}, 错误: {e}", exc_info=True)
            # 如果回测WebSocket通知失败，回退到通用通知
            await self._notify_general_task_update(task)

    async def _sync_backtest_progress_from_task(self, task: Task) -> None:
        """
        从任务状态同步回测进度到进度监控器。

        注意：running 状态下绝不把 task.result.progress_data 回灌到 monitor，
        以避免旧进度在重跑/重建时被错误继承。
        """
        try:
            if task.status != TaskStatus.RUNNING.value:
                return

            progress_data = backtest_progress_monitor.get_progress_data(task.task_id)
            if not progress_data:
                await self._start_backtest_progress(task.task_id)
                progress_data = backtest_progress_monitor.get_progress_data(
                    task.task_id
                )

            if progress_data and self._should_reset_for_new_run(
                task.progress, progress_data
            ):
                self._reset_progress_for_new_run(progress_data)

            if progress_data:
                progress_data.overall_progress = task.progress

            await self._sync_backtest_stages(task.task_id, task.progress)
        except Exception as e:
            logger.warning(
                f"同步回测进度失败: {task.task_id}, 错误: {e}", exc_info=True
            )

    async def _start_backtest_progress(self, task_id: str) -> None:
        """初始化回测进度监控器。"""
        await backtest_progress_monitor.start_backtest_monitoring(
            task_id=task_id,
            backtest_id=f"bt_{task_id[:8]}",
            total_trading_days=0,
        )

    @staticmethod
    def _should_reset_for_new_run(
        task_progress: float, progress_data: BacktestProgressData
    ) -> bool:
        """
        判断是否需要重置 monitor，避免新运行时显示旧 detailed 进度。

        依据：新任务进入 running 时 progress 初始值通常约 10%，而旧运行会残留
        processed_days/current_date 等字段。
        """
        if task_progress > 10.0:
            return False

        return (
            progress_data.processed_trading_days > 0
            or progress_data.current_date is not None
            or progress_data.total_signals_generated > 0
            or progress_data.total_trades_executed > 0
        )

    @staticmethod
    def _reset_progress_for_new_run(progress_data: BacktestProgressData) -> None:
        """重置回测进度监控器详细字段（仅用于新运行起点）。"""
        now = utcnow()
        progress_data.start_time = now
        progress_data.overall_progress = 0.0
        progress_data.current_stage = "initializing"

        progress_data.total_trading_days = 0
        progress_data.processed_trading_days = 0
        progress_data.current_date = None
        progress_data.processing_speed = 0.0
        progress_data.estimated_completion = None
        progress_data.elapsed_time = None

        progress_data.total_signals_generated = 0
        progress_data.total_trades_executed = 0
        progress_data.current_portfolio_value = 0.0

        progress_data.error_message = None
        progress_data.warnings = []

        for stage in progress_data.stages:
            stage.start_time = None
            stage.end_time = None
            stage.progress = 0.0
            stage.status = "pending"
            stage.details = {}

    async def _sync_backtest_stages(self, task_id: str, task_progress: float) -> None:
        """根据 overall_progress 更新阶段状态。"""
        progress_value = float(task_progress)

        def relative_progress(delta: float, denominator: float) -> float:
            return min((delta / denominator) * 100.0, 100.0)

        stage_updates: list[tuple[str, float, str]] = []
        if progress_value >= 30:
            stage_updates.extend(
                [
                    ("initialization", 100.0, "completed"),
                    ("data_loading", 100.0, "completed"),
                    ("strategy_setup", 100.0, "completed"),
                ]
            )
            if progress_value < 90:
                stage_updates.append(
                    (
                        "backtest_execution",
                        relative_progress(progress_value - 30.0, 60.0),
                        "running",
                    )
                )
            else:
                stage_updates.append(("backtest_execution", 100.0, "completed"))
                if progress_value < 95:
                    stage_updates.append(
                        (
                            "metrics_calculation",
                            relative_progress(progress_value - 90.0, 5.0),
                            "running",
                        )
                    )
                else:
                    stage_updates.append(("metrics_calculation", 100.0, "completed"))
                    stage_updates.append(
                        (
                            "data_storage",
                            relative_progress(progress_value - 95.0, 5.0),
                            "running",
                        )
                    )
        elif progress_value >= 25:
            stage_updates.extend(
                [
                    ("initialization", 100.0, "completed"),
                    ("data_loading", 100.0, "completed"),
                    (
                        "strategy_setup",
                        relative_progress(progress_value - 25.0, 5.0),
                        "running",
                    ),
                ]
            )
        elif progress_value >= 10:
            stage_updates.extend(
                [
                    ("initialization", 100.0, "completed"),
                    (
                        "data_loading",
                        relative_progress(progress_value - 10.0, 15.0),
                        "running",
                    ),
                ]
            )
        else:
            stage_updates.append(
                ("initialization", min(progress_value * 10.0, 100.0), "running")
            )

        for stage_name, progress, status in stage_updates:
            await backtest_progress_monitor.update_stage(
                task_id, stage_name, progress=progress, status=status
            )

    async def _notify_general_task_update(self, task):
        """通知普通任务更新"""
        try:
            # 根据任务状态和进度变化发送不同类型的消息
            if task.status == TaskStatus.RUNNING.value:
                # 运行中：发送进度更新
                message = {
                    "type": "task:progress",
                    "task_id": task.task_id,
                    "status": task.status,
                    "progress": task.progress,
                    "timestamp": utcnow().isoformat(),
                }
                await manager.send_to_task_subscribers(task.task_id, message)
                logger.debug(f"已发送任务进度更新: {task.task_id}, 进度: {task.progress}%")

            elif task.status == TaskStatus.COMPLETED.value:
                # 已完成：发送完成消息
                message = {
                    "type": "task:completed",
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "status": task.status,
                    "progress": task.progress,
                    "completed_at": task.completed_at.isoformat()
                    if task.completed_at
                    else None,
                    "results": task.result,
                    "timestamp": utcnow().isoformat(),
                }
                await manager.send_to_task_subscribers(task.task_id, message)
                logger.debug(f"已发送任务完成通知: {task.task_id}")

            elif task.status == TaskStatus.FAILED.value:
                # 失败：发送失败消息
                message = {
                    "type": "task:failed",
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "status": task.status,
                    "error": task.error_message,
                    "error_message": task.error_message,
                    "timestamp": utcnow().isoformat(),
                }
                await manager.send_to_task_subscribers(task.task_id, message)
                logger.debug(f"已发送任务失败通知: {task.task_id}, 错误: {task.error_message}")
            else:
                # 其他状态：发送通用更新
                message = {
                    "type": "task:update",
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "status": task.status,
                    "progress": task.progress,
                    "timestamp": utcnow().isoformat(),
                }
                await manager.send_to_task_subscribers(task.task_id, message)
                logger.debug(
                    "已发送任务状态更新: "
                    f"{task.task_id}, 状态: {task.status}, "
                    f"进度: {task.progress}%"
                )

        except Exception as e:
            logger.error(f"发送任务状态通知失败: {task.task_id}, 错误: {e}", exc_info=True)


# 全局任务通知器实例
task_notifier = TaskNotifier()
