"""
任务状态通知服务

监控数据库中的任务状态变化，并通过WebSocket推送给前端。
由于任务在独立进程中执行，无法直接访问主进程的WebSocket连接，
因此通过数据库同步状态，主进程监控并推送。
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from loguru import logger

from app.api.v1.backtest_websocket import backtest_ws_manager
from app.core.database import SessionLocal
from app.models.task_models import TaskStatus
from app.repositories.task_repository import TaskRepository
from app.services.backtest.execution.backtest_progress_monitor import (
    backtest_progress_monitor,
)
from app.websocket import manager


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
        session = SessionLocal()
        try:
            task_repository = TaskRepository(session)

            # 获取最近更新的任务（运行中或刚完成的任务）
            # 只检查最近1分钟内更新的任务
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=1)

            # 获取所有运行中的任务
            running_tasks = task_repository.get_tasks_by_status(TaskStatus.RUNNING)

            # 获取最近完成或失败的任务
            recent_tasks = task_repository.get_recently_updated_tasks(cutoff_time)

            # 合并任务列表
            all_tasks = set(running_tasks) | set(recent_tasks)

            for task in all_tasks:
                # 检查是否需要通知
                # 使用 progress 字段的变化来判断是否有更新
                last_check = self._last_check_time.get(task.task_id)
                last_progress = self._last_progress.get(task.task_id, -1)  # 记录上次进度

                # 如果进度有变化，或者时间戳有变化，都需要通知
                progress_changed = task.progress != last_progress
                task_update_time = (
                    task.completed_at or task.started_at or task.created_at
                )
                time_changed = not last_check or (
                    task_update_time and task_update_time > last_check
                )

                if not progress_changed and not time_changed:
                    continue  # 任务状态未变化，跳过

                # 更新最后检查时间和进度
                self._last_check_time[task.task_id] = datetime.now(timezone.utc)
                self._last_progress[task.task_id] = task.progress  # 记录当前进度

                # 发送通知
                await self._notify_task_update(task)

        except Exception as e:
            logger.error(f"检查任务状态失败: {e}", exc_info=True)
        finally:
            session.close()

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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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

    async def _sync_backtest_progress_from_task(self, task):
        """从任务状态同步回测进度到进度监控器"""
        try:
            if task.status == TaskStatus.RUNNING.value:
                progress_data = backtest_progress_monitor.get_progress_data(
                    task.task_id
                )
                if not progress_data:
                    # 初始化进度监控
                    await backtest_progress_monitor.start_backtest_monitoring(
                        task_id=task.task_id,
                        backtest_id=str(uuid.uuid4()),
                        total_trading_days=0,
                    )

                progress_data = backtest_progress_monitor.get_progress_data(
                    task.task_id
                )
                if progress_data:
                    # 从数据库读取详细进度数据
                    db_progress_data = {}
                    if task.result and isinstance(task.result, dict):
                        db_progress_data = task.result.get("progress_data", {})
                        logger.info(
                            f"从数据库读取进度数据: task_id={task.task_id}, progress_data={db_progress_data}, result={task.result}"
                        )

                    # 同步详细数据
                    if db_progress_data:
                        progress_data.processed_trading_days = db_progress_data.get(
                            "processed_days", 0
                        )
                        progress_data.total_trading_days = db_progress_data.get(
                            "total_days", 0
                        )
                        progress_data.current_date = db_progress_data.get(
                            "current_date"
                        )
                        progress_data.total_signals_generated = db_progress_data.get(
                            "total_signals", 0
                        )
                        progress_data.total_trades_executed = db_progress_data.get(
                            "total_trades", 0
                        )
                        progress_data.current_portfolio_value = db_progress_data.get(
                            "portfolio_value", 0.0
                        )
                        logger.info(
                            f"同步进度数据完成: processed_days={progress_data.processed_trading_days}, total_days={progress_data.total_trading_days}, signals={progress_data.total_signals_generated}, trades={progress_data.total_trades_executed}, portfolio={progress_data.current_portfolio_value}"
                        )
                    else:
                        logger.warning(
                            f"数据库中没有进度数据: task_id={task.task_id}, result={task.result}, result_type={type(task.result)}"
                        )

                    # 更新总体进度
                    progress_data.overall_progress = task.progress

                    # 根据任务进度正确设置阶段状态
                    if task.progress >= 30:  # 如果进度 >= 30%，说明前几个阶段已完成
                        # 初始化已完成
                        await backtest_progress_monitor.update_stage(
                            task.task_id,
                            "initialization",
                            progress=100,
                            status="completed",
                        )
                        # 数据加载已完成
                        await backtest_progress_monitor.update_stage(
                            task.task_id,
                            "data_loading",
                            progress=100,
                            status="completed",
                        )
                        # 策略设置已完成
                        await backtest_progress_monitor.update_stage(
                            task.task_id,
                            "strategy_setup",
                            progress=100,
                            status="completed",
                        )

                        # 回测执行阶段
                        if task.progress < 90:
                            execution_progress = (task.progress - 30) / 60 * 100
                            await backtest_progress_monitor.update_stage(
                                task.task_id,
                                "backtest_execution",
                                progress=execution_progress,
                                status="running",
                            )
                        else:
                            # 回测执行已完成
                            await backtest_progress_monitor.update_stage(
                                task.task_id,
                                "backtest_execution",
                                progress=100,
                                status="completed",
                            )

                            if task.progress < 95:
                                await backtest_progress_monitor.update_stage(
                                    task.task_id,
                                    "metrics_calculation",
                                    progress=min((task.progress - 90) / 5 * 100, 100),
                                    status="running",
                                )
                            else:
                                # 指标计算已完成
                                await backtest_progress_monitor.update_stage(
                                    task.task_id,
                                    "metrics_calculation",
                                    progress=100,
                                    status="completed",
                                )
                                await backtest_progress_monitor.update_stage(
                                    task.task_id,
                                    "data_storage",
                                    progress=min((task.progress - 95) / 5 * 100, 100),
                                    status="running",
                                )
                    elif task.progress >= 25:
                        # 初始化已完成
                        await backtest_progress_monitor.update_stage(
                            task.task_id,
                            "initialization",
                            progress=100,
                            status="completed",
                        )
                        # 数据加载已完成
                        await backtest_progress_monitor.update_stage(
                            task.task_id,
                            "data_loading",
                            progress=100,
                            status="completed",
                        )
                        # 策略设置进行中
                        await backtest_progress_monitor.update_stage(
                            task.task_id,
                            "strategy_setup",
                            progress=min((task.progress - 25) / 5 * 100, 100),
                            status="running",
                        )
                    elif task.progress >= 10:
                        # 初始化已完成
                        await backtest_progress_monitor.update_stage(
                            task.task_id,
                            "initialization",
                            progress=100,
                            status="completed",
                        )
                        # 数据加载进行中
                        await backtest_progress_monitor.update_stage(
                            task.task_id,
                            "data_loading",
                            progress=min((task.progress - 10) / 15 * 100, 100),
                            status="running",
                        )
                    else:
                        # 初始化进行中
                        await backtest_progress_monitor.update_stage(
                            task.task_id,
                            "initialization",
                            progress=min(task.progress * 10, 100),
                            status="running",
                        )

        except Exception as e:
            logger.warning(f"同步回测进度失败: {task.task_id}, 错误: {e}", exc_info=True)

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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await manager.send_to_task_subscribers(task.task_id, message)
                logger.debug(
                    f"已发送任务状态更新: {task.task_id}, 状态: {task.status}, 进度: {task.progress}%"
                )

        except Exception as e:
            logger.error(f"发送任务状态通知失败: {task.task_id}, 错误: {e}", exc_info=True)


# 全局任务通知器实例
task_notifier = TaskNotifier()
