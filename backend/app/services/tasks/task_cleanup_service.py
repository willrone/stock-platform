"""
任务清理服务
定期清理卡住的任务
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from app.services.tasks.task_monitor import task_monitor

logger = logging.getLogger(__name__)


class TaskCleanupService:
    """任务清理服务"""

    def __init__(
        self,
        cleanup_interval_minutes: int = 30,
        task_timeout_minutes: int = 60,
        auto_cleanup: bool = True,
    ):
        """
        初始化任务清理服务

        Args:
            cleanup_interval_minutes: 清理间隔（分钟）
            task_timeout_minutes: 任务超时时间（分钟）
            auto_cleanup: 是否自动清理
        """
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self.task_timeout_minutes = task_timeout_minutes
        self.auto_cleanup = auto_cleanup
        self.is_running = False
        self.cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动清理服务"""
        if self.is_running:
            logger.warning("任务清理服务已在运行")
            return

        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(
            f"任务清理服务已启动，清理间隔: {self.cleanup_interval_minutes}分钟，任务超时: {self.task_timeout_minutes}分钟"
        )

    async def stop(self):
        """停止清理服务"""
        if not self.is_running:
            return

        self.is_running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("任务清理服务已停止")

    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"任务清理循环出错: {e}", exc_info=True)
                await asyncio.sleep(60)  # 出错后等待1分钟再继续

    async def _perform_cleanup(self):
        """执行清理操作"""
        try:
            # 获取卡住的任务
            stuck_tasks = task_monitor.get_stuck_tasks(self.task_timeout_minutes)

            if not stuck_tasks:
                logger.debug("没有发现卡住的任务")
                return

            logger.info(f"发现 {len(stuck_tasks)} 个卡住的任务")

            if self.auto_cleanup:
                # 自动清理卡住的任务
                result = task_monitor.cleanup_stuck_tasks(
                    timeout_minutes=self.task_timeout_minutes, auto_fix=True
                )

                if result["fixed_tasks"]:
                    logger.info(f"自动清理了 {len(result['fixed_tasks'])} 个卡住的任务")

                if result["failed_tasks"]:
                    logger.warning(f"清理失败 {len(result['failed_tasks'])} 个任务")
            else:
                # 仅记录，不自动清理
                for task in stuck_tasks:
                    logger.warning(
                        f"发现卡住任务: {task['task_id']} ({task['task_name']}) - {task['status']} - {task['progress']}%"
                    )

        except Exception as e:
            logger.error(f"执行任务清理失败: {e}", exc_info=True)

    async def manual_cleanup(self, task_timeout_minutes: Optional[int] = None) -> dict:
        """手动执行清理"""
        timeout = task_timeout_minutes or self.task_timeout_minutes

        try:
            result = task_monitor.cleanup_stuck_tasks(
                timeout_minutes=timeout, auto_fix=True
            )

            logger.info(
                f"手动清理完成: 处理 {result['total_stuck']} 个任务，修复 {len(result['fixed_tasks'])} 个"
            )
            return result

        except Exception as e:
            logger.error(f"手动清理失败: {e}", exc_info=True)
            raise


# 全局任务清理服务实例
task_cleanup_service = TaskCleanupService(
    cleanup_interval_minutes=30,  # 每30分钟检查一次
    task_timeout_minutes=60,  # 任务超时1小时
    auto_cleanup=False,  # 暂时禁用自动清理（优化任务被误杀，待修复竞态条件后再开启）
)
