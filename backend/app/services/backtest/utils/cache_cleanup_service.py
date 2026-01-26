"""
缓存清理服务
定期清理过期的图表缓存和旧的回测数据
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from loguru import logger

from app.core.database import get_async_session
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository
from app.services.backtest.utils.chart_cache_service import chart_cache_service


class CacheCleanupService:
    """缓存清理服务"""

    def __init__(self):
        self.logger = logger.bind(service="cache_cleanup")
        self.is_running = False
        self.cleanup_interval_hours = 6  # 每6小时清理一次
        self.data_retention_days = 30  # 保留30天的数据

    async def start_cleanup_scheduler(self):
        """启动清理调度器"""
        if self.is_running:
            self.logger.warning("清理调度器已在运行")
            return

        self.is_running = True
        self.logger.info(f"启动缓存清理调度器，清理间隔: {self.cleanup_interval_hours}小时")

        try:
            while self.is_running:
                # 执行清理任务
                await self.run_cleanup_tasks()

                # 等待下次清理
                await asyncio.sleep(self.cleanup_interval_hours * 3600)

        except asyncio.CancelledError:
            self.logger.info("清理调度器被取消")
        except Exception as e:
            self.logger.error(f"清理调度器异常: {e}", exc_info=True)
        finally:
            self.is_running = False

    async def stop_cleanup_scheduler(self):
        """停止清理调度器"""
        if not self.is_running:
            self.logger.warning("清理调度器未在运行")
            return

        self.is_running = False
        self.logger.info("停止缓存清理调度器")

    async def run_cleanup_tasks(self) -> Dict[str, Any]:
        """执行清理任务"""
        self.logger.info("开始执行缓存清理任务...")

        cleanup_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "expired_cache_cleaned": 0,
            "old_data_cleaned": {},
            "errors": [],
        }

        try:
            # 1. 清理过期的图表缓存
            expired_count = await self.cleanup_expired_chart_cache()
            cleanup_results["expired_cache_cleaned"] = expired_count

            # 2. 清理旧的回测详细数据
            old_data_results = await self.cleanup_old_backtest_data()
            cleanup_results["old_data_cleaned"] = old_data_results

            self.logger.info(f"缓存清理任务完成: {cleanup_results}")

        except Exception as e:
            error_msg = f"清理任务执行失败: {e}"
            self.logger.error(error_msg, exc_info=True)
            cleanup_results["errors"].append(error_msg)

        return cleanup_results

    async def cleanup_expired_chart_cache(self) -> int:
        """清理过期的图表缓存"""
        try:
            self.logger.info("清理过期的图表缓存...")
            expired_count = await chart_cache_service.cleanup_expired_cache()

            if expired_count > 0:
                self.logger.info(f"清理了 {expired_count} 条过期缓存记录")
            else:
                self.logger.debug("没有过期的缓存记录需要清理")

            return expired_count

        except Exception as e:
            self.logger.error(f"清理过期缓存失败: {e}", exc_info=True)
            return 0

    async def cleanup_old_backtest_data(self) -> Dict[str, int]:
        """清理旧的回测详细数据"""
        try:
            self.logger.info(f"清理 {self.data_retention_days} 天前的回测详细数据...")

            async for session in get_async_session():
                repository = BacktestDetailedRepository(session)
                cleanup_results = await repository.cleanup_old_data(
                    self.data_retention_days
                )
                await session.commit()

                total_cleaned = sum(cleanup_results.values())
                if total_cleaned > 0:
                    self.logger.info(f"清理了旧数据: {cleanup_results}")
                else:
                    self.logger.debug("没有旧数据需要清理")

                return cleanup_results

        except Exception as e:
            self.logger.error(f"清理旧数据失败: {e}", exc_info=True)
            return {}

    async def force_cleanup_task_data(self, task_id: str) -> bool:
        """强制清理特定任务的所有数据"""
        try:
            self.logger.info(f"强制清理任务数据: task_id={task_id}")

            # 1. 清理图表缓存
            cache_success = await chart_cache_service.invalidate_cache(task_id)

            # 2. 清理详细数据
            async for session in get_async_session():
                repository = BacktestDetailedRepository(session)
                data_success = await repository.delete_task_data(task_id)
                await session.commit()

            success = cache_success and data_success

            if success:
                self.logger.info(f"成功清理任务数据: task_id={task_id}")
            else:
                self.logger.warning(f"清理任务数据部分失败: task_id={task_id}")

            return success

        except Exception as e:
            self.logger.error(f"强制清理任务数据失败: {e}", exc_info=True)
            return False

    async def get_cleanup_statistics(self) -> Dict[str, Any]:
        """获取清理统计信息"""
        try:
            # 获取缓存统计
            cache_stats = await chart_cache_service.get_cache_statistics()

            # 获取数据库统计
            async for session in get_async_session():
                repository = BacktestDetailedRepository(session)

                # 这里可以添加更多统计查询
                # 暂时返回基本信息
                db_stats = {
                    "data_retention_days": self.data_retention_days,
                    "cleanup_interval_hours": self.cleanup_interval_hours,
                }

            return {
                "service_status": {
                    "is_running": self.is_running,
                    "cleanup_interval_hours": self.cleanup_interval_hours,
                    "data_retention_days": self.data_retention_days,
                },
                "cache_statistics": cache_stats,
                "database_statistics": db_stats,
                "last_cleanup": None,  # 可以添加上次清理时间记录
            }

        except Exception as e:
            self.logger.error(f"获取清理统计失败: {e}", exc_info=True)
            return {"error": str(e)}

    async def manual_cleanup(
        self,
        cleanup_expired_cache: bool = True,
        cleanup_old_data: bool = True,
        custom_retention_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """手动执行清理任务"""
        self.logger.info("手动执行清理任务...")

        cleanup_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "manual_trigger": True,
            "expired_cache_cleaned": 0,
            "old_data_cleaned": {},
            "errors": [],
        }

        try:
            if cleanup_expired_cache:
                expired_count = await self.cleanup_expired_chart_cache()
                cleanup_results["expired_cache_cleaned"] = expired_count

            if cleanup_old_data:
                # 使用自定义保留天数或默认值
                retention_days = custom_retention_days or self.data_retention_days

                async for session in get_async_session():
                    repository = BacktestDetailedRepository(session)
                    old_data_results = await repository.cleanup_old_data(retention_days)
                    await session.commit()
                    cleanup_results["old_data_cleaned"] = old_data_results

            self.logger.info(f"手动清理任务完成: {cleanup_results}")

        except Exception as e:
            error_msg = f"手动清理任务失败: {e}"
            self.logger.error(error_msg, exc_info=True)
            cleanup_results["errors"].append(error_msg)

        return cleanup_results

    def configure_cleanup_settings(
        self,
        cleanup_interval_hours: Optional[int] = None,
        data_retention_days: Optional[int] = None,
    ):
        """配置清理设置"""
        if cleanup_interval_hours is not None:
            self.cleanup_interval_hours = max(1, cleanup_interval_hours)  # 最少1小时
            self.logger.info(f"更新清理间隔: {self.cleanup_interval_hours}小时")

        if data_retention_days is not None:
            self.data_retention_days = max(1, data_retention_days)  # 最少保留1天
            self.logger.info(f"更新数据保留天数: {self.data_retention_days}天")


# 全局清理服务实例
cache_cleanup_service = CacheCleanupService()


async def start_background_cleanup():
    """启动后台清理服务"""
    try:
        await cache_cleanup_service.start_cleanup_scheduler()
    except Exception as e:
        logger.error(f"启动后台清理服务失败: {e}", exc_info=True)


async def stop_background_cleanup():
    """停止后台清理服务"""
    try:
        await cache_cleanup_service.stop_cleanup_scheduler()
    except Exception as e:
        logger.error(f"停止后台清理服务失败: {e}", exc_info=True)
