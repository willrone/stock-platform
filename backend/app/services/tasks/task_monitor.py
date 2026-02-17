"""
任务监控服务
用于监控和处理卡住的任务
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List

from app.core.database import SessionLocal
from app.models.task_models import TaskStatus
from app.repositories.task_repository import TaskRepository

logger = logging.getLogger(__name__)


class TaskMonitor:
    """任务监控器"""

    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = db_path

    def _get_task_timeout_minutes(self, config_str: str, default_minutes: int) -> int:
        """
        从任务 config 中提取超时时间（分钟）

        优先级：
        1. config.optimization_config.timeout（秒）
        2. config.timeout（秒）
        3. 默认值 default_minutes

        Args:
            config_str: 任务配置 JSON 字符串
            default_minutes: 默认超时时间（分钟）

        Returns:
            超时时间（分钟）
        """
        if not config_str:
            return default_minutes
        try:
            config = json.loads(config_str)
            # 优先从 optimization_config.timeout 读取（秒）
            timeout_seconds = None
            opt_cfg = config.get("optimization_config", {})
            if isinstance(opt_cfg, dict):
                timeout_seconds = opt_cfg.get("timeout")
            # 其次从顶层 timeout 读取
            if timeout_seconds is None:
                timeout_seconds = config.get("timeout")
            if timeout_seconds is not None and timeout_seconds > 0:
                return max(int(timeout_seconds / 60), 1)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return default_minutes

    def get_stuck_tasks(self, timeout_minutes: int = 30) -> List[Dict[str, Any]]:
        """
        获取卡住的任务

        对每个任务单独判断超时：优先使用任务 config 中的 timeout，
        没有则使用 timeout_minutes 默认值。

        Args:
            timeout_minutes: 默认任务超时时间（分钟），任务自带 timeout 优先

        Returns:
            卡住的任务列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 查询所有运行中/排队中的任务（含 config 用于读取自定义超时）
            cursor.execute(
                """
                SELECT task_id, task_name, task_type, status,
                       created_at, started_at, progress, config
                FROM tasks 
                WHERE status IN ('running', 'queued')
                ORDER BY created_at DESC
            """
            )

            now = datetime.now()
            tasks = []
            for row in cursor.fetchall():
                (
                    task_id,
                    task_name,
                    task_type,
                    status,
                    created_at,
                    started_at,
                    progress,
                    config_str,
                ) = row

                # 计算该任务的实际超时阈值
                task_timeout = self._get_task_timeout_minutes(
                    config_str, timeout_minutes
                )

                # 判断是否超时
                ref_time_str = started_at or created_at
                if not ref_time_str:
                    continue
                try:
                    ref_time = datetime.fromisoformat(ref_time_str)
                except ValueError:
                    continue

                elapsed_minutes = (now - ref_time).total_seconds() / 60
                if elapsed_minutes < task_timeout:
                    continue  # 未超时，跳过

                logger.info(
                    f"任务超时: {task_id} ({task_name}) "
                    f"已运行 {elapsed_minutes:.0f} 分钟，"
                    f"超时阈值 {task_timeout} 分钟"
                    f"（{'自定义' if config_str else '默认'}）"
                )

                tasks.append(
                    {
                        "task_id": task_id,
                        "task_name": task_name,
                        "task_type": task_type,
                        "status": status,
                        "created_at": created_at,
                        "started_at": started_at,
                        "progress": progress,
                        "timeout_minutes": task_timeout,
                    }
                )

            return tasks

        except Exception as e:
            logger.error(f"获取卡住任务失败: {e}")
            return []
        finally:
            conn.close()

    def force_complete_task(self, task_id: str, status: str = "cancelled") -> bool:
        """
        强制完成任务

        Args:
            task_id: 任务ID
            status: 新状态 (cancelled, failed, completed)

        Returns:
            是否成功
        """
        try:
            session = SessionLocal()
            task_repository = TaskRepository(session)

            # 更新任务状态
            task = task_repository.update_task_status(
                task_id=task_id,
                status=getattr(TaskStatus, status.upper()),
                progress=100.0,
            )

            if task:
                logger.info(f"强制完成任务: {task_id} -> {status}")
                return True
            else:
                logger.warning(f"任务不存在: {task_id}")
                return False

        except Exception as e:
            logger.error(f"强制完成任务失败: {task_id}, 错误: {e}")
            return False
        finally:
            session.close()

    def cleanup_stuck_tasks(
        self, timeout_minutes: int = 30, auto_fix: bool = False
    ) -> Dict[str, Any]:
        """
        清理卡住的任务

        Args:
            timeout_minutes: 任务超时时间（分钟）
            auto_fix: 是否自动修复

        Returns:
            清理结果
        """
        stuck_tasks = self.get_stuck_tasks(timeout_minutes)

        result = {
            "total_stuck": len(stuck_tasks),
            "fixed_tasks": [],
            "failed_tasks": [],
        }

        if not stuck_tasks:
            logger.info("没有发现卡住的任务")
            return result

        logger.info(f"发现 {len(stuck_tasks)} 个卡住的任务")

        for task in stuck_tasks:
            task_id = task["task_id"]
            task_name = task["task_name"]

            if auto_fix:
                # 自动修复：将任务标记为已取消
                if self.force_complete_task(task_id, "cancelled"):
                    result["fixed_tasks"].append(
                        {
                            "task_id": task_id,
                            "task_name": task_name,
                            "action": "cancelled",
                        }
                    )
                    logger.info(f"自动修复任务: {task_id} ({task_name})")
                else:
                    result["failed_tasks"].append(
                        {"task_id": task_id, "task_name": task_name, "error": "修复失败"}
                    )
            else:
                logger.info(
                    f"发现卡住任务: {task_id} ({task_name}) - {task['status']} - {task['progress']}%"
                )

        return result

    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 统计各状态任务数量
            cursor.execute(
                """
                SELECT status, COUNT(*) as count
                FROM tasks 
                GROUP BY status
            """
            )

            status_counts = {}
            for status, count in cursor.fetchall():
                status_counts[status] = count

            # 统计最近24小时的任务
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM tasks 
                WHERE created_at > ?
            """,
                (yesterday,),
            )

            recent_count = cursor.fetchone()[0]

            return {
                "status_counts": status_counts,
                "recent_24h": recent_count,
                "total_tasks": sum(status_counts.values()),
            }

        except Exception as e:
            logger.error(f"获取任务统计失败: {e}")
            return {}
        finally:
            conn.close()


# 全局任务监控器实例
task_monitor = TaskMonitor()
