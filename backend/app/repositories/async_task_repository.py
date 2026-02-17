"""
异步任务数据存储层 - 处理任务的CRUD操作和数据持久化（异步版本）
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import and_, asc
from sqlalchemy import delete as sql_delete
from sqlalchemy import desc, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm.attributes import flag_modified

from app.core.error_handler import ErrorContext, ErrorSeverity, TaskError
from app.core.logging_config import AuditLogger
from app.models.task_models import Task, TaskStatus, TaskType


class AsyncTaskRepository:
    """异步任务数据仓库"""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    def _to_json_safe(self, value: Any) -> Any:
        """递归转换为可 JSON 序列化的类型"""
        try:
            import numpy as np
        except Exception:
            np = None
        try:
            import pandas as pd
        except Exception:
            pd = None

        from datetime import date, datetime
        from enum import Enum

        if isinstance(value, dict):
            return {k: self._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if pd is not None:
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            if isinstance(value, pd.Series):
                return [self._to_json_safe(v) for v in value.tolist()]
            if isinstance(value, pd.DataFrame):
                return {
                    k: [self._to_json_safe(v) for v in col]
                    for k, col in value.to_dict(orient="list").items()
                }
        if np is not None:
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            if isinstance(value, np.ndarray):
                return [self._to_json_safe(v) for v in value.tolist()]
        return value

    async def create_task(
        self,
        task_name: str,
        task_type: TaskType,
        user_id: str,
        config: Dict[str, Any],
    ) -> Task:
        """创建新任务"""
        try:
            task = Task(
                task_name=task_name,
                task_type=task_type.value,
                user_id=user_id,
                config=config,
                status=TaskStatus.CREATED.value,
                progress=0.0,
                created_at=datetime.utcnow(),
            )

            self.db.add(task)
            await self.db.commit()
            await self.db.refresh(task)

            # 记录审计日志
            AuditLogger.log_user_action(
                action="create_task",
                user_id=user_id,
                resource=f"task:{task.task_id}",
                details={
                    "task_name": task_name,
                    "task_type": task_type.value,
                    "config": config,
                },
            )

            logger.info(f"任务创建成功: {task.task_id}, 类型: {task_type.value}")
            return task

        except Exception as e:
            await self.db.rollback()
            raise TaskError(
                message=f"创建任务失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(user_id=user_id),
                original_exception=e,
            )

    async def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        try:
            result = await self.db.execute(select(Task).filter(Task.task_id == task_id))
            task = result.scalar_one_or_none()
            return task
        except Exception as e:
            raise TaskError(
                message=f"获取任务失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(task_id=task_id),
                original_exception=e,
            )

    async def get_tasks_by_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        status_filter: Optional[TaskStatus] = None,
        task_type_filter: Optional[TaskType] = None,
    ) -> List[Task]:
        """获取用户的任务列表"""
        try:
            query = select(Task).filter(Task.user_id == user_id)

            if status_filter:
                query = query.filter(Task.status == status_filter.value)

            if task_type_filter:
                query = query.filter(Task.task_type == task_type_filter.value)

            query = query.order_by(desc(Task.created_at)).offset(offset).limit(limit)

            result = await self.db.execute(query)
            tasks = result.scalars().all()
            return list(tasks)

        except Exception as e:
            raise TaskError(
                message=f"获取用户任务列表失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(user_id=user_id),
                original_exception=e,
            )

    async def get_tasks_by_status(
        self, status: TaskStatus, limit: int = 100
    ) -> List[Task]:
        """根据状态获取任务列表"""
        try:
            query = (
                select(Task)
                .filter(Task.status == status.value)
                .order_by(asc(Task.created_at))
                .limit(limit)
            )
            result = await self.db.execute(query)
            tasks = result.scalars().all()
            return list(tasks)
        except Exception as e:
            raise TaskError(
                message=f"根据状态获取任务失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: Optional[float] = None,
        result=None,
        error_message: Optional[str] = None,
    ) -> Task:
        """更新任务状态"""
        if result is not None and not isinstance(result, dict):
            raise TypeError(f"result must be a dict, got {type(result)}")

        try:
            task = await self.get_task_by_id(task_id)
            if not task:
                raise TaskError(
                    message=f"任务不存在: {task_id}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(task_id=task_id),
                )

            old_status = task.status
            task.status = status.value

            if progress is not None:
                task.progress = progress

            if result is not None:
                task.result = self._to_json_safe(result)
                flag_modified(task, "result")

            if error_message is not None:
                task.error_message = error_message

            # 更新时间戳
            if status == TaskStatus.RUNNING and not task.started_at:
                task.started_at = datetime.utcnow()
            elif status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ]:
                task.completed_at = datetime.utcnow()

            await self.db.commit()
            await self.db.refresh(task)

            # 记录审计日志
            AuditLogger.log_data_change(
                table="tasks",
                operation="UPDATE",
                record_id=task_id,
                old_values={"status": old_status},
                new_values={"status": status.value},
                user_id=task.user_id,
            )

            logger.info(f"任务状态更新: {task_id}, {old_status} -> {status.value}")
            return task

        except TaskError:
            raise
        except Exception as e:
            await self.db.rollback()
            raise TaskError(
                message=f"更新任务状态失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(task_id=task_id),
                original_exception=e,
            )

    async def update_task_progress(self, task_id: str, progress: float) -> Task:
        """更新任务进度"""
        try:
            task = await self.get_task_by_id(task_id)
            if not task:
                raise TaskError(
                    message=f"任务不存在: {task_id}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(task_id=task_id),
                )

            old_progress = task.progress
            task.progress = progress

            await self.db.commit()
            await self.db.refresh(task)

            logger.debug(f"任务进度更新: {task_id}, {old_progress:.1f}% -> {progress:.1f}%")
            return task

        except TaskError:
            raise
        except Exception as e:
            await self.db.rollback()
            raise TaskError(
                message=f"更新任务进度失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(task_id=task_id),
                original_exception=e,
            )

    async def delete_task(
        self, task_id: str, user_id: str, force: bool = False
    ) -> bool:
        """删除任务"""
        try:
            task = await self.get_task_by_id(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return False

            # 验证用户权限
            if not force and task.user_id != user_id:
                raise TaskError(
                    message="无权限删除此任务",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(task_id=task_id, user_id=user_id),
                )

            # 检测僵尸任务
            is_zombie_task = False
            if task.status == TaskStatus.RUNNING.value:
                now = datetime.utcnow()
                task_age = (
                    (now - task.created_at).total_seconds() / 3600
                    if task.created_at
                    else 0
                )
                time_since_start = (
                    (now - task.started_at).total_seconds() / 3600
                    if task.started_at
                    else 0
                )

                if task_age > 3 or time_since_start > 1.5:
                    is_zombie_task = True
                    logger.info(
                        f"检测到僵尸任务: {task_id}, 创建时间: {task_age:.1f}小时前, 开始运行: {time_since_start:.1f}小时前"
                    )

            # 非强制模式下，只能删除已完成或失败的任务，或僵尸任务
            if not force:
                if task.status not in [
                    TaskStatus.COMPLETED.value,
                    TaskStatus.FAILED.value,
                    TaskStatus.CANCELLED.value,
                ]:
                    if not is_zombie_task:
                        raise TaskError(
                            message=f"该任务正在运行中（状态: {task.status}），请使用强制删除（force=true）或等待任务完成",
                            severity=ErrorSeverity.MEDIUM,
                            context=ErrorContext(task_id=task_id),
                        )
                    else:
                        logger.info(f"自动删除僵尸任务: {task_id}")
            else:
                logger.info(f"强制删除任务: {task_id}, 原状态: {task.status}")

            # 删除相关的详细数据
            try:
                from app.models.backtest_detailed_models import (
                    BacktestBenchmark,
                    BacktestDetailedResult,
                    PortfolioSnapshot,
                    TradeRecord,
                )

                related_tables = [
                    (BacktestDetailedResult, "回测详细结果"),
                    (PortfolioSnapshot, "组合快照"),
                    (TradeRecord, "交易记录"),
                    (BacktestBenchmark, "基准数据"),
                ]

                total_deleted = 0
                for model_class, table_name in related_tables:
                    try:
                        stmt = sql_delete(model_class).where(
                            model_class.task_id == task_id
                        )
                        result = await self.db.execute(stmt)
                        deleted_count = result.rowcount
                        if deleted_count > 0:
                            logger.info(f"删除{table_name}: {deleted_count}条记录")
                            total_deleted += deleted_count
                    except Exception as e:
                        logger.debug(f"删除{table_name}时出错（可能表不存在）: {e}")

                if total_deleted > 0:
                    logger.info(f"已删除任务 {task_id} 的详细数据，共 {total_deleted} 条记录")
                    await self.db.flush()
            except Exception as e:
                logger.warning(f"删除任务详细数据时出错（继续删除主任务）: {e}")

            # 删除主任务
            await self.db.delete(task)
            await self.db.commit()

            # 记录审计日志
            AuditLogger.log_user_action(
                action="delete_task",
                user_id=user_id,
                resource=f"task:{task_id}",
                details={
                    "task_name": task.task_name,
                    "task_type": task.task_type,
                    "force": force,
                },
            )

            logger.info(f"任务删除成功: {task_id}, 强制模式: {force}")
            return True

        except TaskError:
            raise
        except Exception as e:
            await self.db.rollback()
            error_msg = str(e)
            if "foreign key" in error_msg.lower() or "constraint" in error_msg.lower():
                logger.error(f"删除任务失败（数据库约束）: {task_id}, 错误: {error_msg}")
                raise TaskError(
                    message=f"删除任务失败：存在关联数据。请先删除相关数据，或使用强制删除。错误详情: {error_msg}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(task_id=task_id, user_id=user_id),
                    original_exception=e,
                )
            else:
                raise TaskError(
                    message=f"删除任务失败: {error_msg}",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(task_id=task_id, user_id=user_id),
                    original_exception=e,
                )

    async def get_task_statistics(
        self, user_id: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """获取任务统计信息"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            query = select(Task).filter(Task.created_at >= cutoff_date)
            if user_id:
                query = query.filter(Task.user_id == user_id)

            result = await self.db.execute(query)
            tasks = result.scalars().all()

            # 统计各种状态的任务数量
            status_counts = {}
            type_counts = {}

            for task in tasks:
                status_counts[task.status] = status_counts.get(task.status, 0) + 1
                type_counts[task.task_type] = type_counts.get(task.task_type, 0) + 1

            # 计算平均执行时间
            completed_tasks = [t for t in tasks if t.completed_at and t.started_at]
            avg_duration = 0
            if completed_tasks:
                durations = [
                    (t.completed_at - t.started_at).total_seconds()
                    for t in completed_tasks
                ]
                avg_duration = sum(durations) / len(durations)

            return {
                "total_tasks": len(tasks),
                "status_counts": status_counts,
                "type_counts": type_counts,
                "avg_duration_seconds": avg_duration,
                "success_rate": (
                    status_counts.get(TaskStatus.COMPLETED.value, 0)
                    / max(len(tasks), 1)
                ),
                "period_days": days,
            }

        except Exception as e:
            raise TaskError(
                message=f"获取任务统计失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    async def cleanup_old_tasks(self, days: int = 90) -> int:
        """清理旧任务"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # 只清理已完成或失败的任务
            stmt = sql_delete(Task).where(
                and_(
                    Task.completed_at < cutoff_date,
                    or_(
                        Task.status == TaskStatus.COMPLETED.value,
                        Task.status == TaskStatus.FAILED.value,
                    ),
                )
            )

            result = await self.db.execute(stmt)
            deleted_count = result.rowcount

            await self.db.commit()

            logger.info(f"清理旧任务完成: 删除 {deleted_count} 个任务")
            return deleted_count

        except Exception as e:
            await self.db.rollback()
            raise TaskError(
                message=f"清理旧任务失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )
