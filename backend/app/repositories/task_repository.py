"""
任务数据存储层 - 处理任务的CRUD操作和数据持久化
"""

from __future__ import annotations  # 延迟评估类型注解，避免在独立进程中类型解析问题

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast

from loguru import logger
from sqlalchemy import and_, asc, desc, func, or_
from sqlalchemy.orm import Session, defer

from app.core.error_handler import ErrorContext, ErrorSeverity, TaskError
from app.core.logging_config import AuditLogger
from app.models.task_models import (
    BacktestResult,
    ModelInfo,
    PredictionResult,
    Task,
    TaskStatus,
    TaskType,
)


class TaskRepository:
    """任务数据仓库"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def _to_json_safe(self, value: Any) -> Any:
        """递归转换为可 JSON 序列化的类型"""
        np: Any = None
        pd: Any = None
        try:
            import numpy as _np

            np = _np
        except Exception:
            pass
        try:
            import pandas as _pd

            pd = _pd
        except Exception:
            pass

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

    def create_task(
        self,
        task_name: str,
        task_type: TaskType,
        user_id: str,
        config: "Dict[str, Any]",
    ) -> "Task":
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
            self.db.commit()
            self.db.refresh(task)

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
            self.db.rollback()
            raise TaskError(
                message=f"创建任务失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(user_id=user_id),
                original_exception=e,
            )

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        try:
            task = self.db.query(Task).filter(Task.task_id == task_id).first()
            return cast(Optional[Task], task)
        except Exception as e:
            raise TaskError(
                message=f"获取任务失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(task_id=task_id),
                original_exception=e,
            )

    def get_tasks_by_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        status_filter: Optional[TaskStatus] = None,
        task_type_filter: Optional[TaskType] = None,
        exclude_result: bool = False,
    ) -> List[Task]:
        """获取用户的任务列表

        Args:
            exclude_result: 如果为 True，不加载 result 列（列表查询时可显著减少 I/O）。
        """
        try:
            query = self.db.query(Task).filter(Task.user_id == user_id)

            if exclude_result:
                query = query.options(defer(cast(Any, Task.result)))

            if status_filter:
                query = query.filter(Task.status == status_filter.value)

            if task_type_filter:
                query = query.filter(Task.task_type == task_type_filter.value)

            tasks = (
                query.order_by(desc(Task.created_at)).offset(offset).limit(limit).all()
            )
            return cast(List[Task], tasks)

        except Exception as e:
            raise TaskError(
                message=f"获取用户任务列表失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(user_id=user_id),
                original_exception=e,
            )

    def count_tasks_by_user(
        self,
        user_id: str,
        status_filter: Optional[TaskStatus] = None,
        task_type_filter: Optional[TaskType] = None,
    ) -> int:
        """使用 COUNT(*) 高效获取用户任务总数"""
        try:
            query = self.db.query(func.count(Task.task_id)).filter(
                Task.user_id == user_id
            )

            if status_filter:
                query = query.filter(Task.status == status_filter.value)

            if task_type_filter:
                query = query.filter(Task.task_type == task_type_filter.value)

            return query.scalar() or 0

        except Exception as e:
            raise TaskError(
                message=f"获取用户任务总数失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(user_id=user_id),
                original_exception=e,
            )

    def get_tasks_by_status(self, status: TaskStatus, limit: int = 100) -> List[Task]:
        """根据状态获取任务列表"""
        try:
            tasks = (
                self.db.query(Task)
                .filter(Task.status == status.value)
                .order_by(asc(Task.created_at))
                .limit(limit)
                .all()
            )
            return cast(List[Task], tasks)
        except Exception as e:
            raise TaskError(
                message=f"根据状态获取任务失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    def get_recently_updated_tasks(
        self, since: datetime, limit: int = 100
    ) -> List[Task]:
        """获取最近更新的任务列表"""
        try:
            # 使用completed_at或started_at来判断最近更新的任务
            tasks = (
                self.db.query(Task)
                .filter(or_(Task.completed_at >= since, Task.started_at >= since))
                .filter(
                    Task.status.in_(
                        [
                            TaskStatus.COMPLETED.value,
                            TaskStatus.FAILED.value,
                            TaskStatus.CANCELLED.value,
                            TaskStatus.RUNNING.value,  # 也包括运行中的任务
                        ]
                    )
                )
                .order_by(desc(Task.created_at))
                .limit(limit)
                .all()
            )
            return cast(List[Task], tasks)
        except Exception as e:
            raise TaskError(
                message=f"获取最近更新的任务失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: Optional[float] = None,
        result: Any = None,  # 使用 None 作为默认值，避免类型注解问题
        error_message: Optional[str] = None,
    ) -> "Task":
        """更新任务状态"""
        # 在方法内部确保 Any 可用（用于类型检查，但不影响运行时）

        if result is not None and not isinstance(result, dict):
            raise TypeError(f"result must be a dict, got {type(result)}")

        try:
            task = self.get_task_by_id(task_id)
            if not task:
                raise TaskError(
                    message=f"任务不存在: {task_id}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(task_id=task_id),
                )

            task_record = cast(Any, task)
            old_status = cast(str, task.status)
            task_record.status = status.value

            if progress is not None:
                task_record.progress = progress

            if result is not None:
                # 强制更新 result 字段，即使值看起来相同
                from sqlalchemy.orm.attributes import flag_modified

                task_record.result = self._to_json_safe(result)
                flag_modified(task, "result")  # 标记 result 字段为已修改

            if error_message is not None:
                task_record.error_message = error_message

            # 更新时间戳
            if status == TaskStatus.RUNNING and not cast(
                Optional[datetime], task.started_at
            ):
                task_record.started_at = datetime.utcnow()
            elif status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ]:
                task_record.completed_at = datetime.utcnow()

            self.db.commit()
            self.db.refresh(task)

            # 记录审计日志
            AuditLogger.log_data_change(
                table="tasks",
                operation="UPDATE",
                record_id=task_id,
                old_values={"status": old_status},
                new_values={"status": status.value},
                user_id=cast(Optional[str], task.user_id),
            )

            logger.info(f"任务状态更新: {task_id}, {old_status} -> {status.value}")
            return task

        except TaskError:
            raise
        except Exception as e:
            self.db.rollback()
            raise TaskError(
                message=f"更新任务状态失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(task_id=task_id),
                original_exception=e,
            )

    def update_task_progress(self, task_id: str, progress: float) -> "Task":
        """更新任务进度"""
        try:
            task = self.get_task_by_id(task_id)
            if not task:
                raise TaskError(
                    message=f"任务不存在: {task_id}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(task_id=task_id),
                )

            task_record = cast(Any, task)
            old_progress = cast(float, task.progress)
            task_record.progress = progress

            self.db.commit()
            self.db.refresh(task)

            logger.debug(f"任务进度更新: {task_id}, {old_progress:.1f}% -> {progress:.1f}%")
            return task

        except TaskError:
            raise
        except Exception as e:
            self.db.rollback()
            raise TaskError(
                message=f"更新任务进度失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(task_id=task_id),
                original_exception=e,
            )

    def delete_task(self, task_id: str, user_id: str, force: bool = False) -> bool:
        """删除任务"""
        try:
            task = self.get_task_by_id(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return False

            # 验证用户权限（强制模式下跳过权限检查）
            if not force and task.user_id != user_id:
                raise TaskError(
                    message="无权限删除此任务",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(task_id=task_id, user_id=user_id),
                )

            # 检测僵尸任务（状态是运行中但实际已中断）
            is_zombie_task = False
            if task.status == TaskStatus.RUNNING.value:
                # 如果任务开始时间超过一定时间，认为是僵尸任务
                now = datetime.utcnow()
                task_age = (
                    (now - task.created_at).total_seconds() / 3600
                    if task.created_at
                    else 0
                )
                # 使用started_at作为运行开始时间
                time_since_start = (
                    (now - task.started_at).total_seconds() / 3600
                    if task.started_at
                    else 0
                )

                # 判定条件：
                # 1. 任务创建超过3小时，或
                # 2. 任务开始运行超过1.5小时
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

            # 先删除相关的详细数据（如果有的话）
            # 使用同步方式删除，避免异步调用复杂性
            try:
                from sqlalchemy import delete as sql_delete

                from app.models.backtest_detailed_models import (
                    BacktestBenchmark,
                    BacktestDetailedResult,
                    PortfolioSnapshot,
                    TradeRecord,
                )

                # 删除各个详细数据表中的数据
                related_tables = [
                    (BacktestDetailedResult, "回测详细结果"),
                    (PortfolioSnapshot, "组合快照"),
                    (TradeRecord, "交易记录"),
                    (BacktestBenchmark, "基准数据"),
                ]

                total_deleted = 0
                for model_class, table_name in related_tables:
                    try:
                        model = cast(Any, model_class)
                        stmt = sql_delete(model_class).where(model.task_id == task_id)
                        result = self.db.execute(stmt)
                        deleted_count = cast(Any, result).rowcount or 0
                        if deleted_count > 0:
                            logger.info(f"删除{table_name}: {deleted_count}条记录")
                            total_deleted += deleted_count
                    except Exception as e:
                        # 表可能不存在，忽略错误
                        logger.debug(f"删除{table_name}时出错（可能表不存在）: {e}")

                if total_deleted > 0:
                    logger.info(f"已删除任务 {task_id} 的详细数据，共 {total_deleted} 条记录")
                    self.db.flush()  # 刷新但先不提交，等主任务删除一起提交
            except Exception as e:
                # 删除详细数据失败不影响主任务删除
                logger.warning(f"删除任务详细数据时出错（继续删除主任务）: {e}")

            # 删除主任务
            self.db.delete(task)
            self.db.commit()

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
            self.db.rollback()
            error_msg = str(e)
            # 检查是否是数据库约束错误
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

    def get_task_statistics(
        self, user_id: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """获取任务统计信息（使用 SQL 聚合，不加载完整 ORM 对象）"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            def _base_filter(q: Any) -> Any:
                q = q.filter(Task.created_at >= cutoff_date)
                if user_id:
                    q = q.filter(Task.user_id == user_id)
                return q

            # 按状态分组计数
            status_rows = (
                _base_filter(self.db.query(Task.status, func.count(Task.task_id)))
                .group_by(Task.status)
                .all()
            )
            status_counts = {status: count for status, count in status_rows}

            # 按类型分组计数
            type_rows = (
                _base_filter(self.db.query(Task.task_type, func.count(Task.task_id)))
                .group_by(Task.task_type)
                .all()
            )
            type_counts = {task_type: count for task_type, count in type_rows}

            total_tasks = sum(status_counts.values())

            # 计算平均执行时间（仅已完成的任务）
            avg_duration = 0
            completed_with_times = (
                _base_filter(self.db.query(Task.started_at, Task.completed_at))
                .filter(
                    Task.started_at.isnot(None),
                    Task.completed_at.isnot(None),
                )
                .all()
            )
            if completed_with_times:
                durations = [
                    (c.completed_at - c.started_at).total_seconds()
                    for c in completed_with_times
                ]
                avg_duration = sum(durations) / len(durations)

            return {
                "total_tasks": total_tasks,
                "status_counts": status_counts,
                "type_counts": type_counts,
                "avg_duration_seconds": avg_duration,
                "success_rate": (
                    status_counts.get(TaskStatus.COMPLETED.value, 0)
                    / max(total_tasks, 1)
                ),
                "period_days": days,
            }

        except Exception as e:
            raise TaskError(
                message=f"获取任务统计失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    def cleanup_old_tasks(self, days: int = 90) -> int:
        """清理旧任务"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # 只清理已完成或失败的任务
            deleted_count = (
                self.db.query(Task)
                .filter(
                    and_(
                        Task.completed_at < cutoff_date,
                        or_(
                            Task.status == TaskStatus.COMPLETED.value,
                            Task.status == TaskStatus.FAILED.value,
                        ),
                    )
                )
                .delete()
            )

            self.db.commit()

            logger.info(f"清理旧任务完成: 删除 {deleted_count} 个任务")
            return int(deleted_count)

        except Exception as e:
            self.db.rollback()
            raise TaskError(
                message=f"清理旧任务失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )


class PredictionResultRepository:
    """预测结果数据仓库"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def save_prediction_result(
        self,
        task_id: str,
        stock_code: str,
        prediction_date: datetime,
        predicted_price: float,
        predicted_direction: int,
        confidence_score: float,
        confidence_interval_lower: float,
        confidence_interval_upper: float,
        model_id: str,
        features_used: List[str],
        risk_metrics: Dict[str, Any],
    ) -> PredictionResult:
        """保存预测结果"""
        try:
            result = PredictionResult(
                task_id=task_id,
                stock_code=stock_code,
                prediction_date=prediction_date,
                predicted_price=predicted_price,
                predicted_direction=predicted_direction,
                confidence_score=confidence_score,
                confidence_interval_lower=confidence_interval_lower,
                confidence_interval_upper=confidence_interval_upper,
                model_id=model_id,
                features_used=features_used,
                risk_metrics=risk_metrics,
                created_at=datetime.utcnow(),
            )

            self.db.add(result)
            self.db.commit()
            self.db.refresh(result)

            logger.info(f"预测结果保存成功: {task_id}, {stock_code}")
            return result

        except Exception as e:
            self.db.rollback()
            raise TaskError(
                message=f"保存预测结果失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(task_id=task_id, stock_code=stock_code),
                original_exception=e,
            )

    def get_prediction_results_by_task(self, task_id: str) -> List[PredictionResult]:
        """获取任务的预测结果"""
        try:
            results = (
                self.db.query(PredictionResult)
                .filter(PredictionResult.task_id == task_id)
                .order_by(desc(PredictionResult.created_at))
                .all()
            )
            return cast(List[PredictionResult], results)
        except Exception as e:
            raise TaskError(
                message=f"获取预测结果失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(task_id=task_id),
                original_exception=e,
            )

    def get_prediction_results_by_stock(
        self, stock_code: str, limit: int = 100
    ) -> List[PredictionResult]:
        """获取股票的预测历史"""
        try:
            results = (
                self.db.query(PredictionResult)
                .filter(PredictionResult.stock_code == stock_code)
                .order_by(desc(PredictionResult.prediction_date))
                .limit(limit)
                .all()
            )
            return cast(List[PredictionResult], results)
        except Exception as e:
            raise TaskError(
                message=f"获取股票预测历史失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(stock_code=stock_code),
                original_exception=e,
            )


class BacktestResultRepository:
    """回测结果数据仓库"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def save_backtest_result(
        self,
        task_id: str,
        backtest_id: str,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        initial_cash: float,
        final_value: float,
        total_return: float,
        annualized_return: float,
        volatility: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        profit_factor: float,
        total_trades: int,
        trade_history: List[Dict[str, Any]],
    ) -> BacktestResult:
        """保存回测结果"""
        try:
            result = BacktestResult(
                task_id=task_id,
                backtest_id=backtest_id,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                final_value=final_value,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                trade_history=trade_history,
                created_at=datetime.utcnow(),
            )

            self.db.add(result)
            self.db.commit()
            self.db.refresh(result)

            logger.info(f"回测结果保存成功: {task_id}, {backtest_id}")
            return result

        except Exception as e:
            self.db.rollback()
            raise TaskError(
                message=f"保存回测结果失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(task_id=task_id),
                original_exception=e,
            )

    def get_backtest_results_by_task(self, task_id: str) -> List[BacktestResult]:
        """获取任务的回测结果"""
        try:
            results = (
                self.db.query(BacktestResult)
                .filter(BacktestResult.task_id == task_id)
                .order_by(desc(BacktestResult.created_at))
                .all()
            )
            return cast(List[BacktestResult], results)
        except Exception as e:
            raise TaskError(
                message=f"获取回测结果失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(task_id=task_id),
                original_exception=e,
            )


class ModelInfoRepository:
    """模型信息数据仓库"""

    def __init__(self, db_session: Session):
        self.db = db_session

    def save_model_info(
        self,
        model_id: str,
        model_name: str,
        model_type: str,
        version: str,
        file_path: str,
        training_data_start: datetime,
        training_data_end: datetime,
        performance_metrics: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        status: str = "training",
    ) -> ModelInfo:
        """保存模型信息"""
        try:
            model_info = ModelInfo(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version=version,
                file_path=file_path,
                training_data_start=training_data_start,
                training_data_end=training_data_end,
                performance_metrics=performance_metrics,
                hyperparameters=hyperparameters,
                status=status,
                created_at=datetime.utcnow(),
            )

            self.db.add(model_info)
            self.db.commit()
            self.db.refresh(model_info)

            logger.info(f"模型信息保存成功: {model_id}")
            return model_info

        except Exception as e:
            self.db.rollback()
            raise TaskError(
                message=f"保存模型信息失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(model_id=model_id),
                original_exception=e,
            )

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息。

        兼容三类查询：
        1. model_id 精确匹配
        2. model_name 精确匹配
        3. 短别名模糊匹配（如 bank-core3 -> hermes-bank-core3-2024-*）
        """
        try:
            direct_match = (
                self.db.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
            )
            if direct_match:
                return cast(ModelInfo, direct_match)

            exact_name_match = (
                self.db.query(ModelInfo)
                .filter(ModelInfo.model_name == model_id)
                .order_by(desc(ModelInfo.updated_at), desc(ModelInfo.created_at))
                .first()
            )
            if exact_name_match:
                return cast(ModelInfo, exact_name_match)

            normalized_alias = (model_id or "").strip()
            if not normalized_alias:
                return None

            alias_patterns = [
                f"%{normalized_alias}%",
                f"%-{normalized_alias}-%",
            ]
            alias_match = (
                self.db.query(ModelInfo)
                .filter(
                    or_(
                        ModelInfo.model_name.ilike(alias_patterns[0]),
                        ModelInfo.model_name.ilike(alias_patterns[1]),
                    )
                )
                .order_by(
                    desc(ModelInfo.status == "ready"),
                    desc(ModelInfo.updated_at),
                    desc(ModelInfo.created_at),
                )
                .first()
            )
            return cast(Optional[ModelInfo], alias_match)
        except Exception as e:
            raise TaskError(
                message=f"获取模型信息失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(model_id=model_id),
                original_exception=e,
            )

    def get_models_by_type(
        self, model_type: str, status: str = "ready"
    ) -> List[ModelInfo]:
        """根据类型获取模型列表"""
        try:
            models = (
                self.db.query(ModelInfo)
                .filter(
                    and_(ModelInfo.model_type == model_type, ModelInfo.status == status)
                )
                .order_by(desc(ModelInfo.created_at))
                .all()
            )
            return cast(List[ModelInfo], models)
        except Exception as e:
            raise TaskError(
                message=f"获取模型列表失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    def update_model_status(
        self, model_id: str, status: str, deployed_at: Optional[datetime] = None
    ) -> ModelInfo:
        """更新模型状态"""
        try:
            model_info = self.get_model_info(model_id)
            if not model_info:
                raise TaskError(
                    message=f"模型不存在: {model_id}",
                    severity=ErrorSeverity.MEDIUM,
                    context=ErrorContext(model_id=model_id),
                )

            model_record = cast(Any, model_info)
            model_record.status = status
            if deployed_at:
                model_record.deployed_at = deployed_at

            self.db.commit()
            self.db.refresh(model_info)

            logger.info(f"模型状态更新: {model_id}, 状态: {status}")
            return model_info

        except TaskError:
            raise
        except Exception as e:
            self.db.rollback()
            raise TaskError(
                message=f"更新模型状态失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(model_id=model_id),
                original_exception=e,
            )
