"""
任务恢复服务
在应用启动时自动恢复中断的任务
"""

import logging
from typing import List

from app.core.database import SessionLocal
from app.models.task_models import Task
from app.services.tasks.process_executor import get_process_executor

logger = logging.getLogger(__name__)


class TaskRecoveryService:
    """任务恢复服务"""

    def __init__(self):
        self.recovered_tasks: List[str] = []

    def recover_interrupted_tasks(self) -> dict:
        """
        恢复中断的任务

        在应用启动时调用，将 running/queued 状态的任务重新提交到进程池

        Returns:
            恢复结果统计
        """
        session = SessionLocal()
        try:
            # 查询所有 running 或 queued 状态的任务
            interrupted_tasks = (
                session.query(Task).filter(Task.status.in_(["running", "queued"])).all()
            )

            if not interrupted_tasks:
                logger.info("没有需要恢复的任务")
                return {"total": 0, "recovered": 0, "failed": 0, "tasks": []}

            logger.info(f"发现 {len(interrupted_tasks)} 个中断的任务，开始恢复...")

            result = {
                "total": len(interrupted_tasks),
                "recovered": 0,
                "failed": 0,
                "tasks": [],
            }

            process_executor = get_process_executor()

            for task in interrupted_tasks:
                try:
                    # 将任务标记为 pending，准备重新执行
                    task.status = "pending"
                    task.started_at = None
                    session.commit()

                    # 根据任务类型提交到进程池
                    if task.task_type == "hyperparameter_optimization":
                        from app.api.v1.optimization import (
                            execute_optimization_task_simple,
                        )

                        future = process_executor.submit(
                            execute_optimization_task_simple, task.task_id
                        )
                    elif task.task_type == "backtest":
                        from app.api.v1.tasks import execute_backtest_task_simple

                        future = process_executor.submit(
                            execute_backtest_task_simple, task.task_id
                        )
                    elif task.task_type == "prediction":
                        from app.api.v1.tasks import execute_prediction_task_simple

                        future = process_executor.submit(
                            execute_prediction_task_simple, task.task_id
                        )
                    else:
                        logger.warning(
                            f"不支持的任务类型: {task.task_type}, 跳过任务 {task.task_id}"
                        )
                        result["failed"] += 1
                        continue

                    self.recovered_tasks.append(task.task_id)
                    result["recovered"] += 1
                    result["tasks"].append(
                        {
                            "task_id": task.task_id,
                            "task_name": task.task_name,
                            "task_type": task.task_type,
                            "status": "recovered",
                        }
                    )

                    logger.info(f"✅ 恢复任务: {task.task_id} - {task.task_name}")

                except Exception as e:
                    logger.error(f"❌ 恢复任务失败: {task.task_id}, 错误: {e}")
                    result["failed"] += 1
                    result["tasks"].append(
                        {
                            "task_id": task.task_id,
                            "task_name": task.task_name,
                            "task_type": task.task_type,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

            logger.info(
                f"任务恢复完成: 总计 {result['total']}, "
                f"成功 {result['recovered']}, 失败 {result['failed']}"
            )

            return result

        except Exception as e:
            logger.error(f"任务恢复服务失败: {e}", exc_info=True)
            return {"total": 0, "recovered": 0, "failed": 0, "error": str(e)}
        finally:
            session.close()

    def get_recovered_tasks(self) -> List[str]:
        """获取已恢复的任务列表"""
        return self.recovered_tasks.copy()


# 全局任务恢复服务实例
task_recovery_service = TaskRecoveryService()
