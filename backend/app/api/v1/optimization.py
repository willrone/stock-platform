"""
超参优化任务 API
"""

import asyncio
import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from app.api.v1.schemas import HyperparameterOptimizationRequest, StandardResponse
from app.core.database import SessionLocal
from app.models.task_models import TaskStatus, TaskType
from app.repositories.task_repository import TaskRepository
from app.services.tasks.process_executor import get_process_executor
from app.services.tasks.task_execution_engine import (
    HyperparameterOptimizationTaskExecutor,
)
from app.services.tasks.task_queue import QueuedTask, TaskExecutionContext, TaskPriority

router = APIRouter(prefix="/optimization", tags=["超参优化"])


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/tasks", response_model=StandardResponse)
async def create_optimization_task(
    request: HyperparameterOptimizationRequest, db: Session = Depends(get_db)
):
    """创建超参优化任务"""
    try:
        task_repository = TaskRepository(db)

        # 构建任务配置
        config = {
            "stock_codes": request.stock_codes,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "optimization_config": {
                "strategy_name": request.strategy_name,
                "param_space": {
                    name: {
                        "type": space.type,
                        "low": space.low,
                        "high": space.high,
                        "choices": space.choices,
                        "default": space.default,
                        "enabled": space.enabled,
                        "log": space.log,
                    }
                    for name, space in request.param_space.items()
                },
                "objective_config": {
                    "objective_metric": request.objective_config.objective_metric,
                    "direction": request.objective_config.direction,
                    "objective_weights": request.objective_config.objective_weights,
                },
                "n_trials": request.n_trials,
                "optimization_method": request.optimization_method,
                "timeout": request.timeout,
            },
            "backtest_config": request.backtest_config or {},
        }

        # 创建任务
        task = task_repository.create_task(
            task_name=request.task_name,
            task_type=TaskType.HYPERPARAMETER_OPTIMIZATION,
            user_id="default_user",  # TODO: 从认证中获取真实用户ID
            config=config,
        )

        # 将任务提交到进程池执行
        try:
            process_executor = get_process_executor()

            # 使用进程池提交任务
            future = process_executor.submit(
                execute_optimization_task_simple, task.task_id
            )

            logger.info(f"超参优化任务已提交到进程池: {task.task_id}")
        except Exception as submit_error:
            logger.error(f"将任务提交到进程池时出错: {submit_error}", exc_info=True)
            task_repository.update_task_status(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error_message=f"任务提交失败: {str(submit_error)}",
            )

        # 转换为前端期望的格式
        task_data = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "stock_codes": request.stock_codes,
            "created_at": task.created_at.isoformat()
            if task.created_at
            else datetime.now().isoformat(),
            "error_message": task.error_message,
        }

        return StandardResponse(success=True, message="超参优化任务创建成功", data=task_data)

    except Exception as e:
        logger.error(f"创建超参优化任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建超参优化任务失败: {str(e)}")
    finally:
        db.close()


@router.get("/tasks", response_model=StandardResponse)
async def list_optimization_tasks(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """获取超参优化任务列表"""
    try:
        task_repository = TaskRepository(db)

        # 筛选优化任务
        tasks = task_repository.get_tasks_by_user(
            user_id="default_user",  # TODO: 从认证中获取
            limit=limit,
            offset=offset,
            status_filter=TaskStatus[status.upper()] if status else None,
        )

        # 过滤出优化任务
        optimization_tasks = [
            t
            for t in tasks
            if t.task_type == TaskType.HYPERPARAMETER_OPTIMIZATION.value
        ]

        # 转换为前端格式
        task_list = []
        for task in optimization_tasks:
            config = task.config or {}
            optimization_config = config.get("optimization_config", {})

            task_data = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "task_type": task.task_type,
                "status": task.status,
                "progress": task.progress,
                "strategy_name": optimization_config.get("strategy_name", ""),
                "n_trials": optimization_config.get("n_trials", 0),
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": task.completed_at.isoformat()
                if task.completed_at
                else None,
                "error_message": task.error_message,
            }

            # 如果有结果，添加最佳得分
            if task.result:
                result = task.result
                task_data["best_score"] = result.get("best_score")
                task_data["best_trial_number"] = result.get("best_trial_number")

            task_list.append(task_data)

        return StandardResponse(
            success=True,
            message="获取超参优化任务列表成功",
            data={"tasks": task_list, "total": len(task_list)},
        )

    except Exception as e:
        logger.error(f"获取超参优化任务列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取超参优化任务列表失败: {str(e)}")
    finally:
        db.close()


@router.get("/tasks/{task_id}", response_model=StandardResponse)
async def get_optimization_task(task_id: str, db: Session = Depends(get_db)):
    """获取超参优化任务详情"""
    try:
        task_repository = TaskRepository(db)
        task = task_repository.get_task_by_id(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        if task.task_type != TaskType.HYPERPARAMETER_OPTIMIZATION.value:
            raise HTTPException(status_code=400, detail="该任务不是超参优化任务")

        config = task.config or {}
        optimization_config = config.get("optimization_config", {})

        task_data = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "strategy_name": optimization_config.get("strategy_name", ""),
            "stock_codes": config.get("stock_codes", []),
            "start_date": config.get("start_date"),
            "end_date": config.get("end_date"),
            "n_trials": optimization_config.get("n_trials", 0),
            "optimization_method": optimization_config.get(
                "optimization_method", "tpe"
            ),
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat()
            if task.completed_at
            else None,
            "error_message": task.error_message,
            "result": task.result,
        }

        return StandardResponse(success=True, message="获取超参优化任务详情成功", data=task_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取超参优化任务详情失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取超参优化任务详情失败: {str(e)}")
    finally:
        db.close()


@router.get("/tasks/{task_id}/status", response_model=StandardResponse)
async def get_optimization_status(task_id: str, db: Session = Depends(get_db)):
    """获取优化任务实时状态"""
    try:
        task_repository = TaskRepository(db)
        task = task_repository.get_task_by_id(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        result = task.result or {}
        config = task.config or {}
        optimization_config = config.get("optimization_config", {})

        # 从 result 或 config 中获取 n_trials
        n_trials = result.get("n_trials") or optimization_config.get("n_trials", 0)

        status_data = {
            "task_id": task_id,
            "status": task.status,
            "progress": task.progress or 0.0,
            "n_trials": n_trials,
            "completed_trials": result.get("completed_trials", 0),
            "running_trials": result.get("running_trials", 0),
            "pruned_trials": result.get("pruned_trials", 0),
            "failed_trials": result.get("failed_trials", 0),
            "best_score": result.get("best_score"),
            "best_trial_number": result.get("best_trial_number"),
            "best_params": result.get("best_params"),
        }

        logger.debug(
            f"优化任务状态: task_id={task_id}, status={task.status}, progress={task.progress}, "
            f"n_trials={n_trials}, completed={status_data['completed_trials']}, "
            f"running={status_data['running_trials']}, result_keys={list(result.keys())}"
        )

        return StandardResponse(success=True, message="获取优化状态成功", data=status_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取优化状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取优化状态失败: {str(e)}")
    finally:
        db.close()


@router.get("/tasks/{task_id}/param-importance", response_model=StandardResponse)
async def get_param_importance(task_id: str, db: Session = Depends(get_db)):
    """获取参数重要性"""
    try:
        task_repository = TaskRepository(db)
        task = task_repository.get_task_by_id(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        if task.status != TaskStatus.COMPLETED.value:
            raise HTTPException(status_code=400, detail="任务尚未完成")

        result = task.result or {}
        param_importance = result.get("param_importance", {})

        return StandardResponse(
            success=True, message="获取参数重要性成功", data=param_importance
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取参数重要性失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取参数重要性失败: {str(e)}")
    finally:
        db.close()


@router.get("/tasks/{task_id}/pareto-front", response_model=StandardResponse)
async def get_pareto_front(task_id: str, db: Session = Depends(get_db)):
    """获取帕累托前沿（多目标优化时）"""
    try:
        task_repository = TaskRepository(db)
        task = task_repository.get_task_by_id(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")

        if task.status != TaskStatus.COMPLETED.value:
            raise HTTPException(status_code=400, detail="任务尚未完成")

        result = task.result or {}
        pareto_front = result.get("pareto_front", [])

        if not pareto_front:
            raise HTTPException(status_code=400, detail="该任务不是多目标优化任务")

        return StandardResponse(success=True, message="获取帕累托前沿成功", data=pareto_front)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取帕累托前沿失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取帕累托前沿失败: {str(e)}")
    finally:
        db.close()


def execute_optimization_task_simple(task_id: str):
    """
    简化的超参优化任务执行函数（进程池执行）
    """
    import asyncio
    import os
    from datetime import datetime

    from loguru import logger

    process_id = os.getpid()
    task_logger = logger.bind(process_id=process_id, task_id=task_id, log_type="task")

    # 在进程池中需要重新导入，因为进程池会创建新的进程
    from app.core.database import SessionLocal
    from app.models.task_models import TaskStatus, TaskType
    from app.repositories.task_repository import TaskRepository
    from app.services.tasks.task_execution_engine import (
        HyperparameterOptimizationTaskExecutor,
    )
    from app.services.tasks.task_queue import (
        QueuedTask,
        TaskExecutionContext,
        TaskPriority,
    )

    session = SessionLocal()
    task_logger.info(f"开始执行超参优化任务: {task_id}, 进程ID: {process_id}")

    try:
        task_repository = TaskRepository(session)

        task = task_repository.get_task_by_id(task_id)
        if not task:
            task_logger.error(f"任务不存在: {task_id}")
            return

        # 更新任务状态为运行中
        task_repository.update_task_status(
            task_id=task_id, status=TaskStatus.RUNNING, progress=10.0
        )

        # 创建执行器
        executor = HyperparameterOptimizationTaskExecutor(task_repository)

        # 创建 QueuedTask
        queued_task = QueuedTask(
            task_id=task.task_id,
            task_type=TaskType.HYPERPARAMETER_OPTIMIZATION,
            user_id=task.user_id or "default_user",
            config=task.config or {},
            priority=TaskPriority.NORMAL,
            created_at=task.created_at if task.created_at else datetime.now(),
        )

        # 创建执行上下文
        context = TaskExecutionContext(
            task_id=task_id,
            executor_id=f"optimization_executor_{os.getpid()}",
            start_time=datetime.now(),
            progress_callback=lambda progress, message: task_repository.update_task_status(
                task_id, TaskStatus.RUNNING, progress=progress
            ),
            cancel_event=None,
        )

        # 执行任务
        result = executor.execute(queued_task, context)

        task_logger.info(f"超参优化任务执行完成: {task_id}")

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        task_logger.error(
            f"超参优化任务执行失败: {task_id}, 错误类型: {type(e).__name__}, 错误: {e}", exc_info=True
        )
        task_logger.error(f"详细错误信息: {error_details}")

        try:
            if "task_repository" in locals():
                task_repository.update_task_status(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"{type(e).__name__}: {str(e)}",
                )
        except Exception as update_error:
            task_logger.error(f"更新任务状态失败: {update_error}", exc_info=True)
    finally:
        if "session" in locals():
            session.close()
