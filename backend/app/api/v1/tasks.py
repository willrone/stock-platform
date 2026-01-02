"""
任务管理路由
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Optional
from datetime import datetime
from loguru import logger

from app.api.v1.schemas import StandardResponse, TaskCreateRequest
from app.core.database import SessionLocal
from app.repositories.task_repository import TaskRepository, PredictionResultRepository
from app.models.task_models import TaskStatus, TaskType
from app.api.v1.dependencies import execute_prediction_task_simple

router = APIRouter(prefix="/tasks", tags=["任务管理"])


@router.post("", response_model=StandardResponse)
async def create_task(
    request: TaskCreateRequest, 
    background_tasks: BackgroundTasks
):
    """创建任务（支持预测和回测）"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        
        # 确定任务类型
        task_type_str = request.task_type.lower() if request.task_type else "prediction"
        if task_type_str == "backtest":
            task_type = TaskType.BACKTEST
        else:
            task_type = TaskType.PREDICTION
        
        # 构建任务配置
        if task_type == TaskType.PREDICTION:
            if not request.model_id:
                raise HTTPException(status_code=400, detail="预测任务需要提供model_id")
            config = {
                "stock_codes": request.stock_codes,
                "model_id": request.model_id,
                **(request.prediction_config or {})
            }
        else:  # BACKTEST
            if not request.backtest_config:
                raise HTTPException(status_code=400, detail="回测任务需要提供backtest_config")
            config = {
                "stock_codes": request.stock_codes,
                **(request.backtest_config or {})
            }
        
        # 创建任务
        task = task_repository.create_task(
            task_name=request.task_name,
            task_type=task_type,
            user_id="default_user",  # TODO: 从认证中获取真实用户ID
            config=config
        )
        
        # 将任务加入后台执行
        try:
            if task_type == TaskType.PREDICTION:
                background_tasks.add_task(execute_prediction_task_simple, task.task_id)
            else:  # BACKTEST
                from app.api.v1.dependencies import execute_backtest_task_simple
                background_tasks.add_task(execute_backtest_task_simple, task.task_id)
            logger.info(f"任务已加入后台执行: {task.task_id}, 类型: {task_type.value}")
        except Exception as bg_error:
            logger.error(f"将任务加入后台执行时出错: {bg_error}", exc_info=True)
        
        # 转换为前端期望的格式
        task_data = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "stock_codes": request.stock_codes,
            "model_id": config.get("model_id", ""),
            "created_at": task.created_at.isoformat() if task.created_at else datetime.now().isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error_message": task.error_message
        }
        
        return StandardResponse(
            success=True,
            message="任务创建成功",
            data=task_data
        )
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"创建任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")
    finally:
        session.close()


@router.get("", response_model=StandardResponse)
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """获取任务列表"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        
        # 转换状态字符串为TaskStatus枚举
        status_filter = None
        if status:
            try:
                status_filter = TaskStatus(status)
            except ValueError:
                logger.warning(f"无效的任务状态: {status}")
        
        # 获取任务列表
        tasks = task_repository.get_tasks_by_user(
            user_id="default_user",
            limit=limit,
            offset=offset,
            status_filter=status_filter
        )
        
        # 获取总数
        total_tasks = task_repository.get_tasks_by_user(
            user_id="default_user",
            limit=10000,
            offset=0,
            status_filter=status_filter
        )
        total = len(total_tasks)
        
        # 转换为前端期望的格式
        task_list = []
        for task in tasks:
            config = task.config or {}
            stock_codes = config.get("stock_codes", [])
            model_id = config.get("model_id", "")
            
            task_data = {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "status": task.status,
                "progress": task.progress,
                "stock_codes": stock_codes if isinstance(stock_codes, list) else [],
                "model_id": model_id,
                "created_at": task.created_at.isoformat() if task.created_at else datetime.now().isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error_message": task.error_message
            }
            task_list.append(task_data)
        
        return StandardResponse(
            success=True,
            message="任务列表获取成功",
            data={
                "tasks": task_list,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        )
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")
    finally:
        session.close()


@router.get("/{task_id}", response_model=StandardResponse)
async def get_task_detail(task_id: str):
    """获取任务详情"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        prediction_result_repository = PredictionResultRepository(session)
        
        # 获取任务
        task = task_repository.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        
        # 获取任务配置
        config = task.config or {}
        stock_codes = config.get("stock_codes", [])
        model_id = config.get("model_id", "")
        
        # 获取预测结果
        prediction_results = prediction_result_repository.get_prediction_results_by_task(task_id)
        
        # 构建预测结果列表
        predictions = []
        total_confidence = 0.0
        for result in prediction_results:
            prediction = {
                "stock_code": result.stock_code,
                "predicted_direction": result.predicted_direction,
                "predicted_return": (result.predicted_price - 100) / 100 if result.predicted_price else 0,
                "confidence_score": result.confidence_score,
                "confidence_interval": {
                    "lower": result.confidence_interval_lower or 0,
                    "upper": result.confidence_interval_upper or 0
                },
                "risk_assessment": result.risk_metrics or {}
            }
            predictions.append(prediction)
            total_confidence += result.confidence_score
        
        # 计算平均置信度
        average_confidence = total_confidence / len(prediction_results) if prediction_results else 0.0
        
        # 获取回测结果（如果任务类型是回测，或者结果中包含回测数据）
        backtest_results = None
        if task.task_type == "backtest" or (task.result and isinstance(task.result, (dict, str))):
            logger.info(f"处理任务结果: task_id={task_id}, task_type={task.task_type}, result存在={task.result is not None}, result类型={type(task.result)}")
            if task.result:
                try:
                    import json
                    if isinstance(task.result, str):
                        parsed_result = json.loads(task.result)
                    else:
                        parsed_result = task.result
                    
                    # 检查是否包含回测相关的字段
                    is_backtest_data = False
                    if isinstance(parsed_result, dict):
                        # 检查是否包含回测相关的关键字段
                        backtest_keys = ['equity_curve', 'drawdown_curve', 'portfolio', 'risk_metrics', 'trade_history', 'dates']
                        is_backtest_data = any(key in parsed_result for key in backtest_keys)
                    
                    # 如果是回测任务，或者结果中包含回测数据，则使用该结果
                    if task.task_type == "backtest" or is_backtest_data:
                        backtest_results = parsed_result
                        logger.info(f"回测结果解析成功: task_id={task_id}, 包含字段={list(backtest_results.keys())[:20] if isinstance(backtest_results, dict) else '非字典类型'}")
                        if isinstance(backtest_results, dict):
                            logger.info(f"回测结果关键字段: equity_curve={len(backtest_results.get('equity_curve', []))}, "
                                       f"portfolio={backtest_results.get('portfolio') is not None}, "
                                       f"risk_metrics={backtest_results.get('risk_metrics') is not None}")
                    else:
                        logger.debug(f"任务结果不包含回测数据: task_id={task_id}")
                except Exception as e:
                    logger.warning(f"解析回测结果失败: {e}", exc_info=True)
            else:
                if task.task_type == "backtest":
                    logger.warning(f"回测任务但无结果数据: task_id={task_id}, result={task.result}")
        
        # 构建任务详情
        task_detail = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "stock_codes": stock_codes if isinstance(stock_codes, list) else [],
            "model_id": model_id,
            "created_at": task.created_at.isoformat() if task.created_at else datetime.now().isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error_message": task.error_message,
            "results": {
                "total_stocks": len(stock_codes) if isinstance(stock_codes, list) else 0,
                "successful_predictions": len(prediction_results),
                "average_confidence": average_confidence,
                "predictions": predictions,
                "backtest_results": backtest_results  # 添加回测结果
            },
            # 如果有回测结果，直接将回测结果放在顶层，方便前端访问
            "backtest_results": backtest_results if backtest_results is not None else None
        }
        
        # 添加调试日志
        if backtest_results is not None:
            logger.info(f"回测结果详情返回: task_id={task_id}, task_type={task.task_type}, backtest_results存在={backtest_results is not None}, "
                       f"results.backtest_results存在={backtest_results is not None}")
            if backtest_results:
                logger.info(f"回测结果包含字段: {list(backtest_results.keys())[:20] if isinstance(backtest_results, dict) else '非字典'}")
        
        return StandardResponse(
            success=True,
            message="任务详情获取成功",
            data=task_detail
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务详情失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")
    finally:
        session.close()


@router.delete("/{task_id}", response_model=StandardResponse)
async def delete_task(task_id: str):
    """删除任务"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        success = task_repository.delete_task(
            task_id=task_id,
            user_id="default_user"
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"任务不存在或无法删除: {task_id}")
        
        return StandardResponse(
            success=True,
            message="任务删除成功",
            data={"task_id": task_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")
    finally:
        session.close()


@router.post("/{task_id}/stop", response_model=StandardResponse)
async def stop_task(task_id: str):
    """停止运行中的任务"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        
        if task.status not in [TaskStatus.RUNNING.value, TaskStatus.QUEUED.value]:
            raise HTTPException(status_code=400, detail=f"任务状态为 {task.status}，无法停止")
        
        # 更新任务状态为已取消
        task = task_repository.update_task_status(
            task_id=task_id,
            status=TaskStatus.CANCELLED
        )
        
        task_data = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "status": task.status,
            "progress": task.progress
        }
        
        return StandardResponse(
            success=True,
            message="任务已停止",
            data=task_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"停止任务失败: {str(e)}")
    finally:
        session.close()


@router.post("/{task_id}/retry", response_model=StandardResponse)
async def retry_task(
    task_id: str,
    background_tasks: BackgroundTasks
):
    """重新运行失败的任务"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        task = task_repository.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        
        if task.status not in [TaskStatus.FAILED.value, TaskStatus.CANCELLED.value]:
            raise HTTPException(status_code=400, detail=f"任务状态为 {task.status}，无法重试")
        
        # 重置任务状态
        task = task_repository.update_task_status(
            task_id=task_id,
            status=TaskStatus.CREATED,
            progress=0.0
        )
        
        # 添加后台任务来执行预测
        background_tasks.add_task(execute_prediction_task_simple, task_id)
        
        task_data = {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "status": task.status,
            "progress": task.progress
        }
        
        return StandardResponse(
            success=True,
            message="任务已重新创建",
            data=task_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重试任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"重试任务失败: {str(e)}")
    finally:
        session.close()


@router.get("/stats", response_model=StandardResponse)
async def get_task_stats():
    """获取任务统计信息"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        stats = task_repository.get_task_statistics(
            user_id="default_user",
            days=30
        )
        
        # 转换为前端期望的格式
        status_counts = stats.get("status_counts", {})
        task_stats = {
            "total": stats.get("total_tasks", 0),
            "completed": status_counts.get(TaskStatus.COMPLETED.value, 0),
            "running": status_counts.get(TaskStatus.RUNNING.value, 0) + status_counts.get(TaskStatus.QUEUED.value, 0),
            "failed": status_counts.get(TaskStatus.FAILED.value, 0),
            "success_rate": stats.get("success_rate", 0.0)
        }
        
        return StandardResponse(
            success=True,
            message="任务统计获取成功",
            data=task_stats
        )
        
    except Exception as e:
        logger.error(f"获取任务统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取任务统计失败: {str(e)}")
    finally:
        session.close()

