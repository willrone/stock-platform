"""
API依赖注入和共享函数
"""

from datetime import datetime
import logging
from app.core.container import (
    get_data_service, 
    get_indicators_service, 
    get_parquet_manager,
    get_data_sync_engine,
    get_monitoring_service
)
from app.core.database import SessionLocal
from app.repositories.task_repository import TaskRepository, PredictionResultRepository, ModelInfoRepository
from app.models.task_models import TaskStatus
from app.services.tasks import TaskQueueManager

logger = logging.getLogger(__name__)

# 全局任务队列管理器实例
task_queue_manager = TaskQueueManager()

# 启动任务调度器（在模块加载时启动）
try:
    task_queue_manager.start_all_schedulers()
    logger.info("任务队列管理器已启动")
except Exception as e:
    logger.warning(f"任务队列管理器启动失败: {e}")


def get_task_repository():
    """获取任务仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return TaskRepository(session), session
    except:
        session.close()
        raise


def get_prediction_result_repository():
    """获取预测结果仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return PredictionResultRepository(session), session
    except:
        session.close()
        raise


def get_model_info_repository():
    """获取模型信息仓库（使用同步会话）"""
    session = SessionLocal()
    try:
        return ModelInfoRepository(session), session
    except:
        session.close()
        raise


# 简化的任务执行函数（用于后台任务）
def execute_prediction_task_simple(task_id: str):
    """简化的预测任务执行函数（后台任务）"""
    session = SessionLocal()
    try:
        task_repository = TaskRepository(session)
        prediction_result_repository = PredictionResultRepository(session)
        
        # 获取任务
        task = task_repository.get_task_by_id(task_id)
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return
        
        # 更新任务状态为运行中
        task_repository.update_task_status(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            progress=10.0
        )
        
        # 解析任务配置
        config = task.config or {}
        stock_codes = config.get('stock_codes', [])
        model_id = config.get('model_id', 'default_model')
        
        logger.info(f"开始执行预测任务: {task_id}, 股票数量: {len(stock_codes)}")
        
        # 模拟预测执行（这里可以替换为真实的预测逻辑）
        total_stocks = len(stock_codes)
        for i, stock_code in enumerate(stock_codes):
            try:
                # 更新进度
                progress = 10 + (i + 1) * 80 / total_stocks
                task_repository.update_task_status(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,  # 添加必需的status参数
                    progress=progress
                )
                
                # TODO: 这里应该调用真实的预测引擎
                # prediction_result = prediction_engine.predict(stock_code, model_id)
                
                # 模拟预测结果（临时实现）
                import random
                predicted_direction = random.choice([-1, 0, 1])
                confidence_score = random.uniform(0.6, 0.95)
                
                # 保存预测结果
                prediction_result_repository.save_prediction_result(
                    task_id=task_id,
                    stock_code=stock_code,
                    prediction_date=datetime.utcnow(),
                    predicted_price=100.0,  # 临时值
                    predicted_direction=predicted_direction,
                    confidence_score=confidence_score,
                    confidence_interval_lower=confidence_score - 0.1,
                    confidence_interval_upper=confidence_score + 0.1,
                    model_id=model_id,
                    features_used=[],
                    risk_metrics={}
                )
                
                logger.info(f"完成股票预测: {stock_code}, 方向: {predicted_direction}, 置信度: {confidence_score:.2f}")
                
            except Exception as e:
                logger.error(f"预测股票 {stock_code} 失败: {e}", exc_info=True)
                continue
        
        # 更新任务状态为完成
        task_repository.update_task_status(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0
        )
        
        logger.info(f"预测任务完成: {task_id}")
        
    except Exception as e:
        logger.error(f"执行预测任务失败: {task_id}, 错误: {e}", exc_info=True)
        try:
            task_repository.update_task_status(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )
        except:
            pass
    finally:
        session.close()

