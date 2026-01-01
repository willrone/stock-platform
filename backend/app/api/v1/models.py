"""
模型管理路由
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from sqlalchemy import or_
import logging
import uuid
from pathlib import Path
from typing import Optional

from app.api.v1.schemas import StandardResponse, ModelTrainingRequest
from app.core.database import SessionLocal
from app.repositories.task_repository import ModelInfoRepository
from app.models.task_models import ModelInfo
from app.websocket import (
    notify_model_training_progress,
    notify_model_training_completed,
    notify_model_training_failed
)
from app.services.models.hyperparameter_tuning import (
    HyperparameterTuner,
    SearchStrategy,
    HyperparameterSpace
)
from app.services.models.evaluation_report import EvaluationReportGenerator

router = APIRouter(prefix="/models", tags=["模型管理"])
logger = logging.getLogger(__name__)

# 导入模型训练服务
DEEP_TRAINING_AVAILABLE = False
ML_TRAINING_AVAILABLE = False
DeepModelTrainingService = None
DeepModelType = None
DeepTrainingConfig = None
MLModelTrainingService = None
MLModelType = None
MLTrainingConfig = None
ModelStorage = None

try:
    from app.services.models.model_training import (
        ModelTrainingService as DeepModelTrainingService,
        ModelType as DeepModelType,
        TrainingConfig as DeepTrainingConfig
    )
    DEEP_TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"深度学习训练服务导入失败: {e}")

try:
    from app.services.models.model_training_service import (
        ModelTrainingService as MLModelTrainingService,
        ModelType as MLModelType,
        TrainingConfig as MLTrainingConfig
    )
    from app.services.models.model_storage import ModelStorage
    ML_TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"传统ML训练服务导入失败: {e}")

TRAINING_AVAILABLE = DEEP_TRAINING_AVAILABLE or ML_TRAINING_AVAILABLE

# 全局训练服务实例（延迟初始化）
_deep_training_service: Optional[DeepModelTrainingService] = None
_ml_training_service: Optional[MLModelTrainingService] = None
_model_storage: Optional[ModelStorage] = None


def get_deep_training_service() -> DeepModelTrainingService:
    """获取深度学习训练服务实例"""
    global _deep_training_service
    if _deep_training_service is None and DEEP_TRAINING_AVAILABLE:
        _deep_training_service = DeepModelTrainingService()
    return _deep_training_service


def get_ml_training_service() -> MLModelTrainingService:
    """获取传统ML训练服务实例"""
    global _ml_training_service, _model_storage
    if _ml_training_service is None and ML_TRAINING_AVAILABLE:
        if _model_storage is None:
            _model_storage = ModelStorage()
        _ml_training_service = MLModelTrainingService(_model_storage)
    return _ml_training_service

def _format_feature_importance_for_report(feature_importance) -> list:
    """将特征重要性转换为报告格式"""
    if not feature_importance:
        return []
    
    if isinstance(feature_importance, dict):
        # 如果是字典格式 {feature_name: importance}
        return [
            {'name': name, 'importance': float(importance)}
            for name, importance in feature_importance.items()
        ]
    elif isinstance(feature_importance, list):
        # 如果已经是列表格式
        return feature_importance
    else:
        return []

def _normalize_accuracy(metrics: dict) -> float:
    """规范化准确率，确保不为负数"""
    if not isinstance(metrics, dict):
        return 0.0
    
    # 优先使用direction_accuracy（方向准确率）
    accuracy = metrics.get("direction_accuracy")
    if accuracy is not None:
        return max(0.0, min(1.0, float(accuracy)))
    
    # 其次使用accuracy
    accuracy = metrics.get("accuracy")
    if accuracy is not None:
        return max(0.0, min(1.0, float(accuracy)))
    
    # 最后使用r2，但如果是负数则设为0
    r2 = metrics.get("r2", 0.0)
    return max(0.0, float(r2))

def _normalize_performance_metrics_for_report(metrics: dict) -> dict:
    """规范化性能指标用于报告，确保accuracy不为负数"""
    if not isinstance(metrics, dict):
        return {
            'accuracy': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'r2': 0.0
        }
    
    normalized_accuracy = _normalize_accuracy(metrics)
    
    return {
        'accuracy': normalized_accuracy,
        'rmse': metrics.get('rmse', 0.0),
        'mae': metrics.get('mae', 0.0),
        'r2': metrics.get('r2', 0.0),
        'direction_accuracy': metrics.get('direction_accuracy')
    }


async def train_model_task(model_id: str, model_name: str, model_type: str,
                           stock_codes: list, start_date: datetime, end_date: datetime,
                           hyperparameters: dict, enable_hyperparameter_tuning: bool = False):
    """后台训练任务"""
    session = SessionLocal()
    report_generator = EvaluationReportGenerator()
    
    try:
        model_repository = ModelInfoRepository(session)
        model_info = model_repository.get_model_info(model_id)
        
        if not model_info:
            logger.error(f"模型不存在: {model_id}")
            await notify_model_training_failed(model_id, "模型不存在")
            return
        
        try:
            # 发送训练开始通知
            await notify_model_training_progress(model_id, 0.0, "initializing", "开始初始化训练")
            model_info.training_stage = "initializing"
            model_info.training_progress = 0.0
            session.commit()
            
            # 根据模型类型选择训练服务
            if model_type in ['transformer', 'lstm', 'timesnet', 'patchtst', 'informer']:
                # 使用深度学习训练服务
                if not DEEP_TRAINING_AVAILABLE:
                    raise ValueError("深度学习训练服务不可用")
                
                training_service = get_deep_training_service()
                await training_service.initialize()
                
                # 转换模型类型
                model_type_map = {
                    'transformer': DeepModelType.TRANSFORMER,
                    'lstm': DeepModelType.LSTM,
                    'timesnet': DeepModelType.TIMESNET,
                    'patchtst': DeepModelType.PATCHTST,
                    'informer': DeepModelType.INFORMER,
                }
                deep_model_type = model_type_map.get(model_type, DeepModelType.LSTM)
                
                # 超参数调优
                if enable_hyperparameter_tuning:
                    await notify_model_training_progress(model_id, 5.0, "hyperparameter_tuning", "开始超参数搜索")
                    model_info.training_stage = "hyperparameter_tuning"
                    model_info.training_progress = 5.0
                    session.commit()
                    
                    # 定义超参数搜索空间
                    param_space = {
                        'learning_rate': HyperparameterSpace(
                            name='learning_rate',
                            param_type='float',
                            min_value=0.0001,
                            max_value=0.01,
                            step=0.001
                        ),
                        'batch_size': HyperparameterSpace(
                            name='batch_size',
                            param_type='choice',
                            choices=[16, 32, 64, 128]
                        ),
                        'epochs': HyperparameterSpace(
                            name='epochs',
                            param_type='int',
                            min_value=50,
                            max_value=200,
                            step=50
                        )
                    }
                    
                    # 创建调优器
                    tuner = HyperparameterTuner(SearchStrategy.RANDOM_SEARCH)
                    
                    # 定义训练函数
                    async def train_with_params(params):
                        config = DeepTrainingConfig(
                            model_type=deep_model_type,
                            sequence_length=hyperparameters.get('sequence_length', 60),
                            prediction_horizon=hyperparameters.get('prediction_horizon', 5),
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            learning_rate=params['learning_rate'],
                            validation_split=hyperparameters.get('validation_split', 0.2),
                        )
                        _, metrics = await training_service.train_model(
                            model_id=f"{model_id}_trial",
                            stock_codes=stock_codes,
                            config=config,
                            start_date=start_date,
                            end_date=end_date
                        )
                        return {
                            'score': getattr(metrics, 'accuracy', 0.0),
                            'accuracy': getattr(metrics, 'accuracy', 0.0),
                            'sharpe_ratio': getattr(metrics, 'sharpe_ratio', 0.0)
                        }
                    
                    # 执行超参数搜索
                    best_trial = tuner.random_search(param_space, train_with_params, n_trials=10)
                    if best_trial:
                        hyperparameters.update(best_trial.hyperparameters)
                        await notify_model_training_progress(
                            model_id, 10.0, "hyperparameter_tuning",
                            f"超参数搜索完成，最佳得分: {best_trial.score:.4f}",
                            {"best_score": best_trial.score, "best_params": best_trial.hyperparameters}
                        )
                
                # 创建训练配置
                await notify_model_training_progress(model_id, 15.0, "preparing", "准备训练数据")
                model_info.training_stage = "preparing"
                model_info.training_progress = 15.0
                session.commit()
                
                config = DeepTrainingConfig(
                    model_type=deep_model_type,
                    sequence_length=hyperparameters.get('sequence_length', 60),
                    prediction_horizon=hyperparameters.get('prediction_horizon', 5),
                    batch_size=hyperparameters.get('batch_size', 32),
                    epochs=hyperparameters.get('epochs', 100),
                    learning_rate=hyperparameters.get('learning_rate', 0.001),
                    validation_split=hyperparameters.get('validation_split', 0.2),
                )
                
                # 训练模型（这里需要修改训练服务以支持进度回调）
                await notify_model_training_progress(model_id, 20.0, "training", "开始训练模型")
                model_info.training_stage = "training"
                model_info.training_progress = 20.0
                session.commit()
                
                model_file_path, metrics = await training_service.train_model(
                    model_id=model_id,
                    stock_codes=stock_codes,
                    config=config,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # 生成评估报告
                await notify_model_training_progress(model_id, 90.0, "evaluating", "生成评估报告")
                model_info.training_stage = "evaluating"
                model_info.training_progress = 90.0
                session.commit()
                
                report = report_generator.generate_report(
                    model_id=model_id,
                    model_name=model_name,
                    model_type=model_type,
                    version=model_info.version,
                    training_summary={
                        'duration': 0.0,  # TODO: 从训练服务获取
                        'total_samples': 0,
                        'train_samples': 0,
                        'validation_samples': 0,
                        'test_samples': 0,
                        'epochs': config.epochs,
                        'batch_size': config.batch_size,
                        'learning_rate': config.learning_rate
                    },
                    performance_metrics={
                        'accuracy': getattr(metrics, 'accuracy', 0.75),
                        'sharpe_ratio': getattr(metrics, 'sharpe_ratio', 0.0),
                        'total_return': getattr(metrics, 'total_return', 0.0),
                        'max_drawdown': getattr(metrics, 'max_drawdown', 0.0),
                    },
                    feature_importance=[],
                    training_history=[],
                    hyperparameters=hyperparameters,
                    training_data_info={
                        'stock_codes': stock_codes,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    }
                )
                
                # 更新模型信息
                model_info.status = "ready"
                model_info.file_path = model_file_path
                model_info.training_progress = 100.0
                model_info.training_stage = "completed"
                model_info.performance_metrics = {
                    "accuracy": getattr(metrics, 'accuracy', 0.75),
                    "sharpe_ratio": getattr(metrics, 'sharpe_ratio', 0.0),
                    "total_return": getattr(metrics, 'total_return', 0.0),
                    "max_drawdown": getattr(metrics, 'max_drawdown', 0.0),
                }
                model_info.evaluation_report = report_generator.to_dict(report)
                model_info.hyperparameters = hyperparameters
                session.commit()
                
                # 发送完成通知
                await notify_model_training_completed(model_id, model_info.performance_metrics)
                
            elif model_type in ['random_forest', 'linear_regression', 'xgboost', 'lightgbm']:
                # 使用传统ML训练服务
                if not ML_TRAINING_AVAILABLE:
                    raise ValueError("传统ML训练服务不可用")
                
                training_service = get_ml_training_service()
                
                # 超参数调优
                if enable_hyperparameter_tuning:
                    await notify_model_training_progress(model_id, 5.0, "hyperparameter_tuning", "开始超参数搜索")
                    model_info.training_stage = "hyperparameter_tuning"
                    model_info.training_progress = 5.0
                    session.commit()
                    
                    # 定义超参数搜索空间
                    param_space = {
                        'n_estimators': HyperparameterSpace(
                            name='n_estimators',
                            param_type='int',
                            min_value=50,
                            max_value=300,
                            step=50
                        ),
                        'max_depth': HyperparameterSpace(
                            name='max_depth',
                            param_type='int',
                            min_value=3,
                            max_value=20,
                            step=2
                        )
                    }
                    
                    tuner = HyperparameterTuner(SearchStrategy.RANDOM_SEARCH)
                    
                    def train_with_params(params):
                        config = MLTrainingConfig(
                            model_type=MLModelType.RANDOM_FOREST,
                            hyperparameters={**hyperparameters, **params},
                            validation_split=hyperparameters.get('validation_split', 0.2),
                        )
                        result = training_service.train_model(
                            model_name=f"{model_name}_trial",
                            model_type=MLModelType.RANDOM_FOREST,
                            stock_codes=stock_codes,
                            start_date=start_date,
                            end_date=end_date,
                            config=config,
                            created_by="system"
                        )
                        # 安全地获取test_metrics
                        test_metrics = result.test_metrics
                        if not isinstance(test_metrics, dict):
                            test_metrics = {}
                        
                        return {
                            'score': test_metrics.get("accuracy", test_metrics.get("r2", 0.0)),
                            'accuracy': test_metrics.get("accuracy", test_metrics.get("r2", 0.0)),
                            'rmse': test_metrics.get("rmse", 0.0)
                        }
                    
                    best_trial = tuner.random_search(param_space, train_with_params, n_trials=10)
                    if best_trial:
                        hyperparameters.update(best_trial.hyperparameters)
                        await notify_model_training_progress(
                            model_id, 10.0, "hyperparameter_tuning",
                            f"超参数搜索完成，最佳得分: {best_trial.score:.4f}",
                            {"best_score": best_trial.score, "best_params": best_trial.hyperparameters}
                        )
                
                await notify_model_training_progress(model_id, 15.0, "preparing", "准备训练数据")
                model_info.training_stage = "preparing"
                model_info.training_progress = 15.0
                session.commit()
                
                # 转换模型类型
                ml_model_type_map = {
                    'random_forest': MLModelType.RANDOM_FOREST,
                    'linear_regression': MLModelType.LINEAR_REGRESSION,
                }
                ml_model_type = ml_model_type_map.get(model_type, MLModelType.RANDOM_FOREST)
                
                # 创建训练配置
                config = MLTrainingConfig(
                    model_type=ml_model_type,
                    hyperparameters=hyperparameters,
                    validation_split=hyperparameters.get('validation_split', 0.2),
                )
                
                await notify_model_training_progress(model_id, 20.0, "training", "开始训练模型")
                model_info.training_stage = "training"
                model_info.training_progress = 20.0
                session.commit()
                
                # 训练模型
                result = training_service.train_model(
                    model_name=model_name,
                    model_type=ml_model_type,
                    stock_codes=stock_codes,
                    start_date=start_date,
                    end_date=end_date,
                    config=config,
                    created_by="system"
                )
                
                # 生成评估报告
                await notify_model_training_progress(model_id, 90.0, "evaluating", "生成评估报告")
                model_info.training_stage = "evaluating"
                model_info.training_progress = 90.0
                session.commit()
                
                report = report_generator.generate_report(
                    model_id=model_id,
                    model_name=model_name,
                    model_type=model_type,
                    version=model_info.version,
                    training_summary={
                        'duration': result.training_time,
                        'total_samples': 0,
                        'train_samples': 0,
                        'validation_samples': 0,
                        'test_samples': 0,
                        'epochs': 0,
                        'batch_size': 0,
                        'learning_rate': 0.0
                    },
                    performance_metrics=_normalize_performance_metrics_for_report(
                        result.test_metrics if isinstance(result.test_metrics, dict) else {}
                    ),
                    feature_importance=_format_feature_importance_for_report(result.feature_importance),
                    training_history=result.training_history or [],
                    hyperparameters=hyperparameters,
                    training_data_info={
                        'stock_codes': stock_codes,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    }
                )
                
                # 更新模型信息
                model_info.status = "ready"
                model_info.training_progress = 100.0
                model_info.training_stage = "completed"
                # 安全地获取test_metrics
                test_metrics = result.test_metrics
                if not isinstance(test_metrics, dict):
                    test_metrics = {}
                
                # 对于回归模型，优先使用direction_accuracy，如果没有则使用r2（但确保不为负）
                accuracy_value = test_metrics.get("accuracy")
                if accuracy_value is None:
                    # 尝试获取方向准确率
                    accuracy_value = test_metrics.get("direction_accuracy")
                if accuracy_value is None:
                    # 使用R²，但如果是负数则设为0
                    r2_value = test_metrics.get("r2", 0.0)
                    accuracy_value = max(0.0, r2_value)
                
                model_info.performance_metrics = {
                    "accuracy": float(accuracy_value),
                    "rmse": test_metrics.get("rmse", 0.0),
                    "mae": test_metrics.get("mae", 0.0),
                    "r2": test_metrics.get("r2", 0.0),
                    "direction_accuracy": test_metrics.get("direction_accuracy")
                }
                model_info.evaluation_report = report_generator.to_dict(report)
                model_info.hyperparameters = hyperparameters
                session.commit()
                
                await notify_model_training_completed(model_id, model_info.performance_metrics)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            session.commit()
            logger.info(f"模型训练完成: {model_id}")
            
        except Exception as e:
            logger.error(f"模型训练失败: {model_id}, 错误: {e}", exc_info=True)
            model_info.status = "failed"
            model_info.training_stage = "failed"
            # 将错误信息保存到performance_metrics中
            model_info.performance_metrics = {
                "error": str(e),
                "status": "failed"
            }
            session.commit()
            await notify_model_training_failed(model_id, str(e))
            
    except Exception as e:
        logger.error(f"训练任务执行失败: {e}", exc_info=True)
        session.rollback()
        await notify_model_training_failed(model_id, str(e))
    finally:
        session.close()


@router.get("/{model_id}/versions", response_model=StandardResponse)
async def get_model_versions(model_id: str):
    """获取模型的所有版本"""
    session = SessionLocal()
    try:
        # 获取主模型
        model = session.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
        
        # 获取所有版本（包括主模型本身）
        parent_id = model.parent_model_id or model_id
        versions = session.query(ModelInfo).filter(
            (ModelInfo.model_id == parent_id) | (ModelInfo.parent_model_id == parent_id)
        ).order_by(ModelInfo.created_at.desc()).all()
        
        version_list = []
        for v in versions:
            version_list.append({
                "model_id": v.model_id,
                "model_name": v.model_name,
                "version": v.version,
                "status": v.status,
                "accuracy": (v.performance_metrics or {}).get("accuracy", 0.0),
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "is_current": v.model_id == model_id
            })
        
        return StandardResponse(
            success=True,
            message="模型版本列表获取成功",
            data={"versions": version_list}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型版本失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型版本失败: {str(e)}")
    finally:
        session.close()


@router.get("/{model_id}/report", response_model=StandardResponse)
async def get_model_evaluation_report(model_id: str):
    """获取模型评估报告"""
    session = SessionLocal()
    try:
        model = session.query(ModelInfo).filter(ModelInfo.model_id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
        
        if not model.evaluation_report:
            raise HTTPException(status_code=404, detail="该模型尚未生成评估报告")
        
        return StandardResponse(
            success=True,
            message="评估报告获取成功",
            data=model.evaluation_report
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取评估报告失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取评估报告失败: {str(e)}")
    finally:
        session.close()


@router.get("", response_model=StandardResponse)
async def list_models():
    """获取模型列表"""
    session = SessionLocal()
    try:
        model_repository = ModelInfoRepository(session)
        
        # 获取所有模型（包括training、failed等状态）
        models = session.query(ModelInfo).order_by(ModelInfo.created_at.desc()).all()
        
        # 转换为前端期望的格式
        model_list = []
        for model in models:
            # 安全地获取performance_metrics
            performance_metrics = model.performance_metrics
            if isinstance(performance_metrics, str):
                try:
                    import json
                    performance_metrics = json.loads(performance_metrics)
                except:
                    performance_metrics = {}
            if not isinstance(performance_metrics, dict):
                performance_metrics = {}
            
            accuracy = performance_metrics.get("accuracy", 0.0)
            if isinstance(accuracy, dict):
                accuracy = accuracy.get("value", 0.0) if isinstance(accuracy, dict) else 0.0
            
            model_data = {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "model_type": model.model_type,
                "version": model.version,
                "accuracy": float(accuracy) if accuracy else 0.0,
                "created_at": model.created_at.isoformat() if model.created_at else datetime.now().isoformat(),
                "status": model.status,
                "training_progress": model.training_progress or 0.0,
                "training_stage": model.training_stage
            }
            model_list.append(model_data)
        
        return StandardResponse(
            success=True,
            message="模型列表获取成功",
            data={"models": model_list}
        )
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")
    finally:
        session.close()


@router.get("/{model_id}", response_model=StandardResponse)
async def get_model_detail(model_id: str):
    """获取模型详情"""
    session = SessionLocal()
    try:
        model_repository = ModelInfoRepository(session)
        model = model_repository.get_model_info(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
        
        # 转换为前端期望的格式
        performance_metrics = model.performance_metrics or {}
        if isinstance(performance_metrics, str):
            try:
                import json
                performance_metrics = json.loads(performance_metrics)
            except:
                performance_metrics = {}
        if not isinstance(performance_metrics, dict):
            performance_metrics = {}
        
        # 提取准确率（从performance_metrics或计算）
        accuracy = performance_metrics.get("accuracy", 0.0)
        if isinstance(accuracy, dict):
            accuracy = accuracy.get("value", 0.0) if isinstance(accuracy, dict) else 0.0
        
        training_data_period = {}
        if model.training_data_start and model.training_data_end:
            training_data_period = {
                "start": model.training_data_start.isoformat(),
                "end": model.training_data_end.isoformat()
            }
        
        # 从evaluation_report中提取stock_codes
        stock_codes = []
        if model.evaluation_report and isinstance(model.evaluation_report, dict):
            training_data_info = model.evaluation_report.get("training_data_info", {})
            if isinstance(training_data_info, dict):
                stock_codes = training_data_info.get("stock_codes", [])
        
        model_detail = {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "model_type": model.model_type,
            "version": model.version,
            "accuracy": float(accuracy) if accuracy else 0.0,
            "description": f"{model.model_type}模型 - {model.model_name}",
            "performance_metrics": performance_metrics,
            "training_info": {
                "training_data_period": training_data_period,
                "hyperparameters": model.hyperparameters or {},
                "stock_codes": stock_codes
            },
            "created_at": model.created_at.isoformat() if model.created_at else datetime.now().isoformat(),
            "status": model.status
        }
        
        return StandardResponse(
            success=True,
            message="模型详情获取成功",
            data=model_detail
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型详情失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取模型详情失败: {str(e)}")
    finally:
        session.close()


@router.delete("/{model_id}", response_model=StandardResponse)
async def delete_model(model_id: str):
    """删除模型"""
    session = SessionLocal()
    try:
        model_repository = ModelInfoRepository(session)
        model = model_repository.get_model_info(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")
        
        # 不能删除正在训练中的模型
        if model.status == "training":
            raise HTTPException(
                status_code=400,
                detail="无法删除正在训练中的模型，请等待训练完成或取消训练"
            )
        
        # 删除模型文件（如果存在）
        if model.file_path:
            try:
                model_file = Path(model.file_path)
                if model_file.exists():
                    model_file.unlink()
                    logger.info(f"已删除模型文件: {model.file_path}")
            except Exception as e:
                logger.warning(f"删除模型文件失败: {e}，继续删除数据库记录")
        
        # 删除数据库记录
        session.delete(model)
        session.commit()
        
        logger.info(f"模型删除成功: {model_id}")
        
        return StandardResponse(
            success=True,
            message="模型删除成功",
            data={"model_id": model_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"删除模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")
    finally:
        session.close()


@router.post("/train", response_model=StandardResponse)
async def create_training_task(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks
):
    """创建模型训练任务"""
    if not TRAINING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="模型训练服务不可用，请检查依赖安装"
        )
    
    session = SessionLocal()
    try:
        # 验证模型类型
        valid_model_types = [
            'random_forest', 'linear_regression', 'xgboost', 'lightgbm',
            'transformer', 'lstm', 'timesnet', 'patchtst', 'informer'
        ]
        if request.model_type not in valid_model_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型类型: {request.model_type}。支持的类型: {', '.join(valid_model_types)}"
            )
        
        # 生成模型ID
        model_id = str(uuid.uuid4())
        
        # 创建模型目录
        models_dir = Path("backend/data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_file_path = models_dir / f"{model_id}.pkl"
        
        # 解析日期
        try:
            start_date = datetime.fromisoformat(request.start_date)
            end_date = datetime.fromisoformat(request.end_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="日期格式错误，请使用 YYYY-MM-DD 格式"
            )
        
        # 创建模型记录
        model_info = ModelInfo(
            model_id=model_id,
            model_name=request.model_name,
            model_type=request.model_type,
            version="1.0.0",
            file_path=str(model_file_path),
            training_data_start=start_date,
            training_data_end=end_date,
            hyperparameters=request.hyperparameters or {},
            status="training",
            parent_model_id=request.parent_model_id,
            created_at=datetime.utcnow()
        )
        
        session.add(model_info)
        session.commit()
        
        logger.info(f"创建模型训练任务: {model_id}, 模型名称: {request.model_name}, 类型: {request.model_type}")
        
        # 启动后台训练任务
        background_tasks.add_task(
            train_model_task,
            model_id=model_id,
            model_name=request.model_name,
            model_type=request.model_type,
            stock_codes=request.stock_codes,
            start_date=start_date,
            end_date=end_date,
            hyperparameters=request.hyperparameters or {}
        )
        
        return StandardResponse(
            success=True,
            message="模型训练任务已创建，正在后台训练中",
            data={
                "model_id": model_id,
                "model_name": request.model_name,
                "model_type": request.model_type,
                "status": "training"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"创建模型训练任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建模型训练任务失败: {str(e)}")
    finally:
        session.close()

