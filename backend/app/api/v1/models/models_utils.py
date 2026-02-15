"""
模型管理共享工具和配置
包含所有模块共享的导入、全局变量和辅助函数
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from sqlalchemy import or_

from app.api.v1.schemas import ModelTrainingRequest, StandardResponse
from app.core.database import SessionLocal
from app.models.task_models import ModelInfo
from app.repositories.task_repository import ModelInfoRepository
from app.services.models.evaluation_report import EvaluationReportGenerator
from app.services.models.hyperparameter_tuning import (
    HyperparameterSpace,
    SearchStrategy,
)
from app.services.models.lineage_tracker import lineage_tracker
from app.services.models.model_lifecycle_manager import model_lifecycle_manager
from app.websocket import (
    notify_model_training_completed,
    notify_model_training_failed,
    notify_model_training_progress,
)

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
    )
    from app.services.models.model_training import ModelType as DeepModelType
    from app.services.models.model_training import TrainingConfig as DeepTrainingConfig

    DEEP_TRAINING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"深度学习训练服务导入失败: {e}")

try:
    from app.services.models.model_storage import ModelStorage
    from app.services.models.model_training_service import (
        ModelTrainingService as MLModelTrainingService,
    )
    from app.services.models.model_training_service import ModelType as MLModelType
    from app.services.models.model_training_service import (
        TrainingConfig as MLTrainingConfig,
    )

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
            {"name": name, "importance": float(importance)}
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
    """规范化性能指标用于报告，确保accuracy不为负数，保留所有指标"""
    if not isinstance(metrics, dict):
        return {"accuracy": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}

    normalized_accuracy = _normalize_accuracy(metrics)

    # 保留所有传入的指标，数值类指标 None→0.0 防止下游比较报错
    def _safe_float(val, default=0.0):
        """安全转换为 float，None/NaN 返回 default"""
        if val is None:
            return default
        try:
            f = float(val)
            import math
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    normalized_metrics = {
        "accuracy": normalized_accuracy,
        "rmse": _safe_float(metrics.get("rmse")) or (
            _safe_float(metrics.get("mse")) ** 0.5 if metrics.get("mse") else 0.0
        ),
        "mae": _safe_float(metrics.get("mae")),
        "r2": _safe_float(metrics.get("r2")),
        "mse": _safe_float(metrics.get("mse")),
        # 分类指标 - None→0.0
        "precision": _safe_float(metrics.get("precision")),
        "recall": _safe_float(metrics.get("recall")),
        "f1_score": _safe_float(metrics.get("f1_score")),
        # 金融指标 - 保留 None（下游已做 None ���查）
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "win_rate": metrics.get("win_rate"),
        # 其他指标
        "direction_accuracy": metrics.get("direction_accuracy"),
        "information_ratio": metrics.get("information_ratio"),
        "calmar_ratio": metrics.get("calmar_ratio"),
        "profit_factor": metrics.get("profit_factor"),
        "volatility": metrics.get("volatility"),
        # RankIC 指标
        "rank_ic": _safe_float(metrics.get("rank_ic")),
        "rank_ic_ir": _safe_float(metrics.get("rank_ic_ir")),
    }

    return normalized_metrics


# 创建线程池执行器（单例）
_train_executor = None


def get_train_executor():
    """获取训练任务线程池执行器"""
    global _train_executor
    if _train_executor is None:
        _train_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="train_task"
        )
    return _train_executor
