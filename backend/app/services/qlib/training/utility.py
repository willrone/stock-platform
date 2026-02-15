"""
工具函数模块

包含通用工具函数
"""

from typing import Any, Dict, List

from .config import QlibModelType


def get_supported_model_types() -> List[str]:
    """获取支持的模型类型列表"""
    return [model_type.value for model_type in QlibModelType]


def get_model_config_template(model_type: str) -> Dict[str, Any]:
    """获取模型配置模板"""
    # 这里可以返回不同模型类型的默认配置
    templates = {
        "lightgbm": {
            "learning_rate": 0.2,
            "num_leaves": 210,
            "n_estimators": 1000,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
        },
        "xgboost": {
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
        },
        "mlp": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
        },
    }
    return templates.get(model_type, {})


def recommend_models(data_size: int, feature_count: int) -> List[str]:
    """根据数据规模推荐模型"""
    if data_size < 1000:
        return ["linear", "lightgbm"]
    elif data_size < 10000:
        return ["lightgbm", "xgboost", "mlp"]
    else:
        return ["lightgbm", "xgboost", "mlp", "transformer"]


def get_training_recommendations(model_type: str) -> Dict[str, Any]:
    """获取训练建议"""
    recommendations = {
        "lightgbm": {
            "batch_size": None,
            "epochs": None,
            "early_stopping": True,
            "validation_split": 0.2,
        },
        "mlp": {
            "batch_size": 256,
            "epochs": 100,
            "early_stopping": True,
            "validation_split": 0.2,
        },
    }
    return recommendations.get(model_type, {})
