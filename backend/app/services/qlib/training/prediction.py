"""
预测模块

包含模型预测和特征对齐功能
"""

from typing import Any, Dict, List

from loguru import logger


def get_supported_model_types(self) -> List[str]:
    """获取支持的模型类型列表"""
    return self.model_manager.get_supported_models()


def get_model_config_template(self, model_type: str) -> Dict[str, Any]:
    """获取模型配置模板"""
    try:
        metadata = self.model_manager.get_model_metadata(model_type)
        hyperparameter_specs = self.model_manager.get_hyperparameter_specs(model_type)

        if not metadata:
            return {}

        template = {
            "model_info": metadata.to_dict(),
            "hyperparameters": {
                spec.name: spec.default_value for spec in hyperparameter_specs
            },
        }
        return template
    except Exception as e:
        logger.error(f"获取模型配置模板失败: {e}")
        return {}


def recommend_models(
    self, sample_count: int, feature_count: int, task_type: str = "regression"
) -> List[str]:
    """推荐适合的模型"""
    return self.model_manager.recommend_models(sample_count, feature_count, task_type)


def get_training_recommendations(self, model_type: str) -> Dict[str, Any]:
    """获取训练建议"""
    return self.model_manager.get_training_recommendations(model_type)
