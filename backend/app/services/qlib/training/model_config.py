"""
模型配置模块

创建Qlib模型配置
"""

from typing import Any, Dict

from loguru import logger

from .config import QlibTrainingConfig


async def create_qlib_model_config(
    model_manager, config: QlibTrainingConfig
) -> Dict[str, Any]:
    """创建Qlib模型��置"""
    model_name = config.model_type.value

    # 使用模型管理器创建配置
    try:
        qlib_config = model_manager.create_qlib_config(
            model_name, config.hyperparameters
        )
        return qlib_config
    except Exception as e:
        logger.error(f"创建Qlib模型配置失败: {e}")
        raise
