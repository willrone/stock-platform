"""
Qlib 模型预测器

提供使用 Qlib 模型进行预测的功能。
"""

from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

# 检测 Qlib 可用性
try:
    from qlib.utils import init_instance_by_config

    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    init_instance_by_config = None


class QlibModelPredictor:
    """Qlib 模型预测器

    使用 Qlib 模型进行训练和预测。

    Attributes:
        model: Qlib 模型实例
        model_config: 模型配置
        is_fitted: 模型是否已训练
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """初始化模型预测器

        Args:
            model_config: Qlib 模型配置字典
        """
        self.model_config = model_config
        self.model = None
        self.is_fitted = False

        if not QLIB_AVAILABLE:
            logger.warning("Qlib 不可用，模型预测功能将受限")

    def fit(self, dataset: pd.DataFrame) -> "QlibModelPredictor":
        """训练模型

        Args:
            dataset: 训练数据集

        Returns:
            self

        Raises:
            RuntimeError: 当 Qlib 不可用或模型配置未设置时
        """
        if not QLIB_AVAILABLE:
            raise RuntimeError("Qlib 不可用，无法训练模型")

        if self.model_config is None:
            raise RuntimeError("模型配置未设置，请先调用 set_config()")

        try:
            logger.info("开始训练 Qlib 模型...")

            # 创建模型实例
            self.model = init_instance_by_config(self.model_config)

            # 训练模型
            self.model.fit(dataset)
            self.is_fitted = True

            logger.info("Qlib 模型训练完成")
            return self

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """使用模型进行预测

        Args:
            dataset: 预测数据集

        Returns:
            预测结果 DataFrame

        Raises:
            RuntimeError: 当模型未训练时
        """
        if not QLIB_AVAILABLE:
            logger.warning("Qlib 不可用，返回空预测结果")
            return pd.DataFrame()

        if not self.is_fitted or self.model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")

        try:
            logger.info("开始 Qlib 模型预测...")

            predictions = self.model.predict(dataset)

            # 转换预测结果格式
            if isinstance(predictions, pd.Series):
                predictions = predictions.to_frame("prediction")
            elif not isinstance(predictions, pd.DataFrame):
                predictions = pd.DataFrame(predictions, columns=["prediction"])

            logger.info(f"Qlib 预测完成: {len(predictions)} 条预测结果")
            return predictions

        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return pd.DataFrame()

    def fit_predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """训练模型并进行预测

        Args:
            dataset: 数据集（用于训练和预测）

        Returns:
            预测结果 DataFrame
        """
        self.fit(dataset)
        return self.predict(dataset)

    def set_config(self, model_config: Dict[str, Any]) -> "QlibModelPredictor":
        """设置模型配置

        Args:
            model_config: Qlib 模型配置字典

        Returns:
            self
        """
        self.model_config = model_config
        self.model = None
        self.is_fitted = False
        return self

    def get_model(self) -> Optional[Any]:
        """获取底层模型实例

        Returns:
            Qlib 模型实例，如果未训练则返回 None
        """
        return self.model

    def is_available(self) -> bool:
        """检查预测器是否可用

        Returns:
            是否可用
        """
        return QLIB_AVAILABLE

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """获取特征重要性（如果模型支持）

        Returns:
            特征重要性 DataFrame，如果不支持则返回 None
        """
        if not self.is_fitted or self.model is None:
            return None

        try:
            if hasattr(self.model, "get_feature_importance"):
                importance = self.model.get_feature_importance()
                if isinstance(importance, dict):
                    return pd.DataFrame(
                        list(importance.items()),
                        columns=["feature", "importance"],
                    ).sort_values("importance", ascending=False)
                return importance
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {e}")

        return None
