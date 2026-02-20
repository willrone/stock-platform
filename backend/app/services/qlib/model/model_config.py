"""
Qlib 模型配置构建器

提供各种 Qlib 模型的配置生成功能。
"""

from typing import Any, Dict, Optional

from loguru import logger


class QlibModelConfigBuilder:
    """Qlib 模型配置构建器

    支持的模型类型：
    - LightGBM (LGBModel)
    - XGBoost (XGBModel)
    - MLP (DNNModelPytorch)

    Attributes:
        default_lgb_params: LightGBM 默认参数
        default_xgb_params: XGBoost 默认参数
        default_mlp_params: MLP 默认参数
    """

    # LightGBM 默认参数（对齐 Qlib 官方基准 qlib/examples/benchmarks/LightGBM）
    DEFAULT_LGB_PARAMS = {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
        "num_boost_round": 1000,
        "early_stopping_rounds": 50,
        "min_child_samples": 100,
    }

    # XGBoost 默认参数
    DEFAULT_XGB_PARAMS = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 8,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }

    # MLP 默认参数
    DEFAULT_MLP_PARAMS = {
        "lr": 0.001,
        "batch_size": 2048,
        "max_steps": 8000,
        "early_stop_rounds": 50,
        "optimizer": "adam",
    }

    # 模型类型映射
    MODEL_MAPPING = {
        "lightgbm": ("LGBModel", "qlib.contrib.model.gbdt"),
        "lgb": ("LGBModel", "qlib.contrib.model.gbdt"),
        "xgboost": ("XGBModel", "qlib.contrib.model.xgboost"),
        "xgb": ("XGBModel", "qlib.contrib.model.xgboost"),
        "mlp": ("DNNModelPytorch", "qlib.contrib.model.pytorch_nn"),
        "dnn": ("DNNModelPytorch", "qlib.contrib.model.pytorch_nn"),
    }

    def __init__(self):
        """初始化模型配置构建器"""
        self.default_params = {
            "lightgbm": self.DEFAULT_LGB_PARAMS.copy(),
            "xgboost": self.DEFAULT_XGB_PARAMS.copy(),
            "mlp": self.DEFAULT_MLP_PARAMS.copy(),
        }

    def build(
        self,
        model_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建模型配置

        Args:
            model_type: 模型类型（lightgbm, xgboost, mlp）
            hyperparameters: 自定义超参数，会覆盖默认值

        Returns:
            Qlib 模型配置字典

        Raises:
            ValueError: 当模型类型不支持时
        """
        model_type_lower = model_type.lower()

        if model_type_lower not in self.MODEL_MAPPING:
            supported = list(self.MODEL_MAPPING.keys())
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: {supported}")

        class_name, module_path = self.MODEL_MAPPING[model_type_lower]

        # 获取默认参数
        default_key = self._get_default_key(model_type_lower)
        kwargs = self.default_params.get(default_key, {}).copy()

        # 合并自定义参数
        if hyperparameters:
            kwargs.update(hyperparameters)

        config = {
            "class": class_name,
            "module_path": module_path,
            "kwargs": kwargs,
        }

        logger.info(f"构建 {model_type} 模型配置完成")
        return config

    def _get_default_key(self, model_type: str) -> str:
        """获取默认参数的键名

        Args:
            model_type: 模型类型

        Returns:
            默认参数键名
        """
        mapping = {
            "lightgbm": "lightgbm",
            "lgb": "lightgbm",
            "xgboost": "xgboost",
            "xgb": "xgboost",
            "mlp": "mlp",
            "dnn": "mlp",
        }
        return mapping.get(model_type, "lightgbm")

    def build_lgb_config(
        self, hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """构建 LightGBM 模型配置

        Args:
            hyperparameters: 自定义超参数

        Returns:
            LightGBM 模型配置
        """
        return self.build("lightgbm", hyperparameters)

    def build_xgb_config(
        self, hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """构建 XGBoost 模型配置

        Args:
            hyperparameters: 自定义超参数

        Returns:
            XGBoost 模型配置
        """
        return self.build("xgboost", hyperparameters)

    def build_mlp_config(
        self, hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """构建 MLP 模型配置

        Args:
            hyperparameters: 自定义超参数

        Returns:
            MLP 模型配置
        """
        return self.build("mlp", hyperparameters)

    def get_supported_models(self) -> Dict[str, str]:
        """获取支持的模型类型

        Returns:
            模型类型及其描述
        """
        return {
            "lightgbm": "LightGBM 梯度提升模型",
            "xgboost": "XGBoost 梯度提升模型",
            "mlp": "多层感知机神经网络",
        }

    def update_default_params(self, model_type: str, params: Dict[str, Any]) -> None:
        """更新默认参数

        Args:
            model_type: 模型类型
            params: 要更新的参数
        """
        default_key = self._get_default_key(model_type.lower())
        if default_key in self.default_params:
            self.default_params[default_key].update(params)
            logger.info(f"更新 {model_type} 默认参数: {params}")
