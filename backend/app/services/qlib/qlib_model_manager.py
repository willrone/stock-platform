"""
Qlib模型配置管理器

支持传统ML模型和深度学习模型的统一配置管理
包括Transformer、Informer、TimesNet、PatchTST等现代模型
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from loguru import logger

# 检测可用的深度学习框架
try:
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch不可用，深度学习模型将不可用")

try:
    import qlib

    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("Qlib不可用，某些功能将受限")


class ModelCategory(Enum):
    """模型类别"""

    TRADITIONAL_ML = "traditional_ml"  # 传统机器学习
    DEEP_LEARNING = "deep_learning"  # 深度学习
    TIME_SERIES = "time_series"  # 时间序列专用


class ModelComplexity(Enum):
    """模型复杂度"""

    LOW = "low"  # 低复杂度，快速训练
    MEDIUM = "medium"  # 中等复杂度
    HIGH = "high"  # 高复杂度，需要更多资源


@dataclass
class ModelMetadata:
    """模型元数据"""

    name: str
    display_name: str
    category: ModelCategory
    complexity: ModelComplexity
    description: str
    supported_tasks: List[str]  # 支持的任务类型：regression, classification, forecasting
    min_samples: int  # 最小样本数要求
    recommended_features: int  # 推荐特征数
    training_time_estimate: str  # 训练时间估计
    memory_requirement: str  # 内存需求

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class HyperparameterSpec:
    """超参数规格"""

    name: str
    param_type: str  # int, float, choice, bool
    default_value: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class BaseModelAdapter(ABC):
    """模型适配器基类"""

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """获取模型元数据"""
        pass

    @abstractmethod
    def get_hyperparameter_specs(self) -> List[HyperparameterSpec]:
        """获取超参数规格"""
        pass

    @abstractmethod
    def create_qlib_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """创建Qlib配置"""
        pass

    @abstractmethod
    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """验证超参数"""
        pass


class LightGBMAdapter(BaseModelAdapter):
    """LightGBM模型适配器"""

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="lightgbm",
            display_name="LightGBM",
            category=ModelCategory.TRADITIONAL_ML,
            complexity=ModelComplexity.MEDIUM,
            description="高效的梯度提升决策树，适合大规模数据和特征工程",
            supported_tasks=["regression", "classification"],
            min_samples=1000,
            recommended_features=50,
            training_time_estimate="5-15分钟",
            memory_requirement="中等",
        )

    def get_hyperparameter_specs(self) -> List[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="learning_rate",
                param_type="float",
                default_value=0.1,
                min_value=0.01,
                max_value=0.3,
                description="学习率，控制每次迭代的步长",
            ),
            HyperparameterSpec(
                name="num_leaves",
                param_type="int",
                default_value=31,
                min_value=10,
                max_value=300,
                description="叶子节点数，控制模型复杂度",
            ),
            HyperparameterSpec(
                name="max_depth",
                param_type="int",
                default_value=-1,
                min_value=3,
                max_value=15,
                description="树的最大深度，-1表示无限制",
            ),
            HyperparameterSpec(
                name="min_data_in_leaf",
                param_type="int",
                default_value=20,
                min_value=5,
                max_value=100,
                description="叶子节点最小样本数",
            ),
            HyperparameterSpec(
                name="feature_fraction",
                param_type="float",
                default_value=0.9,
                min_value=0.4,
                max_value=1.0,
                description="特征采样比例",
            ),
            HyperparameterSpec(
                name="bagging_fraction",
                param_type="float",
                default_value=0.8,
                min_value=0.4,
                max_value=1.0,
                description="样本采样比例",
            ),
            HyperparameterSpec(
                name="lambda_l1",
                param_type="float",
                default_value=0.0,
                min_value=0.0,
                max_value=100.0,
                description="L1正则化系数",
            ),
            HyperparameterSpec(
                name="lambda_l2",
                param_type="float",
                default_value=0.0,
                min_value=0.0,
                max_value=100.0,
                description="L2正则化系数",
            ),
            HyperparameterSpec(
                name="num_iterations",
                param_type="int",
                default_value=100,
                min_value=10,
                max_value=1000,
                description="训练迭代次数（epochs），LightGBM使用num_iterations参数",
            ),
        ]

    def create_qlib_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "huber",  # 使用Huber损失，对异常值更鲁棒
                "huber_delta": hyperparameters.get("huber_delta", 0.1),  # Huber损失的delta参数
                "learning_rate": hyperparameters.get("learning_rate", 0.1),
                "num_leaves": hyperparameters.get("num_leaves", 31),
                "max_depth": hyperparameters.get("max_depth", -1),
                "min_data_in_leaf": hyperparameters.get("min_data_in_leaf", 20),
                "feature_fraction": hyperparameters.get("feature_fraction", 0.9),
                "bagging_fraction": hyperparameters.get("bagging_fraction", 0.8),
                "lambda_l1": hyperparameters.get("lambda_l1", 0.0),
                "lambda_l2": hyperparameters.get("lambda_l2", 0.0),
                "num_threads": 20,
                "verbose": -1,  # 禁用LightGBM的默认输出，但保留训练历史
            },
        }

        # 添加epoch数配置（LightGBM使用num_iterations或n_estimators）
        num_iterations = None
        if "num_iterations" in hyperparameters:
            num_iterations = hyperparameters["num_iterations"]
        elif "n_estimators" in hyperparameters:
            num_iterations = hyperparameters["n_estimators"]
        elif "epochs" in hyperparameters:
            num_iterations = hyperparameters["epochs"]

        if num_iterations:
            config["kwargs"]["num_iterations"] = num_iterations

        return config

    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """验证超参数"""
        try:
            lr = hyperparameters.get("learning_rate", 0.1)
            if not (0.01 <= lr <= 0.3):
                return False

            num_leaves = hyperparameters.get("num_leaves", 31)
            if not (10 <= num_leaves <= 300):
                return False

            return True
        except:
            return False


class LinearAdapter(BaseModelAdapter):
    """线性回归模型适配器"""

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="linear",
            display_name="线性回归",
            category=ModelCategory.TRADITIONAL_ML,
            complexity=ModelComplexity.LOW,
            description="简单的线性回归模型，训练快速，高可解释性",
            supported_tasks=["regression"],
            min_samples=100,
            recommended_features=10,
            training_time_estimate="1-5分钟",
            memory_requirement="低",
        )

    def get_hyperparameter_specs(self) -> List[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="alpha",
                param_type="float",
                default_value=0.01,
                min_value=0.0001,
                max_value=1.0,
                description="正则化强度",
            ),
            HyperparameterSpec(
                name="fit_intercept",
                param_type="bool",
                default_value=True,
                description="是否拟合截距项",
            ),
            HyperparameterSpec(
                name="normalize",
                param_type="bool",
                default_value=False,
                description="是否标准化特征",
            ),
        ]

    def create_qlib_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "class": "LinearModel",
            "module_path": "qlib.contrib.model.linear",
            "kwargs": {
                "alpha": hyperparameters.get("alpha", 0.01),
                "fit_intercept": hyperparameters.get("fit_intercept", True),
                "normalize": hyperparameters.get("normalize", False),
            },
        }

    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        try:
            alpha = hyperparameters.get("alpha", 0.01)
            if not (0.0001 <= alpha <= 1.0):
                return False
            return True
        except:
            return False


class XGBoostAdapter(BaseModelAdapter):
    """XGBoost模型适配器"""

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="xgboost",
            display_name="XGBoost",
            category=ModelCategory.TRADITIONAL_ML,
            complexity=ModelComplexity.MEDIUM,
            description="极端梯度提升，在结构化数据上表现优异",
            supported_tasks=["regression", "classification"],
            min_samples=500,
            recommended_features=30,
            training_time_estimate="10-20分钟",
            memory_requirement="中等",
        )

    def get_hyperparameter_specs(self) -> List[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="learning_rate",
                param_type="float",
                default_value=0.1,
                min_value=0.01,
                max_value=0.3,
                description="学习率",
            ),
            HyperparameterSpec(
                name="max_depth",
                param_type="int",
                default_value=6,
                min_value=3,
                max_value=15,
                description="树的最大深度",
            ),
            HyperparameterSpec(
                name="n_estimators",
                param_type="int",
                default_value=100,
                min_value=50,
                max_value=1000,
                description="树的数量",
            ),
            HyperparameterSpec(
                name="subsample",
                param_type="float",
                default_value=0.8,
                min_value=0.5,
                max_value=1.0,
                description="样本采样比例",
            ),
            HyperparameterSpec(
                name="colsample_bytree",
                param_type="float",
                default_value=0.8,
                min_value=0.5,
                max_value=1.0,
                description="特征采样比例",
            ),
        ]

    def create_qlib_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        config = {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "learning_rate": hyperparameters.get("learning_rate", 0.1),
                "max_depth": hyperparameters.get("max_depth", 6),
                "n_estimators": hyperparameters.get("n_estimators", 100),
                "subsample": hyperparameters.get("subsample", 0.8),
                "colsample_bytree": hyperparameters.get("colsample_bytree", 0.8),
                "random_state": 42,
            },
        }

        # 支持num_iterations或epochs作为n_estimators的别名
        if "num_iterations" in hyperparameters:
            config["kwargs"]["n_estimators"] = hyperparameters["num_iterations"]
        elif "epochs" in hyperparameters:
            config["kwargs"]["n_estimators"] = hyperparameters["epochs"]

        return config

    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        try:
            lr = hyperparameters.get("learning_rate", 0.1)
            if not (0.01 <= lr <= 0.3):
                return False
            return True
        except:
            return False


class MLPAdapter(BaseModelAdapter):
    """多层感知机适配器"""

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="mlp",
            display_name="多层感知机 (MLP)",
            category=ModelCategory.DEEP_LEARNING,
            complexity=ModelComplexity.MEDIUM,
            description="经典的深度神经网络，适合非线性特征学习",
            supported_tasks=["regression", "classification"],
            min_samples=2000,
            recommended_features=20,
            training_time_estimate="15-30分钟",
            memory_requirement="中等",
        )

    def get_hyperparameter_specs(self) -> List[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="lr",
                param_type="float",
                default_value=0.001,
                min_value=0.0001,
                max_value=0.01,
                description="学习率",
            ),
            HyperparameterSpec(
                name="max_steps",
                param_type="int",
                default_value=8000,
                min_value=1000,
                max_value=20000,
                description="最大训练步数",
            ),
            HyperparameterSpec(
                name="batch_size",
                param_type="int",
                default_value=2000,
                min_value=256,
                max_value=8192,
                description="批次大小",
            ),
            HyperparameterSpec(
                name="early_stop_rounds",
                param_type="int",
                default_value=50,
                min_value=10,
                max_value=200,
                description="早停轮数",
            ),
        ]

    def create_qlib_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "class": "DNNModelPytorch",
            "module_path": "qlib.contrib.model.pytorch_nn",
            "kwargs": {
                "lr": hyperparameters.get("lr", 0.001),
                "max_steps": hyperparameters.get("max_steps", 8000),
                "batch_size": hyperparameters.get("batch_size", 2000),
                "early_stop_rounds": hyperparameters.get("early_stop_rounds", 50),
                "eval_steps": 100,
            },
        }

    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        try:
            lr = hyperparameters.get("lr", 0.001)
            if not (0.0001 <= lr <= 0.01):
                return False
            return True
        except:
            return False


class TransformerAdapter(BaseModelAdapter):
    """Transformer模型适配器"""

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="transformer",
            display_name="Transformer",
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.HIGH,
            description="基于注意力机制的序列模型，擅长捕捉长期依赖",
            supported_tasks=["forecasting", "regression"],
            min_samples=5000,
            recommended_features=15,
            training_time_estimate="30-60分钟",
            memory_requirement="高",
        )

    def get_hyperparameter_specs(self) -> List[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="d_model",
                param_type="int",
                default_value=128,
                min_value=64,
                max_value=512,
                description="模型维度",
            ),
            HyperparameterSpec(
                name="nhead",
                param_type="choice",
                default_value=8,
                choices=[4, 8, 16],
                description="注意力头数",
            ),
            HyperparameterSpec(
                name="num_layers",
                param_type="int",
                default_value=4,
                min_value=2,
                max_value=8,
                description="编码器层数",
            ),
            HyperparameterSpec(
                name="dropout",
                param_type="float",
                default_value=0.1,
                min_value=0.0,
                max_value=0.5,
                description="Dropout比例",
            ),
            HyperparameterSpec(
                name="learning_rate",
                param_type="float",
                default_value=0.001,
                min_value=0.0001,
                max_value=0.01,
                description="学习率",
            ),
        ]

    def create_qlib_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        # 注意：这里返回的是自定义配置，需要在训练引擎中特殊处理
        return {
            "class": "CustomTransformerModel",
            "module_path": "app.services.qlib.custom_models",
            "kwargs": {
                "d_model": hyperparameters.get("d_model", 128),
                "nhead": hyperparameters.get("nhead", 8),
                "num_layers": hyperparameters.get("num_layers", 4),
                "dropout": hyperparameters.get("dropout", 0.1),
                "learning_rate": hyperparameters.get("learning_rate", 0.001),
            },
        }

    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        try:
            d_model = hyperparameters.get("d_model", 128)
            nhead = hyperparameters.get("nhead", 8)

            # d_model必须能被nhead整除
            if d_model % nhead != 0:
                return False

            return True
        except:
            return False


class InformerAdapter(BaseModelAdapter):
    """Informer模型适配器"""

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="informer",
            display_name="Informer",
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.HIGH,
            description="高效的长序列时间序列预测模型，基于稀疏注意力机制",
            supported_tasks=["forecasting", "regression"],
            min_samples=8000,
            recommended_features=10,
            training_time_estimate="45-90分钟",
            memory_requirement="高",
        )

    def get_hyperparameter_specs(self) -> List[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="d_model",
                param_type="int",
                default_value=512,
                min_value=128,
                max_value=1024,
                description="模型维度",
            ),
            HyperparameterSpec(
                name="n_heads",
                param_type="choice",
                default_value=8,
                choices=[4, 8, 16],
                description="注意力头数",
            ),
            HyperparameterSpec(
                name="e_layers",
                param_type="int",
                default_value=2,
                min_value=1,
                max_value=4,
                description="编码器层数",
            ),
            HyperparameterSpec(
                name="d_layers",
                param_type="int",
                default_value=1,
                min_value=1,
                max_value=3,
                description="解码器层数",
            ),
            HyperparameterSpec(
                name="factor",
                param_type="int",
                default_value=5,
                min_value=3,
                max_value=10,
                description="稀疏注意力因子",
            ),
        ]

    def create_qlib_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "class": "CustomInformerModel",
            "module_path": "app.services.qlib.custom_models",
            "kwargs": hyperparameters,
        }

    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        try:
            d_model = hyperparameters.get("d_model", 512)
            n_heads = hyperparameters.get("n_heads", 8)

            if d_model % n_heads != 0:
                return False

            return True
        except:
            return False


class TimesNetAdapter(BaseModelAdapter):
    """TimesNet模型适配器"""

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="timesnet",
            display_name="TimesNet",
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.HIGH,
            description="基于2D卷积的时间序列分析模型，能够捕捉多周期模式",
            supported_tasks=["forecasting", "regression"],
            min_samples=6000,
            recommended_features=12,
            training_time_estimate="40-80分钟",
            memory_requirement="高",
        )

    def get_hyperparameter_specs(self) -> List[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="d_model",
                param_type="int",
                default_value=64,
                min_value=32,
                max_value=256,
                description="模型维度",
            ),
            HyperparameterSpec(
                name="d_ff",
                param_type="int",
                default_value=256,
                min_value=128,
                max_value=1024,
                description="前馈网络维度",
            ),
            HyperparameterSpec(
                name="num_kernels",
                param_type="int",
                default_value=6,
                min_value=3,
                max_value=10,
                description="卷积核数量",
            ),
            HyperparameterSpec(
                name="top_k",
                param_type="int",
                default_value=5,
                min_value=3,
                max_value=8,
                description="Top-K周期选择",
            ),
        ]

    def create_qlib_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "class": "CustomTimesNetModel",
            "module_path": "app.services.qlib.custom_models",
            "kwargs": hyperparameters,
        }

    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        return True  # TimesNet参数相对灵活


class PatchTSTAdapter(BaseModelAdapter):
    """PatchTST模型适配器"""

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="patchtst",
            display_name="PatchTST",
            category=ModelCategory.TIME_SERIES,
            complexity=ModelComplexity.HIGH,
            description="基于补丁的时间序列Transformer，高效处理长序列",
            supported_tasks=["forecasting", "regression"],
            min_samples=4000,
            recommended_features=8,
            training_time_estimate="25-50分钟",
            memory_requirement="中高",
        )

    def get_hyperparameter_specs(self) -> List[HyperparameterSpec]:
        return [
            HyperparameterSpec(
                name="patch_len",
                param_type="int",
                default_value=16,
                min_value=8,
                max_value=32,
                description="补丁长度",
            ),
            HyperparameterSpec(
                name="stride",
                param_type="int",
                default_value=8,
                min_value=4,
                max_value=16,
                description="补丁步长",
            ),
            HyperparameterSpec(
                name="d_model",
                param_type="int",
                default_value=128,
                min_value=64,
                max_value=512,
                description="模型维度",
            ),
            HyperparameterSpec(
                name="n_heads",
                param_type="choice",
                default_value=8,
                choices=[4, 8, 16],
                description="注意力头数",
            ),
            HyperparameterSpec(
                name="num_layers",
                param_type="int",
                default_value=3,
                min_value=2,
                max_value=6,
                description="Transformer层数",
            ),
        ]

    def create_qlib_config(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "class": "CustomPatchTSTModel",
            "module_path": "app.services.qlib.custom_models",
            "kwargs": hyperparameters,
        }

    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        try:
            patch_len = hyperparameters.get("patch_len", 16)
            stride = hyperparameters.get("stride", 8)

            # stride应该小于等于patch_len
            if stride > patch_len:
                return False

            return True
        except:
            return False


class QlibModelManager:
    """Qlib模型配置管理器"""

    def __init__(self):
        self.adapters: Dict[str, BaseModelAdapter] = {}
        self._register_default_adapters()

        logger.info(f"Qlib模型管理器初始化完成，支持 {len(self.adapters)} 种模型")

    def _register_default_adapters(self):
        """注册默认的模型适配器"""
        # 传统ML模型
        self.adapters["lightgbm"] = LightGBMAdapter()
        self.adapters["xgboost"] = XGBoostAdapter()
        self.adapters["linear"] = LinearAdapter()  # 添加线性回归适配器
        self.adapters["mlp"] = MLPAdapter()

        # 深度学习模型（仅在PyTorch可用时注册）
        if PYTORCH_AVAILABLE:
            self.adapters["transformer"] = TransformerAdapter()
            self.adapters["informer"] = InformerAdapter()
            self.adapters["timesnet"] = TimesNetAdapter()
            self.adapters["patchtst"] = PatchTSTAdapter()
        else:
            logger.warning("PyTorch不可用，深度学习模型将不可用")

    def register_adapter(self, name: str, adapter: BaseModelAdapter):
        """注册自定义模型适配器"""
        self.adapters[name] = adapter
        logger.info(f"注册自定义模型适配器: {name}")

    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return list(self.adapters.keys())

    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """获取模型元数据"""
        adapter = self.adapters.get(model_name)
        if adapter:
            return adapter.get_metadata()
        return None

    def get_all_models_metadata(self) -> Dict[str, ModelMetadata]:
        """获取所有模型的元数据"""
        metadata = {}
        for name, adapter in self.adapters.items():
            metadata[name] = adapter.get_metadata()
        return metadata

    def get_hyperparameter_specs(self, model_name: str) -> List[HyperparameterSpec]:
        """获取模型的超参数规格"""
        adapter = self.adapters.get(model_name)
        if adapter:
            return adapter.get_hyperparameter_specs()
        return []

    def create_qlib_config(
        self, model_name: str, hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建Qlib模型配置"""
        adapter = self.adapters.get(model_name)
        if not adapter:
            raise ValueError(f"不支持的模型类型: {model_name}")

        # 验证超参数
        if not adapter.validate_hyperparameters(hyperparameters):
            raise ValueError(f"模型 {model_name} 的超参数验证失败")

        return adapter.create_qlib_config(hyperparameters)

    def validate_hyperparameters(
        self, model_name: str, hyperparameters: Dict[str, Any]
    ) -> bool:
        """验证超参数"""
        adapter = self.adapters.get(model_name)
        if adapter:
            return adapter.validate_hyperparameters(hyperparameters)
        return False

    def get_models_by_category(self, category: ModelCategory) -> List[str]:
        """按类别获取模型"""
        models = []
        for name, adapter in self.adapters.items():
            metadata = adapter.get_metadata()
            if metadata.category == category:
                models.append(name)
        return models

    def get_models_by_complexity(self, complexity: ModelComplexity) -> List[str]:
        """按复杂度获取模型"""
        models = []
        for name, adapter in self.adapters.items():
            metadata = adapter.get_metadata()
            if metadata.complexity == complexity:
                models.append(name)
        return models

    def recommend_models(
        self, sample_count: int, feature_count: int, task_type: str = "regression"
    ) -> List[str]:
        """根据数据特征推荐模型"""
        recommendations = []

        for name, adapter in self.adapters.items():
            metadata = adapter.get_metadata()

            # 检查任务类型支持
            if task_type not in metadata.supported_tasks:
                continue

            # 检查样本数要求
            if sample_count < metadata.min_samples:
                continue

            # 根据特征数和样本数评分
            score = 0

            # 样本数评分
            if sample_count >= metadata.min_samples * 2:
                score += 2
            elif sample_count >= metadata.min_samples:
                score += 1

            # 特征数评分
            if feature_count >= metadata.recommended_features:
                score += 1

            # 复杂度评分（样本少时偏向简单模型）
            if sample_count < 5000 and metadata.complexity == ModelComplexity.LOW:
                score += 1
            elif sample_count >= 10000 and metadata.complexity == ModelComplexity.HIGH:
                score += 1

            if score > 0:
                recommendations.append((name, score))

        # 按评分排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in recommendations]

    def export_config_template(self, model_name: str, file_path: str):
        """导出模型配置模板"""
        adapter = self.adapters.get(model_name)
        if not adapter:
            raise ValueError(f"不支持的模型类型: {model_name}")

        metadata = adapter.get_metadata()
        hyperparameter_specs = adapter.get_hyperparameter_specs()

        template = {
            "model_name": model_name,
            "metadata": metadata.to_dict(),
            "hyperparameters": {
                spec.name: {
                    "type": spec.param_type,
                    "default": spec.default_value,
                    "description": spec.description,
                    "min_value": spec.min_value,
                    "max_value": spec.max_value,
                    "choices": spec.choices,
                }
                for spec in hyperparameter_specs
            },
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

        logger.info(f"模型配置模板已导出: {file_path}")

    def get_training_recommendations(self, model_name: str) -> Dict[str, Any]:
        """获取训练建议"""
        metadata = self.get_model_metadata(model_name)
        if not metadata:
            return {}

        return {
            "min_samples": metadata.min_samples,
            "recommended_features": metadata.recommended_features,
            "training_time_estimate": metadata.training_time_estimate,
            "memory_requirement": metadata.memory_requirement,
            "complexity": metadata.complexity.value,
            "category": metadata.category.value,
            "tips": self._get_training_tips(model_name),
        }

    def _get_training_tips(self, model_name: str) -> List[str]:
        """获取训练提示"""
        tips_map = {
            "lightgbm": ["适合大规模数据和高维特征", "建议使用特征工程提升效果", "可以通过调整num_leaves控制过拟合"],
            "xgboost": ["在结构化数据上表现优异", "建议使用交叉验证选择参数", "注意调整正则化参数防止过拟合"],
            "transformer": ["需要大量数据才能发挥优势", "建议使用GPU加速训练", "注意序列长度对内存的影响"],
            "informer": ["专门用于长序列预测", "需要充足的GPU内存", "适合处理复杂的时间模式"],
        }

        return tips_map.get(model_name, ["请参考模型文档进行配置"])
