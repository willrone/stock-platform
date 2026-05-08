"""
Qlib集成服务模块（轻量懒加载入口）。

该模块避免在包导入时 eager import 重依赖子模块，
从而减少 qlib/psutil 等可选依赖缺失带来的导入失败。
"""

from importlib import import_module
from typing import Any, Dict, List, Tuple

# 名称 -> (子模块, 属性名)
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # 数据提供器
    "EnhancedQlibDataProvider": ("enhanced_qlib_provider", "EnhancedQlibDataProvider"),
    "Alpha158Calculator": ("enhanced_qlib_provider", "Alpha158Calculator"),
    "FactorCache": ("enhanced_qlib_provider", "FactorCache"),
    "QlibDataAdapter": ("qlib_data_adapter", "QlibDataAdapter"),
    # 训练引擎
    "UnifiedQlibTrainingEngine": (
        "unified_qlib_training_engine",
        "UnifiedQlibTrainingEngine",
    ),
    "QlibTrainingConfig": ("unified_qlib_training_engine", "QlibTrainingConfig"),
    "QlibTrainingResult": ("unified_qlib_training_engine", "QlibTrainingResult"),
    "QlibModelType": ("unified_qlib_training_engine", "QlibModelType"),
    # 模型管理器
    "QlibModelManager": ("qlib_model_manager", "QlibModelManager"),
    "ModelMetadata": ("qlib_model_manager", "ModelMetadata"),
    "HyperparameterSpec": ("qlib_model_manager", "HyperparameterSpec"),
    "ModelCategory": ("qlib_model_manager", "ModelCategory"),
    "ModelComplexity": ("qlib_model_manager", "ModelComplexity"),
    # 自定义模型
    "CustomTransformerModel": ("custom_models", "CustomTransformerModel"),
    "CustomInformerModel": ("custom_models", "CustomInformerModel"),
    "CustomTimesNetModel": ("custom_models", "CustomTimesNetModel"),
    "CustomPatchTSTModel": ("custom_models", "CustomPatchTSTModel"),
}

# 支持 `from app.services.qlib import official_workflow` 这类子模块导入
_LAZY_SUBMODULES = {
    "official_workflow",
    "enhanced_qlib_provider",
    "qlib_data_adapter",
    "qlib_model_manager",
    "unified_qlib_training_engine",
    "custom_models",
}

# 仅检查 custom_models 是否可导入，不触发 provider/training engine 导入
try:
    import_module(f"{__name__}.custom_models")
    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    CUSTOM_MODELS_AVAILABLE = False

__all__ = [
    # 数据提供器
    "EnhancedQlibDataProvider",
    "Alpha158Calculator",
    "FactorCache",
    "QlibDataAdapter",
    # 训练引擎
    "UnifiedQlibTrainingEngine",
    "QlibTrainingConfig",
    "QlibTrainingResult",
    "QlibModelType",
    # 模型管理器
    "QlibModelManager",
    "ModelMetadata",
    "HyperparameterSpec",
    "ModelCategory",
    "ModelComplexity",
    # 可用性标志
    "CUSTOM_MODELS_AVAILABLE",
]

# 如果自定义模型可用，添加到导出列表
if CUSTOM_MODELS_AVAILABLE:
    __all__.extend(
        [
            "CustomTransformerModel",
            "CustomInformerModel",
            "CustomTimesNetModel",
            "CustomPatchTSTModel",
        ]
    )


def __getattr__(name: str) -> Any:
    if name == "CUSTOM_MODELS_AVAILABLE":
        return CUSTOM_MODELS_AVAILABLE

    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__) | _LAZY_SUBMODULES)
