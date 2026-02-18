"""
ML 模型加载器

负责加载统一训练引擎和 legacy 格式的模型，
提取模型元数据（特征集类型、标签类型）。
"""

import glob
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

_logger = logging.getLogger(__name__)

# 模型类型常量
MODEL_TYPE_LEGACY = "legacy"
MODEL_TYPE_UNIFIED = "unified"

# 默认特征集和标签类型（向后兼容）
DEFAULT_FEATURE_SET = "alpha158"
DEFAULT_LABEL_TYPE = "regression"


@dataclass
class LoadedModelPair:
    """加载后的模型对，包含元数据"""

    lgb_model: Any
    xgb_model: Any
    model_type: str  # "legacy" | "unified"
    feature_set: str  # "alpha158" | "technical_62" | "custom"
    label_type: str  # "regression" | "binary"
    binary_threshold: float  # 仅 binary 模式有意义
    feature_columns: List[str] = field(default_factory=list)  # 训练时的特征列名


def load_model_pair(
    model_dir: Path,
    lgb_model_id: Optional[str],
    xgb_model_id: Optional[str],
) -> Optional[LoadedModelPair]:
    """加载模型对（自动检测 legacy / unified 格式）

    Args:
        model_dir: 模型文件所在目录
        lgb_model_id: LGB 模型 ID（None 则尝试 legacy）
        xgb_model_id: XGB 模型 ID（None 则尝试 legacy）

    Returns:
        LoadedModelPair 或 None
    """
    if not model_dir.exists():
        return None

    if lgb_model_id or xgb_model_id:
        return _load_unified_models(
            model_dir,
            lgb_model_id,
            xgb_model_id,
        )
    return _load_legacy_models(model_dir)


def _load_legacy_models(model_dir: Path) -> Optional[LoadedModelPair]:
    """加载 legacy 格式模型（lgb_model.pkl / xgb_model.pkl）"""
    lgb_path = model_dir / "lgb_model.pkl"
    xgb_path = model_dir / "xgb_model.pkl"

    lgb_model = _load_pickle(lgb_path)
    xgb_model = _load_pickle(xgb_path)

    if lgb_model and xgb_model:
        _logger.info("Legacy 模型加载完成（62 特征）")
        return LoadedModelPair(
            lgb_model=lgb_model,
            xgb_model=xgb_model,
            model_type=MODEL_TYPE_LEGACY,
            feature_set="technical_62",
            label_type="regression",
            binary_threshold=0.0,
        )
    return None


def _load_unified_models(
    model_dir: Path,
    lgb_model_id: Optional[str],
    xgb_model_id: Optional[str],
) -> Optional[LoadedModelPair]:
    """加载统一引擎训练的模型，提取元数据"""
    lgb_result = _load_single_unified(model_dir, lgb_model_id, "LGB")
    xgb_result = _load_single_unified(model_dir, xgb_model_id, "XGB")

    lgb_model = lgb_result[0] if lgb_result else None
    xgb_model = xgb_result[0] if xgb_result else None

    if not (lgb_model or xgb_model):
        return None

    # 如果只有一个模型，复用它作为另一个
    if lgb_model and not xgb_model:
        _logger.info("仅有 LGB 模型，复用为 XGB")
        xgb_model = lgb_model
    elif xgb_model and not lgb_model:
        _logger.info("仅有 XGB 模型，复用为 LGB")
        lgb_model = xgb_model

    # 从任一模型的 config 中提取元数据
    metadata = lgb_result[1] if lgb_result else xgb_result[1]
    feature_set = metadata.get("feature_set", DEFAULT_FEATURE_SET)
    label_type = metadata.get("label_type", DEFAULT_LABEL_TYPE)
    binary_threshold = metadata.get("binary_threshold", 0.003)
    feature_columns = metadata.get("feature_columns", [])

    _logger.info(f"统一引擎模型加载完成: feature_set={feature_set}, "
                 f"label_type={label_type}, feature_columns={len(feature_columns)}")
    return LoadedModelPair(
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        model_type=MODEL_TYPE_UNIFIED,
        feature_set=feature_set,
        label_type=label_type,
        binary_threshold=binary_threshold,
        feature_columns=feature_columns,
    )


def _load_single_unified(
    model_dir: Path,
    model_id: Optional[str],
    label: str,
) -> Optional[tuple]:
    """加载单个统一引擎模型，返回 (booster, metadata_dict)"""
    if not model_id:
        return None

    pattern = str(model_dir / f"{model_id}_qlib_*.pkl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        _logger.warning(f"{label} 模型未找到: {pattern}")
        return None

    pkl_path = Path(matches[-1])
    obj = _load_pickle(pkl_path)
    if not isinstance(obj, dict) or "model" not in obj:
        _logger.warning(f"{label} 模型格式异常: {type(obj)}")
        return None

    # 提取元数据
    metadata = obj.get("config", {})
    # 统一引擎新增的训练元数据
    training_meta = obj.get("training_meta", {})
    merged_meta = {**metadata, **training_meta}

    # 提取原生 Booster
    qlib_model = obj["model"]
    booster = getattr(qlib_model, "model", None)
    if booster is None:
        _logger.warning(f"{label} 无法提取 Booster")
        return None

    _logger.info(f"{label} 模型已加载: {Path(pkl_path).name}")
    return (booster, merged_meta)


def _load_pickle(path: Path) -> Any:
    """安全加载 pickle 文件"""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        _logger.error(f"加载模型失败 {path}: {e}")
        return None
