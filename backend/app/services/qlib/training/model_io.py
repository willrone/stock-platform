"""
模型IO模块

包含模型保存和加载功能
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .qlib_check import QLIB_AVAILABLE

if QLIB_AVAILABLE:
    pass


async def save_qlib_model(
    model: Any,
    model_id: str,
    model_config: Dict[str, Any],
    training_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """保存Qlib模型

    Args:
        model: 训练好的模型对象
        model_id: 模型唯一标识
        model_config: Qlib 模型配置（超参数等）
        training_meta: 训练元数据（feature_set, label_type 等），
                       供回测策略加载时识别模型特征
    """
    try:
        # 创建模型保存目录
        from app.core.config import settings

        models_dir = Path(settings.MODEL_STORAGE_PATH)
        models_dir.mkdir(parents=True, exist_ok=True)

        # 生成模型文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_id}_qlib_{timestamp}"

        # 保存模型（使用pickle格式）
        model_path = models_dir / f"{model_filename}.pkl"

        payload = {
            "model": model,
            "config": model_config,
            "timestamp": timestamp,
        }
        # 保存训练元数据（特征集、标签类型等）
        if training_meta:
            payload["training_meta"] = training_meta

        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"Qlib模型保存成功: {model_path}")
        return str(model_path)

    except Exception as e:
        logger.error(f"保存Qlib模型失败: {e}")
        raise


async def load_qlib_model(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    """加载Qlib模型"""
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        config = model_data["config"]

        logger.info(f"Qlib模型加载成功: {model_path}")
        return model, config

    except Exception as e:
        logger.error(f"加载Qlib模型失败: {e}")
        raise


def _align_prediction_features(model: Any, dataset: pd.DataFrame) -> pd.DataFrame:
    """对齐预测数据特征列以匹配训练特征"""
    try:
        base_model = model.model if hasattr(model, "model") else model
        feature_names = None

        if hasattr(base_model, "feature_name"):
            try:
                feature_names = base_model.feature_name()
            except Exception:
                feature_names = None
        if feature_names is None and hasattr(base_model, "feature_name_"):
            feature_names = list(base_model.feature_name_)
        if (
            feature_names is None
            and hasattr(base_model, "booster_")
            and hasattr(base_model.booster_, "feature_name")
        ):
            feature_names = base_model.booster_.feature_name()

        if not feature_names:
            return dataset

        normalized_feature_names = []
        for name in feature_names:
            if isinstance(name, bytes):
                normalized_feature_names.append(name.decode(errors="ignore"))
            else:
                normalized_feature_names.append(str(name))

        dataset_columns = [str(col) for col in dataset.columns]
        has_named_match = any(
            name in dataset_columns for name in normalized_feature_names
        )
        name_mismatch = (
            all(name.startswith("Column_") for name in normalized_feature_names)
            and not has_named_match
        )

        if name_mismatch:
            data = dataset.values
            feature_count = len(normalized_feature_names)
            if data.shape[1] < feature_count:
                pad_width = feature_count - data.shape[1]
                data = np.hstack([data, np.zeros((data.shape[0], pad_width))])
            elif data.shape[1] > feature_count:
                data = data[:, :feature_count]
            logger.info(
                "预测特征使用位置对齐: model_features={}, dataset_features={}",
                feature_count,
                dataset.shape[1],
            )
            return pd.DataFrame(
                data, index=dataset.index, columns=normalized_feature_names
            )

        aligned = dataset.copy()
        missing = []
        for name in normalized_feature_names:
            if name not in aligned.columns:
                aligned[name] = 0.0
                missing.append(name)
        if missing and len(missing) == len(normalized_feature_names):
            data = dataset.values
            feature_count = len(normalized_feature_names)
            if data.shape[1] < feature_count:
                pad_width = feature_count - data.shape[1]
                data = np.hstack([data, np.zeros((data.shape[0], pad_width))])
            elif data.shape[1] > feature_count:
                data = data[:, :feature_count]
            logger.info(
                "预测特征全部缺失，回退到位置对齐: model_features={}, dataset_features={}",
                feature_count,
                dataset.shape[1],
            )
            return pd.DataFrame(
                data, index=dataset.index, columns=normalized_feature_names
            )

        aligned = aligned[normalized_feature_names]
        if missing:
            logger.info("预测特征缺失补齐: count={}, sample={}", len(missing), missing[:5])
        return aligned

    except Exception as e:
        logger.warning(f"对齐预测特征失败，使用原始数据: {e}")
        return dataset
