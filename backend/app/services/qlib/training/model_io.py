"""
模型IO模块

包含模型保存和加载功能
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from loguru import logger

from .qlib_check import QLIB_AVAILABLE

if QLIB_AVAILABLE:
    import qlib


def _align_prediction_features(
        self, model: Any, dataset: pd.DataFrame
    ) -> pd.DataFrame:
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

