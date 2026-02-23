"""
ML 特征适配器

根据模型的特征集类型，在回测时使用对应的特征计算方式。
桥接训练引擎的特征计算模块和回测策略。
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# 延迟导入避免循环依赖
_qlib_builder = None
_tech_computer = None


def _get_qlib_builder():
    global _qlib_builder
    if _qlib_builder is None:
        from .qlib_feature_builder import build_qlib_features
        _qlib_builder = build_qlib_features
    return _qlib_builder


def _get_tech_computer():
    global _tech_computer
    if _tech_computer is None:
        from app.services.qlib.training.technical_feature_computer import (
            compute_stock_technical_features,
        )
        _tech_computer = compute_stock_technical_features
    return _tech_computer


def build_feature_matrix(
    data: pd.DataFrame, feature_set: str,
) -> Optional[np.ndarray]:
    """根据特征集类型构建特征矩阵

    Args:
        data: OHLCV DataFrame
        feature_set: "alpha158" | "technical_62"

    Returns:
        特征矩阵 (n_samples, n_features)，失败返回 None
    """
    if feature_set == "alpha158":
        return _build_alpha158_matrix(data)
    if feature_set == "technical_62":
        return _build_technical62_matrix(data)
    _logger.warning(f"未知特征集: {feature_set}")
    return None


def build_feature_matrix_batch(
    df: pd.DataFrame, feature_set: str,
) -> Optional[np.ndarray]:
    """批量构建特征矩阵（多股票）

    Args:
        df: 包含 stock_code 列的 DataFrame
        feature_set: "alpha158" | "technical_62"

    Returns:
        特征矩阵，失败返回 None
    """
    if feature_set == "alpha158":
        return _build_alpha158_batch(df)
    if feature_set == "technical_62":
        return _build_technical62_batch(df)
    _logger.warning(f"未知特征集: {feature_set}")
    return None


def _build_alpha158_matrix(data: pd.DataFrame) -> Optional[np.ndarray]:
    """构建 alpha158 特征矩阵（44 个 Qlib 特征）"""
    builder = _get_qlib_builder()
    qlib_df = builder(data)
    if qlib_df is None:
        _logger.warning("Alpha158 特征构建失败")
        return None
    _logger.info(f"Alpha158 特征矩阵: {qlib_df.shape}")
    return qlib_df.fillna(0).values


def _build_technical62_matrix(
    data: pd.DataFrame,
) -> Optional[np.ndarray]:
    """构建 technical_62 特征矩阵（62 个手工特征）"""
    from app.services.qlib.training.feature_sets import (
        TECHNICAL_62_FEATURES,
    )

    computer = _get_tech_computer()
    # 列名适配：回测数据用 close，训练用 $close
    adapted = _adapt_column_names(data)
    result = computer(adapted)

    feature_names = TECHNICAL_62_FEATURES
    matrix = _extract_ordered_features(result, feature_names)
    _logger.info(f"Technical62 特征矩阵: ({len(data)}, {len(feature_names)})")
    return matrix


def _build_alpha158_batch(df: pd.DataFrame) -> Optional[np.ndarray]:
    """批量构建 alpha158 特征"""
    builder = _get_qlib_builder()
    all_parts: List[pd.DataFrame] = []

    for stock_code, stock_df in df.groupby("stock_code"):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        if len(stock_df) < 60:
            continue
        qf = builder(stock_df)
        if qf is not None:
            all_parts.append(qf)

    if not all_parts:
        _logger.warning("Alpha158 批量特征构建失败")
        return None

    combined = pd.concat(all_parts, ignore_index=True)
    _logger.info(f"Alpha158 批量特征矩阵: {combined.shape}")
    return combined.fillna(0).values


def _build_technical62_batch(
    df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """批量构建 technical_62 特征"""
    from app.services.qlib.training.feature_sets import (
        TECHNICAL_62_FEATURES,
    )

    computer = _get_tech_computer()
    all_parts: List[pd.DataFrame] = []

    for stock_code, stock_df in df.groupby("stock_code"):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        if len(stock_df) < 60:
            continue
        adapted = _adapt_column_names(stock_df)
        result = computer(adapted)
        all_parts.append(result)

    if not all_parts:
        _logger.warning("Technical62 批量特征构建失败")
        return None

    combined = pd.concat(all_parts, ignore_index=True)
    return _extract_ordered_features(combined, TECHNICAL_62_FEATURES)


def _adapt_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """适配列名：回测数据 close→$close 等"""
    mapping = {}
    for col in ["close", "high", "low", "open", "volume"]:
        dollar_col = f"${col}"
        if col in data.columns and dollar_col not in data.columns:
            mapping[col] = dollar_col
    if mapping:
        return data.rename(columns=mapping)
    return data


def _extract_ordered_features(
    df: pd.DataFrame, feature_names: List[str],
) -> np.ndarray:
    """按指定顺序提取特征列，缺失填 0"""
    result = pd.DataFrame(index=df.index)
    for name in feature_names:
        result[name] = df[name] if name in df.columns else 0.0
    return result.fillna(0).values
