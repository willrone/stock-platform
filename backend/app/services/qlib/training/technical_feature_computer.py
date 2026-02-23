"""
手工技术指标特征计算器

将体系A的62个手工特征整合为可复用的计算模块，
供统一训练引擎和回测策略共同使用。
"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from .feature_sets import (
    TECHNICAL_62_FEATURES,
    compute_bollinger_features,
    compute_ma_features,
    compute_macd_features,
    compute_momentum_features,
    compute_pattern_features,
    compute_position_features,
    compute_return_features,
    compute_rsi_features,
    compute_volatility_features,
    compute_volume_features,
)

EPSILON = 1e-10


def compute_stock_technical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """为单只股票计算62个技术指标特征

    Args:
        df: 包含 $close/$high/$low/$open/$volume 列的 DataFrame

    Returns:
        添加了技术指标列的 DataFrame
    """
    close = _get_col(df, "close")
    high = _get_col(df, "high")
    low = _get_col(df, "low")
    open_ = _get_col(df, "open")
    volume = _get_col(df, "volume")

    if close is None:
        logger.warning("缺少收盘价列，无法计算技术指标")
        return df

    result = df.copy()
    all_features: Dict[str, pd.Series] = {}

    # 收益率 + 动量
    ret_feats = compute_return_features(
        _wrap_ohlcv(close, high, low, open_, volume)
    )
    all_features.update(ret_feats)
    all_features.update(
        compute_momentum_features(ret_feats, close)
    )

    # MA / 波动率 / 成交量
    all_features.update(compute_ma_features(close))
    all_features.update(compute_volatility_features(close))
    if volume is not None:
        all_features.update(
            compute_volume_features(close, volume)
        )

    # RSI / MACD / 布林带
    all_features.update(compute_rsi_features(close))
    all_features.update(compute_macd_features(close))
    all_features.update(compute_bollinger_features(close))

    # 价格形态 / 位置 / ATR+ADX
    if all(s is not None for s in [high, low, open_]):
        all_features.update(
            compute_pattern_features(close, high, low, open_)
        )
        all_features.update(
            compute_position_features(close, high, low)
        )

    for name, series in all_features.items():
        result[name] = series

    return result


def compute_cross_sectional_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """计算截面特征（每日排名、相对强度）

    需要多只股票的合并数据，包含 date 列或 MultiIndex。

    Args:
        df: 包含 date 列和技术指标的 DataFrame

    Returns:
        添加了截面特征的 DataFrame
    """
    result = df.copy()
    date_col = _resolve_date_col(result)
    if date_col is None:
        logger.warning("无法确定日期列，跳过截面特征")
        return result

    # 排名特征
    rank_cols = [
        ("return_1d", "return_1d_rank"),
        ("return_5d", "return_5d_rank"),
        ("return_20d", "return_20d_rank"),
        ("$volume", "volume_rank"),
        ("volatility_20", "volatility_20_rank"),
    ]
    for src, dst in rank_cols:
        if src in result.columns:
            result[dst] = result.groupby(date_col)[src].rank(
                pct=True
            )

    # 市场上涨比例
    if "return_1d" in result.columns:
        result["market_up_ratio"] = result.groupby(date_col)[
            "return_1d"
        ].transform(lambda x: (x > 0).sum() / max(len(x), 1))

    # 相对强度
    _add_relative_strength(result, date_col)

    return result


def _add_relative_strength(
    df: pd.DataFrame, date_col: str,
) -> None:
    """添加相对强度特征（原地修改）"""
    for period, col_name in [
        ("return_5d", "relative_strength_5d"),
        ("return_20d", "relative_strength_20d"),
    ]:
        if period in df.columns:
            market_mean = df.groupby(date_col)[period].transform(
                "mean"
            )
            df[col_name] = df[period] - market_mean

    if (
        "relative_strength_5d" in df.columns
        and "relative_strength_20d" in df.columns
    ):
        df["relative_momentum"] = (
            df["relative_strength_5d"]
            - df["relative_strength_20d"]
        )


def _get_col(
    df: pd.DataFrame, name: str,
) -> Optional[pd.Series]:
    """获取列，支持 $close 和 close 两种命名"""
    for candidate in [f"${name}", name, name.capitalize()]:
        if candidate in df.columns:
            return df[candidate]
    return None


def _wrap_ohlcv(
    close: pd.Series,
    high: Optional[pd.Series],
    low: Optional[pd.Series],
    open_: Optional[pd.Series],
    volume: Optional[pd.Series],
) -> pd.DataFrame:
    """将 OHLCV Series 包装为 DataFrame（$前缀列名）"""
    data = {"$close": close}
    if high is not None:
        data["$high"] = high
    if low is not None:
        data["$low"] = low
    if open_ is not None:
        data["$open"] = open_
    if volume is not None:
        data["$volume"] = volume
    return pd.DataFrame(data, index=close.index)


def _resolve_date_col(df: pd.DataFrame) -> Optional[str]:
    """确定日期列名"""
    if isinstance(df.index, pd.MultiIndex):
        return df.index.names[1]
    for col in ["date", "datetime", "trade_date"]:
        if col in df.columns:
            return col
    return None
