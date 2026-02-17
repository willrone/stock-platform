"""
特征集定义模块

定义不同的特征集（Alpha158、手工技术指标等），
供统一训练引擎选择使用。
"""

from enum import Enum
from typing import Dict, List

import numpy as np
import pandas as pd


class FeatureSetType(Enum):
    """特征集类型"""

    ALPHA158 = "alpha158"
    TECHNICAL_62 = "technical_62"
    CUSTOM = "custom"


# 62 个手工技术指标特征名（与体系A一致）
TECHNICAL_62_FEATURES: List[str] = [
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "momentum_short",
    "momentum_long",
    "momentum_reversal",
    "momentum_strength_5",
    "momentum_strength_10",
    "momentum_strength_20",
    "relative_strength_5d",
    "relative_strength_20d",
    "relative_momentum",
    "return_1d_rank",
    "return_5d_rank",
    "return_20d_rank",
    "volume_rank",
    "volatility_20_rank",
    "market_up_ratio",
    "ma_ratio_5",
    "ma_ratio_10",
    "ma_ratio_20",
    "ma_ratio_60",
    "ma_slope_5",
    "ma_slope_10",
    "ma_slope_20",
    "ma_alignment",
    "volatility_5",
    "volatility_20",
    "volatility_60",
    "vol_regime",
    "volatility_skew",
    "vol_ratio",
    "vol_ma_ratio",
    "vol_std",
    "vol_price_diverge",
    "rsi_6",
    "rsi_14",
    "rsi_diff",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_slope",
    "bb_position",
    "bb_width",
    "body",
    "wick_upper",
    "wick_lower",
    "range_pct",
    "consecutive_up",
    "consecutive_down",
    "price_pos_20",
    "price_pos_60",
    "dist_high_20",
    "dist_low_20",
    "dist_high_60",
    "dist_low_60",
    "atr_pct",
    "di_diff",
    "adx",
]

EPSILON = 1e-10


def get_feature_set_names(feature_set: FeatureSetType) -> List[str]:
    """获取特征集的特征名列表"""
    if feature_set == FeatureSetType.TECHNICAL_62:
        return list(TECHNICAL_62_FEATURES)
    return []


def compute_return_features(
    df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """计算收益率特征（6个）"""
    close = df["$close"]
    result: Dict[str, pd.Series] = {}
    for period in [1, 2, 3, 5, 10, 20]:
        result[f"return_{period}d"] = close.pct_change(period)
    return result


def compute_momentum_features(
    returns_dict: Dict[str, pd.Series],
    close: pd.Series,
) -> Dict[str, pd.Series]:
    """计算动量特征（6个）"""
    result: Dict[str, pd.Series] = {}
    result["momentum_short"] = returns_dict["return_5d"] - returns_dict["return_10d"]
    result["momentum_long"] = returns_dict["return_10d"] - returns_dict["return_20d"]
    result["momentum_reversal"] = -returns_dict["return_1d"]

    daily_returns = close.pct_change()
    for period in [5, 10, 20]:
        up_days = (daily_returns > 0).rolling(period).sum()
        result[f"momentum_strength_{period}"] = up_days / period
    return result


def compute_ma_features(close: pd.Series) -> Dict[str, pd.Series]:
    """计算移动平均特征（9个）"""
    result: Dict[str, pd.Series] = {}
    mas: Dict[int, pd.Series] = {}
    for window in [5, 10, 20, 60]:
        ma = close.rolling(window).mean()
        mas[window] = ma
        result[f"ma_ratio_{window}"] = close / ma - 1
        result[f"ma_slope_{window}"] = ma.pct_change(5)
    result["ma_alignment"] = (mas[5] > mas[10]).astype(int) + (
        mas[10] > mas[20]
    ).astype(int)
    return result


def compute_volatility_features(
    close: pd.Series,
) -> Dict[str, pd.Series]:
    """计算波动率特征（5个）"""
    returns = close.pct_change()
    result: Dict[str, pd.Series] = {}
    for window in [5, 20, 60]:
        result[f"volatility_{window}"] = returns.rolling(window).std()
    result["vol_regime"] = result["volatility_5"] / (result["volatility_20"] + EPSILON)
    result["volatility_skew"] = returns.rolling(20).skew()
    return result


def compute_volume_features(
    close: pd.Series,
    volume: pd.Series,
) -> Dict[str, pd.Series]:
    """计算成交量特征（4个）"""
    result: Dict[str, pd.Series] = {}
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    result["vol_ratio"] = volume / (vol_ma20 + 1)
    result["vol_ma_ratio"] = vol_ma5 / (vol_ma20 + 1)
    result["vol_std"] = volume.rolling(20).std() / (vol_ma20 + 1)
    price_up = (close > close.shift(1)).astype(int)
    vol_up = (volume > volume.shift(1)).astype(int)
    result["vol_price_diverge"] = (price_up != vol_up).astype(int)
    return result


def compute_rsi_features(close: pd.Series) -> Dict[str, pd.Series]:
    """计算RSI特征（3个）"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    result: Dict[str, pd.Series] = {}
    for period in [6, 14]:
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + EPSILON)
        result[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    result["rsi_diff"] = result["rsi_6"] - result["rsi_14"]
    return result


def compute_macd_features(close: pd.Series) -> Dict[str, pd.Series]:
    """计算MACD特征（4个）"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result: Dict[str, pd.Series] = {}
    result["macd"] = ema12 - ema26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    result["macd_hist_slope"] = result["macd_hist"].diff(3)
    return result


def compute_pattern_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
) -> Dict[str, pd.Series]:
    """计算价格形态特征（6个）"""
    result: Dict[str, pd.Series] = {}
    result["body"] = (close - open_) / (open_ + EPSILON)
    result["wick_upper"] = (high - np.maximum(close, open_)) / (high - low + EPSILON)
    result["wick_lower"] = (np.minimum(close, open_) - low) / (high - low + EPSILON)
    result["range_pct"] = (high - low) / (close + EPSILON)
    result["consecutive_up"] = (close > close.shift(1)).rolling(5).sum()
    result["consecutive_down"] = (close < close.shift(1)).rolling(5).sum()
    return result


def compute_position_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> Dict[str, pd.Series]:
    """计算价格位置 + ATR/ADX 特征（9个）"""
    result: Dict[str, pd.Series] = {}
    for window in [20, 60]:
        high_n = high.rolling(window).max()
        low_n = low.rolling(window).min()
        result[f"price_pos_{window}"] = (close - low_n) / (high_n - low_n + EPSILON)
        result[f"dist_high_{window}"] = (high_n - close) / (close + EPSILON)
        result[f"dist_low_{window}"] = (close - low_n) / (close + EPSILON)
    # ATR
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr_pct"] = tr.rolling(14).mean() / (close + EPSILON)
    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + EPSILON)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + EPSILON)
    result["di_diff"] = plus_di - minus_di
    result["adx"] = (plus_di - minus_di).abs().rolling(14).mean()
    return result


def compute_bollinger_features(
    close: pd.Series,
) -> Dict[str, pd.Series]:
    """计算布林带特征（2个）"""
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    return {
        "bb_position": (close - bb_mid) / (2 * bb_std + EPSILON),
        "bb_width": 4 * bb_std / (bb_mid + EPSILON),
    }
