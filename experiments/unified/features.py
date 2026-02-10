"""
特征工程模块

计算 62 个手工技术指标 + 截面特征 + 市场中性化
"""
from typing import List

import numpy as np
import pandas as pd

from .constants import (
    ATR_PERIOD,
    BB_WINDOW,
    EPSILON,
    MA_WINDOWS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MOMENTUM_STRENGTH_PERIODS,
    PRICE_POSITION_WINDOWS,
    RETURN_PERIODS,
    RSI_PERIODS,
    VOLATILITY_WINDOWS,
)


def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算收益率特征（6 个）"""
    close = df["close"]
    for period in RETURN_PERIODS:
        df[f"return_{period}d"] = close.pct_change(period)
    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算动量特征（6 个）"""
    df["momentum_short"] = df["return_5d"] - df["return_10d"]
    df["momentum_long"] = df["return_10d"] - df["return_20d"]
    df["momentum_reversal"] = -df["return_1d"]

    returns = df["close"].pct_change()
    for period in MOMENTUM_STRENGTH_PERIODS:
        up_days = (returns > 0).rolling(period).sum()
        df[f"momentum_strength_{period}"] = up_days / period
    return df


def compute_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算移动平均特征（9 个）"""
    close = df["close"]
    for window in MA_WINDOWS:
        ma = close.rolling(window).mean()
        df[f"ma_ratio_{window}"] = close / ma - 1
        df[f"ma_slope_{window}"] = ma.pct_change(5)

    ma_5 = close.rolling(5).mean()
    ma_10 = close.rolling(10).mean()
    ma_20 = close.rolling(20).mean()
    df["ma_alignment"] = (
        (ma_5 > ma_10).astype(int) + (ma_10 > ma_20).astype(int)
    )
    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算波动率特征（5 个）"""
    returns = df["close"].pct_change()
    for window in VOLATILITY_WINDOWS:
        df[f"volatility_{window}"] = returns.rolling(window).std()

    df["vol_regime"] = df["volatility_5"] / (df["volatility_20"] + EPSILON)
    df["volatility_skew"] = returns.rolling(20).skew()
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算成交量特征（4 个）"""
    volume = df["volume"]
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()

    df["vol_ratio"] = volume / (vol_ma20 + 1)
    df["vol_ma_ratio"] = vol_ma5 / (vol_ma20 + 1)
    df["vol_std"] = volume.rolling(20).std() / (vol_ma20 + 1)

    price_up = (df["close"] > df["close"].shift(1)).astype(int)
    vol_up = (volume > volume.shift(1)).astype(int)
    df["vol_price_diverge"] = (price_up != vol_up).astype(int)
    return df


def compute_rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 RSI 特征（3 个）"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    for period in RSI_PERIODS:
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + EPSILON)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    df["rsi_diff"] = df["rsi_6"] - df["rsi_14"]
    return df


def compute_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MACD 特征（4 个）"""
    close = df["close"]
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff(3)
    return df


def compute_bb_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算布林带特征（2 个）"""
    close = df["close"]
    bb_mid = close.rolling(BB_WINDOW).mean()
    bb_std = close.rolling(BB_WINDOW).std()
    df["bb_position"] = (close - bb_mid) / (2 * bb_std + EPSILON)
    df["bb_width"] = 4 * bb_std / (bb_mid + EPSILON)
    return df


def compute_price_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算价格形态特征（6 个）"""
    close, open_, high, low = df["close"], df["open"], df["high"], df["low"]
    df["body"] = (close - open_) / (open_ + EPSILON)
    df["wick_upper"] = (high - np.maximum(close, open_)) / (high - low + EPSILON)
    df["wick_lower"] = (np.minimum(close, open_) - low) / (high - low + EPSILON)
    df["range_pct"] = (high - low) / (close + EPSILON)
    df["consecutive_up"] = (close > close.shift(1)).rolling(5).sum()
    df["consecutive_down"] = (close < close.shift(1)).rolling(5).sum()
    return df


def compute_price_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算价格位置特征（6 个）"""
    close, high, low = df["close"], df["high"], df["low"]
    for window in PRICE_POSITION_WINDOWS:
        high_n = high.rolling(window).max()
        low_n = low.rolling(window).min()
        df[f"price_pos_{window}"] = (close - low_n) / (high_n - low_n + EPSILON)
        df[f"dist_high_{window}"] = (high_n - close) / (close + EPSILON)
        df[f"dist_low_{window}"] = (close - low_n) / (close + EPSILON)
    return df


def compute_atr_adx_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算 ATR 和 ADX 特征（3 个）"""
    close, high, low = df["close"], df["high"], df["low"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    df["atr_pct"] = df["atr"] / (close + EPSILON)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr14 = tr.rolling(ATR_PERIOD).mean()
    plus_di = 100 * plus_dm.rolling(ATR_PERIOD).mean() / (atr14 + EPSILON)
    minus_di = 100 * minus_dm.rolling(ATR_PERIOD).mean() / (atr14 + EPSILON)
    df["di_diff"] = plus_di - minus_di
    df["adx"] = (plus_di - minus_di).abs().rolling(ATR_PERIOD).mean()
    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算单只股票的全部 53 个时序特征"""
    df = df.copy().sort_values("date").reset_index(drop=True)

    feature_steps = [
        compute_return_features,
        compute_momentum_features,
        compute_ma_features,
        compute_volatility_features,
        compute_volume_features,
        compute_rsi_features,
        compute_macd_features,
        compute_bb_features,
        compute_price_pattern_features,
        compute_price_position_features,
        compute_atr_adx_features,
    ]
    for step in feature_steps:
        df = step(df)

    return df


def compute_regression_label(df: pd.DataFrame) -> pd.DataFrame:
    """计算回归标签：未来 1 天收益率"""
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1
    return df


def compute_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算截面特征（9 个）：排名 + 相对强度 + 市场状态"""
    rank_cols = ["return_1d", "return_5d", "return_20d", "volume", "volatility_20"]
    for col in rank_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True)

    df["market_up_ratio"] = df.groupby("date")["return_1d"].transform(
        lambda x: (x > 0).sum() / max(len(x), 1)
    )

    for period in ["5d", "20d"]:
        col = f"return_{period}"
        if col in df.columns:
            market_mean = df.groupby("date")[col].transform("mean")
            df[f"relative_strength_{period}"] = df[col] - market_mean

    if "relative_strength_5d" in df.columns:
        df["relative_momentum"] = (
            df["relative_strength_5d"] - df["relative_strength_20d"]
        )
    return df


def neutralize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    市场中性化处理

    对所有数值特征做截面去均值（减去当日截面均值），
    降低 market_up_ratio 等市场 beta 的过度影响。
    """
    feature_cols = get_feature_columns()
    numeric_features = [c for c in feature_cols if c in df.columns]

    for col in numeric_features:
        cross_mean = df.groupby("date")[col].transform("mean")
        df[f"{col}"] = df[col] - cross_mean

    return df


def get_feature_columns(include_fundamental: bool = False) -> List[str]:
    """
    获取特征列名列表

    Args:
        include_fundamental: 是否包含基本面因子（默认 False，保持向后兼容）

    Returns:
        62 个技术因子 + 可选 13 个基本面因子
    """
    technical = get_technical_feature_columns()

    if not include_fundamental:
        return technical

    from .fundamental_factors import get_fundamental_feature_names

    return technical + get_fundamental_feature_names()


def get_technical_feature_columns() -> List[str]:
    """获取 62 个技术因子列名（与策略推理保持一致）"""
    return [
        # 收益率 (6)
        "return_1d", "return_2d", "return_3d", "return_5d", "return_10d", "return_20d",
        # 动量 (6)
        "momentum_short", "momentum_long", "momentum_reversal",
        "momentum_strength_5", "momentum_strength_10", "momentum_strength_20",
        # 截面-相对强度 (3)
        "relative_strength_5d", "relative_strength_20d", "relative_momentum",
        # 截面-排名 (5)
        "return_1d_rank", "return_5d_rank", "return_20d_rank",
        "volume_rank", "volatility_20_rank",
        # 截面-市场状态 (1)
        "market_up_ratio",
        # 移动平均 (9)
        "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",
        "ma_slope_5", "ma_slope_10", "ma_slope_20", "ma_alignment",
        # 波动率 (5)
        "volatility_5", "volatility_20", "volatility_60", "vol_regime", "volatility_skew",
        # 成交量 (4)
        "vol_ratio", "vol_ma_ratio", "vol_std", "vol_price_diverge",
        # RSI (3)
        "rsi_6", "rsi_14", "rsi_diff",
        # MACD (4)
        "macd", "macd_signal", "macd_hist", "macd_hist_slope",
        # 布林带 (2)
        "bb_position", "bb_width",
        # 价格形态 (6)
        "body", "wick_upper", "wick_lower", "range_pct",
        "consecutive_up", "consecutive_down",
        # 价格位置 (6)
        "price_pos_20", "price_pos_60",
        "dist_high_20", "dist_low_20", "dist_high_60", "dist_low_60",
        # ATR/ADX (3)
        "atr_pct", "di_diff", "adx",
    ]
