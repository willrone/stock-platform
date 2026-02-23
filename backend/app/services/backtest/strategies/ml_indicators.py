"""
ML 策略指标计算

为 ML 集成策略的 fallback 和风控提供技术指标计算。
复用训练引擎的特征计算模块，避免重复代码。
"""

from typing import Dict

import pandas as pd

EPSILON = 1e-10


def compute_strategy_indicators(
    data: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """计算策略所需的技术指标（用于 fallback 和风控）

    Args:
        data: 包含 close/high/low/open/volume 的 DataFrame

    Returns:
        指标名 → Series 的字典
    """
    if data is None or len(data) == 0:
        return {}

    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    returns = close.pct_change()

    indicators: Dict[str, pd.Series] = {}
    _add_return_features(indicators, close, returns)
    _add_momentum_features(indicators, returns)
    _add_rsi_features(indicators, close)
    _add_macd_features(indicators, close)
    _add_volatility_features(indicators, returns)
    _add_volume_features(indicators, close, volume)
    _add_bollinger_features(indicators, close)
    _add_ma_features(indicators, close)
    return indicators


def _add_return_features(
    ind: Dict[str, pd.Series], close: pd.Series, returns: pd.Series,
) -> None:
    """收益率特征"""
    for p in [1, 2, 3, 5, 10, 20]:
        ind[f"return_{p}d"] = close.pct_change(p)


def _add_momentum_features(
    ind: Dict[str, pd.Series], returns: pd.Series,
) -> None:
    """动量特征"""
    ind["momentum_short"] = ind.get("return_5d", returns) - ind.get("return_10d", returns)


def _add_rsi_features(
    ind: Dict[str, pd.Series], close: pd.Series,
) -> None:
    """RSI 特征"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    for p in [6, 14]:
        avg_gain = gain.ewm(span=p, adjust=False).mean()
        avg_loss = loss.ewm(span=p, adjust=False).mean()
        rs = avg_gain / (avg_loss + EPSILON)
        ind[f"rsi_{p}"] = 100 - (100 / (1 + rs))
    ind["rsi_diff"] = ind["rsi_6"] - ind["rsi_14"]


def _add_macd_features(
    ind: Dict[str, pd.Series], close: pd.Series,
) -> None:
    """MACD 特征"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ind["macd"] = ema12 - ema26
    ind["macd_signal"] = ind["macd"].ewm(span=9, adjust=False).mean()
    ind["macd_hist"] = ind["macd"] - ind["macd_signal"]
    ind["macd_hist_slope"] = ind["macd_hist"].diff(3)


def _add_volatility_features(
    ind: Dict[str, pd.Series], returns: pd.Series,
) -> None:
    """波动率特征"""
    for w in [5, 20, 60]:
        ind[f"volatility_{w}"] = returns.rolling(w).std()


def _add_volume_features(
    ind: Dict[str, pd.Series], close: pd.Series, volume: pd.Series,
) -> None:
    """成交量特征"""
    vol_ma20 = volume.rolling(20).mean()
    ind["vol_ratio"] = volume / (vol_ma20 + 1)


def _add_bollinger_features(
    ind: Dict[str, pd.Series], close: pd.Series,
) -> None:
    """布林带特征"""
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    ind["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + EPSILON)


def _add_ma_features(
    ind: Dict[str, pd.Series], close: pd.Series,
) -> None:
    """移动平均特征"""
    mas = {}
    for w in [5, 10, 20]:
        mas[w] = close.rolling(w).mean()
    ind["ma_alignment"] = (
        (mas[5] > mas[10]).astype(int) + (mas[10] > mas[20]).astype(int)
    )
