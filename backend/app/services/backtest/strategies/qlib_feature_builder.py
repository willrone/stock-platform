"""
Qlib 模型特征构建器

为回测策略中的 Qlib 模型（44 特征，Column_0~Column_43）生成
与训练时一致的特征。特征来源：
  - 基础 OHLCV + VWAP（6 列）
  - 技术指标（23 列）
  - 基本面特征（8 列）
  - 数据集准备阶段追加特征（7 列）
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 44 个特征的固定顺序（与训练时 Column_0~43 一一对应）
QLIB_FEATURE_ORDER = [
    "$open", "$high", "$low", "$close", "$volume", "$vwap",
    "OBV", "MA5", "KDJ_K", "KDJ_D", "KDJ_J", "MA10",
    "STOCH_K", "WILLIAMS_R", "RSI", "ATR", "STOCH_D", "MA20",
    "SMA", "EMA", "WMA", "CCI", "BOLLINGER_UPPER", "BOLLINGER_MIDDLE",
    "BOLLINGER_LOWER", "MACD", "MACD_SIGNAL", "MACD_HISTOGRAM", "MA60",
    "price_change", "price_change_5d", "price_change_20d",
    "volatility_5d", "volatility_20d", "volume_change",
    "volume_ma_ratio", "price_position",
    "RET1", "RET5", "RET20", "STD5", "STD20", "VOL1", "VOL5",
]

QLIB_FEATURE_COUNT = len(QLIB_FEATURE_ORDER)
EPSILON = 1e-10


def build_qlib_features(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """从 OHLCV 数据构建 44 个 Qlib 特征。

    Args:
        data: 包含 open/high/low/close/volume 列的 DataFrame

    Returns:
        包含 44 列特征的 DataFrame（列名为 Column_0~Column_43），
        行索引与输入一致；数据不足时返回 None。
    """
    if data is None or len(data) < 60:
        return None

    try:
        features = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        open_ = data["open"]
        volume = data["volume"]

        _add_base_ohlcv(features, open_, high, low, close, volume)
        _add_technical_indicators(features, close, high, low, volume)
        _add_fundamental_features(features, close, volume)
        _add_dataset_prep_features(features, close, volume)

        # 按训练时的固定顺序排列，缺失列填 0
        result = pd.DataFrame(index=data.index)
        for col_name in QLIB_FEATURE_ORDER:
            result[col_name] = features[col_name] if col_name in features else 0.0

        result = result.fillna(0)
        # 重命名为 Column_0 ~ Column_43
        result.columns = [f"Column_{i}" for i in range(QLIB_FEATURE_COUNT)]
        return result

    except Exception as e:
        logger.error(f"构建 Qlib 特征失败: {e}")
        return None


# ── 私有辅助函数 ─────────────────────────────────────────


def _add_base_ohlcv(
    feat: pd.DataFrame,
    open_: pd.Series, high: pd.Series,
    low: pd.Series, close: pd.Series, volume: pd.Series,
) -> None:
    """添加基础 OHLCV + VWAP（6 列）"""
    feat["$open"] = open_
    feat["$high"] = high
    feat["$low"] = low
    feat["$close"] = close
    feat["$volume"] = volume
    # VWAP 近似：(high + low + close) / 3
    feat["$vwap"] = (high + low + close) / 3


def _add_technical_indicators(
    feat: pd.DataFrame,
    close: pd.Series, high: pd.Series,
    low: pd.Series, volume: pd.Series,
) -> None:
    """添加技术指标（23 列）"""
    _calc_obv(feat, close, volume)
    _calc_ma(feat, close)
    _calc_kdj(feat, close, high, low)
    _calc_stochastic(feat, close, high, low)
    _calc_williams_r(feat, close, high, low)
    _calc_rsi(feat, close)
    _calc_atr(feat, close, high, low)
    _calc_ema_sma_wma(feat, close)
    _calc_cci(feat, close, high, low)
    _calc_bollinger(feat, close)
    _calc_macd(feat, close)


def _add_fundamental_features(
    feat: pd.DataFrame, close: pd.Series, volume: pd.Series,
) -> None:
    """添加基本面特征（8 列）"""
    feat["price_change"] = close.pct_change()
    feat["price_change_5d"] = close.pct_change(5)
    feat["price_change_20d"] = close.pct_change(20)
    feat["volatility_5d"] = close.pct_change().rolling(5).std()
    feat["volatility_20d"] = close.pct_change().rolling(20).std()
    feat["volume_change"] = volume.pct_change()
    feat["volume_ma_ratio"] = volume / (volume.rolling(20).mean() + EPSILON)
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    feat["price_position"] = (close - low_20) / (high_20 - low_20 + EPSILON)


def _add_dataset_prep_features(
    feat: pd.DataFrame, close: pd.Series, volume: pd.Series,
) -> None:
    """添加数据集准备阶段追加的特征（7 列）"""
    feat["RET1"] = close.pct_change(1)
    feat["RET5"] = close.pct_change(5)
    feat["RET20"] = close.pct_change(20)
    feat["STD5"] = close.rolling(5).std()
    feat["STD20"] = close.rolling(20).std()
    feat["VOL1"] = volume.pct_change(1)
    feat["VOL5"] = volume.pct_change(5)


# ── 技术指标计算 ─────────────────────────────────────────


def _calc_obv(feat: pd.DataFrame, close: pd.Series, volume: pd.Series) -> None:
    direction = np.sign(close.diff()).fillna(0)
    feat["OBV"] = (direction * volume).cumsum()


def _calc_ma(feat: pd.DataFrame, close: pd.Series) -> None:
    for w in [5, 10, 20, 60]:
        feat[f"MA{w}"] = close.rolling(w).mean()


def _calc_kdj(
    feat: pd.DataFrame, close: pd.Series,
    high: pd.Series, low: pd.Series,
) -> None:
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + EPSILON) * 100
    feat["KDJ_K"] = rsv.ewm(com=2, adjust=False).mean()
    feat["KDJ_D"] = feat["KDJ_K"].ewm(com=2, adjust=False).mean()
    feat["KDJ_J"] = 3 * feat["KDJ_K"] - 2 * feat["KDJ_D"]


def _calc_stochastic(
    feat: pd.DataFrame, close: pd.Series,
    high: pd.Series, low: pd.Series,
) -> None:
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    feat["STOCH_K"] = (close - low_14) / (high_14 - low_14 + EPSILON) * 100
    feat["STOCH_D"] = feat["STOCH_K"].rolling(3).mean()


def _calc_williams_r(
    feat: pd.DataFrame, close: pd.Series,
    high: pd.Series, low: pd.Series,
) -> None:
    high_14 = high.rolling(14).max()
    low_14 = low.rolling(14).min()
    feat["WILLIAMS_R"] = (high_14 - close) / (high_14 - low_14 + EPSILON) * -100


def _calc_rsi(feat: pd.DataFrame, close: pd.Series) -> None:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / (loss + EPSILON)
    feat["RSI"] = 100 - (100 / (1 + rs))


def _calc_atr(
    feat: pd.DataFrame, close: pd.Series,
    high: pd.Series, low: pd.Series,
) -> None:
    tr = pd.Series(
        np.maximum.reduce([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ]),
        index=close.index,
    )
    feat["ATR"] = tr.rolling(14).mean()


def _calc_ema_sma_wma(feat: pd.DataFrame, close: pd.Series) -> None:
    feat["SMA"] = close.rolling(20).mean()
    feat["EMA"] = close.ewm(span=20, adjust=False).mean()
    weights = np.arange(1, 21)
    feat["WMA"] = close.rolling(20).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True,
    )


def _calc_cci(
    feat: pd.DataFrame, close: pd.Series,
    high: pd.Series, low: pd.Series,
) -> None:
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(20).mean()
    tp_md = tp.rolling(20).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True,
    )
    feat["CCI"] = (tp - tp_ma) / (0.015 * tp_md + EPSILON)


def _calc_bollinger(feat: pd.DataFrame, close: pd.Series) -> None:
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    feat["BOLLINGER_UPPER"] = mid + 2 * std
    feat["BOLLINGER_MIDDLE"] = mid
    feat["BOLLINGER_LOWER"] = mid - 2 * std


def _calc_macd(feat: pd.DataFrame, close: pd.Series) -> None:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    feat["MACD"] = ema12 - ema26
    feat["MACD_SIGNAL"] = feat["MACD"].ewm(span=9, adjust=False).mean()
    feat["MACD_HISTOGRAM"] = feat["MACD"] - feat["MACD_SIGNAL"]
