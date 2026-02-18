"""
ML 特征适配器

根据模型的特征集类型，在回测时使用对应的特征计算方式。
桥接训练引擎的特征计算模块和回测策略。
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# 延迟导入避免循环依赖
_qlib_builder = None
_tech_computer = None
_indicator_calculator = None


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


def _get_indicator_calculator():
    """获取训练时使用的同一个 TechnicalIndicatorCalculator 实例"""
    global _indicator_calculator
    if _indicator_calculator is None:
        from app.services.prediction.technical_indicators import (
            TechnicalIndicatorCalculator,
        )

        _indicator_calculator = TechnicalIndicatorCalculator()
    return _indicator_calculator


def build_feature_matrix(
    data: pd.DataFrame,
    feature_set: str,
    feature_columns: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """根据特征集类型构建特征矩阵

    Args:
        data: OHLCV DataFrame
        feature_set: "alpha158" | "technical_62"
        feature_columns: 训练时保存的特征列名列表（优先使用）

    Returns:
        特征矩阵 (n_samples, n_features)，失败返回 None
    """
    if feature_columns:
        return _build_from_saved_columns(data, feature_columns)
    if feature_set == "alpha158":
        return _build_alpha158_matrix(data)
    if feature_set == "technical_62":
        return _build_technical62_matrix(data)
    _logger.warning(f"未知特征集: {feature_set}")
    return None


def build_feature_matrix_batch(
    df: pd.DataFrame,
    feature_set: str,
    feature_columns: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """批量构建特征矩阵（多股票）

    Args:
        df: 包含 stock_code 列的 DataFrame
        feature_set: "alpha158" | "technical_62"
        feature_columns: 训练时保存的特征列名列表（优先使用）

    Returns:
        特征矩阵，失败返回 None
    """
    if feature_columns:
        return _build_from_saved_columns_batch(df, feature_columns)
    if feature_set == "alpha158":
        return _build_alpha158_batch(df)
    if feature_set == "technical_62":
        return _build_technical62_batch(df)
    _logger.warning(f"未知特征集: {feature_set}")
    return None


def _build_alpha158_matrix(data: pd.DataFrame) -> Optional[np.ndarray]:
    """构建 alpha158 特征矩阵（33 个 Qlib 特征，与训练一致）"""
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
    from app.services.qlib.training.feature_sets import TECHNICAL_62_FEATURES

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
    from app.services.qlib.training.feature_sets import TECHNICAL_62_FEATURES

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
    df: pd.DataFrame,
    feature_names: List[str],
) -> np.ndarray:
    """按指定顺序提取特征列，缺失填 0"""
    result = pd.DataFrame(index=df.index)
    for name in feature_names:
        result[name] = df[name] if name in df.columns else 0.0
    return result.fillna(0).values



# ── 基于保存的 feature_columns 构建特征（复用训练管道） ──────────────────


def _compute_features_via_training_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """用纯 pandas 向量化计算技术指标，与训练时 TechnicalIndicatorCalculator 输出一致。

    参数已对齐训练代码 (technical_indicators.py TechnicalIndicatorCalculator):
    - KDJ: k_period=9, 平滑 K=(2/3)*prev_K+(1/3)*RSV, D=(2/3)*prev_D+(1/3)*K
    - STOCH: k_period=14, D=SMA(K, 3)
    - ATR: 首值 SMA(TR, 14)，后续指数平滑 (prev*(period-1)+TR)/period
    - SMA: period=20
    - EMA: period=20, 首值用 SMA 初始化
    - WMA: period=20

    内存优化：使用 float32 减少 ~50% 内存，用 ewm 替代 Python for-loop。
    """
    close = data["close"].astype(np.float32)
    high = data["high"].astype(np.float32)
    low = data["low"].astype(np.float32)
    open_ = data["open"].astype(np.float32)
    volume = data["volume"].astype(np.float32)

    features = pd.DataFrame(index=data.index)

    # ── 移动平均线 ──
    features["MA5"] = close.rolling(5).mean()
    features["MA10"] = close.rolling(10).mean()
    features["MA20"] = close.rolling(20).mean()
    features["MA60"] = close.rolling(60).mean()
    features["SMA"] = close.rolling(20).mean()  # 训练时 SMA 默认 period=20
    # EMA: 训练时 period=20 — 用 pandas ewm 替代 Python for-loop（内存+速度优化）
    features["EMA"] = close.ewm(span=20, adjust=False).mean()
    # WMA: 训练时 period=20 — 预计算权重避免每次 lambda 重建
    wma_period = 20
    _wma_weights = np.arange(1, wma_period + 1, dtype=np.float32)
    _wma_wsum = float(_wma_weights.sum())
    features["WMA"] = close.rolling(wma_period).apply(
        lambda x: np.dot(x, _wma_weights) / _wma_wsum, raw=True
    )

    # ── RSI (14) ──
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["RSI"] = 100 - (100 / (1 + rs))

    # ── KDJ (k_period=9，与训练时 calculate_kdj 一致) ──
    kdj_period = 9
    low_kdj = low.rolling(kdj_period).min()
    high_kdj = high.rolling(kdj_period).max()
    rsv = (close - low_kdj) / (high_kdj - low_kdj).replace(0, np.nan) * 100
    # K/D 平滑：K = (2/3)*prev_K + (1/3)*RSV 等价于 ewm(com=2, adjust=False)
    features["KDJ_K"] = rsv.ewm(com=2, adjust=False).mean()
    features["KDJ_D"] = features["KDJ_K"].ewm(com=2, adjust=False).mean()
    features["KDJ_J"] = 3 * features["KDJ_K"] - 2 * features["KDJ_D"]

    # ── Stochastic (k_period=14, d_period=3 SMA，与训练时 calculate_stochastic 一致) ──
    stoch_period = 14
    low_stoch = low.rolling(stoch_period).min()
    high_stoch = high.rolling(stoch_period).max()
    stoch_k = (close - low_stoch) / (high_stoch - low_stoch).replace(0, np.nan) * 100
    stoch_d = stoch_k.rolling(3).mean()  # D = SMA(K, 3)
    features["STOCH_K"] = stoch_k
    features["STOCH_D"] = stoch_d

    # ── Williams %R ──
    features["WILLIAMS_R"] = (high_stoch - close) / (high_stoch - low_stoch).replace(0, np.nan) * -100

    # ── MACD ──
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features["MACD"] = macd_line
    features["MACD_SIGNAL"] = signal_line
    features["MACD_HISTOGRAM"] = macd_line - signal_line

    # ── Bollinger Bands (20, 2) ──
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features["BOLLINGER_UPPER"] = bb_mid + 2 * bb_std
    features["BOLLINGER_MIDDLE"] = bb_mid
    features["BOLLINGER_LOWER"] = bb_mid - 2 * bb_std

    # ── ATR (14) - 用 ewm 近似 Wilder 指数平滑 ──
    atr_period = 14
    # 使用 np.maximum 替代 pd.concat().max() 避免 attrs 比较导致的 Series truth value 错误
    tr1 = (high - low).values
    tr2 = np.abs((high - close.shift(1)).values)
    tr3 = np.abs((low - close.shift(1)).values)
    tr = pd.Series(np.fmax(np.fmax(tr1, tr2), tr3), index=data.index)
    # ewm(span=14) 近似 Wilder 平滑 (prev*(n-1)+TR)/n
    features["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()

    # ── CCI (20) ──
    tp = (high + low + close) / 3
    tp_ma = tp.rolling(20).mean()
    tp_md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    features["CCI"] = (tp - tp_ma) / (0.015 * tp_md.replace(0, np.nan))

    # ── OBV ──
    obv_sign = np.sign(close.diff()).fillna(0)
    features["OBV"] = (obv_sign * volume).cumsum()

    # ── 价格变动 ──
    features["price_change"] = close.pct_change(1)
    features["price_change_5d"] = close.pct_change(5)
    features["price_change_20d"] = close.pct_change(20)

    # ── 波动率 ──
    features["volatility_5d"] = close.pct_change().rolling(5).std()
    features["volatility_20d"] = close.pct_change().rolling(20).std()

    # ── 成交量 ──
    features["volume_change"] = volume.pct_change(1)
    vol_ma5 = volume.rolling(5).mean()
    features["volume_ma_ratio"] = volume / vol_ma5.replace(0, np.nan)

    # ── 价格位置 ──
    features["price_position"] = (close - low_stoch) / (high_stoch - low_stoch).replace(0, np.nan)

    # ── STD ──
    features["STD5"] = close.rolling(5).std()
    features["STD20"] = close.rolling(20).std()

    # ── 原始 OHLCV ──
    features["open"] = open_.values
    features["high"] = high.values
    features["low"] = low.values
    features["close"] = close.values
    features["volume"] = volume.values
    if "adj_close" in data.columns:
        features["adj_close"] = data["adj_close"].astype(np.float32).values

    _logger.info(f"向量化技术指标计算完成: {len(features.columns)} 列")
    return features


def _build_from_saved_columns(
    data: pd.DataFrame,
    feature_columns: List[str],
) -> Optional[np.ndarray]:
    """根据训练时保存的特征列名构建特征矩阵（单只股票）

    使用训练时的同一个 TechnicalIndicatorCalculator 计算特征，
    确保推理特征与训练完全一致。
    """
    if data is None or (hasattr(data, '__len__') and len(data) < 60):
        return None
    try:
        # 确保传入的是 DataFrame
        if isinstance(data, pd.Series):
            _logger.warning("_build_from_saved_columns 收到 Series，跳过")
            return None
        all_features = _compute_features_via_training_pipeline(data)
        result = _extract_ordered_features(all_features, feature_columns)
        matched = sum(1 for c in feature_columns if c in all_features.columns)
        _logger.info(
            f"基于训练管道构建特征: 需要 {len(feature_columns)} 列, "
            f"匹配 {matched}/{len(feature_columns)}"
        )
        if matched < len(feature_columns):
            missing = [c for c in feature_columns if c not in all_features.columns]
            _logger.warning(f"缺失特征列（填0）: {missing}")
        return result
    except Exception as e:
        _logger.error(f"基于训练管道构建特征失败: {e}", exc_info=True)
        return None


def _build_from_saved_columns_batch(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Optional[np.ndarray]:
    """根据训练时保存的特征列名批量构建特征矩阵（多股票）"""
    all_parts: List[np.ndarray] = []

    for stock_code, stock_df in df.groupby("stock_code"):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        if len(stock_df) < 60:
            continue
        matrix = _build_from_saved_columns(stock_df, feature_columns)
        if matrix is not None:
            all_parts.append(matrix)

    if not all_parts:
        _logger.warning("基于训练管道批量特征构建失败")
        return None

    combined = np.vstack(all_parts)
    _logger.info(f"基于训练管道批量特征矩阵: {combined.shape}")
    return combined
