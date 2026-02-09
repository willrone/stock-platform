"""
简化版 Alpha 因子计算

当 Qlib Alpha158 不可用时，提供基础因子计算功能。
"""

from typing import List

import numpy as np
import pandas as pd
from loguru import logger


def calculate_simplified_alpha_factors(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算简化版 Alpha 因子

    当 Qlib Alpha158 不可用时使用此函数计算基础因子。

    Args:
        data: 包含 OHLCV 数据的 DataFrame，列名需要有 $ 前缀

    Returns:
        包含计算出的因子的 DataFrame
    """
    if data.empty:
        return pd.DataFrame()

    # 确保数据有正确的列
    required_cols = ["$close", "$high", "$low", "$volume"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.warning(f"缺少必要列: {missing_cols}")
        return pd.DataFrame(index=data.index)

    factors = pd.DataFrame(index=data.index)

    try:
        # 价格收益率因子
        for period in [5, 10, 20, 30]:
            factors[f"RESI{period}"] = data["$close"].pct_change(periods=period)

        # 移动平均因子
        for period in [5, 10, 20, 30]:
            factors[f"MA{period}"] = data["$close"].rolling(period).mean()

        # 标准差因子
        for period in [5, 10, 20, 30]:
            factors[f"STD{period}"] = data["$close"].rolling(period).std()

        # 成交量标准差因子
        for period in [5, 10, 20, 30]:
            factors[f"VSTD{period}"] = data["$volume"].rolling(period).std()

        # 相关性因子（价格与成交量）
        for period in [5, 10, 20, 30]:
            factors[f"CORR{period}"] = (
                data["$close"].rolling(period).corr(data["$volume"])
            )

        # 最高价因子
        for period in [5, 10, 20, 30]:
            factors[f"MAX{period}"] = data["$high"].rolling(period).max()

        # 最低价因子
        for period in [5, 10, 20, 30]:
            factors[f"MIN{period}"] = data["$low"].rolling(period).min()

        # 分位数比率因子
        for period in [5, 10, 20, 30]:
            q80 = data["$close"].rolling(period).quantile(0.8)
            q20 = data["$close"].rolling(period).quantile(0.2)
            factors[f"QTLU{period}"] = q80 / (q20 + 1e-8)  # 避免除零

        # 填充无穷大和NaN值
        factors = factors.replace([np.inf, -np.inf], np.nan)
        factors = factors.ffill().fillna(0)

        logger.debug(f"计算了 {len(factors.columns)} 个简化 Alpha 因子")
        return factors

    except Exception as e:
        logger.error(f"简化 Alpha 因子计算失败: {e}")
        return pd.DataFrame(index=data.index)


def calculate_basic_factors_for_stock(
    stock_data: pd.DataFrame, stock_code: str
) -> pd.DataFrame:
    """
    为单个股票计算基础因子

    Args:
        stock_data: 单个股票的 OHLCV 数据
        stock_code: 股票代码

    Returns:
        包含计算出的因子的 DataFrame
    """
    try:
        # 确保数据有正确的列
        required_cols = ["$close", "$high", "$low", "$volume", "$open"]
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        if missing_cols:
            logger.warning(f"股票 {stock_code} 缺少必要列: {missing_cols}")
            return pd.DataFrame(index=stock_data.index)

        factors = pd.DataFrame(index=stock_data.index)

        close = stock_data["$close"]
        high = stock_data["$high"]
        low = stock_data["$low"]
        volume = stock_data["$volume"]
        open_ = stock_data["$open"]

        # 基础价格因子
        factors["KMID"] = (close - open_) / open_
        factors["KLEN"] = (high - low) / close
        factors["KUP"] = (high - close) / close
        factors["KLOW"] = (close - low) / close

        # 价格收益率（不同周期）
        for period in [1, 2, 3, 5, 10, 20, 30, 60]:
            factors[f"RET{period}"] = close.pct_change(period)

        # 移动平均
        for period in [5, 10, 20, 30, 60]:
            factors[f"MA{period}"] = close.rolling(period).mean()

        # 标准差
        for period in [5, 10, 20, 30, 60]:
            factors[f"STD{period}"] = close.rolling(period).std()

        # 最大值/最小值
        for period in [5, 10, 20, 30, 60]:
            factors[f"MAX{period}"] = close.rolling(period).max()
            factors[f"MIN{period}"] = close.rolling(period).min()

        # 量价相关性
        for period in [5, 10, 20, 30, 60]:
            factors[f"CORR{period}"] = close.rolling(period).corr(volume)

        # 成交量因子
        for period in [5, 10, 20, 30, 60]:
            factors[f"VMA{period}"] = volume.rolling(period).mean()
            factors[f"VSTD{period}"] = volume.rolling(period).std()

        # 填充缺失值
        factors = factors.bfill().fillna(0)

        logger.debug(f"股票 {stock_code} 计算了 {len(factors.columns)} 个基础因子")
        return factors

    except Exception as e:
        logger.error(f"计算股票 {stock_code} 的基础因子失败: {e}")
        return pd.DataFrame(index=stock_data.index)


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    计算 RSI 指标

    Args:
        series: 价格序列
        period: 计算周期

    Returns:
        RSI 值序列
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
