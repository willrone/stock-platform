"""
策略边界条件测试 (P5)

测试各策略在极端输入下的鲁棒性：
- 空 DataFrame
- 全 NaN 价格数据
- 只有 1 行数据
- 极端参数（rsi_period=1）

源码：app/services/backtest/strategies/technical/basic_strategies.py
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from app.services.backtest.strategies.technical.basic_strategies import (
    RSIStrategy,
    MovingAverageStrategy,
    MACDStrategy,
)


def _make_df(close, n=None, start="2023-01-01"):
    """构造标准 DataFrame"""
    if isinstance(close, (list, np.ndarray)):
        close = np.array(close, dtype=float)
        n = len(close)
    else:
        close = np.full(n, close, dtype=float)
    dates = pd.bdate_range(start=start, periods=n)
    return pd.DataFrame({
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": np.full(n, 1_000_000, dtype=float),
    }, index=dates)


def _build_rsi(**overrides):
    config = {
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
        "enable_trend_alignment": False,
        "enable_volume_confirm": False,
        "enable_multi_rsi": False,
        "enable_cooldown": False,
        "enable_divergence": False,
    }
    config.update(overrides)
    return RSIStrategy(config)


def _build_ma(**overrides):
    config = {
        "short_window": 5,
        "long_window": 20,
        "enable_trend_filter": False,
    }
    config.update(overrides)
    return MovingAverageStrategy(config)


def _build_macd(**overrides):
    config = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    }
    config.update(overrides)
    return MACDStrategy(config)


# ============================================================
# 空 DataFrame 测试
# ============================================================

class TestEmptyDataFrame:
    """空 DataFrame 输入不崩溃"""

    def _empty_df(self):
        return pd.DataFrame(
            columns=["close", "high", "low", "volume"],
            dtype=float,
        )

    def test_rsi_calculate_indicators_empty(self):
        """RSI calculate_indicators 空数据不崩溃"""
        strategy = _build_rsi()
        df = self._empty_df()
        indicators = strategy.calculate_indicators(df)
        assert "rsi" in indicators
        assert len(indicators["rsi"]) == 0

    def test_rsi_precompute_empty(self):
        """RSI precompute_all_signals 空数据不崩溃"""
        strategy = _build_rsi()
        df = self._empty_df()
        result = strategy.precompute_all_signals(df)
        # 可能返回 None 或空 Series，都可接受
        if result is not None:
            assert len(result) == 0

    def test_ma_calculate_indicators_empty(self):
        """MA calculate_indicators 空数据不崩溃"""
        strategy = _build_ma()
        df = self._empty_df()
        indicators = strategy.calculate_indicators(df)
        assert "sma_short" in indicators

    def test_macd_calculate_indicators_empty(self):
        """MACD calculate_indicators 空数据不崩溃"""
        strategy = _build_macd()
        df = self._empty_df()
        indicators = strategy.calculate_indicators(df)
        assert "macd" in indicators


# ============================================================
# 全 NaN 价格数据测试
# ============================================================

class TestAllNaNData:
    """全 NaN 价格数据不崩溃"""

    def _nan_df(self, n=50):
        dates = pd.bdate_range(start="2023-01-01", periods=n)
        return pd.DataFrame({
            "close": np.full(n, np.nan),
            "high": np.full(n, np.nan),
            "low": np.full(n, np.nan),
            "volume": np.full(n, np.nan),
        }, index=dates)

    def test_rsi_indicators_all_nan(self):
        """RSI 全 NaN 数据计算指标不崩溃"""
        strategy = _build_rsi()
        df = self._nan_df()
        indicators = strategy.calculate_indicators(df)
        # RSI 应全为 NaN
        assert indicators["rsi"].isna().all()

    def test_rsi_precompute_all_nan(self):
        """RSI 全 NaN 数据 precompute 不崩溃"""
        strategy = _build_rsi()
        df = self._nan_df()
        result = strategy.precompute_all_signals(df)
        # 不崩溃即可；可能返回 None 或全 0 的 Series
        if result is not None:
            assert len(result) == len(df)

    def test_rsi_generate_signals_all_nan(self):
        """RSI 全 NaN 数据 generate_signals 不崩溃"""
        strategy = _build_rsi()
        df = self._nan_df()
        date = df.index[20]
        signals = strategy.generate_signals(df, date)
        assert isinstance(signals, list)

    def test_ma_precompute_all_nan(self):
        """MA 全 NaN 数据 precompute 不崩溃"""
        strategy = _build_ma()
        df = self._nan_df()
        result = strategy.precompute_all_signals(df)
        if result is not None:
            assert len(result) == len(df)

    def test_macd_precompute_all_nan(self):
        """MACD 全 NaN 数据 precompute 不崩溃"""
        strategy = _build_macd()
        df = self._nan_df()
        result = strategy.precompute_all_signals(df)
        if result is not None:
            assert len(result) == len(df)


# ============================================================
# 只有 1 行数据测试
# ============================================================

class TestSingleRowData:
    """只有 1 行数据不崩溃"""

    def _single_df(self):
        return _make_df([100.0])

    def test_rsi_single_row(self):
        """RSI 单行数据不崩溃"""
        strategy = _build_rsi()
        df = self._single_df()
        indicators = strategy.calculate_indicators(df)
        assert len(indicators["rsi"]) == 1

    def test_rsi_precompute_single_row(self):
        """RSI precompute 单行数据不崩溃"""
        strategy = _build_rsi()
        df = self._single_df()
        result = strategy.precompute_all_signals(df)
        if result is not None:
            assert len(result) == 1

    def test_rsi_generate_signals_single_row(self):
        """RSI generate_signals 单行数据不崩溃"""
        strategy = _build_rsi()
        df = self._single_df()
        signals = strategy.generate_signals(df, df.index[0])
        assert isinstance(signals, list)
        # 数据不足，不应产生信号
        assert len(signals) == 0

    def test_ma_single_row(self):
        """MA 单行数据不崩溃"""
        strategy = _build_ma()
        df = self._single_df()
        indicators = strategy.calculate_indicators(df)
        assert len(indicators["sma_short"]) == 1

    def test_macd_single_row(self):
        """MACD 单行数据不崩溃"""
        strategy = _build_macd()
        df = self._single_df()
        indicators = strategy.calculate_indicators(df)
        assert len(indicators["macd"]) == 1


# ============================================================
# 极端参数测试
# ============================================================

class TestExtremeParameters:
    """极端参数不崩溃"""

    def test_rsi_period_1(self):
        """rsi_period=1 不崩溃"""
        strategy = _build_rsi(rsi_period=1)
        df = _make_df(np.random.RandomState(42).uniform(90, 110, 50))
        indicators = strategy.calculate_indicators(df)
        # rsi_period=1 应能计算出 RSI
        assert len(indicators["rsi"]) == 50

    def test_rsi_period_1_precompute(self):
        """rsi_period=1 precompute 不崩溃"""
        strategy = _build_rsi(rsi_period=1)
        df = _make_df(np.random.RandomState(42).uniform(90, 110, 50))
        result = strategy.precompute_all_signals(df)
        if result is not None:
            assert len(result) == 50

    def test_rsi_period_1_generate_signals(self):
        """rsi_period=1 generate_signals 不崩溃"""
        strategy = _build_rsi(rsi_period=1)
        df = _make_df(np.random.RandomState(42).uniform(90, 110, 50))
        # 取中间位置的日期
        signals = strategy.generate_signals(df, df.index[25])
        assert isinstance(signals, list)

    def test_rsi_very_large_period(self):
        """rsi_period 大于数据长度不崩溃"""
        strategy = _build_rsi(rsi_period=100)
        df = _make_df(np.random.RandomState(42).uniform(90, 110, 50))
        indicators = strategy.calculate_indicators(df)
        # 数据不足，RSI 应全为 NaN
        assert indicators["rsi"].isna().all()

    def test_ma_short_equals_long(self):
        """MA 短期 == 长期窗口不崩溃"""
        strategy = _build_ma(short_window=10, long_window=10)
        df = _make_df(np.random.RandomState(42).uniform(90, 110, 50))
        result = strategy.precompute_all_signals(df)
        if result is not None:
            assert len(result) == 50

    def test_rsi_thresholds_inverted(self):
        """RSI 超卖阈值 > 超买阈值时不崩溃（虽然无意义）"""
        strategy = _build_rsi(oversold_threshold=80, overbought_threshold=20)
        df = _make_df(np.random.RandomState(42).uniform(90, 110, 50))
        result = strategy.precompute_all_signals(df)
        if result is not None:
            assert len(result) == 50

    def test_divergence_lookback_larger_than_data(self):
        """divergence_lookback 大于数据长度时不崩溃"""
        strategy = _build_rsi(
            enable_divergence=True,
            divergence_lookback=100,
        )
        df = _make_df(np.random.RandomState(42).uniform(90, 110, 50))
        result = strategy.precompute_all_signals(df)
        if result is not None:
            assert len(result) == 50
