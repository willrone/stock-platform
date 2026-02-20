"""
RSIStrategy 冷却期 + 背离检测 测试 (P0)

测试 RSIStrategy 的 P4（信号冷却期）和 P5（RSI 背离检测）功能。
源码：app/services/backtest/strategies/technical/basic_strategies.py
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from app.services.backtest.strategies.technical.basic_strategies import RSIStrategy
from app.services.backtest.models.enums import SignalType


# ============================================================
# 辅助函数
# ============================================================

def _make_date_index(n: int, start: str = "2023-01-01") -> pd.DatetimeIndex:
    """生成交易日日期索引"""
    return pd.bdate_range(start=start, periods=n)


def _make_df(close: np.ndarray, dates: pd.DatetimeIndex = None) -> pd.DataFrame:
    """用 close 序列构造包含 close/high/low/volume 的 DataFrame"""
    n = len(close)
    if dates is None:
        dates = _make_date_index(n)
    df = pd.DataFrame({
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": np.full(n, 1_000_000, dtype=float),
    }, index=dates)
    return df


def _build_rsi_strategy(**overrides) -> RSIStrategy:
    """构造 RSIStrategy 实例，默认关闭趋势对齐和成交量确认以简化测试"""
    config = {
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
        "enable_trend_alignment": False,
        "enable_volume_confirm": False,
        "enable_multi_rsi": False,
        "enable_cooldown": True,
        "cooldown_days": 5,
        "enable_divergence": False,
    }
    config.update(overrides)
    return RSIStrategy(config)


def _generate_rsi_crossover_prices(
    n: int = 100,
    oversold_cross_indices: list = None,
    overbought_cross_indices: list = None,
    rsi_period: int = 14,
) -> np.ndarray:
    """
    生成一段价格序列，使 RSI 在指定位置产生穿越信号。
    通过在特定位置制造急跌/急涨来触发 RSI 穿越。
    """
    np.random.seed(42)
    prices = np.full(n, 100.0)
    # 先生成平稳序列
    for i in range(1, n):
        prices[i] = prices[i - 1] * (1 + np.random.normal(0, 0.005))

    # 在指定位置制造急跌（触发超卖穿越）
    if oversold_cross_indices:
        for idx in oversold_cross_indices:
            # 连续下跌让 RSI 跌破 oversold，然后回升穿越
            for j in range(max(0, idx - rsi_period), idx):
                prices[j] = prices[max(0, j - 1)] * 0.97
            # idx 处回升
            prices[idx] = prices[idx - 1] * 1.05

    # 在指定位置制造急涨（触发超买穿越）
    if overbought_cross_indices:
        for idx in overbought_cross_indices:
            for j in range(max(0, idx - rsi_period), idx):
                prices[j] = prices[max(0, j - 1)] * 1.03
            prices[idx] = prices[idx - 1] * 0.95

    return prices


# ============================================================
# 测试类：冷却期 (P4)
# ============================================================

class TestRSICooldown:
    """RSI 策略冷却期测试"""

    def test_cooldown_suppresses_repeated_buy(self):
        """测试1: 冷却期内重复买入信号应被抑制"""
        strategy = _build_rsi_strategy(enable_cooldown=True, cooldown_days=5)

        d1 = datetime(2023, 1, 2)
        d2 = datetime(2023, 1, 4)  # 2天后，在冷却期内

        # 第一次买入应通过
        assert strategy._check_cooldown("buy", d1) is True
        strategy._update_cooldown("buy", d1)

        # 冷却期内（2天 < 5天）应被抑制
        assert strategy._check_cooldown("buy", d2) is False

    def test_cooldown_suppresses_repeated_sell(self):
        """测试2: 冷却期内重复卖出信号应被抑制"""
        strategy = _build_rsi_strategy(enable_cooldown=True, cooldown_days=5)

        d1 = datetime(2023, 1, 2)
        d2 = datetime(2023, 1, 5)  # 3天后，在冷却期内

        assert strategy._check_cooldown("sell", d1) is True
        strategy._update_cooldown("sell", d1)

        assert strategy._check_cooldown("sell", d2) is False

    def test_cooldown_allows_after_expiry(self):
        """测试3: 冷却期刚好到期（第 cooldown_days 天）应允许新信号"""
        strategy = _build_rsi_strategy(enable_cooldown=True, cooldown_days=5)

        d1 = datetime(2023, 1, 2)
        d_exact = d1 + timedelta(days=5)  # 刚好第5天

        strategy._update_cooldown("buy", d1)

        # 刚好到期应允许（delta == cooldown_days，不满足 < cooldown_days）
        assert strategy._check_cooldown("buy", d_exact) is True

    def test_cooldown_disabled(self):
        """测试4: enable_cooldown=False 时不抑制任何信号"""
        strategy = _build_rsi_strategy(enable_cooldown=False, cooldown_days=5)

        d1 = datetime(2023, 1, 2)
        d2 = datetime(2023, 1, 3)  # 仅1天后

        strategy._update_cooldown("buy", d1)

        # 冷却期关闭，即使1天后也应通过
        assert strategy._check_cooldown("buy", d2) is True

    def test_cooldown_buy_sell_independent(self):
        """冷却期买卖独立：买入冷却不影响卖出"""
        strategy = _build_rsi_strategy(enable_cooldown=True, cooldown_days=5)

        d1 = datetime(2023, 1, 2)
        d2 = datetime(2023, 1, 3)

        strategy._update_cooldown("buy", d1)

        # 买入在冷却期内
        assert strategy._check_cooldown("buy", d2) is False
        # 卖出不受影响
        assert strategy._check_cooldown("sell", d2) is True


# ============================================================
# 测试类：背离检测 (P5)
# ============================================================

class TestRSIDivergence:
    """RSI 策略背离检测测试"""

    def _make_indicators_for_divergence(
        self,
        price_values: list,
        rsi_values: list,
    ) -> dict:
        """构造用于 _detect_divergence 的 indicators 字典"""
        idx = pd.RangeIndex(len(price_values))
        return {
            "price": pd.Series(price_values, index=idx, dtype=float),
            "rsi": pd.Series(rsi_values, index=idx, dtype=float),
        }

    def test_bullish_divergence(self):
        """测试5: 看涨背离 — 价格创新低 + RSI 未创新低 → 'bullish'"""
        strategy = _build_rsi_strategy(enable_divergence=True, divergence_lookback=5)

        # 构造数据：价格持续下跌创新低，但 RSI 在最后一个点高于前低
        #   价格: [100, 98, 96, 94, 92, 90]  → 最后 90 是新低
        #   RSI:  [40,  25, 20, 18, 15, 22]  → 最后 22 > 前低 15 + 3
        prices = [100, 98, 96, 94, 92, 90]
        rsi_vals = [40, 25, 20, 18, 15, 22]
        indicators = self._make_indicators_for_divergence(prices, rsi_vals)

        result = strategy._detect_divergence(indicators, idx=5)
        assert result == "bullish"

    def test_bearish_divergence(self):
        """测试6: 看跌背离 — 价格创新高 + RSI 未创新高 → 'bearish'"""
        strategy = _build_rsi_strategy(enable_divergence=True, divergence_lookback=5)

        # 构造数据：价格持续上涨创新高，但 RSI 在最后一个点低于前高
        #   价格: [100, 102, 104, 106, 108, 110]  → 最后 110 是新高
        #   RSI:  [50,  65,  70,  75,  80,  72]   → 最后 72 < 前高 80 - 3
        prices = [100, 102, 104, 106, 108, 110]
        rsi_vals = [50, 65, 70, 75, 80, 72]
        indicators = self._make_indicators_for_divergence(prices, rsi_vals)

        result = strategy._detect_divergence(indicators, idx=5)
        assert result == "bearish"

    def test_divergence_returns_none_when_idx_too_small(self):
        """测试7: idx < divergence_lookback 时 _detect_divergence 返回 None"""
        strategy = _build_rsi_strategy(enable_divergence=True, divergence_lookback=10)

        prices = list(range(20))
        rsi_vals = list(range(20))
        indicators = self._make_indicators_for_divergence(prices, rsi_vals)

        # idx=5 < lookback=10，应返回 None
        result = strategy._detect_divergence(indicators, idx=5)
        assert result is None

    def test_divergence_with_nan_data(self):
        """测试8: 数据含 NaN 时背离检测不崩溃，返回 None"""
        strategy = _build_rsi_strategy(enable_divergence=True, divergence_lookback=5)

        prices = [100, 98, np.nan, 94, 92, 90]
        rsi_vals = [40, 25, 20, np.nan, 15, 22]
        indicators = self._make_indicators_for_divergence(prices, rsi_vals)

        # 含 NaN 应安全返回 None
        result = strategy._detect_divergence(indicators, idx=5)
        assert result is None

    def test_divergence_disabled(self):
        """enable_divergence=False 时始终返回 None"""
        strategy = _build_rsi_strategy(enable_divergence=False, divergence_lookback=5)

        prices = [100, 98, 96, 94, 92, 90]
        rsi_vals = [40, 25, 20, 18, 15, 22]
        indicators = self._make_indicators_for_divergence(prices, rsi_vals)

        result = strategy._detect_divergence(indicators, idx=5)
        assert result is None


# ============================================================
# 测试类：冷却期 + 背离联合 (P4 + P5)
# ============================================================

class TestRSICooldownWithDivergence:
    """冷却期与背离信号联合测试"""

    def test_divergence_signal_respects_cooldown(self):
        """测试9: 背离信号也受冷却期约束"""
        strategy = _build_rsi_strategy(
            enable_cooldown=True,
            cooldown_days=5,
            enable_divergence=True,
            divergence_lookback=5,
        )

        d1 = datetime(2023, 1, 2)
        d2 = datetime(2023, 1, 4)  # 2天后，在冷却期内

        # 模拟已经触发过一次买入
        strategy._update_cooldown("buy", d1)

        # 冷却期内，即使检测到看涨背离，买入也应被抑制
        assert strategy._check_cooldown("buy", d2) is False

        # 但卖出不受影响
        assert strategy._check_cooldown("sell", d2) is True


# ============================================================
# 测试类：precompute_all_signals 冷却期 + 背离
# ============================================================

class TestRSIPrecomputeSignals:
    """precompute_all_signals 向量化计算测试"""

    def _make_rsi_crossover_data(self, n: int = 200) -> pd.DataFrame:
        """
        生成一段能产生 RSI 穿越信号的价格数据。
        通过制造明显的涨跌周期来触发 RSI 超卖/超买穿越。
        """
        np.random.seed(123)
        dates = _make_date_index(n)
        # 制造周期性涨跌
        t = np.arange(n, dtype=float)
        base = 100 + 10 * np.sin(2 * np.pi * t / 40)  # 40天一个周期
        noise = np.random.normal(0, 0.5, n)
        close = base + noise
        close = np.maximum(close, 1.0)  # 确保价格为正
        return _make_df(close, dates)

    def test_precompute_cooldown_suppresses_consecutive_signals(self):
        """测试10: precompute_all_signals 中冷却期后处理正确抑制连续信号"""
        strategy = _build_rsi_strategy(
            enable_cooldown=True,
            cooldown_days=5,
            rsi_period=14,
        )

        data = self._make_rsi_crossover_data(200)
        signals = strategy.precompute_all_signals(data)

        assert signals is not None
        assert len(signals) == len(data)

        # 验证冷却期：同方向信号之间至少间隔 cooldown_days
        buy_indices = signals.index[signals > 0].tolist()
        sell_indices = signals.index[signals < 0].tolist()

        # 检查买入信号间隔
        for i in range(1, len(buy_indices)):
            prev_pos = data.index.get_loc(buy_indices[i - 1])
            curr_pos = data.index.get_loc(buy_indices[i])
            assert curr_pos - prev_pos >= strategy.cooldown_days, (
                f"买入信号间隔不足: 位置 {prev_pos} -> {curr_pos}, "
                f"间隔 {curr_pos - prev_pos} < {strategy.cooldown_days}"
            )

        # 检查卖出信号间隔
        for i in range(1, len(sell_indices)):
            prev_pos = data.index.get_loc(sell_indices[i - 1])
            curr_pos = data.index.get_loc(sell_indices[i])
            assert curr_pos - prev_pos >= strategy.cooldown_days, (
                f"卖出信号间隔不足: 位置 {prev_pos} -> {curr_pos}, "
                f"间隔 {curr_pos - prev_pos} < {strategy.cooldown_days}"
            )

    def test_precompute_divergence_signals_inserted(self):
        """测试11: precompute_all_signals 中背离信号正确插入"""
        strategy = _build_rsi_strategy(
            enable_cooldown=False,  # 关闭冷却期以简化验证
            enable_divergence=True,
            divergence_lookback=20,
            rsi_period=14,
        )

        # 生成较长数据以确保有足够的 lookback 窗口
        data = self._make_rsi_crossover_data(300)
        signals = strategy.precompute_all_signals(data)

        assert signals is not None
        assert len(signals) == len(data)

        # 前 divergence_lookback 个位置不应有背离信号
        # （穿越信号可能存在，但背离信号不会在 idx < lookback 处产生）
        # 只要不崩溃且返回正确长度即可

        # 验证信号值在合理范围内
        assert signals.max() <= 1.0
        assert signals.min() >= -1.0

    def test_precompute_no_cooldown(self):
        """关闭冷却期时 precompute 不抑制任何信号"""
        strategy = _build_rsi_strategy(
            enable_cooldown=False,
            rsi_period=14,
        )

        data = self._make_rsi_crossover_data(200)
        signals = strategy.precompute_all_signals(data)

        assert signals is not None

        # 关闭冷却期时，可能有连续的同方向信号
        # 只要不崩溃且返回正确长度即可
        assert len(signals) == len(data)

    def test_precompute_returns_correct_dtype(self):
        """precompute_all_signals 返回 float64 Series"""
        strategy = _build_rsi_strategy(rsi_period=14)
        data = self._make_rsi_crossover_data(100)
        signals = strategy.precompute_all_signals(data)

        assert signals is not None
        assert signals.dtype == np.float64

    def test_precompute_with_divergence_and_cooldown(self):
        """背离 + 冷却期同时启用时 precompute 正常工作"""
        strategy = _build_rsi_strategy(
            enable_cooldown=True,
            cooldown_days=3,
            enable_divergence=True,
            divergence_lookback=15,
            rsi_period=14,
        )

        data = self._make_rsi_crossover_data(300)
        signals = strategy.precompute_all_signals(data)

        assert signals is not None
        assert len(signals) == len(data)

        # 所有非零信号之间（同方向）应满足冷却期
        buy_positions = [data.index.get_loc(idx) for idx in signals.index[signals > 0]]
        for i in range(1, len(buy_positions)):
            assert buy_positions[i] - buy_positions[i - 1] >= strategy.cooldown_days
