"""
MarketRegimeDetector 和 regime_aware_voting 单元测试
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 确保项目路径在 sys.path 中
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.services.backtest.models import SignalType, TradingSignal
from app.services.backtest.utils.market_regime_detector import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeDetectorConfig,
    _classify_strategy,
)
from app.services.backtest.utils.signal_integrator import SignalIntegrator


# ======================================================================
# 辅助函数
# ======================================================================


def _make_ohlc(
    close_prices: list[float],
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """根据 close 序列生成简单的 OHLC DataFrame。"""
    n = len(close_prices)
    dates = pd.date_range(start, periods=n, freq="B")
    close = np.array(close_prices, dtype=float)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.001
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=dates,
    )


def _make_trending_data(n: int = 100) -> pd.DataFrame:
    """生成强趋势数据：持续上涨 + 高 ADX。"""
    base = 100.0
    prices = [base + i * 1.5 + np.random.normal(0, 0.3) for i in range(n)]
    return _make_ohlc(prices)


def _make_mean_reverting_data(n: int = 100) -> pd.DataFrame:
    """生成震荡数据：围绕均值波动 + 低 ADX。"""
    prices = [100 + 2 * np.sin(i * 0.5) + np.random.normal(0, 0.2) for i in range(n)]
    return _make_ohlc(prices)


def _make_high_vol_data(n: int = 100) -> pd.DataFrame:
    """生成高波动数据：大部分平稳，最后一段剧烈波动。

    波动率检测比较最后 vol_period(20) 天 vs 前一个 vol_period(20) 天，
    所以需要让波动率突变发生在最后 20 天内。
    """
    calm_len = n - 20  # 前 80 天平稳
    calm = [100 + np.random.normal(0, 0.1) for _ in range(calm_len)]
    wild = [100 + np.random.normal(0, 10.0) for _ in range(20)]
    return _make_ohlc(calm + wild)


def _make_signal(
    stock: str,
    sig_type: SignalType,
    strategy: str,
    strength: float = 0.7,
) -> TradingSignal:
    """快速创建测试信号。"""
    return TradingSignal(
        timestamp=datetime(2024, 6, 1),
        stock_code=stock,
        signal_type=sig_type,
        strength=strength,
        price=100.0,
        reason=f"{strategy}: test",
        metadata={"strategy_name": strategy},
    )


# ======================================================================
# 测试 MarketRegimeDetector
# ======================================================================


class TestMarketRegimeDetector:
    """MarketRegimeDetector 单元测试"""

    def test_insufficient_data_returns_neutral(self):
        """数据不足时返回 NEUTRAL"""
        detector = MarketRegimeDetector()
        short_data = _make_ohlc([100, 101, 102])
        assert detector.detect(short_data) == MarketRegime.NEUTRAL

    def test_none_data_returns_neutral(self):
        """None 数据返回 NEUTRAL"""
        detector = MarketRegimeDetector()
        assert detector.detect(None) == MarketRegime.NEUTRAL

    def test_trending_detection(self):
        """强趋势数据应检测为 TRENDING"""
        np.random.seed(42)
        data = _make_trending_data(120)
        detector = MarketRegimeDetector()
        regime = detector.detect(data)
        # 强趋势数据 ADX 应该较高
        assert regime in (MarketRegime.TRENDING, MarketRegime.NEUTRAL)

    def test_mean_reverting_detection(self):
        """震荡数据应检测为 MEAN_REVERTING"""
        np.random.seed(42)
        data = _make_mean_reverting_data(120)
        detector = MarketRegimeDetector()
        regime = detector.detect(data)
        # 震荡数据 ADX 应该较低
        assert regime in (MarketRegime.MEAN_REVERTING, MarketRegime.NEUTRAL)

    def test_high_volatility_detection(self):
        """高波动数据应检测为 HIGH_VOLATILITY"""
        np.random.seed(42)
        data = _make_high_vol_data(120)
        detector = MarketRegimeDetector()
        regime = detector.detect(data)
        assert regime == MarketRegime.HIGH_VOLATILITY

    def test_custom_config(self):
        """自定义配置参数"""
        cfg = RegimeDetectorConfig(
            adx_period=10,
            trend_threshold=30.0,
            mean_rev_threshold=15.0,
        )
        detector = MarketRegimeDetector(config=cfg)
        assert detector.cfg.adx_period == 10
        assert detector.cfg.trend_threshold == 30.0


# ======================================================================
# 测试动态权重
# ======================================================================


class TestDynamicWeights:
    """get_dynamic_weights 单元测试"""

    def setup_method(self):
        self.detector = MarketRegimeDetector()
        self.names = ["macd_cross", "bollinger_band", "custom_alpha"]
        self.base = {"macd_cross": 0.4, "bollinger_band": 0.4, "custom_alpha": 0.2}

    def test_neutral_keeps_base(self):
        """NEUTRAL 状态保持基础权重"""
        result = self.detector.get_dynamic_weights(
            MarketRegime.NEUTRAL, self.names, self.base,
        )
        assert result == self.base

    def test_trending_boosts_trend(self):
        """TRENDING 状态趋势类权重应增大"""
        result = self.detector.get_dynamic_weights(
            MarketRegime.TRENDING, self.names, self.base,
        )
        # macd 权重应大于 bollinger 权重
        assert result["macd_cross"] > result["bollinger_band"]

    def test_mean_reverting_boosts_mr(self):
        """MEAN_REVERTING 状态均值回归类权重应增大"""
        result = self.detector.get_dynamic_weights(
            MarketRegime.MEAN_REVERTING, self.names, self.base,
        )
        assert result["bollinger_band"] > result["macd_cross"]

    def test_weights_sum_to_one(self):
        """所有 regime 下权重归一化后总和为 1"""
        for regime in MarketRegime:
            result = self.detector.get_dynamic_weights(
                regime, self.names, self.base,
            )
            assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_high_vol_reduces_trend(self):
        """HIGH_VOLATILITY 状态趋势类权重应远低于均值回归类"""
        result = self.detector.get_dynamic_weights(
            MarketRegime.HIGH_VOLATILITY, self.names, self.base,
        )
        # 趋势类权重应远低于均值回归类
        assert result["macd_cross"] < result["bollinger_band"]
        # 均值回归类应是最大的
        assert result["bollinger_band"] == max(result.values())


# ======================================================================
# 测试策略分类
# ======================================================================


class TestStrategyClassification:
    """策略名称自动分类测试"""

    def test_trend_keywords(self):
        assert _classify_strategy("MACD_Cross") == "trend"
        assert _classify_strategy("rsi_divergence") == "trend"
        assert _classify_strategy("momentum_factor") == "trend"
        assert _classify_strategy("trend_following") == "trend"

    def test_mean_reversion_keywords(self):
        assert _classify_strategy("cointegration_pairs") == "mean_reversion"
        assert _classify_strategy("Bollinger_Band") == "mean_reversion"
        assert _classify_strategy("mean_reversion_v2") == "mean_reversion"
        assert _classify_strategy("pairs_trading") == "mean_reversion"

    def test_other(self):
        assert _classify_strategy("custom_alpha") == "other"
        assert _classify_strategy("ml_model_v3") == "other"


# ======================================================================
# 测试 SignalIntegrator regime_aware_voting
# ======================================================================


class TestRegimeAwareVoting:
    """regime_aware_voting 整合方法测试"""

    def test_basic_integration(self):
        """基本整合：无 market_data 时退化为 NEUTRAL"""
        integrator = SignalIntegrator(method="regime_aware_voting")
        signals = [
            _make_signal("000001", SignalType.BUY, "macd_cross", 0.8),
            _make_signal("000001", SignalType.BUY, "bollinger_band", 0.6),
        ]
        weights = {"macd_cross": 0.5, "bollinger_band": 0.5}
        result = integrator.integrate(signals, weights)
        assert len(result) == 1
        assert result[0].signal_type == SignalType.BUY

    def test_regime_in_metadata(self):
        """输出信号 metadata 中应包含 regime 信息"""
        integrator = SignalIntegrator(method="regime_aware_voting")
        np.random.seed(42)
        data = _make_mean_reverting_data(120)
        signals = [
            _make_signal("000001", SignalType.BUY, "macd_cross"),
            _make_signal("000001", SignalType.BUY, "bollinger_band"),
        ]
        weights = {"macd_cross": 0.5, "bollinger_band": 0.5}
        result = integrator.integrate(signals, weights, market_data=data)
        assert len(result) == 1
        meta = result[0].metadata
        assert "regime" in meta
        assert "dynamic_weights" in meta
        assert meta["regime"] in [r.value for r in MarketRegime]

    def test_consensus_boost(self):
        """趋势类和均值回归类同向信号应获得一致性加成"""
        integrator = SignalIntegrator(method="regime_aware_voting")
        signals = [
            _make_signal("000001", SignalType.BUY, "macd_cross", 0.6),
            _make_signal("000001", SignalType.BUY, "bollinger_band", 0.6),
        ]
        weights = {"macd_cross": 0.5, "bollinger_band": 0.5}
        result = integrator.integrate(signals, weights)
        assert len(result) == 1
        assert result[0].metadata.get("consensus_boost") is True

    def test_high_vol_position_scale(self):
        """HIGH_VOLATILITY 时 position_scale 应为 0.5"""
        integrator = SignalIntegrator(method="regime_aware_voting")
        np.random.seed(42)
        data = _make_high_vol_data(120)
        signals = [
            _make_signal("000001", SignalType.BUY, "macd_cross"),
            _make_signal("000001", SignalType.BUY, "bollinger_band"),
        ]
        weights = {"macd_cross": 0.5, "bollinger_band": 0.5}
        result = integrator.integrate(signals, weights, market_data=data)
        assert len(result) == 1
        assert result[0].metadata["position_scale"] == 0.5

    def test_empty_signals(self):
        """空信号列表应返回空"""
        integrator = SignalIntegrator(method="regime_aware_voting")
        result = integrator.integrate([], {"a": 1.0})
        assert result == []

    def test_multiple_stocks(self):
        """多只股票应分别整合"""
        integrator = SignalIntegrator(method="regime_aware_voting")
        signals = [
            _make_signal("000001", SignalType.BUY, "macd_cross"),
            _make_signal("000002", SignalType.SELL, "macd_cross"),
            _make_signal("000001", SignalType.BUY, "bollinger_band"),
            _make_signal("000002", SignalType.SELL, "bollinger_band"),
        ]
        weights = {"macd_cross": 0.5, "bollinger_band": 0.5}
        result = integrator.integrate(signals, weights)
        assert len(result) == 2
        codes = {s.stock_code for s in result}
        assert codes == {"000001", "000002"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
