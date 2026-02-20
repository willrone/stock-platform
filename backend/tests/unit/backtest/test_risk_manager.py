"""
风控管理器测试 (P2)

测试 RiskManager 的止损止盈触发、最大回撤熔断逻辑。
源码：app/services/backtest/core/risk_manager.py
"""

import pytest
from datetime import datetime, timedelta

from app.services.backtest.core.risk_manager import (
    RiskManager,
    PositionPriceInfo,
    CircuitBreakerEvent,
    CIRCUIT_BREAKER_RECOVERY_RATIO,
)
from app.services.backtest.models.data_models import BacktestConfig
from app.services.backtest.models.enums import SignalType


def _make_config(**overrides) -> BacktestConfig:
    """构造 BacktestConfig，提供合理默认值"""
    defaults = {
        "initial_cash": 100000.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.15,
        "max_drawdown_pct": None,
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)


def _make_position_info(
    code: str = "000001.SZ",
    quantity: int = 100,
    avg_cost: float = 10.0,
    current_price: float = 10.0,
    ts: datetime = None,
) -> PositionPriceInfo:
    return PositionPriceInfo(
        stock_code=code,
        quantity=quantity,
        avg_cost=avg_cost,
        current_price=current_price,
        timestamp=ts or datetime(2023, 6, 1),
    )


# ============================================================
# 止损止盈测试
# ============================================================

class TestStopLossTakeProfit:
    """止损止盈触发测试"""

    def test_stop_loss_triggered(self):
        """亏损达��止损阈值时应产生卖出信号"""
        config = _make_config(stop_loss_pct=0.05)
        rm = RiskManager(config)

        # 成本 10，当前 9.4 → 亏损 6% > 5% 止损线
        pos = _make_position_info(avg_cost=10.0, current_price=9.4)
        signals = rm.check_stop_loss_take_profit({"000001.SZ": pos})

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.SELL
        assert "止损" in signals[0].reason

    def test_take_profit_triggered(self):
        """盈利达到止盈阈值时应产生卖出信号"""
        config = _make_config(take_profit_pct=0.15)
        rm = RiskManager(config)

        # 成本 10，当前 11.6 → 盈利 16% > 15% 止盈线
        pos = _make_position_info(avg_cost=10.0, current_price=11.6)
        signals = rm.check_stop_loss_take_profit({"000001.SZ": pos})

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.SELL
        assert "止盈" in signals[0].reason

    def test_no_signal_within_threshold(self):
        """盈亏在阈值范围内不应产生信号"""
        config = _make_config(stop_loss_pct=0.05, take_profit_pct=0.15)
        rm = RiskManager(config)

        # 成本 10，当前 10.5 → 盈利 5%，未达止盈
        pos = _make_position_info(avg_cost=10.0, current_price=10.5)
        signals = rm.check_stop_loss_take_profit({"000001.SZ": pos})

        assert len(signals) == 0

    def test_stop_loss_disabled(self):
        """止损阈值为 0 时不触发止损"""
        config = _make_config(stop_loss_pct=0.0, take_profit_pct=0.15)
        rm = RiskManager(config)

        # 亏损 20% 但止损关闭
        pos = _make_position_info(avg_cost=10.0, current_price=8.0)
        signals = rm.check_stop_loss_take_profit({"000001.SZ": pos})

        assert len(signals) == 0

    def test_zero_quantity_ignored(self):
        """持仓数量为 0 时不产生信号"""
        config = _make_config(stop_loss_pct=0.05)
        rm = RiskManager(config)

        pos = _make_position_info(quantity=0, avg_cost=10.0, current_price=5.0)
        signals = rm.check_stop_loss_take_profit({"000001.SZ": pos})

        assert len(signals) == 0

    def test_multiple_positions(self):
        """多只股票同时检查止损止盈"""
        config = _make_config(stop_loss_pct=0.05, take_profit_pct=0.15)
        rm = RiskManager(config)

        positions = {
            "000001.SZ": _make_position_info(code="000001.SZ", avg_cost=10.0, current_price=9.0),  # 止损
            "000002.SZ": _make_position_info(code="000002.SZ", avg_cost=10.0, current_price=12.0),  # 止盈
            "000003.SZ": _make_position_info(code="000003.SZ", avg_cost=10.0, current_price=10.5),  # 正常
        }
        signals = rm.check_stop_loss_take_profit(positions)

        assert len(signals) == 2
        codes = {s.stock_code for s in signals}
        assert "000001.SZ" in codes
        assert "000002.SZ" in codes


# ============================================================
# 最大回撤熔断测试
# ============================================================

class TestCircuitBreaker:
    """最大回撤熔断测试"""

    def test_circuit_breaker_triggered(self):
        """回撤超过阈值时触发熔断"""
        config = _make_config(max_drawdown_pct=0.10)
        rm = RiskManager(config)

        d = datetime(2023, 6, 1)
        # 初始 100000，跌到 89000 → 回撤 11% > 10%
        rm.update_circuit_breaker(89000.0, d)

        assert rm.circuit_breaker_active is True
        assert len(rm.circuit_breaker_events) == 1
        assert rm.circuit_breaker_events[0].event_type == "triggered"

    def test_circuit_breaker_not_triggered(self):
        """回撤未超过阈值时不触发"""
        config = _make_config(max_drawdown_pct=0.10)
        rm = RiskManager(config)

        d = datetime(2023, 6, 1)
        rm.update_circuit_breaker(95000.0, d)  # 回撤 5% < 10%

        assert rm.circuit_breaker_active is False

    def test_circuit_breaker_recovery(self):
        """回撤缩小到恢复阈值时解除熔断"""
        config = _make_config(max_drawdown_pct=0.10)
        rm = RiskManager(config)

        d1 = datetime(2023, 6, 1)
        d2 = datetime(2023, 6, 5)

        # 触发熔断
        rm.update_circuit_breaker(89000.0, d1)
        assert rm.circuit_breaker_active is True

        # 恢复到回撤 < 10% * 50% = 5%，即净值 > 95000
        rm.update_circuit_breaker(96000.0, d2)
        assert rm.circuit_breaker_active is False
        assert len(rm.circuit_breaker_events) == 2
        assert rm.circuit_breaker_events[1].event_type == "recovered"

    def test_circuit_breaker_filters_buy_signals(self):
        """熔断激活时过滤掉买入信号，保留卖出信号"""
        config = _make_config(max_drawdown_pct=0.10)
        rm = RiskManager(config)

        # 触发熔断
        rm.update_circuit_breaker(89000.0, datetime(2023, 6, 1))

        from app.services.backtest.models.data_models import TradingSignal
        signals = [
            TradingSignal(
                timestamp=datetime(2023, 6, 2),
                stock_code="000001.SZ",
                signal_type=SignalType.BUY,
                strength=0.8,
                price=10.0,
                reason="test buy",
            ),
            TradingSignal(
                timestamp=datetime(2023, 6, 2),
                stock_code="000002.SZ",
                signal_type=SignalType.SELL,
                strength=0.8,
                price=20.0,
                reason="test sell",
            ),
        ]

        filtered = rm.filter_signals_by_circuit_breaker(signals)

        assert len(filtered) == 1
        assert filtered[0].signal_type == SignalType.SELL

    def test_circuit_breaker_disabled(self):
        """max_drawdown_pct=None 时不启用熔断"""
        config = _make_config(max_drawdown_pct=None)
        rm = RiskManager(config)

        rm.update_circuit_breaker(50000.0, datetime(2023, 6, 1))  # 回撤 50%

        assert rm.circuit_breaker_active is False

    def test_peak_value_updates(self):
        """净值创新高时更新 peak"""
        config = _make_config(max_drawdown_pct=0.10)
        rm = RiskManager(config)

        rm.update_circuit_breaker(110000.0, datetime(2023, 6, 1))
        assert rm.peak_portfolio_value == 110000.0

        rm.update_circuit_breaker(105000.0, datetime(2023, 6, 2))
        assert rm.peak_portfolio_value == 110000.0  # 不应降低

    def test_circuit_breaker_summary(self):
        """get_circuit_breaker_summary 返回正确摘要"""
        config = _make_config(max_drawdown_pct=0.10)
        rm = RiskManager(config)

        rm.update_circuit_breaker(89000.0, datetime(2023, 6, 1))
        rm.update_circuit_breaker(96000.0, datetime(2023, 6, 5))

        summary = rm.get_circuit_breaker_summary()
        assert summary["total_triggers"] == 1
        assert summary["total_recoveries"] == 1
        assert summary["currently_active"] is False
