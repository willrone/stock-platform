"""
回测执行器测试 (P1)

测试 backtest_loop_executor 中可独立测试的函数。
主循环 execute_backtest_loop 依赖数据库/任务状态，此处仅测试
_check_and_execute_stop_loss_take_profit 等可隔离的逻辑。

源码：app/services/backtest/execution/backtest_loop_executor.py
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from app.services.backtest.execution.backtest_loop_executor import (
    _check_and_execute_stop_loss_take_profit,
)
from app.services.backtest.core.risk_manager import (
    RiskManager,
    PositionPriceInfo,
)
from app.services.backtest.models.data_models import (
    BacktestConfig,
    Position,
    TradingSignal,
    Trade,
)
from app.services.backtest.models.enums import SignalType


def _make_config(**overrides) -> BacktestConfig:
    defaults = {
        "initial_cash": 100000.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.15,
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)


class TestCheckAndExecuteStopLossTakeProfit:
    """_check_and_execute_stop_loss_take_profit 独立函数测试"""

    def _make_portfolio_manager(self, positions: dict, execute_result=None):
        """构造 mock PortfolioManager"""
        pm = Mock()
        pm.positions = positions
        pm.config = _make_config()
        if execute_result is None:
            # 默认返回一笔成功交易
            mock_trade = Mock()
            pm.execute_signal.return_value = (mock_trade, None)
        else:
            pm.execute_signal.return_value = execute_result
        return pm

    def test_no_positions_returns_zero(self):
        """无持仓时返回 0"""
        config = _make_config()
        rm = RiskManager(config)
        pm = self._make_portfolio_manager(positions={})

        result = _check_and_execute_stop_loss_take_profit(
            rm, pm, {}, datetime(2023, 6, 1)
        )
        assert result == 0

    def test_stop_loss_executes_trade(self):
        """止损触发时执行卖出交易"""
        config = _make_config(stop_loss_pct=0.05)
        rm = RiskManager(config)

        # 模拟持仓：成本 10，当前价 9.0（亏损 10%）
        positions = {
            "000001.SZ": Mock(quantity=100, avg_cost=10.0),
        }
        pm = self._make_portfolio_manager(positions)

        current_prices = {"000001.SZ": 9.0}
        result = _check_and_execute_stop_loss_take_profit(
            rm, pm, current_prices, datetime(2023, 6, 1)
        )

        assert result == 1
        pm.execute_signal.assert_called_once()

    def test_no_trigger_returns_zero(self):
        """盈亏在阈值内不执行交易"""
        config = _make_config(stop_loss_pct=0.05, take_profit_pct=0.15)
        rm = RiskManager(config)

        positions = {
            "000001.SZ": Mock(quantity=100, avg_cost=10.0),
        }
        pm = self._make_portfolio_manager(positions)

        current_prices = {"000001.SZ": 10.5}  # 盈利 5%，未达止盈
        result = _check_and_execute_stop_loss_take_profit(
            rm, pm, current_prices, datetime(2023, 6, 1)
        )

        assert result == 0
        pm.execute_signal.assert_not_called()

    def test_trade_execution_failure(self):
        """交易执行失败时不计入交易数"""
        config = _make_config(stop_loss_pct=0.05)
        rm = RiskManager(config)

        positions = {
            "000001.SZ": Mock(quantity=100, avg_cost=10.0),
        }
        pm = self._make_portfolio_manager(
            positions, execute_result=(None, "资金不足")
        )

        current_prices = {"000001.SZ": 9.0}
        result = _check_and_execute_stop_loss_take_profit(
            rm, pm, current_prices, datetime(2023, 6, 1)
        )

        assert result == 0

    def test_price_missing_for_position(self):
        """持仓股票无当前价格时跳过"""
        config = _make_config(stop_loss_pct=0.05)
        rm = RiskManager(config)

        positions = {
            "000001.SZ": Mock(quantity=100, avg_cost=10.0),
        }
        pm = self._make_portfolio_manager(positions)

        # 当前价格中没有该股票
        current_prices = {}
        result = _check_and_execute_stop_loss_take_profit(
            rm, pm, current_prices, datetime(2023, 6, 1)
        )

        assert result == 0
