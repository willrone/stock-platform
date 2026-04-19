from __future__ import annotations

from datetime import datetime

import pytest

from app.services.backtest.core.portfolio_manager import PortfolioManager
from app.services.backtest.core.portfolio_manager_array import PortfolioManagerArray
from app.services.backtest.models import BacktestConfig, SignalType, TradingSignal


def _make_signal(signal_type: SignalType, *, price: float = 10.0) -> TradingSignal:
    return TradingSignal(
        timestamp=datetime(2024, 1, 2),
        stock_code="000001.SZ",
        signal_type=signal_type,
        strength=1.0,
        price=price,
        reason="test",
    )


@pytest.mark.parametrize(
    "manager_factory",
    [
        lambda config: PortfolioManager(config),
        lambda config: PortfolioManagerArray(config, ["000001.SZ"]),
    ],
)
def test_portfolio_managers_apply_open_close_and_min_cost(manager_factory) -> None:
    config = BacktestConfig(
        initial_cash=2000.0,
        commission_rate=0.001,
        slippage_rate=0.0,
        max_position_size=0.5,
        cash_reserve_ratio=0.0,
        board_lot_size=100,
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5.0,
    )
    manager = manager_factory(config)

    buy_trade, buy_reason = manager.execute_signal(
        _make_signal(SignalType.BUY, price=10.0),
        {"000001.SZ": 10.0},
    )
    assert buy_reason is None
    assert buy_trade is not None
    assert buy_trade.quantity == 100
    assert buy_trade.commission == pytest.approx(5.0)
    assert manager.cash == pytest.approx(995.0)

    sell_trade, sell_reason = manager.execute_signal(
        _make_signal(SignalType.SELL, price=12.0),
        {"000001.SZ": 12.0},
    )
    assert sell_reason is None
    assert sell_trade is not None
    assert sell_trade.quantity == 100
    assert sell_trade.commission == pytest.approx(5.0)
    assert sell_trade.pnl == pytest.approx(195.0)
    assert manager.cash == pytest.approx(2190.0)
    assert manager.total_commission == pytest.approx(10.0)
