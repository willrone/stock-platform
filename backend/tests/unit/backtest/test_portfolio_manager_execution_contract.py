"""portfolio_manager / portfolio_manager_array execution contract tests.

锁定 position sizing / cash reserve / board lot / affordability 等买入执行合同，
确保 dict 版与 array 版在可观察行为上保持可比较。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pytest

from app.services.backtest.core.portfolio_manager import PortfolioManager
from app.services.backtest.core.portfolio_manager_array import PortfolioManagerArray
from app.services.backtest.models import BacktestConfig, SignalType, TradingSignal
from app.services.backtest.utils.rejection_reason_classifier import (
    classify_rejection_reason,
)

ManagerKind = Literal["classic", "array"]


@pytest.fixture(params=["classic", "array"])
def manager_kind(request: pytest.FixtureRequest) -> ManagerKind:
    return request.param  # type: ignore[return-value]


def _build_signal(price: float, *, stock_code: str = "000001.SZ") -> TradingSignal:
    return TradingSignal(
        timestamp=datetime(2024, 1, 2, 9, 30),
        stock_code=stock_code,
        signal_type=SignalType.BUY,
        strength=0.8,
        price=price,
        reason="execution contract test",
    )


def _build_manager(kind: ManagerKind, config: BacktestConfig):
    if kind == "classic":
        return PortfolioManager(config)
    return PortfolioManagerArray(config, ["000001.SZ"])


@pytest.mark.parametrize("manager_kind", ["classic", "array"])
def test_buy_quantity_respects_max_position_size_and_board_lot(manager_kind: ManagerKind) -> None:
    config = BacktestConfig(
        initial_cash=10_000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_position_size=0.25,
        cash_reserve_ratio=0.0,
        board_lot_size=100,
    )
    manager = _build_manager(manager_kind, config)

    trade, reason = manager.execute_signal(_build_signal(10.0), {"000001.SZ": 10.0})

    assert reason is None
    assert trade is not None
    assert trade.action == "BUY"
    assert trade.quantity == 200
    assert trade.price == 10.0
    assert manager.cash == pytest.approx(8_000.0)
    assert manager.positions["000001.SZ"].quantity == 200
    assert manager.positions["000001.SZ"].avg_cost == pytest.approx(10.0)


@pytest.mark.parametrize("manager_kind", ["classic", "array"])
def test_cash_reserve_ratio_caps_buying_power(manager_kind: ManagerKind) -> None:
    config = BacktestConfig(
        initial_cash=10_000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_position_size=1.0,
        cash_reserve_ratio=0.25,
        board_lot_size=100,
    )
    manager = _build_manager(manager_kind, config)

    trade, reason = manager.execute_signal(_build_signal(10.0), {"000001.SZ": 10.0})

    assert reason is None
    assert trade is not None
    assert trade.quantity == 700
    assert manager.cash == pytest.approx(3_000.0)
    assert manager.positions["000001.SZ"].quantity == 700


@pytest.mark.parametrize("manager_kind", ["classic", "array"])
def test_buy_rejection_classifies_insufficient_buy_quantity(manager_kind: ManagerKind) -> None:
    config = BacktestConfig(
        initial_cash=1_000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_position_size=1.0,
        cash_reserve_ratio=0.0,
        board_lot_size=100,
    )
    manager = _build_manager(manager_kind, config)

    trade, reason = manager.execute_signal(_build_signal(11.0), {"000001.SZ": 11.0})

    assert trade is None
    assert reason is not None
    assert classify_rejection_reason(reason) == "insufficient_buy_quantity"


@pytest.mark.parametrize("manager_kind", ["classic", "array"])
def test_buy_rejection_classifies_position_limit_after_existing_holding(manager_kind: ManagerKind) -> None:
    config = BacktestConfig(
        initial_cash=10_000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        max_position_size=0.2,
        cash_reserve_ratio=0.0,
        board_lot_size=100,
    )
    manager = _build_manager(manager_kind, config)

    first_trade, first_reason = manager.execute_signal(
        _build_signal(10.0), {"000001.SZ": 10.0}
    )
    second_trade, second_reason = manager.execute_signal(
        _build_signal(10.0), {"000001.SZ": 10.0}
    )

    assert first_reason is None
    assert first_trade is not None
    assert second_trade is None
    assert second_reason is not None
    assert classify_rejection_reason(second_reason) == "position_limit"
    assert manager.positions["000001.SZ"].quantity == 200


@pytest.mark.parametrize("manager_kind", ["classic", "array"])
def test_buy_rejection_classifies_affordability_gap_from_commission(manager_kind: ManagerKind) -> None:
    config = BacktestConfig(
        initial_cash=10_000.0,
        commission_rate=0.01,
        slippage_rate=0.0,
        max_position_size=1.0,
        cash_reserve_ratio=0.0,
        board_lot_size=100,
    )
    manager = _build_manager(manager_kind, config)

    trade, reason = manager.execute_signal(_build_signal(10.0), {"000001.SZ": 10.0})

    assert trade is None
    assert reason is not None
    assert classify_rejection_reason(reason) == "insufficient_funds"


@pytest.mark.parametrize("manager_kind", ["classic", "array"])
def test_buy_success_keeps_array_and_classic_outputs_comparable(manager_kind: ManagerKind) -> None:
    config = BacktestConfig(
        initial_cash=20_000.0,
        commission_rate=0.001,
        slippage_rate=0.001,
        max_position_size=0.3,
        cash_reserve_ratio=0.1,
        board_lot_size=100,
    )
    manager = _build_manager(manager_kind, config)

    trade, reason = manager.execute_signal(_build_signal(20.0), {"000001.SZ": 20.0})

    assert reason is None
    assert trade is not None
    assert trade.action == "BUY"
    assert trade.quantity == 200
    assert trade.price == pytest.approx(20.02)
    assert trade.commission == pytest.approx(4.004)
    assert trade.slippage_cost == pytest.approx(4.0)
    assert manager.total_commission == pytest.approx(4.004)
    assert manager.total_slippage == pytest.approx(4.0)
    assert manager.positions["000001.SZ"].quantity == 200
    assert manager.positions_without_cost["000001.SZ"].quantity == 200
    assert manager.cash == pytest.approx(15_991.996)
    assert manager.cash_without_cost == pytest.approx(16_000.0)
