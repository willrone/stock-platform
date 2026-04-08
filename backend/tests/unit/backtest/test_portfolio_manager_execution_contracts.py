"""portfolio_manager / array execution contract tests.

锁定 position sizing、cash reserve、board lot 与不足一手时的当前执行合同，
确保 dict / array 两套组合管理器在关键约束上保持可比较。
"""

from __future__ import annotations

from datetime import datetime

from app.services.backtest.core.portfolio_manager import PortfolioManager
from app.services.backtest.core.portfolio_manager_array import PortfolioManagerArray
from app.services.backtest.models import BacktestConfig, SignalType, TradingSignal
from app.services.backtest.utils.rejection_reason_classifier import (
    classify_rejection_reason,
)

TEST_STOCK_CODE = "000001.SZ"
TEST_TIMESTAMP = datetime(2024, 1, 2, 9, 30)
TEST_PRICE = 100.0


def _build_config(**overrides: float | int) -> BacktestConfig:
    """Create a deterministic backtest config for execution contract tests."""
    config_values: dict[str, float | int | bool | str] = {
        "initial_cash": 100_000.0,
        "commission_rate": 0.0,
        "slippage_rate": 0.0,
        "max_position_size": 1.0,
        "cash_reserve_ratio": 0.0,
        "board_lot_size": 100,
    }
    config_values.update(overrides)
    return BacktestConfig(**config_values)


def _build_buy_signal(price: float = TEST_PRICE) -> TradingSignal:
    """Return a minimal deterministic buy signal for contract checks."""
    return TradingSignal(
        timestamp=TEST_TIMESTAMP,
        stock_code=TEST_STOCK_CODE,
        signal_type=SignalType.BUY,
        strength=0.9,
        price=price,
        reason="execution contract test",
    )


def _build_managers(
    config: BacktestConfig,
) -> tuple[PortfolioManager, PortfolioManagerArray]:
    """Create dict and array portfolio managers with the same config."""
    return (
        PortfolioManager(config),
        PortfolioManagerArray(config, stock_codes=[TEST_STOCK_CODE]),
    )


def _execute_buy(
    config: BacktestConfig,
    *,
    price: float = TEST_PRICE,
) -> tuple[tuple[object | None, str | None], tuple[object | None, str | None]]:
    """Execute the same buy signal against dict and array managers."""
    portfolio_manager, portfolio_manager_array = _build_managers(config)
    signal = _build_buy_signal(price=price)
    current_prices = {TEST_STOCK_CODE: price}
    return (
        portfolio_manager.execute_signal(signal, current_prices),
        portfolio_manager_array.execute_signal(signal, current_prices),
    )


def test_board_lot_rounding_matches_position_sizing_across_managers() -> None:
    """max_position_size 与 board_lot_size 应共同决定一致的可买数量。"""
    config = _build_config(max_position_size=0.23, board_lot_size=100)

    (trade_dict, reason_dict), (trade_array, reason_array) = _execute_buy(config)

    assert reason_dict is None
    assert reason_array is None
    assert trade_dict is not None
    assert trade_array is not None
    assert trade_dict.quantity == 200
    assert trade_array.quantity == 200
    assert trade_dict.quantity == trade_array.quantity
    assert trade_dict.quantity % config.board_lot_size == 0
    assert trade_array.quantity % config.board_lot_size == 0


def test_cash_reserve_ratio_reduces_buy_quantity_for_both_managers() -> None:
    """cash_reserve_ratio 应在两套实现里一致收缩买入数量。"""
    no_reserve_config = _build_config(max_position_size=1.0, cash_reserve_ratio=0.0)
    reserve_config = _build_config(max_position_size=1.0, cash_reserve_ratio=0.25)

    (no_reserve_trade_dict, _), (no_reserve_trade_array, _) = _execute_buy(
        no_reserve_config
    )
    (reserve_trade_dict, _), (reserve_trade_array, _) = _execute_buy(reserve_config)

    assert no_reserve_trade_dict is not None
    assert no_reserve_trade_array is not None
    assert reserve_trade_dict is not None
    assert reserve_trade_array is not None
    assert no_reserve_trade_dict.quantity == 1000
    assert no_reserve_trade_array.quantity == 1000
    assert reserve_trade_dict.quantity == 700
    assert reserve_trade_array.quantity == 700
    assert reserve_trade_dict.quantity < no_reserve_trade_dict.quantity
    assert reserve_trade_array.quantity < no_reserve_trade_array.quantity


def test_second_buy_hits_position_limit_in_both_managers() -> None:
    """已占满 max_position_size 后，二次买入应被归类为 position_limit。"""
    config = _build_config(max_position_size=0.2, board_lot_size=100)
    portfolio_manager, portfolio_manager_array = _build_managers(config)
    signal = _build_buy_signal()
    current_prices = {TEST_STOCK_CODE: TEST_PRICE}

    first_trade_dict, first_reason_dict = portfolio_manager.execute_signal(
        signal, current_prices
    )
    first_trade_array, first_reason_array = portfolio_manager_array.execute_signal(
        signal, current_prices
    )
    second_trade_dict, second_reason_dict = portfolio_manager.execute_signal(
        signal, current_prices
    )
    second_trade_array, second_reason_array = portfolio_manager_array.execute_signal(
        signal, current_prices
    )

    assert first_reason_dict is None
    assert first_reason_array is None
    assert first_trade_dict is not None
    assert first_trade_array is not None
    assert first_trade_dict.quantity == 200
    assert first_trade_array.quantity == 200
    assert second_trade_dict is None
    assert second_trade_array is None
    assert second_reason_dict is not None
    assert second_reason_array is not None
    assert classify_rejection_reason(second_reason_dict) == "position_limit"
    assert classify_rejection_reason(second_reason_array) == "position_limit"


def test_partial_room_below_one_board_lot_is_insufficient_buy_quantity() -> None:
    """剩余仓位不足一手时，当前合同应归类为 insufficient_buy_quantity。"""
    config = _build_config(max_position_size=0.205, board_lot_size=100)
    portfolio_manager, portfolio_manager_array = _build_managers(config)
    signal = _build_buy_signal()
    current_prices = {TEST_STOCK_CODE: TEST_PRICE}

    first_trade_dict, first_reason_dict = portfolio_manager.execute_signal(
        signal, current_prices
    )
    first_trade_array, first_reason_array = portfolio_manager_array.execute_signal(
        signal, current_prices
    )
    second_trade_dict, second_reason_dict = portfolio_manager.execute_signal(
        signal, current_prices
    )
    second_trade_array, second_reason_array = portfolio_manager_array.execute_signal(
        signal, current_prices
    )

    assert first_reason_dict is None
    assert first_reason_array is None
    assert first_trade_dict is not None
    assert first_trade_array is not None
    assert first_trade_dict.quantity == 200
    assert first_trade_array.quantity == 200
    assert second_trade_dict is None
    assert second_trade_array is None
    assert second_reason_dict is not None
    assert second_reason_array is not None
    assert classify_rejection_reason(second_reason_dict) == "insufficient_buy_quantity"
    assert classify_rejection_reason(second_reason_array) == "insufficient_buy_quantity"
