from __future__ import annotations

from datetime import datetime

from app.services.backtest.core.portfolio_manager import PortfolioManager
from app.services.backtest.execution.trade_modes import (
    TradeModeExecutionContext,
    get_trade_mode_executor,
)
from app.services.backtest.models import BacktestConfig, SignalType, TradingSignal


class _AlwaysValidStrategy:
    def validate_signal(self, signal, portfolio_value, current_positions):
        return True, None


def _build_signal(
    current_date: datetime, stock_code: str, score: float, price: float
) -> TradingSignal:
    return TradingSignal(
        timestamp=current_date,
        stock_code=stock_code,
        signal_type=SignalType.BUY,
        strength=min(1.0, abs(score) * 100),
        price=price,
        reason="ranking score {score:.4f}",
        metadata={
            "ranking_score": score,
            "signal_role": "ranking_score",
        },
    )


def test_topk_dropout_trade_mode_rotates_worst_holding_into_best_new_name() -> None:
    executor = get_trade_mode_executor("topk_dropout")
    portfolio_manager = PortfolioManager(
        BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            max_position_size=0.6,
            cash_reserve_ratio=0.0,
            board_lot_size=100,
        )
    )
    strategy = _AlwaysValidStrategy()

    day_one = datetime(2024, 1, 2)
    day_two = datetime(2024, 1, 3)
    prices = {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0}

    first_result = executor.execute(
        TradeModeExecutionContext(
            current_date=day_one,
            all_signals=[
                _build_signal(day_one, "AAA", 0.90, prices["AAA"]),
                _build_signal(day_one, "BBB", 0.80, prices["BBB"]),
                _build_signal(day_one, "CCC", 0.10, prices["CCC"]),
            ],
            current_prices=prices,
            portfolio_manager=portfolio_manager,
            strategy=strategy,
            strategy_config={"topk": 2, "n_drop": 1},
            stock_universe=["AAA", "BBB", "CCC"],
        )
    )

    assert first_result.trades_this_day == 2
    assert set(portfolio_manager.positions.keys()) == {"AAA", "BBB"}

    second_result = executor.execute(
        TradeModeExecutionContext(
            current_date=day_two,
            all_signals=[
                _build_signal(day_two, "AAA", 0.20, prices["AAA"]),
                _build_signal(day_two, "BBB", 0.95, prices["BBB"]),
                _build_signal(day_two, "CCC", 0.90, prices["CCC"]),
            ],
            current_prices=prices,
            portfolio_manager=portfolio_manager,
            strategy=strategy,
            strategy_config={"topk": 2, "n_drop": 1},
            stock_universe=["AAA", "BBB", "CCC"],
        )
    )

    assert second_result.trades_this_day == 2
    assert set(portfolio_manager.positions.keys()) == {"BBB", "CCC"}
    assert {item["signal_type"] for item in second_result.executed_trade_signals} == {
        "BUY",
        "SELL",
    }


def test_topk_dropout_trade_mode_ignores_candidates_without_current_price() -> None:
    executor = get_trade_mode_executor("topk_dropout")
    portfolio_manager = PortfolioManager(
        BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            max_position_size=0.6,
            cash_reserve_ratio=0.0,
            board_lot_size=100,
        )
    )
    strategy = _AlwaysValidStrategy()
    current_date = datetime(2024, 1, 2)

    result = executor.execute(
        TradeModeExecutionContext(
            current_date=current_date,
            all_signals=[
                _build_signal(current_date, "AAA", 0.90, 10.0),
                _build_signal(current_date, "BBB", 0.80, 20.0),
                _build_signal(current_date, "CCC", 0.70, 0.0),
            ],
            current_prices={"AAA": 10.0, "BBB": 20.0},
            portfolio_manager=portfolio_manager,
            strategy=strategy,
            strategy_config={"topk": 3, "n_drop": 1},
            stock_universe=["AAA", "BBB", "CCC"],
        )
    )

    assert result.trades_this_day == 2
    assert set(portfolio_manager.positions.keys()) == {"AAA", "BBB"}
    assert result.unexecuted_signals == []


def test_topk_dropout_trade_mode_does_not_trade_without_ranking_signals() -> None:
    executor = get_trade_mode_executor("topk_dropout")
    portfolio_manager = PortfolioManager(
        BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            max_position_size=0.6,
            cash_reserve_ratio=0.0,
            board_lot_size=100,
        )
    )

    result = executor.execute(
        TradeModeExecutionContext(
            current_date=datetime(2024, 1, 2),
            all_signals=[],
            current_prices={"AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
            portfolio_manager=portfolio_manager,
            strategy=_AlwaysValidStrategy(),
            strategy_config={"topk": 2, "n_drop": 1},
            stock_universe=["AAA", "BBB", "CCC"],
        )
    )

    assert result.trades_this_day == 0
    assert result.executed_trade_signals == []
    assert result.unexecuted_signals == []
    assert portfolio_manager.positions == {}


def test_topk_dropout_trade_mode_respects_hold_threshold_buffer() -> None:
    executor = get_trade_mode_executor("topk_dropout")
    portfolio_manager = PortfolioManager(
        BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            max_position_size=0.6,
            cash_reserve_ratio=0.0,
            board_lot_size=100,
        )
    )
    strategy = _AlwaysValidStrategy()

    day_one = datetime(2024, 1, 2)
    day_two = datetime(2024, 1, 3)
    prices = {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0}

    executor.execute(
        TradeModeExecutionContext(
            current_date=day_one,
            all_signals=[
                _build_signal(day_one, "AAA", 0.90, prices["AAA"]),
                _build_signal(day_one, "BBB", 0.80, prices["BBB"]),
                _build_signal(day_one, "CCC", 0.10, prices["CCC"]),
                _build_signal(day_one, "DDD", 0.05, prices["DDD"]),
            ],
            current_prices=prices,
            portfolio_manager=portfolio_manager,
            strategy=strategy,
            strategy_config={"topk": 2, "n_drop": 1, "hold_thresh": 1},
            stock_universe=["AAA", "BBB", "CCC", "DDD"],
        )
    )
    assert set(portfolio_manager.positions.keys()) == {"AAA", "BBB"}

    buffered_result = executor.execute(
        TradeModeExecutionContext(
            current_date=day_two,
            all_signals=[
                _build_signal(day_two, "BBB", 0.95, prices["BBB"]),
                _build_signal(day_two, "CCC", 0.90, prices["CCC"]),
                _build_signal(day_two, "AAA", 0.85, prices["AAA"]),
                _build_signal(day_two, "DDD", 0.10, prices["DDD"]),
            ],
            current_prices=prices,
            portfolio_manager=portfolio_manager,
            strategy=strategy,
            strategy_config={"topk": 2, "n_drop": 1, "hold_thresh": 1},
            stock_universe=["AAA", "BBB", "CCC", "DDD"],
        )
    )

    assert buffered_result.trades_this_day == 0
    assert set(portfolio_manager.positions.keys()) == {"AAA", "BBB"}


def test_topk_dropout_trade_mode_rebalances_once_holding_falls_beyond_buffer() -> None:
    executor = get_trade_mode_executor("topk_dropout")
    portfolio_manager = PortfolioManager(
        BacktestConfig(
            initial_cash=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            max_position_size=0.6,
            cash_reserve_ratio=0.0,
            board_lot_size=100,
        )
    )
    strategy = _AlwaysValidStrategy()

    day_one = datetime(2024, 1, 2)
    day_two = datetime(2024, 1, 3)
    prices = {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0}

    executor.execute(
        TradeModeExecutionContext(
            current_date=day_one,
            all_signals=[
                _build_signal(day_one, "AAA", 0.90, prices["AAA"]),
                _build_signal(day_one, "BBB", 0.80, prices["BBB"]),
                _build_signal(day_one, "CCC", 0.10, prices["CCC"]),
                _build_signal(day_one, "DDD", 0.05, prices["DDD"]),
            ],
            current_prices=prices,
            portfolio_manager=portfolio_manager,
            strategy=strategy,
            strategy_config={"topk": 2, "n_drop": 1, "hold_thresh": 1},
            stock_universe=["AAA", "BBB", "CCC", "DDD"],
        )
    )
    assert set(portfolio_manager.positions.keys()) == {"AAA", "BBB"}

    rebalance_result = executor.execute(
        TradeModeExecutionContext(
            current_date=day_two,
            all_signals=[
                _build_signal(day_two, "BBB", 0.95, prices["BBB"]),
                _build_signal(day_two, "CCC", 0.90, prices["CCC"]),
                _build_signal(day_two, "DDD", 0.85, prices["DDD"]),
                _build_signal(day_two, "AAA", 0.10, prices["AAA"]),
            ],
            current_prices=prices,
            portfolio_manager=portfolio_manager,
            strategy=strategy,
            strategy_config={"topk": 2, "n_drop": 1, "hold_thresh": 1},
            stock_universe=["AAA", "BBB", "CCC", "DDD"],
        )
    )

    assert rebalance_result.trades_this_day == 2
    assert set(portfolio_manager.positions.keys()) == {"BBB", "CCC"}
