"""Property tests for the current backtest engine contracts.

These tests used to exercise a removed synchronous backtest facade.  They now
cover the maintained surfaces: strategy signal generation, portfolio accounting,
executor parameter validation, async report aggregation, and performance metrics.
"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from app.services.backtest import (
    BacktestConfig,
    BacktestExecutor,
    PortfolioManager,
    SignalType,
    StrategyFactory,
    TradingSignal,
)
from app.services.backtest.execution import backtest_executor as executor_module


class _AsyncSessionContext:
    """Minimal async context manager for repository mocks."""

    def __init__(self, session: object) -> None:
        self._session = session

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _sample_stock_data(days: int = 252, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2024-01-01", periods=days)
    returns = rng.normal(0.001, 0.02, days)
    prices = 100 * np.exp(np.cumsum(returns))
    data = pd.DataFrame(
        {
            "open": prices * (1 + rng.normal(0, 0.005, days)),
            "high": prices * (1 + np.abs(rng.normal(0, 0.01, days))),
            "low": prices * (1 - np.abs(rng.normal(0, 0.01, days))),
            "close": prices,
            "volume": rng.integers(1_000_000, 10_000_000, days),
        },
        index=dates,
    )
    data["high"] = np.maximum(data["high"], data["close"])
    data["low"] = np.minimum(data["low"], data["close"])
    data.attrs["stock_code"] = "000001.SZ"
    return data


def _patch_executor_run(
    executor: BacktestExecutor,
    *,
    stock_codes: list[str],
    metrics: dict[str, float],
) -> ExitStack:
    trading_dates = pd.bdate_range("2024-01-01", periods=20).to_pydatetime().tolist()
    stock_data = {
        code: pd.DataFrame(
            {
                "open": range(100, 100 + len(trading_dates)),
                "high": range(101, 101 + len(trading_dates)),
                "low": range(99, 99 + len(trading_dates)),
                "close": range(100, 100 + len(trading_dates)),
                "volume": [1_000_000] * len(trading_dates),
            },
            index=pd.DatetimeIndex(trading_dates),
        )
        for code in stock_codes
    }
    portfolio_manager = Mock()
    portfolio_manager.get_performance_metrics.return_value = metrics
    portfolio_manager.trades = [
        SimpleNamespace(
            trade_id="T000001",
            stock_code=stock_codes[0],
            action="BUY",
            quantity=100,
            price=10.0,
            timestamp=trading_dates[0],
            commission=1.0,
            slippage_cost=0.1,
            pnl=0.0,
            cumulative_pnl=0.0,
        )
    ]
    portfolio_manager.portfolio_history = [
        {"date": trading_dates[0], "portfolio_value": 100_000.0, "cash": 99_000.0}
    ]
    portfolio_manager.total_commission = 1.0
    portfolio_manager.total_slippage = 0.1
    portfolio_manager.get_performance_metrics_without_cost.return_value = {}

    stack = ExitStack()
    stack.enter_context(
        patch.object(
            executor_module.AdvancedStrategyFactory,
            "create_strategy",
            return_value=object(),
        )
    )
    stack.enter_context(
        patch.object(executor.data_loader, "load_multiple_stocks", return_value=stock_data)
    )
    stack.enter_context(patch.object(executor, "_get_trading_calendar", return_value=trading_dates))
    stack.enter_context(patch.object(executor, "_build_date_index"))
    stack.enter_context(patch.object(executor, "_precompute_strategy_signals"))
    stack.enter_context(
        patch.object(executor, "_extract_precomputed_signals_to_dict", return_value={})
    )
    stack.enter_context(
        patch.object(
            executor,
            "_build_aligned_arrays",
            return_value={
                "stock_codes": stock_codes,
                "close": [[100.0] * len(trading_dates) for _ in stock_codes],
                "signal": [[0] * len(trading_dates) for _ in stock_codes],
            },
        )
    )
    stack.enter_context(
        patch.object(executor, "_prepare_strategy_backtest_data", AsyncMock(return_value=None))
    )
    stack.enter_context(
        patch.object(executor_module, "PortfolioManagerArray", return_value=portfolio_manager)
    )
    stack.enter_context(
        patch.object(
            executor,
            "_execute_backtest_loop",
            AsyncMock(
                return_value={
                    "total_signals": len(stock_codes) * 2,
                    "trading_days": len(trading_dates),
                    "executed_trades": 1,
                }
            ),
        )
    )
    stack.enter_context(patch.object(executor, "_calculate_additional_metrics", return_value={}))
    stack.enter_context(
        patch(
            "app.core.database.get_async_session_context",
            return_value=_AsyncSessionContext(MagicMock()),
        )
    )
    return stack


class TestBacktestEngineAccuracy:
    """Property tests for the maintained backtest components."""

    @given(
        strategy_name=st.sampled_from(["moving_average", "rsi", "macd"]),
        short_window=st.integers(min_value=3, max_value=10),
        long_window=st.integers(min_value=15, max_value=30),
    )
    @settings(max_examples=40, deadline=None)
    def test_strategy_signal_generation_accuracy(
        self, strategy_name: str, short_window: int, long_window: int
    ) -> None:
        """Strategies should emit well-formed TradingSignal objects."""
        assume(short_window < long_window)
        if strategy_name == "moving_average":
            config = {
                "short_window": short_window,
                "long_window": long_window,
                "signal_threshold": 0.02,
            }
        elif strategy_name == "rsi":
            config = {
                "rsi_period": min(short_window + 5, 14),
                "oversold_threshold": 30,
                "overbought_threshold": 70,
            }
        else:
            config = {
                "fast_period": short_window,
                "slow_period": long_window,
                "signal_period": 9,
            }

        strategy = StrategyFactory.create_strategy(strategy_name, config)
        test_data = _sample_stock_data()
        current_date = test_data.index[-50]
        signals = strategy.generate_signals(test_data, current_date)

        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.stock_code == "000001.SZ"
            assert signal.signal_type in {SignalType.BUY, SignalType.SELL, SignalType.HOLD}
            assert 0 <= signal.strength <= 1
            assert signal.price > 0
            assert signal.timestamp == current_date
            assert signal.reason

        indicators = strategy.calculate_indicators(test_data)
        assert indicators
        for indicator in indicators.values():
            assert isinstance(indicator, pd.Series)
            assert len(indicator) == len(test_data)
            assert not indicator.isnull().all()

    @given(
        initial_cash=st.floats(min_value=50_000, max_value=500_000),
        commission_rate=st.floats(min_value=0.0001, max_value=0.01),
        max_position_size=st.floats(min_value=0.1, max_value=0.5),
    )
    @settings(max_examples=40, deadline=None)
    def test_portfolio_management_accuracy(
        self, initial_cash: float, commission_rate: float, max_position_size: float
    ) -> None:
        """PortfolioManager should update cash, positions, and trades consistently."""
        config = BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            max_position_size=max_position_size,
            slippage_rate=0.0,
            cash_reserve_ratio=0.0,
            board_lot_size=1,
        )
        portfolio_manager = PortfolioManager(config)
        buy_signal = TradingSignal(
            timestamp=datetime(2024, 6, 15),
            stock_code="000001.SZ",
            signal_type=SignalType.BUY,
            strength=0.8,
            price=100.0,
            reason="test buy",
        )

        buy_trade, buy_error = portfolio_manager.execute_signal(
            buy_signal, {"000001.SZ": 100.0}
        )

        assert buy_error is None
        assert buy_trade is not None
        assert buy_trade.action == "BUY"
        assert buy_trade.quantity > 0
        assert portfolio_manager.cash < initial_cash
        position = portfolio_manager.positions["000001.SZ"]
        assert position.quantity == buy_trade.quantity

        expected_cash = initial_cash - (buy_trade.quantity * buy_trade.price + buy_trade.commission)
        assert abs(portfolio_manager.cash - expected_cash) < 0.01

        sell_signal = TradingSignal(
            timestamp=datetime(2024, 6, 25),
            stock_code="000001.SZ",
            signal_type=SignalType.SELL,
            strength=0.7,
            price=105.0,
            reason="test sell",
        )
        sell_trade, sell_error = portfolio_manager.execute_signal(
            sell_signal, {"000001.SZ": 105.0}
        )

        assert sell_error is None
        assert sell_trade is not None
        assert sell_trade.action == "SELL"
        assert sell_trade.quantity == position.quantity
        assert "000001.SZ" not in portfolio_manager.positions

    @given(
        returns_data=st.lists(
            st.floats(min_value=-0.1, max_value=0.1), min_size=50, max_size=200
        )
    )
    @settings(max_examples=40, deadline=None)
    def test_performance_metrics_accuracy(self, returns_data: list[float]) -> None:
        """Performance metrics should stay numeric and within expected bounds."""
        config = BacktestConfig(initial_cash=100_000)
        portfolio_manager = PortfolioManager(config)
        portfolio_values = [config.initial_cash]
        for daily_return in returns_data:
            portfolio_values.append(max(portfolio_values[-1] * (1 + daily_return), 1_000))

        base_date = datetime(2024, 1, 1)
        for i, value in enumerate(portfolio_values):
            portfolio_manager.equity_curve.append((base_date + timedelta(days=i), value))
            portfolio_manager.portfolio_history.append(
                {
                    "date": base_date + timedelta(days=i),
                    "cash": value * 0.1,
                    "portfolio_value": value,
                    "positions": {},
                    "total_trades": i,
                }
            )
            portfolio_manager.portfolio_history_without_cost.append(
                {
                    "date": base_date + timedelta(days=i),
                    "cash": value * 0.1,
                    "portfolio_value": value,
                    "positions": {},
                    "total_trades": i,
                }
            )

        metrics = portfolio_manager.get_performance_metrics()

        assert isinstance(metrics.get("total_return"), float)
        assert isinstance(metrics.get("annualized_return"), float)
        assert metrics["volatility"] >= 0
        assert -1 <= metrics["max_drawdown"] <= 0
        assert 0 <= metrics["win_rate"] <= 1
        assert metrics["total_trades"] >= 0
        expected_total_return = (portfolio_values[-1] - config.initial_cash) / config.initial_cash
        assert abs(metrics["total_return"] - expected_total_return) < 0.01

    @given(
        strategy_name=st.sampled_from(["moving_average", "rsi", "macd"]),
        stock_count=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=20, deadline=None)
    def test_validate_backtest_parameters_accepts_current_contract(
        self, strategy_name: str, stock_count: int
    ) -> None:
        """Executor parameter validation should accept supported strategies and universes."""
        executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)
        stock_codes = [f"00000{i}.SZ" for i in range(1, stock_count + 1)]

        assert executor.validate_backtest_parameters(
            strategy_name=strategy_name,
            stock_codes=stock_codes,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 1),
            strategy_config={},
        ) is True

    @pytest.mark.asyncio
    @given(stock_count=st.integers(min_value=1, max_value=3))
    @settings(max_examples=10, deadline=None)
    async def test_async_backtest_report_contract(self, stock_count: int) -> None:
        """Async executor should aggregate reports for one or more stocks."""
        stock_codes = [f"00000{i}.SZ" for i in range(1, stock_count + 1)]
        metrics = {
            "total_return": 0.05 * stock_count,
            "annualized_return": 0.06,
            "volatility": 0.2,
            "sharpe_ratio": 0.3,
            "max_drawdown": -0.04,
            "win_rate": 0.5,
            "profit_factor": 1.2,
            "total_trades": 1,
        }
        executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)

        with _patch_executor_run(executor, stock_codes=stock_codes, metrics=metrics):
            result = await executor.run_backtest(
                strategy_name="rsi",
                stock_codes=stock_codes,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 3, 1),
                strategy_config={"rsi_period": 14},
                backtest_config=BacktestConfig(initial_cash=100_000.0),
            )

        assert result["strategy_name"] == "rsi"
        assert result["stock_codes"] == stock_codes
        assert result["metrics"]["total_return"] == metrics["total_return"]
        assert result["total_signals"] == stock_count * 2
        assert result["trading_days"] == 20
        assert isinstance(result["trade_history"], list)
        assert set(result["perf_breakdown"]).issuperset(
            {"strategy_setup_s", "data_loading_s", "main_loop_s", "total_wall_s"}
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
