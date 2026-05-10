"""Current backtest engine contract tests.

The historical version of this file targeted the removed synchronous
``SimpleBacktestEngine`` facade.  These tests now lock the public contracts that
remain supported by the production backtest stack: config/data model shape,
portfolio execution, ORM result serialization, executor validation, and async
run_backtest aggregation behavior.
"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest

from app.models.task_models import BacktestResult
from app.services.backtest import (
    BacktestConfig,
    BacktestExecutor,
    OrderType,
    PortfolioManager,
    SignalType,
    Trade,
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


def _patch_executor_run(
    executor: BacktestExecutor,
    *,
    stock_codes: list[str] | None = None,
    metrics: dict[str, float] | None = None,
    loop_result: dict[str, int] | None = None,
) -> ExitStack:
    """Patch expensive executor dependencies while preserving aggregation code."""
    codes = stock_codes or ["000001.SZ"]
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
        for code in codes
    }
    portfolio_manager = Mock()
    portfolio_manager.get_performance_metrics.return_value = metrics or {
        "total_return": 0.12,
        "annualized_return": 0.13,
        "volatility": 0.2,
        "sharpe_ratio": 0.65,
        "max_drawdown": -0.08,
        "win_rate": 0.5,
        "profit_factor": 1.8,
        "total_trades": 2,
    }
    portfolio_manager.trades = [
        SimpleNamespace(
            trade_id="T000001",
            stock_code=codes[0],
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
        patch.object(
            executor.data_loader, "load_multiple_stocks", return_value=stock_data
        )
    )
    stack.enter_context(
        patch.object(executor, "_get_trading_calendar", return_value=trading_dates)
    )
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
                "stock_codes": codes,
                "close": [[100.0] * len(trading_dates) for _ in codes],
                "signal": [[0] * len(trading_dates) for _ in codes],
            },
        )
    )
    stack.enter_context(
        patch.object(
            executor, "_prepare_strategy_backtest_data", AsyncMock(return_value=None)
        )
    )
    stack.enter_context(
        patch.object(
            executor_module, "PortfolioManagerArray", return_value=portfolio_manager
        )
    )
    stack.enter_context(
        patch.object(
            executor,
            "_execute_backtest_loop",
            AsyncMock(
                return_value=loop_result
                or {
                    "total_signals": 3,
                    "trading_days": len(trading_dates),
                    "executed_trades": 1,
                }
            ),
        )
    )
    stack.enter_context(
        patch.object(executor, "_calculate_additional_metrics", return_value={})
    )
    stack.enter_context(
        patch(
            "app.core.database.get_async_session_context",
            return_value=_AsyncSessionContext(MagicMock()),
        )
    )
    return stack


class TestBacktestConfig:
    """Backtest configuration model tests."""

    def test_backtest_config_creation(self) -> None:
        config = BacktestConfig(
            initial_cash=50_000.0,
            commission_rate=0.002,
            max_position_size=0.3,
        )

        assert config.initial_cash == 50_000.0
        assert config.commission_rate == 0.002
        assert config.max_position_size == 0.3

    def test_backtest_config_defaults(self) -> None:
        config = BacktestConfig()

        assert config.initial_cash == 100_000.0
        assert config.commission_rate == 0.001
        assert config.max_position_size == 0.2
        assert config.board_lot_size == 100


class TestTrade:
    """Trade data model tests."""

    def test_trade_creation(self) -> None:
        trade = Trade(
            trade_id="T000001",
            stock_code="000001.SZ",
            action="SELL",
            quantity=1_000,
            price=11.0,
            timestamp=datetime(2024, 1, 5),
            commission=10.0,
            slippage_cost=1.0,
            pnl=1_000.0,
            cumulative_pnl=1_000.0,
        )

        assert trade.stock_code == "000001.SZ"
        assert trade.action == "SELL"
        assert trade.pnl == 1_000.0
        assert trade.commission == 10.0
        assert trade.cumulative_pnl == 1_000.0

    def test_order_type_remains_order_style_enum(self) -> None:
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"


class TestBacktestResult:
    """Backtest result ORM serialization tests."""

    def test_backtest_result_to_dict(self) -> None:
        result = BacktestResult(
            task_id="task-1",
            backtest_id="bt-1",
            strategy_name="rsi",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_cash=100_000.0,
            final_value=120_000.0,
            total_return=0.2,
            annualized_return=0.2,
            max_drawdown=-0.1,
            sharpe_ratio=1.5,
            total_trades=10,
            win_rate=0.6,
            profit_factor=2.0,
            volatility=0.15,
            trade_history=[],
        )

        result_dict = result.to_dict()

        assert result_dict["portfolio"]["total_return"] == 0.2
        assert result_dict["risk_metrics"]["sharpe_ratio"] == 1.5
        assert result_dict["trading_stats"]["win_rate"] == 0.6
        assert result_dict["strategy_name"] == "rsi"
        assert "period" in result_dict
        assert "portfolio" in result_dict
        assert "risk_metrics" in result_dict
        assert "trading_stats" in result_dict


class TestPortfolioManager:
    """Portfolio execution tests for the current backtest engine."""

    def test_portfolio_manager_creation(self) -> None:
        config = BacktestConfig(initial_cash=10_000.0)
        portfolio = PortfolioManager(config)

        assert portfolio.config == config
        assert portfolio.cash == config.initial_cash
        assert portfolio.positions == {}
        assert portfolio.trades == []

    def test_buy_and_sell_signal_execution(self) -> None:
        config = BacktestConfig(
            initial_cash=20_000.0,
            max_position_size=0.5,
            cash_reserve_ratio=0.0,
            board_lot_size=1,
            commission_rate=0.001,
            slippage_rate=0.0,
        )
        portfolio = PortfolioManager(config)

        buy_signal = TradingSignal(
            timestamp=datetime(2024, 1, 1),
            stock_code="000001.SZ",
            signal_type=SignalType.BUY,
            strength=1.0,
            price=10.0,
            reason="unit test buy",
        )
        buy_trade, buy_error = portfolio.execute_signal(buy_signal, {"000001.SZ": 10.0})

        assert buy_error is None
        assert buy_trade is not None
        assert buy_trade.action == "BUY"
        assert "000001.SZ" in portfolio.positions
        assert portfolio.cash < config.initial_cash

        sell_signal = TradingSignal(
            timestamp=datetime(2024, 1, 2),
            stock_code="000001.SZ",
            signal_type=SignalType.SELL,
            strength=1.0,
            price=12.0,
            reason="unit test sell",
        )
        sell_trade, sell_error = portfolio.execute_signal(
            sell_signal, {"000001.SZ": 12.0}
        )

        assert sell_error is None
        assert sell_trade is not None
        assert sell_trade.action == "SELL"
        assert sell_trade.pnl > 0
        assert "000001.SZ" not in portfolio.positions
        assert len(portfolio.trades) == 2


class TestBacktestExecutor:
    """BacktestExecutor public contract tests."""

    def test_executor_creation(self) -> None:
        executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)

        stats = executor.get_execution_statistics()
        assert stats["total_backtests"] == 0
        assert stats["successful_backtests"] == 0
        assert isinstance(stats["available_strategies"], list)

    def test_validate_backtest_parameters(self) -> None:
        executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)

        assert (
            executor.validate_backtest_parameters(
                strategy_name="rsi",
                stock_codes=["000001.SZ"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 3, 1),
                strategy_config={"rsi_period": 14},
            )
            is True
        )

    @pytest.mark.asyncio
    async def test_run_backtest_returns_current_report_contract(self) -> None:
        executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)

        with _patch_executor_run(executor):
            result = await executor.run_backtest(
                strategy_name="rsi",
                stock_codes=["000001.SZ"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 3, 1),
                strategy_config={"rsi_period": 14},
                backtest_config=BacktestConfig(initial_cash=50_000.0),
            )

        assert result["strategy_name"] == "rsi"
        assert result["stock_codes"] == ["000001.SZ"]
        assert result["backtest_config"]["initial_cash"] == 50_000.0
        assert result["metrics"]["total_return"] == 0.12
        assert result["total_signals"] == 3
        assert result["trading_days"] == 20
        assert result["trade_history"][0]["stock_code"] == "000001.SZ"
        assert set(result["perf_breakdown"]).issuperset(
            {"data_loading_s", "main_loop_s", "metrics_s", "total_wall_s"}
        )

    @pytest.mark.asyncio
    async def test_run_backtest_supports_multiple_stocks(self) -> None:
        executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)
        stock_codes = ["000001.SZ", "000002.SZ"]

        with _patch_executor_run(executor, stock_codes=stock_codes):
            result = await executor.run_backtest(
                strategy_name="rsi",
                stock_codes=stock_codes,
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 3, 1),
                strategy_config={"rsi_period": 14},
            )

        assert result["stock_codes"] == stock_codes
        assert result["trading_days"] == 20
        assert isinstance(result["portfolio_history"], list)
