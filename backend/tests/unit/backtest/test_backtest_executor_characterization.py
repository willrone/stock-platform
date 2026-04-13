"""backtest_executor characterization tests.

锁定 run_backtest 当前对外可见的结果聚合行为，作为后续拆分 executor
前的回归护栏。
"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.execution import backtest_executor as executor_module
from app.services.backtest.execution.backtest_executor import BacktestExecutor


class _AsyncSessionContext:
    """Minimal async context manager for repository mocks."""

    def __init__(self, session: object) -> None:
        self._session = session

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _build_stock_data(trading_dates: list[datetime]) -> dict[str, pd.DataFrame]:
    """Create one minimal stock dataset for run_backtest scaffolding."""
    close_prices = list(range(100, 100 + len(trading_dates)))
    stock_frame = pd.DataFrame(
        {
            "open": close_prices,
            "high": close_prices,
            "low": close_prices,
            "close": close_prices,
            "volume": [1_000_000] * len(trading_dates),
        },
        index=pd.DatetimeIndex(trading_dates),
    )
    return {"000001.SZ": stock_frame}


def _build_report_template() -> dict[str, object]:
    """Return the base report payload produced before aggregation patches."""
    return {
        "strategy_name": "macd",
        "metrics": {"total_return": 0.12, "sharpe_ratio": 1.5},
        "backtest_config": {"initial_cash": 100000.0},
        "trade_history": [{"stock_code": "000001.SZ", "action": "BUY"}],
        "portfolio_history": [{"date": "2024-01-01", "value": 100000.0}],
        "cost_statistics": {"total_commission": 12.3},
    }


def _build_progress_monitor() -> MagicMock:
    """Create an async-friendly progress monitor stub."""
    progress_monitor = MagicMock()
    progress_monitor.start_backtest_monitoring = AsyncMock()
    progress_monitor.update_stage = AsyncMock()
    progress_monitor.complete_backtest = AsyncMock()
    progress_monitor.set_error = AsyncMock()
    progress_monitor.get_progress_data.return_value = SimpleNamespace(
        total_trading_days=0
    )
    return progress_monitor


def _run_backtest_patch_stack(
    executor: BacktestExecutor,
    *,
    loop_result: dict[str, int],
    report_template: dict[str, object],
    progress_monitor: MagicMock | None = None,
    signal_stats: dict[str, object] | None = None,
    signal_error: Exception | None = None,
) -> ExitStack:
    """Patch run_backtest dependencies so tests only lock aggregation behavior."""
    trading_dates = pd.bdate_range("2024-01-01", periods=20).to_pydatetime().tolist()
    stock_data = _build_stock_data(trading_dates)
    portfolio_manager = Mock()
    portfolio_manager.get_performance_metrics.return_value = {
        "total_return": 0.12,
        "sharpe_ratio": 1.5,
    }

    signal_repo = MagicMock()
    if signal_error is None:
        signal_repo.get_signal_statistics = AsyncMock(return_value=signal_stats or {})
    else:
        signal_repo.get_signal_statistics = AsyncMock(side_effect=signal_error)

    task_repo = MagicMock()
    task_repo.get_task_by_id.return_value = SimpleNamespace(result={})

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
    stack.enter_context(
        patch.object(executor, "_get_trading_calendar", return_value=trading_dates)
    )
    stack.enter_context(patch.object(executor, "_build_date_index"))
    stack.enter_context(patch.object(executor, "_precompute_strategy_signals"))
    stack.enter_context(
        patch.object(executor, "_extract_precomputed_signals_to_dict", return_value={})
    )
    stack.enter_context(
        patch.object(executor, "_build_aligned_arrays", return_value={})
    )
    stack.enter_context(
        patch.object(executor, "_execute_backtest_loop", AsyncMock(return_value=loop_result))
    )
    stack.enter_context(
        patch.object(
            executor.report_builder,
            "build_report",
            return_value=report_template.copy(),
        )
    )
    stack.enter_context(
        patch.object(executor_module, "PortfolioManagerArray", return_value=portfolio_manager)
    )

    if progress_monitor is not None:
        stack.enter_context(
            patch.object(executor_module, "backtest_progress_monitor", progress_monitor)
        )
        stack.enter_context(patch("app.core.database.SessionLocal", return_value=MagicMock()))
        stack.enter_context(
            patch("app.repositories.task_repository.TaskRepository", return_value=task_repo)
        )
        stack.enter_context(
            patch(
                "app.core.database.get_async_session_context",
                return_value=_AsyncSessionContext(MagicMock()),
            )
        )
        stack.enter_context(
            patch(
                "app.repositories.backtest_detailed_repository.BacktestDetailedRepository",
                return_value=signal_repo,
            )
        )

    return stack


@pytest.mark.asyncio
async def test_run_backtest_preserves_characterization_fields_without_task_id() -> None:
    """锁定 run_backtest 直接返回的关键聚合字段。"""
    executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)
    loop_result = {"total_signals": 11, "trading_days": 20, "executed_trades": 7}

    with _run_backtest_patch_stack(
        executor,
        loop_result=loop_result,
        report_template=_build_report_template(),
    ):
        result = await executor.run_backtest(
            strategy_name="macd",
            stock_codes=["000001.SZ"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_config={"fast_period": 12, "slow_period": 26},
        )

    assert result["metrics"]["total_return"] == 0.12
    assert result["trade_history"][0]["stock_code"] == "000001.SZ"
    assert result["portfolio_history"][0]["value"] == 100000.0
    assert result["total_signals"] == 11
    assert result["trading_days"] == 20
    assert "signal_execution_summary" not in result
    assert set(result["perf_breakdown"]).issuperset(
        {
            "strategy_setup_s",
            "data_loading_s",
            "precompute_signals_s",
            "align_arrays_s",
            "main_loop_s",
            "metrics_s",
            "report_generation_s",
            "total_wall_s",
        }
    )
    assert result["perf_breakdown"]["total_wall_s"] > 0


@pytest.mark.asyncio
async def test_run_backtest_with_task_id_adds_signal_execution_summary() -> None:
    """锁定 task_id 存在时 signal_execution_summary 的透传行为。"""
    executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)
    progress_monitor = _build_progress_monitor()
    loop_result = {"total_signals": 13, "trading_days": 20, "executed_trades": 5}
    signal_stats = {
        "execution_rate": 0.5,
        "execution_rate_actionable": 0.8,
        "raw_signal_count": 20,
        "actionable_signal_count": 10,
        "executed_signal_count": 8,
        "top_rejection_reasons": [{"reason": "limit_up", "count": 2}],
    }

    with _run_backtest_patch_stack(
        executor,
        loop_result=loop_result,
        report_template=_build_report_template(),
        progress_monitor=progress_monitor,
        signal_stats=signal_stats,
    ):
        result = await executor.run_backtest(
            strategy_name="macd",
            stock_codes=["000001.SZ"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_config={"fast_period": 12, "slow_period": 26},
            task_id="task-446",
        )

    assert result["total_signals"] == 13
    assert result["trading_days"] == 20
    assert result["signal_execution_summary"] == signal_stats
    progress_monitor.start_backtest_monitoring.assert_awaited_once()
    assert progress_monitor.update_stage.await_count >= 8
    progress_monitor.complete_backtest.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_backtest_with_task_id_falls_back_to_empty_signal_summary() -> None:
    """锁定信号统计查询失败时降级为空对象的当前行为。"""
    executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)
    progress_monitor = _build_progress_monitor()
    loop_result = {"total_signals": 9, "trading_days": 20, "executed_trades": 3}

    with _run_backtest_patch_stack(
        executor,
        loop_result=loop_result,
        report_template=_build_report_template(),
        progress_monitor=progress_monitor,
        signal_error=RuntimeError("repo unavailable"),
    ):
        result = await executor.run_backtest(
            strategy_name="macd",
            stock_codes=["000001.SZ"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            strategy_config={"fast_period": 12, "slow_period": 26},
            task_id="task-446",
        )

    assert result["total_signals"] == 9
    assert result["trading_days"] == 20
    assert result["signal_execution_summary"] == {}


def test_topk_dropout_price_lookup_includes_non_held_candidates_without_precomputed_signal_matrix() -> None:
    """Ranking rebalance must still load candidate prices even when only current holdings are in the portfolio."""
    executor = BacktestExecutor(data_dir="/tmp", enable_parallel=False)
    portfolio_manager = Mock()
    portfolio_manager.positions = {"601398.SH": object(), "601288.SH": object()}

    strategy = Mock()
    strategy.get_trade_mode.return_value = "topk_dropout"

    aligned_arrays = {
        "stock_codes": ["600036.SH", "601288.SH", "601398.SH"],
        "signal": np.zeros((3, 1), dtype=np.int8),
    }

    need_codes = executor._determine_price_lookup_codes(
        strategy=strategy,
        portfolio_manager=portfolio_manager,
        aligned_arrays=aligned_arrays,
        date_index=0,
    )

    assert need_codes == {"600036.SH", "601288.SH", "601398.SH"}
