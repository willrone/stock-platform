"""Unit tests for backtest report builder extraction."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from app.services.backtest.reporting import (
    BacktestReportBuilder,
    BacktestReportBuildInput,
)


class _DummyPortfolioManager:
    """Minimal portfolio manager stub for report builder tests."""

    def __init__(self) -> None:
        self.trades = [
            {
                "trade_id": "trade-1",
                "stock_code": "000001.SZ",
                "action": "BUY",
                "quantity": 100,
                "price": 10.0,
                "timestamp": datetime(2024, 1, 2),
                "commission": 1.0,
                "slippage_cost": 0.2,
                "pnl": 0.0,
            },
            {
                "trade_id": "trade-2",
                "stock_code": "000001.SZ",
                "action": "SELL",
                "quantity": 100,
                "price": 11.0,
                "timestamp": datetime(2024, 2, 2),
                "commission": 1.0,
                "slippage_cost": 0.2,
                "pnl": 100.0,
            },
        ]
        self.portfolio_history = [
            {
                "date": datetime(2024, 1, 31),
                "portfolio_value": 101000.0,
                "portfolio_value_without_cost": 101050.0,
                "cash": 5000.0,
                "positions": {"000001.SZ": {"quantity": 100}},
            },
            {
                "date": datetime(2024, 2, 29),
                "portfolio_value": 102500.0,
                "portfolio_value_without_cost": 102600.0,
                "cash": 102500.0,
                "positions": {},
            },
        ]
        self.total_commission = 2.0
        self.total_slippage = 0.4

    def get_portfolio_value(self, _current_prices: dict[str, float]) -> float:
        """Return a fallback valuation."""
        return 99999.0

    def get_performance_metrics_without_cost(self) -> dict[str, float]:
        """Return minimal no-cost metrics."""
        return {
            "mean": 0.01,
            "std": 0.02,
            "annualized_return": 0.20,
            "information_ratio": 1.1,
            "max_drawdown": 0.05,
        }


@pytest.fixture
def build_input() -> BacktestReportBuildInput:
    """Create a reusable report build input."""
    config = SimpleNamespace(
        initial_cash=100000.0,
        commission_rate=0.0003,
        slippage_rate=0.001,
        max_position_size=0.2,
    )
    performance_metrics = {
        "total_return": 0.025,
        "annualized_return": 0.18,
        "volatility": 0.12,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.08,
        "total_trades": 2,
        "win_rate": 0.5,
        "profit_factor": 1.3,
        "winning_trades": 1,
        "losing_trades": 1,
    }
    return BacktestReportBuildInput(
        strategy_name="macd",
        stock_codes=["000001.SZ"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 2, 29),
        config=config,
        portfolio_manager=_DummyPortfolioManager(),
        performance_metrics=performance_metrics,
        strategy_config={"fast_period": 12, "slow_period": 26},
    )


def test_build_report_preserves_legacy_schema(build_input: BacktestReportBuildInput) -> None:
    """Builder should preserve legacy report payload shape."""
    builder = BacktestReportBuilder()

    report = builder.build_report(build_input)

    assert report["strategy_name"] == "macd"
    assert report["final_value"] == 102500.0
    assert report["metrics"]["sharpe_ratio"] == 1.5
    assert report["backtest_config"]["strategy_config"]["fast_period"] == 12
    assert report["trade_history"][1]["action"] == "SELL"
    assert report["portfolio_history"][0]["positions_count"] == 1
    assert report["cost_statistics"]["total_cost"] == pytest.approx(2.4)
    assert report["cost_statistics"]["cost_ratio"] == pytest.approx(0.000024)
    assert report["excess_return_without_cost"]["information_ratio"] == 1.1
    assert report["excess_return_with_cost"]["information_ratio"] == 1.5


def test_attach_runtime_diagnostics_keeps_loop_fields_normalized(
    build_input: BacktestReportBuildInput,
) -> None:
    """Runtime diagnostics should be attached without mutating input counters."""
    builder = BacktestReportBuilder()
    report = builder.build_report(build_input)
    perf_breakdown = {"report_generation_s": 0.2, "total_wall_s": 1.0}

    builder.attach_runtime_diagnostics(
        report,
        {"total_signals": 13, "trading_days": 20},
        perf_breakdown,
    )
    perf_breakdown["report_generation_s"] = 999

    assert report["total_signals"] == 13
    assert report["trading_days"] == 20
    assert report["perf_breakdown"]["report_generation_s"] == pytest.approx(0.2)


def test_attach_signal_execution_summary_fills_missing_defaults() -> None:
    """Signal summary normalization should match executor legacy defaults."""
    builder = BacktestReportBuilder()
    report: dict[str, object] = {}

    builder.attach_signal_execution_summary(
        report,
        {"executed_signal_count": 3, "top_rejection_reasons": [{"reason": "limit_up"}]},
    )

    assert report["signal_execution_summary"] == {
        "execution_rate": 0.0,
        "execution_rate_actionable": 0.0,
        "raw_signal_count": 0,
        "actionable_signal_count": 0,
        "executed_signal_count": 3,
        "top_rejection_reasons": [{"reason": "limit_up"}],
    }
