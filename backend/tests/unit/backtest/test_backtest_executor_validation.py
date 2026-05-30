"""BacktestExecutor parameter validation contract tests."""

from __future__ import annotations

from datetime import datetime

from app.services.backtest import BacktestExecutor


def test_validate_backtest_parameters_accepts_portfolio_strategy_config() -> None:
    executor = BacktestExecutor(data_dir="../data", enable_parallel=False)

    assert executor.validate_backtest_parameters(
        strategy_name="portfolio",
        stock_codes=["000001.SZ"],
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 2, 2),
        strategy_config={
            "strategies": [
                {
                    "name": "moving_average",
                    "weight": 1.0,
                    "config": {
                        "short_window": 5,
                        "long_window": 20,
                        "signal_threshold": 0.005,
                    },
                }
            ],
            "integration_method": "weighted_voting",
        },
    )
