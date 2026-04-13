from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

api_package = ModuleType("app.api")
api_package.__path__ = [str(BACKEND_ROOT / "app" / "api")]
v1_package = ModuleType("app.api.v1")
v1_package.__path__ = [str(BACKEND_ROOT / "app" / "api" / "v1")]
sys.modules.setdefault("app.api", api_package)
sys.modules.setdefault("app.api.v1", v1_package)

from app.api.v1.backtest import run_backtest
from app.api.v1.schemas import BacktestRequest


@pytest.mark.asyncio
async def test_run_backtest_model_request_injects_model_id_into_strategy_config() -> None:
    request = BacktestRequest(
        strategy_name="model",
        model_id="bank-core3",
        stock_codes=["000001.SZ"],
        start_date="2024-01-01T00:00:00",
        end_date="2024-02-15T00:00:00",
        initial_cash=100000.0,
    )

    executor = MagicMock()
    executor.validate_backtest_parameters = MagicMock(return_value=True)
    executor.run_backtest = AsyncMock(
        return_value={
            "strategy_name": "model_signal",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-02-15T00:00:00",
            "initial_cash": 100000.0,
            "final_value": 108000.0,
            "total_return": 0.08,
            "annualized_return": 0.24,
            "volatility": 0.12,
            "sharpe_ratio": 1.1,
            "max_drawdown": -0.05,
            "total_trades": 4,
            "win_rate": 0.5,
            "profit_factor": 1.4,
            "portfolio_history": [
                {"date": "2024-01-02", "portfolio_value": 100000.0},
                {"date": "2024-02-15", "portfolio_value": 108000.0},
            ],
            "trade_history": [],
        }
    )

    with patch("app.api.v1.backtest.BacktestExecutor", return_value=executor), patch(
        "app.api.v1.backtest.BacktestConfig",
        side_effect=lambda **kwargs: type("DummyBacktestConfig", (), kwargs)(),
    ):
        response = await run_backtest(request)

    executor.validate_backtest_parameters.assert_called_once()
    validate_kwargs = executor.validate_backtest_parameters.call_args.kwargs
    assert validate_kwargs["strategy_name"] == "model_signal"
    assert validate_kwargs["strategy_config"]["model_id"] == "bank-core3"

    run_kwargs = executor.run_backtest.await_args.kwargs
    assert run_kwargs["strategy_name"] == "model_signal"
    assert run_kwargs["strategy_config"]["model_id"] == "bank-core3"
    assert response.success is True
    assert response.data["strategy_name"] == "model_signal"


@pytest.mark.asyncio
async def test_run_backtest_topk_dropout_alias_normalizes_to_model_topk_dropout() -> None:
    request = BacktestRequest(
        strategy_name="topk_dropout",
        model_id="bank-core3",
        stock_codes=["000001.SZ", "000002.SZ"],
        start_date="2024-01-01T00:00:00",
        end_date="2024-02-15T00:00:00",
        initial_cash=100000.0,
        strategy_config={
            "topk": 2,
            "n_drop": 1,
            "benchmark": "SH000300",
        },
    )

    executor = MagicMock()
    executor.validate_backtest_parameters = MagicMock(return_value=True)
    executor.run_backtest = AsyncMock(
        return_value={
            "strategy_name": "model_topk_dropout",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-02-15T00:00:00",
            "initial_cash": 100000.0,
            "final_value": 108000.0,
            "total_return": 0.08,
            "annualized_return": 0.24,
            "volatility": 0.12,
            "sharpe_ratio": 1.1,
            "max_drawdown": -0.05,
            "total_trades": 4,
            "win_rate": 0.5,
            "profit_factor": 1.4,
            "portfolio_history": [],
            "trade_history": [],
        }
    )

    with patch("app.api.v1.backtest.BacktestExecutor", return_value=executor), patch(
        "app.api.v1.backtest.BacktestConfig",
        side_effect=lambda **kwargs: type("DummyBacktestConfig", (), kwargs)(),
    ):
        response = await run_backtest(request)

    validate_kwargs = executor.validate_backtest_parameters.call_args.kwargs
    assert validate_kwargs["strategy_name"] == "model_topk_dropout"
    assert validate_kwargs["strategy_config"]["model_id"] == "bank-core3"
    assert validate_kwargs["strategy_config"]["topk"] == 2
    assert validate_kwargs["strategy_config"]["n_drop"] == 1
    assert validate_kwargs["strategy_config"]["benchmark"] == "SH000300"

    run_kwargs = executor.run_backtest.await_args.kwargs
    assert run_kwargs["strategy_name"] == "model_topk_dropout"
    assert run_kwargs["strategy_config"]["model_id"] == "bank-core3"
    assert run_kwargs["strategy_config"]["topk"] == 2
    assert run_kwargs["strategy_config"]["n_drop"] == 1
    assert response.success is True
    assert response.data["strategy_name"] == "model_topk_dropout"


@pytest.mark.asyncio
async def test_run_backtest_propagates_runtime_portfolio_constraints() -> None:
    request = BacktestRequest(
        strategy_name="model_topk_dropout",
        model_id="bank-core3",
        stock_codes=["000001.SZ", "000002.SZ"],
        start_date="2024-01-01T00:00:00",
        end_date="2024-02-15T00:00:00",
        initial_cash=100000.0,
        strategy_config={
            "topk": 2,
            "n_drop": 1,
            "benchmark": "SH000300",
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
            "max_position_size": 0.1,
            "cash_reserve_ratio": 0.2,
            "board_lot_size": 200,
        },
    )

    executor = MagicMock()
    executor.validate_backtest_parameters = MagicMock(return_value=True)
    executor.run_backtest = AsyncMock(
        return_value={
            "strategy_name": "model_topk_dropout",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-02-15T00:00:00",
            "initial_cash": 100000.0,
            "final_value": 108000.0,
            "total_return": 0.08,
            "annualized_return": 0.24,
            "volatility": 0.12,
            "sharpe_ratio": 1.1,
            "max_drawdown": -0.05,
            "total_trades": 4,
            "win_rate": 0.5,
            "profit_factor": 1.4,
            "portfolio_history": [],
            "trade_history": [],
        }
    )

    with patch("app.api.v1.backtest.BacktestExecutor", return_value=executor), patch(
        "app.api.v1.backtest.BacktestConfig",
        side_effect=lambda **kwargs: type("DummyBacktestConfig", (), kwargs)(),
    ):
        response = await run_backtest(request)

    runtime_cfg = executor.run_backtest.await_args.kwargs["backtest_config"]
    assert runtime_cfg.commission_rate == 0.001
    assert runtime_cfg.slippage_rate == 0.0005
    assert runtime_cfg.max_position_size == 0.1
    assert runtime_cfg.cash_reserve_ratio == 0.2
    assert runtime_cfg.board_lot_size == 200
    assert response.success is True
