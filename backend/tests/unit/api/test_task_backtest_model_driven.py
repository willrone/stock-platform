from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

api_package = ModuleType("app.api")
api_package.__path__ = [str(BACKEND_ROOT / "app" / "api")]
v1_package = ModuleType("app.api.v1")
v1_package.__path__ = [str(BACKEND_ROOT / "app" / "api" / "v1")]
sys.modules.setdefault("app.api", api_package)
sys.modules.setdefault("app.api.v1", v1_package)

from app.api.v1.dependencies import execute_backtest_task_simple


class _DummySession:
    def close(self) -> None:
        return None


class _FakeLoop:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode

    def run_until_complete(self, coro):
        if self.mode == "run":
            return asyncio.run(coro)
        coro.close()
        return None

    def close(self) -> None:
        return None


def test_execute_backtest_task_injects_top_level_model_id_into_strategy_config() -> None:
    task = SimpleNamespace(
        task_id="task-model-driven-1",
        config={
            "stock_codes": ["600036.SH", "601288.SH", "601398.SH"],
            "strategy_name": "model_signal",
            "model_id": "bank-core3-model",
            "start_date": "2025-01-01T00:00:00",
            "end_date": "2025-12-31T00:00:00",
            "initial_cash": 100000.0,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
            "strategy_config": {
                "horizon": "short_term",
                "buy_threshold": 0.0005,
                "sell_threshold": -0.0005,
            },
        },
    )

    repository = MagicMock()
    repository.get_task_by_id.return_value = task
    repository.update_task_status = MagicMock()

    executor = MagicMock()
    executor.run_backtest = AsyncMock(
        return_value={
            "strategy_name": "model_signal",
            "total_return": 0.1,
            "trade_history": [],
            "portfolio_history": [],
        }
    )

    with patch("app.api.v1.dependencies.SessionLocal", return_value=_DummySession()), patch(
        "app.api.v1.dependencies.TaskRepository", return_value=repository
    ), patch("app.services.backtest.BacktestExecutor", return_value=executor), patch(
        "app.services.backtest.BacktestConfig",
        side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
    ), patch("nest_asyncio.apply", return_value=None), patch(
        "asyncio.new_event_loop",
        side_effect=[_FakeLoop(mode="run"), _FakeLoop(mode="skip")],
    ), patch("asyncio.set_event_loop", return_value=None):
        execute_backtest_task_simple("task-model-driven-1")

    run_kwargs = executor.run_backtest.await_args.kwargs
    assert run_kwargs["strategy_name"] == "model_signal"
    assert run_kwargs["strategy_config"]["model_id"] == "bank-core3-model"
    assert run_kwargs["strategy_config"]["horizon"] == "short_term"
    assert run_kwargs["strategy_config"]["buy_threshold"] == 0.0005
    assert run_kwargs["strategy_config"]["sell_threshold"] == -0.0005


def test_execute_backtest_task_normalizes_model_topk_dropout_alias() -> None:
    task = SimpleNamespace(
        task_id="task-model-ranking-1",
        config={
            "stock_codes": ["600036.SH", "601288.SH", "601398.SH"],
            "strategy_name": "topk_dropout",
            "model_id": "bank-core3-model",
            "start_date": "2025-01-01T00:00:00",
            "end_date": "2025-12-31T00:00:00",
            "initial_cash": 100000.0,
            "commission_rate": 0.0003,
            "slippage_rate": 0.0001,
            "strategy_config": {
                "horizon": "short_term",
                "topk": 2,
                "n_drop": 1,
                "benchmark": "SH000300",
            },
        },
    )

    repository = MagicMock()
    repository.get_task_by_id.return_value = task
    repository.update_task_status = MagicMock()

    executor = MagicMock()
    executor.run_backtest = AsyncMock(
        return_value={
            "strategy_name": "model_topk_dropout",
            "total_return": 0.1,
            "trade_history": [],
            "portfolio_history": [],
        }
    )

    with patch("app.api.v1.dependencies.SessionLocal", return_value=_DummySession()), patch(
        "app.api.v1.dependencies.TaskRepository", return_value=repository
    ), patch("app.services.backtest.BacktestExecutor", return_value=executor), patch(
        "app.services.backtest.BacktestConfig",
        side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
    ), patch("nest_asyncio.apply", return_value=None), patch(
        "asyncio.new_event_loop",
        side_effect=[_FakeLoop(mode="run"), _FakeLoop(mode="skip")],
    ), patch("asyncio.set_event_loop", return_value=None):
        execute_backtest_task_simple("task-model-ranking-1")

    run_kwargs = executor.run_backtest.await_args.kwargs
    assert run_kwargs["strategy_name"] == "model_topk_dropout"
    assert run_kwargs["strategy_config"]["model_id"] == "bank-core3-model"
    assert run_kwargs["strategy_config"]["topk"] == 2
    assert run_kwargs["strategy_config"]["n_drop"] == 1
    assert run_kwargs["strategy_config"]["benchmark"] == "SH000300"
