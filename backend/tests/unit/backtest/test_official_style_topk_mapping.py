from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

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

from app.api.v1.backtest import _normalize_backtest_strategy_request
from app.api.v1.dependencies import _normalize_task_backtest_strategy_config
from app.api.v1.schemas import BacktestRequest
from app.services.backtest.utils.official_style_params import (
    apply_official_style_topk_dropout_params,
    derive_official_style_topk_dropout_params,
)


@pytest.mark.parametrize(
    ("pool_size", "expected"),
    [
        (3, {"topk": 2, "n_drop": 1, "hold_thresh": 0}),
        (5, {"topk": 2, "n_drop": 1, "hold_thresh": 1}),
        (15, {"topk": 3, "n_drop": 1, "hold_thresh": 2}),
        (20, {"topk": 4, "n_drop": 1, "hold_thresh": 2}),
    ],
)
def test_derive_official_style_topk_dropout_params(pool_size: int, expected: dict[str, int]) -> None:
    assert derive_official_style_topk_dropout_params(pool_size) == expected


def test_apply_official_style_topk_dropout_params_fills_missing_ranking_parameters() -> None:
    strategy_config = apply_official_style_topk_dropout_params(
        strategy_name="topk_dropout",
        stock_codes=["600036.SH", "601288.SH", "600519.SH", "000001.SZ", "000651.SZ"],
        strategy_config={"official_style": True},
    )

    assert strategy_config["topk"] == 2
    assert strategy_config["n_drop"] == 1
    assert strategy_config["hold_thresh"] == 1
    assert strategy_config["official_style"] is True


def test_apply_official_style_topk_dropout_params_preserves_explicit_values() -> None:
    strategy_config = apply_official_style_topk_dropout_params(
        strategy_name="model_topk_dropout",
        stock_codes=["600036.SH", "601288.SH", "600519.SH", "000001.SZ", "000651.SZ"],
        strategy_config={
            "official_style": True,
            "topk": 3,
            "n_drop": 2,
            "hold_thresh": 0,
        },
    )

    assert strategy_config["topk"] == 3
    assert strategy_config["n_drop"] == 2
    assert strategy_config["hold_thresh"] == 0


def test_apply_official_style_topk_dropout_params_ignores_non_ranking_strategies() -> None:
    strategy_config = apply_official_style_topk_dropout_params(
        strategy_name="model_signal",
        stock_codes=["600036.SH", "601288.SH", "600519.SH"],
        strategy_config={"official_style": True},
    )

    assert strategy_config == {"official_style": True}


def test_normalize_backtest_request_applies_official_style_mapping() -> None:
    strategy_name, strategy_config = _normalize_backtest_strategy_request(
        BacktestRequest(
            strategy_name="topk_dropout",
            stock_codes=["600036.SH", "601288.SH", "600519.SH", "000001.SZ", "000651.SZ"],
            start_date="2020-01-01T00:00:00",
            end_date="2020-08-01T00:00:00",
            model_id="alpha158-model",
            strategy_config={"official_style": True},
        )
    )

    assert strategy_name == "model_topk_dropout"
    assert strategy_config["model_id"] == "alpha158-model"
    assert strategy_config["topk"] == 2
    assert strategy_config["n_drop"] == 1
    assert strategy_config["hold_thresh"] == 1


def test_normalize_task_backtest_strategy_config_applies_official_style_mapping() -> None:
    strategy_name, strategy_config = _normalize_task_backtest_strategy_config(
        {
            "strategy_name": "topk_dropout",
            "stock_codes": ["600036.SH", "601288.SH", "600519.SH", "000001.SZ", "000651.SZ"],
            "model_id": "alpha360-model",
            "strategy_config": {"official_style": True},
        }
    )

    assert strategy_name == "model_topk_dropout"
    assert strategy_config["model_id"] == "alpha360-model"
    assert strategy_config["topk"] == 2
    assert strategy_config["n_drop"] == 1
    assert strategy_config["hold_thresh"] == 1
