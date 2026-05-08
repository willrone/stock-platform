from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.strategy_factory import StrategyFactory


@pytest.mark.asyncio
async def test_model_prediction_strategy_prepares_predictions_and_emits_threshold_signals() -> (
    None
):
    strategy = StrategyFactory.create_strategy(
        "model_signal",
        {
            "model_id": "bank-core3",
            "buy_threshold": 0.02,
            "sell_threshold": -0.02,
            "horizon": "short_term",
        },
    )

    trading_dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ]
    )
    stock_data = {
        "000001.SZ": pd.DataFrame(
            {
                "open": [10.0, 10.1, 10.2],
                "high": [10.1, 10.2, 10.3],
                "low": [9.9, 10.0, 10.1],
                "close": [10.0, 10.1, 10.2],
                "volume": [1000, 1000, 1000],
            },
            index=trading_dates,
        )
    }
    stock_data["000001.SZ"].attrs["stock_code"] = "000001.SZ"

    mocked_returns = pd.Series([0.03, 0.0, -0.04], index=trading_dates)

    with patch(
        "app.services.backtest.strategies.model_prediction_base.PredictionEngine"
    ) as mock_engine_cls:
        mock_engine = mock_engine_cls.return_value
        mock_engine.predict_return_series = AsyncMock(return_value=mocked_returns)

        await strategy.prepare_backtest_data(
            stock_data,
            start_date=datetime(2024, 1, 2),
            end_date=datetime(2024, 1, 4),
        )

    precomputed = strategy.precompute_all_signals(stock_data["000001.SZ"])

    assert precomputed.loc[pd.Timestamp("2024-01-02")] == SignalType.BUY
    assert pd.isna(precomputed.loc[pd.Timestamp("2024-01-03")])
    assert precomputed.loc[pd.Timestamp("2024-01-04")] == SignalType.SELL

    buy_signals = strategy.generate_signals(
        stock_data["000001.SZ"], pd.Timestamp("2024-01-02")
    )
    sell_signals = strategy.generate_signals(
        stock_data["000001.SZ"], pd.Timestamp("2024-01-04")
    )

    assert buy_signals[0].signal_type == SignalType.BUY
    assert buy_signals[0].metadata["model_id"] == "bank-core3"
    assert buy_signals[0].metadata["predicted_return"] == pytest.approx(0.03)
    assert sell_signals[0].signal_type == SignalType.SELL
    assert sell_signals[0].metadata["predicted_return"] == pytest.approx(-0.04)


@pytest.mark.asyncio
async def test_model_prediction_strategy_only_emits_on_state_changes() -> None:
    strategy = StrategyFactory.create_strategy(
        "model_signal",
        {
            "model_id": "bank-core3",
            "buy_threshold": 0.02,
            "sell_threshold": -0.02,
            "horizon": "short_term",
        },
    )

    trading_dates = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    )
    stock_data = {
        "000001.SZ": pd.DataFrame(
            {
                "open": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
                "high": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
                "low": [9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
                "close": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
                "volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000],
            },
            index=trading_dates,
        )
    }
    stock_data["000001.SZ"].attrs["stock_code"] = "000001.SZ"

    mocked_returns = pd.Series(
        [0.03, 0.04, 0.01, -0.03, -0.04, 0.0, 0.05],
        index=trading_dates,
    )

    with patch(
        "app.services.backtest.strategies.model_prediction_base.PredictionEngine"
    ) as mock_engine_cls:
        mock_engine = mock_engine_cls.return_value
        mock_engine.predict_return_series = AsyncMock(return_value=mocked_returns)

        await strategy.prepare_backtest_data(
            stock_data,
            start_date=datetime(2024, 1, 2),
            end_date=datetime(2024, 1, 10),
        )

    precomputed = strategy.precompute_all_signals(stock_data["000001.SZ"])

    expected = {
        pd.Timestamp("2024-01-02"): SignalType.BUY,
        pd.Timestamp("2024-01-03"): None,
        pd.Timestamp("2024-01-04"): None,
        pd.Timestamp("2024-01-05"): SignalType.SELL,
        pd.Timestamp("2024-01-08"): None,
        pd.Timestamp("2024-01-09"): None,
        pd.Timestamp("2024-01-10"): SignalType.BUY,
    }
    for date, signal_type in expected.items():
        actual = precomputed.loc[date]
        if signal_type is None:
            assert pd.isna(actual)
        else:
            assert actual == signal_type

    signal_map = {
        date: strategy.generate_signals(stock_data["000001.SZ"], date)
        for date in trading_dates
    }

    assert signal_map[pd.Timestamp("2024-01-02")][0].signal_type == SignalType.BUY
    assert signal_map[pd.Timestamp("2024-01-03")] == []
    assert signal_map[pd.Timestamp("2024-01-04")] == []
    assert signal_map[pd.Timestamp("2024-01-05")][0].signal_type == SignalType.SELL
    assert signal_map[pd.Timestamp("2024-01-08")] == []
    assert signal_map[pd.Timestamp("2024-01-09")] == []
    assert signal_map[pd.Timestamp("2024-01-10")][0].signal_type == SignalType.BUY
