from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.services.backtest.strategies.strategy_factory import StrategyFactory


@pytest.mark.asyncio
async def test_model_topk_dropout_strategy_emits_ranking_scores_for_each_stock_date() -> None:
    strategy = StrategyFactory.create_strategy(
        "model_topk_dropout",
        {
            "model_id": "bank-core3",
            "topk": 2,
            "n_drop": 1,
            "horizon": "short_term",
            "score_scale": 100.0,
        },
    )

    trading_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
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
        ),
        "000002.SZ": pd.DataFrame(
            {
                "open": [20.0, 20.1, 20.2],
                "high": [20.1, 20.2, 20.3],
                "low": [19.9, 20.0, 20.1],
                "close": [20.0, 20.1, 20.2],
                "volume": [1000, 1000, 1000],
            },
            index=trading_dates,
        ),
    }
    stock_data["000001.SZ"].attrs["stock_code"] = "000001.SZ"
    stock_data["000002.SZ"].attrs["stock_code"] = "000002.SZ"

    predicted_returns = {
        "000001.SZ": pd.Series([0.03, 0.01, -0.02], index=trading_dates),
        "000002.SZ": pd.Series([0.01, 0.04, 0.02], index=trading_dates),
    }

    with patch(
        "app.services.backtest.strategies.model_prediction_base.PredictionEngine"
    ) as mock_engine_cls:
        mock_engine = mock_engine_cls.return_value
        mock_engine.predict_return_series = AsyncMock(
            side_effect=lambda stock_code, config, start_date, end_date: predicted_returns[stock_code]
        )

        await strategy.prepare_backtest_data(
            stock_data,
            start_date=datetime(2024, 1, 2),
            end_date=datetime(2024, 1, 4),
        )

    day_one_signal = strategy.generate_signals(
        stock_data["000001.SZ"], pd.Timestamp("2024-01-02")
    )
    day_two_signal = strategy.generate_signals(
        stock_data["000002.SZ"], pd.Timestamp("2024-01-03")
    )

    assert strategy.get_trade_mode() == "topk_dropout"
    assert strategy.get_trade_mode_config()["topk"] == 2
    assert strategy.get_trade_mode_config()["n_drop"] == 1

    assert len(day_one_signal) == 1
    assert day_one_signal[0].metadata["ranking_score"] == pytest.approx(0.03)
    assert day_one_signal[0].metadata["predicted_return"] == pytest.approx(0.03)
    assert day_one_signal[0].metadata["model_id"] == "bank-core3"

    assert len(day_two_signal) == 1
    assert day_two_signal[0].metadata["ranking_score"] == pytest.approx(0.04)
    assert day_two_signal[0].metadata["trade_mode"] == "topk_dropout"
