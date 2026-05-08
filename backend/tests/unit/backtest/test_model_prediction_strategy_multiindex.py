from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.services.backtest.strategies.strategy_factory import StrategyFactory


@pytest.mark.asyncio
async def test_model_prediction_strategy_flattens_multiindex_prediction_series_for_single_stock() -> (
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

    trading_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    data = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2],
            "high": [10.1, 10.2, 10.3],
            "low": [9.9, 10.0, 10.1],
            "close": [10.0, 10.1, 10.2],
            "volume": [1000, 1000, 1000],
        },
        index=trading_dates,
    )
    data.attrs["stock_code"] = "000001.SZ"
    stock_data = {"000001.SZ": data}

    multiindex_returns = pd.Series(
        [0.03, 0.0, -0.04],
        index=pd.MultiIndex.from_product(
            [["000001.SZ"], trading_dates], names=["instrument", "datetime"]
        ),
    )

    with patch(
        "app.services.backtest.strategies.model_prediction_base.PredictionEngine"
    ) as mock_engine_cls:
        mock_engine = mock_engine_cls.return_value
        mock_engine.predict_return_series = AsyncMock(return_value=multiindex_returns)

        await strategy.prepare_backtest_data(
            stock_data,
            start_date=datetime(2024, 1, 2),
            end_date=datetime(2024, 1, 4),
        )

    precomputed = strategy.precompute_all_signals(stock_data["000001.SZ"])

    assert precomputed.loc[pd.Timestamp("2024-01-02")] is not None
    assert precomputed.loc[pd.Timestamp("2024-01-04")] is not None
