from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.services.backtest.execution.data_loader import DataLoader


def _build_price_frame(start: str, periods: int) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=periods)
    prices = list(range(100, 100 + periods))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1_000_000] * periods,
        },
        index=dates,
    )


def test_is_data_valid_accepts_fully_covered_short_window() -> None:
    loader = DataLoader(data_dir="/tmp", max_workers=None)
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 2, 10)
    data = _build_price_frame("2026-01-05", periods=27)

    assert loader._is_data_valid(data, start_date, end_date) is True
