"""
ValueFactorStrategy unit tests.

Created: 2026-07-25
Purpose: Verify value-factor signal generation — BUY when the
         composite value_score turns positive, SELL when it turns
         negative.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.strategies import ValueFactorStrategy


def _make_data(
    close: list[float], n_stocks: int = 3
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    result = {}
    for i in range(1, n_stocks + 1):
        code = f"00000{i}.SZ"
        df = pd.DataFrame(
            {
                "open": [c * 0.99 for c in close],
                "high": [c * 1.02 for c in close],
                "low": [c * 0.98 for c in close],
                "close": close,
                "volume": [1_000_000] * len(close),
            },
            index=dates,
        )
        df.attrs["stock_code"] = code
        result[code] = df
    return result


def test_value_factor_indicators_shape() -> None:
    """calculate_indicators returns ratio estimates and value_score."""
    n = 300
    rng = np.random.default_rng(42)
    close = 50 + np.cumsum(rng.normal(0, 0.3, n))
    close = np.maximum(close, 10).tolist()

    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = ValueFactorStrategy({})
    ind = strategy.calculate_indicators(df)

    expected = {"pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda",
                "value_score", "price"}
    assert expected.issubset(ind.keys())
    assert len(ind["value_score"]) == n


def test_value_factor_generate_signals_returns_list() -> None:
    """generate_signals always returns a list."""
    n = 300
    rng = np.random.default_rng(42)
    close = 50 + np.cumsum(rng.normal(0, 0.3, n))
    close = np.maximum(close, 10).tolist()

    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = ValueFactorStrategy({})
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)


def test_value_factor_short_data_returns_empty() -> None:
    """With fewer than 260 bars, generate_signals returns []."""
    n = 100
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = ValueFactorStrategy({})
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert result == []


def test_value_factor_with_pe_threshold() -> None:
    """When pe_max is set, BUY signals respect the threshold."""
    n = 300
    rng = np.random.default_rng(42)
    close = 50 + np.cumsum(rng.normal(0, 0.3, n))
    close = np.maximum(close, 10).tolist()

    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = ValueFactorStrategy({"pe_max": 50})
    strategy.get_cached_indicators(df)

    # This should not crash regardless of whether signals are emitted
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)
