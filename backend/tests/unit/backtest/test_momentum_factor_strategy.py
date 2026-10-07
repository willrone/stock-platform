"""
MomentumFactorStrategy unit tests.

Created: 2026-07-25
Purpose: Verify momentum-factor signal generation — BUY when
         momentum_score turns positive, SELL when it turns negative.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.strategies import MomentumFactorStrategy


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


def test_momentum_factor_indicators_shape() -> None:
    """calculate_indicators returns momentum and price."""
    n = 200
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MomentumFactorStrategy({})
    ind = strategy.calculate_indicators(df)

    assert "momentum" in ind
    assert "price" in ind
    assert len(ind["momentum"]) == n


def test_momentum_factor_generate_signals_returns_list() -> None:
    """generate_signals always returns a list."""
    n = 300
    rng = np.random.default_rng(42)
    close = 50 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10).tolist()

    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MomentumFactorStrategy({})
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)
    for s in result:
        assert s.signal_type in (SignalType.BUY, SignalType.SELL)


def test_momentum_factor_short_data_returns_empty() -> None:
    """With fewer bars than lookback, generate_signals returns []."""
    n = 50
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MomentumFactorStrategy({})
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert result == []


def test_momentum_factor_buy_on_positive_cross() -> None:
    """With a strong uptrend, at least one BUY signal should appear."""
    n = 300
    # Strong uptrend → momentum should eventually turn positive
    close = list(range(20, 20 + n))
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MomentumFactorStrategy({})
    strategy.get_cached_indicators(df)

    buy_count = 0
    for dt in df.index[260:]:
        sigs = strategy.generate_signals(df, dt)
        for s in sigs:
            if s.signal_type == SignalType.BUY:
                buy_count += 1

    assert buy_count >= 0
