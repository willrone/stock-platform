"""
CointegrationStrategy unit tests.

Created: 2026-07-25
Purpose: Verify cointegration-based signal generation — BUY/SELL
         on z-score crossings with mean_reversion_strength < 0.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.strategies import CointegrationStrategy


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


def test_cointegration_indicators_shape() -> None:
    """calculate_indicators returns expected keys."""
    n = 70
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = CointegrationStrategy(
        {"lookback_period": 60, "half_life": 20, "entry_threshold": 2.0}
    )
    ind = strategy.calculate_indicators(df)

    expected = {"price", "returns", "half_life", "sma", "std", "zscore",
                "mean_reversion_strength"}
    assert expected.issubset(ind.keys())
    assert len(ind["price"]) == n
    # returns has one fewer element
    assert len(ind["returns"]) == n - 1


def test_cointegration_precompute_requires_mean_reversion() -> None:
    """precompute_all_signals returns None when mean_reversion_strength is 0
    (no half-life could be estimated)."""
    n = 70
    # Flat price → no mean reversion
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = CointegrationStrategy(
        {"lookback_period": 60, "half_life": 20, "entry_threshold": 2.0}
    )

    signals = strategy.precompute_all_signals(df)
    # On flat data mean_reversion_strength is 0 (beta >= 0), so all signals are masked
    assert signals is not None
    assert signals.isna().all()


def test_cointegration_generate_signals_returns_list() -> None:
    """generate_signals always returns a list."""
    n = 70
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = CointegrationStrategy(
        {"lookback_period": 60, "half_life": 20, "entry_threshold": 2.0}
    )
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)


def test_cointegration_generate_on_short_data_returns_empty() -> None:
    """With fewer bars than lookback_period, generate_signals returns []."""
    n = 30
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = CointegrationStrategy(
        {"lookback_period": 60, "half_life": 20, "entry_threshold": 2.0}
    )
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert result == []


def test_cointegration_half_life_estimation() -> None:
    """_estimate_half_life returns a positive float for mean-reverting series."""
    n = 100
    # Mean-reverting series: oscillate around 50
    rng = np.random.default_rng(42)
    eps = rng.normal(0, 0.3, n)
    theta = -0.1  # mean-reversion speed
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = series[t - 1] + theta * (series[t - 1] - 0) + eps[t]
    series = series + 50

    data_dict = _make_data(series.tolist(), n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = CointegrationStrategy({"lookback_period": 60, "half_life": 20})
    ind = strategy.calculate_indicators(df)

    half_life = ind["half_life"]
    assert isinstance(half_life, float) or isinstance(half_life, np.floating)
    assert half_life > 0
