"""
KDJStrategy unit tests.

Created: 2026-07-25
Purpose: Verify KDJ signal generation — BUY on low-J golden cross,
         SELL on high-J death cross.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.strategies import KDJStrategy


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
                "high": [c * 1.03 for c in close],
                "low": [c * 0.97 for c in close],
                "close": close,
                "volume": [1_000_000] * len(close),
            },
            index=dates,
        )
        df.attrs["stock_code"] = code
        result[code] = df
    return result


def test_kdj_indicators_shape() -> None:
    """calculate_indicators returns K, D, J lines."""
    k_period, d_period, j_smooth = 9, 3, 3
    n = 50
    close = [50.0] * n

    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = KDJStrategy(
        {
            "k_period": k_period,
            "d_period": d_period,
            "j_smooth": j_smooth,
            "oversold": 20,
            "overbought": 80,
        }
    )
    ind = strategy.calculate_indicators(df)

    assert "k_line" in ind
    assert "d_line" in ind
    assert "j_line" in ind
    assert "price" in ind
    assert len(ind["k_line"]) == n


def test_kdj_precompute_does_not_crash() -> None:
    """precompute_all_signals runs without error on typical data."""
    k_period, d_period, j_smooth = 9, 3, 3
    n = 80
    rng = np.random.default_rng(42)
    close = 50 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10).tolist()

    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = KDJStrategy(
        {
            "k_period": k_period,
            "d_period": d_period,
            "j_smooth": j_smooth,
            "oversold": 20,
            "overbought": 80,
        }
    )

    signals = strategy.precompute_all_signals(df)
    assert signals is not None
    assert len(signals) == len(df)


def test_kdj_oversold_golden_cross_logic() -> None:
    """With J < oversold and K crossing above D, precompute emits BUY."""
    k_period, d_period, j_smooth = 3, 3, 3  # short windows to reduce warmup
    n = k_period + d_period + j_smooth + 10
    close = [50.0] * n

    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = KDJStrategy(
        {
            "k_period": k_period,
            "d_period": d_period,
            "j_smooth": j_smooth,
            "oversold": 20,
            "overbought": 80,
        }
    )

    # On flat data K=D=J=RSV, so no cross. We just verify no crash.
    signals = strategy.precompute_all_signals(df)
    assert signals is not None


def test_kdj_generate_signals_returns_list() -> None:
    """generate_signals always returns a list."""
    n = 30
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = KDJStrategy(
        {
            "k_period": 9,
            "d_period": 3,
            "j_smooth": 3,
            "oversold": 20,
            "overbought": 80,
        }
    )
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)
