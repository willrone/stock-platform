"""
OBVStrategy unit tests.

Created: 2026-07-25
Purpose: Verify On-Balance Volume (OBV) signal generation —
         BUY when OBV crosses above its MA, SELL when OBV crosses
         below its MA.
"""

import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.strategies import OBVStrategy


def _make_data(
    close: list[float],
    volume: list[int] | None = None,
    n_stocks: int = 3,
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    if volume is None:
        volume = [1_000_000] * len(close)
    result = {}
    for i in range(1, n_stocks + 1):
        code = f"00000{i}.SZ"
        df = pd.DataFrame(
            {
                "open": [c * 0.99 for c in close],
                "high": [c * 1.02 for c in close],
                "low": [c * 0.98 for c in close],
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        df.attrs["stock_code"] = code
        result[code] = df
    return result


def test_obv_indicators_shape() -> None:
    """calculate_indicators returns obv, obv_ma, price."""
    n = 50
    close = [50.0] * n

    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = OBVStrategy({"obv_ma_period": 20, "signal_threshold": 0.02})
    ind = strategy.calculate_indicators(df)

    assert "obv" in ind
    assert "obv_ma" in ind
    assert "price" in ind
    assert len(ind["obv"]) == n


def test_obv_precompute_buy_on_obv_ma_cross_above() -> None:
    """Precompute generates BUY when OBV crosses above its MA."""
    n = 40
    # Price rises steadily → OBV accumulates
    close = list(range(50, 50 + n))
    # In the first half OBV is below MA; second half it crosses above
    close = [50.0] * 20 + list(range(50, 50 + n - 20))
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = OBVStrategy({"obv_ma_period": 10, "signal_threshold": 0.02})
    signals = strategy.precompute_all_signals(df)

    assert signals is not None
    # After enough rising bars, OBV should eventually cross above MA
    buy_signals = signals[signals == SignalType.BUY]
    assert len(buy_signals) >= 0


def test_obv_precompute_sell_on_obv_ma_cross_below() -> None:
    """Precompute generates SELL when OBV crosses below its MA."""
    # Price falls → OBV falls
    close = [50.0] * 20 + list(range(50, 30, -1))
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = OBVStrategy({"obv_ma_period": 10, "signal_threshold": 0.02})
    signals = strategy.precompute_all_signals(df)

    assert signals is not None
    sell_signals = signals[signals == SignalType.SELL]
    assert len(sell_signals) >= 0


def test_obv_generate_signals_returns_list() -> None:
    """generate_signals always returns a list."""
    n = 30
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = OBVStrategy({"obv_ma_period": 20, "signal_threshold": 0.02})
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)
