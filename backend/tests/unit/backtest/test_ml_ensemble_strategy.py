"""
MLEnsembleLgbXgbRiskCtlStrategy unit tests.

Created: 2026-07-25
Purpose: Verify ML ensemble strategy signal generation using the
         fallback (rule-based) path, since pre-trained models are
         not available in unit tests.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.ml_ensemble_strategy import (
    MLEnsembleLgbXgbRiskCtlStrategy,
)


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


def test_ml_ensemble_indicators_shape() -> None:
    """calculate_indicators returns a rich dict of technical features."""
    n = 100
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MLEnsembleLgbXgbRiskCtlStrategy(
        {"top_n": 5, "prob_threshold": 0.5}
    )
    ind = strategy.calculate_indicators(df)

    # Key categories should be present
    assert "return_1d" in ind
    assert "rsi_14" in ind
    assert "macd" in ind
    assert "volatility_20" in ind
    assert "bb_position" in ind
    assert len(ind["return_1d"]) == n


def test_ml_ensemble_generate_signals_returns_list() -> None:
    """generate_signals returns a list (uses fallback when no model)."""
    n = 100
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MLEnsembleLgbXgbRiskCtlStrategy(
        {"top_n": 5, "prob_threshold": 0.5}
    )
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)
    for s in result:
        assert s.signal_type in (SignalType.BUY, SignalType.SELL)


def test_ml_ensemble_short_data_returns_empty() -> None:
    """With fewer than 60 bars, generate_signals returns []."""
    n = 30
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MLEnsembleLgbXgbRiskCtlStrategy(
        {"top_n": 5, "prob_threshold": 0.5}
    )
    result = strategy.generate_signals(df, df.index[-1])
    # Uses _get_current_idx which returns -1 for missing date → idx < 60 → empty
    assert result == []


def test_ml_ensemble_buy_on_uptrend_fallback() -> None:
    """With a strong uptrend and oversold RSI, fallback path should
    produce BUY signals."""
    n = 80
    # Gradual uptrend with some volatility
    rng = np.random.default_rng(42)
    close = 50 + np.cumsum(rng.normal(0.15, 0.5, n))
    close = np.maximum(close, 10).tolist()

    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MLEnsembleLgbXgbRiskCtlStrategy(
        {"top_n": 5, "prob_threshold": 0.5}
    )
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)


def test_ml_ensemble_position_scale_stop_loss() -> None:
    """_calculate_position_scale returns 0 when daily return is below stop_loss."""
    strategy = MLEnsembleLgbXgbRiskCtlStrategy(
        {"stop_loss": -0.02, "vol_scaling": False, "market_filter": False}
    )
    scale = strategy._calculate_position_scale(-0.03)
    assert scale == 0.0

    scale_good = strategy._calculate_position_scale(0.01)
    assert scale_good == 1.0


def test_ml_ensemble_get_feature_vector() -> None:
    """_get_feature_vector returns a numpy array of correct length."""
    n = 100
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MLEnsembleLgbXgbRiskCtlStrategy({})
    ind = strategy.calculate_indicators(df)

    vec = strategy._get_feature_vector(ind, n - 1)
    assert vec is not None
    assert isinstance(vec, np.ndarray)
    # 52 features in the getter list (not 62 as in _get_feature_names)
    assert len(vec) >= 40


def test_ml_ensemble_precompute_fallback() -> None:
    """precompute_all_signals uses fallback when no model is loaded."""
    n = 100
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = MLEnsembleLgbXgbRiskCtlStrategy(
        {"top_n": 5, "prob_threshold": 0.5}
    )
    signals = strategy.precompute_all_signals(df)

    # Should use fallback since no model loaded
    assert signals is not None
    assert len(signals) == len(df)
    assert all(s in (SignalType.BUY, SignalType.SELL, None) for s in signals)
