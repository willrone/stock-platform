"""Numba-accelerated technical indicators.

This module is optional: if numba is not installed, callers should fall back to
pandas/ta-lib implementations.

Design goals:
- Pure numpy inputs/outputs (fast, JIT-friendly)
- No pandas dependency inside JIT functions

Notes:
- First call triggers JIT compilation; benchmark results should exclude the first-call
  compilation time (warmup).
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap


@njit(cache=True)
def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average, adjust=False semantics.

    Args:
        values: 1d float64 array
        span: ewm span

    Returns:
        1d float64 array
    """
    n = values.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    alpha = 2.0 / (span + 1.0)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


@njit(cache=True)
def macd(values: np.ndarray, fast: int, slow: int, signal: int):
    """MACD (DIF/DEA/HIST) using EMA.

    Returns:
        macd_line, macd_signal, macd_hist
    """
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


@njit(cache=True)
def rsi_wilder(values: np.ndarray, period: int) -> np.ndarray:
    """RSI with Wilder smoothing (EMA of gains/losses).

    This is closer to TA-Lib RSI than a simple rolling mean.
    """
    n = values.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out

    out[:] = np.nan

    # gains/losses
    gain = 0.0
    loss = 0.0

    # seed using first period diffs
    for i in range(1, min(period + 1, n)):
        diff = values[i] - values[i - 1]
        if diff > 0:
            gain += diff
        else:
            loss -= diff

    if n <= period:
        return out

    avg_gain = gain / period
    avg_loss = loss / period

    if avg_loss == 0.0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        diff = values[i] - values[i - 1]
        g = diff if diff > 0 else 0.0
        l = -diff if diff < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out
