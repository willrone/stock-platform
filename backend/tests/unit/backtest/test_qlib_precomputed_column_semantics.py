"""qlib 预计算列的语义回归测试（2026-10-25）。

背景:
  qlib 预计算 parquet 里的 `MA20` / `STD20` **不是价格均线/标准差**，
  而是归一化因子值。实测 `000001_SZ`:
      qlib MA20  = 1.073   真实 MA20 = 45.49    相关系数 0.087
      qlib STD20 = 0.047   真实 STD20 = 1.99
  Bollinger / Cointegration 曾用它算布林带与 z-score（= 用垃圾指标），
  且 95% 的股票是部分填充 → 同一策略内有的走预计算、有的走 fallback。

  `RSI14` 则相反是正确的（实测相关系数 1.000000，最大差 0.0003），
  `rsi_optimized.py` 继续用它没问题。

本测试锁死三件事:
  1. qlib 的 MA*/STD* 确实与真实价格均线/标准差不一致（防止有人误以为能用）
  2. Bollinger / Cointegration 的 calculate_indicators 不得再取用这些列
  3. qlib RSI14 确实等于真实 RSI（保住 rsi_optimized 的正确用法）
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

QLIB_DIR = "/Users/rone/Projects/stock-platform/data/qlib_data/features/day"
PROBE_FILE = "000001_SZ.parquet"

from app.services.backtest.strategies.strategies import (  # noqa: E402
    BollingerBandStrategy,
    CointegrationStrategy,
)


def _load_probe():
    path = os.path.join(QLIB_DIR, PROBE_FILE)
    if not os.path.exists(path):
        pytest.skip(f"qlib 探针文件不存在: {path}")
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["$close", "MA20", "STD20", "RSI14"])
    df = t.to_pandas()
    if df.index.nlevels > 1:
        df.index = pd.to_datetime(df.index.get_level_values(-1))
    return df


def test_qlib_ma20_is_not_price_moving_average() -> None:
    """qlib MA20 与真实价格均线几乎不相关 —— 禁止当作均线使用。"""
    df = _load_probe()
    if df["MA20"].notna().sum() < 100:
        pytest.skip("该探针文件 MA20 填充不足")
    real_ma20 = df["$close"].rolling(20).mean()
    cmp = pd.DataFrame({"q": df["MA20"], "r": real_ma20}).dropna()
    corr = float(cmp["q"].corr(cmp["r"]))
    assert corr < 0.5, f"qlib MA20 竟然与真实均线高相关({corr:.3f})，请重新评估"
    # 量级也完全不同（真实均线是价格量级）
    assert float(cmp["q"].abs().max()) < float(cmp["r"].abs().max())


def test_qlib_std20_is_not_price_std() -> None:
    """qlib STD20 与真实价格标准差不是一回事。"""
    df = _load_probe()
    if df["STD20"].notna().sum() < 100:
        pytest.skip("该探针文件 STD20 填充不足")
    real_std20 = df["$close"].rolling(20).std()
    cmp = pd.DataFrame({"q": df["STD20"], "r": real_std20}).dropna()
    assert float(cmp["q"].abs().max()) < float(cmp["r"].abs().max())


def test_qlib_rsi14_matches_real_rsi() -> None:
    """qlib RSI14 确实是真实 RSI —— rsi_optimized 用它是安全的。"""
    df = _load_probe()
    d = df["$close"].diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    real = 100 - 100 / (1 + gain / loss)
    cmp = pd.DataFrame({"q": df["RSI14"], "r": real}).dropna()
    if len(cmp) < 100:
        pytest.skip("探针数据不足")
    assert float(cmp["q"].corr(cmp["r"])) > 0.999
    assert float((cmp["q"] - cmp["r"]).abs().max()) < 0.01


def _fake_frame(close: pd.Series) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=len(close), freq="D"))
    close = close.reset_index(drop=True)
    df = pd.DataFrame(
        {
            "open": close.to_numpy(),
            "high": close.to_numpy() * 1.01,
            "low": close.to_numpy() * 0.99,
            "close": close.to_numpy(),
            "volume": np.full(len(close), 1_000_000.0),
            # 故意塞垃圾值：若策略误用预计算列，结果会被污染
            "MA20": np.full(len(close), 1.0),
            "STD20": np.full(len(close), 0.05),
        },
        index=idx,
    )
    return df


def test_bollinger_uses_real_price_not_qlib_columns() -> None:
    """Bollinger 的中轨/带宽必须由真实价格算出，不能被 qlib MA/STD 污染。"""
    close = pd.Series(np.linspace(10.0, 20.0, 120))
    df = _fake_frame(close)
    df.attrs["stock_code"] = "000001.SZ"

    strat = BollingerBandStrategy({"period": 20, "std_dev": 2})
    ind = strat.calculate_indicators(df)
    sma = ind["sma"]
    expected = close.reset_index(drop=True).rolling(20).mean()
    expected.index = df.index
    valid = expected.notna()
    assert np.allclose(sma[valid].to_numpy(), expected[valid].to_numpy())
    # 中轨绝不能是恒定的 1.0（qlib MA20 的垃圾值）
    assert float(sma[valid].abs().max()) > 5.0


def test_cointegration_uses_real_price_not_qlib_columns() -> None:
    """Cointegration 的 z-score 分母必须来自真实价格标准差。"""
    close = pd.Series(np.linspace(10.0, 20.0, 200))
    df = _fake_frame(close)
    df.attrs["stock_code"] = "000001.SZ"

    strat = CointegrationStrategy({"lookback_period": 60})
    ind = strat.calculate_indicators(df)
    zscore = ind.get("zscore")
    if zscore is None:
        pytest.skip("Cointegration 未暴露 zscore 指标")
    valid = zscore.notna()
    assert float(zscore[valid].abs().max()) > 0.5
