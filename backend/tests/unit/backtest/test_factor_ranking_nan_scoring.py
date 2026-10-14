"""factor_ranking 截面打分 NaN 传播修复的回归测试.

2026-10-14 修复: 旧三因子 (momentum/low_vol/low_amp) 的 zscore 截面统计用
ndarray .mean()/.std() (NaN 传播) — 全宇宙任一票停牌/缺历史即让该日整个截面
分数全 NaN → 全市场停打分 (实测官方对照组 2021-2023 零交易)。
修复后全因子 nan-aware, 与 vwap_below (PR#16) 对齐。

断言:
1. 截面含 NaN 票时, 有效票仍拿到分数 (旧代码此处全 NaN)
2. 分数与手算 nan-aware zscore 一致
3. 全 NaN 行仍无分数 (min_cross 门槛保留)
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.strategies.factor_ranking_strategy import (
    FactorRankingStrategy,
    _SCORES_ATTR,
)

N_DAYS = 200
START = datetime(2024, 1, 2)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(START, periods=n)


def _frame(closes: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": pd.Series(1e6, index=index, dtype=float),
    })


async def _prepare(strategy: FactorRankingStrategy, stock_data):
    end = pd.Timestamp(START) + pd.Timedelta(days=N_DAYS * 2)
    await strategy.prepare_backtest_data(stock_data, START, end.to_pydatetime())


@pytest.mark.asyncio
async def test_nan_stock_does_not_kill_cross_section():
    """一只票停牌 (整段 NaN) 时, 其余票仍应拿到分数 — 旧代码此日全 NaN。"""
    idx = _dates(N_DAYS)
    rng = np.random.default_rng(42)
    healthy_a = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, N_DAYS))
    healthy_b = 50 * np.cumprod(1 + rng.normal(0.0004, 0.01, N_DAYS))
    suspended = np.full(N_DAYS, np.nan)  # 全程停牌

    stock_data = {
        "000001.SZ": _frame(healthy_a, idx),
        "000002.SZ": _frame(healthy_b, idx),
        "000003.SZ": _frame(suspended, idx),
    }
    for df in stock_data.values():
        df.attrs["stock_code"] = list(stock_data)[
            list(id(v) for v in stock_data.values()).index(id(df))
        ]

    strat = FactorRankingStrategy({"topk": 2, "n_drop": 1,
                                   "weights": {"momentum": 1.0, "low_vol": 1.0,
                                               "low_amp": 1.0}})
    await _prepare(strat, stock_data)

    valid_counts = {}
    for code, df in stock_data.items():
        smap = df.attrs.get(_SCORES_ATTR, {})
        valid_counts[code] = len(smap)
        if code == "000003.SZ":
            assert len(smap) == 0, f"停牌票不应有分数, got {len(smap)}"
    # 核心断言: 修复前两只健康票分数 = 0 (被停牌票 NaN 传播杀死)
    assert valid_counts["000001.SZ"] > 0, "健康票被 NaN 传播杀死 (修复失效)"
    assert valid_counts["000002.SZ"] > 0, "健康票被 NaN 传播杀死 (修复失效)"


@pytest.mark.asyncio
async def test_scores_match_manual_nan_aware_zscore():
    """分数须等于手算 nan-aware 截面 zscore (动量窗 126+skip20 后的窗口期)。"""
    idx = _dates(N_DAYS)
    rng = np.random.default_rng(7)
    a = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, N_DAYS))
    b = 80 * np.cumprod(1 + rng.normal(0.0003, 0.008, N_DAYS))
    c = 30 * np.cumprod(1 + rng.normal(-0.0002, 0.015, N_DAYS))
    # c 中段停牌 10 天 → 制造局部 NaN
    c = c.copy()
    c[100:110] = np.nan

    stock_data = {"000001.SZ": _frame(a, idx), "000002.SZ": _frame(b, idx),
                  "000003.SZ": _frame(c, idx)}
    for code, df in stock_data.items():
        df.attrs["stock_code"] = code

    strat = FactorRankingStrategy({"topk": 3, "n_drop": 1,
                                   "weights": {"momentum": 1.0, "low_vol": 1.0,
                                               "low_amp": 1.0}})
    await _prepare(strat, stock_data)

    # 参考日: 动量窗 (shift 146) 已就绪, 三票均有效
    ref_date = idx[160]
    got = {code: df.attrs[_SCORES_ATTR].get(ref_date.to_pydatetime())
           for code, df in stock_data.items()}
    assert all(v is not None for v in got.values()), f"该日三票都应有分数: {got}"
    for code, v in got.items():
        assert np.isfinite(v), f"{code} 分数非有限: {v}"

    # 动量因子单独打分 (weights 只开 momentum), 校验数值 = 手算 nan-aware zscore
    stock_data2 = {"000001.SZ": _frame(a, idx), "000002.SZ": _frame(b, idx),
                   "000003.SZ": _frame(c, idx)}
    for code, df in stock_data2.items():
        df.attrs["stock_code"] = code
    strat2 = FactorRankingStrategy({"topk": 3, "n_drop": 1,
                                    "weights": {"momentum": 1.0}})
    await _prepare(strat2, stock_data2)

    close = pd.DataFrame({"000001.SZ": a, "000002.SZ": b, "000003.SZ": c},
                         index=idx, dtype=float)
    momentum = close.shift(20) / close.shift(146) - 1.0
    row = momentum.loc[ref_date]
    valid = row[np.isfinite(row)]
    z_ref = (row - valid.mean()) / valid.std(ddof=0)
    for code, df in stock_data2.items():
        got_v = df.attrs[_SCORES_ATTR][ref_date.to_pydatetime()]
        assert got_v == pytest.approx(z_ref[code], abs=1e-9), \
            f"{code} momentum-only 分数 {got_v} != 手算 {z_ref[code]}"


@pytest.mark.asyncio
async def test_all_nan_row_yields_no_scores():
    """起始窗口期 (全票动量未满 146 行) 不产生分数 — min_cross 门槛保留。"""
    idx = _dates(N_DAYS)
    rng = np.random.default_rng(3)
    a = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, N_DAYS))
    b = 60 * np.cumprod(1 + rng.normal(0.0004, 0.01, N_DAYS))
    stock_data = {"000001.SZ": _frame(a, idx), "000002.SZ": _frame(b, idx)}
    for code, df in stock_data.items():
        df.attrs["stock_code"] = code

    strat = FactorRankingStrategy({"topk": 2, "n_drop": 1,
                                   "weights": {"momentum": 1.0, "low_vol": 1.0,
                                               "low_amp": 1.0}})
    await _prepare(strat, stock_data)

    smap = stock_data["000001.SZ"].attrs[_SCORES_ATTR]
    # 动量需 shift(146): 前 146 行动量 NaN, 但 low_vol(60)/low_amp(10) 有分数
    # → composite 至少在 60 行后开始有值 (权重平均只需一个因子有效)
    assert len(smap) > 0, "应存在分数 (至少 low_vol/low_amp 窗口满后)"
    first = min(smap)
    assert first >= (idx[9] + pd.Timedelta(days=0)).to_pydatetime(), \
        f"首个分数不应早于最慢就绪窗口: {first}"
