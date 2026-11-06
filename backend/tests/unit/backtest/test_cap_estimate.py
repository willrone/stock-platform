"""cap_estimate 契约测试：市值推导、单位闸门、分层边界、缓存复用。

核心原则沿用宇宙闸门：查不到 = 排除，不是放行。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.cap_estimate import (
    estimate_caps,
    filter_tier,
    tier_bounds,
    tier_stats,
)

START, END = "2024-01-01", "2024-01-10"
DATES = pd.bdate_range(START, END).strftime("%Y-%m-%d").tolist()


def _write(root: str, code: str, *, price: float, shares: float, amount_mult: float = 1.0) -> None:
    """造一只票：股本 shares、价格 price、成交额按换手率反推（amount 单位元，除非 amount_mult 改口径）。"""
    daily_vol = shares * 0.01  # 每天换手 1%
    turn = 1.0
    amount = daily_vol * price * amount_mult
    df = pd.DataFrame({
        "date": DATES,
        "close": [price] * len(DATES),
        "volume": [daily_vol] * len(DATES),
        "amount": [amount] * len(DATES),
        "turn": [turn] * len(DATES),
    })
    d = os.path.join(root, "parquet", "baostock_daily")
    os.makedirs(d, exist_ok=True)
    df.to_parquet(os.path.join(d, code.replace(".", "_") + ".parquet"))


def _root(tmp_path) -> str:
    root = tmp_path / "baostock"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def test_cap_equals_amount_over_turn(tmp_path):
    """市值 = 成交额 / 换手率。1 亿股 × 10 元 = 10 亿。"""
    root = _root(tmp_path)
    _write(root, "600001.SH", price=10.0, shares=1e8)
    caps = estimate_caps(root, START, END, ["600001.SH"])
    assert caps["600001.SH"] == pytest.approx(10e8, rel=1e-6)


def test_unit_assumption_break_raises(tmp_path):
    """若 amount 口径变成千元（常见于其它数据源），必须抛错，不能产出错市值。"""
    root = _root(tmp_path)
    _write(root, "600002.SH", price=10.0, shares=1e8, amount_mult=1e-3)
    with pytest.raises(ValueError, match="单位假设不成立"):
        estimate_caps(root, START, END, ["600002.SH"])


def test_missing_turn_excludes_stock(tmp_path):
    """换手率缺失的票不给市值（后续分层里会被排除），不许瞎估。"""
    root = _root(tmp_path)
    _write(root, "600003.SH", price=10.0, shares=1e8)
    p = os.path.join(root, "parquet", "baostock_daily", "600003_SH.parquet")
    d = pd.read_parquet(p)
    d["turn"] = np.nan
    d.to_parquet(p)
    caps = estimate_caps(root, START, END, ["600003.SH"])
    assert "600003.SH" not in caps


def test_tier_bounds_and_filter(tmp_path):
    """100 只票按市值排 1..100 亿：low/mid/high 三段互不重叠，且查不到市值的票被剔。"""
    root = _root(tmp_path)
    codes = []
    for i in range(1, 101):
        code = f"60{i:04d}.SH"
        codes.append(code)
        _write(root, code, price=10.0, shares=i * 1e7)  # 市值 = i 亿
    codes.append("609999.SH")  # 没有数据文件 → 不可估
    caps = estimate_caps(root, START, END, codes)
    q25, q75 = tier_bounds(caps)
    assert q25 < q75
    low = filter_tier(codes, caps, "low")
    mid = filter_tier(codes, caps, "mid")
    high = filter_tier(codes, caps, "high")
    assert "609999.SH" not in low + mid + high  # 查不到 = 排除
    assert set(low) & set(mid) == set() and set(mid) & set(high) == set()
    assert max(caps[c] for c in low) < min(caps[c] for c in mid) <= max(caps[c] for c in mid) < min(caps[c] for c in high)
    assert len(low) + len(mid) + len(high) == len(caps)
    st = tier_stats(caps)
    assert st["n"] == 100 and st["median_yi"] == pytest.approx(50.5, abs=0.6)


def test_unknown_tier_and_small_universe_raise(tmp_path):
    root = _root(tmp_path)
    _write(root, "600004.SH", price=10.0, shares=1e8)
    caps = estimate_caps(root, START, END, ["600004.SH"])
    with pytest.raises(ValueError, match="未知 tier"):
        filter_tier(["600004.SH"], caps, "small")
    with pytest.raises(ValueError, match="太少"):
        tier_bounds(caps)


def test_cache_is_reused(tmp_path):
    """第二次调用走缓存：即使底层文件被删，也不再重算（缓存命中）。"""
    root = _root(tmp_path)
    codes = [f"60{i:04d}.SH" for i in range(1, 51)]
    for i, code in enumerate(codes, start=1):
        _write(root, code, price=10.0, shares=i * 1e7)
    cache = str(tmp_path / "cap_cache.json")
    first = estimate_caps(root, START, END, codes, cache)
    assert os.path.exists(cache)
    meta = json.load(open(cache))
    assert meta["n_codes"] == 50 and meta["method"].startswith("median(amount")
    for code in codes:
        os.remove(os.path.join(root, "parquet", "baostock_daily", code.replace(".", "_") + ".parquet"))
    second = estimate_caps(root, START, END, codes, cache)
    assert second == first and len(second) == 50
