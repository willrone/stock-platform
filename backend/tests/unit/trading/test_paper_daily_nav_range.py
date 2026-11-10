"""`paper_daily_nav.py --as-of / --range`（方案 D：历史区间回溯）的门禁测试。

用**合成数据**（tmp_path + 手写 ledger + 小 adj_close 矩阵），
**不依赖真实 `data/paper/`** → 跑得快。

## 本文件守什么

| # | 门禁 | 为什么 |
|---|---|---|
| a | **默认行为不变** | 不带 `--as-of/--range` 时，逐位等同改造前（既有门禁也依赖） |
| b | `--as-of` 只产出**那一个点** | 历史单点回溯 |
| c | `--range` 只产出**区间内**的点 | 历史区间回溯 |
| d | **区间收窄不改变重放口径** | 区间起点之前的持仓必须算对 → 否则区间内市值是错的 |
| e | **非交易日端点 → 报错**（不静默取整） | 静默取整会让"我查了 10-26"变成"我查了 10-23" |
| f | `--as-of` + `--range` 同给 → 取交集 | 语义明确 |
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[3]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SCRIPT = BACKEND / "scripts" / "paper_daily_nav.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dn():
    return _load(SCRIPT, "paper_daily_nav_range_test")


# ── 合成数据 ──────────────────────────────────────────────────────────
DATES = pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20",
                        "2026-10-21", "2026-10-22"])

HOLDINGS: Dict[str, Any] = {
    "initial_capital": 1_000_000.0,
    "cash": 989_985.0,
    "positions": {},
}


def make_adj() -> pd.DataFrame:
    return pd.DataFrame({
        "600000.SH": [10.0, 10.5, 11.0, 12.0, 13.0],
    }, index=DATES)


LEDGER: List[Dict[str, str]] = [
    {"date": "2026-10-16", "action": "BUY", "code": "600000.SH",
     "name": "浦发", "price": "10.0", "shares": "1000", "amount": "10000.0",
     "commission": "5.0", "stamp_tax": "0.0", "slippage": "10.0",
     "score": "0.5", "cost_amount": "10000.0", "realized_pnl": "",
     "cash_after": "989985.0", "portfolio_value": "989985.0"},
]


@pytest.fixture
def wired(dn, monkeypatch):
    """把 `paper_daily_nav` 的账本/持仓源接到合成数据上。"""
    monkeypatch.setattr(dn, "read_ledger", lambda s: list(LEDGER))
    monkeypatch.setattr(dn.pp, "load_holdings",
                        lambda s, cap: dict(HOLDINGS))
    return dn


def _dates(points):
    return [p["date"] for p in points]


# ── a：默认行为不变 ───────────────────────────────────────────────────
def test_default_produces_all_trading_days_since_first_trade(wired):
    points, diag = wired.build_line_points("x", make_adj(), DATES)
    assert _dates(points) == [str(d.date()) for d in DATES]
    assert diag["window_start"] is None
    assert diag["window_end"] is None


# ── b：--as-of 只产出那一个点 ─────────────────────────────────────────
def test_as_of_produces_single_point(wired):
    points, diag = wired.build_line_points("x", make_adj(), DATES,
                                           as_of="2026-10-20")
    assert _dates(points) == ["2026-10-20"]
    assert diag["window_end"] == "2026-10-20"


def test_as_of_value_matches_default_run_at_same_day(wired):
    """`--as-of` 的点必须与默认跑出来的**同一天**的点**逐位相等**。"""
    adj = make_adj()
    full, _ = wired.build_line_points("x", adj, DATES)
    one, _ = wired.build_line_points("x", adj, DATES, as_of="2026-10-20")
    by_date = {p["date"]: p for p in full}
    assert one[0] == by_date["2026-10-20"]


# ── c：--range 只产出区间内的点 ───────────────────────────────────────
def test_range_produces_only_window(wired):
    points, diag = wired.build_line_points(
        "x", make_adj(), DATES, range_="2026-10-19:2026-10-21")
    assert _dates(points) == ["2026-10-19", "2026-10-20", "2026-10-21"]
    assert diag["window_start"] == "2026-10-19"
    assert diag["window_end"] == "2026-10-21"


def test_range_values_match_default_run(wired):
    """区间内的点必须与默认跑出来的**同一天**的点逐位相等。"""
    adj = make_adj()
    full, _ = wired.build_line_points("x", adj, DATES)
    by_date = {p["date"]: p for p in full}
    win, _ = wired.build_line_points("x", adj, DATES,
                                     range_="2026-10-19:2026-10-21")
    for p in win:
        assert p == by_date[p["date"]]


# ── d：区间收窄不改变重放口径 ─────────────────────────────────────────
def test_range_does_not_change_replay_semantics(wired, monkeypatch):
    """区间起点之前的成交**仍然要参与重放**。

    构造：先在 10-16 买、10-20 卖，区间取 10-19~10-22 →
    10-19 那天必须**还持有**（不能因为区间从 10-19 开始就把 10-16 的买入丢掉）。
    """
    rows = [
        {"date": "2026-10-16", "action": "BUY", "code": "600000.SH",
         "price": "10.0", "shares": "1000", "amount": "10000.0",
         "commission": "5.0", "stamp_tax": "0.0", "slippage": "10.0",
         "score": "", "cost_amount": "", "realized_pnl": "",
         "cash_after": "", "portfolio_value": ""},
        {"date": "2026-10-20", "action": "SELL", "code": "600000.SH",
         "price": "11.0", "shares": "1000", "amount": "11000.0",
         "commission": "5.0", "stamp_tax": "5.5", "slippage": "11.0",
         "score": "", "cost_amount": "", "realized_pnl": "",
         "cash_after": "", "portfolio_value": ""},
    ]
    monkeypatch.setattr(wired, "read_ledger", lambda s: rows)
    adj = make_adj()
    win, _ = wired.build_line_points("x", adj, DATES,
                                     range_="2026-10-19:2026-10-22")
    by_date = {p["date"]: p for p in win}
    # 10-19：仍持仓 1000 股 → n_positions = 1
    assert by_date["2026-10-19"]["n_positions"] == 1
    # 10-20：已清仓 → n_positions = 0
    assert by_date["2026-10-20"]["n_positions"] == 0
    # 10-19 的市值 = 1000×10×(10.5/10) = 10500；现金 989985
    assert by_date["2026-10-19"]["value"] == pytest.approx(1_000_485.0)


# ── e：非交易日端点 → 报错（不静默取整）──────────────────────────────
def test_as_of_non_trading_day_raises(wired):
    with pytest.raises(ValueError, match="不是交易日"):
        wired.build_line_points("x", make_adj(), DATES, as_of="2026-10-18")


def test_range_non_trading_day_raises(wired):
    with pytest.raises(ValueError, match="不是交易日"):
        wired.build_line_points("x", make_adj(), DATES,
                                range_="2026-10-18:2026-10-21")


def test_range_bad_format_raises(wired):
    with pytest.raises(ValueError, match="START:END"):
        wired.build_line_points("x", make_adj(), DATES, range_="2026-10-19")


def test_range_start_after_end_raises(wired):
    with pytest.raises(ValueError, match="起点晚于终点"):
        wired.build_line_points("x", make_adj(), DATES,
                                range_="2026-10-21:2026-10-19")


# ── f：--as-of + --range 同给 → 取交集 ───────────────────────────────
def test_as_of_and_range_take_intersection(wired):
    points, diag = wired.build_line_points(
        "x", make_adj(), DATES, as_of="2026-10-20",
        range_="2026-10-19:2026-10-22")
    assert _dates(points) == ["2026-10-19", "2026-10-20"]
    assert diag["window_start"] == "2026-10-19"
    assert diag["window_end"] == "2026-10-20"
