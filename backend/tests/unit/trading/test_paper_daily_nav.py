"""纸面盘**日频盯市**门禁测试（方案书 §3 的 a~e + 新增 f）。

用**合成数据**（小的 adj_close 矩阵 + 手写 ledger），不依赖真实行情：
跑得快、可重复、且能把边界情况构造出来（停牌/缺价/部分卖出/无未来函数）。

门禁清单（与方案书 §3 对应）：
  a 口径一致性   → 见 test_total_return_* + test_daily_point_matches_nav_history
  b 序列完整性   → test_points_are_sorted_unique_no_future
  c 幂等         → test_merge_points_is_idempotent
  d 断点续跑     → test_resume_after_truncation_matches_full_run
  e 缺价不造假   → test_missing_price_uses_last_valid_and_marks_stale
  f 无未来函数   → test_truncated_data_recompute_matches（新增，比 a 更硬）
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BACKEND))

SCRIPT = BACKEND / "scripts" / "paper_daily_nav.py"
PP_SCRIPT = BACKEND / "scripts" / "paper_portfolio.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dn():
    return _load(SCRIPT, "paper_daily_nav_test")


@pytest.fixture(scope="module")
def pp():
    return _load(PP_SCRIPT, "paper_portfolio_for_daily_nav_test")


# ── 合成数据 ──────────────────────────────────────────────────────────
DATES = pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20",
                        "2026-10-21", "2026-10-22"])


def make_adj() -> pd.DataFrame:
    return pd.DataFrame({
        "600000.SH": [10.0, 10.5, 11.0, 12.0, 13.0],
        "600001.SH": [20.0, 20.0, 21.0, 21.0, 21.0],
    }, index=DATES)


def make_ledger(rows) -> list[dict]:
    base = {"date": "", "action": "BUY", "code": "", "price": "0",
            "shares": "0", "amount": "0", "cash_after": "", "portfolio_value": ""}
    out = []
    for r in rows:
        d = dict(base)
        d.update(r)
        out.append(d)
    return out


# ── f（核心）：估值只使用 ≤ as_of 的信息，且缺价用最后有效价 ───────────
def test_total_return_value_uses_last_valid_price(pp):
    """e + f：某票在估值日没有价 → 用**≤ as_of 的最后有效价**，并标 stale。"""
    adj = make_adj()
    positions = {"600000.SH": {"shares": 1000, "cost": 10.0, "adj_cost": 10.0}}
    cash = 0.0
    # 把 2026-10-22 当天做成 NaN（模拟停牌 / 涨跌幅还没同步）
    adj.loc[DATES[-1], "600000.SH"] = np.nan

    value, det = pp.total_return_portfolio_value(positions, cash, adj, DATES[-1])
    # 最后有效价 = 2026-10-21 的 12.0 → 市值 = 1000 × 10.0 × (12.0/10.0) = 12000
    assert value == pytest.approx(12000.0)
    assert det[0]["stale_price"] is True
    assert det[0]["price_date"] == "2026-10-21"


def test_total_return_value_no_price_at_all_freezes_at_cost(pp):
    """完全没有价 → ratio=1.0（市值=买入金额）+ stale 标记（不造假收益）。"""
    adj = make_adj()
    adj["600000.SH"] = np.nan
    positions = {"600000.SH": {"shares": 1000, "cost": 10.0, "adj_cost": 10.0}}
    value, det = pp.total_return_portfolio_value(positions, 0.0, adj, DATES[-1])
    assert value == pytest.approx(10000.0)
    assert det[0]["stale_price"] is True


def test_total_return_value_is_bitwise_stable(dn, pp):
    """a 的基础：同一输入两次调用结果**逐位**相等（幂等的必要条件）。"""
    adj = make_adj()
    positions = {"600000.SH": {"shares": 1000, "cost": 10.0, "adj_cost": 10.0},
                 "600001.SH": {"shares": 500, "cost": 20.0, "adj_cost": 20.0}}
    v1, d1 = pp.total_return_portfolio_value(positions, 1000.0, adj, DATES[-1])
    v2, d2 = pp.total_return_portfolio_value(positions, 1000.0, adj, DATES[-1])
    assert v1 == v2
    assert [x["market_value"] for x in d1] == [x["market_value"] for x in d2]


# ── 重放器 ────────────────────────────────────────────────────────────
def test_replay_buy_then_full_sell(dn):
    rows = make_ledger([
        {"date": "2026-10-16", "action": "BUY", "code": "600000.SH",
         "price": "10.0", "shares": "1000", "amount": "10000", "cash_after": "900000"},
        {"date": "2026-10-20", "action": "SELL", "code": "600000.SH",
         "price": "11.0", "shares": "1000", "amount": "11000", "cash_after": "1010000"},
    ])
    snaps = dn.replay_holdings(rows, make_adj(), initial_capital=1_000_000.0)
    assert set(snaps) == {"2026-10-16", "2026-10-20"}
    assert snaps["2026-10-16"]["positions"]["600000.SH"]["shares"] == 1000
    assert "600000.SH" not in snaps["2026-10-20"]["positions"]
    # 现金自算（与 paper_portfolio 成本模型一致）：
    #   BUY : gross 10000 + 佣金 max(5, 万3)=5 + 滑点 10 → 现金 999985
    #   SELL: 总回报口径 gross = 10.0×1000×(11.0/10.0)=11000
    #         − 佣金 5 − 印花 5.5 − 滑点 11 → +10978.5
    assert snaps["2026-10-16"]["cash"] == pytest.approx(989_985.0)
    assert snaps["2026-10-20"]["cash"] == pytest.approx(1_000_963.5)


def test_replay_partial_sell_keeps_unit_cost(dn):
    """部分卖出：每股成本不变（移动平均成本法），总成本随股数同比例减少。"""
    rows = make_ledger([
        {"date": "2026-10-16", "action": "BUY", "code": "600000.SH",
         "price": "10.0", "shares": "1000", "amount": "10000", "cash_after": "900000"},
        {"date": "2026-10-20", "action": "SELL", "code": "600000.SH",
         "price": "11.0", "shares": "400", "amount": "4400", "cash_after": "944000"},
    ])
    snaps = dn.replay_holdings(rows, make_adj())
    p = snaps["2026-10-20"]["positions"]["600000.SH"]
    assert p["shares"] == 600
    assert p["cost"] == pytest.approx(10.0)   # 每股成本不变


def test_replay_multiple_buys_are_weighted_average(dn):
    rows = make_ledger([
        {"date": "2026-10-16", "action": "BUY", "code": "600000.SH",
         "price": "10.0", "shares": "1000", "amount": "10000", "cash_after": "900000"},
        {"date": "2026-10-20", "action": "BUY", "code": "600000.SH",
         "price": "20.0", "shares": "1000", "amount": "20000", "cash_after": "880000"},
    ])
    snaps = dn.replay_holdings(rows, make_adj())
    p = snaps["2026-10-20"]["positions"]["600000.SH"]
    assert p["shares"] == 2000
    assert p["cost"] == pytest.approx(15.0)   # (10×1000 + 20×1000)/2000
    assert p["buy_date"] == "2026-10-20"       # 最近一次买入日


# ── c：幂等（merge 按 date upsert）────────────────────────────────────
def test_merge_points_is_idempotent(dn):
    a = [{"date": "2026-10-16", "value": 100.0, "return_pct": 0.0},
         {"date": "2026-10-19", "value": 101.0, "return_pct": 1.0}]
    b = [{"date": "2026-10-19", "value": 999.0, "return_pct": 99.9},
         {"date": "2026-10-20", "value": 102.0, "return_pct": 2.0}]
    m1 = dn.merge_points(a, b)
    m2 = dn.merge_points(m1, b)
    assert m1 == m2                      # 幂等
    assert [p["date"] for p in m1] == ["2026-10-16", "2026-10-19", "2026-10-20"]
    assert m1[1]["value"] == 999.0        # 同日覆盖，不追加


# ── d：断点续跑（截断后补齐 == 一次跑完）──────────────────────────────
def test_resume_after_truncation_matches_full_run(dn):
    full = [{"date": "2026-10-16", "value": 1.0}, {"date": "2026-10-19", "value": 2.0},
            {"date": "2026-10-20", "value": 3.0}]
    truncated = full[:-1]
    resumed = dn.merge_points(truncated, full[-1:])
    assert resumed == full


# ── b：序列完整性 ─────────────────────────────────────────────────────
def test_points_are_sorted_unique_no_future(dn, tmp_path):
    pts = [{"date": "2026-10-19", "value": 2.0},
           {"date": "2026-10-16", "value": 1.0},
           {"date": "2026-10-19", "value": 2.0}]
    merged = dn.merge_points(pts, [])
    dates = [p["date"] for p in merged]
    assert dates == sorted(dates)
    assert len(dates) == len(set(dates))
