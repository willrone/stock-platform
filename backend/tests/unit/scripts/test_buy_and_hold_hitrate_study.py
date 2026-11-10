"""buy_and_hold_hitrate_study 的契约测试（合成面板，不读真实库也能跑；只有两条口径断言读真实文件）。

锁五件事：
1. 收益口径必须是含分红的 pct_chg 链（数据源哪天换成未复权，测试当场红）；
2. 池里没有退市股是事实，不许把命中率当真实命中率引用；
3. 整数手 + 佣金 + 印花税的结算数学要和独立复算一致；
4. 选股规则不许把「亏损股（PE<0）」「没 ROE 数据的股」「买不起整手的股」当成便宜/优质选进来；
5. roe 只能取披露日 ≤ 入场日的记录（点位口径，不许回灌未来）。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from scripts.buy_and_hold_hitrate_study import (
    BAOSTOCK_DIR,
    CAPITAL,
    FEE_RATE,
    HAIRCUT_RETURN,
    LOT,
    MIN_FEE,
    STAMP_RATE,
    audit_adj_chain,
    features_at,
    haircut_one,
    pick,
    roe_as_of,
    settle,
)

CODES = ["000001.SZ", "600000.SH"]
NO_DATA = "CI 上没有真实数据层（data/ 被 .gitignore 排除）；这条口径断言只在有数据层的机器上有效，本地必须绿"


def _panel(dates, px, rets):
    cal = pd.DatetimeIndex(dates)
    adj = pd.DataFrame({c: np.cumprod(1.0 + np.array(rets[c], dtype=float)) for c in CODES}, index=cal)
    raw = pd.DataFrame({c: np.array(px[c], dtype=float) for c in CODES}, index=cal)
    zero = lambda v: pd.DataFrame({c: [v] * len(cal) for c in CODES}, index=cal, dtype=float)  # noqa: E731
    return {"adj": adj, "raw": raw, "ret": pd.DataFrame({c: np.array(rets[c], dtype=float) for c in CODES}, index=cal),
            "amount": zero(1e8), "pe": zero(10.0), "pb": zero(1.0), "st": zero(0.0), "halt": zero(0.0),
            "bars": pd.DataFrame({c: np.arange(1.0, len(cal) + 1.0) for c in CODES}, index=cal)}


def _dates(n=400):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2022-01-03", periods=n)]


@pytest.mark.skipif(
    not os.path.exists(os.path.join(BAOSTOCK_DIR, "600000_SH.parquet")),
    reason="数据层缺失：pct_chg 口径断言需要真实 baostock_daily 日线（%s）" % NO_DATA,
)
def test_adj_chain_actually_contains_dividends():
    """整份研究的收益口径靠 pct_chg 链。若它其实等于未复权收益，命中率结论全是假的。"""
    a = audit_adj_chain()
    assert a["gap_days"] >= 2, a                       # 找到除息日（close 与 pct_chg 不一致的日子）
    assert a["gap_mean_pct_on_gap_days"] > 0.5, a      # 不一致的幅度是分红量级，不是四舍五入
    assert a["verdict"] == "close 未复权、pct_chg 含分红"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(BAOSTOCK_DIR, "600000_SH.parquet")),
    reason="数据层缺失：退市股缺席这件事必须对着真实库断言（%s）" % NO_DATA,
)
def test_pool_has_no_delisted_stocks_and_must_be_disclosed():
    """池只有存续股：这些真实退过市的代码在库里根本不存在 → 任何命中率都是上界。"""
    for code in ("300104_SZ", "000018_SZ", "002231_SZ", "300216_SZ"):
        assert not os.path.exists(os.path.join(BAOSTOCK_DIR, f"{code}.parquet")), code
    assert os.path.exists(os.path.join(BAOSTOCK_DIR, "600000_SH.parquet"))


def test_settle_math_matches_independent_replay():
    """每只 +3%/年：整数手、买佣、卖佣+印花税必须逐笔落到现金上，终值与独立复算一致（0.05pp 内）。
    顺带锁一件事：组合收益是按 9 万本金算的 → 整数手剩下的现金会拖后腿，不许假装满仓。"""
    dates = _dates(760)
    entry = pd.Timestamp(dates[0])
    px = {"000001.SZ": [10.0] * len(dates), "600000.SH": [40.0] * len(dates)}
    panel = _panel(dates, px, {c: [0.03 / 252] * len(dates) for c in CODES})
    res = settle(panel, entry, 1, CODES, k=2)
    assert res and res["n_picks"] == 2
    cash_back = 0.0
    for s in res["stocks"]:
        lots = int((CAPITAL / 2) // (s["entry_price"] * LOT))
        assert s["lots"] == lots and lots > 0
        cost = lots * LOT * s["entry_price"]
        assert abs(s["cost"] - cost) < 0.01
        gross = cost * (1 + s["total_return_pct"] / 100)
        net = gross - max(cost * FEE_RATE, MIN_FEE) - max(gross * FEE_RATE, MIN_FEE) - gross * STAMP_RATE
        assert abs((net / cost - 1) * 100 - s["net_return_pct"]) < 0.02, (net / cost - 1, s["net_return_pct"])
        cash_back += net
    expected_port = cash_back / CAPITAL - 1
    assert abs(expected_port * 100 - res["portfolio_total_pct"]) < 0.05
    assert res["stock_hit_rate_pct"] == 100.0
    # 满仓本应 ≈ +3%；按 9 万算只有 ~+1.9% → 闲置现金的拖累必须被算进来
    assert res["portfolio_total_pct"] < 2.9, res["portfolio_total_pct"]
    assert 0.5 < res["idle_cash_pct"] < 5.0, res["idle_cash_pct"]


def test_losing_panel_gives_zero_hit_rate_not_a_fluke():
    dates = _dates(760)
    entry = pd.Timestamp(dates[0])
    px = {"000001.SZ": [10.0] * len(dates), "600000.SH": [20.0] * len(dates)}
    panel = _panel(dates, px, {c: [-0.20 / 252] * len(dates) for c in CODES})
    res = settle(panel, entry, 1, CODES, k=2)
    assert res["stock_hit_rate_pct"] == 0.0
    assert res["portfolio_total_pct"] < -19.0


def test_price_cap_excludes_unaffordable_stocks():
    """9 万买 K=2 只 → 每只 4.5 万 → 单价 >450 元的股票必须被剔除（不是"少买一手"）。
    注意入场日必须有 ≥120 根历史 bar（次新股被剔），所以用第 200 根 bar 入场。"""
    dates = _dates(400)
    entry = pd.Timestamp(dates[200])
    px = {"000001.SZ": [10.0] * len(dates), "600000.SH": [900.0] * len(dates)}
    panel = _panel(dates, px, {c: [0.001] * len(dates) for c in CODES})
    df = features_at(panel, entry, {}, price_cap=CAPITAL / 2 / LOT)
    assert len(df) == 1, "买不起整手的高价股必须整个被剔掉"
    assert "600000.SH" not in df.index and "000001.SZ" in df.index


def test_rules_never_pick_loss_making_or_missing_data_stocks():
    dates = _dates(400)
    entry = pd.Timestamp(dates[200])
    px = {"000001.SZ": [10.0] * len(dates), "600000.SH": [20.0] * len(dates)}
    panel = _panel(dates, px, {c: [0.001] * len(dates) for c in CODES})
    panel["pe"] = pd.DataFrame({"000001.SZ": [-8.0] * len(panel["pe"].index),      # 亏损股：PE 为负
                                "600000.SH": [12.0] * len(panel["pe"].index)}, index=panel["pe"].index)
    df = features_at(panel, entry, {}, price_cap=1e9)
    assert pick(df, "low_pe", 2) == ["600000.SH"], "PE<0 不是便宜，是亏损"
    assert pick(df, "high_roe", 2) == [], "没有 ROE 数据不许被当成高 ROE 选进来"
    panel["pb"] = pd.DataFrame({"000001.SZ": [0.0] * len(panel["pb"].index),
                                "600000.SH": [0.8] * len(panel["pb"].index)}, index=panel["pb"].index)
    df2 = features_at(panel, entry, {}, price_cap=1e9)
    assert pick(df2, "low_pb", 2) == ["600000.SH"], "PB=0 是脏数据不是深度价值"


def test_roe_is_point_in_time_only():
    roe = {"600000": [(pd.Timestamp("2022-04-20"), 12.0), (pd.Timestamp("2023-04-25"), 3.0)]}
    assert roe_as_of(roe, "600000.SH", pd.Timestamp("2022-01-04")) is None, "入场前没有已披露记录"
    assert roe_as_of(roe, "600000.SH", pd.Timestamp("2022-05-01")) == 12.0
    assert roe_as_of(roe, "600000.SH", pd.Timestamp("2023-06-01")) == 3.0, "不许用未来才披露的记录"


def test_delisting_haircut_only_lowers_results():
    cohort = {"years": 3.0, "stocks": [{"cost": 4500.0, "net_return_pct": 60.0},
                                       {"cost": 4500.0, "net_return_pct": 40.0},
                                       {"cost": 4500.0, "net_return_pct": 20.0},
                                       {"cost": 4500.0, "net_return_pct": -10.0}]}
    clean_hit = 75.0
    pes = haircut_one(cohort, 0.05, "悲观")
    rnd = haircut_one(cohort, 0.05, "随机")
    assert pes["hit_rate_pct"] <= clean_hit and rnd["hit_rate_pct"] <= clean_hit
    assert pes["annualized_pct"] < 40.0 and rnd["annualized_pct"] < 40.0
    assert pes["annualized_pct"] <= rnd["annualized_pct"] + 1e-9, "悲观界必须不优于随机界"
    assert abs(pes["hit_rate_pct"] - 50.0) < 1e-6, "1/4 只被换成 -70% → 命中率 2/4"
