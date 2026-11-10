"""§12.84 长窗口命中率研究的契约测试。

A 组（合成面板，不依赖数据落库）锁的是结算与口径本身：分红必须进收益、
退市必须真的亏钱、特征只能看过去、手数必须整数、现金不能为负。
B 组（读产物 JSON）锁的是「无偏宇宙真的被用了」：样本里必须有退市股、
必须有持仓期内被摘牌的、悲观口径必须比乐观口径差 —— 少一条就红。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import scripts.buy_and_hold_hitrate_long as M
from scripts.buy_and_hold_hitrate_long import CAPITAL, COMM_MIN, COMM_RATE, LOT, STAMP, settle

OUT_JSON = os.path.join(M.ROOT, "data", "stocks", "buy_and_hold_hitrate_long.json")


def _synthetic_panel() -> M.Panel:
    """三只票：A 分红连续（tr_index 平滑、raw 除息跳空）、B 上市后 1 年被摘牌、C 入场后暴涨（后视陷阱）。"""
    cal = pd.DatetimeIndex(pd.bdate_range("2010-01-04", periods=1200))
    p = M.Panel()
    p.cal = cal
    p.meta = {}
    p.dates = {}
    p.close = {}
    p.tr = {}
    p.amount = {}

    def fill(key, close, hfq, out=None, delisted=False):
        p.dates[key] = cal
        p.close[key] = np.asarray(close, dtype="float32")
        p.tr[key] = np.asarray(hfq, dtype="float32")
        p.amount[key] = np.full(len(cal), 5e8, dtype="float32")
        p.meta[key] = {"delisted": delisted, "outDate": out}

    n = len(cal)
    # A：raw 每年除息 5 元，tr_index 恒定 → 用 raw 算会得出巨额亏损，用总回报算是 0 ± 费用
    raw_a, hfq_a = [], []
    price = 100.0
    for i in range(n):
        if i % 252 == 251:
            price -= 5.0
        raw_a.append(price)
        hfq_a.append(100.0)
    fill("600001.SH", raw_a, hfq_a)

    # B：正常涨一年后突然没 bar（摘牌 2013-01-04）
    raw_b = [10.0 * (1.001 ** i) for i in range(n)]
    hfq_b = list(raw_b)
    cut = int((cal >= pd.Timestamp("2013-01-04")).argmax())
    raw_b = raw_b[:cut] + [np.nan] * (n - cut)
    hfq_b = hfq_b[:cut] + [np.nan] * (n - cut)
    p.dates["000005.SZ"] = cal[:cut]
    p.close["000005.SZ"] = np.asarray(raw_b[:cut], dtype="float32")
    p.tr["000005.SZ"] = np.asarray(hfq_b[:cut], dtype="float32")
    p.amount["000005.SZ"] = np.full(cut, 3e8, dtype="float32")
    p.meta["000005.SZ"] = {"delisted": True, "outDate": pd.Timestamp("2013-01-04")}

    # C：入场后暴涨（特征不许看到未来），入场前是低波动
    raw_c = [20.0] * n
    hfq_c = list(raw_c)
    after = int((cal >= pd.Timestamp("2014-01-06")).argmax())
    for i in range(after, n):
        hfq_c[i] = 20.0 * (1.01 ** (i - after))
        raw_c[i] = hfq_c[i]
    fill("600002.SH", raw_c, hfq_c)
    return p


def _entry() -> pd.Timestamp:
    return pd.Timestamp("2012-01-03")


def test_total_return_caliber_makes_dividends_a_zero_not_a_loss():
    p = _synthetic_panel()
    r = settle(p, ["600001.SH"], _entry(), 1)
    assert r["n"] == 1
    # raw 口径会被算成一年亏 5%（除息），自建总回报口径必须是 0 ± 费用量级
    assert abs(r["picks"][0]["ret_pct"]) < 0.05, f"分红没进收益：{r['picks'][0]['ret_pct']}"
    assert r["nav_a"] < CAPITAL, "费用必须真的离开净值"
    assert CAPITAL - r["nav_a"] < 300, f"只有两只手的佣金，拖累不该超过 300 元：{CAPITAL - r['nav_a']}"


def test_delisted_position_hits_both_calibers_and_pessimistic_is_worse():
    p = _synthetic_panel()
    r = settle(p, ["000005.SZ"], _entry(), 3)
    assert r["delisted_in_horizon"] == 1
    assert r["picks"][0]["caliber"] == "退市A(末根卖出)"
    assert r["nav_b"] < r["nav_a"], "悲观口径必须严格更差，否则退市没被记账"
    # 摘牌前这只票只涨了 ~0.1%/天×252 天 ≈ 28%，A 口径赚钱、B 口径直接归零
    assert r["nav_b"] < CAPITAL < r["nav_a"]
    assert r["ann_b_pct"] < 0.0 < r["ann_a_pct"]


def test_features_never_see_post_entry_data():
    p = _synthetic_panel()
    i = p.asof_index("600002.SH", _entry())
    f = M.features_at(p, "600002.SH", i)
    assert f is not None
    assert abs(f["mom12"]) < 1e-9, "入场后的暴涨混进了 12 月动量 = 后视偏差"
    assert f["vol250"] < 1e-9
    entry_idx = int((p.dates["600002.SH"] <= _entry()).sum()) - 1
    assert entry_idx == i


def test_cash_never_negative_and_lots_are_integers():
    p = _synthetic_panel()
    r = settle(p, ["600001.SH", "000005.SZ", "600002.SH"], _entry(), 3)
    for x in r["picks"]:
        assert x["lots"] >= 1 and float(x["lots"]).is_integer()
        assert x["lots"] * LOT * x["buy_price"] <= CAPITAL / 3 + 1.0
    # 独立复算：净值 = 本金 + Σ每笔净亏（费用已含在 pnl 内）
    replay = CAPITAL + sum(x["pnl_a"] for x in r["picks"])
    assert abs(replay - r["nav_a"]) <= 1.0, f"净值与逐笔复算差 {replay - r['nav_a']:.2f} 元"


def test_survivor_settles_at_exit_date_not_at_panel_end():
    """活着的票必须按到期日结算、不许被算成退市 —— 否则持有期会被悄悄拉长成「拿到面板末尾」。"""
    p = _synthetic_panel()
    entry, horizon = _entry(), 2
    d = p.dates["600002.SH"]
    exit_target = entry + pd.DateOffset(years=horizon)
    assert exit_target < d[-1], "合成面板太短，这条测试失去意义"
    res = M.settle(p, ["600002.SH"], entry, horizon)
    assert res["n"] == 1
    assert res["delisted_in_horizon"] == 0, res
    assert res["picks"][0]["caliber"] == "到期卖出", res["picks"][0]
    j = int(np.searchsorted(d.values, np.datetime64(exit_target), side="right") - 1)
    assert j < len(d) - 1, (d[j], d[-1])
    assert abs((d[j] - exit_target).days) <= 15, (d[j], exit_target)


def test_price_cap_excludes_unaffordable_stocks():
    """K 只等权分 9 万 → 单价超过 9万/K/100 的票根本买不起，必须在选择阶段剔除。"""
    p = _synthetic_panel()
    k = 2
    codes = M.pick(p, _entry(), "low_vol", k, 7)
    for c in codes:
        i = p.asof_index(c, _entry())
        assert p.close[c][i] <= CAPITAL / k / LOT + 1e-9
    # 造一只 500 元的票：K=2 上限 450 元 → 即便波动率最低也不许选
    p.close["600999.SH"] = np.full(1200, 500.0, dtype="float32")
    p.tr["600999.SH"] = np.full(1200, 500.0, dtype="float32")
    p.amount["600999.SH"] = np.full(1200, 9e9, dtype="float32")
    p.dates["600999.SH"] = p.dates["600001.SH"]
    p.meta["600999.SH"] = {"delisted": False, "outDate": None}
    codes = M.pick(p, _entry(), "low_vol", k, 7)
    assert "600999.SH" not in codes


# ---------------- B 组：产物 JSON 的无偏宇宙闸门 ----------------

def _data() -> dict:
    return json.load(open(OUT_JSON))


@pytest.mark.skipif(not os.path.exists(OUT_JSON), reason="长窗口研究还没跑（跑完这条必须绿）")
def test_universe_actually_contains_delisted_stocks():
    d = _data()
    u = d["universe"]
    assert u["panel_stocks"] >= 5000, u
    assert u["panel_delisted"] >= 250, f"退市股没进样本：{u}"


@pytest.mark.skipif(not os.path.exists(OUT_JSON), reason="长窗口研究还没跑（跑完这条必须绿）")
def test_delistings_were_actually_experienced_in_holdings():
    d = _data()
    tot = sum(r["delisted_in_horizon"] for r in d["detail"])
    cohorts_with = sum(1 for r in d["detail"] if r["delisted_in_horizon"] > 0)
    assert tot >= 10, f"10 期 × 100 次选股一次退市都没碰上 = 池子被动过（{tot}）"
    assert cohorts_with >= 3, cohorts_with


@pytest.mark.skipif(not os.path.exists(OUT_JSON), reason="长窗口研究还没跑（跑完这条必须绿）")
def test_nav_matches_per_trade_replay_and_pessimistic_gap_is_real():
    d = _data()
    worst_gap = 0.0
    for r in d["detail"]:
        for run in r["runs"]:
            replay = CAPITAL + sum(x["pnl_a"] for x in run["picks"])
            assert abs(replay - run["nav_a"]) <= 1.0, (r["entry"], r["rule"], run["nav_a"], replay)
            if run["delisted_in_horizon"]:
                recovered = 0.0
                for x in run["picks"]:
                    if x["caliber"] != "到期卖出":
                        sell = x["lots"] * LOT * x["buy_price"] * (1.0 + x["ret_pct"] / 100.0)
                        recovered += sell - max(sell * COMM_RATE, COMM_MIN) - sell * STAMP
                gap = run["nav_a"] - run["nav_b"]
                worst_gap = max(worst_gap, abs(gap - recovered))
                assert gap >= recovered - 5.0, (r["entry"], r["rule"], gap, recovered)
    assert worst_gap < 50.0, f"退市记账与复算最大偏差 {worst_gap:.2f} 元"


@pytest.mark.skipif(not os.path.exists(OUT_JSON), reason="长窗口研究还没跑（跑完这条必须绿）")
def test_every_pick_was_affordable_and_lots_integral():
    d = _data()
    for r in d["detail"]:
        cap = CAPITAL / r["k"] / LOT
        for run in r["runs"]:
            for x in run["picks"]:
                assert x["buy_price"] <= cap + 1e-6, (r["entry"], r["rule"], x)
                assert isinstance(x["lots"], int) and x["lots"] >= 1


@pytest.mark.skipif(not os.path.exists(OUT_JSON), reason="长窗口研究还没跑（跑完这条必须绿）")
def test_random_rule_used_three_seeds():
    """单种子随机对照极差可达 0.19–0.46 夏普 —— 一条随机基准线不许只跑一个种子。"""
    d = _data()
    rnd = [r for r in d["detail"] if r["rule"] == "random"]
    assert rnd, "没有随机基准"
    assert all(len(r["runs"]) >= 3 for r in rnd), [len(r["runs"]) for r in rnd]
    assert len({run["seed"] for r in rnd for run in r["runs"]}) >= 3


@pytest.mark.skipif(not os.path.exists(OUT_JSON), reason="长窗口研究还没跑（跑完这条必须绿）")
def test_valuation_rules_are_declared_untestable_not_silently_dropped():
    d = _data()
    assert d["excluded_rules"], "低PE/低PB/高ROE 为什么不能测，必须写明，不许默默删掉"
    assert any("退市" in s for s in d["excluded_rules"])
    rules = {s["rule"] for s in d["summary"]}
    assert {"low_vol", "mom12", "big_amount", "random"} <= rules
    assert not ({"low_pe", "low_pb", "high_roe"} & rules)
