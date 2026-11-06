"""汇总 ¥90,000 本金的全部回测产物：因子臂 vs 随机对照（多种子取均值 + 均值标准误）。

规则：随机对照必须 ≥3 种子；σ 大时明确写"样本不足"，不许用单种子定结论。
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st

BASE = "/Users/rone/Projects/stock-platform/data/strategy_baseline"
PREFIX = "ablation_2026-11-06_"


def load(f: str) -> dict:
    d = json.load(open(f))
    m = d.get("__meta__") or {}
    cfg = m.get("strategy_config") or {}
    ec = d.get("equity_curve") or []
    v = [float(x[1]) for x in ec]
    ann = vol = 0.0
    if len(v) > 2:
        r = [math.log(v[i] / v[i - 1]) for i in range(1, len(v))]
        ann = (v[-1] / v[0]) ** (244.0 / len(v)) - 1
        sd = st.stdev(r)
        vol = sd * math.sqrt(244)
    ph = d.get("portfolio_history") or []
    cash = sum(float(x.get("cash", 0)) for x in ph) / max(1, len(ph)) / d["backtest_config"]["initial_cash"]
    return dict(
        name=os.path.basename(f)[len(PREFIX): -len("_lag1.json")],
        line="随机对照" if "random_ctrl" in f else "25f 因子",
        capital=d["backtest_config"]["initial_cash"],
        topk=cfg.get("topk"),
        tier=(m.get("cap_tier") or {}).get("tier"),
        uni=m.get("universe_after_gate"),
        sharpe=d["sharpe_ratio"],
        ann=ann * 100,
        vol=vol * 100,
        ret=d["total_return"] * 100,
        dd=d["max_drawdown"] * 100,
        trades=d["total_trades"],
        cost=d["cost_statistics"]["cost_ratio"] * 100,
        cash=cash * 100,
        stocks=d.get("stocks_traded"),
        commission=d["cost_statistics"]["total_commission"],
    )


def stats(vals: list[float]):
    m = st.mean(vals)
    if len(vals) < 2:
        return m, 0.0, 0.0, len(vals)
    sd = st.stdev(vals)
    return m, sd, sd / math.sqrt(len(vals)), len(vals)


def main() -> None:
    rows = [load(f) for f in sorted(glob.glob(os.path.join(BASE, PREFIX + "*_lag1.json")))]
    rows = [r for r in rows if r["capital"] == 90000.0]
    if not rows:
        print("无 ¥90,000 本金产物")
        return
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["line"], r["tier"], r["topk"]), []).append(r)

    print("=== ¥90,000 本金 · 闸门后宇宙 · lag1 · 周频 ===")
    print(f"{'臂':10s} {'层':>6s} {'topk':>5s} {'宇宙':>6s} {'种子':>4s} {'夏普均值':>9s} {'年化均值':>9s} {'回撤均值':>9s} {'笔数均值':>9s} {'成本均值':>9s} {'现金均值':>9s} {'票数均值':>9s}")
    for key in sorted(groups, key=lambda k: (str(k[1]), k[2], k[0])):
        line, tier, topk = key
        g = groups[key]
        m, sd, mse, n = stats([x["sharpe"] for x in g])
        ann = st.mean([x["ann"] for x in g])
        dd = st.mean([x["dd"] for x in g])
        tr = st.mean([x["trades"] for x in g])
        cost = st.mean([x["cost"] for x in g])
        cash = st.mean([x["cash"] for x in g])
        stk = st.mean([x["stocks"] for x in g])
        uni = g[0]["uni"]
        flag = "" if (line == "25f 因子" or n >= 3) else "  ← 种子不足"
        print(f"{line:10s} {str(tier):>6s} {topk:5d} {str(uni):>6s} {n:4d} {m:9.4f} {ann:8.2f}% {dd:8.2f}% {tr:9.0f} {cost:8.2f}% {cash:8.1f}% {stk:9.0f}{flag}")

    print("\n=== 因子 − 随机（同层同 topk，随机取种子均值）===")
    for tier in sorted({k[1] for k in groups}, key=lambda x: str(x)):
        for topk in sorted({k[2] for k in groups if k[1] == tier}):
            fac = groups.get(("25f 因子", tier, topk))
            rnd = groups.get(("随机对照", tier, topk))
            if not fac or not rnd:
                continue
            fm = st.mean([x["sharpe"] for x in fac])
            rm, rsd, rmse, n = stats([x["sharpe"] for x in rnd])
            d_sh = fm - rm
            d_ann = st.mean([x["ann"] for x in fac]) - st.mean([x["ann"] for x in rnd])
            d_dd = st.mean([x["dd"] for x in fac]) - st.mean([x["dd"] for x in rnd])
            if rmse > 0:
                z = d_sh / rmse
                verdict = "超额显著" if z >= 2 else ("不确定" if z > 0 else "因子不如随机")
                print(f"  tier={str(tier):>6s} topk={topk:3d} 种子={n} Δ夏普 {d_sh:+.4f}（{z:.1f} 均值标准误, 随机σ {rsd:.4f}）→ {verdict}"
                      f"；Δ年化 {d_ann:+.2f}pp；Δ回撤 {d_dd:+.2f}pp")
            else:
                print(f"  tier={str(tier):>6s} topk={topk:3d} 种子={n} Δ夏普 {d_sh:+.4f} → 样本不足，不可判")


if __name__ == "__main__":
    main()
