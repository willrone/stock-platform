"""小资金可执行性核算：把目标权重换成真实"手"（ETF 100 份/手），看实际权重、剩余现金、成本。

用途：本金 <10 万时，策略的可行边界由**颗粒度**和**最低佣金 5 元/笔**决定，不由因子决定。
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

ETF_DIR = "/Users/rone/Projects/stock-platform/data/etf/parquet"
A, B = "2023-05-12", "2026-05-08"
COMMISSION_RATE, MIN_COMMISSION = 0.0003, 5.0  # 券商常见：万3，单笔最低 5 元；ETF 免印花税
LOT = 100


def load(code: str, a: str, b: str) -> pd.Series:
    d = pd.read_parquet(os.path.join(ETF_DIR, f"{code}.parquet"))
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d = d[(d["date"] >= a) & (d["date"] <= b)].sort_values("date")
    p = pd.to_numeric(d["close"], errors="coerce")
    ok = np.isfinite(p) & (p > 0)
    return pd.Series(p[ok].astype(float).values, index=d.loc[ok, "date"])


def commission(amount: float) -> float:
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def plan(capital: float, weights: dict[str, tuple[str, float]], price_date: str) -> None:
    """按最新价把目标权重换算成整数手，报实际权重与成本。"""
    label = " + ".join(f"{name}{w*100:.0f}%" for name, (code, w) in weights.items())
    print(f"\n目标：{label}｜本金 {capital:,.0f} 元")
    print(f"{'标的':16s} {'代码':8s} {'目标价':>9s} {'目标金额':>10s} {'手数':>5s} {'实际金额':>10s} {'实际权重':>9s} {'单边佣金':>9s}")
    total_in, total_comm = 0.0, 0.0
    prices = {}
    for name, (code, w) in weights.items():
        s = load(code, price_date, price_date)
        if s.empty:
            s = load(code, "2026-01-01", "2026-11-05")
        p = float(s.iloc[-1])
        prices[code] = p
        target = capital * w
        lots = int(target // (p * LOT))
        actual = lots * LOT * p
        comm = commission(actual)
        total_in += actual
        total_comm += comm
        print(f"{name:16s} {code:8s} {p:9.3f} {target:10,.0f} {lots:5d} {actual:10,.0f} {actual/capital*100:8.1f}% {comm:9.2f}")
    print(f"  合计投入 {total_in:,.0f} 元（{total_in/capital*100:.1f}%），闲置现金 {capital-total_in:,.0f} 元"
          f"；建仓佣金合计 {total_comm:.2f} 元 = 本金的 {total_comm/capital*100:.3f}%")
    for name, (code, w) in weights.items():
        unit = prices[code] * LOT
        print(f"    {name} 一手 = {unit:,.0f} 元 = 本金的 {unit/capital*100:.1f}%（颗粒度）")


def simulate(capital: float, weights: dict[str, tuple[str, float]], rebalance: str = "yearly") -> None:
    """按整数手建仓、每年首个交易日按目标权重换回整数手，算真实净值（含 5 元最低佣金）。"""
    codes = [c for _, (c, _) in weights.items()]
    series = {c: load(c, A, B) for c in codes}
    idx = series[codes[0]].index
    for c in codes:
        idx = idx.intersection(series[c].index)
    units = {c: 0.0 for c in codes}
    cash = capital
    navs = []
    trades = 0
    for i, dt in enumerate(idx):
        if i == 0 or (rebalance == "yearly" and dt[:4] != idx[i - 1][:4]):
            total = cash + sum(units[c] * float(series[c].loc[dt]) for c in codes)
            for name, (c, w) in weights.items():  # weights 的 key 是名称，值是 (代码, 权重) —— 别解反
                p = float(series[c].loc[dt])
                want_lots = int((total * w) // (p * LOT))
                want_units = want_lots * LOT          # units[c] 存的是"份"，不是"手"，别混
                delta_units = want_units - units[c]
                if delta_units:
                    amount = abs(delta_units) * p
                    cash -= delta_units * p
                    cash -= commission(amount)
                    units[c] = want_units
                    trades += 1
        navs.append(cash + sum(units[c] * float(series[c].loc[dt]) for c in codes))
    nav = pd.Series(navs, index=idx, dtype=float)
    r = nav.pct_change(fill_method=None).fillna(0.0)
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (244.0 / len(nav)) - 1
    vol = r.std(ddof=1) * math.sqrt(244)
    dd = float((nav / nav.cummax() - 1).min())
    print(f"  {rebalance} 再平衡（整数手 + 5元最低佣金）：累计 {(nav.iloc[-1]/nav.iloc[0]-1)*100:+.2f}% "
          f"年化 {ann*100:+.2f}% 波动 {vol*100:.2f}% 夏普 {ann/vol:.4f} 回撤 {dd*100:+.2f}% 调仓笔数 {trades}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=90_000.0)
    args = ap.parse_args()
    cap = args.capital
    CANDIDATES = {
        "红利低波50/城投债ETF 60/40": {"红利低波50": ("515450", 0.60), "城投债ETF": ("511220", 0.40)},
        "红利低波50/城投债ETF 50/50": {"红利低波50": ("515450", 0.50), "城投债ETF": ("511220", 0.50)},
        "红利低波50/国债ETF 60/40": {"红利低波50": ("515450", 0.60), "国债ETF": ("511010", 0.40)},
        "红利低波50/城投债/国债 50/25/25": {"红利低波50": ("515450", 0.50), "城投债ETF": ("511220", 0.25), "国债ETF": ("511010", 0.25)},
    }
    print(f"=== 建仓颗粒度核算（本金 {cap:,.0f} 元，按 2026 年最新收盘）===")
    for label, w in CANDIDATES.items():
        plan(cap, w, "2026-10-01")
    print("\n=== 回测窗口 2023-05-12→2026-05-08 的整数手模拟（含最低佣金）===")
    for label, w in CANDIDATES.items():
        print(f"\n{label}")
        simulate(cap, w, "buy_hold")
        simulate(cap, w, "yearly")
