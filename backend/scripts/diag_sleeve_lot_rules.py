"""袖套整手分配的策略对比（快，秒级）：哪种补手规则能把「类间等权」压得最准。

背景：袖套每只目标权重 = 袖套净值/9（50 万 → 55,555 元），
但 ETF 的**一手金额**差 20 倍：
  518880 黄金ETF ~800 元/手、159937 ~800、511260 国债ETF ~11,000、
  511010 ~12,000、511020 ~11,000、159980 商品ETF ~150、159985 ~120、159981 ~200。
所以「每只 1/9」在债券类根本凑不出来 —— 只能靠补手规则去逼近。

对比 4 种规则（都用「先向下取整 + 再补手」）：
  A 只补一次、按**股数**零头最大优先（当前实现）
  B 只补一次、按**金额**零头最大优先
  C 只补一次、按**误差下降量** 2s−lot 最大优先，且只在 >0 时补
  D 不补手（纯向下取整）

用法：python3 scripts/diag_sleeve_lot_rules.py
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.trading.multi_asset_service import (  # noqa: E402
    ETF_COMMISSION,
    ETF_PARQUET_FROZEN,
    ETF_SLIPPAGE,
    SLEEVE_HELD,
    load_etf_bars,
    sleeve_target_weights,
)

BOARD_LOT = 100
CAPITAL = 500_000.0
TRADE_DAY = pd.Timestamp("2026-10-19")


def build(strategy: str) -> Dict[str, Any]:
    codes = sorted({c for lst in SLEEVE_HELD.values() for c in lst})
    bars = load_etf_bars(codes, base_dir=ETF_PARQUET_FROZEN)
    target = sleeve_target_weights()
    value = CAPITAL
    px = {c: float(bars.get(c, TRADE_DAY, "close")["price"]) for c in codes}

    plan: List[Dict[str, Any]] = []
    for code in codes:
        want = value * target[code]
        buy_px = px[code] * (1 + ETF_SLIPPAGE)
        raw = want / buy_px
        lots = int(raw // BOARD_LOT)
        s = raw - lots * BOARD_LOT          # 股数零头
        plan.append({
            "code": code, "px": px[code], "buy_px": buy_px, "lots": lots,
            "s_shares": s, "s_value": s * buy_px,
            "lot_value": BOARD_LOT * buy_px,
        })

    # 可分配的闲钱 = 现金 − 向下取整后的总成本
    floor_cost = sum(it["lots"] * it["lot_value"] * (1 + ETF_COMMISSION)
                     for it in plan)
    free = CAPITAL - floor_cost

    if strategy in ("A", "B", "C"):
        used = set()
        while True:
            best = None
            for it in plan:
                if it["code"] in used:
                    continue
                need = it["lot_value"] * (1 + ETF_COMMISSION)
                if need > free + 1e-6:
                    continue
                if strategy == "A":
                    key = it["s_shares"]
                elif strategy == "B":
                    key = it["s_value"]
                else:  # C：误差下降量 2s − lot
                    key = 2 * it["s_value"] - it["lot_value"]
                    if key <= 0:
                        continue
                if best is None or key > best[0]:
                    best = (key, it)
            if best is None:
                break
            it = best[1]
            free -= it["lot_value"] * (1 + ETF_COMMISSION)
            it["lots"] += 1
            used.add(it["code"])

    # 算类权重
    vals: Dict[str, float] = {}
    total_invested = 0.0
    for it in plan:
        v = it["lots"] * BOARD_LOT * it["px"]
        total_invested += v
        cls = next((k for k, lst in SLEEVE_HELD.items() if it["code"] in lst), None)
        vals[cls] = vals.get(cls, 0.0) + v
    tot = sum(vals.values())
    cw = {k: vals[k] / tot for k in vals}
    dev = max(abs(cw[k] - 1 / 3) for k in cw)
    return {
        "strategy": strategy, "class_w": cw, "max_dev": dev,
        "invested": total_invested, "invested_ratio": total_invested / CAPITAL,
        "idle": CAPITAL - total_invested,
        "plan": plan,
    }


def main() -> int:
    print(f"袖套 {CAPITAL:,.0f} 元，每只目标 {CAPITAL/9:,.0f} 元，"
          f"成交日 {TRADE_DAY.date()}\n")
    print(f"{'策略':<34}{'最大类偏差':>12}{'投入率':>10}{'闲置':>10}")
    for st, desc in (("D", "不补手（纯向下取整）"),
                     ("A", "补一次·按股数零头最大优先"),
                     ("B", "补一次·按金额零头最大优先"),
                     ("C", "补一次·按误差下降量(2s−lot)")):
        r = build(st)
        print(f"{st} {desc:<30}{r['max_dev']*100:>11.2f}%"
              f"{r['invested_ratio']*100:>9.2f}%{r['idle']:>10,.0f}")
    print()
    r = build("C")
    print("策略 C 逐只明细：")
    for it in sorted(r["plan"], key=lambda x: x["code"]):
        cls = next((k for k, lst in SLEEVE_HELD.items() if it["code"] in lst), "")
        print(f"  {it['code']} {cls:<10} 价 {it['px']:>8.3f} "
              f"一手 {it['lot_value']:>9,.0f} 手数 {it['lots']:>5d} "
              f"缺口 {it['s_value']:>9,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
