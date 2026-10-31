#!/usr/bin/env python3
"""诊断：袖套撮合器的「买 / 卖超配」顺序对**类间等权**的影响。

## 为什么必须做这个实验

多资产线的核心承诺是「gold / bond / commodity 三类**各 1/3**」。
parity 回测（产品口径，2023-05 → 2026-05，141 次调仓）实测：
类权重最大偏离**均值 2.85 pt / 最差 7.10 pt** —— 承诺没有兑现。

根因是撮合顺序：

    ① 卖非目标 → ② 买不足的（pro-rata 缩放） → ③ 卖超配的

② 在 ③ 之前 → **③ 卖超配释放的现金，这一轮用不上**；
② 只受「调仓前就有的现金」约束（≈4%）。
于是当某类超配（比如黄金涨到 40%），② 想把债券/商品买回来时现金不够，
pro-rata 把买入额按比例砍掉 → **欠配的类永远补不回来**。

## 本脚本做什么

只跑袖套（9 只 ETF，无 A 腿）→ 秒级，可以自由切换口径做 A/B：

  A. buy_first（现状）：② 买 → ③ 卖超配
  B. sell_first（修好）：③ 卖超配 → ② 买

对比三件事：
  1. 类权重最大偏离（调仓日快照）
  2. 袖套自己的夏普
  3. 袖套净值末值

## 口径（必须与研究侧一致，否则不可比）

  - 成交价 = 当日 close ± 滑点（ETF_SLIPPAGE）
  - 佣金 0.03%（ETF 免印花税），买卖都扣
  - 每 21 个交易日调一次（研究侧 P3.build_targets 的网格）
  - 目标：三类各 1/3，类内 3 只等权 → 每只 1/9

## 结论用法

如果 B 明显优于 A → 说明「先买后卖」是**实现缺陷**，必须修，
且修完要**重跑全部多资产研究**（研究端口径也会变，1.4114 这个基准作废）。
如果 B 也不明显好 → 说明「类间等权」在 21 天调仓 + 整手约束下
**物理上做不到**，应该改承诺（改成「月度再平衡至近似等权」）而不是改实现。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.services.trading.multi_asset_service import (  # noqa: E402
    ETF_COMMISSION,
    ETF_SLIPPAGE,
    ETF_PARQUET_FROZEN,
    SLEEVE_CLASSES,
    SLEEVE_HELD,
    load_etf_bars,
)

CAPITAL = 500_000.0
REB_EVERY = 21
BOARD_LOT = 100

# 3 类 × 3 只
ALL_CODES: List[str] = sorted({c for lst in SLEEVE_HELD.values() for c in lst})
CLASS_OF: Dict[str, str] = {
    c: cls for cls, lst in SLEEVE_HELD.items() for c in lst
}


def target_weights() -> Dict[str, float]:
    """三类各 1/3，类内等权 → 每只 1/9。"""
    w: Dict[str, float] = {}
    for cls in SLEEVE_CLASSES:
        codes = SLEEVE_HELD[cls]
        for c in codes:
            w[c] = (1.0 / len(SLEEVE_CLASSES)) / len(codes)
    return w


def load_closes(codes: List[str]) -> pd.DataFrame:
    bars = load_etf_bars(codes, base_dir=ETF_PARQUET_FROZEN)
    series = {}
    for c in codes:
        df = bars.frames[c]
        s = df.set_index("date")["close"].astype(float)
        series[c] = s
    out = pd.DataFrame(series).sort_index()
    # ⚠️ 截到**所有标的都有数据**的区间：159981 商品 ETF 上市晚，
    #    前面几年是 NaN → 类权重会算成 0 → 偏离 999pt 的假象。
    #    必须用共同可用区间，否则 A/B 结论不可信。
    valid = out.dropna()
    if len(valid) == 0:
        raise SystemExit("❌ 9 只 ETF 没有共同可用区间")
    lo, hi = valid.index[0], valid.index[-1]
    out = out.loc[lo:hi]
    n_gap = int(out.isna().any(axis=1).sum())
    # 区间内部偶发缺价（停牌/接口抖动）→ 前向填充；
    # ETF 在非交易日缺行是正常的，填充后每个交易日都有价格。
    out = out.ffill()
    out = out.dropna()
    print(f"[diag] 共同可用区间 {out.index[0].date()} → {out.index[-1].date()}"
          f"  {len(out)} 天（区间内缺价 {n_gap} 天已前向填充）")
    assert not out.isna().any().any(), "仍有 NaN，价格序列不完整"
    return out


def simulate(
    closes: pd.DataFrame,
    order: str,
    board_lot: int = BOARD_LOT,
    capital: float = CAPITAL,
) -> Dict[str, Any]:
    """按给定顺序跑袖套。order ∈ {'buy_first', 'sell_first'}"""
    target = target_weights()
    px_all = closes
    dates = list(px_all.index)

    cash = capital
    shares: Dict[str, float] = {c: 0.0 for c in ALL_CODES}

    nav: List[Tuple[pd.Timestamp, float]] = []
    reb_log: List[Dict[str, Any]] = []

    for i, d in enumerate(dates):
        px = {c: float(px_all.at[d, c]) for c in ALL_CODES}
        if (i % REB_EVERY) == 0:
            value = cash + sum(shares[c] * px[c] for c in ALL_CODES)

            # ── 计算想买 / 想卖 ──────────────────────────────────────
            under: Dict[str, float] = {}
            over: Dict[str, float] = {}
            for c in ALL_CODES:
                want = value * target[c]
                cur = shares[c] * px[c]
                if cur < want - 1e-6:
                    under[c] = want - cur
                elif cur > want + 1e-6:
                    over[c] = cur - want

            def do_buy(pool_cash: float) -> None:
                nonlocal cash
                if not under:
                    return
                need = sum(under.values())
                scale = 1.0
                if need > 0 and pool_cash < need * (1 + ETF_COMMISSION):
                    scale = pool_cash / (need * (1 + ETF_COMMISSION))
                for c, amt in under.items():
                    a = amt * scale
                    if a <= 0:
                        continue
                    if board_lot > 0:
                        # 整手：向下取整，零头回填用「误差下降最大」准则
                        buy_px = px[c] * (1 + ETF_SLIPPAGE)
                        lots = int((a / buy_px) // board_lot)
                        while lots > 0:
                            cost = lots * board_lot * buy_px
                            if cost * (1 + ETF_COMMISSION) <= cash:
                                break
                            lots -= 1
                        if lots <= 0:
                            continue
                        sh = lots * board_lot
                        amt_actual = sh * buy_px
                        cost = amt_actual * ETF_COMMISSION
                        if amt_actual + cost > cash + 1e-6:
                            continue
                    else:
                        buy_px = px[c] * (1 + ETF_SLIPPAGE)
                        sh = a / buy_px
                        amt_actual = a
                        cost = amt_actual * ETF_COMMISSION
                    cash -= amt_actual + cost
                    shares[c] += sh

            def do_sell() -> None:
                nonlocal cash
                for c, amt in over.items():
                    sell_px = px[c] * (1 - ETF_SLIPPAGE)
                    sh = amt / sell_px
                    if board_lot > 0:
                        sh = float(int(sh // board_lot) * board_lot)
                    if sh <= 0:
                        continue
                    sh = min(sh, shares[c])
                    proceeds = sh * sell_px
                    commission = proceeds * ETF_COMMISSION
                    cash += proceeds - commission
                    shares[c] -= sh

            if order == "sell_first":
                do_sell()
                do_buy(cash)
            else:
                do_buy(cash)
                do_sell()

            # 调仓后快照
            cls_val = {cls: 0.0 for cls in SLEEVE_CLASSES}
            for c in ALL_CODES:
                cls_val[CLASS_OF[c]] += shares[c] * px[c]
            tot = sum(cls_val.values())
            cw = {k: v / tot for k, v in cls_val.items()} if tot > 0 else {}
            invested = tot / value if value > 0 else 0.0
            reb_log.append({
                "date": str(d.date()),
                "dev": max(abs(v - 1 / 3) for v in cw.values()) if cw else 9.99,
                "cw": cw,
                "invested": invested,
            })

        # 每日盯市
        v = cash + sum(shares[c] * float(px_all.at[d, c]) for c in ALL_CODES)
        nav.append((d, v))

    # ── 指标 ─────────────────────────────────────────────────────────
    s = pd.Series({d: v for d, v in nav}).sort_index()
    rets = s.pct_change(fill_method=None).dropna()
    ann_vol = float(rets.std(ddof=1) * np.sqrt(252))
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)) if ann_vol > 0 else 0.0
    dd = float((s / s.cummax() - 1).min())
    devs = [r["dev"] for r in reb_log]
    return {
        "order": order,
        "board_lot": board_lot,
        "n_reb": len(reb_log),
        "sharpe": round(sharpe, 4),
        "ann_vol": round(ann_vol, 4),
        "maxdd": round(dd, 4),
        "final": round(float(s.iloc[-1]), 2),
        "dev_mean": round(float(np.mean(devs)), 4) if devs else None,
        "dev_max": round(float(np.max(devs)), 4) if devs else None,
        "dev_last10_mean": round(float(np.mean(devs[-10:])), 4) if devs else None,
        "invested_mean": round(float(np.mean([r["invested"] for r in reb_log])), 4) if reb_log else None,
        "reb_log": reb_log,
        "nav": [[str(d.date()), round(v, 2)] for d, v in nav],
    }


def main() -> int:
    closes = load_closes(ALL_CODES)
    print(f"[diag] ETF 行情 {closes.index[0].date()} → {closes.index[-1].date()}"
          f"  {len(closes)} 天")

    results = {}
    for order in ("buy_first", "sell_first"):
        for lot in (0, BOARD_LOT):
            key = f"{order}_lot{lot}"
            r = simulate(closes, order, board_lot=lot)
            results[key] = r
            print(f"[diag] {key:22s} 夏普 {r['sharpe']:+.4f} | 回撤 {r['maxdd']*100:6.2f}% "
                  f"| 末值 {r['final']:>12,.0f} | 类权重偏离 均 {r['dev_mean']*100:5.2f}pt "
                  f"最大 {r['dev_max']*100:5.2f}pt | 投入 {r['invested_mean']*100:5.2f}%")

    out = os.path.join(ROOT, "scripts", "diag_sleeve_class_order.json")
    with open(out, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(f"[diag] 产物 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
