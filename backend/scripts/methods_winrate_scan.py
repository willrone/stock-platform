#!/usr/bin/env python3
"""胜率杠杆扫描（组合构建层范围内）。

背景：用户目标是「夏普 ≥1.5、胜率 ≥ 75%、回撤 ≤20%」。
- 引擎基线 honest_composite：夏普 0.7992 / **胜率 67.06%** / 回撤 −16.46%
- 中性化后：夏普 1.0524 / 胜率 57.64% / 回撤 −10.62%（夏普↑ 胜率↓）

引擎的胜率口径（`portfolio_manager_array.py` L666-674）：
  胜率 = 盈利的 SELL 笔数 / (盈利 SELL + 亏损 SELL)，即**逐笔平仓**口径。

## 测哪些杠杆（都在组合构建层，不改因子）
1. **n_drop**：每周强制卖出的最差持仓数。n_drop 越大 → 越频繁地「割掉还没跌的票」
   → 胜率低；n_drop 越小 → 只卖真正最差的 → 胜率高，但换手也低。
2. **topk**：持仓数。topk 小 → 集中 → 单笔波动大。
3. **hold_thresh**（引擎语义 = 排名缓冲）：掉出 topk+hold_thresh 才卖。
4. **调仓频率**：周频 vs 日频 vs 月频。

## 输出
每组：夏普 / 收益 / 回撤 / **胜率** / 盈亏比 / 笔数 —— 逐笔口径，
与引擎一致（卖出平仓笔）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import methods_lab as M  # noqa: E402
from methods_research_v2 import neutralize_score  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "backtest")
TRIALS: List[Dict[str, Any]] = []


def record(name, group, res, note="", extra=None):
    e = {
        "name": name, "group": group,
        "sharpe_ratio": round(float(res["sharpe_ratio"]), 4),
        "total_return": round(float(res["total_return"]), 4),
        "max_drawdown": round(float(res["max_drawdown"]), 4),
        "total_trades": int(res["total_trades"]),
        "volatility": round(float(res.get("volatility", 0.0)), 4),
        "yearly": res.get("yearly", {}),
        "note": note,
    }
    if extra:
        e.update(extra)
    TRIALS.append(e)
    print(f"  {name[:38]:40s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}  {note[:40]}", flush=True)
    return e


def simulate_winrate(
    score: np.ndarray,
    mats: Dict[str, Any],
    trad: Dict[str, np.ndarray],
    cfg: L.SimConfig,
) -> Dict[str, Any]:
    """复刻 ensemble_lab.simulate，但按**引擎口径**算胜率（逐笔平仓）。

    引擎口径：卖出平仓时按 FIFO 均价算 pnl，pnl>0 算赢。
    """
    master: pd.DatetimeIndex = mats["master"]
    close = mats["close"]
    n_d, n_s = close.shape
    codes = mats["codes"]
    sc = score.copy()
    sc[~np.isfinite(sc)] = np.nan
    lu = L.limit_up_mask(mats)
    sc[lu] = np.nan
    reb = L._rebalance_mask(master, cfg)
    sc[~reb] = np.nan
    if cfg.lag > 0:
        lagged = np.full_like(sc, np.nan)
        lagged[cfg.lag:] = sc[:-cfg.lag]
        sc = lagged

    cash = cfg.initial_cash
    qty = np.zeros(n_s)
    avg_cost = np.zeros(n_s)
    last_px = np.zeros(n_s)
    n_rows_seen = np.zeros(n_s, dtype=np.int64)
    can_buy = trad["can_buy"]
    can_sell = trad["can_sell"]
    max_buy = trad["max_buy_amount"]
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    sell_pnls: List[float] = []
    n_buys = 0

    for t in range(n_d):
        px = close[t]
        valid = np.isfinite(px) & (px > 0)
        if not valid.any():
            continue
        last_px[valid] = px[valid]
        n_rows_seen += valid.astype(np.int64)
        warm = n_rows_seen >= 21
        row = sc[t]
        have = valid & np.isfinite(row) & warm
        if have.any():
            idx = np.where(have)[0]
            order = idx[np.lexsort((np.array([codes[i] for i in idx]),
                                     -row[idx]))]
            holdings = np.where(qty > 0)[0]
            holdings_set = set(int(c) for c in holdings)

            def _pv() -> float:
                return cash + float(np.sum(qty * np.where(
                    np.isfinite(last_px), last_px, 0.0)))

            def _buy(c: int, pv: float) -> None:
                nonlocal cash, n_buys
                if not can_buy[t, c]:
                    return
                px_c = px[c]
                max_pos_val = pv * cfg.max_position_size
                avail = min(max_pos_val, cash * (1 - cfg.cash_reserve))
                if max_buy is not None:
                    mb = max_buy[t, c]
                    if np.isfinite(mb) and mb > 0:
                        avail = min(avail, mb)
                if avail <= 0:
                    return
                q = int(avail / px_c / cfg.board_lot) * cfg.board_lot
                if q <= 0:
                    return
                cost = q * px_c * (1 + cfg.slippage)
                comm = max(cost * cfg.commission, 0.0)
                if cost + comm > cash:
                    return
                cash -= cost + comm
                old_q = qty[c]
                new_q = old_q + q
                avg_cost[c] = ((avg_cost[c] * old_q + cost) / new_q
                               if new_q else 0.0)
                qty[c] = new_q
                n_buys += 1

            if len(holdings) < cfg.topk:
                buy = [int(c) for c in order if int(c) not in holdings_set
                       ][: cfg.topk - len(holdings)]
                for c in buy:
                    _buy(c, _pv())
            else:
                rank = {int(c): i for i, c in enumerate(order)}
                held = [c for c in holdings if int(c) in rank]
                held_sorted = sorted(held, key=lambda c: rank[int(c)])
                sell = [c for c in reversed(held_sorted)
                        if rank[int(c)] >= cfg.topk + cfg.hold_thresh
                        ][: cfg.n_drop]
                sold = 0
                for c in sell:
                    if not can_sell[t, c]:
                        continue
                    px_c = px[c]
                    proceeds = qty[c] * px_c * (1 - cfg.slippage)
                    comm = max(proceeds * cfg.commission, 0.0)
                    cash += proceeds - comm
                    q = qty[c]
                    cost = q * avg_cost[c]
                    qty[c] = 0.0
                    avg_cost[c] = 0.0
                    sold += 1
                    if cost > 0:
                        sell_pnls.append(proceeds - cost)
                buy = [int(c) for c in order
                       if int(c) not in holdings_set][:sold]
                for c in buy:
                    _buy(c, _pv())

        val_px = np.where(np.isfinite(px) & (px > 0), px, last_px)
        equity_curve.append((master[t], cash + float(np.sum(qty * val_px))))

    m = L._metrics(equity_curve, len(sell_pnls), master)
    wins = [p for p in sell_pnls if p > 0]
    losses = [p for p in sell_pnls if p < 0]
    denom = len(wins) + len(losses)
    m["win_rate"] = (len(wins) / denom) if denom else 0.0
    m["n_sell_closes"] = len(sell_pnls)
    m["payoff_ratio"] = (
        float(np.sum(wins) / abs(np.sum(losses))) if losses else float("nan"))
    m["avg_win"] = float(np.mean(wins)) if wins else 0.0
    m["avg_loss"] = float(np.mean(losses)) if losses else 0.0
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_winrate_scan.json")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 92)
    print("胜率杠杆扫描（组合构建层）· 真实盘口径 lag=1")
    print("=" * 92)

    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    s42, _ = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(mats["close"], mats["volume"], 20)
    neut = neutralize_score(s42, {"size": size, "illiq": illiq})

    res_all: Dict[str, Any] = {
        "meta": {
            "window": f"{mats['master'][0].date()}→{mats['master'][-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(mats["master"]),
            "win_rate_definition": "盈利 SELL 笔数 / (盈利 SELL + 亏损 SELL)"
                                   "（与 portfolio_manager_array.py L666-674 一致）",
            "caliber": {"lag": 1, "enable_tradability": True,
                        "commission": 0.0003, "slippage": 0.0001,
                        "board_lot": 100, "cash_reserve": 0.05},
            "engine_baseline_42f": {
                "sharpe_ratio": 0.7991795250472001,
                "win_rate": 0.6705882352941176,
                "max_drawdown": -0.16460289642093642,
                "total_trades": 1080,
            },
            "engine_neut_size_illiq": {
                "sharpe_ratio": 1.0524378432259367,
                "win_rate": 0.5764331210191083,
                "max_drawdown": -0.10622457102291379,
                "total_trades": 687,
            },
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "trials": [],
    }

    print("\n  -- 基线复现（模拟器，逐笔胜率口径）--")
    for nm, sc in (("base", s42), ("neut", neut)):
        cfg = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                          max_position_size=(1 - 0.05) / 60)
        r = simulate_winrate(sc, mats, trad, cfg)
        e = record(f"WR_base_{nm}", "WR", r, f"{nm} topk=60 d=5")
        e["win_rate"] = round(float(r["win_rate"]), 4)
        e["payoff_ratio"] = round(float(r["payoff_ratio"]), 4)
        e["n_sell_closes"] = int(r["n_sell_closes"])
        print(f"      胜率 {r['win_rate']:.2%}  盈亏比 {r['payoff_ratio']:.3f}  "
              f"平仓笔数 {r['n_sell_closes']}")

    print("\n  -- n_drop 扫描（胜率的核心杠杆）--")
    for tag, sc in (("base", s42), ("neut", neut)):
        for n_drop in (1, 2, 3, 5, 8, 12):
            cfg = L.SimConfig(topk=60, n_drop=n_drop, lag=1,
                              rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / 60)
            r = simulate_winrate(sc, mats, trad, cfg)
            e = record(f"WR_{tag}_d{n_drop}", "WR", r,
                       f"{tag} n_drop={n_drop}")
            e["win_rate"] = round(float(r["win_rate"]), 4)
            e["payoff_ratio"] = round(float(r["payoff_ratio"]), 4)
            print(f"      胜率 {r['win_rate']:.2%}  盈亏比 {r['payoff_ratio']:.3f}")

    print("\n  -- topk 扫描 --")
    for tag, sc in (("base", s42), ("neut", neut)):
        for topk in (20, 30, 40, 60, 80, 100):
            cfg = L.SimConfig(topk=topk, n_drop=max(1, topk // 12), lag=1,
                              rebalance_days="weekly",
                              max_position_size=(1 - 0.05) / topk)
            r = simulate_winrate(sc, mats, trad, cfg)
            e = record(f"WR_{tag}_k{topk}", "WR", r, f"{tag} topk={topk}")
            e["win_rate"] = round(float(r["win_rate"]), 4)
            e["payoff_ratio"] = round(float(r["payoff_ratio"]), 4)
            print(f"      胜率 {r['win_rate']:.2%}  盈亏比 {r['payoff_ratio']:.3f}")

    print("\n  -- 调仓频率 --")
    for tag, sc in (("base", s42), ("neut", neut)):
        for freq in ("daily", "weekly", "monthly"):
            cfg = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days=freq,
                              max_position_size=(1 - 0.05) / 60)
            r = simulate_winrate(sc, mats, trad, cfg)
            e = record(f"WR_{tag}_{freq}", "WR", r, f"{tag} {freq}")
            e["win_rate"] = round(float(r["win_rate"]), 4)
            e["payoff_ratio"] = round(float(r["payoff_ratio"]), 4)
            print(f"      胜率 {r['win_rate']:.2%}  盈亏比 {r['payoff_ratio']:.3f}")

    res_all["trials"] = TRIALS
    res_all["n_trials"] = len(TRIALS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
