#!/usr/bin/env python3
"""中性化后**胜率下降**的原因分析（胜率是用户目标之一：≥75%）。

现象（真实引擎 3 年）：
  基线（42 因子）      夏普 0.7992  胜率 67.06%  笔数 1080
  + size+illiq 中性化  夏普 1.0524  胜率 57.64%  笔数 687

夏普涨了但胜率跌了 —— 必须查清是「机制变化」还是「统计口径」。
三条可能：
  H1. **口径效应**：笔数从 1080 → 687。中性化后单笔平均持仓期更长，
      交易变少 → 样本从 1080 笔掉到 687 笔 → 胜率统计噪声变大。
      且「平仓即算一笔」的胜率定义对**持有越久**越不利（要跌过买入价
      才算亏）。
  H2. **收益形态变化**：中性化后收益更依赖**少数大赢家**（右尾更厚），
      胜率下降但盈亏比上升 → 夏普上升。这是「趋势型 alpha」的特征。
  H3. **真的变差**：中性化破坏了选股，只靠少数票撑着。

## 检验
  W1. 逐笔盈亏分布（模拟器可拿到逐笔）→ 分位数、盈亏比、最大单笔贡献
  W2. 去掉最大 1/5/10 笔后的夏普 → 若夏普塌了 = H3（靠少数票撑着）
  W3. 持有期分布 → H1
  W4. 分年胜率
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

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import methods_lab as M  # noqa: E402
from methods_research_v2 import neutralize_score  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "backtest")


def simulate_with_trades(
    score: np.ndarray,
    mats: Dict[str, Any],
    trad: Dict[str, np.ndarray],
    cfg: L.SimConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """复刻 ensemble_lab.simulate，但**记录逐笔交易**（买/卖成对）。

    只在研究层用（不改 ensemble_lab.py）。
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
    trades: List[Dict[str, Any]] = []
    open_lots: Dict[int, List[Dict[str, Any]]] = {}

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
                nonlocal cash
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
                open_lots.setdefault(int(c), []).append(
                    {"date": master[t], "qty": q, "cost": cost,
                     "px": px_c})

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
                    qty[c] = 0.0
                    avg_cost[c] = 0.0
                    sold += 1
                    lots = open_lots.get(int(c), [])
                    if lots:
                        # FIFO 平仓
                        rem = q
                        cost_basis = 0.0
                        entry_date = lots[0]["date"]
                        for lot in lots:
                            take = min(rem, lot["qty"])
                            cost_basis += take * lot["px"]
                            lot["qty"] -= take
                            rem -= take
                            if rem <= 0:
                                break
                        lots = [l for l in lots if l["qty"] > 1e-9]
                        if lots:
                            open_lots[int(c)] = lots
                        else:
                            open_lots.pop(int(c), None)
                        if cost_basis > 0:
                            trades.append({
                                "code": codes[int(c)],
                                "exit_date": master[t],
                                "entry_date": entry_date,
                                "gross_pnl": proceeds - cost_basis,
                                "ret": proceeds / cost_basis - 1.0,
                                "hold_days": (master[t] - entry_date).days,
                            })
                buy = [int(c) for c in order
                       if int(c) not in holdings_set][:sold]
                for c in buy:
                    _buy(c, _pv())

        val_px = np.where(np.isfinite(px) & (px > 0), px, last_px)
        equity_curve.append((master[t], cash + float(np.sum(qty * val_px))))

    m = L._metrics(equity_curve, len(trades), master)
    return m, trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_winrate.json")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--ndrop", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 88)
    print("胜率下降归因（基线 vs 中性化）")
    print("=" * 88)

    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    CFG = L.SimConfig(topk=args.topk, n_drop=args.ndrop, lag=1,
                      rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / args.topk)
    s42, _ = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(mats["close"], mats["volume"], 20)
    neut = neutralize_score(s42, {"size": size, "illiq": illiq})

    out: Dict[str, Any] = {
        "meta": {"window": f"{mats['master'][0].date()}→"
                          f"{mats['master'][-1].date()}",
                 "n_codes": len(mats["codes"]),
                 "generated_at": pd.Timestamp.now().isoformat(timespec="seconds")},
        "variants": {},
    }

    for nm, sc in (("base", s42), ("neut", neut)):
        m, trades = simulate_with_trades(sc, mats, trad, CFG)
        if not trades:
            print(f"  {nm}: 无逐笔记录", flush=True)
            continue
        rets = np.array([t["ret"] for t in trades])
        wins = rets > 0
        win_rate = float(wins.mean())
        gp = rets[rets > 0].sum()
        gl = -rets[rets < 0].sum()
        payoff = float(gp / gl) if gl > 0 else float("nan")
        hold = np.array([t["hold_days"] for t in trades
                         if t["hold_days"] is not None], dtype=float)
        # 去掉最大 N 笔后的夏普（用逐笔收益近似组合日收益的集中度）
        order = np.argsort(-rets)
        contrib = rets[order]
        cum = np.cumsum(contrib)
        total = rets.sum()
        top1 = float(contrib[0] / total) if total != 0 else float("nan")
        top5 = float(cum[:5].sum() / total) if total != 0 else float("nan")
        top10 = float(cum[:10].sum() / total) if total != 0 else float("nan")
        print(f"\n  --- {nm} ---")
        print(f"    夏普 {m['sharpe_ratio']:.3f}  笔数 {len(trades)}  "
              f"胜率 {win_rate:.2%}  盈亏比 {payoff:.3f}")
        print(f"    单笔收益: 均值 {rets.mean():+.4%} 中位 {np.median(rets):+.4%} "
              f"最好 {rets.max():+.2%} 最差 {rets.min():+.2%}")
        print(f"    分位: p10 {np.quantile(rets,0.1):+.2%} "
              f"p25 {np.quantile(rets,0.25):+.2%} "
              f"p75 {np.quantile(rets,0.75):+.2%} "
              f"p90 {np.quantile(rets,0.9):+.2%}")
        print(f"    持有期(自然日): 中位 {np.median(hold):.0f} "
              f"均值 {hold.mean():.1f} p90 {np.quantile(hold,0.9):.0f}")
        print(f"    收益集中度: 最大1笔占总盈亏 {top1:+.1%}  "
              f"前5笔 {top5:+.1%}  前10笔 {top10:+.1%}")
        out["variants"][nm] = {
            "sharpe": round(float(m["sharpe_ratio"]), 4),
            "total_return": round(float(m["total_return"]), 4),
            "max_drawdown": round(float(m["max_drawdown"]), 4),
            "n_trades": len(trades),
            "win_rate": round(win_rate, 4),
            "payoff_ratio": round(payoff, 4),
            "mean_trade_ret": round(float(rets.mean()), 5),
            "median_trade_ret": round(float(np.median(rets)), 5),
            "best_trade": round(float(rets.max()), 4),
            "worst_trade": round(float(rets.min()), 4),
            "q10": round(float(np.quantile(rets, 0.1)), 4),
            "q90": round(float(np.quantile(rets, 0.9)), 4),
            "median_hold_days": float(np.median(hold)),
            "mean_hold_days": round(float(hold.mean()), 2),
            "top1_pnl_share": round(top1, 4),
            "top5_pnl_share": round(top5, 4),
            "top10_pnl_share": round(top10, 4),
        }

    print("\n" + "=" * 88)
    print("结论判读")
    print("=" * 88)
    b = out["variants"].get("base")
    n = out["variants"].get("neut")
    if b and n:
        print(f"  胜率 {b['win_rate']:.2%} → {n['win_rate']:.2%}  "
              f"({n['win_rate']-b['win_rate']:+.2%})")
        print(f"  盈亏比 {b['payoff_ratio']:.3f} → {n['payoff_ratio']:.3f}")
        print(f"  中位持有 {b['median_hold_days']:.0f} → "
              f"{n['median_hold_days']:.0f} 日")
        print(f"  前10笔盈亏占比 {b['top10_pnl_share']:+.1%} → "
              f"{n['top10_pnl_share']:+.1%}")
        if n["payoff_ratio"] > b["payoff_ratio"]:
            print("  → 盈亏比上升 + 胜率下降 = **收益形态变右偏**（H2），"
                  "夏普上升是真实的")
        if n["median_hold_days"] > b["median_hold_days"]:
            print("  → 持有期变长 → 胜率统计受持有期影响（H1 也有贡献）")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
