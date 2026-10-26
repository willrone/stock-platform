#!/usr/bin/env python3
"""引擎保真版模拟器 —— 复刻 ``TopkDropoutTradeModeExecutor`` 的真实语义。

## 与 ensemble_lab.simulate 的差异（2026-10-27 实测对账后修正）
1. **卖出候选**：引擎的 ``sell_candidates`` 要求
   ``code in tradeable_codes``（当天有分数）且
   ``rank_index.get(code, len(ranked)) >= topk + hold_thresh``。
   → 持仓当天**没有分数**（NaN）时 rank 取默认值 = len(ranked)（最差），
   但 ``code in tradeable_codes`` 为假 → **不卖**。
   旧模拟器把 NaN 持仓当成 rank=最差 → 会误卖。
2. **买入候选**：引擎 ``buy_candidates = [code for code,_ in ranked
   if code not in holdings_set][:n_drop]`` —— 从**完整排名**里取，
   不是只从 topk 里取；且已经持有的会被 holdings_set 过滤。
3. **买入金额**：引擎用 ``portfolio_value * max_position_size`` 作为**上限**，
   再与 ``cash * (1 - cash_reserve)`` 取小；且**允许对已有持仓加仓**
   （``available = max_position_value - current_position_value``）。
4. **卖出后不立即补买同一天已被卖出的股票**（holdings_set 在换仓前快照）。

## 用法
    from ensemble_sim import simulate, SimConfig
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import ensemble_lab as L  # noqa: E402


class SimConfig(L.SimConfig):
    pass


def simulate(
    score: np.ndarray,
    mats: Dict[str, Any],
    trad: Dict[str, np.ndarray],
    cfg: SimConfig,
    selector: Optional[Any] = None,
) -> Dict[str, Any]:
    """复刻引擎 TopK-Dropout 执行（42f/rsi 实测与引擎一致）。"""
    master: pd.DatetimeIndex = mats["master"]
    close: np.ndarray = mats["close"]
    n_d, n_s = close.shape
    codes = mats["codes"]
    sel = selector or L.TopKEqualWeightSelector(cfg)

    # ---- 分数预处理：涨停掩码 → 调仓日掩码 → lag（与策略类同序）----
    sc = score.copy()
    sc[~np.isfinite(sc)] = np.nan
    sc[L.limit_up_mask(mats)] = np.nan
    reb = L._rebalance_mask(master, cfg)
    sc[~reb] = np.nan
    if cfg.lag > 0:
        lagged = np.full_like(sc, np.nan)
        lagged[cfg.lag:] = sc[:-cfg.lag]
        sc = lagged
    if hasattr(sel, "_score"):
        sel._score = sc

    cash = cfg.initial_cash
    qty = np.zeros(n_s, dtype=np.float64)
    last_px = np.zeros(n_s, dtype=np.float64)
    equity_curve = []
    n_trades = 0

    can_buy = trad["can_buy"] if cfg.use_tradability else np.ones((n_d, n_s), bool)
    can_sell = trad["can_sell"] if cfg.use_tradability else np.ones((n_d, n_s), bool)
    max_buy = trad["max_buy_amount"] if cfg.use_tradability else None

    n_rows_seen = np.zeros(n_s, dtype=np.int64)

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
            order = idx[np.lexsort((np.array([codes[i] for i in idx]), -row[idx]))]
            holdings = np.where(qty > 0)[0]
            holdings_set = set(int(c) for c in holdings)

            def _pv() -> float:
                return cash + float(np.sum(qty * np.where(
                    np.isfinite(last_px), last_px, 0.0)))

            def _buy(c: int) -> bool:
                nonlocal cash, n_trades
                if not can_buy[t, c]:
                    return False
                pc = px[c]
                if not np.isfinite(pc) or pc <= 0:
                    return False
                # 引擎：max_position_value = PV * mps；再与 cash*(1-reserve) 取小
                max_pos_val = _pv() * cfg.max_position_size
                cur_val = qty[c] * pc
                avail = min(max_pos_val - cur_val, cash * (1 - cfg.cash_reserve))
                if max_buy is not None:
                    mb = max_buy[t, c]
                    if np.isfinite(mb) and mb > 0:
                        avail = min(avail, mb)
                if avail <= 0:
                    return False
                q = int(avail / pc / cfg.board_lot) * cfg.board_lot
                if q <= 0:
                    return False
                cost = q * pc * (1 + cfg.slippage)
                comm = max(cost * cfg.commission, 0.0)
                if cost + comm > cash:
                    return False
                cash -= cost + comm
                old_q = qty[c]
                new_q = old_q + q
                avg = 0.0
                qty[c] = new_q
                n_trades += 1
                return True

            keep, tgt_w, buy_pool = sel.select(order, t, holdings, len(holdings))
            keep_set = set(int(c) for c in keep)

            if len(holdings) < cfg.topk:
                # 初始建仓：从完整排名里取未持有的
                buy = [int(c) for c in buy_pool
                       if int(c) not in holdings_set][: cfg.topk - len(holdings)]
                for c in buy:
                    _buy(c)
            else:
                # ---- 卖出：引擎语义 ----
                # tradeable_codes = 当天有分数的集合（ranked 的 key）
                rank = {int(c): i for i, c in enumerate(order)}
                n_ranked = len(order)
                held_sorted = sorted(
                    [int(c) for c in holdings],
                    key=lambda c: rank.get(int(c), n_ranked),
                )
                exit_rank = cfg.topk + cfg.hold_thresh + max(0, cfg.buffer)
                # 引擎：code in tradeable_codes（=ranked 的 key）才可卖
                sell = [c for c in reversed(held_sorted)
                        if c in rank and rank[c] >= exit_rank][: cfg.n_drop]
                sold = 0
                for c in sell:
                    if not can_sell[t, c]:
                        continue
                    pc = px[c]
                    if not np.isfinite(pc) or pc <= 0:
                        continue
                    proceeds = qty[c] * pc * (1 - cfg.slippage)
                    comm = max(proceeds * cfg.commission, 0.0)
                    cash += proceeds - comm
                    qty[c] = 0.0
                    sold += 1
                    n_trades += 1
                # ---- 买入：引擎语义（从完整排名取，剔除换仓前的持仓）----
                buy = [int(c) for c in buy_pool
                       if int(c) not in holdings_set][:sold]
                for c in buy:
                    _buy(c)

                # 分层/逆波动率变体：强制调平到目标持仓集合
                if getattr(sel, "rebalance_to_targets", False) and tgt_w:
                    pv = _pv()
                    for c in sorted(keep_set):
                        w_target = float(tgt_w.get(int(c), 0.0))
                        if w_target <= 0:
                            continue
                        pc = px[c]
                        if not np.isfinite(pc) or pc <= 0:
                            continue
                        cur_val = qty[c] * pc
                        tgt_val = pv * w_target
                        if tgt_val > cur_val * 1.02:
                            need = tgt_val - cur_val
                            if not can_buy[t, c]:
                                continue
                            q = int(need / pc / cfg.board_lot) * cfg.board_lot
                            if q <= 0:
                                continue
                            cost = q * pc * (1 + cfg.slippage)
                            comm = max(cost * cfg.commission, 0.0)
                            if cost + comm > cash:
                                continue
                            cash -= cost + comm
                            qty[c] += q
                            n_trades += 1
                        elif cur_val > tgt_val * 1.02:
                            if not can_sell[t, c]:
                                continue
                            excess = cur_val - tgt_val
                            q_sell = int(excess / pc / cfg.board_lot) * cfg.board_lot
                            q_sell = min(q_sell, int(qty[c]))
                            if q_sell <= 0:
                                continue
                            proceeds = q_sell * pc * (1 - cfg.slippage)
                            comm = max(proceeds * cfg.commission, 0.0)
                            cash += proceeds - comm
                            qty[c] -= q_sell
                            n_trades += 1

        val_px = np.where(np.isfinite(px) & (px > 0), px, last_px)
        equity = cash + float(np.sum(qty * val_px))
        equity_curve.append((master[t], equity))

    return L._metrics(equity_curve, n_trades, master)


def simulate_subset(
    score: np.ndarray,
    mats: Dict[str, Any],
    trad: Dict[str, np.ndarray],
    cfg: SimConfig,
    rows: np.ndarray,
) -> Dict[str, Any]:
    """在 master 的子区间上跑（用于 OOS 评估）。"""
    sub_mats = {k: v for k, v in mats.items()
                if k not in ("master", "codes", "stock_data")}
    sub_mats["master"] = mats["master"][rows]
    sub_mats["codes"] = mats["codes"]
    sub_mats["stock_data"] = mats.get("stock_data", {})
    sub_trad = {k: v[rows] for k, v in trad.items()}
    return simulate(score[rows], sub_mats, sub_trad, cfg)
