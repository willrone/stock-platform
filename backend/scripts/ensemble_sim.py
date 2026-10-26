#!/usr/bin/env python3
"""引擎保真版模拟器 —— 复刻 ``TopkDropoutTradeModeExecutor`` 的真实语义。

## 与 ensemble_lab.simulate 的差异（2026-10-27 实测逐月/逐笔对账后修正）
1. **卖出候选**：引擎的 ``sell_candidates`` 要求
   ``code in tradeable_codes``（当天在 ranked 里 = 当天有分数）且
   ``rank_index.get(code, len(ranked)) >= topk + hold_thresh``。
   → 持仓当天**没有分数**（NaN）的票不在 ranked 里 → **不会被卖**。
   旧模拟器把 NaN 持仓当成 rank=最差 → 会误卖。
2. **买入候选**：引擎 ``buy_candidates = [code for code,_ in ranked
   if code not in holdings_set][:n_drop]`` —— 从**完整排名**里取
   （不是只从 topk 里取），且换仓前已持有的会被 holdings_set 过滤。
3. **买入金额**：引擎用 ``portfolio_value * max_position_size`` 作为**上限**，
   再与 ``cash * (1 - cash_reserve)`` 取小；且允许对已有持仓加仓
   （``available = max_position_value - current_position_value``）。
4. **涨停掩码**：引擎的 tradability 矩阵带窗口前一日收盘价（prev_close），
   窗口首日的涨跌停也能判定 → 模拟器沿用同一份 trad 缓存。

## 保真度（2026-10-27 用 verify_10y_pit.py --lag 1 实测引擎对照）
    腿      引擎 SR     模拟器 SR    Δ
    42f     0.7992     0.7485      -0.051
    rsi     0.7716     0.7712      -0.0004
    fr      0.6202     0.6102      -0.010
    mf     -0.4523    -0.4656      -0.013

## 用法
    sc = prepare_score(score, mats, cfg)
    res = simulate(sc, mats, trad, cfg)
    res_is = simulate(sc, mats, trad, cfg, rows=is_rows)
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import ensemble_lab as L  # noqa: E402


class SimConfig(L.SimConfig):
    pass


def limit_up_mask_cached(mats: Dict[str, Any]) -> np.ndarray:
    """涨停掩码（缓存到 mats，避免每次 simulate 重算）。"""
    if "_lu_mask" not in mats:
        mats["_lu_mask"] = L.limit_up_mask(mats)
    return mats["_lu_mask"]


def prepare_score(
    score: np.ndarray,
    mats: Dict[str, Any],
    cfg: SimConfig,
) -> np.ndarray:
    """分数预处理（与策略类同序）：涨停掩码 → 调仓日掩码 → lag。

    必须在**完整时间轴**上做，然后才能切 IS/OOS 子区间
    （否则 OOS 第一天会因为缺前一天信号而无仓）。
    """
    master: pd.DatetimeIndex = mats["master"]
    sc = score.copy()
    sc[~np.isfinite(sc)] = np.nan
    sc[limit_up_mask_cached(mats)] = np.nan
    reb = L._rebalance_mask(master, cfg)
    sc[~reb] = np.nan
    if cfg.lag > 0:
        lagged = np.full_like(sc, np.nan)
        lagged[cfg.lag:] = sc[:-cfg.lag]
        sc = lagged
    return sc


def simulate(
    prepared_score: np.ndarray,
    mats: Dict[str, Any],
    trad: Dict[str, np.ndarray],
    cfg: SimConfig,
    selector: Optional[Any] = None,
    rows: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """复刻引擎 TopK-Dropout 执行。

    prepared_score : 已经过 prepare_score 的分数矩阵（完整时间轴）
    rows           : 只跑 master 的这些行（IS/OOS 切分）
    """
    master: pd.DatetimeIndex = mats["master"]
    close: np.ndarray = mats["close"]
    codes = mats["codes"]
    sel = selector or L.TopKEqualWeightSelector(cfg)

    sc = prepared_score
    if rows is not None:
        sc = sc[rows]
        master = master[rows]
        close = close[rows]
        trad = {k: v[rows] for k, v in trad.items()}
    n_d, n_s = close.shape

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
                qty[c] += q
                n_trades += 1
                return True

            keep, tgt_w, buy_pool = sel.select(order, t, holdings, len(holdings))
            keep_set = set(int(c) for c in keep)

            if len(holdings) < cfg.topk:
                buy = [int(c) for c in buy_pool
                       if int(c) not in holdings_set][: cfg.topk - len(holdings)]
                for c in buy:
                    _buy(c)
            else:
                # ---- 卖出：引擎语义 ----
                rank = {int(c): i for i, c in enumerate(order)}
                held_sorted = sorted(
                    [int(c) for c in holdings], key=lambda c: rank.get(int(c), 0))
                exit_rank = cfg.topk + cfg.hold_thresh + max(0, cfg.buffer)
                # 引擎：code 必须在 ranked（当天有分数）里才会被卖
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
                # ---- 买入：引擎语义（完整排名，剔除换仓前持仓）----
                buy = [int(c) for c in buy_pool
                       if int(c) not in holdings_set][:sold]
                for c in buy:
                    _buy(c)

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
