#!/usr/bin/env python3
"""周频交易信号 + 纸面账本 (42f topk30/ndrop3, 周一调仓用上周五分数).

用途: 每周五收盘后跑一遍, 输出下周一要买/卖/持的股票、股数、金额、预估成本,
并把成交记进纸面账本。默认用 100 万纸面资金, 不碰真钱。

与回测的对应关系:
  weekly_lag1_livecost = 周五收盘分 → 周一收盘执行。本脚本默认按**周五收盘价**
  记纸面成交 (比回测乐观一个隔夜跳空), 用于先跑通流程; 后续可加 --fill 用
  周一实际收盘价重记。

输出:
  data/signals/signal_<date>.json   当日信号快照
  data/signals/ledger.csv           逐笔纸面成交账本
  data/signals/paper_holdings.json  纸面组合 (现金 + 持仓)

用法:
  env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/weekly_signal.py
  ... --topk 30 --ndrop 3 --capital 1000000 --force   # 非周五强制跑
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

BACKEND_ROOT = "/Users/rone/Projects/stock-platform/backend"
ROOT = os.path.dirname(BACKEND_ROOT)
sys.path.insert(0, BACKEND_ROOT)

_spec = importlib.util.spec_from_file_location(
    "ds", os.path.join(BACKEND_ROOT, "scripts", "daily_signal.py"))
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)

SIGNAL_DIR = os.path.join(ROOT, "data", "signals")
LEDGER = os.path.join(SIGNAL_DIR, "ledger.csv")
PAPER = os.path.join(SIGNAL_DIR, "paper_holdings.json")
RAW_DIR = os.path.join(ROOT, "data", "tdx_daily", "raw")

COMMISSION = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.0005       # 卖出
SLIPPAGE = 0.001         # 单边
CASH_RESERVE = 0.05
BOARD_LOT = 100

# 22f 固定版 (2026-10-19 rolling walk-forward 验证):
#   MIN* +1, STD/CORR/CORD* -1, MAX30/MAX60 +1; 剔除 MAX5/10/20.
#   2021-2026 OOS: 9.4% 年化 / -27.1% 回撤 / 0.685 夏普.
F22 = ["MIN5", "STD5", "CORR5", "CORD5", "MIN10", "STD10", "CORR10", "CORD10",
       "MIN20", "STD20", "CORR20", "CORD20", "MIN30", "MAX30", "STD30", "CORR30", "CORD30",
       "MIN60", "MAX60", "STD60", "CORR60", "CORD60"]
SIGNS_22F = {f: (1.0 if f.startswith("MIN") or f in ("MAX30", "MAX60") else -1.0)
             for f in F22}


def compute_composite_22f(frames, master):
    """22f 基础版复合: 与 rolling walk-forward 的 selected 因子/符号完全一致。"""
    close = ds.to_mat(frames, master, "close")
    vol = ds.to_mat(frames, master, "volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close.pct_change(fill_method=None)
        vchg = vol.pct_change(fill_method=None)
        cands = {}
        for n in (5, 10, 20, 30, 60):
            cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
            cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
            cands[f"STD{n}"] = close.rolling(n).std() / close
            cands[f"CORR{n}"] = close.rolling(n).corr(vol)
            cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)
    acc = np.zeros((len(master), len(frames)))
    wsum = np.zeros_like(acc)
    for f in F22:
        ds.zscore_add(acc, wsum, cands[f].to_numpy(dtype=np.float64),
                      SIGNS_22F[f], ds.MIN_CROSS)
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    lu = close.pct_change(fill_method=None).to_numpy() >= ds.LIMIT_UP
    comp[lu] = np.nan
    return pd.DataFrame(comp, index=master, columns=list(frames))


def limit_pct(code: str) -> float:
    c2 = code[:2]
    if c2 == "30":
        return 0.20
    if c2 == "68":
        return 0.20
    if c2 in ("92", "83", "87"):
        return 0.30
    return 0.10


def load_name_map() -> dict:
    if not os.path.isdir(RAW_DIR):
        return {}
    dates = sorted(d for d in os.listdir(RAW_DIR) if d.startswith("date="))
    if not dates:
        return {}
    try:
        df = pd.read_parquet(os.path.join(RAW_DIR, dates[-1], "part.parquet"),
                             columns=["ts_code", "name"])
        return dict(zip(df["ts_code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def load_paper(capital: float) -> dict:
    if os.path.exists(PAPER):
        with open(PAPER) as fh:
            return json.load(fh)
    return {"cash": float(capital), "positions": {}, "initial_capital": float(capital)}


def save_paper(p: dict) -> None:
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    with open(PAPER, "w") as fh:
        json.dump(p, fh, ensure_ascii=False, indent=1)


def append_ledger(rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    exists = os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="周频信号 + 纸面账本")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--ndrop", type=int, default=3)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--force", action="store_true", help="非周五也跑")
    args = ap.parse_args()

    codes = ds.list_universe()
    frames, panel, master = ds.load_data(codes)
    comp = compute_composite_22f(frames, master)
    latest = comp.index[-1]

    if latest.dayofweek != 4 and not args.force:
        print(f"今天 {latest.date()} 不是周五 (dayofweek={latest.dayofweek})，不调仓。"
              f"加 --force 可强制跑。", flush=True)
        return

    names = load_name_map()
    paper = load_paper(args.capital)
    positions = paper.get("positions", {})
    cash = float(paper.get("cash", 0.0))

    close = ds.to_mat(frames, master, "close").loc[latest]
    vol = ds.to_mat(frames, master, "volume").loc[latest]
    day_ret = close / ds.to_mat(frames, master, "close").shift(1).loc[latest] - 1.0

    rank = comp.loc[latest].dropna().sort_values(ascending=False)
    # 实盘安全过滤: 剔除 ST (5% 涨跌停 + 退市风险; 回测的 9.5% 涨停掩码对 ST 失效)
    st_codes = {c for c in rank.index if "ST" in names.get(c, "").upper()}
    if st_codes:
        rank = rank.drop(index=[c for c in st_codes if c in rank.index])
    rank_index = {c: i for i, c in enumerate(rank.index)}
    top = list(rank.index[:args.topk])

    held = list(positions.keys())
    total_value = cash + sum(float(positions[c].get("shares", 0)) * float(close.get(c, 0) or 0)
                             for c in held)
    target_weight = (1.0 - CASH_RESERVE) / args.topk

    def px(c: str) -> float:
        v = close.get(c, np.nan)
        return float(v) if np.isfinite(v) else float("nan")

    # ── 卖出: 持仓中掉出 topk 的, 最差优先, 最多 ndrop ──
    fell_out = [c for c in held if c not in set(top)]
    fell_out.sort(key=lambda c: rank_index.get(c, len(rank) + 1), reverse=True)
    sells = []
    for c in fell_out[:args.ndrop]:
        p = px(c)
        if not np.isfinite(p) or p <= 0:
            continue
        if float(vol.get(c, 0) or 0) <= 0:
            continue  # 停牌
        if float(day_ret.get(c, 0) or 0) <= -limit_pct(c) + 1e-4:
            continue  # 跌停不可卖
        shares = int(positions[c].get("shares", 0))
        if shares <= 0:
            continue
        gross = shares * p
        commission = max(MIN_COMMISSION, gross * COMMISSION)
        stamp = gross * STAMP_TAX
        slip = gross * SLIPPAGE
        net = gross - commission - stamp - slip
        cash += net
        sells.append({"code": c, "name": names.get(c, ""), "price": p, "shares": shares,
                      "amount": round(gross, 2), "commission": round(commission, 2),
                      "stamp_tax": round(stamp, 2), "slippage": round(slip, 2),
                      "net": round(net, 2), "score": round(float(rank.get(c, np.nan)), 4)
                      if c in rank else ""})
        del positions[c]

    # ── 买入: 新进 topk 的, 最强优先 ──
    # 首次建仓 / 持仓不足 topk 时补满; 否则最多 ndrop
    need = args.topk - len(positions)
    max_buys = max(need, args.ndrop) if len(positions) < args.topk else args.ndrop
    buy_candidates = [c for c in top if c not in positions][:max_buys]
    buys = []
    for c in buy_candidates:
        p = px(c)
        if not np.isfinite(p) or p <= 0:
            continue
        if float(vol.get(c, 0) or 0) <= 0:
            continue  # 停牌
        # 涨停已被打分掩码剔除, 这里再兜底
        if float(day_ret.get(c, 0) or 0) >= limit_pct(c) - 1e-4:
            continue
        target_amount = total_value * target_weight
        # 含成本的可买金额
        unit_cost = p * (1 + SLIPPAGE + COMMISSION)
        shares = int(target_amount / unit_cost // BOARD_LOT) * BOARD_LOT
        if shares <= 0:
            continue
        # 现金约束
        max_shares = int(cash / unit_cost // BOARD_LOT) * BOARD_LOT
        shares = min(shares, max_shares)
        if shares <= 0:
            continue
        gross = shares * p
        commission = max(MIN_COMMISSION, gross * COMMISSION)
        slip = gross * SLIPPAGE
        total_cost = gross + commission + slip
        if total_cost > cash:
            continue
        cash -= total_cost
        positions[c] = {"shares": shares, "cost": round(p, 4),
                        "buy_date": str(latest.date())}
        buys.append({"code": c, "name": names.get(c, ""), "price": p, "shares": shares,
                     "amount": round(gross, 2), "commission": round(commission, 2),
                     "slippage": round(slip, 2), "total_cost": round(total_cost, 2),
                     "score": round(float(rank[c]), 4)})

    # ── 落盘 ──
    end_value = cash + sum(float(positions[c]["shares"]) * float(close.get(c, 0) or 0)
                           for c in positions)
    paper.update({"cash": round(cash, 2), "positions": positions,
                  "last_signal": str(latest.date()),
                  "last_value": round(end_value, 2),
                  "fill_assumption": "Friday close"})
    save_paper(paper)

    rows = []
    for s in sells:
        rows.append({"date": str(latest.date()), "action": "SELL", "code": s["code"],
                     "name": s["name"], "price": s["price"], "shares": s["shares"],
                     "amount": s["amount"], "commission": s["commission"],
                     "stamp_tax": s["stamp_tax"], "slippage": s["slippage"],
                     "score": s["score"], "cash_after": round(cash, 2),
                     "portfolio_value": round(end_value, 2)})
    for b in buys:
        rows.append({"date": str(latest.date()), "action": "BUY", "code": b["code"],
                     "name": b["name"], "price": b["price"], "shares": b["shares"],
                     "amount": b["amount"], "commission": b["commission"],
                     "stamp_tax": 0.0, "slippage": b["slippage"],
                     "score": b["score"], "cash_after": round(cash, 2),
                     "portfolio_value": round(end_value, 2)})
    append_ledger(rows)

    snapshot = {"meta": {"date": str(latest.date()), "topk": args.topk,
                         "n_drop": args.ndrop, "cash_reserve": CASH_RESERVE,
                         "fill_assumption": "Friday close",
                         "generated_at": datetime.now().isoformat()},
                "portfolio": {"cash": round(cash, 2), "value": round(end_value, 2),
                              "n_positions": len(positions)},
                "sells": sells, "buys": buys,
                "holds": [c for c in top if c in positions and c not in {s["code"] for s in sells}]}
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    snap_path = os.path.join(SIGNAL_DIR, f"signal_{latest.date()}.json")
    with open(snap_path, "w") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=1)

    # 同步给前端 /paper-signal 页面（Next static public JSON）
    try:
        frontend_public = os.path.join(ROOT, "frontend", "public")
        os.makedirs(frontend_public, exist_ok=True)
        with open(os.path.join(frontend_public, "paper-signal.json"), "w") as fh:
            json.dump(snapshot, fh, ensure_ascii=False, indent=1)
        print(f"[frontend] {os.path.join(frontend_public, 'paper-signal.json')}", flush=True)
    except Exception as exc:
        print(f"[frontend] 同步失败: {exc}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"周频信号 {latest.date()} (周五) → 下周一执行", flush=True)
    print(f"组合总值 ¥{end_value:,.0f} | 现金 ¥{cash:,.0f} | "
          f"{len(positions)} 只 × {target_weight*100:.2f}%", flush=True)
    print(f"\n◆ 卖出 {len(sells)} 只", flush=True)
    for s in sells:
        print(f"  {s['code']} {s['name']}  {s['shares']}股 @ {s['price']:.2f}  "
              f"净得 ¥{s['net']:,.0f}", flush=True)
    print(f"\n◆ 买入 {len(buys)} 只", flush=True)
    for b in buys:
        print(f"  {b['code']} {b['name']}  {b['shares']}股 @ {b['price']:.2f}  "
              f"成本 ¥{b['total_cost']:,.0f}  score={b['score']:+.3f}", flush=True)
    print(f"\n◆ 持有 {len(snapshot['holds'])} 只", flush=True)
    print(f"[saved] {snap_path}", flush=True)


if __name__ == "__main__":
    main()
