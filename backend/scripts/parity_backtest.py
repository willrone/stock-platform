"""产品口径复现回测：用**产品真实代码路径**跑一遍历史，看能不能复现研究端数字。

为什么必须做这件事：
  研究端（P3/P18）用的是「小数股 + 研究撮合器 + 研究成本口径」，
  产品端（`app/services/trading/multi_asset_service.py` + `scripts/paper_portfolio.py`）
  用的是「整手 100 股 + 产品撮合器 + 产品成本口径」。
  两者**必然有差**，而且差多少目前是未知的 —— 头条数字 1.4114 是研究端口径。

本脚本不重写逻辑，而是**直接调用产品的 `PP.run_multi_asset_line`**，
按周频（周五信号 → 下一交易日成交）跑完整个窗口，再按**日频**盯市算净值，
最后用**研究端同一套夏普口径**（日收益、pstdev、×√244）出数。

用法：
  python3 scripts/parity_backtest.py            # main 窗口
  python3 scripts/parity_backtest.py --oos      # OOS 窗口
输出：scripts/parity_backtest.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import daily_signal as ds  # noqa: E402
import paper_portfolio as PP  # noqa: E402
from app.services.trading.multi_asset_service import (  # noqa: E402
    ETF_PARQUET_FROZEN,
    load_etf_bars,
    missing_codes,
)

CAPITAL = 1_000_000.0

#: 研究端的窗口（multi_asset_leg_experiment.START/END、OOS_START/OOS_END）
WINDOWS = {
    "main": (pd.Timestamp("2023-05-08"), pd.Timestamp("2026-05-08")),
    "oos": (pd.Timestamp("2021-05-11"), pd.Timestamp("2023-05-08")),
}

#: 研究端夏普口径：日收益、总体标准差、×√244（**不是** 252）
SHARPE_ANN = 244.0


def sharpe_research(nav: List[Tuple[str, float]]) -> float:
    """与研究端 `multi_asset_leg_experiment.sharpe` **完全同口径**。"""
    if len(nav) < 3:
        return 0.0
    rets = []
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        if p0 and p0 > 0:
            rets.append(p1 / p0 - 1.0)
    if len(rets) < 3:
        return 0.0
    mu = statistics.fmean(rets)
    sd = statistics.pstdev(rets)
    return mu / sd * math.sqrt(SHARPE_ANN) if sd > 0 else 0.0


def maxdd(nav: List[Tuple[str, float]]) -> float:
    peak = None
    mdd = 0.0
    for _, v in nav:
        peak = v if peak is None else max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def annualized_ret(nav: List[Tuple[str, float]]) -> float:
    if len(nav) < 2 or not nav[0][1] or nav[0][1] <= 0:
        return 0.0
    years = len(nav) / 244.0
    if years <= 0:
        return 0.0
    return (nav[-1][1] / nav[0][1]) ** (1.0 / years) - 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", action="store_true", help="跑 OOS 窗口而不是 main")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    wname = "oos" if args.oos else "main"
    start, end = WINDOWS[wname]

    t0 = time.time()
    # ── 1) 数据 ────────────────────────────────────────────────────────
    codes = ds.list_universe()
    frames, panel, master = ds.load_data(codes)
    names = PP.load_name_map()
    extras = PP.load_market_extras(codes, master) if PP.BAOSTOCK_READY else None
    adj_close = PP.load_adjusted_close(codes, master) if PP.BAOSTOCK_READY else None
    print(f"[parity] 数据加载 {time.time()-t0:.0f}s", flush=True)

    # ── 2) to_mat 缓存（否则每周重建 380MB 矩阵 ×156 周）──────────────
    _mat_cache: Dict[str, pd.DataFrame] = {}
    _orig_to_mat = ds.to_mat

    def cached_to_mat(frames_, master_, col):
        if col not in _mat_cache:
            _mat_cache[col] = _orig_to_mat(frames_, master_, col)
        return _mat_cache[col]

    ds.to_mat = cached_to_mat

    # ── 3) 42f 打分矩阵只算一次（否则每次调仓重算 → 156 × 40s）────────
    comp_all = ds.compute_composite(frames, panel, master)
    print(f"[parity] 42f 矩阵 {time.time()-t0:.0f}s", flush=True)

    def score_cached(frames_, master_, panel_=None, explain_date=None):
        # 与 score_42f 等价：composite 值一致，只少了「为什么买它」的因子归因列
        return comp_all

    PP.score_42f = score_cached

    # ── 4) 袖套行情（冻结快照，与研究端同源）─────────────────────────
    sleeve_codes = PP._sleeve_codes()
    miss = missing_codes(sleeve_codes, base_dir=ETF_PARQUET_FROZEN)
    if miss:
        print(f"❌ 袖套 ETF 缺数据 {miss}")
        return 1
    etf_bars = load_etf_bars(sleeve_codes, base_dir=ETF_PARQUET_FROZEN)

    # 袖套行情统一走**冻结快照**（与研究端同源；实盘目录会被周任务重拉，
    # 前复权重锚会让产品口径的数字和历史研究对不上）。
    _orig_load_etf_bars = PP.load_etf_bars

    _bars_cache: Dict[str, Any] = {}

    def _load_etf_bars_frozen(codes, **_kw):
        # 156 周 × 9 个 parquet 每次重读太慢 → 缓存
        if not _bars_cache:
            _bars_cache.update({"bars": _orig_load_etf_bars(
                codes, base_dir=ETF_PARQUET_FROZEN)})
        return _bars_cache["bars"]

    def _missing_codes_frozen(codes, **_kw):
        return [c for c in codes if not os.path.exists(
            os.path.join(ETF_PARQUET_FROZEN, f"{c}.parquet"))]

    PP.load_etf_bars = _load_etf_bars_frozen
    PP.missing_codes = _missing_codes_frozen

    # ── 5) 逐周调仓（产品真实代码路径）────────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="parity_")
    PP.PAPER_DIR = tmpdir

    in_win = master[(master >= start) & (master <= end)]
    fridays = [d for d in in_win if d.weekday() == 4]  # 周五信号
    # 第一个信号日之前要有一个交易日做建仓
    nav: List[Tuple[str, float]] = []
    detail: List[Dict[str, Any]] = []
    close_mat = _mat_cache.get("close")
    if close_mat is None:
        close_mat = _orig_to_mat(frames, master, "close")
        _mat_cache["close"] = close_mat

    def portfolio_value(positions: Dict[str, Any], cash: float, day: pd.Timestamp) -> float:
        v = cash
        for code, pos in positions.items():
            if pos.get("leg") == "sleeve":
                bar = etf_bars.get(code, day, "close")
                px = bar["price"] if bar else float(pos.get("cost") or 0)
            else:
                px = float(close_mat.at[day, code]) if code in close_mat.columns else np.nan
                if not np.isfinite(px):
                    px = float(pos.get("cost") or 0)
            v += float(pos.get("shares", 0)) * px
        return v

    last_day: Optional[pd.Timestamp] = None
    pending: Optional[Dict[str, Any]] = None
    for i, sig_day in enumerate(fridays):
        after = master[master > sig_day]
        if len(after) == 0:
            break
        trade_day = after[0]
        if trade_day > end:
            break
        res = PP.run_multi_asset_line(
            frames, panel, master, names, CAPITAL, sig_day,
            extras=extras, adj_close=adj_close,
        )
        # 从上一个调仓日到本次成交日之间，按上一份持仓盯市
        if pending is not None:
            prev_pos, prev_cash = pending["positions"], pending["cash"]
            for d in master[(master > last_day) & (master < trade_day)]:
                nav.append((str(d.date()), portfolio_value(prev_pos, prev_cash, d)))
        h = PP.load_holdings("multi_asset_final", CAPITAL)
        pending = {"positions": h.get("positions") or {}, "cash": float(h.get("cash") or 0)}
        nav.append((str(trade_day.date()),
                    portfolio_value(pending["positions"], pending["cash"], trade_day)))
        last_day = trade_day
        legs = res.get("legs") or {}
        detail.append({
            "signal_date": str(sig_day.date()),
            "trade_date": str(trade_day.date()),
            "value": round(res["value"], 2),
            "cash": round(res["cash"], 2),
            "n_positions": res["n_positions"],
            "a_invested": (legs.get("a") or {}).get("invested_ratio"),
            "s_invested": (legs.get("sleeve") or {}).get("invested_ratio"),
            "s_class_w": (legs.get("sleeve") or {}).get("class_weights_now"),
        })
        if (i + 1) % 10 == 0:
            print(f"[parity] {i+1}/{len(fridays)} 周  {time.time()-t0:.0f}s  "
                  f"净值 {nav[-1][1]:,.0f}", flush=True)

    if pending is not None:
        for d in master[(master > last_day) & (master <= end)]:
            nav.append((str(d.date()),
                        portfolio_value(pending["positions"], pending["cash"], d)))

    # ── 6) 出数 ────────────────────────────────────────────────────────
    sh = sharpe_research(nav)
    dd = maxdd(nav)
    ar = annualized_ret(nav)
    rets = []
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        if p0 and p0 > 0:
            rets.append(p1 / p0 - 1.0)
    vol = statistics.pstdev(rets) * math.sqrt(SHARPE_ANN) if len(rets) > 2 else 0.0

    out = {
        "window": wname,
        "start": str(start.date()),
        "end": str(end.date()),
        "n_trading_days": len(nav),
        "capital": CAPITAL,
        "sharpe": round(sh, 4),
        "maxdd": round(dd, 4),
        "ann_return": round(ar, 4),
        "ann_vol": round(vol, 4),
        "final_value": round(nav[-1][1], 2) if nav else 0.0,
        "research_sharpe": {"main": 1.4114, "oos": 1.0691}.get(wname),
        "note": (
            "产品口径（整手 100 股 + 产品撮合器 + 产品成本 + A腿 50 万预算）"
            "跑历史得到的数字。research_sharpe 是研究端口径（研究端夏普）。"
        ),
        "rebalances": detail,
    }
    outp = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f"parity_backtest_{wname}.json")
    with open(outp, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print()
    print(f"[parity] 窗口 {wname} {out['start']} → {out['end']}  "
          f"交易日 {len(nav)}")
    print(f"[parity] 产品口径夏普 {sh:.4f}  年化 {ar:.2%}  波动 {vol:.2%}  "
          f"回撤 {dd:.2%}  末值 {nav[-1][1]:,.0f}")
    print(f"[parity] 研究口径夏普 {out['research_sharpe']}  "
          f"→ 差 {sh - out['research_sharpe']:+.4f}")
    print(f"[parity] 产物 → {outp}")
    print(f"[parity] 总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
