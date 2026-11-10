#!/usr/bin/env python3
"""纸面盘 ↔ 回测引擎 可成交性对账（2026-10-26）

## 为什么
子代理指出**回测与纸面盘是两条独立的可成交性实现**：
- 纸面盘 `paper_portfolio.py` → `trading/execution_rules.check_tradability()`
  （有 name / trade_status / amount 等 baostock 附加字段）
- 回测引擎 `backtest/execution/tradability.py` → 自己算矩阵
  （只有 qlib OHLCV，无 name / trade_status / amount）

两边**同一个信号在同一天可能给出不同结论** → 「回测说能买、纸面盘买不到」
→ 纸面业绩永远达不到回测预期，用户无法判断信号能不能兑现。

本脚本对**同一天、同一只票**逐票对比两边的 can_buy / can_sell 判定，
把分歧定位到具体票、具体原因。

## 判定哪些
1. 停牌：volume == 0
2. 涨停 / 跌停：按板块动态阈值（主板 10% / 创业板科创板 20% / 北交所 30% / ST 5%）
3. 一字板：high == low 且涨跌停
4. 成交额门槛（5% 参与率 / 500 万门槛）
5. ST / 退市整理期

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/reconcile_tradability.py --date 2026-10-23
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from datetime import datetime

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="对账日期 YYYY-MM-DD（默认取 qlib 特征的数据终点）")
    ap.add_argument("--limit", type=int, default=1200, help="抽查股票数")
    ap.add_argument("--scan-days", type=int, default=400, help="回溯多少个自然日做多日对账")
    ap.add_argument("--out", default="data/backtest/reconcile_tradability.json")
    args = ap.parse_args()

    # ── qlib 特征的数据终点（回测侧的数据源）────────────────────
    # 重要发现：回测引擎读 `data/qlib_data/features/day/`，该目录的数据
    # **只到 2026-05-08**；纸面盘读 baostock/TDX（到 2026-10-23）。
    # 这本身是一处「数据源分叉」—— 对账必须落在 qlib 范围内才有意义。
    feat_dir = os.path.join(ROOT, "data", "qlib_data", "features", "day")
    qlib_end = None
    for f in sorted(os.listdir(feat_dir))[:1]:
        try:
            df = pd.read_parquet(os.path.join(feat_dir, f), columns=["date"])
            df["date"] = pd.to_datetime(df["date"])
            qlib_end = df["date"].max()
        except Exception:
            pass
    if args.date:
        day = pd.Timestamp(args.date)
    elif qlib_end is not None:
        day = pd.Timestamp(qlib_end)
    else:
        day = pd.Timestamp("2026-05-08")
    print(f"[reconcile] 对账日期 {day.date()}"
          + (f"（qlib 特征数据终点 {qlib_end.date()}）" if qlib_end is not None else ""),
          flush=True)

    # ── 1. 回测侧：qlib 特征 → tradability 矩阵 ────────────────────
    from app.services.backtest.execution.data_loader import DataLoader
    from app.services.backtest.execution.tradability import build_tradability

    dl = DataLoader(data_dir=os.path.join(ROOT, "data"))
    codes = []
    with open(os.path.join(ROOT, "data", "qlib_official_data", "instruments", "all.txt")) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if parts and "_" in parts[0]:
                c, _, m = parts[0].partition("_")
                if m.upper() in ("SH", "SZ", "BJ") and c.isdigit():
                    codes.append(f"{c}.{m.upper()}")
    codes = sorted(codes)[: args.limit]

    # 扫一段窗口：多日对账才能覆盖到「有涨停/停牌」的日子
    # （单日 0 分歧可能只是那天没���么可拦，说服力不足）
    # 多留 20 天前置历史：让窗口首日也能算出涨跌幅
    # （否则首日 ret=NaN → tradability 跳过 → 误报分歧）
    start = day - pd.Timedelta(days=args.scan_days) - pd.Timedelta(days=20)
    s = start.to_pydatetime()
    e = (day + pd.Timedelta(days=5)).to_pydatetime()
    sd = {}
    for c in codes:
        try:
            d = dl.load_stock_data(c, s, e)
        except Exception:
            continue
        if d is not None and not d.empty:
            sd[c] = d
    print(f"[reconcile] 回测侧加载 {len(sd)} 只", flush=True)

    # 交易日（用股票池的并集，避免某天没数据）
    all_idx = pd.DatetimeIndex([])
    for d in list(sd.values())[:200]:
        all_idx = all_idx.union(d.index)
    all_idx = all_idx.sort_values()
    # 对账窗口 = [day - scan_days, day]；**前置 20 天只用于算首日涨跌幅**，
    # 不能进 dates（否则首日 ret 仍是 NaN → 误报分歧）
    window_start = day - pd.Timedelta(days=args.scan_days)
    dates = [x for x in all_idx if window_start <= x <= day]
    # 窗口首日的 ret 需用窗口前一日的 close 补（与引擎侧同一套处理）
    _prev = {}
    for _c, _d in sd.items():
        _pre = _d[_d.index < dates[0]]
        if len(_pre):
            _cc = [x for x in ("$close", "close") if x in _pre.columns]
            if _cc:
                _v = pd.to_numeric(_pre[_cc[0]], errors="coerce").dropna()
                if len(_v):
                    _prev[_c] = float(_v.iloc[-1])
    m = build_tradability(sd, dates, list(sd.keys()), prev_close=_prev)
    print(f"[reconcile] 对账窗口 {dates[0].date()} → {dates[-1].date()}"
          f"（{len(dates)} 个交易日）", flush=True)

    # ── 2. 纸面盘侧：baostock 附加字段 → check_tradability ───────
    from app.services.trading.execution_rules import check_tradability, load_st_names

    bd = os.path.join(ROOT, "data", "parquet", "baostock_daily")
    st_names = load_st_names()

    def load_ba(code: str):
        p = os.path.join(bd, code.replace(".", "_") + ".parquet")
        if not os.path.exists(p):
            return None
        try:
            df = pd.read_parquet(p)
        except Exception:
            return None
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        row = df[df["date"] == day]
        if row.empty:
            return None
        return row.iloc[0]

    n_bt = n_paper = 0
    diff_buy = []
    diff_sell = []
    paper_buy = {}
    paper_sell = {}

    # 纸面盘侧：把 baostock 日线整段读进来（一次读一只，缓存）
    ba_cache: dict = {}

    def ba_rows(code: str):
        if code in ba_cache:
            return ba_cache[code]
        bd_dir = os.path.join(ROOT, "data", "parquet", "baostock_daily")
        p = os.path.join(bd_dir, code.replace(".", "_") + ".parquet")
        df = None
        if os.path.exists(p):
            try:
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            except Exception:
                df = None
        ba_cache[code] = df
        return df

    day_stats = []
    for r_idx, dday in enumerate(dates):
        n_day = 0
        n_block_bt = 0
        n_block_paper = 0
        for code, d in sd.items():
            if dday not in d.index:
                continue
            c_idx = m.codes.index(code)
            bt_buy = bool(m.can_buy[r_idx, c_idx])
            bt_sell = bool(m.can_sell[r_idx, c_idx])
            bt_reason = ""
            if not bt_buy and m.reason_buy is not None:
                bt_reason = str(m.reason_buy[r_idx, c_idx])
            elif not bt_sell and m.reason_sell is not None:
                bt_reason = str(m.reason_sell[r_idx, c_idx])
            if not bt_buy:
                n_block_bt += 1

            bdf = ba_rows(code)
            if bdf is None or dday not in bdf.index:
                continue
            row = bdf.loc[dday]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            n_day += 1
            if not bt_buy:
                n_block_paper += 1

            close = _f(row.get("close"))
            prev = None
            try:
                pos = bdf.index.get_indexer([dday], method="ffill")[0]
                if pos > 0:
                    prev = _f(bdf["close"].iloc[pos - 1])
            except Exception:
                prev = None
            pct = (close / prev - 1.0) if (close and prev and prev > 0) else None
            name = st_names.get(code, {}).get("name", "")
            t = check_tradability(
                code,
                name=name,
                close=close,
                open_price=_f(row.get("open")),
                high=_f(row.get("high")),
                low=_f(row.get("low")),
                pct_chg=pct,
                volume=_f(row.get("volume")),
                amount=_f(row.get("amount")),
                trade_status=(int(row["trade_status"])
                             if "trade_status" in row and row["trade_status"] is not None
                             else None),
                is_st=1 if st_names.get(code, {}).get("is_st") else None,
                pct_reliable=True,
            )
            n_bt += 1
            n_paper += 1

            if t.can_buy != bt_buy:
                diff_buy.append({
                    "date": str(dday.date()), "code": code, "name": name,
                    "backtest_can_buy": bt_buy, "paper_can_buy": t.can_buy,
                    "backtest_reason": bt_reason, "paper_reason": t.reason,
                    "pct": None if pct is None else round(pct, 4),
                })
            if t.can_sell != bt_sell:
                diff_sell.append({
                    "date": str(dday.date()), "code": code, "name": name,
                    "backtest_can_sell": bt_sell, "paper_can_sell": t.can_sell,
                    "backtest_reason": bt_reason, "paper_reason": t.reason,
                    "pct": None if pct is None else round(pct, 4),
                })
        day_stats.append({
            "date": str(dday.date()), "compared": n_day,
            "backtest_blocked_buy": n_block_bt, "paper_blocked_buy": n_block_paper,
        })

    print(f"\n[reconcile] 对比 {n_paper} 只（两边都有数据）")
    print(f"[reconcile] 买入判定分歧 {len(diff_buy)} 只 "
          f"({len(diff_buy)/max(n_paper,1)*100:.2f}%)")
    print(f"[reconcile] 卖出判定分歧 {len(diff_sell)} 只 "
          f"({len(diff_sell)/max(n_paper,1)*100:.2f}%)")

    # 分歧原因归类
    def classify(diffs, key):
        cnt = collections.Counter()
        for d in diffs:
            r = d.get(key) or ""
            if "停牌" in r:
                cnt["停牌"] += 1
            elif "涨停" in r or "一字" in r:
                cnt["涨停/跌停"] += 1
            elif "成交额" in r or "门槛" in r:
                cnt["流动性"] += 1
            elif r:
                cnt[r[:14]] += 1
            else:
                cnt["（无原因）"] += 1
        return cnt.most_common(8)

    if diff_buy:
        print("\n[reconcile] 买入分歧原因（回测侧）：")
        for k, v in classify(diff_buy, "backtest_reason"):
            print(f"    {k}: {v}")
        print("  样例：")
        for d in diff_buy[:6]:
            print(f"    {d['code']} {d['name']} 回测可买={d['backtest_can_buy']} "
                  f"({d['backtest_reason']}) | 纸面可买={d['paper_can_buy']} "
                  f"({d['paper_reason']}) pct={d['pct']}")
    if diff_sell:
        print("\n[reconcile] 卖出分歧原因（回测侧）：")
        for k, v in classify(diff_sell, "backtest_reason"):
            print(f"    {k}: {v}")
        print("  样例：")
        for d in diff_sell[:6]:
            print(f"    {d['code']} {d['name']} 回测可卖={d['backtest_can_sell']} "
                  f"({d['backtest_reason']}) | 纸面可卖={d['paper_can_sell']} "
                  f"({d['paper_reason']}) pct={d['pct']}")

    # 有拦截的日子（证明对账覆盖到了真正的分歧场景）
    _blocked_days = [x for x in day_stats if x["backtest_blocked_buy"] > 0]
    _heavy = sorted(_blocked_days, key=lambda x: -x["backtest_blocked_buy"])[:5]
    print(f"\n[reconcile] 有拦截（回测侧不可买）的日子: "
          f"{len(_blocked_days)}/{len(day_stats)} 天")
    for x in _heavy:
        print(f"    {x['date']}: 回测不可买 {x['backtest_blocked_buy']} / "
              f"纸面不可买 {x['paper_blocked_buy']}（对比 {x['compared']} 只）")

    out = {
        "date": str(day.date()),
        "window": f"{dates[0].date()}→{dates[-1].date()}",
        "days": len(day_stats),
        "days_with_block": len(_blocked_days),
        "compared": n_paper,
        "buy_diff": len(diff_buy),
        "sell_diff": len(diff_sell),
        "buy_diff_pct": round(len(diff_buy) / max(n_paper, 1) * 100, 3),
        "sell_diff_pct": round(len(diff_sell) / max(n_paper, 1) * 100, 3),
        "day_stats": day_stats,
        "buy_diff_samples": diff_buy[:60],
        "sell_diff_samples": diff_sell[:60],
    }
    op = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as fh:
        import json
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[reconcile] 已写入 {op}")

    ok = len(diff_buy) == 0 and len(diff_sell) == 0
    print(f"[reconcile] {'✅ 两边完全一致' if ok else '⚠️ 存在分歧（见上）'}")
    return 0


def _f(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def load_ba_prev(code: str, day: pd.Timestamp):
    """取上一交易日收盘价（baostock 口径）。"""
    bd = os.path.join(ROOT, "data", "parquet", "baostock_daily")
    p = os.path.join(bd, code.replace(".", "_") + ".parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    prev = df[df["date"] < day].sort_values("date")
    if prev.empty:
        return None
    return float(prev["close"].iloc[-1])


if __name__ == "__main__":
    raise SystemExit(main())
