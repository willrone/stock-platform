#!/usr/bin/env python3
"""财务因子的 RankIC 验证（2026-10-27）

## 目的
检验「财务因子是否真的有 alpha」—— 这是今晚所有价量因子改造全部证伪后，
唯一剩下的「引入新信息」路径。

对照基准：现有 22 个价量因子的 RankIC(fwd5) 全部显著，|t| 4.13~18.54，
最强 CORR10 = −0.0673。**财务因子只要 IC 同量级且与价量因子低相关，
就值得全量接入。**

## PIT 正确性
财务数据按 **`pubDate`（公布日）** 对齐：因子值从公布当天才可用。
用 statDate（报告期）会引入未来函数 —— 2024Q4 财报在 2025-03 才公布。

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_fundamental_ic.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.path.join(ROOT, "data", "backtest", "fundamental_ic.json")
FUND_FILE = os.path.join(ROOT, "data", "fundamentals", "fundamentals_pit.json")

FWD = 5


def main() -> int:
    if not os.path.exists(FUND_FILE):
        print(f"[fic] 缺少 {FUND_FILE}，先跑 fetch_fundamentals.py")
        return 1
    with open(FUND_FILE) as fh:
        raw = json.load(fh)
    print(f"[fic] 财务数据：{len(raw)} 只股票", flush=True)
    if not raw:
        return 1

    # ── 加载价格（同一股票池，用于前瞻收益）────────────────────────
    from app.services.backtest.execution.data_loader import DataLoader
    dl = DataLoader(data_dir=os.path.join(ROOT, "data"))
    codes = sorted(raw.keys())
    s, e = datetime(2019, 1, 1), datetime(2026, 5, 8)
    close = {}
    for i, c in enumerate(codes):
        try:
            d = dl.load_stock_data(c, s, e)
        except Exception:
            continue
        if d is None or d.empty:
            continue
        col = "$close" if "$close" in d.columns else "close"
        close[c] = pd.to_numeric(d[col], errors="coerce")
        if (i + 1) % 50 == 0:
            print(f"  价格 {i+1}/{len(codes)}", flush=True)
    if not close:
        print("[fic] 没有价格数据")
        return 1
    idx = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for v in close.values()])))
    close_df = pd.DataFrame(close).reindex(idx)
    fwd = (close_df.shift(-FWD) / close_df - 1.0)
    print(f"[fic] 价格矩阵 {close_df.shape}  交易日 {len(idx)}", flush=True)

    # ── 构建 PIT 因子面板（阶梯函数：公布日才生效）──────────────────
    factor_names = sorted({f for recs in raw.values() for _, _, v in recs for f in v})
    print(f"[fic] 因子：{factor_names}", flush=True)
    panels = {f: pd.DataFrame(np.nan, index=idx, columns=list(close_df.columns))
              for f in factor_names}

    for code, recs in raw.items():
        if code not in close_df.columns:
            continue
        for pub, _stat, vals in recs:
            ts = pd.Timestamp(pub)
            # 归到该日期所在（或之前）的交易日；从这一天起生效
            pos = idx.searchsorted(ts, side="left")
            if pos >= len(idx):
                continue
            start_ts = idx[pos]
            for f, v in vals.items():
                panels[f].loc[start_ts:, code] = v

    # ── 逐因子算逐日 RankIC ─────────────────────────────────────
    rows = []
    for f, panel in panels.items():
        ics = []
        for i in range(len(idx) - FWD):
            x = panel.iloc[i]
            y = fwd.iloc[i]
            ok = x.notna() & y.notna()
            if ok.sum() < 20:
                continue
            xv, yv = x[ok], y[ok]
            if xv.nunique() < 3 or yv.nunique() < 3:
                continue
            r = xv.rank().corr(yv.rank())
            if pd.notna(r):
                ics.append(float(r))
        if len(ics) < 60:
            print(f"  {f:<22} 样本不足（{len(ics)} 天）", flush=True)
            continue
        arr = np.array(ics)
        icir = arr.mean() / (arr.std() + 1e-12)
        rows.append({
            "factor": f,
            "mean_ic": round(float(arr.mean()), 5),
            "std_ic": round(float(arr.std()), 5),
            "icir": round(float(icir), 4),
            "t_stat": round(float(icir * np.sqrt(len(arr))), 2),
            "ic_positive_ratio": round(float((arr > 0).mean()), 4),
            "n_days": len(arr),
        })
    rows.sort(key=lambda r: -abs(r["t_stat"]))
    print(f"\n{'因子':<22}{'meanIC':>9}{'ICIR':>8}{'t':>7}{'IC>0':>8}")
    print("-" * 56)
    for r in rows:
        print(f"{r['factor']:<22}{r['mean_ic']:>9.4f}{r['icir']:>8.3f}"
              f"{r['t_stat']:>7.2f}{r['ic_positive_ratio']*100:>7.0f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "window": f"{idx[0].date()}→{idx[-1].date()}",
            "n_stocks": int(close_df.shape[1]),
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "财务因子 RankIC(fwd5)；按 pubDate 对齐（PIT）",
            },
            "factors": rows,
        }, fh, ensure_ascii=False, indent=1)
    print(f"\n[fic] 已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
