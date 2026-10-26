#!/usr/bin/env python3
"""按**任意窗口**重新抽各腿的分数（与引擎 prepare_backtest_data 完全同口径）。

## 为什么需要
引擎分段跑（IS/OOS 各起一次）时，``prepare_backtest_data`` 只看得到该段的
数据 → 60 日滚动因子在窗口开头冷启动、面板因子读到的行数不同、截面股票数
不同（5238 → 5202）→ **腿的分数矩阵与全窗口抽的不一样**。
用全窗口腿去模拟分段跑 → IS/OOS 结论会偏（实测引擎 IS SR 0.666 vs 模拟器 1.147）。

本脚本按给定窗口重抽腿，让模拟器与引擎分段跑同口径。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/ensemble_legs_window.py --start 2023-05-08 --end 2025-05-08 \
      --tag is --legs 42f mf
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import ensemble_legs as LEGS  # noqa: E402

OUT = "/Users/rone/.hermes/cache/scratch/stock_ens/cache/legs_window"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--legs", nargs="*", default=["42f", "mf"])
    args = ap.parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    os.makedirs(OUT, exist_ok=True)

    from app.services.backtest.execution.data_loader import DataLoader
    codes = L.universe_from_instruments()
    dl = DataLoader(ROOT + "/data", max_workers=8)
    stock_data = dl.load_multiple_stocks(codes, start, end)
    kept = [c for c in codes if c in stock_data]
    print(f"[load] {len(kept)}/{len(codes)}  {start.date()}→{end.date()}", flush=True)

    master = pd.DatetimeIndex(sorted(set().union(*[
        set(pd.DatetimeIndex(stock_data[c].index)) for c in kept])))
    pos_of = {d: i for i, d in enumerate(master)}
    print(f"[load] master {len(master)} 日", flush=True)

    for key in args.legs:
        path = os.path.join(OUT, f"{args.tag}_{key}.npz")
        if os.path.exists(path):
            print(f"[leg] {key} 命中缓存", flush=True)
            continue
        spec = LEGS.LEG_DEFS[key]
        cls = LEGS.import_cls(*spec["cls"])
        t0 = time.perf_counter()
        strat = cls(dict(spec["cfg"]))
        asyncio.run(strat.prepare_backtest_data(stock_data, start, end))
        out = np.full((len(master), len(kept)), np.nan)
        n = 0
        for j, c in enumerate(kept):
            sm = stock_data[c].attrs.get(spec["attr"]) or {}
            for d, v in sm.items():
                i = pos_of.get(pd.Timestamp(d))
                if i is not None and np.isfinite(v):
                    out[i, j] = v
                    n += 1
        np.savez_compressed(path, score=out, master=master.to_numpy(
            dtype="datetime64[ns]"),
            codes=np.array(kept, dtype="U16"))
        print(f"[leg] {key}: {n} 有效分数 ({time.perf_counter()-t0:.0f}s) → {path}",
              flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
