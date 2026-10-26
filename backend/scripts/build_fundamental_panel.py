#!/usr/bin/env python3
"""构建财务因子 PIT 面板（date × stock），供回测引擎直接加载（2026-10-27 04:20）

## 为什么
IC 验证结论：A股 fwd5 上经典质量因子（ROE/净利率/负债率/EPS）**全部无效**；
显著的是 gross_margin(负)、roe_yoy(负)、asset_yoy(负)、cfps(正)、周转率(正)。
这些显著因子本质是「低估值/低关注度」的反向代理 —— **不是质量 alpha**。

仍值得测一次「25 价量因子 + 财务因子」的合成，理由：
**这是唯一还没试过的「新信息」路径**，且理论上与价量因子低相关。
（此前所有「对现有因子做加权/减法/加趋势/缩 topk」的操作全部证伪。）

## PIT 对齐
财务数据按 **报告期 + 120 天**（覆盖 A 股全部法定公布截止日）作为最早可得日，
按交易日阶梯填充。对季报偏保守 → IC 偏低而不会虚高。

## 产出
`data/backtest/fundamental_panel.npz`
  - `dates`: str[ndates]  交易日
  - `codes`: str[ncodes]  股票代码（600000.SH 形式）
  - `<factor>`: float32[ndates, ncodes]，NaN = 不可用

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/build_fundamental_panel.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, os.path.join(ROOT, "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FUND = os.path.join(ROOT, "data", "fundamentals", "financial_ak_full.json")
OUT = os.path.join(ROOT, "data", "backtest", "fundamental_panel.npz")

# 只保留 IC 检验显著（|t| ≥ 2.5）的因子
FACTORS = [
    "gross_margin",        # IC −0.0103, t −4.45（低者涨）
    "roe_yoy",             # IC −0.0097, t −3.90（低者涨）
    "asset_yoy",           # IC −0.0079, t −3.40（低者涨）
    "cfps",                # IC +0.0070, t +3.19（高者涨）
    "inventory_turnover",  # IC +0.0050, t +2.63
    "asset_turnover",      # IC +0.0048, t +2.58
]

# 回测窗口（与主线一致）
START, END = "2023-05-08", "2026-05-08"


def main() -> int:
    if not os.path.exists(FUND):
        print(f"[panel] 缺少 {FUND}")
        return 1
    with open(FUND) as fh:
        raw = json.load(fh)
    print(f"[panel] 财务数据 {len(raw)} 只", flush=True)

    # ── 交易日历（用平台自己的价格数据，保证与回测引擎同一套日期）──
    from app.services.backtest.execution.data_loader import DataLoader
    dl = DataLoader(data_dir=os.path.join(ROOT, "data"))
    codes = []
    inst = os.path.join(ROOT, "data", "qlib_official_data", "instruments", "all.txt")
    with open(inst) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if parts and "_" in parts[0]:
                c, _, m = parts[0].partition("_")
                if m.upper() in ("SH", "SZ", "BJ") and c.isdigit():
                    codes.append(f"{c}.{m.upper()}")
    codes = sorted(codes)
    probe = dl.load_stock_data(codes[0], pd.Timestamp(START), pd.Timestamp(END))
    dates = pd.DatetimeIndex(sorted(set(probe.index))).tz_localize(None)
    dates = dates[(dates >= pd.Timestamp(START)) & (dates <= pd.Timestamp(END))]
    print(f"[panel] 交易日 {len(dates)}  {dates[0].date()}→{dates[-1].date()}",
          flush=True)

    n_c = len(codes)
    col_idx = {c: i for i, c in enumerate(codes)}
    mats = {f: np.full((len(dates), n_c), np.nan, dtype=np.float32)
            for f in FACTORS}

    # 6 位 → 完整代码
    alias = {c.split(".")[0]: c for c in codes}

    n_filled = 0
    for code6, recs in raw.items():
        full = alias.get(str(code6).split(".")[0])
        if full is None:
            continue
        j = col_idx[full]
        # 按可得日排序 → 阶梯填充
        rows = []
        for rec in recs:
            avail = rec[0]
            vals = rec[1] if len(rec) == 2 else rec[2]
            rows.append((avail, vals))
        rows.sort(key=lambda x: x[0])
        # 找每个可得日对应的交易日位置
        pos = dates.searchsorted(pd.DatetimeIndex([r[0] for r in rows]), side="left")
        for (avail, vals), p in zip(rows, pos):
            if p >= len(dates):
                continue
            for f in FACTORS:
                v = vals.get(f)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv != fv:
                    continue
                mats[f][p:, j] = fv      # 阶梯填充（之后有更新则覆盖）
            n_filled += 1

    print(f"[panel] 填充 {n_filled} 条财报记录", flush=True)
    for f in FACTORS:
        m = mats[f]
        cov = float(np.isfinite(m).sum()) / m.size
        print(f"  {f:<22} 覆盖率 {cov*100:5.1f}%", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT,
        dates=np.array([d.strftime("%Y-%m-%d") for d in dates]),
        codes=np.array(codes),
        **{f: mats[f] for f in FACTORS},
    )
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"[panel] 已写入 {OUT}（{size_mb:.0f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
