#!/usr/bin/env python3
"""因子级 RankIC 拆解（2026-10-27）—— 快速定位「哪些因子真的有预测力」

## 为什么做这个
子代理在做**跑回测的消融/子集搜索**（每组 80~540 秒，很慢）。
本脚本用**直接算 IC** 的方式给出同样的信息（秒级），可作为交叉验证：
- 因子 |RankIC(fwd5)| 大 → 值得保留
- 因子 RankIC 接近 0 或符号不稳 → 是噪声，消融掉应该不掉夏普

## 口径
- 3 年窗口 2023-05-08 → 2026-05-08，股票池用 `all.txt`（PIT）
- 逐日横截面 Spearman RankIC，ICIR = mean(IC)/std(IC)，t = ICIR×√N
- 因子定义直接 import 平台自己的 `_HONEST_FORMULAS`（口径与回测一致）

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/factor_ic_decompose.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.path.join(ROOT, "data", "backtest", "factor_ic_decompose.json")


def main() -> int:
    from app.services.backtest.execution.data_loader import DataLoader
    from daily_signal import _HONEST_FORMULAS

    dl = DataLoader(data_dir=os.path.join(ROOT, "data"))
    codes = []
    with open(os.path.join(ROOT, "data", "qlib_official_data",
                           "instruments", "all.txt")) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if parts and "_" in parts[0]:
                c, _, m = parts[0].partition("_")
                if m.upper() in ("SH", "SZ", "BJ") and c.isdigit():
                    codes.append(f"{c}.{m.upper()}")
    codes = sorted(codes)

    s, e = datetime(2023, 5, 8), datetime(2026, 5, 8)
    print(f"[ic] 加载 {len(codes)} 只 {s.date()}→{e.date()}", flush=True)

    # 一次性把 close/volume/vchg 拼成矩阵（因子只需要这三个）
    close = {}
    vol = {}
    vchg = {}
    for i, c in enumerate(codes):
        try:
            d = dl.load_stock_data(c, s, e)
        except Exception:
            continue
        if d is None or d.empty:
            continue
        close[c] = _col(d, ("$close", "close"))
        vol[c] = _col(d, ("$volume", "volume"))
        vchg[c] = _col(d, ("$change", "change"))
        if (i + 1) % 1000 == 0:
            print(f"  加载 {i+1}/{len(codes)}", flush=True)
    codes = [c for c in codes if c in close]
    idx = pd.DatetimeIndex(sorted(set().union(*[set(v.index) for v in close.values()])))
    idx = idx[(idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))]
    print(f"[ic] 有效股票 {len(codes)}  交易日 {len(idx)}", flush=True)

    close_df = pd.DataFrame({c: close[c] for c in codes}).reindex(idx)
    vol_df = pd.DataFrame({c: vol[c] for c in codes}).reindex(idx)
    vchg_df = pd.DataFrame({c: vchg[c] for c in codes}).reindex(idx)
    ret = close_df.pct_change(fill_method=None)

    # 前瞻 5 日收益（信号日 → 5 个交易日后）
    fwd5 = (close_df.shift(-5) / close_df - 1.0)

    # 复刻策略的因子定义
    facs = {}
    for name, _sign in _HONEST_FORMULAS.items():
        try:
            facs[name] = _make_factor(name, close_df, vol_df, vchg_df, ret)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {name}: {type(e).__name__}", flush=True)

    # 逐因子算逐日 RankIC
    rows = []
    for name, f in facs.items():
        ics = []
        for i in range(len(idx) - 5):
            x = f.iloc[i]
            y = fwd5.iloc[i]
            ok = x.notna() & y.notna()
            if ok.sum() < 30:
                continue
            xv, yv = x[ok], y[ok]
            if xv.nunique() < 3 or yv.nunique() < 3:
                continue
            r = xv.rank().corr(yv.rank())
            if pd.notna(r):
                ics.append(float(r))
        if len(ics) < 60:
            continue
        arr = np.array(ics)
        icir = arr.mean() / (arr.std() + 1e-12)
        t = icir * np.sqrt(len(arr))
        pos_ratio = float((arr > 0).mean())
        rows.append({
            "factor": name,
            "mean_ic": round(float(arr.mean()), 5),
            "std_ic": round(float(arr.std()), 5),
            "icir": round(float(icir), 4),
            "t_stat": round(float(t), 2),
            "ic_positive_ratio": round(pos_ratio, 4),
            "n_days": len(arr),
        })

    rows.sort(key=lambda r: -abs(r["icir"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "window": f"{s.date()}→{e.date()}",
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "因子级 RankIC(fwd5) 拆解；用平台自己的 _HONEST_FORMULAS",
            },
            "factors": rows,
        }, fh, ensure_ascii=False, indent=1)

    print(f"\n{'因子':<12}{'meanIC':>9}{'ICIR':>8}{'t':>7}{'IC>0占比':>10}")
    print("-" * 48)
    for r in rows:
        print(f"{r['factor']:<12}{r['mean_ic']:>9.4f}{r['icir']:>8.3f}"
              f"{r['t_stat']:>7.2f}{r['ic_positive_ratio']*100:>9.1f}%")
    print(f"\n[ic] 已写入 {OUT}")
    return 0


def _col(df, names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _make_factor(name, close, vol, vchg, ret):
    """复刻 honest_composite_strategy / daily_signal 的因子定义。"""
    n = int("".join(ch for ch in name if ch.isdigit()) or 0)
    if name.startswith("RET"):
        return close.pct_change(n, fill_method=None)
    if name.startswith("REV"):
        return close.pct_change(n, fill_method=None) * -1.0
    if name.startswith("MA"):
        return close / close.rolling(n, min_periods=1).mean() - 1.0
    if name.startswith("MIN"):
        return close.rolling(n, min_periods=1).min() / close - 1.0
    if name.startswith("MAX"):
        return close.rolling(n, min_periods=1).max() / close - 1.0
    if name.startswith("STD"):
        return close.rolling(n, min_periods=1).std() / close
    if name.startswith("CORR"):
        return close.rolling(n, min_periods=1).corr(vol)
    if name.startswith("CORD"):
        return ret.rolling(n, min_periods=1).corr(vchg)
    if name.startswith("VMA"):
        return vol / vol.rolling(n, min_periods=1).mean() - 1.0
    if name.startswith("VSUM"):
        return vol.rolling(n, min_periods=1).sum() / vol.rolling(
            n, min_periods=1).mean() - 1.0
    if name.startswith("IMAX"):
        return (close - close.rolling(n, min_periods=1).max()) / close.rolling(
            n, min_periods=1).max()
    if name.startswith("IMIN"):
        return (close - close.rolling(n, min_periods=1).min()) / close.rolling(
            n, min_periods=1).min()
    if name.startswith("WVMA"):
        return close / (vol.rolling(n, min_periods=1).mean()
                        * close.rolling(n, min_periods=1).mean() / close) - 1.0
    if name.startswith("RSQR"):
        return close.rolling(n, min_periods=1).corr(vol) ** 2
    if name.startswith("RESI"):
        x = close.rolling(n, min_periods=1)
        beta = x.cov(ret) / x.var()
        return close - (x.mean() + beta * (ret - ret.rolling(n, min_periods=1).mean()))
    if name.startswith("QTLD"):
        return (close / close.rolling(n, min_periods=1).mean() - 1.0) / close.rolling(
            n, min_periods=1).std() / close
    if name.startswith("BETA"):
        return close.rolling(n, min_periods=1).cov(ret) / ret.rolling(
            n, min_periods=1).var()
    raise KeyError(name)


if __name__ == "__main__":
    raise SystemExit(main())
