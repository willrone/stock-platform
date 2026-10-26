#!/usr/bin/env python3
"""逐因子 IC 诊断：17 个 Alpha158 面板因子在 features_full vs features/day 下
   的方向和强度 —— 定位"42 因子比 25 因子差"的原因。

对每个因子算：逐日 rank-IC（对 5 日前瞻收益），分别在：
  A. features_full 面板目录
  B. features/day 面板目录（旧的、日期重复的）
上计算，并对比 honest_composite 里写死的 sign。

如果某个因子的 IC 符号和写死的 sign 相反 → sign 错了 → 42 因子必然变差。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.backtest.execution.data_loader import DataLoader  # noqa: E402
from app.services.backtest.strategies.honest_composite_strategy import (  # noqa: E402
    _HONEST_FORMULAS, _ALPHAPANEL_FACTORS,
)
from app.services.backtest.strategies.gbm_lab_strategy import (  # noqa: E402
    DEFAULT_PANEL_DIR, LEGACY_PANEL_DIR,
)
from search_gbm_params import universe_from_instruments, QLIB_OFFICIAL  # noqa: E402


def rank_ic(mat: np.ndarray, fwd: np.ndarray, min_n: int = 50):
    ics = []
    for t in range(mat.shape[0]):
        x, y = mat[t], fwd[t]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_n:
            continue
        xr = pd.Series(x[ok]).rank().to_numpy()
        yr = pd.Series(y[ok]).rank().to_numpy()
        if xr.std() == 0 or yr.std() == 0:
            continue
        ics.append(float(np.corrcoef(xr, yr)[0, 1]))
    if not ics:
        return 0.0, 0, 0.0
    a = np.array(ics)
    ir = float(a.mean() / a.std() * np.sqrt(252)) if a.std() > 0 else 0.0
    return float(a.mean()), len(a), ir


def load_panel(code, cols, panel_dir, start, end):
    path = os.path.join(panel_dir, code.replace(".", "_") + ".parquet")
    try:
        pf = pd.read_parquet(path)
    except Exception:
        return None
    try:
        pf = pf.reset_index()
        if "date" not in pf.columns:
            return None
        pf["date"] = pd.to_datetime(pf["date"])
        pf = pf.set_index("date").sort_index()
        pf = pf.loc[pd.Timestamp(start):pd.Timestamp(end)]
        pf = pf[~pf.index.duplicated(keep="first")]
        miss = [c for c in cols if c not in pf.columns]
        if miss:
            return None
        return pf[cols]
    except Exception:
        return None


def main():
    dl = DataLoader(data_dir=os.path.join(ROOT, "data"))
    start = datetime(2023, 5, 8)
    end = datetime(2026, 5, 8)
    all_codes = universe_from_instruments(QLIB_OFFICIAL)
    rng = np.random.default_rng(42)
    codes = sorted(rng.choice(all_codes, size=900, replace=False).tolist())

    stock_data = {}
    for c in codes:
        df = dl.load_stock_data(c, start, end)
        if df is not None and not df.empty:
            stock_data[c] = df
    codes = list(stock_data.keys())
    indexes = [pd.DatetimeIndex(stock_data[c].index) for c in codes]
    master = indexes[0].append(indexes[1:]).unique().sort_values()
    n_dates, n_stocks = len(master), len(codes)
    print(f"[factor_ic] 股票 {n_stocks} × 日期 {n_dates}", flush=True)

    # close / fwd
    close = np.full((n_dates, n_stocks), np.nan)
    for j, c in enumerate(codes):
        df = stock_data[c]
        pos = master.get_indexer(pd.DatetimeIndex(df.index))
        ok = pos >= 0
        close[pos[ok], j] = pd.to_numeric(df["close"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan)[ok]
    cdf = pd.DataFrame(close, index=master)
    fwd = (cdf.shift(-5) / cdf - 1.0).to_numpy(dtype=np.float64)

    panel_cols = list(_ALPHAPANEL_FACTORS.keys())
    out = {}
    for tag, pdir in (("features_full", DEFAULT_PANEL_DIR),
                      ("features_day", LEGACY_PANEL_DIR)):
        mats = np.full((n_dates, n_stocks, len(panel_cols)), np.nan)
        cov = np.zeros(len(panel_cols))
        for j, c in enumerate(codes):
            pf = load_panel(c, panel_cols, pdir, start, end)
            if pf is None:
                continue
            pos = master.get_indexer(pd.DatetimeIndex(pf.index))
            ok = pos >= 0
            if not ok.any():
                continue
            blk = pf.to_numpy(dtype=np.float64, na_value=np.nan)
            mats[pos[ok], j, :] = blk[ok]
            for k in range(len(panel_cols)):
                cov[k] += np.isfinite(blk[ok, k]).sum()
        rows = {}
        for k, fname in enumerate(panel_cols):
            ic, nd, ir = rank_ic(mats[:, :, k], fwd)
            sign = _ALPHAPANEL_FACTORS[fname]
            cover = cov[k] / max(n_dates * n_stocks, 1)
            rows[fname] = {
                "ic": round(ic, 5), "icir": round(ir, 3), "days": nd,
                "sign": sign, "signed_ic": round(ic * sign, 5),
                "coverage": round(float(cover), 4),
                "sign_ok": bool(ic * sign > 0),
            }
        out[tag] = rows
        bad = [f for f, v in rows.items() if not v["sign_ok"]]
        print(f"\n[{tag}] 覆盖率均值={np.mean([v['coverage'] for v in rows.values()]):.3f}")
        for f, v in rows.items():
            flag = "  ← 符号反了!" if not v["sign_ok"] else ""
            print(f"  {f:8s} IC={v['ic']:+.5f} ICIR={v['icir']:+7.2f} "
                  f"sign={v['sign']:+.0f} signed_IC={v['signed_ic']:+.5f} "
                  f"覆盖={v['coverage']:.3f}{flag}")
        print(f"  → 符号反了的因子: {bad}")

    outp = os.path.join(ROOT, "data", "backtest", "gbm_factor_ic.json")
    with open(outp, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n[factor_ic] 已写入 {outp}")


if __name__ == "__main__":
    main()
