#!/usr/bin/env python3
"""为什么 GBM 打不过等权因子复合？—— 诊断脚本（不走引擎，纯打分层分析）

对同一套 42 因子，比较：
1. 等权复合分 vs 模型分的 **逐日 rank-IC**（对 1/5/20 日前瞻收益）
2. 模型分的 IC 是不是真的比复合分高
3. 模型分与复合分的**相关性**（模型是不是只是把复合分学了一遍）
4. 不同目标变量（raw / rank / cs_z / ind_neutral / vol_std）下的 IC
5. purge 前后的 IC 差异（look-ahead 到底贡献了多少虚高）

结论用于决定搜索策略：**如果模型 IC 打不过复合分，就不该指望模型线单独
把夏普拉到 1.2+，只能走"模型分 + 等权分集成"这条路。**
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.backtest.strategies.gbm_lab_strategy import (  # noqa: E402
    GbmLabStrategy, DEFAULT_PANEL_DIR,
)


def _rank_ic(a: np.ndarray, b: np.ndarray, min_n: int = 50) -> tuple:
    """逐日 rank-IC → (mean_ic, ic_ir, n_days)"""
    ics = []
    for t in range(a.shape[0]):
        x, y = a[t], b[t]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < min_n:
            continue
        xr = pd.Series(x[ok]).rank().to_numpy()
        yr = pd.Series(y[ok]).rank().to_numpy()
        if xr.std() == 0 or yr.std() == 0:
            continue
        ics.append(float(np.corrcoef(xr, yr)[0, 1]))
    if not ics:
        return 0.0, 0.0, 0
    arr = np.array(ics)
    ir = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0.0
    return float(arr.mean()), ir, len(arr)


def main():
    # 用 DataLoader 拿引擎同源数据
    from app.services.backtest.execution.data_loader import DataLoader
    dl = DataLoader(data_dir=os.path.join(ROOT, "data"))
    start = datetime(2023, 5, 8)
    end = datetime(2026, 5, 8)

    # 抽 1500 只股票（够算横截面 IC，又不至于太慢）
    sys.path.insert(0, os.path.join(BACKEND, "scripts"))
    from search_gbm_params import universe_from_instruments, QLIB_OFFICIAL
    all_codes = universe_from_instruments(QLIB_OFFICIAL)
    rng = np.random.default_rng(42)
    codes = sorted(rng.choice(all_codes, size=1500, replace=False).tolist())

    print(f"[diag] 加载 {len(codes)} 只股票 …", flush=True)
    stock_data = {}
    for i, c in enumerate(codes):
        df = dl.load_stock_data(c, start, end)
        if df is None or df.empty:
            continue
        stock_data[c] = df
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(codes)}", flush=True)
    codes = list(stock_data.keys())
    print(f"[diag] 有效股票 {len(codes)}", flush=True)

    indexes = [pd.DatetimeIndex(stock_data[c].index) for c in codes]
    master = indexes[0].append(indexes[1:]).unique().sort_values()
    n_dates, n_stocks = len(master), len(codes)
    print(f"[diag] master 日期 {n_dates} × 股票 {n_stocks}", flush=True)

    results = {}

    for tag, cfg in (
        ("base_cs_z", {"target": "cs_z", "horizon": 1, "label_anchor": "trade"}),
        ("h5_cs_z", {"target": "cs_z", "horizon": 5, "label_anchor": "trade"}),
        ("h20_cs_z", {"target": "cs_z", "horizon": 20, "label_anchor": "trade"}),
        ("h1_raw", {"target": "raw", "horizon": 1, "label_anchor": "trade"}),
        ("h5_raw", {"target": "raw", "horizon": 5, "label_anchor": "trade"}),
        ("h1_rank", {"target": "rank", "horizon": 1, "label_anchor": "trade"}),
        ("h5_rank", {"target": "rank", "horizon": 5, "label_anchor": "trade"}),
        ("h5_ind", {"target": "ind_neutral", "horizon": 5,
                    "label_anchor": "trade"}),
        ("h5_vol", {"target": "vol_std", "horizon": 5,
                    "label_anchor": "trade"}),
    ):
        conf = {
            "topk": 60, "n_drop": 5, "rebalance_days": "weekly",
            "rebalance_weekday": 4, "model": "lgbm", "n_estimators": 100,
            "max_depth": 3, "learning_rate": 0.05, "retrain": "monthly",
            "train_window": "expand", "warmup_months": 2,
            "feature_set": "f42", "use_presence": False, "target_vol": 0.20,
            "blend": 1.0, "purge": True, "n_jobs": 4, "lag": 1,
            "panel_dir": DEFAULT_PANEL_DIR,
        }
        conf.update(cfg)
        s = GbmLabStrategy(conf)
        s._start, s._end = start, end

        # 复刻 prepare_backtest_data 的核心部分（只算分，不落 engine）
        feats = s._build_features(stock_data, codes, master)
        raw, names, signs = feats["raw"], feats["names"], feats["signs"]
        close, vol = feats["close"], feats["vol"]

        anchor = s.lag if s.label_anchor == "trade" else 0
        with np.errstate(divide="ignore", invalid="ignore"):
            if anchor > 0:
                fwd = (close.shift(-(anchor + s.horizon)) /
                       close.shift(-anchor) - 1.0).to_numpy(dtype=np.float64)
            else:
                fwd = (close.shift(-s.horizon) / close - 1.0).to_numpy(
                    dtype=np.float64)
        target = s._make_target(close, fwd, vol)

        n_fac = len(raw)
        z_all = np.zeros((n_dates, n_stocks, n_fac), dtype=np.float32)
        pres_all = np.zeros((n_dates, n_stocks, n_fac), dtype=np.float32)
        min_cross = max(2, min(30, n_stocks // 10))
        for k in range(n_fac):
            mat = raw[k]
            ok = np.isfinite(mat)
            cnt = ok.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
                sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
                z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan)
            z = np.where(cnt >= min_cross, z, 0.0)
            z = np.where(np.isfinite(z), z, 0.0)
            z_all[:, :, k] = z
            pres_all[:, :, k] = ok

        acc = np.zeros((n_dates, n_stocks))
        wsum = np.zeros((n_dates, n_stocks))
        for k in range(len(names)):
            sgn = signs[k]
            ok = pres_all[:, :, k] > 0
            acc[ok] += z_all[:, :, k][ok] * sgn
            wsum[ok] += 1
        comp_eq = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)

        # 逐月重训 → 模型分
        pts = s._retrain_points(master)
        comp = np.full((n_dates, n_stocks), np.nan)
        for mi, k0 in enumerate(pts):
            k1 = pts[mi + 1] if mi + 1 < len(pts) else n_dates
            if mi < s.warmup_months:
                comp[k0:k1] = comp_eq[k0:k1]
                continue
            s._train_and_score(z_all, pres_all, target, k0, k1, comp,
                               master, mi)

        # ── 评估：对 1/5/20 日前瞻收益的 rank-IC ────────────────────
        row = {}
        for h_eval in (1, 5, 20):
            with np.errstate(divide="ignore", invalid="ignore"):
                ev = (close.shift(-h_eval) / close - 1.0).to_numpy(dtype=np.float64)
            m_ic, m_ir, nd = _rank_ic(comp, ev)
            e_ic, e_ir, _ = _rank_ic(comp_eq, ev)
            row[f"h{h_eval}"] = {
                "model_ic": m_ic, "model_icir": m_ir,
                "eq_ic": e_ic, "eq_icir": e_ir, "days": nd,
            }
        # 模型分 vs 复合分 的秩相关（模型是不是只学了复合分）
        cors = []
        for t in range(n_dates):
            a, b = comp[t], comp_eq[t]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 50:
                continue
            ar = pd.Series(a[ok]).rank().to_numpy()
            br = pd.Series(b[ok]).rank().to_numpy()
            if ar.std() == 0 or br.std() == 0:
                continue
            cors.append(float(np.corrcoef(ar, br)[0, 1]))
        row["model_vs_eq_corr"] = float(np.mean(cors)) if cors else 0.0
        results[tag] = row
        print(f"[diag] {tag}: " + " | ".join(
            f"{k}: 模型IC={v['model_ic']:.4f}/ICIR={v['model_icir']:.2f} "
            f"复合IC={v['eq_ic']:.4f}/ICIR={v['eq_icir']:.2f}"
            for k, v in row.items() if k.startswith("h")
        ) + f" | 模型~复合相关={row['model_vs_eq_corr']:.3f}", flush=True)

        del z_all, pres_all, raw
        import gc
        gc.collect()

    import json
    out = os.path.join(ROOT, "data", "backtest", "gbm_diag_ic.json")
    with open(out, "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(f"\n[diag] 已写入 {out}")


if __name__ == "__main__":
    main()
