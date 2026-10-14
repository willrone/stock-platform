"""qlib 特征面板横截面 RankIC 筛选 (信号侧研究弧第一步).

窗口: 2021-05-11 → 2026-05-08 (qlib 特征面板实际覆盖)
宇宙: 全市场 (features/day 下全部 ~5513 只, 逐日有效截面)
特征: block2 Alpha158 集 (~186 列, VWAP0 全 NaN 剔除)

数据要点 (2026-10-13 实测):
  - 每个 parquet 纵向拼两块: block1 = $OHLCV + 传统指标 (仅真实交易日),
    block2 = Alpha158 (含节假日的全 weekdays 行, 节假日行特征为 ffill/NaN)
  - b1 日期 ⊂ b2 日期; 前向收益必须在真实交易日历上 shift, 否则跨节假日 shift 错位
  - 故日历 = 抽样股票 $close 有效日期的并集 (≈1210 交易日), 特征与价格都 reindex 到该日历

三道门 (对齐研究纪律):
  G1 主门: fwd1 RankIC BH-FDR q < 0.05 (t 用 N/horizon 有效样本校正重叠)
  G2 子期同向: 2021-05→2023-05 与 2023-05→2026-05 两段 IC 同号
  G3 邻近周期同向: fwd1/fwd5/fwd20 三 horizon IC 同号
幸存 = 三门全过。

输出: data/strategy_baseline/factor_ic_study_5y.json
"""
import json
import os
import time
from datetime import datetime
from math import erfc, sqrt

import numpy as np
import pandas as pd

FEAT_DIR = "/Users/rone/projects/stock-platform/data/qlib_data/features/day"
OUT_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/factor_ic_study_5y.json"
SMOKE_N = int(os.environ.get("IC_SMOKE_N", "0"))  # >0 = 抽样冒烟模式
SPLIT = pd.Timestamp("2023-05-09")  # 子期分界 (对齐既有研究的 2023-26 子期惯例)
HORIZONS = {"fwd1": 1, "fwd5": 5, "fwd20": 20}
EXCLUDE = {"$open", "$high", "$low", "$close", "$volume", "ts_code", "LABEL0", "VWAP0"}
B2_DEF = "CNTP5"  # block2 定义列 (Alpha158 专有)

t0 = time.time()
files = sorted(f for f in os.listdir(FEAT_DIR) if f.endswith(".parquet"))
if SMOKE_N > 0:
    files = files[:: max(1, len(files) // SMOKE_N)][:SMOKE_N]
    OUT_JSON = OUT_JSON.replace(".json", f"_smoke{SMOKE_N}.json")
print(f"[load] {len(files)} files", flush=True)

# ---- 日历: 抽样股票 $close 有效日期的并集 (真实交易日) ----
date_sets = []
step = max(1, len(files) // 20)
for f in files[::step]:
    x = pd.read_parquet(os.path.join(FEAT_DIR, f))
    idx = pd.DatetimeIndex(x.index.get_level_values("date"))
    ok = x["$close"].notna().to_numpy()
    date_sets.append(set(idx[ok]))
dates = pd.DatetimeIndex(sorted(set().union(*date_sets)))
n_d = len(dates)
date_pos = {d: i for i, d in enumerate(dates)}
print(f"[calendar] {n_d} trading dates {dates[0].date()} -> {dates[-1].date()}", flush=True)

n_s = len(files)
code_to_col = {f[:-8]: i for i, f in enumerate(files)}
close_mat = np.full((n_d, n_s), np.nan, dtype=np.float32)
feature_mats: dict[str, np.ndarray] = {}
feat_names: list[str] = []

t1 = time.time()
for k, f in enumerate(files):
    code = f[:-8]
    col = code_to_col[code]
    try:
        df = pd.read_parquet(os.path.join(FEAT_DIR, f))
    except Exception:
        continue
    idx = pd.DatetimeIndex(df.index.get_level_values("date"))
    df = df.copy()
    df.index = idx
    b1 = df[df["$close"].notna()]
    if not b1.empty:
        pos = np.array([date_pos[d] for d in b1.index if d in date_pos], dtype=np.int64)
        close_mat[pos, col] = b1["$close"].to_numpy(dtype=np.float32)[: len(pos)]
    b2 = df[df[B2_DEF].notna()] if B2_DEF in df.columns else df[df["$close"].isna()]
    if not b2.empty:
        pos = np.array([date_pos[d] for d in b2.index if d in date_pos], dtype=np.int64)
        if not feat_names:
            for c in df.columns:
                if c in EXCLUDE or c == B2_DEF:
                    continue
                if b2[c].notna().any():
                    feat_names.append(c)
                    feature_mats[c] = np.full((n_d, n_s), np.nan, dtype=np.float32)
        for c in feat_names:
            vals = b2[c].to_numpy(dtype=np.float32)
            n = min(len(pos), len(vals))
            feature_mats[c][pos[:n], col] = vals[:n]
    if (k + 1) % 500 == 0:
        print(f"[load] {k+1}/{n_s} elapsed {time.time()-t1:.0f}s", flush=True)

print(f"[load] done {time.time()-t1:.0f}s, features={len(feat_names)}", flush=True)

# ---- 前向收益 (真实交易日历 shift) ----
close_df = pd.DataFrame(close_mat, index=dates)
fwd = {
    name: (close_df.shift(-h) / close_df - 1.0).to_numpy(dtype=np.float32)
    for name, h in HORIZONS.items()
}


def rowwise_rank(mat: np.ndarray) -> np.ndarray:
    return pd.DataFrame(mat).rank(axis=1).to_numpy(dtype=np.float32)


fwd_rank = {n: rowwise_rank(m) for n, m in fwd.items()}
fwd_mask = {n: np.isfinite(m) for n, m in fwd_rank.items()}
print("[ic] forward ranks done", flush=True)


def masked_rowwise_pearson(xr, yr, ymask):
    m = np.isfinite(xr) & ymask
    n = m.sum(axis=1).astype(np.float64)
    good = n >= 30
    x = np.where(m, xr, 0.0).astype(np.float64)
    y = np.where(m, yr, 0.0).astype(np.float64)
    sx, sy = x.sum(1), y.sum(1)
    mx = np.where(good, sx / np.maximum(n, 1), 0.0)
    my = np.where(good, sy / np.maximum(n, 1), 0.0)
    xd = np.where(m, xr - mx[:, None], 0.0)
    yd = np.where(m, yr - my[:, None], 0.0)
    cov = (xd * yd).sum(1)
    vx = np.sqrt((xd * xd).sum(1))
    vy = np.sqrt((yd * yd).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = cov / (vx * vy)
    ic[~good] = np.nan
    return ic.astype(np.float32)


def bh_fdr(pvals, q=0.05):
    ok = np.isfinite(pvals)
    p = pvals[ok]
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    below = ranked <= q * np.arange(1, n + 1) / n
    reject = np.zeros(n, dtype=bool)
    if below.any():
        reject[order[: np.where(below)[0].max() + 1]] = True
    out = np.zeros(len(pvals), dtype=bool)
    out[ok] = reject
    return out


def p_two_sided(t):
    return erfc(abs(t) / sqrt(2.0))


results = {}
split_i = int(np.searchsorted(dates.values, np.datetime64(SPLIT)))
fwd1_p = []
for fi, name in enumerate(feat_names):
    xr = rowwise_rank(feature_mats[name])
    stats = {}
    for hn, hr in fwd_rank.items():
        ic_s = masked_rowwise_pearson(xr, hr, fwd_mask[hn])
        ic_s = ic_s[np.isfinite(ic_s)]
        if len(ic_s) < 100:
            stats[hn] = None
            continue
        h = HORIZONS[hn]
        n_eff = max(1.0, len(ic_s) / h)
        mean, std = float(ic_s.mean()), float(ic_s.std(ddof=1))
        t = mean / std * sqrt(n_eff) if std > 0 else 0.0
        seg1, seg2 = ic_s[:split_i], ic_s[split_i:]
        stats[hn] = {
            "ic_mean": round(mean, 5), "ic_std": round(std, 5),
            "icir": round(mean / std, 4) if std > 0 else 0.0,
            "t": round(float(t), 3), "n_days": int(len(ic_s)),
            "seg1_ic": round(float(seg1.mean()), 5) if len(seg1) > 100 else None,
            "seg2_ic": round(float(seg2.mean()), 5) if len(seg2) > 100 else None,
        }
    results[name] = stats
    fwd1_p.append(p_two_sided(stats["fwd1"]["t"]) if stats.get("fwd1") else 1.0)
    if (fi + 1) % 40 == 0:
        print(f"[ic] {fi+1}/{len(feat_names)} elapsed {time.time()-t1:.0f}s", flush=True)

fwd1_reject = bh_fdr(np.array(fwd1_p), q=0.05)

survivors = []
for i, name in enumerate(feat_names):
    s = results[name]
    f1, f5, f20 = s.get("fwd1"), s.get("fwd5"), s.get("fwd20")
    g1 = bool(fwd1_reject[i])
    g2 = bool(
        f1 and f1["seg1_ic"] is not None and f1["seg2_ic"] is not None
        and f1["seg1_ic"] * f1["seg2_ic"] > 0
    )
    g3 = bool(
        f1 and f5 and f20
        and len({np.sign(f1["ic_mean"]), np.sign(f5["ic_mean"]), np.sign(f20["ic_mean"])}) == 1
        and np.sign(f1["ic_mean"]) != 0
    )
    s["gates"] = {"G1_fdr": g1, "G2_subperiod": g2, "G3_horizon": g3}
    s["survivor"] = g1 and g2 and g3
    if s["survivor"]:
        survivors.append(name)

survivors.sort(key=lambda c: -abs(results[c]["fwd1"]["icir"]))
print(f"\n[survivors] {len(survivors)} / {len(feat_names)}", flush=True)
for c in survivors:
    f1 = results[c]["fwd1"]
    print(f"  {c:22s} ic={f1['ic_mean']:+.4f} icir={f1['icir']:+.3f} t={f1['t']:+.1f} "
          f"seg1={f1['seg1_ic']:+.4f} seg2={f1['seg2_ic']:+.4f}", flush=True)

out = {
    "meta": {
        "window": [str(dates[0].date()), str(dates[-1].date())],
        "n_dates": int(n_d), "n_stocks": int(n_s), "n_features": len(feat_names),
        "split": str(SPLIT.date()), "horizons": HORIZONS,
        "gates": "G1 fwd1 BH-FDR q<0.05; G2 两子期同号; G3 fwd1/5/20 同号",
        "generated_at": datetime.now().isoformat(),
    },
    "results": results,
    "survivors": survivors,
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(f"[done] {time.time()-t0:.0f}s -> {OUT_JSON}", flush=True)
