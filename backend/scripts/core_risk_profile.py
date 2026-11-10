"""主仓风险画像（§12.80 ①）：50/50 必须按**收益等权**算，不能按**价格等权**算。

踩过的坑：`df.mean(axis=1)` 直接平均价格 → 511220 单价 10 元、515450 单价 1.5 元，
债腿白占 87% 权重，算出"波动 2.81% / 回撤 −3.84%"的算术假象。
正确做法：先算每只的日收益，再等权复合。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CODES = ("515450", "511220")
RISK_FREE = 0.0136  # 与台账 §12.60 一致


def series(code: str) -> pd.Series:
    df = pd.read_parquet(ROOT / "data/etf/parquet" / f"{code}.parquet")
    return pd.Series(df["close"].astype(float).values,
                     index=pd.to_datetime(df["date"].astype(str).str[:10]))


def load() -> pd.DataFrame:
    return pd.DataFrame({c: series(c) for c in CODES}).dropna()


def ann_from_ret(r: pd.Series) -> float:
    return float((1 + r).prod() ** (252.0 / len(r)) - 1) * 100


def max_dd_from_ret(r: pd.Series) -> float:
    nav = (1 + r).cumprod()
    return float(((nav - nav.cummax()) / nav.cummax()).min()) * 100


def sharpe_from_ret(r: pd.Series) -> float:
    vol = float(r.std() * np.sqrt(252))
    return (ann_from_ret(r) / 100 - RISK_FREE) / vol


def main() -> None:
    px = load()
    ret = px.pct_change().dropna()
    r_eq = ret.mean(axis=1)          # 收益等权（=每日再平衡的上界口径）

    out: dict = {"window": [str(px.index[0].date()), str(px.index[-1].date())],
                 "note": "前复权收盘价（§12.80 ⓪ 东财双口径审计确认）→ 收益已含分红；"
                         "若改用不复权价，同窗口主仓年化只有 2.03%（分红是主仓收益的主体）",
                 "legs": {}, "rolling": {}, "subwindows": {}}

    print("=== 单腿 ===")
    for c in CODES:
        r = ret[c]
        out["legs"][c] = {"annualized_pct": round(ann_from_ret(r), 2),
                          "max_dd_pct": round(max_dd_from_ret(r), 2),
                          "vol_pct": round(float(r.std() * np.sqrt(252)) * 100, 2),
                          "sharpe": round(sharpe_from_ret(r), 4)}
        print(f"  {c} 年化 {ann_from_ret(r):6.2f}%  dd {max_dd_from_ret(r):7.2f}%  "
              f"波动 {r.std()*np.sqrt(252)*100:5.2f}%  夏普 {sharpe_from_ret(r):.4f}")
    corr = float(ret[CODES[0]].corr(ret[CODES[1]]))
    out["legs_corr"] = round(corr, 3)
    print(f"  两腿日收益相关 {corr:.3f}")

    print(f"\n=== 组合（收益等权）年化 {ann_from_ret(r_eq):6.2f}%  "
          f"dd {max_dd_from_ret(r_eq):7.2f}%  波动 {r_eq.std()*np.sqrt(252)*100:5.2f}%  "
          f"夏普 {sharpe_from_ret(r_eq):.4f}")
    out["combo_daily_rebalanced"] = {"annualized_pct": round(ann_from_ret(r_eq), 2),
                                     "max_dd_pct": round(max_dd_from_ret(r_eq), 2),
                                     "vol_pct": round(float(r_eq.std() * np.sqrt(252)) * 100, 2),
                                     "sharpe": round(sharpe_from_ret(r_eq), 4)}

    yearly = []
    print("\n=== 自然年（年初再平衡）===")
    for y in sorted({d.year for d in ret.index}):
        sub = ret[ret.index.year == y]
        if len(sub) < 40:
            continue
        ra = float((1 + sub[CODES[0]]).prod() - 1) * 100
        rb = float((1 + sub[CODES[1]]).prod() - 1) * 100
        rc = 0.5 * ra + 0.5 * rb
        yearly.append({"year": int(y), "combo_pct": round(rc, 2),
                       "515450_pct": round(ra, 2), "511220_pct": round(rb, 2),
                       "n_days": len(sub)})
        print(f"  {y}: 组合 {rc:6.2f}%   515450 {ra:6.2f}%   511220 {rb:6.2f}%")
    tot = float(np.prod([1 + y["combo_pct"] / 100 for y in yearly]))
    yrs = sum(y["n_days"] for y in yearly) / 252
    out["combo_annual_rebalanced"] = {"cumulative_pct": round(tot * 100 - 100, 2),
                                      "years": round(yrs, 2),
                                      "annualized_pct": round((tot ** (1 / yrs) - 1) * 100, 2),
                                      "by_year": yearly}
    print(f"  年度再平衡累计 {tot*100-100:.2f}% / {yrs:.2f} 年 → 年化 {(tot**(1/yrs)-1)*100:.2f}%")

    nav = (1 + r_eq).cumprod()
    print("\n=== 滚动窗口 ===")
    for tag, n in (("12m", 252), ("6m", 126), ("3m", 63)):
        s = ((nav / nav.shift(n) - 1) * 100).dropna()
        out["rolling"][tag] = {"n": int(len(s)), "median_pct": round(float(np.median(s)), 2),
                               "p25_pct": round(float(np.percentile(s, 25)), 2),
                               "p10_pct": round(float(np.percentile(s, 10)), 2),
                               "worst_pct": round(float(s.min()), 2),
                               "share_negative_pct": round(float((s < 0).mean() * 100), 1)}
        print(f"  {tag} n={len(s)} 中位 {np.median(s):6.2f}% p25 {np.percentile(s,25):6.2f}% "
              f"p10 {np.percentile(s,10):6.2f}% 最差 {s.min():6.2f}% 负占比 {(s<0).mean()*100:5.1f}%")
    for c in CODES:
        n = (1 + ret[c]).cumprod()
        s = ((n / n.shift(252) - 1) * 100).dropna()
        out["rolling"][f"{c}_12m"] = {"worst_pct": round(float(s.min()), 2),
                                      "p1_pct": round(float(np.percentile(s, 1)), 2),
                                      "share_negative_pct": round(float((s < 0).mean() * 100), 1)}
        print(f"  {c} 滚动12月 最差 {s.min():6.2f}%  p1 {np.percentile(s,1):6.2f}%  负占比 {(s<0).mean()*100:5.1f}%")

    print("\n=== 子窗口对照（为什么不能拿一个窗口当'稳'）===")
    for tag, s0, s1 in (("全史", "2020-02-26", "2026-11-09"),
                        ("台账窗口 2023-05→2026-05", "2023-05-12", "2026-05-08"),
                        ("含 2022 债赎潮 2021-11→2024-11", "2021-11-10", "2024-11-08"),
                        ("逆风起点 2022-01→2024-01", "2022-01-04", "2024-01-04"),
                        ("最近一年 2025-11→2026-11", "2025-11-09", "2026-11-09")):
        r = r_eq[(r_eq.index >= s0) & (r_eq.index <= s1)]
        if len(r) < 60:
            continue
        out["subwindows"][tag] = {"annualized_pct": round(ann_from_ret(r), 2),
                                  "max_dd_pct": round(max_dd_from_ret(r), 2),
                                  "sharpe": round(sharpe_from_ret(r), 4),
                                  "n_days": len(r)}
        print(f"  {tag:30} 年化 {ann_from_ret(r):6.2f}%  dd {max_dd_from_ret(r):7.2f}%  "
              f"夏普 {sharpe_from_ret(r):6.4f}  n={len(r)}")

    path = ROOT / "data/etf/core_risk_profile.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n→ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
