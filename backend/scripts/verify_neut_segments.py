#!/usr/bin/env python3
"""暴露中性化（size+illiq）的分段稳定性检验（2026-10-27 02:45）

## 为什么必须做
子代理报告：**暴露中性化**（Barra APT 风格逐日截面 OLS 残差化）
- 3 年全样本：0.799 → **1.031**，回撤 −16.5% → −12.4%
- IS（23-05→25-05）0.767 / OOS（25-05→26-05）1.595
- 但 **5 年窗口失效**（0.635 → 0.548）

→ IS+OOS 两段都有效、5 年失效 = 前 3 年有效、含更早年份就失效。
**必须用分段检验判定它是真信号还是只在特定时期有效** ——
这正是我今晚对 topk 做的那个检验（topk 就是这样被判定为过拟合的）。

## 口径
真实盘口径：lag=1 + 可成交性开 + topk60/ndrop5 + 5238 只 + 3 年窗口。
把 3 年切成 3 段各跑一遍，看 **base vs 中性化 的相对优势是否稳定**。

## 方法
- size 代理 = 20 日 log 成交额均值（平台无市值数据）
- illiq = Amihud(2002) ILLIQ 20 日均值取 log1p
- 中性化 = 逐日截面 OLS（y=复合分, X=[size, illiq, 1]）取残差
- 暴露全部用「当日及之前」数据 → 无前视

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_neut_segments.py
"""
from __future__ import annotations

import json
import os
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ensemble_lab as L  # noqa: E402
import methods_lab as M  # noqa: E402
import methods_volmanaged as V  # noqa: E402
from methods_research_v2 import neutralize_score  # noqa: E402

OUT = os.path.join(ROOT, "data", "backtest", "neut_segments.json")

#: 3 段各 1 年
SEGMENTS = [
    ("seg1_23-24", "2023-05-08", "2024-05-08"),
    ("seg2_24-25", "2024-05-08", "2025-05-08"),
    ("seg3_25-26", "2025-05-08", "2026-05-08"),
]


def metrics_from_nav(nav: pd.Series) -> dict:
    nav = nav.copy()
    nav.index = pd.DatetimeIndex(nav.index)
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 30:
        return {"error": "too short"}
    ann_r = float(r.mean()) * 243
    ann_v = float(r.std(ddof=1)) * (243 ** 0.5)
    mdd = float((nav / nav.cummax() - 1.0).min())
    yearly = {}
    for y, g in nav.groupby(pd.DatetimeIndex(nav.index).year):
        if len(g) > 1:
            yearly[str(y)] = round(float(g.iloc[-1] / g.iloc[0] - 1.0), 6)
    return {
        "sharpe_ratio": round(ann_r / ann_v, 4) if ann_v > 0 else 0.0,
        "total_return": round(float(nav.iloc[-1] / nav.iloc[0] - 1.0), 6),
        "max_drawdown": round(mdd, 6),
        "volatility": round(ann_v, 6),
        "yearly": yearly,
        "n_days": len(nav),
    }


def main() -> int:
    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    master = mats["master"]
    print(f"[seg] 窗口 {master[0].date()}→{master[-1].date()}  "
          f"股票 {mats['close'].shape[1]}  交易日 {len(master)}", flush=True)

    # ── 构造暴露矩阵（全部只用当日及之前数据）────────────────────
    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size_proxy = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(mats["close"], mats["volume"], 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = mats["close"][1:] / mats["close"][:-1] - 1.0
        ret = np.vstack([np.full((1, mats["close"].shape[1]), np.nan), ret])
    vol_proxy = pd.DataFrame(ret).rolling(60, min_periods=20).std().to_numpy()
    print("[seg] 暴露矩阵就绪", flush=True)

    # ── 打分：42f 原始 + 中性化(size+illiq)────────────────────
    s42, _ = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    s_neut = neutralize_score(s42, {"size": size_proxy, "illiq": illiq})
    print("[seg] 中性化完成", flush=True)

    cfg = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / 60)

    results = {}
    for label, s, e in SEGMENTS:
        lo, hi = pd.Timestamp(s), pd.Timestamp(e)
        sel = np.where((master >= lo) & (master < hi))[0]
        if len(sel) < 60:
            print(f"[seg] {label}: 交易日不足，跳过", flush=True)
            continue
        row = {}
        for name, score in (("base_42f", s42), ("neut_sizeliq", s_neut)):
            # 截取区间内的矩阵（暴露/分数的前视问题：区间起点前有数据，
            # 但滚动窗口本身只用历史 → 安全）
            sc_seg = score[sel].copy()
            mats_seg = {
                "close": mats["close"][sel], "open": mats.get("open", mats["close"])[sel],
                "high": mats.get("high", mats["close"])[sel],
                "low": mats.get("low", mats["close"])[sel],
                "volume": mats["volume"][sel], "codes": mats["codes"],
                "master": master[sel],
            }
            trad_seg = {
                "can_buy": trad["can_buy"][sel], "can_sell": trad["can_sell"][sel],
                "max_buy_amount": trad.get("max_buy_amount", np.full(
                    (len(sel), mats["close"].shape[1]), np.nan))[sel],
            }
            try:
                eq, _ = V.equity_series(sc_seg, mats_seg, trad_seg, cfg)
                m = metrics_from_nav(pd.Series(eq, index=master[sel]))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                m = {"error": f"{type(e).__name__}: {e}"}
            row[name] = m
            if "error" not in m:
                print(f"  {label:<14} {name:<14} 夏普={m['sharpe_ratio']:6.3f} "
                      f"收益={m['total_return']*100:+7.2f}% 回撤={m['max_drawdown']*100:6.2f}%",
                      flush=True)
        if "error" not in row.get("base_42f", {}) and "error" not in row.get("neut_sizeliq", {}):
            d = row["neut_sizeliq"]["sharpe_ratio"] - row["base_42f"]["sharpe_ratio"]
            row["delta_sharpe"] = round(d, 4)
            print(f"  {label:<14} → 中性化带来的夏普变化: {d:+.3f}", flush=True)
        results[label] = row

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "window": f"{master[0].date()}→{master[-1].date()}",
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "暴露中性化（size+illiq）的分段稳定性检验；"
                        "delta_sharpe 为正说明该段中性化有效",
            },
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[seg] 已写入 {OUT}")

    ds = [r.get("delta_sharpe") for r in results.values()
          if r.get("delta_sharpe") is not None]
    if ds:
        print(f"\n[seg] 各段 Δ夏普: {[round(x,3) for x in ds]}")
        print(f"[seg] 全为正 → 中性化是**稳定有效**的；"
              f"否则 → 只在特定时期有效（过拟合风险）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
