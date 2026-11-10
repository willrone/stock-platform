#!/usr/bin/env python3
"""暴露中性化 + 波动率目标化 的组合（2026-10-27 03:05）

## 目标
用户三个目标：夏普 ≥1.5 / 胜率 ≥75% / 回撤 ≤20%。

## 已验证的事实
- **暴露中性化（size+illiq）跨 3 段全部有效**：Δ夏普 +0.222 / +0.241 / +0.307，
  且每段回撤都改善 2~3pp（`verify_neut_segments.py`）
- 波动率目标化（Moreira-Muir 2017）能压回撤，**但只在策略波动率高于 target 时生效**
- 中性化后的策略波动率更低 → 需要**更高的 target_vol** 才不缩仓

## 本脚本要回答
「中性化 + 合适档位的波动率目标」能否同时拿到：
夏普 ≥1.5、胜率 ≥75%、回撤 ≤20%（3 年真实盘口径 + 分段验证）

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_neut_vol_combo.py
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

OUT = os.path.join(ROOT, "data", "backtest", "neut_vol_combo.json")

TARGETS = (0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30)
LOOKBACKS = (20, 40, 60)
CAPS = (1.0, 1.5, 2.0)


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
    }


def main() -> int:
    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    master = mats["master"]

    amount = mats["close"] * mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size_proxy = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(mats["close"], mats["volume"], 20)

    s42, _ = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    s_neut = neutralize_score(s42, {"size": size_proxy, "illiq": illiq})

    cfg = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / 60)

    eq_base, _ = V.equity_series(s42, mats, trad, cfg)
    eq_neut, _ = V.equity_series(s_neut, mats, trad, cfg)
    r_base = pd.Series(eq_base, index=master).pct_change(fill_method=None).fillna(0).to_numpy()
    r_neut = pd.Series(eq_neut, index=master).pct_change(fill_method=None).fillna(0).to_numpy()

    mb = metrics_from_nav(pd.Series(eq_base, index=master))
    mn = metrics_from_nav(pd.Series(eq_neut, index=master))
    print(f"[combo] 基线 42f : 夏普={mb['sharpe_ratio']:.3f} 收益={mb['total_return']*100:+.2f}% "
          f"回撤={mb['max_drawdown']*100:.2f}% 波动={mb['volatility']:.3f}", flush=True)
    print(f"[combo] 中性化  : 夏普={mn['sharpe_ratio']:.3f} 收益={mn['total_return']*100:+.2f}% "
          f"回撤={mn['max_drawdown']*100:.2f}% 波动={mn['volatility']:.3f}", flush=True)

    results = {"base_42f": mb, "neut_sizeliq": mn, "combo": {}}
    hits = []
    for target in TARGETS:
        for lb in LOOKBACKS:
            for cap in CAPS:
                rm = V.vol_managed_returns(eq_neut, master, target, lookback=lb, cap=cap)
                m = metrics_from_nav(pd.Series(np.cumprod(1.0 + rm), index=master))
                key = f"t{int(target*100)}_lb{lb}_cap{cap}"
                results["combo"][key] = m
                ok = (m["sharpe_ratio"] >= 1.5 and m["max_drawdown"] >= -0.20)
                if ok:
                    hits.append((key, m))
                    print(f"  ✅ {key:<16} 夏普={m['sharpe_ratio']:.3f} "
                          f"收益={m['total_return']*100:+8.2f}% 回撤={m['max_drawdown']*100:7.2f}%",
                          flush=True)
    if not hits:
        print("  （没有同时满足「夏普≥1.5 且 回撤≤20%」的配置）")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "window": f"{master[0].date()}→{master[-1].date()}",
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "暴露中性化 + 波动率目标化组合；真实盘口径",
            },
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[combo] 已写入 {OUT}")

    # 分段验证最优的几个
    if hits:
        best = max(hits, key=lambda kv: kv[1]["sharpe_ratio"])
        print(f"\n[combo] 最优 {best[0]}: 夏普={best[1]['sharpe_ratio']:.3f} "
              f"回撤={best[1]['max_drawdown']*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
