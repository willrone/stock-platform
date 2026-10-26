#!/usr/bin/env python3
"""25 因子 vs 42 因子在 5 年窗口的对比（2026-10-27 06:55）

## 为什么必须做
- 3 年窗口：25 因子 **0.880** > 42 因子 0.799 → 主线已切 25 因子
- 但 5 年窗口（更可信，含 2021-2023 熊市/震荡）：子代理只测了 42 因子
  （base_42f = 0.635），**没有 25 因子的对照**
- 暴露中性化（size+illiq）在 3 年 IS/OOS 都有效（0.767 / 1.595），
  却在 5 年失效（0.548）→ 需要确认「25 因子更优」是不是也只是近 3 年现象

## 口径
复用子代理的实验室（ensemble_lab）：它的 tradability 走正式的
`app/services/backtest/execution/tradability.py`，且 3 年基线与原生引擎
逐位一致（42f 0.7991795 / 25f 0.8795669）→ 口径可信。

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_5y_25f_vs_42f.py
"""
from __future__ import annotations

import json
import os
import sys

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ensemble_lab as L  # noqa: E402
import methods_lab as M  # noqa: E402
import methods_volmanaged as V  # noqa: E402

OUT = os.path.join(ROOT, "data", "backtest", "verify_5y_25f_vs_42f.json")


def metrics_from_nav(nav: pd.Series) -> dict:
    nav = nav.copy()
    nav.index = pd.DatetimeIndex(nav.index)
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 60:
        return {"error": "too short"}
    ann_r = float(r.mean()) * 243
    ann_v = float(r.std(ddof=1)) * (243 ** 0.5)
    peak = nav.cummax()
    mdd = float((nav / peak - 1.0).min())
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
    print(f"[5y] 窗口 {master[0].date()}→{master[-1].date()}  "
          f"股票数 {mats['close'].shape[1]}  交易日 {len(master)}", flush=True)

    cfg = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / 60)

    results = {}
    for label, panel_aligned in (("25f", False), ("42f", True)):
        print(f"\n[5y] --- {label} ---", flush=True)
        try:
            score, _ = L.build_42f_score(
                mats, panel_aligned=panel_aligned,
                panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"),
            )
            eq, _ = V.equity_series(score, mats, trad, cfg)
            nav = pd.Series(eq, index=master)
            m = metrics_from_nav(nav)
            results[label] = m
            print(f"  夏普={m['sharpe_ratio']:.3f} 收益={m['total_return']*100:+.2f}% "
                  f"回撤={m['max_drawdown']*100:.2f}% 波动={m['volatility']:.3f}",
                  flush=True)
            if m.get("yearly"):
                print("  分年: " + "  ".join(
                    f"{k}:{v*100:+.1f}%" for k, v in sorted(m["yearly"].items())))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[label] = {"error": f"{type(e).__name__}: {e}"}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "window": f"{master[0].date()}→{master[-1].date()}",
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "25 公式因子 vs 42 因子（Alpha158 面板真生效）在 5 年窗口的对比",
            },
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[5y] 已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
