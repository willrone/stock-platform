#!/usr/bin/env python3
"""基线 + 波动率目标化验证（2026-10-27 06:00）

## 为什么单独做这个
子代理在「候选策略」上叠了波动率目标化（Moreira & Muir, JFE 2017）拿到
5 年夏普 1.07~1.24。**但候选本身含「暴露中性化」，而中性化在 5 年上是失效的**
（基线 0.635 → neut_si 0.548）。

→ 需要验证**不含中性化、只加波动率目标化**的基线是什么表现。
这是最干净的检验：波动率目标化是**学术界公认方法**（Moreira-Muir 2017），
不是数据挖掘出来的参数。

**关键性质**：时变缩放（不是常数缩放）确实能提升夏普 —— 高波动时降仓、
低波动时加仓，这本身就是一种择时。常数缩放才是不改变夏普的。

## 实现
σ̂ 用**滞后**已实现波动率（只用 ≤ t-1 的收益）→ 无前视；
σ̂ 必须年化（×√252）才能和年化 target_vol 同量纲。

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_vol_managed_baseline.py
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

OUT = os.path.join(ROOT, "data", "backtest", "vol_managed_baseline.json")
NAV_CACHE = os.path.join(ROOT, "data", "backtest", "_blend_nav_25f.json")


def vol_managed_returns(r: np.ndarray, target_vol: float,
                         lookback: int = 20, cap: float = 2.0) -> np.ndarray:
    """Moreira-Muir 波动率目标化：f_t = clip(target_vol / σ̂_{t-1}, 0, cap)。"""
    mp = max(2, min(10, lookback))
    vol = (pd.Series(r).rolling(lookback, min_periods=mp).std().shift(1)
           * np.sqrt(252)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(np.isfinite(vol) & (vol > 1e-9), target_vol / vol, 1.0)
    f = np.clip(np.nan_to_num(f, nan=1.0), 0.0, cap)
    return r * f


def metrics(r: np.ndarray) -> dict:
    nav = np.cumprod(1.0 + r)
    rr = pd.Series(r).dropna()
    if len(rr) < 60:
        return {"error": "too short"}
    ann_r = float(rr.mean()) * 243
    ann_v = float(rr.std(ddof=1)) * (243 ** 0.5)
    peak = np.maximum.accumulate(nav)
    mdd = float(((nav / peak) - 1.0).min())
    return {
        "sharpe_ratio": round(ann_r / ann_v, 4) if ann_v > 0 else 0.0,
        "total_return": round(float(nav[-1] - 1.0), 6),
        "annualized_return": round(float(nav[-1] ** (243 / max(len(nav), 1)) - 1.0), 6),
        "max_drawdown": round(mdd, 6),
        "volatility": round(ann_v, 6),
    }


def main() -> int:
    with open(NAV_CACHE) as fh:
        d = json.load(fh)
    navs = {k: pd.Series(v) for k, v in d["navs"].items()}
    for k in navs:
        navs[k].index = pd.to_datetime(navs[k].index)
        navs[k] = navs[k].sort_index()

    idx = None
    for s in navs.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = idx.sort_values()

    results = {}
    for name in ("honest_composite", "rsi_71421"):
        nav = navs[name].reindex(idx).ffill()
        r = (nav / nav.iloc[0]).pct_change(fill_method=None).fillna(0.0).to_numpy()
        base = metrics(r)
        print(f"\n{'='*74}\n{name}  基准: 夏普={base['sharpe_ratio']:.3f} "
              f"收益={base['total_return']*100:+.2f}% 回撤={base['max_drawdown']*100:.2f}%"
              f" 波动={base['volatility']:.3f}\n{'='*74}")
        results[name] = {"baseline": base, "variants": {}}
        for target in (0.08, 0.10, 0.12, 0.15, 0.20):
            for lb in (20, 40, 60):
                for cap in (1.0, 1.5, 2.0):
                    rm = vol_managed_returns(r, target, lookback=lb, cap=cap)
                    m = metrics(rm)
                    key = f"t{int(target*100)}_lb{lb}_cap{cap}"
                    results[name]["variants"][key] = m
                    if cap == 1.0:
                        print(f"  target={target:.0%} lb={lb:3d} cap={cap:.1f}  "
                              f"夏普={m['sharpe_ratio']:6.3f} 收益={m['total_return']*100:+8.2f}% "
                              f"回撤={m['max_drawdown']*100:7.2f}% "
                              f"波动={m['volatility']:.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "基线 + Moreira-Muir 波动率目标化（滞后年化波动率，无前视）",
                "window": f"{idx[0].date()}→{idx[-1].date()}",
            },
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n已写入 {OUT}")

    print("\n=== 满足「夏普≥1.5 且 回撤≤20%」的配置 ===")
    hit = 0
    for name, blob in results.items():
        for k, m in blob["variants"].items():
            if m.get("sharpe_ratio", 0) >= 1.5 and m.get("max_drawdown", -1) >= -0.20:
                print(f"  ✅ {name} {k}: 夏普={m['sharpe_ratio']:.3f} "
                      f"回撤={m['max_drawdown']*100:.2f}%")
                hit += 1
    if not hit:
        print("  （无）")
    print("\n=== 各策略里「夏普最高 且 回撤≤20%」的配置 ===")
    for name, blob in results.items():
        ok = [(k, m) for k, m in blob["variants"].items()
              if m.get("max_drawdown", -1) >= -0.20]
        if ok:
            k, m = max(ok, key=lambda kv: kv[1]["sharpe_ratio"])
            print(f"  {name}: {k} → 夏普={m['sharpe_ratio']:.3f} "
                  f"回撤={m['max_drawdown']*100:.2f}% "
                  f"收益={m['total_return']*100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
