#!/usr/bin/env python3
"""多策略「资金分配」合成（2026-10-27）

## 为什么用资金分配而不是分数合成
引擎自带的 `StrategyPortfolio` 只有 4 种**信号整合**方法
（weighted_voting / rank_sum / borda / consensus_topk），都是把多个策略的
信号混成一份再选 TopK —— 那会让策略之间互相污染（一个策略的高分票
挤掉另一个策略的好票）。

**资金分配**是另一种更稳的做法：每个策略管自己的一笔资金、各自独立选股
独立建仓，组合收益 = Σ wᵢ × rᵢ(t)。

## 数学上等价，且零引擎改动
每个策略单独跑 → 拿到它的**日度收益率序列** rᵢ(t)（= 100% 资金投在该策略
时的日收益）。资金分配下，组合的日收益就是 Σ wᵢ·rᵢ(t)，再累乘成净值。
**这与真实的资金分配完全等价**（各策略的成本已各自计入）。

## 口径
`lag=1` + 可成交性开 + 5238 只 + 3 年真实盘口径。
各策略用**面板因子真生效**的正式实现（不是加速版）—— 这里要的是真实数字。

用法：
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/verify_strategy_blend.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

OUT = os.path.join(ROOT, "data", "backtest", "strategy_blend.json")

#: 策略 → (cfg, max_position_size)
STRATS = {
    "honest_composite": ({"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                          "rebalance_weekday": 4, "lag": 1}, 0.95 / 60),
    "rsi_71421": ({"topk": 30, "n_drop": 3, "rebalance_days": "weekly",
                   "rebalance_weekday": 4, "lag": 1}, 0.95 / 30),
    "factor_ranking": ({"topk": 50, "n_drop": 5, "rebalance_days": "weekly",
                        "rebalance_weekday": 4, "lag": 1}, 0.95 / 50),
    "gbm_voltgt": ({"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                    "rebalance_weekday": 4, "n_estimators": 100,
                    "max_depth": 3, "learning_rate": 0.05,
                    "warmup_months": 2, "target_vol": 0.20,
                    "vt_window": 20, "lag": 1}, 0.95 / 60),
}

#: 要试的权重组合
BLENDS = [
    {"honest_composite": 1.0},
    {"honest_composite": 0.7, "gbm_voltgt": 0.3},
    {"honest_composite": 0.6, "gbm_voltgt": 0.4},
    {"honest_composite": 0.5, "gbm_voltgt": 0.5},
    {"honest_composite": 0.8, "rsi_71421": 0.2},
    {"honest_composite": 0.5, "rsi_71421": 0.5},
    {"honest_composite": 0.6, "gbm_voltgt": 0.2, "rsi_71421": 0.2},
    {"honest_composite": 0.5, "gbm_voltgt": 0.25, "rsi_71421": 0.25},
    {"honest_composite": 0.4, "gbm_voltgt": 0.3, "rsi_71421": 0.3},
    {"honest_composite": 0.34, "gbm_voltgt": 0.33, "rsi_71421": 0.33},
]


def universe() -> list:
    path = os.path.join(ROOT, "data", "qlib_official_data", "instruments", "all.txt")
    codes = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if parts and "_" in parts[0]:
                c, _, m = parts[0].partition("_")
                if m.upper() in ("SH", "SZ", "BJ") and c.isdigit():
                    codes.append(f"{c}.{m.upper()}")
    return sorted(codes)


def run_strategy(name: str, codes, start, end, no_panel=False) -> dict:
    cfg, mps = STRATS[name]
    if no_panel:
        cfg = dict(cfg)
        cfg["panel_dir"] = "/nonexistent/panel_off"
    ex = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    res = asyncio.run(
        ex.run_backtest(
            strategy_name=name,
            stock_codes=codes,
            start_date=start, end_date=end,
            strategy_config=dict(cfg),
            backtest_config=BacktestConfig(
                initial_cash=1_000_000.0,
                commission_rate=0.0003, slippage_rate=0.0001,
                max_position_size=mps,
                cash_reserve_ratio=0.05, board_lot_size=100,
                stop_loss_pct=0.0, take_profit_pct=0.0,
                rebalance_frequency="daily",
                open_cost=0.0, close_cost=0.0, min_cost=0.0,
                enable_tradability=True,
            ),
        )
    )
    eq = res.get("equity_curve") or []
    out = {
        "sharpe_ratio": res.get("sharpe_ratio"),
        "total_return": res.get("total_return"),
        "max_drawdown": res.get("max_drawdown"),
        "total_trades": res.get("total_trades"),
        "win_rate": res.get("win_rate"),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    if eq:
        s = pd.Series({d: float(v) for d, v in eq}).sort_index()
        out["_nav"] = s
    return out


def metrics_from_nav(nav: pd.Series) -> dict:
    """从日度净值算夏普/回撤/收益/年化。"""
    nav = nav.copy()
    nav.index = pd.DatetimeIndex(nav.index)
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 30:
        return {"error": "too short"}
    ann_r = float(r.mean()) * 243
    ann_v = float(r.std(ddof=1)) * (243 ** 0.5)
    sharpe = ann_r / ann_v if ann_v > 0 else 0.0
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    dd = float((nav / nav.cummax() - 1.0).min())
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / max(years, 1e-9)) - 1.0
            if nav.iloc[0] > 0 else 0.0)
    yearly = {}
    for y, g in nav.groupby(pd.DatetimeIndex(nav.index).year):
        if len(g) > 1:
            yearly[str(y)] = round(float(g.iloc[-1] / g.iloc[0] - 1.0), 6)
    return {
        "sharpe_ratio": round(sharpe, 4),
        "total_return": round(total, 6),
        "annualized_return": round(cagr, 6),
        "max_drawdown": round(dd, 6),
        "volatility": round(ann_v, 6),
        "yearly": yearly,
        "n_days": len(nav),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--out", default="strategy_blend.json")
    ap.add_argument("--no-panel", action="store_true",
                    help="用 25 公式因子版加速（方向性验证用）")
    args = ap.parse_args()

    end = datetime(2026, 5, 8)
    start = datetime(end.year - args.years, end.month, end.day)
    codes = universe()
    print(f"[blend] 窗口 {start.date()}→{end.date()}  股票池 {len(codes)}", flush=True)

    # 1) 各策略单独跑，拿日度净值（带缓存，避免重跑）
    tag = "25f" if args.no_panel else "42f"
    cache_path = os.path.join(ROOT, "data", "backtest", f"_blend_nav_{tag}.json")
    navs: Dict[str, pd.Series] = {}
    singles: Dict[str, dict] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                _c = json.load(fh)
            for k, v in (_c.get("navs") or {}).items():
                navs[k] = pd.Series(v)
            singles = _c.get("singles") or {}
            print(f"[blend] 用缓存的净值曲线（{len(navs)} 条）: {cache_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[blend] 缓存读取失败: {e}")
            navs = {}
    for name in (STRATS if not navs else []):
        print(f"\n[blend] --- 单跑 {name} ---", flush=True)
        try:
            r = run_strategy(name, codes, start, end, args.no_panel)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ERROR {name}: {e}")
            continue
        singles[name] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(f"  夏普={r.get('sharpe_ratio')}  收益={r.get('total_return')}  "
              f"回撤={r.get('max_drawdown')}  笔数={r.get('total_trades')}  "
              f"({r.get('wall_s')}s)", flush=True)
        if "_nav" in r:
            navs[name] = r["_nav"]
            print(f"  净值点数 {len(r['_nav'])}", flush=True)

    if len(navs) < 2:
        print("[blend] 至少需要 2 条净值曲线")
        return 1

    # 2) 组合
    results = {}
    idx = None
    for s in navs.values():
        idx = s.index if idx is None else idx.union(s.index)
    idx = idx.sort_values()

    for blend in BLENDS:
        keys = [k for k in blend if k in navs]
        if len(keys) != len(blend):
            continue
        # 各自归一到 1，然后按权重做日度收益加权
        rets = []
        wsum = 0.0
        for k in keys:
            w = blend[k]
            nav = navs[k].reindex(idx).ffill()
            base = nav.iloc[0] if nav.iloc[0] else 1.0
            r = (nav / base).pct_change(fill_method=None).fillna(0.0)
            rets.append(w * r)
            wsum += w
        if wsum <= 0:
            continue
        comb = pd.concat(rets, axis=1).sum(axis=1) / wsum
        nav = (1.0 + comb).cumprod()
        nav.index = idx
        m = metrics_from_nav(nav)
        label = "+".join(f"{k}:{blend[k]:.2f}" for k in keys)
        results[label] = m
        print(f"\n[blend] {label}")
        print(f"  夏普={m['sharpe_ratio']:.3f}  收益={m['total_return']*100:+.2f}%  "
              f"回撤={m['max_drawdown']*100:.2f}%  年化={m['annualized_return']*100:+.2f}%")
        if m.get("yearly"):
            print("  分年: " + "  ".join(
                f"{k[-2:]}:{v*100:+.1f}%" for k, v in sorted(m["yearly"].items())))

    try:
        with open(cache_path, "w") as fh:
            json.dump({"navs": {k: {str(d): float(v) for d, v in ser.items()}
                                for k, ser in navs.items()},
                       "singles": singles}, fh)
        print(f"[blend] 净值缓存已写 {cache_path}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[blend] 缓存写入失败: {e}")

    os.makedirs(os.path.dirname(os.path.join(ROOT, "data", "backtest", args.out)),
                exist_ok=True)
    outp = os.path.join(ROOT, "data", "backtest", args.out)
    with open(outp, "w") as fh:
        json.dump({
            "window": f"{start.date()}→{end.date()}",
            "methodology": {
                "lag": 1, "enable_tradability": True, "realistic": True,
                "note": "多策略资金分配合成：各策略独立满仓取日度收益率，"
                        "按资金权重线性组合（数学上等价于真实资金分配）",
            },
            "singles": singles,
            "results": results,
        }, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[blend] 已写入 {outp}", flush=True)
    if results:
        best = max(results.items(), key=lambda kv: kv[1].get("sharpe_ratio") or 0)
        print(f"[blend] 夏普最高: {best[0]} ({best[1]['sharpe_ratio']:.3f}, "
              f"回撤 {best[1]['max_drawdown']*100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
