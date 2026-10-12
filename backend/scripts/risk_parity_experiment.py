#!/usr/bin/env python3
"""风险平价配置实验 (stochastic + factor_ranking + bollinger 三腿)。

背景: 五轮夏普研究全部证伪(止损/overlay/组合vol/信号网格) → 唯一未试的
结构性杠杆: 横截面配权(非择时)。三腿来自确定性基线的风险调整前三:
  stochastic    夏普 0.81 / 年化 26.6% / 回撤 -26.1%  (动量振荡, 高波)
  factor_ranking 夏普 0.92 / 年化 11.5% / 回撤 -10.6%  (横截面, 低波)
  bollinger     夏普 0.61 / 年化 21.7% / 回撤 -27.2%  (通道, 中波)
两两相关性低(基线 vs_benchmark correlation 0.3-0.35 区间), 风险平价预期:
在组合层面压波动, 看能否把组合夏普推过单腿天花板 0.81。

机制(纯组合层, 零引擎改动):
  每腿独立跑全市场 5513 回测(确定性引擎) → 取净值曲线
  每日组合收益 = Σ w_i(t) × r_i(t)
  w 在再平衡日按 1/近60日已实现波动 归一化, 月度(21交易日)再平衡
  权重只用 ≤t 数据 → 无未来函数; 再平衡成本忽略(单腿 0.04%×12次/年, 可忽略)

变体:
  R0 静态等权 1/3×3          (对照: diversification 基线)
  R1 风险平价 月度60d        (主实验)
  R2 风险平价 月度20d        (更敏感, 看平台/刀锋)
  C1 stochastic 单腿         (天花板对照 0.81)
  C2 factor_ranking 单腿     (0.92)

口径: 全市场 5513(与基线同宇宙, 避 500 宇宙失真), 2023-05-12→2026-05-08, 100万。
产物: data/strategy_baseline/risk_parity_experiment.json
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_ROOT)

_spec = importlib.util.spec_from_file_location(
    "rsb", os.path.join(BACKEND_ROOT, "scripts", "run_strategy_baseline.py")
)
assert _spec is not None
rsb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rsb)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.core import portfolio_manager_array as pma_module  # noqa: E402

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "risk_parity_experiment.json")
START, END = datetime(2023, 5, 12), datetime(2026, 5, 8)

LEGS = ["stochastic", "factor_ranking", "bollinger"]

_orig_snapshot = pma_module.PortfolioManagerArray.record_portfolio_snapshot


def install_stash(stash: dict) -> None:
    def patched(self, date, current_prices):
        stash["pm"] = self
        _orig_snapshot(self, date, current_prices)

    pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched


def uninstall() -> None:
    pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snapshot


def run_leg(ex, strat: str, codes: list, cfg) -> list:
    stash: dict = {}
    install_stash(stash)
    try:
        asyncio.run(
            ex.run_backtest(
                strategy_name=strat,
                stock_codes=codes,
                start_date=START,
                end_date=END,
                strategy_config={},
                backtest_config=cfg,
            )
        )
        pm = stash.get("pm")
        eq = getattr(pm, "equity_curve", None) if pm else None
        assert eq, f"{strat}: no equity curve captured"
        return eq
    finally:
        uninstall()


def _align(equities: list) -> tuple:
    """日期对齐: 返回 (dates, {leg: np.array values})。缺失日沿用上一值。"""
    maps = {}
    for s, eq in zip(LEGS, equities):
        m = {}
        for d, v in eq:
            dd = d.date() if hasattr(d, "date") else d
            m[dd] = float(v)
        maps[s] = m
    dates = sorted(set.intersection(*[set(m) for m in maps.values()]))
    if not dates:
        raise RuntimeError("leg equity curves share no dates")
    vals = {}
    for s, m in maps.items():
        arr = np.array([m.get(d, np.nan) for d in dates], dtype=float)
        arr = pd.Series(arr).ffill().to_numpy()
        vals[s] = arr
    return [pd.Timestamp(d) for d in dates], vals


def _ret_stats_from_returns(r: np.ndarray, init=1_000_000.0):
    eq = init * np.cumprod(1.0 + r)
    vals = np.concatenate([[init], eq])
    base = np.where(vals[:-1] > 0, vals[:-1], 1.0)
    rets = np.diff(vals) / base
    years = max(1.0, len(rets) / 252.0)
    total = vals[-1] / vals[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(rets.std() * math.sqrt(252))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(((vals - peak) / peak).min())
    return {
        "total_return": round(total, 6),
        "annualized_return": round(ann, 6),
        "volatility": round(vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 6),
    }


def _yearly_from_returns(r: np.ndarray, dates: list):
    by = {}
    cum = 1.0
    for d, ri in zip(dates, r):
        y = int(d.year) if hasattr(d, "year") else int(str(d)[:4])
        cum *= 1 + ri
        by.setdefault(y, [cum / (1 + ri), cum])  # start, running end
    return {str(y): round(end / start - 1.0, 4) for y, (start, end) in by.items()}


def risk_parity_weights(vals: dict, dates: list, lookback: int, rebal: int) -> np.ndarray:
    """返回 (n_days, n_legs) 权重矩阵, t 日权重由 t-lookback..t 的波动决定, 每 rebal 日更新。"""
    names = LEGS
    n = len(dates)
    rets = {s: np.diff(vals[s]) / np.where(vals[s][:-1] > 0, vals[s][:-1], 1.0) for s in names}
    w = np.zeros((n, len(names)))
    last_w = np.ones(len(names)) / len(names)
    for t in range(n):
        if t > 0 and (t % rebal == 0):
            invs = []
            for s in names:
                # 严格无未来函数: t 日开盘前决策, vol 只用 ≤t-1 日收益
                window = rets[s][max(0, t - lookback): t - 1]
                v = float(np.std(window) * math.sqrt(252)) if len(window) >= 20 else 1.0
                invs.append(1.0 / max(v, 1e-4))
            ssum = sum(invs)
            last_w = np.array(invs) / ssum
        w[t] = last_w
    return w


def combine(vals: dict, dates: list, w: np.ndarray):
    names = LEGS
    n = len(dates)
    r = np.zeros(n)
    for i, s in enumerate(names):
        leg_r = np.zeros(n)
        leg_r[1:] = np.diff(vals[s]) / np.where(vals[s][:-1] > 0, vals[s][:-1], 1.0)
        r += leg_r * w[:, i]
    return r


def main() -> None:
    codes = rsb.list_full_universe()
    print(f"[universe] 全市场 {len(codes)} 只, 三腿: {LEGS}", flush=True)
    ex = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    cfg = BacktestConfig(
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        max_position_size=0.2,
        cash_reserve_ratio=0.05,
        board_lot_size=100,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        rebalance_frequency="daily",
        open_cost=0.0,
        close_cost=0.0,
        min_cost=0.0,
    )

    # ---- 跑三腿 ----
    leg_eqs = {}
    for s in LEGS:
        t0 = time.perf_counter()
        leg_eqs[s] = run_leg(ex, s, codes, cfg)
        print(f"[leg] {s}: n={len(leg_eqs[s])} wall={time.perf_counter()-t0:.0f}s", flush=True)

    dates, vals = _align([leg_eqs[s] for s in LEGS])
    print(f"[align] 共同交易日 {len(dates)}: {dates[0].date()} → {dates[-1].date()}", flush=True)

    results: dict = {}

    def save():
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "window": [str(START), str(END)],
                "universe": "full-market-5513",
                "legs": LEGS,
                "note": "风险平价配置: 月度60d/20d 逆波动配权; 纯组合层, 无未来函数",
            },
            "results": results,
        }
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)

    # R0 静态等权
    w0 = np.ones((len(dates), 3)) / 3
    r0 = combine(vals, dates, w0)
    rs0 = _ret_stats_from_returns(r0)
    results["R0_等权13"] = {"stats": rs0, "yearly": _yearly_from_returns(r0, dates)}
    print(f"[R0] 等权: sharpe={rs0['sharpe_ratio']} ann={rs0['annualized_return']} mdd={rs0['max_drawdown']}", flush=True)
    save()

    # R1/R2 风险平价
    for tag, lb, rb in (("R1_风险平价月度60d", 60, 21), ("R2_风险平价月度20d", 20, 21)):
        w = risk_parity_weights(vals, dates, lookback=lb, rebal=rb)
        rc = combine(vals, dates, w)
        rsc = _ret_stats_from_returns(rc)
        # 平均权重(后 90% 区间, 排除冷启动)
        avg_w = {s: float(w[int(len(w) * 0.1):, i].mean()) for i, s in enumerate(LEGS)}
        results[tag] = {
            "lookback": lb,
            "rebal_days": rb,
            "stats": rsc,
            "avg_weights": avg_w,
            "yearly": _yearly_from_returns(rc, dates),
        }
        print(
            f"[{tag}] sharpe={rsc['sharpe_ratio']} ann={rsc['annualized_return']} "
            f"mdd={rsc['max_drawdown']} avg_w={ {k: round(v,3) for k,v in avg_w.items()} }",
            flush=True,
        )
        save()

    # C1/C2 单腿对照(直接取腿统计)
    for s in ("stochastic", "factor_ranking"):
        arr = vals[s]
        lr = np.zeros(len(arr))
        lr[1:] = np.diff(arr) / np.where(arr[:-1] > 0, arr[:-1], 1.0)
        rsc = _ret_stats_from_returns(lr)
        results[f"C_{s}单腿"] = {"stats": rsc, "yearly": _yearly_from_returns(lr, dates)}
        print(f"[C_{s}] sharpe={rsc['sharpe_ratio']} ann={rsc['annualized_return']} mdd={rsc['max_drawdown']}", flush=True)
        save()

    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
