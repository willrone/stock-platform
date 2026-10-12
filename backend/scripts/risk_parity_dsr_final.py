#!/usr/bin/env python3
"""风险平价组合复算 + DSR/PBO 统计终审（纯 numpy+math, 无 scipy 依赖）。

背景: risk_parity_experiment 跑出 R1(风险平价月度60d) 夏普 1.230 > 单腿最优 0.910,
五轮以来首个真改善。本脚本:
  1) 三腿全市场重跑(确定性引擎), 独立复算组合 stats + 分年(验证自洽)
  2) Bailey-Lopez de Prado 终审:
     DSR = deflated Sharpe, 校正多重试验(N)+偏峰度
     PBO = CSCV 组合交叉验证, 对"同一组合策略的参数变体"(lookback 20/40/60)
           判断参数选择是否过拟合(PBO>0.5 = 过拟合证据)
  3) 单腿/组合/关键信号候选逐个算 DSR
口径: 全市场 5513, 2023-05-12→2026-05-08, 确定性引擎, 无未来函数。
产物: data/strategy_baseline/risk_parity_dsr_final.json
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
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "risk_parity_dsr_final.json")
START, END = datetime(2023, 5, 12), datetime(2026, 5, 8)
LEGS = ["stochastic", "factor_ranking", "bollinger"]
N_TRIALS = 40  # 五轮研究独立试验数(保守上界, 用于 DSR)

SQRT252 = math.sqrt(252.0)

_orig_snapshot = pma_module.PortfolioManagerArray.record_portfolio_snapshot


def install_stash(stash: dict) -> None:
    def patched(self, date, current_prices):
        stash["pm"] = self
        _orig_snapshot(self, date, current_prices)

    pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched


def uninstall() -> None:
    pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snapshot


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _skew_kurt(x: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    m2 = float(((x - mu) ** 2).mean())
    m3 = float(((x - mu) ** 3).mean())
    m4 = float(((x - mu) ** 4).mean())
    skew = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    kurt = m4 / m2 ** 2 if m2 > 0 else 3.0
    return skew, kurt


def equity_to_returns(eq) -> tuple:
    vals = np.array([float(v) for _, v in eq])
    dates = [pd.Timestamp(d) for d, _ in eq]
    base = np.where(vals[:-1] > 0, vals[:-1], 1.0)
    r = np.concatenate([[0.0], np.diff(vals) / base])
    return dates, r


def combo_stats(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    eq = 1e6 * np.cumprod(1.0 + r)
    vals = np.concatenate([[1e6], eq])
    rr = np.diff(vals) / np.where(vals[:-1] > 0, vals[:-1], 1.0)
    years = max(1.0, len(rr) / 252.0)
    total = vals[-1] / vals[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(rr.std(ddof=1) * SQRT252)
    sharpe = float(rr.mean() / rr.std(ddof=1) * SQRT252) if rr.std() > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(((vals - peak) / peak).min())
    return {
        "total_return": round(total, 6),
        "annualized_return": round(ann, 6),
        "volatility": round(vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 6),
    }


def yearly_from_equity(dates: list, vals: np.ndarray) -> dict:
    """分年收益(上年末→本年末) = V[本年末]/V[上年末]-1, 首年基准=1.0。
    vals 与 dates 对齐(净值序列, V[0]=1 对应 r[0]=0)。
    自洽性: 各年(1+yr)累乘 严格等于 total(无跨年歧义)。"""
    vlist = [float(v) for v in vals]
    years: list[int] = []
    last_i: dict[int, int] = {}
    for i, dt in enumerate(dates):
        y = int(pd.Timestamp(dt).year)
        if not years or years[-1] != y:
            years.append(y)
        last_i[y] = i
    out = {}
    prev_end: float = 1.0
    for y in years:
        v = vlist[last_i[y]]
        out[str(y)] = round(v / prev_end - 1.0, 4)
        prev_end = v
    return out


def yearly_returns(dates: list, r: np.ndarray) -> dict:
    """r 为日收益序列(r[0]=0, 长度=len(dates))→ 净值 cumprod(1+r)(与 dates 对齐)→ 分年(上年末口径)。"""
    vals = np.cumprod(1.0 + np.asarray(r, dtype=float))
    return yearly_from_equity(dates, vals)


def annualized_sharpe(daily: np.ndarray) -> float:
    r = np.asarray(daily, dtype=float)[1:]
    mu, sd = r.mean(), r.std(ddof=1)
    return float(mu / sd * SQRT252) if sd > 0 else 0.0


def deflated_sharpe(daily: np.ndarray, n_trials: int = N_TRIALS) -> dict:
    """Bailey-Lopez de Prado (2014) deflated Sharpe, 正态极值近似。"""
    r = np.asarray(daily, dtype=float)[1:]
    T = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr_d = mu / sd if sd > 0 else 0.0
    sr_ann = sr_d * SQRT252
    skew, kurt = _skew_kurt(r)
    var_sr = max(1e-12, (1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2) / T)
    sd_sr = math.sqrt(var_sr)
    sr0 = sd_sr * math.sqrt(2.0 * math.log(max(2, n_trials)))
    dsr = norm_cdf((sr_d - sr0) / sd_sr)
    return {
        "sharpe_annualized": round(sr_ann, 4),
        "n_obs": T,
        "skew": round(skew, 4),
        "kurt": round(kurt, 4),
        "sr0_threshold": round(sr0 * SQRT252, 4),
        "deflated_sharpe": round(dsr, 5),
        "pass_95": bool(dsr > 0.95),
        "n_trials": n_trials,
    }


def pbo_cscv(rets: dict, n_splits: int = 200, seed: int = 7) -> dict:
    """PBO: 200 次随机 50/50 对半分, 前半选优→后半排名。
    PBO>0.5 = 前半选出的"最优"在后半系统性跑输 = 选择过拟合证据。"""
    names = list(rets)
    R = np.stack([np.asarray(rets[n], dtype=float)[1:] for n in names], axis=1)
    T, V = R.shape
    acc = 0.0
    rng = np.random.default_rng(seed)
    for i in range(n_splits):
        idx = rng.permutation(T)
        A, B = R[idx[: T // 2]], R[idx[T // 2 :]]
        sa, sb = A.std(0, ddof=1), B.std(0, ddof=1)
        srA = A.mean(0) / np.where(sa > 0, sa, 1.0) * SQRT252
        srB = B.mean(0) / np.where(sb > 0, sb, 1.0) * SQRT252
        best = int(np.argmax(srA))
        acc += (1 + int((srB > srB[best]).sum())) / V
    pbo = acc / n_splits
    return {
        "pbo": round(float(pbo), 4),
        "n_variants": V,
        "n_splits": n_splits,
        "variants": names,
        "note": "随机50/50对半分; 前半选优→后半排名均值/V; PBO>0.5=过拟合证据",
    }


def risk_parity_w(leg_r: dict, n: int, lookback: int, rebal: int) -> np.ndarray:
    w = np.ones((n, len(LEGS))) / len(LEGS)
    last = np.ones(len(LEGS)) / len(LEGS)
    for t in range(1, n):
        if t % rebal == 0:
            invs = []
            for i, s in enumerate(LEGS):
                win = leg_r[s][max(0, t - lookback): t - 1]  # 止于 t-2 日收益(t-2 收盘已知), t 日权重无未来函数
                v = float(np.std(win, ddof=1) * SQRT252) if len(win) >= 20 else 1.0
                invs.append(1.0 / max(v, 1e-4))
            ssum = sum(invs)
            last = np.array(invs) / ssum
        w[t] = last
    return w


def run_leg(ex, strat, codes, cfg, scfg=None):
    stash: dict = {}
    install_stash(stash)
    try:
        res = asyncio.run(
            ex.run_backtest(
                strategy_name=strat,
                stock_codes=codes,
                start_date=START,
                end_date=END,
                strategy_config=scfg or {},
                backtest_config=cfg,
            )
        )
        pm = stash.get("pm")
        eq = getattr(pm, "equity_curve", None) if pm else None
        return res, eq
    finally:
        uninstall()


def main() -> None:
    codes = rsb.list_full_universe()
    print(f"[universe] {len(codes)} 只, 三腿 {LEGS}", flush=True)
    ex = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    cfg = BacktestConfig(
        initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
        max_position_size=0.2, cash_reserve_ratio=0.05, board_lot_size=100,
        stop_loss_pct=0.05, take_profit_pct=0.15, rebalance_frequency="daily",
        open_cost=0.0, close_cost=0.0, min_cost=0.0,
    )

    out = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "window": [str(START), str(END)],
            "n_trials_for_dsr": N_TRIALS,
            "note": "风险平价复算+DSR/PBO终审; 全市场5513; 确定性引擎; 夏普=μ/σ·√252标准口径(引擎AR/σ口径有bug,见文档); 分年=上年末→本年末净值端点(累乘严格自洽于total)",
        },
        "legs": {},
        "combos": {},
        "returns": {},
        "dsr": {},
        "pbo": {},
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    def save():
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=str)

    # ---- 1) 三腿重跑 ----
    leg_dates = {}
    leg_r = {}
    for s in LEGS:
        t0 = time.perf_counter()
        res, eq = run_leg(ex, s, codes, cfg)
        d, r = equity_to_returns(eq)
        leg_dates[s] = d
        leg_r[s] = r
        out["legs"][s] = {
            "stats": combo_stats(r),
            "yearly": yearly_returns(d, r),
            "engine_reported": {k: res.get(k) for k in ("sharpe_ratio", "total_return", "max_drawdown", "total_trades")},
            "n_days": len(eq),
        }
        out["returns"][f"C_{s}单腿"] = {"dates": [str(pd.Timestamp(x)) for x in d], "daily": [float(x) for x in r]}
        print(f"[leg] {s}: engine_sharpe={res.get('sharpe_ratio')} curve={out['legs'][s]['stats']['sharpe_ratio']} n={len(eq)} wall={time.perf_counter()-t0:.0f}s", flush=True)
    save()

    # 日期一致性(三腿同宇宙同窗口, 应完全一致)
    base_dates = leg_dates[LEGS[0]]
    for s in LEGS:
        assert len(leg_dates[s]) == len(base_dates), f"{s} 日期数不一致"
    n = len(base_dates)
    print(f"[align] {n} 天, 三腿日期一致", flush=True)

    # ---- 2) 组合: R0 等权 + R1 lookback 20/40/60 ----
    w0 = np.ones((n, 3)) / 3
    r0 = np.sum([leg_r[s] * w0[:, i] for i, s in enumerate(LEGS)], axis=0)
    out["combos"]["R0_等权"] = {
        "stats": combo_stats(r0),
        "yearly": yearly_returns(base_dates, r0),
    }
    combo_rets = {"R0_等权": r0}
    for lb in (20, 40, 60):
        w = risk_parity_w(leg_r, n, lookback=lb, rebal=21)
        rc = np.sum([leg_r[s] * w[:, i] for i, s in enumerate(LEGS)], axis=0)
        combo_rets[f"R1_lb{lb}"] = rc
        out["combos"][f"R1_lb{lb}"] = {
            "stats": combo_stats(rc),
            "yearly": yearly_returns(base_dates, rc),
            "avg_weights": {s: round(float(w[int(n * 0.1):, i].mean()), 3) for i, s in enumerate(LEGS)},
        }
    for tag, c in out["combos"].items():
        s = c["stats"]
        print(f"[{tag}] sharpe={s['sharpe_ratio']} ann={s['annualized_return']} mdd={s['max_drawdown']} yearly={c['yearly']}", flush=True)
    save()

    # ---- 3) DSR: 单腿 + 组合 ----
    for s in LEGS:
        ds = deflated_sharpe(leg_r[s])
        out["dsr"][f"C_{s}单腿"] = ds
        print(f"[DSR] {s}单腿: DSR={ds['deflated_sharpe']} pass95={ds['pass_95']} sharpe={ds['sharpe_annualized']}", flush=True)
    for tag, rr in combo_rets.items():
        ds = deflated_sharpe(rr)
        out["dsr"][tag] = ds
        out["returns"][tag] = {"dates": [str(pd.Timestamp(d)) for d in base_dates], "daily": [float(x) for x in rr]}
        print(f"[DSR] {tag}: DSR={ds['deflated_sharpe']} pass95={ds['pass_95']}", flush=True)
    save()

    # ---- 4) PBO: 组合参数变体(lookback 20/40/60 + 等权) ----
    out["pbo"] = pbo_cscv(combo_rets)
    print(f"[PBO] {out['pbo']['pbo']} across {out['pbo']['n_variants']} 变体 {out['pbo']['variants']}", flush=True)
    save()

    # ---- 5) DSR: 关键信号候选(单腿市场口径) ----
    extra = [
        ("G3_35_65_k10", "stochastic", {"oversold": 35, "overbought": 65, "k_period": 10, "d_period": 3}),
        ("sig_15_85_k14", "stochastic", {"oversold": 15, "overbought": 85, "k_period": 14, "d_period": 3}),
        ("macd", "macd", None),
        ("obv", "obv", None),
        ("kdj", "kdj", None),
    ]
    for tag, strat, scfg in extra:
        t0 = time.perf_counter()
        res, eq = run_leg(ex, strat, codes, cfg, scfg)
        d, r = equity_to_returns(eq)
        ds = deflated_sharpe(r)
        out["dsr"][tag] = {**ds, "universe": "full-market", "engine_sharpe": res.get("sharpe_ratio")}
        print(f"[DSR] {tag}: DSR={ds['deflated_sharpe']} pass95={ds['pass_95']} sharpe={ds['sharpe_annualized']} wall={time.perf_counter()-t0:.0f}s", flush=True)
        save()

    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
