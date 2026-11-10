#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P14 · 袖套可优化空间的上界（Oracle 上界实验）

**这个实验不找"更好的方案"，只回答一个问题：袖套内部到底还剩多少可优化空间。**

背景（P13）：
  - 袖套层权重优化（ERC/逆波动/最小方差）在袖套层真有效（5/5 窗口夏普全升），
    在组合层全败（main 1.4114 → 1.3126）。
  - 根因：纯方差目标不含 μ，必然全压 bond → 丢收益 + 丢分散化。
  - 随机族最优成员在 main/oos 的组合夏普是 1.5318 / 1.1916，
    连"类等权"基线（1.4114 / 1.0691）都只是打赢随机族**均值**，没打赢最优成员。

所以在设计任何新优化器之前，先把**上界**算出来。

三层上界（全部含未来信息，**不可实现，仅用于定标**）：

| 代号 | 含义 | 自由参数 | 过拟合程度 |
|---|---|---|---|
| `oracle_s` | 每个调仓日用**未来 252 个交易日**的 μ/Σ 解出使组合夏普最大的袖套权重 | 9 | 低（用聚合统计，不用逐期噪声） |
| `oracle_j` | 同上，且 **A 股腿权重放开**（0.05~0.95） | 9 + 1 | 低 |
| `oracle_h` | 每个持有期（21 天）内**事后**最大化已实现组合夏普 | 9 | 极高（21 个观测拟 9 个参数） |

判决（相对随机族最优成员 1.5318 / 1.1916）：
  U < 1.53            → 空间不存在，正式封死袖套优化
  1.53 ≤ U < 1.8     → 空间小，资源转向 A 股腿
  U ≥ 1.8            → 空间大，值得研究如何逼近

⚠️ **Oracle 用未来信息，它的夏普必然被高估。**
   它只能用于「空间存在性」判据，**绝不能与任何可实现方案比大小，更不能进定稿。**

用法：
    env -u PYTHONPATH .venv-py313/bin/python3 scripts/multi_asset_p14_oracle_bound.py \
        --phase ab --window main
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p13_asset_weight as P13  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    END, OOS_END, OOS_START, START, maxdd, merge_by_date, sharpe,
)

OUT_DIR = os.path.join(ROOT, "data", "backtest")

P5_FIXED = P13.P5_FIXED
CLASSES = P13.CLASSES
REB_EVERY = P13.REB_EVERY
A_LEG_W = P13.A_LEG_W

ORACLE_HORIZON = 252      # 统计 oracle 的前瞻交易日数（1 年）
MIN_ORACLE_OBS = 60       # oracle 估计的最少观测
SINGLE_CAP = 0.40         # 单标的权重上限（与 P4 的 cap40 同量级，防止病态解）
A_LEG_LO, A_LEG_HI = 0.05, 0.95

RANDOM_BEST_MAIN = 1.5318   # P13 rand_code_dyn_s7
RANDOM_BEST_OOS = 1.1916    # P13 rand_code_s7
CURRENT_MAIN = 1.4114       # P3/P4/P12/P13 定稿
CURRENT_OOS = 1.0691


# ───────────────────────── 优化器 ─────────────────────────

def _combined_stats(w: np.ndarray, w_a: float,
                    mu: np.ndarray, Sigma: np.ndarray, c: np.ndarray,
                    mu_a: float, var_a: float) -> Tuple[float, float]:
    """给定袖套权重 w（单纯形上）与 A 腿权重 w_a，返回组合的 (μ, σ)（日频）。"""
    mu_s = float(w @ mu)
    var_s = float(w @ Sigma @ w)
    if var_s < 0.0:
        var_s = 0.0
    cov_sa = float(w @ c)
    mu_c = w_a * mu_a + (1.0 - w_a) * mu_s
    var_c = (w_a ** 2) * var_a + ((1.0 - w_a) ** 2) * var_s \
        + 2.0 * w_a * (1.0 - w_a) * cov_sa
    return mu_c, math.sqrt(max(var_c, 1e-18))


def _sharpe_obj(w: np.ndarray, w_a: float,
                mu: np.ndarray, Sigma: np.ndarray, c: np.ndarray,
                mu_a: float, var_a: float) -> float:
    mu_c, sd_c = _combined_stats(w, w_a, mu, Sigma, c, mu_a, var_a)
    if sd_c <= 0:
        return -1e9
    return mu_c / sd_c


def _solve_oracle(mu: np.ndarray, Sigma: np.ndarray, c: np.ndarray,
                  mu_a: float, var_a: float,
                  free_a_leg: bool,
                  a_leg_fixed: float = A_LEG_W) -> Optional[np.ndarray]:
    """在（袖套权重 + 可选 A 腿权重）上最大化组合夏普。

    返回 x：free_a_leg=False → x 是 9 维袖套权重（和为 1）；
            free_a_leg=True  → x 是 10 维（前 9 维袖套权重和为 1，第 10 维是 A 腿权重）。
    """
    try:
        from scipy.optimize import minimize
    except Exception as e:            # pragma: no cover
        raise SystemExit(f"[p14] ❌ 缺 scipy: {e}")

    n = len(mu)
    cons = [{"type": "eq", "fun": lambda x: float(np.sum(x[:n]) - 1.0)}]
    bounds = [(0.0, SINGLE_CAP)] * n
    if free_a_leg:
        bounds.append((A_LEG_LO, A_LEG_HI))
        x0 = np.concatenate([np.full(n, 1.0 / n), [a_leg_fixed]])
    else:
        x0 = np.full(n, 1.0 / n)

    def obj(x):
        w = x[:n]
        wa = x[n] if free_a_leg else a_leg_fixed
        return -_sharpe_obj(w, wa, mu, Sigma, c, mu_a, var_a)

    best = None
    for start in (x0,):
        try:
            res = minimize(obj, start, method="SLSQP", bounds=bounds,
                           constraints=cons,
                           options={"maxiter": 400, "ftol": 1e-10})
        except Exception:
            continue
        if res is None:
            continue
        x = np.asarray(res.x, dtype=float)
        x[:n] = np.clip(x[:n], 0.0, SINGLE_CAP)
        s = x[:n].sum()
        if s <= 1e-9:
            continue
        x[:n] = x[:n] / s
        if free_a_leg:
            x[n] = float(np.clip(x[n], A_LEG_LO, A_LEG_HI))
        val = -obj(x)
        if best is None or val > best[0]:
            best = (val, x)
    if best is None:
        return None
    return best[1]


def _hindsight_oracle_period(ret_s: np.ndarray, ret_a: np.ndarray,
                             free_a_leg: bool) -> Optional[np.ndarray]:
    """逐持有期事后最优：在**已实现**的持有期收益上直接最大化组合夏普。

    ret_s: (T, 9) 持有期内各袖套标的的日收益
    ret_a: (T,)   持有期内 A 腿日收益
    这是"任何月度调仓规则的理论上限"，过拟合程度极高，仅作天花板参考。
    """
    n = ret_s.shape[1]
    if ret_s.shape[0] < 5:
        return None
    mu = ret_s.mean(axis=0)
    Sigma = np.cov(ret_s, rowvar=False)
    Sigma = np.atleast_2d(Sigma)
    c = np.array([np.cov(ret_s[:, j], ret_a)[0, 1] for j in range(n)])
    mu_a = float(ret_a.mean())
    var_a = float(np.var(ret_a, ddof=1))

    def obj(x):
        w = x[:n]
        wa = x[n] if free_a_leg else A_LEG_W
        mu_c, sd_c = _combined_stats(w, wa, mu, Sigma, c, mu_a, var_a)
        return -mu_c / sd_c if sd_c > 0 else 1e9

    cons = [{"type": "eq", "fun": lambda x: float(np.sum(x[:n]) - 1.0)}]
    bounds = [(0.0, 1.0)] * n
    if free_a_leg:
        bounds.append((0.0, 1.0))
        x0 = np.concatenate([np.full(n, 1.0 / n), [A_LEG_W]])
    else:
        x0 = np.full(n, 1.0 / n)
    from scipy.optimize import minimize
    try:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 400, "ftol": 1e-12})
    except Exception:
        return None
    if res is None:
        return None
    x = np.asarray(res.x, dtype=float)
    x[:n] = np.clip(x[:n], 0.0, None)
    s = x[:n].sum()
    if s <= 1e-9:
        return None
    x[:n] = x[:n] / s
    if free_a_leg:
        x[n] = float(np.clip(x[n], 0.0, 1.0))
    return x


# ───────────────────────── 数据装配 ─────────────────────────

def load_window(window: str) -> Dict[str, Any]:
    start, end = (START, END) if window == "main" else (OOS_START, OOS_END)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    held = P5_FIXED
    codes = [c for lst in held.values() for c in lst]
    class_of = {c: cl for cl, lst in held.items() for c in lst}

    # 日历口径必须与 P3/P6/P13 逐位一致：
    #   日历按**基线候选并集**建，价格索引覆盖全池（基线候选 ∪ P3 池）
    pool = P3.build_pool(start)
    universe = L3.load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    if len(calendar) < 300:
        raise SystemExit(f"[p14] ❌ 交易日历过短 {len(calendar)}，退出")
    R, rdates = P13.build_returns(calendar, prices, codes)

    # 估计矩阵（与 P13 同口径）：窗口前 500 自然日 + 窗口内
    est_start = (start - __import__("datetime").timedelta(days=500)).strftime("%Y-%m-%d")
    est_cal = sorted({d for c in codes for d in prices.get(c, {})
                      if est_start <= d <= e_str} | set(calendar))
    R_est, _ = P13.build_returns(est_cal, prices, codes)

    # A 腿 NAV（复用缓存，与 P3/P4/P12/P13 同源）
    cache = os.path.join(OUT_DIR, f"_a_share_nav_{window}.json")
    if not os.path.exists(cache):
        raise SystemExit(f"[p14] ❌ 找不到 A 腿缓存 {cache}")
    with open(cache, encoding="utf-8") as fh:
        blob = json.load(fh)
    a_nav = [(d, float(v)) for d, v in blob["a_share_nav"]]

    # 把 A 腿日收益对齐到 calendar 的「前值→当日」结构（与 R[k] 同口径）
    amap = dict(a_nav)
    a_ret = np.full(len(calendar), np.nan)
    for k in range(1, len(calendar)):
        d0, d1 = calendar[k - 1], calendar[k]
        if d0 in amap and d1 in amap and amap[d0] > 0:
            a_ret[k] = amap[d1] / amap[d0] - 1.0
    # A 腿日收益对齐到 est_cal（缺失为 NaN：窗口前 A 腿 NAV 不可得）
    a_ret_est = np.full(len(est_cal), np.nan)
    for k in range(1, len(est_cal)):
        d0, d1 = est_cal[k - 1], est_cal[k]
        if d0 in amap and d1 in amap and amap[d0] > 0:
            a_ret_est[k] = amap[d1] / amap[d0] - 1.0

    return {
        "window": window, "start": s_str, "end": e_str, "held": held,
        "codes": codes, "class_of": class_of, "calendar": calendar,
        "R": R, "a_nav": a_nav, "a_ret": a_ret,
        "est_cal": est_cal, "R_est": R_est, "a_ret_est": a_ret_est,
        "est_pos": {d: i for i, d in enumerate(est_cal)},
        "prices": prices,
    }


def oracle_schedule(w: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """mode:
      oracle_s  统计 oracle，袖套权重自由（用未来 252 天）
      oracle_j  统计 oracle + A 腿权重自由（用未来 252 天）
      oracle_h  逐持有期事后最优（用未来 21 天）
      past_s    **可实现对照**：同样的优化器，但用信号日**之前**的 252 天
      past_j    **可实现对照**：past_s + A 腿权重自由
    """
    calendar = w["calendar"]
    R = w["R"]
    a_ret = w["a_ret"]
    n = len(calendar)
    codes = w["codes"]
    free_a = mode in ("oracle_j", "past_j")
    use_past = mode.startswith("past")
    if use_past:
        R = w["R_est"]
        a_ret = w["a_ret_est"]

    sched: Dict[str, Dict[str, float]] = {}      # signal_date → {code: w}
    a_sched: Dict[str, float] = {}               # signal_date → A 腿权重
    fallback = 0
    stats = {"n_solved": 0, "n_fallback": 0, "sharpe_est": []}

    for i in range(1, n):
        if (i - 1) % REB_EVERY != 0:
            continue
        signal_date = calendar[i - 1]
        if use_past:
            # 可实现对照：估计窗口 = 信号日之前（est_cal 口径）
            p = w["est_pos"].get(signal_date)
            if p is None:
                continue
            hi = p
            lo = max(0, p - ORACLE_HORIZON)
        else:
            lo = i - 1                              # 未来收益从信号日收盘开始
            if mode == "oracle_h":
                hi = min(n - 1, lo + REB_EVERY)      # 持有期 = 21 天
            else:
                hi = min(n - 1, lo + ORACLE_HORIZON)  # 统计 oracle = 1 年前瞻
        min_obs = 10 if mode == "oracle_h" else MIN_ORACLE_OBS
        if hi - lo < min_obs:
            # 观测不足 → 退回等权（记录）
            cw = {c: 1.0 / len(codes) for c in codes}
            sched[signal_date] = cw
            a_sched[signal_date] = A_LEG_W
            fallback += 1
            continue

        seg_s = R[lo:hi]                            # (T, 9)
        seg_a = a_ret[lo:hi]                        # (T,)
        ok = ~np.isnan(seg_a)
        seg_s = seg_s[ok]
        seg_a = seg_a[ok]
        if seg_s.shape[0] < min_obs:
            cw = {c: 1.0 / len(codes) for c in codes}
            sched[signal_date] = cw
            a_sched[signal_date] = A_LEG_W
            fallback += 1
            continue

        if mode == "oracle_h":
            x = _hindsight_oracle_period(seg_s, seg_a, free_a_leg=False)
            if x is None:
                cw = {c: 1.0 / len(codes) for c in codes}
                sched[signal_date] = cw
                a_sched[signal_date] = A_LEG_W
                fallback += 1
                continue
            wv = x[:len(codes)]
            wa = A_LEG_W
        else:
            mu = np.nanmean(seg_s, axis=0)
            Sigma = np.cov(seg_s, rowvar=False)
            Sigma = np.atleast_2d(Sigma)
            Sigma = np.nan_to_num(Sigma, nan=0.0)
            cvec = np.array([np.cov(seg_s[:, j], seg_a)[0, 1]
                             for j in range(len(codes))])
            mu_a = float(np.mean(seg_a))
            var_a = float(np.var(seg_a, ddof=1))
            x = _solve_oracle(mu, Sigma, cvec, mu_a, var_a, free_a_leg=free_a)
            if x is None:
                cw = {c: 1.0 / len(codes) for c in codes}
                sched[signal_date] = cw
                a_sched[signal_date] = A_LEG_W
                fallback += 1
                continue
            wv = x[:len(codes)]
            wa = float(x[len(codes)]) if free_a else A_LEG_W
            stats["sharpe_est"].append(
                float(_sharpe_obj(wv, wa, mu, Sigma, cvec, mu_a, var_a)
                      * math.sqrt(252)))

        sched[signal_date] = {c: float(wv[j]) for j, c in enumerate(codes)}
        a_sched[signal_date] = wa
        stats["n_solved"] += 1

    stats["n_fallback"] = fallback
    return {"sched": sched, "a_sched": a_sched, "stats": stats}


def merge_variable(a_nav: List[Tuple[str, float]],
                   s_nav: List[Tuple[str, float]],
                   a_sched: Dict[str, float],
                   exec_of: Dict[str, str]) -> List[Tuple[str, float]]:
    """按时间变动的 A 腿权重合并（oracle_j 用）。权重按调仓日阶跃。"""
    smap = dict(s_nav)
    amap = dict(a_nav)
    adates = [d for d, _ in a_nav]
    keys = sorted(set(adates) & set(smap))
    out: List[Tuple[str, float]] = []
    cur_a = A_LEG_W
    cur_sig = None
    sig_sorted = sorted(exec_of.keys())
    for d in keys:
        # 找到 d 之前最近一次调仓（按 exec 日期）
        while sig_sorted:
            sig = sig_sorted[0]
            if exec_of[sig] <= d:
                cur_sig = sig
                sig_sorted.pop(0)
            else:
                break
        if cur_sig is not None:
            cur_a = a_sched.get(cur_sig, A_LEG_W)
        av = amap[d]
        sv = smap[d]
        out.append((d, cur_a * av + (1.0 - cur_a) * sv))
    return out


def _rebal_dates(calendar: List[str]) -> List[Dict[str, str]]:
    """与 P13.build_targets_code 完全同口径的调仓信号日列表。"""
    out = []
    for i in range(1, len(calendar)):
        if (i - 1) % REB_EVERY != 0:
            continue
        out.append({"date": calendar[i - 1], "exec": calendar[i]})
    return out


def evaluate(w: Dict[str, Any], sched: Dict[str, Dict[str, float]],
             a_sched: Dict[str, float]) -> Dict[str, Any]:
    """把 oracle 权重表喂进 P13 的撮合器，得到袖套 NAV，再合并 A 腿。"""
    calendar = w["calendar"]
    codes = w["codes"]
    idx = P13.build_price_index(w["prices"])

    def weight_fn(signal_date: str, _i: int) -> Optional[Dict[str, float]]:
        cw = sched.get(signal_date)
        if not cw:
            return None
        s = sum(cw.values())
        if s <= 0:
            return None
        return {c: v / s for c, v in cw.items() if v > 0}

    targets, wmeta = P13.build_targets_code(calendar, weight_fn, REB_EVERY)
    s_nav, rec = P13.simulate_portfolio(calendar, idx, targets)
    exec_of = {row["date"]: row["exec"] for row in wmeta["weight_log"]}

    if all(abs(a_sched.get(k, A_LEG_W) - A_LEG_W) < 1e-9 for k in a_sched):
        comb = merge_by_date([w["a_nav"], s_nav], [A_LEG_W, 1.0 - A_LEG_W])
    else:
        comb = merge_variable(w["a_nav"], s_nav, a_sched, exec_of)

    # 权重统计
    cls_avg = {c: 0.0 for c in CLASSES}
    class_of = w["class_of"]
    rows = wmeta["weight_log"]
    for row in rows:
        for code, wt in row["code_weights"].items():
            cls_avg[class_of[code]] += wt / max(1, len(rows))
    a_avg = sum(a_sched.get(r["date"], A_LEG_W) for r in rows) / max(1, len(rows))

    return {
        "sleeve_sharpe": round(sharpe(s_nav), 4),
        "sleeve_return": round(P13.nav_return(s_nav), 6),
        "sleeve_maxdd": round(maxdd(s_nav), 6),
        "combined_sharpe": round(sharpe(comb), 4),
        "combined_return": round(P13.nav_return(comb), 6),
        "combined_maxdd": round(maxdd(comb), 6),
        "turnover_ratio": rec["turnover_ratio"],
        "n_rebalances": rec["n_rebalances"],
        "class_weights_avg": {k: round(v, 4) for k, v in cls_avg.items()},
        "a_leg_weight_avg": round(a_avg, 4),
    }


def run_ab(window: str) -> Dict[str, Any]:
    w = load_window(window)
    out: Dict[str, Any] = {
        "phase": "ab", "window": window, "start": w["start"], "end": w["end"],
        "held": w["held"], "codes": w["codes"], "n_calendar": len(w["calendar"]),
        "oracle_horizon": ORACLE_HORIZON, "single_cap": SINGLE_CAP,
        "note": "Oracle 用未来信息 → 夏普必然被高估；只用于「空间存在性」判据，"
                "不得与可实现方案比大小，不得进定稿。",
        "results": {},
    }
    cur = CURRENT_MAIN if window == "main" else CURRENT_OOS
    rnd = RANDOM_BEST_MAIN if window == "main" else RANDOM_BEST_OOS
    out["reference"] = {"current_final_sharpe": cur, "random_best_sharpe": rnd}

    # ── 基线复现门：等权调度走同一条撮合/合并路径，必须逐位复现 P3 定稿 ──
    eq_sched = {r["date"]: {c: 1.0 / len(w["codes"]) for c in w["codes"]}
                for r in _rebal_dates(w["calendar"])}
    base_m = evaluate(w, eq_sched, {k: A_LEG_W for k in eq_sched})
    out["baseline_gate"] = {"expected": cur, "got": base_m["combined_sharpe"],
                             "ok": abs(base_m["combined_sharpe"] - cur) <= 5e-4}
    print(f"[p14] {window} 基线复现门: base_equal 组合夏普 "
          f"{base_m['combined_sharpe']}（定稿 {cur}）"
          f" → {'✅' if out['baseline_gate']['ok'] else '❌'}", flush=True)
    if not out["baseline_gate"]["ok"]:
        raise SystemExit("[p14] ❌ 基线未复现，全部 oracle 结果不可比，退出")
    out["baseline_metrics"] = base_m

    for mode in ("oracle_s", "oracle_j", "oracle_h", "past_s", "past_j"):
        sc = oracle_schedule(w, mode)
        m = evaluate(w, sc["sched"], sc["a_sched"])
        m["oracle_stats"] = {k: v for k, v in sc["stats"].items() if k != "sharpe_est"}
        m["oracle_est_sharpe_mean"] = (
            round(float(np.mean(sc["stats"]["sharpe_est"])), 4)
            if sc["stats"]["sharpe_est"] else None)
        m["vs_current"] = round(m["combined_sharpe"] - cur, 4)
        m["vs_random_best"] = round(m["combined_sharpe"] - rnd, 4)
        out["results"][mode] = m
        print(f"[p14] {window} {mode:10s} 袖套夏普 {m['sleeve_sharpe']:8.4f} "
              f"| 组合夏普 {m['combined_sharpe']:8.4f} "
              f"(vs 定稿 {m['vs_current']:+.4f}, vs 随机最优 {m['vs_random_best']:+.4f}) "
              f"| A腿权重均值 {m['a_leg_weight_avg']:.3f} "
              f"| 类权重 {m['class_weights_avg']} "
              f"| 换手 {m['turnover_ratio']:.2f}x "
              f"| fallback {m['oracle_stats']['n_fallback']}", flush=True)

    # 判决
    U_s = out["results"]["oracle_s"]["combined_sharpe"]
    U_j = out["results"]["oracle_j"]["combined_sharpe"]
    U_h = out["results"]["oracle_h"]["combined_sharpe"]
    if U_s < rnd:
        verdict = "空间不存在（统计 oracle 都打不过随机族最优成员）→ 正式封死袖套优化"
    elif U_s < 1.8:
        verdict = "空间小（≤1.8）→ 资源转向 A 股腿"
    else:
        verdict = "空间大（≥1.8）→ 值得研究如何逼近上界"
    out["verdict"] = {
        "U_oracle_sleeve_only": U_s, "U_oracle_joint": U_j,
        "U_oracle_hindsight": U_h, "threshold_random_best": rnd,
        "past_sleeve_sharpe": out["results"]["past_s"]["combined_sharpe"],
        "past_joint_sharpe": out["results"]["past_j"]["combined_sharpe"],
        "oracle_minus_past": round(U_s - out["results"]["past_s"]["combined_sharpe"], 4),
        "verdict": verdict,
    }
    print(f"[p14] {window} 判决: U_sleeve={U_s} U_joint={U_j} U_hindsight={U_h} "
          f"| {verdict}", flush=True)

    path = os.path.join(OUT_DIR, f"multi_asset_p14_{window}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"[p14] ✓ 已写入 {path}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="ab", choices=["ab"])
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    args = ap.parse_args()
    run_ab(args.window)


if __name__ == "__main__":
    main()
