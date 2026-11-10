#!/usr/bin/env python3
"""P13 · 袖套标的层配权 —— 「类等权 + 类内等权」是中性默认吗？

回答 `P5_CLASS_AUDIT_2026-10-31.md` §5 留下的开放问题：

    把三类摊开看**实际的风险源数**：gold 3 只 ρ≈0.99 → **1** 个；
    bond 3 只长久期利率债 → **1** 个；commodity 3 只 ρ 0.11~0.62 → **3** 个。
    当前定稿给三类**同等权重**（各 16.67%），但它们承载的独立风险源数是 1/1/3。
    也就是说「类内等权 + 类间等权」不是中性默认结构，它隐含了
    **「每类等权 = 每类风险相等」的错误假设**。

P5 给出的下一步是「资产层面的风险平价」（9 个标的），但该编号后来被
P6（主腿权重敏感性）占用 → **标的层面的配权至今没有做过**。P13 补上。

## 变量隔离（唯一变量 = 袖套内部 9 只 ETF 的目标权重）

- A 股腿权重 w 固定 0.50（P12 已证 w 曲线不可识别，不动）
- 持仓集固定 = P5 定稿 9 只，**不重新选股**
- 再平衡 = 每 21 交易日（与 P3/P4/P5 同口径）
- 估计窗口 L = 120 交易日（与 P4 walk-forward 同值，**不调参**）
- 成本 = 佣金 0.03% + 滑点 0.01%（与全项目同口径）；成本 ×2 作为稳健性对照

## 方案

| key | 权重口径 | 估计窗口 |
|---|---|---|
| `base_equal` | 类间等权 + 类内等权（= 每只 1/9） | 静态 |
| `code_invvol` | 9 只按逆波动率 | L=120 滚动 |
| `code_erc` | 9 只等风险贡献（ERC, Spinu CCD） | L=120 滚动 |
| `code_minvar` | 9 只全局最小方差（SLSQP，长仓约束） | L=120 滚动 |
| `class_rp` | **类级**风险平价（3 类，类内等权）—— 细化到标的层是否有增量？ | L=120 滚动 |
| `rand_code_s*` | 随机固定权重 ×6 | 静态 |
| `rand_code_dyn_s*` | 随机滚动权重 ×6 | L=120 滚动 |

`base_equal` 与「每只 1/9」是同一件事（3 类 × 3 只，类内等权 + 类间等权
⇒ 每只 1/9）—— 这本身就是第一个发现：**当前的"类等权"没有任何类粒度信息，
它就是 9 只标的的等权**。任何标的层的自由度都只能来自「类内不均」。

## 判据（事前注册，针对每个优化方案）

(a) main 窗口袖套夏普 ≥ 基线 + 0.05
(b) main 窗口组合夏普 ≥ 基线 + 0.05（合成 A 股腿 50%）
(c) 5 窗口方向一致：main / oos / w2013 / w2016 / w2019 的 Δ袖套夏普**全为正**
(d) 迁移一致性：**同一个方案**在 ≥ 4/5 窗口排第一（不能只在 2 个窗口赢）
(e) 打得过全部随机权重（max over 6 seeds）
(f) 换手 ≤ 3x（类级 riskparity 在 P4 里换手 10.5x —— 那就是估计噪声）
(g) 成本 ×2 后 Δ袖套夏普仍 > 0

## 核心诊断（无论判据结果如何都要给）

1. **风险贡献分解**：类等权下三类对袖套总方差的实际风险贡献（Euler 分解），
   与 P5 §5 说的「风险源数 1/1/3」对照。
2. **"类"这个抽象是否被数据支持**：ERC 最优解按类聚合后的权重是否仍接近 1/3？
   若接近 → 类粒度被支持（当前结构 ≈ 风险中性）；
   若偏离且跨窗口不稳 → 类粒度是人为约束，标的层更自由但没有稳定解。

## 口径纪律（沿用本项目铁律）

- **零技巧基准 = 买入持有等权** `mean_i[P_i(t)/P_i(t0)]`，
  绝不用「日频再平衡等权」当基准（那是给对手塞策略）
- **袖套指标 ≠ 组合指标**：两者都报，且必报 ρ(袖套, A股腿)
- **长历史是降级袖套**（gold+bond，commodity 无 2019 前数据）→ 只报袖套口径，
  **不构成组合层判决**（P2/P4/P12 三次踩过同一条坑）

跑法：
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\
        scripts/multi_asset_p13_asset_weight.py --phase ab --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\
        scripts/multi_asset_p13_asset_weight.py --phase ab --window oos
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\
        scripts/multi_asset_p13_asset_weight.py --phase long
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    END, OOS_END, OOS_START, START, maxdd, merge_by_date, sharpe,
)

OUT_DIR = os.path.join(ROOT, "data", "backtest")

# ── 定稿持仓集（P5 固化）────────────────────────────────
P5_FIXED: Dict[str, List[str]] = {
    "gold": ["518880", "159934", "159937"],
    "bond": ["511260", "511010", "511020"],
    "commodity": ["159980", "159985", "159981"],
}
CLASSES = ["gold", "bond", "commodity"]
# 长历史降级袖套：commodity 无 2019 前数据（510170 是 P5 判定最该剔除的标的）
LONG_FIXED: Dict[str, List[str]] = {
    "gold": ["518880", "159934", "159937"],
    "bond": ["511260", "511010", "511020"],
}

REB_EVERY = 21          # 每月再平衡（与 P3/P4/P5 同口径）
L_EST = 120             # 估计窗口（与 P4 walk-forward 同值，不调参）
MIN_OBS = 60            # 窗口内至少要有多少个有效观测才允许估计
COMMISSION = 0.0003
SLIPPAGE = 0.0001
SEEDS = (20261030, 7, 99, 1234, 555, 8888)
A_LEG_W = 0.5           # A 股腿权重（P12 已证不可调）

OPT_SCHEMES = ["code_invvol", "code_erc", "code_minvar", "class_rp"]


# ───────────────────────── 价格 / 收益 ─────────────────────────


def build_price_index(prices: Dict[str, Dict[str, Dict[str, float]]]
                      ) -> Dict[str, Dict[str, Any]]:
    """把 {code: {date: {open, close}}} 转成 L3.simulate 用的索引结构。"""
    idx: Dict[str, Dict[str, Any]] = {}
    for code, rows in prices.items():
        ds = sorted(rows)
        idx[code] = {
            "dates": ds,
            "open": [rows[d]["open"] for d in ds],
            "close": [rows[d]["close"] for d in ds],
        }
    return idx


def build_returns(calendar: List[str],
                  prices: Dict[str, Dict[str, Dict[str, float]]],
                  codes: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """close-to-close 日收益矩阵（行 = calendar，列 = codes），缺失为 NaN。

    前值填充：与撮合器 `_px` 的行为一致（用最近的历史收盘价）。
    """
    n, m = len(calendar), len(codes)
    close = np.full((n, m), np.nan)
    for j, code in enumerate(codes):
        rows = prices.get(code, {})
        last = None
        for i, d in enumerate(calendar):
            p = rows.get(d, {}).get("close")
            if p is not None:
                last = p
            close[i, j] = last if last is not None else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        R = close[1:] / close[:-1] - 1.0
    R[~np.isfinite(R)] = np.nan
    return R, list(calendar[1:])


# ───────────────────────── 权重估计器 ─────────────────────────


def _trailing(R: np.ndarray, i_end: int) -> Optional[np.ndarray]:
    """取截至第 i_end 行（含）的 L 行收益。行 0 对应 calendar[1]。"""
    lo = max(0, i_end - L_EST + 1)
    seg = R[lo:i_end + 1]
    if seg.shape[0] < MIN_OBS:
        return None
    if np.isnan(seg).all(axis=1).any():
        # 某一天全部缺失 → 该行不可用，剔除
        seg = seg[~np.isnan(seg).all(axis=1)]
    if seg.shape[0] < MIN_OBS:
        return None
    return seg


def _cov(seg: np.ndarray) -> Optional[np.ndarray]:
    """协方差 + 有效观测数。观测不足或奇异 → None（退回等权）。"""
    cnt = np.sum(~np.isnan(seg), axis=0)
    if (cnt < MIN_OBS).any():
        return None
    c = np.cov(seg, rowvar=False)
    if c.ndim == 0:
        c = np.array([[float(c)]])
    if not np.all(np.isfinite(c)):
        return None
    return c


def w_invvol(seg: np.ndarray, codes: Sequence[str]) -> Optional[Dict[str, float]]:
    sig = np.nanstd(seg, axis=0, ddof=1)
    if not np.all(np.isfinite(sig)) or (sig <= 0).any():
        return None
    inv = 1.0 / sig
    w = inv / inv.sum()
    return dict(zip(codes, w.tolist()))


def w_erc(seg: np.ndarray, codes: Sequence[str]) -> Optional[Dict[str, float]]:
    """等风险贡献（Spinu 2013 CCD），长仓、权重和 = 1。"""
    c = _cov(seg)
    if c is None:
        return None
    w = _erc_solve(c)
    if w is None:
        return None
    return dict(zip(codes, w.tolist()))


def _erc_solve(cov: np.ndarray, iters: int = 300, tol: float = 1e-12) -> Optional[np.ndarray]:
    n = cov.shape[0]
    if n == 0:
        return None
    w = np.ones(n) / n
    for _ in range(iters):
        prev = w.copy()
        cvar = float(w @ cov @ w)
        if not np.isfinite(cvar) or cvar <= 0:
            return None
        for i in range(n):
            a = float(cov[i, i])
            b = float(sum(cov[i, j] * w[j] for j in range(n) if j != i))
            if a <= 1e-18:
                w[i] = (cvar / n) / b if b > 1e-18 else 0.0
            else:
                disc = b * b + 4.0 * a * cvar / n
                if disc < 0:
                    return None
                w[i] = (-b + math.sqrt(disc)) / (2.0 * a)
            if w[i] < 0:
                w[i] = 0.0
        s = w.sum()
        if s <= 0:
            return None
        w = w / s
        if np.max(np.abs(w - prev)) < tol:
            break
    return w


def w_minvar(seg: np.ndarray, codes: Sequence[str]) -> Optional[Dict[str, float]]:
    """全局最小方差，长仓约束 + 权重和 = 1（SLSQP）。"""
    c = _cov(seg)
    if c is None:
        return None
    from scipy.optimize import minimize

    n = c.shape[0]

    def obj(w: np.ndarray) -> float:
        return float(w @ c @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (c @ w)

    x0 = np.ones(n) / n
    cons = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},)
    bnds = tuple((0.0, 1.0) for _ in range(n))
    try:
        r = minimize(obj, x0, jac=jac, bounds=bnds, constraints=cons,
                     method="SLSQP", options={"maxiter": 500, "ftol": 1e-14})
    except Exception:
        return None
    if not getattr(r, "success", False):
        return None
    w = np.clip(np.asarray(r.x, dtype=float), 0.0, None)
    s = w.sum()
    if s <= 0:
        return None
    return dict(zip(codes, (w / s).tolist()))


def w_class_rp(seg: np.ndarray, codes: Sequence[str],
               class_of: Dict[str, str], class_list: Sequence[str]
               ) -> Optional[Dict[str, float]]:
    """类级风险平价：类篮子收益的 ERC → 类权重 → 类内等权。"""
    c = _cov(seg)
    if c is None:
        return None
    idx = {code: j for j, code in enumerate(codes)}
    baskets: List[np.ndarray] = []
    for cl in class_list:
        members = [idx[code] for code in codes if class_of.get(code) == cl]
        if not members:
            return None
        baskets.append(np.nanmean(seg[:, members], axis=1))
    bcov = np.cov(np.array(baskets), rowvar=False)
    if bcov.ndim == 0:
        bcov = np.array([[float(bcov)]])
    if not np.all(np.isfinite(bcov)):
        return None
    cw = _erc_solve(bcov)
    if cw is None:
        return None
    out: Dict[str, float] = {}
    for cl, w_cl in zip(class_list, cw.tolist()):
        members = [code for code in codes if class_of.get(code) == cl]
        for code in members:
            out[code] = w_cl / len(members)
    return out


def w_rand_static(seg: Optional[np.ndarray], codes: Sequence[str],
                  seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(codes)))
    return dict(zip(codes, w.tolist()))


def w_rand_dyn(seg: Optional[np.ndarray], codes: Sequence[str],
               seed: int, rebal_idx: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed * 100003 + rebal_idx)
    w = rng.dirichlet(np.ones(len(codes)))
    return dict(zip(codes, w.tolist()))


# ───────────────────────── 目标权重 / 撮合 ─────────────────────────


def build_targets_code(calendar: List[str], weight_fn, reb_every: int = REB_EVERY):
    """标的层目标权重表：weight_fn(signal_date) → {code: weight}。"""
    targets: Dict[str, Dict[str, float]] = {}
    log: List[Dict[str, Any]] = []
    for i in range(1, len(calendar)):
        if (i - 1) % reb_every != 0:
            continue
        signal_date, exec_date = calendar[i - 1], calendar[i]
        cw = weight_fn(signal_date, i - 1)
        if not cw:
            continue
        t = {c: float(w) for c, w in cw.items() if w > 0}
        if not t:
            continue
        s = sum(t.values())
        if abs(s - 1.0) > 1e-6:
            t = {c: w / s for c, w in t.items()}
        targets[exec_date] = t
        log.append({"date": signal_date, "exec": exec_date,
                    "code_weights": {k: round(v, 6) for k, v in t.items()}})
    return targets, {"n_rebalances": len(log), "weight_log": log}


def simulate_portfolio(calendar: List[str],
                       idx: Dict[str, Dict[str, Any]],
                       targets: Dict[str, Dict[str, float]],
                       cost_mult: float = 1.0) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """撮合器（独立实现，语义与 L3.simulate 一致）。

    与 L3.simulate 的差异只有两点，都是刻意的：
      1. 价格索引由参数传入（长历史用另一套目录，不能依赖全局单例）
      2. 成本可乘倍数（成本 ×2 稳健性对照）

    口径一致性由**基线复现门**证明：base_equal 必须逐位复现 P3 定稿。
    """
    commission = COMMISSION * cost_mult
    slippage = SLIPPAGE * cost_mult

    def px(code: str, d: str, field: str) -> Optional[float]:
        node = idx.get(code)
        if not node:
            return None
        i = bisect.bisect_right(node["dates"], d) - 1
        if i < 0:
            return None
        return float(node[field][i])

    missing = set()
    for _d, t in targets.items():
        missing |= {c for c in t if c not in idx}
    if missing:
        raise SystemExit(f"[p13] ❌ 价格索引缺少 {len(missing)} 个代码: "
                         f"{sorted(missing)[:5]}")

    capital = 1_000_000.0
    cash = capital
    shares: Dict[str, float] = {}
    nav: List[Tuple[str, float]] = []
    total_cost = 0.0
    total_turnover = 0.0
    n_rebal = 0

    for d in calendar:
        tgt = targets.get(d)
        if tgt:
            n_rebal += 1
            val = cash
            for code, sh in shares.items():
                p = px(code, d, "open")
                if p is not None:
                    val += sh * p
            # 1) 先卖不在目标里的
            for code in list(shares.keys()):
                if code in tgt:
                    continue
                p = px(code, d, "open")
                if p is None:
                    continue
                proceeds = shares[code] * p * (1 - slippage)
                cash += proceeds
                total_cost += proceeds * commission
                total_turnover += proceeds
                del shares[code]
            # 2) 买（按比例缩放，顺序无关）
            buys: List[Tuple[str, float]] = []
            for code, w in tgt.items():
                if w <= 0:
                    continue
                p = px(code, d, "open")
                if p is None:
                    continue
                want = val * w
                cur = shares.get(code, 0.0) * p
                if cur >= want:
                    continue
                buys.append((code, want - cur))
            if buys:
                need = sum(a for _, a in buys)
                scale = 1.0
                if need > 0 and cash < need * (1 + commission):
                    scale = cash / (need * (1 + commission))
                for code, amount in buys:
                    if amount <= 0:
                        continue
                    amount *= scale
                    cost = amount * commission
                    cash -= amount + cost
                    total_cost += cost
                    total_turnover += amount
                    buy_px = px(code, d, "open") * (1 + slippage)
                    shares[code] = shares.get(code, 0.0) + amount / buy_px
            # 3) 超配的卖回目标权重
            for code in list(shares.keys()):
                if code not in tgt:
                    continue
                p = px(code, d, "open")
                if p is None:
                    continue
                cur = shares[code] * p
                want = val * tgt[code]
                if cur <= want:
                    continue
                sell_px = p * (1 - slippage)
                proceeds = min(shares[code] * sell_px, cur - want)
                if proceeds <= 0:
                    continue
                cash += proceeds
                total_cost += proceeds * commission
                total_turnover += proceeds
                shares[code] -= proceeds / sell_px

        v = cash
        for code, sh in shares.items():
            p = px(code, d, "close")
            if p is not None:
                v += sh * p
        nav.append((d, v))

    rec = {
        "commission_total": round(total_cost, 4),
        "turnover_total": round(total_turnover, 4),
        "turnover_ratio": round(total_turnover / capital, 6),
        "n_rebalances": n_rebal,
        "cost_mult": cost_mult,
    }
    return nav, rec


# ───────────────────────── 诊断 / 基准 ─────────────────────────


def buyhold_equal(calendar: List[str],
                  prices: Dict[str, Dict[str, Dict[str, float]]],
                  codes: Sequence[str]) -> List[Tuple[str, float]]:
    """零技巧基准：买入持有等权 mean_i[P_i(t)/P_i(t0)]。

    只纳入在窗口首个交易日就有价格的标的（否则分母会随时间变化，
    那就不是买入持有了）。**绝不用日频再平衡等权当基准**。
    """
    if not calendar:
        return []
    d0 = calendar[0]
    use = [c for c in codes if d0 in prices.get(c, {})]
    if not use:
        return []
    series: Dict[str, Dict[str, float]] = {}
    for c in use:
        rows = prices[c]
        base = rows[d0]["close"]
        series[c] = {d: rows[d]["close"] / base for d in calendar if d in rows}
    out: List[Tuple[str, float]] = []
    for d in calendar:
        vals = [series[c][d] for c in use if d in series[c]]
        if vals:
            out.append((d, sum(vals) / len(vals)))
    return out


def risk_contrib_by_class(weights: Dict[str, float],
                           cov: np.ndarray,
                           codes: Sequence[str],
                           class_of: Dict[str, str],
                           class_list: Sequence[str]) -> Dict[str, float]:
    """Euler 风险贡献分解：RC_i = w_i(Σw)_i / (w'Σw)，再按类聚合。"""
    w = np.array([weights.get(c, 0.0) for c in codes], dtype=float)
    if w.sum() <= 0:
        return {c: 0.0 for c in class_list}
    sw = cov @ w
    denom = float(w @ sw)
    if denom <= 0:
        return {c: 0.0 for c in class_list}
    rc = w * sw / denom
    out: Dict[str, float] = {c: 0.0 for c in class_list}
    for j, code in enumerate(codes):
        cl = class_of.get(code)
        if cl in out:
            out[cl] += float(rc[j])
    return {c: round(v, 6) for c, v in out.items()}


def class_weight_avg(weight_log: List[Dict[str, Any]],
                     class_of: Dict[str, str],
                     class_list: Sequence[str]) -> Dict[str, Any]:
    """各调仓日类权重的平均（按标的层权重聚回类层）。"""
    acc: Dict[str, List[float]] = {c: [] for c in class_list}
    for row in weight_log:
        cw = row.get("code_weights", {})
        tot = sum(cw.values()) or 1.0
        for cl in class_list:
            s = sum(w for code, w in cw.items() if class_of.get(code) == cl)
            acc[cl].append(s / tot)
    return {c: (round(sum(v) / len(v), 6) if v else None) for c, v in acc.items()}


def nav_return(nav: List[Tuple[str, float]]) -> float:
    if len(nav) < 2 or nav[0][1] <= 0:
        return 0.0
    return nav[-1][1] / nav[0][1] - 1.0


def nav_vol(nav: List[Tuple[str, float]]) -> float:
    if len(nav) < 3:
        return 0.0
    rets = [nav[i][1] / nav[i - 1][1] - 1.0 for i in range(1, len(nav))]
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / len(rets)
    return math.sqrt(var) * math.sqrt(244.0)


def corr(a: List[Tuple[str, float]], b: List[Tuple[str, float]]) -> Optional[float]:
    da, db = dict(a), dict(b)
    dates = sorted(set(da) & set(db))
    if len(dates) < 30:
        return None
    ra = [da[dates[i]] / da[dates[i - 1]] - 1 for i in range(1, len(dates))]
    rb = [db[dates[i]] / db[dates[i - 1]] - 1 for i in range(1, len(dates))]
    if len(ra) < 10:
        return None
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da_ = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db_ = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da_ <= 0 or db_ <= 0:
        return None
    return round(num / (da_ * db_), 6)


# ───────────────────────── 方案执行 ─────────────────────────


def run_scheme(calendar: List[str],
               R: np.ndarray,
               est_pos: Dict[str, int],
               codes: List[str],
               class_of: Dict[str, str],
               class_list: List[str],
               scheme: str,
               idx: Dict[str, Dict[str, Any]],
               cost_mult: float = 1.0) -> Dict[str, Any]:
    """跑一个方案，返回净值 + 记录。"""
    n_fallback = 0
    rebal_counter = {"n": 0}

    def weight_fn(_signal_date: str, i_signal: int) -> Optional[Dict[str, float]]:
        nonlocal n_fallback
        rebal_counter["n"] += 1
        # ⚠️ 信号日 = calendar[i_signal]，当日收盘已知 → 最新可用收益是
        #    R[j-1]（j = 信号日在**估计矩阵**里的行号）。
        #    取到 R[j] 会把「信号日→次日」这段未来收益算进估计 → 前视偏差。
        # ⚠️ 估计矩阵必须**覆盖窗口之前**的历史（L=120 交易日）：
        #    只用窗口内的话，窗口头 3 次调仓都凑不满 120 天 → 优化方案被
        #    静默降级成等权，而基线不是 → 起点不公平（P4 的 walk-forward
        #    用的是「信号日之前」的数据，本来就包含窗口前的部分）。
        j = est_pos.get(_signal_date)
        seg = _trailing(R, j - 1) if j is not None else None
        if scheme == "base_equal":
            w = 1.0 / len(codes)
            return {c: w for c in codes}
        if scheme.startswith("rand_code_dyn_s"):
            # 方案名里带的**就是 seed 值本身**（如 rand_code_dyn_s20261030），
            # 不能拿它当 SEEDS 的下标（SEEDS 是 6 个小整数，不是 20261030）
            seed = int(scheme.rsplit("_s", 1)[1])
            return w_rand_dyn(seg, codes, seed, rebal_counter["n"])
        if scheme.startswith("rand_code_s"):
            seed = int(scheme.rsplit("_s", 1)[1])
            return w_rand_static(seg, codes, seed)
        if seg is None:
            n_fallback += 1
            w = 1.0 / len(codes)
            return {c: w for c in codes}
        if scheme == "code_invvol":
            out = w_invvol(seg, codes)
        elif scheme == "code_erc":
            out = w_erc(seg, codes)
        elif scheme == "code_minvar":
            out = w_minvar(seg, codes)
        elif scheme == "class_rp":
            out = w_class_rp(seg, codes, class_of, class_list)
        else:
            raise SystemExit(f"[p13] ❌ 未知方案 {scheme}")
        if out is None:
            n_fallback += 1
            w = 1.0 / len(codes)
            return {c: w for c in codes}
        return out

    targets, wmeta = build_targets_code(calendar, weight_fn)
    nav, rec = simulate_portfolio(calendar, idx, targets, cost_mult=cost_mult)
    return {"nav": nav, "rec": rec, "wlog": wmeta["weight_log"],
            "n_rebalances": wmeta["n_rebalances"], "fallback": n_fallback}


def scheme_metrics(nav: List[Tuple[str, float]],
                   rec: Dict[str, Any],
                   a_nav: Optional[List[Tuple[str, float]]],
                   wlog: List[Dict[str, Any]],
                   class_of: Dict[str, str],
                   class_list: List[str],
                   fallback: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sleeve_sharpe": round(sharpe(nav), 4),
        "sleeve_return": round(nav_return(nav), 6),
        "sleeve_maxdd": round(maxdd(nav), 6),
        "sleeve_vol": round(nav_vol(nav), 6),
        "turnover_ratio": rec["turnover_ratio"],
        "n_rebalances": rec["n_rebalances"],
        "fallback_equal": fallback,
        "class_weights_avg": class_weight_avg(wlog, class_of, class_list),
    }
    if a_nav:
        comb = merge_by_date([a_nav, nav], [A_LEG_W, 1.0 - A_LEG_W])
        out["combined_sharpe"] = round(sharpe(comb), 4)
        out["combined_return"] = round(nav_return(comb), 6)
        out["combined_maxdd"] = round(maxdd(comb), 6)
        out["rho_sleeve_ashare"] = corr(nav, a_nav)
    return out


# ───────────────────────── 阶段 AB ─────────────────────────


def run_ab(window: str) -> Dict[str, Any]:
    import multi_asset_leg_experiment as M

    start, end = (START, END) if window == "main" else (OOS_START, OOS_END)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    out_path = os.path.join(OUT_DIR, f"multi_asset_p13_{window}.json")
    nav_cache = os.path.join(OUT_DIR, f"_a_share_nav_{window}.json")

    # ── A 股腿（缓存优先；模块全局窗口必须同步）──
    a_nav: List[Tuple[str, float]] = []
    if os.path.exists(nav_cache):
        with open(nav_cache, encoding="utf-8") as fh:
            blob = json.load(fh)
        a_nav = [(d, float(v)) for d, v in blob["a_share_nav"]]
        print(f"[p13] 读缓存 A 股腿 {window}: {len(a_nav)} 点", flush=True)
    else:
        M.START, M.END = start, end
        from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
        codes_uni = universe_from_instruments(QLIB_OFFICIAL)
        print(f"[p13] 跑 A 股腿 {window}（{len(codes_uni)} 只）...", flush=True)
        rec_a, a_nav = M.run_a_share_leg(codes_uni)
        ver = M.verify_baseline(rec_a)
        print(f"[p13] A 腿基线复现门: {json.dumps(ver, ensure_ascii=False)}", flush=True)
        if ver.get("checked") and not ver.get("ok"):
            raise SystemExit("[p13] ❌ A 股腿基线复现失败，退出")
        with open(nav_cache, "w", encoding="utf-8") as fh:
            json.dump({"a_share_nav": a_nav, "a_share_leg": rec_a}, fh, ensure_ascii=False)

    # ── ETF 价格 / 日历 ──
    pool = P3.build_pool(start)
    universe = L3.load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    held = {c: list(P5_FIXED[c]) for c in P5_FIXED}
    print(f"[p13] 固定持仓集(P5 定稿): {json.dumps(held, ensure_ascii=False)}", flush=True)

    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    idx = build_price_index(prices)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    if len(calendar) < 300:
        raise SystemExit(f"[p13] ❌ 交易日历过短 {len(calendar)}，退出")
    print(f"[p13] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)

    codes = [c for lst in held.values() for c in lst]
    class_of = {c: cl for cl, lst in held.items() for c in lst}
    class_list = list(held.keys())

    # 估计矩阵：**窗口前 500 个自然日** + 窗口内，保证每个信号日都有满 120 天历史
    est_start = (start - __import__("datetime").timedelta(days=500)).strftime("%Y-%m-%d")
    est_cal = sorted({d for c in codes for d in prices.get(c, {})
                      if est_start <= d <= e_str} | set(calendar))
    est_pos = {d: i for i, d in enumerate(est_cal)}
    R, _ = build_returns(est_cal, prices, codes)
    print(f"[p13] 估计矩阵 {len(est_cal)} 天（{est_cal[0]} → {est_cal[-1]}），"
          f"窗口内 {len(calendar)} 天", flush=True)

    # ── 方案清单 ──
    schemes = ["base_equal"] + OPT_SCHEMES
    schemes += [f"rand_code_s{s}" for s in SEEDS]
    schemes += [f"rand_code_dyn_s{s}" for s in SEEDS]

    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}
    wlogs: Dict[str, List[Dict[str, Any]]] = {}
    for sc in schemes:
        r = run_scheme(calendar, R, est_pos, codes, class_of, class_list, sc, idx)
        m = scheme_metrics(r["nav"], r["rec"], a_nav, r["wlog"], class_of,
                           class_list, r["fallback"])
        results[sc] = m
        navs[sc] = r["nav"]
        wlogs[sc] = r["wlog"]
        print(f"[p13] {sc:22s} 袖套夏普 {m['sleeve_sharpe']:.4f} "
              f"组合 {m.get('combined_sharpe')} 换手 {m['turnover_ratio']:.2f}x",
              flush=True)

    # ── 成本 ×2 对照 ──
    cost2: Dict[str, Any] = {}
    for sc in ["base_equal"] + OPT_SCHEMES:
        r = run_scheme(calendar, R, est_pos, codes, class_of, class_list, sc, idx,
                       cost_mult=2.0)
        cost2[sc] = {
            "sleeve_sharpe": round(sharpe(r["nav"]), 4),
            "turnover_ratio": r["rec"]["turnover_ratio"],
        }
        if a_nav:
            comb = merge_by_date([a_nav, r["nav"]], [A_LEG_W, 1.0 - A_LEG_W])
            cost2[sc]["combined_sharpe"] = round(sharpe(comb), 4)
        print(f"[p13] [成本×2] {sc:22s} 袖套夏普 {cost2[sc]['sleeve_sharpe']:.4f}",
              flush=True)

    # ── 零技巧基准：买入持有等权 ──
    bh = buyhold_equal(calendar, prices, codes)
    bh_metrics = {
        "sleeve_sharpe": round(sharpe(bh), 4),
        "sleeve_return": round(nav_return(bh), 6),
        "sleeve_maxdd": round(maxdd(bh), 6),
        "sleeve_vol": round(nav_vol(bh), 6),
        "n_codes": len({c for c in codes if calendar[0] in prices.get(c, {})}),
    }
    if a_nav:
        comb_bh = merge_by_date([a_nav, bh], [A_LEG_W, 1.0 - A_LEG_W])
        bh_metrics["combined_sharpe"] = round(sharpe(comb_bh), 4)
        bh_metrics["combined_return"] = round(nav_return(comb_bh), 6)
        bh_metrics["combined_maxdd"] = round(maxdd(comb_bh), 6)
    print(f"[p13] 买入持有等权基准: {json.dumps(bh_metrics, ensure_ascii=False)}",
          flush=True)

    # ── 风险贡献分解（类等权 vs 标的层 ERC）──
    rc: Dict[str, Any] = {}
    for sc in ("base_equal", "code_erc", "code_minvar", "code_invvol"):
        acc: Dict[str, List[float]] = {c: [] for c in class_list}
        for row in wlogs.get(sc, []):
            j = est_pos.get(row["date"])
            if j is None:
                continue
            seg = _trailing(R, j - 1)
            if seg is None:
                continue
            c = _cov(seg)
            if c is None:
                continue
            contrib = risk_contrib_by_class(row.get("code_weights", {}), c, codes,
                                            class_of, class_list)
            for k, v in contrib.items():
                acc[k].append(v)
        rc[sc] = {k: (round(sum(v) / len(v), 4) if v else None) for k, v in acc.items()}
        rc[sc + "_n_obs"] = len(wlogs.get(sc, []))
    print(f"[p13] 风险贡献分解: {json.dumps(rc, ensure_ascii=False)}", flush=True)

    # ── 基线复现门：base_equal 必须逐位复现 P3 定稿 ──
    p3p = os.path.join(OUT_DIR, f"multi_asset_p3_{window}.json")
    with open(p3p, encoding="utf-8") as fh:
        p3 = json.load(fh)
    p3res = p3.get("results", {})
    p3key = ("bond_lowrho3_noliq" if "bond_lowrho3_noliq" in p3res
             else "bond_lowrho3")
    exp = float(p3res[p3key]["combined_sharpe"])
    got = float(results["base_equal"]["combined_sharpe"])
    ok = abs(got - exp) <= 5e-4
    print(f"[p13] 基线复现门: base_equal 组合夏普 {got:.4f}（P3[{p3key}] {exp:.4f}）"
          f" → {'✅' if ok else '❌'}", flush=True)
    if not ok:
        raise SystemExit(f"[p13] ❌ 基线漂移 {got} != {exp}，全部配置不可比，退出")

    out = {
        "phase": "ab",
        "window": window,
        "start": s_str,
        "end": e_str,
        "n_calendar": len(calendar),
        "held": held,
        "codes": codes,
        "class_of": class_of,
        "a_leg_weight": A_LEG_W,
        "estimate_window": L_EST,
        "reb_every": REB_EVERY,
        "baseline_gate": {"p3_key": p3key, "expected": exp, "got": got, "ok": ok},
        "results": results,
        "cost_x2": cost2,
        "buyhold_equal_benchmark": bh_metrics,
        "risk_contrib_by_class": rc,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[p13] ✓ 已写入 {out_path}", flush=True)
    return out


# ───────────────────────── 阶段 long ─────────────────────────


def load_long_prices(codes: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """长历史 ETF 价格（data/etf/parquet_long/）。"""
    import pandas as pd

    d = os.path.join(ROOT, "data", "etf", "parquet_long")
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for code in codes:
        path = os.path.join(d, f"{code}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values("date")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        rows: Dict[str, Dict[str, float]] = {}
        for _, r in df.iterrows():
            rows[str(r["date"])] = {"open": float(r["open"]), "close": float(r["close"])}
        if rows:
            out[code] = rows
    return out


def run_long() -> Dict[str, Any]:
    out_path = os.path.join(OUT_DIR, "multi_asset_p13_long.json")
    WINDOWS = [
        ("w2013", datetime(2013, 5, 8), datetime(2016, 5, 8)),
        ("w2016", datetime(2016, 5, 8), datetime(2019, 5, 8)),
        ("w2019", datetime(2019, 5, 8), datetime(2021, 5, 11)),
    ]
    long_codes = sorted({c for lst in LONG_FIXED.values() for c in lst})
    prices = load_long_prices(long_codes)
    missing = [c for c in long_codes if c not in prices]
    if missing:
        raise SystemExit(f"[p13] ❌ 长历史 ETF 缺数据: {missing}")
    idx = build_price_index(prices)

    results: Dict[str, Any] = {}
    for label, start, end in WINDOWS:
        s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        # 只保留窗口内有数据的标的（与 P12 长历史同口径）
        held = {cl: [c for c in lst if any(s_str <= d <= e_str for d in prices[c])]
                for cl, lst in LONG_FIXED.items()}
        held = {cl: lst for cl, lst in held.items() if lst}
        codes = [c for lst in held.values() for c in lst]
        class_of = {c: cl for cl, lst in held.items() for c in lst}
        class_list = list(held.keys())
        calendar = sorted({d for c in codes for d in prices[c]
                           if s_str <= d <= e_str})
        if len(calendar) < 300:
            print(f"[p13] ⚠️ {label} 日历过短 {len(calendar)}，跳过", flush=True)
            continue
        print(f"[p13] === 长历史 {label} {s_str}~{e_str} "
              f"日历 {len(calendar)} 天 持仓 {json.dumps(held, ensure_ascii=False)}",
              flush=True)
        # 估计矩阵同样覆盖窗口前历史（长历史起点更早，ETF 上市时间不统一，
        # 估计不足时会退回等权并记入 fallback_equal）
        est_start = (start - __import__("datetime").timedelta(days=500)).strftime("%Y-%m-%d")
        est_cal = sorted({d for c in codes for d in prices.get(c, {})
                          if est_start <= d <= e_str} | set(calendar))
        est_pos = {d: i for i, d in enumerate(est_cal)}
        R, _ = build_returns(est_cal, prices, codes)

        schemes = ["base_equal"] + OPT_SCHEMES
        schemes += [f"rand_code_s{s}" for s in SEEDS]
        schemes += [f"rand_code_dyn_s{s}" for s in SEEDS]
        w: Dict[str, Any] = {}
        for sc in schemes:
            r = run_scheme(calendar, R, est_pos, codes, class_of, class_list, sc, idx)
            w[sc] = {
                "sleeve_sharpe": round(sharpe(r["nav"]), 4),
                "sleeve_return": round(nav_return(r["nav"]), 6),
                "sleeve_maxdd": round(maxdd(r["nav"]), 6),
                "sleeve_vol": round(nav_vol(r["nav"]), 6),
                "turnover_ratio": r["rec"]["turnover_ratio"],
                "n_rebalances": r["rec"]["n_rebalances"],
                "fallback_equal": r["fallback"],
                "class_weights_avg": class_weight_avg(r["wlog"], class_of, class_list),
            }
        bh = buyhold_equal(calendar, prices, codes)
        w["buyhold_equal_benchmark"] = {
            "sleeve_sharpe": round(sharpe(bh), 4),
            "sleeve_return": round(nav_return(bh), 6),
            "sleeve_maxdd": round(maxdd(bh), 6),
        }
        results[label] = {
            "window": [s_str, e_str],
            "held": held,
            "n_calendar": len(calendar),
            "results": w,
        }
        print(f"[p13] {label} base_equal 袖套夏普 "
              f"{w['base_equal']['sleeve_sharpe']:.4f} | "
              f"erc {w['code_erc']['sleeve_sharpe']:.4f} | "
              f"minvar {w['code_minvar']['sleeve_sharpe']:.4f}", flush=True)

    out = {
        "phase": "long",
        "note": ("长历史是**降级袖套**（commodity 无 2019 前数据 → 只有 gold+bond），"
                 "且窗口内无数据的标的被剔除 → 只报袖套口径，不构成组合层判决"),
        "estimate_window": L_EST,
        "reb_every": REB_EVERY,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[p13] ✓ 已写入 {out_path}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="ab", choices=["ab", "long"])
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    args = ap.parse_args()
    if args.phase == "ab":
        run_ab(args.window)
    else:
        run_long()


if __name__ == "__main__":
    main()
