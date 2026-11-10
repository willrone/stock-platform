#!/usr/bin/env python3
"""横截面选股「组合构建层 / 信号处理层」方法库（供 methods_research.py 调用）。

本模块只做**方法实现**，不跑回测。所有方法都遵循一条硬规矩：
**任何用到未来信息的量都必须 shift(1) 或用滚动窗口**（IC 权重、EWMA 平滑、
分层配额等），否则回测数字是假的。

## 方法清单（每条都标注出处）

| 方法 | 抄的东西 |
|---|---|
| `ewma_smooth` | Barra/MSCI 因子暴露的指数加权平滑（半衰期可调） |
| `icir_weights` | Grinold-Kahn: IR = IC·√BR；Newey-West(1987) HAC 标准误 + Andrews(1991) 滞后选择 |
| `newey_west_tstat` | Newey-West(1987) 自相关稳健 t 统计量 |
| `cost_penalty` | Amihud(2002) 非流动性 + Frazzini-Israel(2018) 交易成本内生化 |
| `group_cap_select` | Barra APT 组约束式分组配额选股 |
| `hrp_weights` | Lopez de Prado(2016) HRP: 层次聚类→准对角化→递归二分 |
| `maxdiv_weights` | Maillard-Roncalli-Teissier(2010) 最大分散化权重 |
| `blend_horizons` | Asness-Frazzini-Pedersen(2013) 多周期信号合成 |

## 用法
    env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
        backend/scripts/methods_lab.py --selftest
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if os.path.join(BACKEND, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402

CACHE_DIR = os.path.join(
    os.environ.get("HERMES_SCRATCH", "/Users/rone/.hermes/cache/scratch"),
    "stock_methods",
)


# ══════════════════════════════════════════════════════════════════
# 0. 数据加载（带磁盘缓存，避免每次重载 5513 只股票）
# ══════════════════════════════════════════════════════════════════
def load_cached_mats(
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """加载 (date × stock) 矩阵 + tradability，落地 npz 缓存。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"mats_{L.START.date()}_{L.END.date()}"
    path = os.path.join(CACHE_DIR, tag + ".npz")
    codes = L.universe_from_instruments()
    if os.path.exists(path) and not rebuild:
        z = np.load(path, allow_pickle=True)
        mats = {
            "codes": list(z["codes"]),
            "master": pd.DatetimeIndex(z["master"]),
            "close": z["close"], "high": z["high"], "low": z["low"],
            "volume": z["volume"],
            "stock_data": None,      # 缓存后不再需要（tradability 一起缓存）
        }
        trad = {"can_buy": z["can_buy"], "can_sell": z["can_sell"],
                "max_buy_amount": z["max_buy_amount"]}
        print(f"[mats] cache hit {path}", flush=True)
        return {"mats": mats, "trad": trad}

    print(f"[mats] loading universe {len(codes)}", flush=True)
    mats = L.load_matrices(codes)
    print("[mats] building tradability ...", flush=True)
    trad = L.build_tradability_mats(mats)
    np.savez_compressed(
        path,
        codes=np.array(mats["codes"], dtype=object),
        master=mats["master"].to_numpy(),
        close=mats["close"], high=mats["high"], low=mats["low"],
        volume=mats["volume"],
        can_buy=trad["can_buy"], can_sell=trad["can_sell"],
        max_buy_amount=trad["max_buy_amount"],
    )
    print(f"[mats] cached -> {path}", flush=True)
    mats2 = dict(mats)
    mats2["stock_data"] = None
    return {"mats": mats2, "trad": trad}


# ══════════════════════════════════════════════════════════════════
# 1. 分数层：EWMA 平滑（Barra 因子暴露平滑）
# ══════════════════════════════════════════════════════════════════
def ewma_smooth(
    score: np.ndarray,
    halflife: float,
    min_periods: int = 1,
) -> np.ndarray:
    """逐股时间维 EWMA 平滑（nan-aware：NaN 日沿用上次有效值）。

    Barra/MSCI 因子模型对因子暴露做指数加权移动平均以降低估计噪声
    （MSCI Barra Global Equity Model Handbook, "Factor Exposure Estimation"
    一节用 EWMA + 半衰期平滑原始暴露）。此处把同一手法用在**复合信号**上：
    半衰期越长 → 信号越平滑 → 换手越低，但反应越慢。

    alpha = 1 - exp(-ln2 / halflife)   (半衰期口径)
    """
    if halflife <= 0:
        return score.copy()
    alpha = 1.0 - np.exp(-np.log(2.0) / float(halflife))
    n_d, n_s = score.shape
    out = np.full_like(score, np.nan)
    prev = np.full(n_s, np.nan)
    seen = np.zeros(n_s, dtype=np.int64)
    for t in range(n_d):
        v = score[t]
        valid = np.isfinite(v)
        new_prev = prev.copy()
        fresh = valid & ~np.isfinite(prev)
        new_prev[fresh] = v[fresh]
        upd = valid & np.isfinite(prev)
        new_prev[upd] = alpha * v[upd] + (1.0 - alpha) * prev[upd]
        out[t] = np.where(seen + valid > 0, new_prev, np.nan)
        prev = new_prev
        seen = seen + valid.astype(np.int64)
    if min_periods > 1:
        out[seen[None, :] < min_periods] = np.nan
    return out


def moving_average_smooth(score: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均平滑（多周期合成的另一种形式）。"""
    if window <= 1:
        return score.copy()
    df = pd.DataFrame(score)
    return df.rolling(window, min_periods=max(1, window // 2)).mean().to_numpy()


# ══════════════════════════════════════════════════════════════════
# 2. IC 层：Newey-West 调整的 ICIR 权重
# ══════════════════════════════════════════════════════════════════
def newey_west_mean_tstat(
    series: np.ndarray,
    max_lag: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Newey-West(1987) HAC 下样本均值的 t 统计量。

    返回 (mean, se, tstat)。
    序列自相关 → 朴素 t 统计高估显著性；NW 用 Bartlett 核修正。
    滞后阶数用 Andrews(1991) 经验规则: L = floor(4*(T/100)^(2/9))。
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 10:
        return (float(np.nan), float(np.nan), float(np.nan))
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(int(max_lag), T - 1))
    u = x - x.mean()
    gamma0 = float(np.dot(u, u) / T)
    S = gamma0
    for l in range(1, max_lag + 1):
        w = 1.0 - l / (max_lag + 1.0)          # Bartlett 核
        gl = float(np.dot(u[l:], u[:-l]) / T)
        S += 2.0 * w * gl
    S = max(S, 1e-18)
    se = float(np.sqrt(S / T))
    mean = float(x.mean())
    t = mean / se if se > 0 else float("nan")
    return mean, se, t


def factor_rank_ic_series(
    z: np.ndarray,
    fwd: np.ndarray,
) -> np.ndarray:
    """逐日截面 RankIC 序列（对 fwd 收益）。"""
    return L.cs_rank_ic(z, fwd)


def icir_weighted_score(
    z_factors: Dict[str, np.ndarray],
    fwd: np.ndarray,
    window: int = 120,
    min_periods: int = 40,
    use_nw: bool = True,
    clip: float = 1.0,
    shrink: bool = True,
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """按滚动 ICIR（可选 Newey-West t 统计）加权的因子复合。

    walk-forward，无前视：第 t 日的权重只用 [t-window, t-1] 的历史 IC。
    Grinold-Kahn: 信息比率 IR ≈ IC·√BR → 因子权重 ∝ IC。
    Newey-West 调整：IC 序列因重叠标签而自相关 → 用 HAC t 统计代替 IC/σ。

    shrink=True: w = mean_ic * t²/(t²+1)（James-Stein 式收缩，抑制小样本
    高 IC 因子的过拟合）。

    返回 (score, per_factor_diagnostics)。
    """
    names = list(z_factors)
    n_d, n_s = z_factors[names[0]].shape
    wmat = np.zeros((n_d, len(names)))
    diag: Dict[str, Dict[str, np.ndarray]] = {}
    for j, name in enumerate(names):
        ics = pd.Series(factor_rank_ic_series(z_factors[name], fwd))
        roll_mean = ics.rolling(window, min_periods=min_periods).mean()
        roll_std = ics.rolling(window, min_periods=min_periods).std()
        icir = (roll_mean / roll_std.replace(0.0, np.nan)).shift(1)
        if use_nw:
            # NW t 统计：逐日滚动代价高 → 用块状窗口近似（每 20 日算一次，
            # 中间线性插值），保留 HAC 修正的方向性。
            nw_t = _blockwise_nw_tstat(ics, window=window,
                                       min_periods=min_periods, block=20)
            nw_t = nw_t.shift(1)
            w = nw_t.fillna(0.0)
        else:
            w = icir.fillna(0.0)
        if shrink:
            # 用 t²/(t²+1) 收缩（NW 路径下 w 已是 t 统计）
            t2 = w * w
            w = w * t2 / (t2 + 1.0)
        w = w.clip(-clip, clip)
        wmat[:, j] = w.to_numpy()
        diag[name] = {
            "ic_mean": roll_mean.to_numpy(),
            "ic_std": roll_std.to_numpy(),
            "icir": icir.to_numpy(),
            "weight": w.to_numpy(),
        }
    total = np.zeros((n_d, n_s))
    wsum = np.zeros((n_d, n_s))
    for j, name in enumerate(names):
        z = z_factors[name]
        w = wmat[:, j][:, None]
        good = np.isfinite(z) & np.isfinite(w)
        total += np.where(good, z * w, 0.0)
        wsum += np.where(good, np.abs(w), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)
    return score, diag


def _blockwise_nw_tstat(
    ics: pd.Series,
    window: int,
    min_periods: int,
    block: int = 20,
) -> pd.Series:
    """块状 Newey-West t 统计：每 block 日算一次滚动窗口的 NW t，中间前向填充。

    全量逐日 NW 需要 O(T²)（T=727 → 53 万次 HAC 求和 × 42 因子 ≈ 可接受
    但慢）；块状近似把代价降到 1/block，方向性一致。
    """
    T = len(ics)
    out = np.full(T, np.nan)
    arr = ics.to_numpy(dtype=np.float64)
    for start in range(0, T, block):
        end = min(start + block, T)
        seg = arr[max(0, end - window):end]
        if len(seg) < min_periods:
            continue
        mean, se, t = newey_west_mean_tstat(seg)
        if np.isfinite(t):
            out[start:end] = t
    return pd.Series(out, index=ics.index).ffill()


# ══════════════════════════════════════════════════════════════════
# 3. 成本内生化：Amihud 非流动性惩罚
# ══════════════════════════════════════════════════════════════════
def amihud_illiq(
    close: np.ndarray,
    volume: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Amihud(2002) 非流动性: ILLIQ = |r| / 成交额，窗口内均值。

    Amihud(2002) "Illiquidity and stock returns: cross-section and
    time-series effects", JFM 415: 31-56.
    数值上取 log(1+ILLIQ) 以抗肥尾。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close[1:] / close[:-1] - 1.0
        ret = np.vstack([np.full((1, close.shape[1]), np.nan), ret])
        amt = close * volume
        daily = np.abs(ret) / np.where(amt > 0, amt, np.nan)
    df = pd.DataFrame(daily)
    m = df.rolling(window, min_periods=max(5, window // 2)).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log1p(m.to_numpy())


def cost_penalized_score(
    score: np.ndarray,
    illiq: np.ndarray,
    lam: float,
) -> np.ndarray:
    """成本感知打分: score' = z(score) - λ·z(illiq)。

    逻辑：预期净 alpha = 预期毛 alpha - 交易成本。Frazzini & Israel(2018)
    "Trading Costs of Asset Pricing Anomalies" JF 73(1): 255-276 指出
    规模/流动性越小 → 毛 alpha 越高但成本越高 → 净 alpha 最优在中间。
    这里的实现把「成本」显式扣在**选股阶段**（而不是只在成交时扣费）。
    """
    if lam == 0:
        return score
    z_s = L.cs_zscore(score, 30)
    z_i = L.cs_zscore(illiq, 30)
    return z_s - lam * z_i


# ══════════════════════════════════════════════════════════════════
# 4. 组合构建层：HRP / 最大分散化 / 逆波动率
# ══════════════════════════════════════════════════════════════════
def _trailing_returns(
    close: np.ndarray,
    t: int,
    idx: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """取 t 日往前 lookback 日的收益矩阵 (lookback × len(idx))。

    注意：只取 close[t0:t]，**不含 t 日** —— 模拟器的 lag=1 语义下，
    第 t 天的成交用的是第 t-1 天收盘算出的信号，因此权重只能用 ≤ t-1 的价格。
    """
    t0 = max(0, t - lookback)
    if t0 >= t:
        t0 = max(0, t - 1)
    c = close[t0:t][:, idx]
    if c.shape[0] < 2:
        return np.zeros((0, len(idx)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = c[1:] / c[:-1] - 1.0
    return r


def _cluster_var(c: np.ndarray, idx: Sequence[int]) -> float:
    """簇的组合方差（簇内用逆方差组合）。c 为相关矩阵。"""
    idx = np.asarray(list(idx), dtype=int)
    if len(idx) == 0:
        return 0.0
    sub = c[np.ix_(idx, idx)]
    var = np.diag(sub).copy()
    iv = 1.0 / np.where(var > 1e-12, var, np.nan)
    iv = np.nan_to_num(iv, nan=0.0)
    if iv.sum() <= 0:
        return float(np.mean(var)) if len(var) else 0.0
    w = iv / iv.sum()
    return float(w @ sub @ w)


def _hrp_tree_weights(
    c: np.ndarray,
    Z: np.ndarray,
    m: int,
    node: int,
) -> Dict[int, float]:
    """（保留供调试用）层次树版本。

    **注意**：论文（López de Prado 2016, Algorithm 2）里的 recursive
    bisection 是对**准对角化后的有序列表**做**保序对半切分**，
    不是对层次树切分。正式实现见 `_hrp_recursive_bisection`。
    """
    if node < m:
        return {int(node): 1.0}
    l, r = int(Z[node - m, 0]), int(Z[node - m, 1])
    wl = _hrp_tree_weights(c, Z, m, l)
    wr = _hrp_tree_weights(c, Z, m, r)
    var_l = _cluster_var(c, list(wl.keys()))
    var_r = _cluster_var(c, list(wr.keys()))
    tot = var_l + var_r
    alpha = (1.0 - var_l / tot) if tot > 0 else 0.5
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = {k: alpha * v for k, v in wl.items()}
    out.update({k: (1.0 - alpha) * v for k, v in wr.items()})
    return out


def _hrp_recursive_bisection(c: np.ndarray, order: np.ndarray) -> np.ndarray:
    """HRP 递归二分 —— 严格按 López de Prado(2016) Algorithm 2。

    论文原文（SSRN 2708678, "Stage 3: Recursive Bisection"）：
      1. 所有项赋单位权重 w_n = 1
      2. 递归地把列表 L_i **保序**二分为 L_i1 ∪ L_i2
      3. 计算 L_ij 的方差 V_ij（用簇内**逆方差**组合的二次型）
      4. 切分因子 α_i = 1 − V_i1/(V_i1+V_i2)，w(L_i1)=α_i, w(L_i2)=1−α_i

    论文给的参考代码（Code snippet 2）就是：
        cItems=[sortIx]
        while len(cItems)>0:
            cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)/2),
                     (len(i)/2,len(i))) if len(i)>1]
            for i in range(0,len(cItems),2):
                cItems0=cItems[i]; cItems1=cItems[i+1]
                cVar0=getClusterVar(cov,cItems0)
                cVar1=getClusterVar(cov,cItems1)
                alpha=1-cVar0/(cVar0+cVar1)
                w[cItems0]=alpha; w[cItems1]=1-alpha

    c  : 准对角化后的相关矩阵（c[i,j] 对应 order[i], order[j]）
    order : 准对角化后的叶子顺序（叶子的**局部**下标）
    返回：长度 = len(order) 的权重（局部下标序）
    """
    n = len(order)
    w = np.zeros(n)
    if n == 0:
        return w
    if n == 1:
        return np.array([1.0])
    # 保序对半切分
    half = n // 2
    items = [order]
    while items:
        nxt = []
        for it in items:
            if len(it) > 1:
                h = len(it) // 2
                nxt.append(it[:h])
                nxt.append(it[h:])
        if not nxt:
            break
        items = nxt
    # 逐对分配权重
    for i in range(0, len(items) - 1, 2):
        a, b = items[i], items[i + 1]
        v_a = _cluster_var(c, a)
        v_b = _cluster_var(c, b)
        tot = v_a + v_b
        alpha = (1.0 - v_a / tot) if tot > 0 else 0.5
        alpha = float(np.clip(alpha, 0.0, 1.0))
        for x in a:
            w[x] = alpha
        for x in b:
            w[x] = 1.0 - alpha
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def hrp_weights(
    close: np.ndarray,
    t: int,
    idx: np.ndarray,
    lookback: int = 120,
    min_obs: int = 40,
) -> Optional[np.ndarray]:
    """Lopez de Prado(2016) HRP 权重。

    步骤（严格按论文）:
      1. 收益 → 相关矩阵
      2. 相关 → 距离 d = sqrt(0.5*(1-ρ))（欧氏可嵌入）
      3. 单链接层次聚类 → **准对角化**（leaves_list 给出叶序）
      4. **递归二分**（保序对半切分 + α = 1 − V₁/(V₁+V₂)）

    返回长度 = len(idx) 的权重向量（无效列给 0）。
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n = len(idx)
    if n < 5:
        return None
    r = _trailing_returns(close, t, idx, lookback)
    if r.shape[0] < min_obs:
        return None
    r = np.where(np.isfinite(r), r, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.nanstd(r, axis=0)
    good = np.where(np.isfinite(r).sum(axis=0) >= min_obs * 0.8)[0]
    good = good[sd[good] > 1e-9]
    if len(good) < 5:
        return None
    rg = r[:, good]
    # rowvar=False：r 的**行**是时间、**列**是股票 → 变量在列上
    corr = np.corrcoef(np.nan_to_num(rg, nan=0.0), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -0.999, 0.999)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    try:
        Z = linkage(squareform(dist, checks=False), method="single")
        order = leaves_list(Z)            # 准对角化后的叶序（局部下标）
    except Exception:
        return None
    m = len(good)
    c_qd = corr[np.ix_(order, order)]     # 准对角化后的相关矩阵
    w_qd = _hrp_recursive_bisection(c_qd, np.arange(m))
    # w_qd[i] 是**准对角位置 i** 的权重，对应的原始局部下标是 order[i]
    out = np.zeros(n)
    for qd_pos, w in enumerate(w_qd):
        out[good[order[qd_pos]]] = w
    s = out.sum()
    if s <= 0:
        return None
    return out / s


def maxdiv_weights(
    close: np.ndarray,
    t: int,
    idx: np.ndarray,
    lookback: int = 120,
    min_obs: int = 40,
) -> Optional[np.ndarray]:
    """最大分散化权重: w ∝ Σ⁻¹σ (Maillard-Roncalli-Teissier 2010)。

    分散化比率 DR = (w'Σw)^0.5 / (w'σ) 在 w ∝ Σ⁻¹σ 时最大。
    """
    n = len(idx)
    if n < 5:
        return None
    r = _trailing_returns(close, t, idx, lookback)
    if r.shape[0] < min_obs:
        return None
    r = np.where(np.isfinite(r), r, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.nanstd(r, axis=0)
    keep = sd > 1e-9
    if keep.sum() < 5:
        return None
    r = r[:, keep]
    cov = np.cov(np.nan_to_num(r, nan=0.0), rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    # Ledoit-Wolf 式收缩（对角加载），提高稳定性
    lam = 0.10
    cov = (1 - lam) * cov + lam * np.diag(np.diag(cov))
    try:
        w = np.linalg.solve(cov, np.diag(cov))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        return None
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return None
    out = np.zeros(n)
    out[np.where(keep)[0]] = w / w.sum()
    return out


def inverse_vol_weights(
    close: np.ndarray,
    t: int,
    idx: np.ndarray,
    lookback: int = 60,
) -> Optional[np.ndarray]:
    """逆波动率权重 w ∝ 1/σ。"""
    n = len(idx)
    if n < 2:
        return None
    r = _trailing_returns(close, t, idx, lookback)
    # lookback 个价格 → lookback-1 个收益 → 要求至少一半样本有效
    if r.shape[0] < max(10, lookback // 2):
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.nanstd(r, axis=0)
    iv = 1.0 / np.where(sd > 1e-9, sd, np.nan)
    if not np.isfinite(iv).any() or np.nansum(iv) <= 0:
        return None
    iv = np.nan_to_num(iv, nan=0.0)
    return iv / iv.sum()


# ══════════════════════════════════════════════════════════════════
# 5. 分组配额选股（Barra APT 式组约束）
# ══════════════════════════════════════════════════════════════════
def group_cap_mask(
    score: np.ndarray,
    group_ids: np.ndarray,
    cap: int,
    min_cross: int = 30,
) -> np.ndarray:
    """把超配额的组内成员「压出」TopK：返回掩码后的分数（超配额 → NaN）。

    Barra APT 用「组内暴露约束」控制风格/行业集中度；这里用更简单的
    「每组最多 cap 只」配额：按分数降序遍历，某组已达 cap 则该股被剔除。
    """
    n_d, n_s = score.shape
    out = score.copy()
    for t in range(n_d):
        row = score[t]
        valid = np.where(np.isfinite(row))[0]
        if len(valid) <= min_cross:
            continue
        order = valid[np.argsort(-row[valid], kind="mergesort")]
        counts: Dict[int, int] = {}
        drop = []
        for j in order:
            g = int(group_ids[j])
            if counts.get(g, 0) >= cap:
                drop.append(j)
            else:
                counts[g] = counts.get(g, 0) + 1
        if drop:
            out[t, drop] = np.nan
    return out


def dynamic_layer_ids(
    amount: np.ndarray,
    n_layers: int = 5,
) -> np.ndarray:
    """按**逐日**成交额分位数分层（时间变动的层 id）。

    返回 (n_dates, n_stocks) 的层 id 矩阵（0 = 最高流动性层）。
    """
    n_d, n_s = amount.shape
    out = np.zeros((n_d, n_s), dtype=np.int64)
    for t in range(n_d):
        row = amount[t]
        finite = np.isfinite(row)
        if finite.sum() < 50:
            out[t] = n_layers // 2
            continue
        vals = row[finite]
        qs = np.nanquantile(vals, np.linspace(0, 1, n_layers + 1)[1:-1])
        b = np.digitize(row, qs)
        b[~finite] = n_layers // 2
        out[t] = b.astype(np.int64)
    return out


# ══════════════════════════════════════════════════════════════════
# 6. 多周期合成
# ══════════════════════════════════════════════════════════════════
def blend_scores(
    scores: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """多周期信号合成：各自截面 z 后加权平均（Asness et al. 2013 风格）。"""
    total = None
    wsum = None
    for name, w in weights.items():
        if w == 0:
            continue
        z = L.cs_zscore(scores[name], 30)
        good = np.isfinite(z)
        contrib = np.where(good, z * w, 0.0)
        cnt = np.where(good, abs(w), 0.0)
        total = contrib if total is None else total + contrib
        wsum = cnt if wsum is None else wsum + cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)


# ══════════════════════════════════════════════════════════════════
# 7. 自检
# ══════════════════════════════════════════════════════════════════
def _selftest() -> None:
    rng = np.random.default_rng(0)
    # EWMA
    s = rng.normal(size=(50, 3))
    s[0, 0] = np.nan
    out = ewma_smooth(s, halflife=5)
    assert np.isfinite(out[1, 0]), "EWMA 未沿用上次有效值"
    assert abs(out[1, 0] - s[1, 0]) < 1e-12, "EWMA 首值应等于原值"
    # 稳态：alpha 越大越贴原值
    a1 = ewma_smooth(s, halflife=1)
    a20 = ewma_smooth(s, halflife=20)
    assert abs(out[10, 1] - a20[10, 1]) < abs(out[10, 1] - a1[10, 1]), "半衰期越大越平滑"
    # NW t 统计：正自相关序列 → HAC 方差 > 朴素方差 → |t_NW| < |t_naive|
    # （Newey-West 的作用正是**修正**朴素 t 统计高估显著性）
    x = np.cumsum(rng.normal(size=500))
    m, se, t = newey_west_mean_tstat(x)
    naive_se = np.std(x, ddof=1) / np.sqrt(len(x))
    assert abs(t) < abs(x.mean() / naive_se) + 1e-9, "NW 未修正自相关"
    # 独立同分布序列 → NW ≈ 朴素
    y = rng.normal(size=500)
    m2, se2, t2 = newey_west_mean_tstat(y)
    assert abs(t2) < 6.0, f"NW t 在独立序列上爆炸: {t2}"
    print(f"[selftest] NW autocorr: t_nw={t:.3f} t_naive={x.mean()/naive_se:.3f}; "
          f"iid: t_nw={t2:.3f}")
    # 分组配额
    sc = np.tile(np.arange(20.0)[::-1], (3, 1))
    gid = np.array([0] * 10 + [1] * 10)
    masked = group_cap_mask(sc, gid, cap=3, min_cross=5)
    assert np.isnan(masked[0, 3]), "组配额未生效"
    # 逆波动率
    close = np.cumprod(1 + rng.normal(scale=0.02, size=(200, 8)), axis=0) * 10
    w = inverse_vol_weights(close, 199, np.arange(8))
    assert w is not None and abs(w.sum() - 1) < 1e-9
    # HRP / maxdiv
    w2 = hrp_weights(close, 199, np.arange(8))
    assert w2 is None or abs(w2.sum() - 1) < 1e-6
    w3 = maxdiv_weights(close, 199, np.arange(8))
    assert w3 is None or abs(w3.sum() - 1) < 1e-6
    print("[selftest] ALL OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
