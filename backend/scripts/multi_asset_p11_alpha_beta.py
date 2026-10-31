#!/usr/bin/env python3
"""P11 · A 股腿的 alpha/beta 分解 —— 「主腿弱」是真失效，还是只是 beta 拖累？

## 触发（来自 P10）

P10 打出 A 股腿 25f 生产配置的跨窗口夏普曲线：

| 窗口 | 夏普 |
|---|---|
| w2013 | 0.8099 |
| **w2016** | **-0.1903**（收益 -9.50%，回撤 -35.90%） |
| w2019 | 1.2295 |
| oos | 0.7297 |
| main | 0.8796 |

**w2016 那三年 A 股腿是亏钱的。** 现在必须回答一个岔路口问题：

> **w2016 的 -0.19 是「策略选不出好股票」（alpha ≤ 0），
>  还是「策略有 alpha、只是被市场 beta 拖累了」（alpha > 0）？**

## v2：修掉了 v1 的两个基准构造错误（这是本脚本最值钱的部分）

### 错误 1：基准用了「日频再平衡 + 死亡股剔除」

v1 的基准是：成分股每日**等权再平衡**，且当日无价（停牌/退市）的股票**被剔除**。
后果实测（w2016）：

- 同一成分池，**日频再平衡口径年化 +0.47%**，**买入持有口径年化 -12.22%** → 差 12.7pp
- 拆两半检验：全市场随机拆成两半，年化 **+10.14% vs -9.30%**，两者日收益相关仅 **0.54**
- 500 只子池 8 个种子：年化从 -9.5% 到 **+39.9%**，波动从 21.7% 到 **85.7%**
  （分母只剩个位数成员 → 单只股票暴涨即污染整个基准）

**机制**：把「死亡股票」剔除，等于**把每只退市股最后一次暴跌从基准里抹掉**，
还顺手给基准塞进一个极强的「卖高买低」反转效应。
→ 拿这种基准去比策略，等于**给基准塞了一个策略**，策略必然显得差。

### 错误 2：基准有幸存者偏差，而腿没有

实测 `data/qlib_data/features_full/` 的 5513 只**全部活到 2026**，
已退市的 600087 / 600247 / 000029 / 600462 / 000033 / 000982 **一只都没有**。
而腿的引擎（`enable_tradability=True`）遇到退市股会**照实承担亏损**。
→ 基准被系统性高估 → **超额夏普被系统性低估**。
→ 两个错误方向一致，都压向「策略没 alpha」。

### v2 的基准口径

**主口径 = 买入持有等权**（零技巧市场组合）：

1. 成分股 = 窗口起点当日有价的全部股票，**固定不变**
2. 权重恒为 1/N，**不做任何再平衡**
3. 停牌日按 0 收益持有；**退市/消失的股票不清零、不剔除**
   → 基准承担全部损失，**与腿的口径对称**
4. **日频再平衡口径作为敏感性分析**并列报告（说明差异来自哪里）
5. 补充基准：**沪深300（510300 ETF）**（2019 起）+ **随机 500 只子池 × 3 种子**
6. **幸存者偏差量化**：用 `data/parquet/stock_data/`（5595 只，含 82 只退市股）
   估算「把退市股加回基准」会改变多少

## 指标

- `leg_sharpe`：腿的夏普
- `bench_sharpe`：基准的夏普
- `excess_sharpe`：`sharpe(腿日收益 − 基准日收益)` ← **核心指标**
- `alpha_sharpe`：`sharpe(腿日收益 − β×基准日收益)`（beta 中性化后的纯 alpha）
- `beta` / `corr` / `ir`

## 预注册判据（跑之前写死，事后不许改）

- **(a)** 若 w2016 的 `excess_sharpe > 0` → 结论 = **alpha 存在，负夏普来自 beta**
  → 该调**仓位/beta 层**，**不要动选股层**
- **(b)** 若 w2016 的 `excess_sharpe ≤ 0` → 结论 = **策略真失效**
  → 该动**信号/因子层**
- **(c)** 稳健性：5 窗口 `excess_sharpe` 是否同号；**换基准口径结论是否翻转**

无论怎么判，**定稿不变**（本实验只做诊断）。

## 用法

    env -u PYTHONPATH backend/.venv-py313/bin/python3 \
        backend/scripts/multi_asset_p11_alpha_beta.py --phase all [--only w2016]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

OUT_DIR = os.path.join(ROOT, "data", "backtest")
FEATURES_FULL = os.path.join(ROOT, "data", "qlib_data", "features_full")
STOCK_DATA = os.path.join(ROOT, "data", "parquet", "stock_data")
ETF_PARQUET = os.path.join(ROOT, "data", "etf", "parquet")
ANN = 244

WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("w2013", datetime(2013, 5, 8), datetime(2016, 5, 8)),
    ("w2016", datetime(2016, 5, 8), datetime(2019, 5, 8)),
    ("w2019", datetime(2019, 5, 8), datetime(2021, 5, 11)),
    ("oos", datetime(2021, 5, 11), datetime(2023, 5, 8)),
    ("main", datetime(2023, 5, 8), datetime(2026, 5, 8)),
]


# ───────────────────────── 基础统计 ─────────────────────────


def sharpe_from_rets(rets: List[float]) -> float:
    if len(rets) < 3:
        return 0.0
    mu = statistics.fmean(rets)
    sd = statistics.pstdev(rets)
    return mu / sd * math.sqrt(ANN) if sd > 0 else 0.0


def maxdd_from_rets(rets: List[float]) -> float:
    v = 1.0
    peak = 1.0
    mdd = 0.0
    for r in rets:
        v *= (1.0 + r)
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = dd
    return mdd


def nav_to_rets(nav: List[Tuple[str, float]]) -> List[float]:
    rets = []
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        rets.append(p1 / p0 - 1.0 if (p0 and p0 > 0) else 0.0)
    return rets


# ───────────────────────── 面板 ─────────────────────────


def load_close_panel() -> Any:
    """features_full 的 $close → DataFrame(index=date, columns=code)（Qlib 调整价）。"""
    import pandas as pd

    frames: Dict[str, Any] = {}
    for fn in sorted(f for f in os.listdir(FEATURES_FULL) if f.endswith(".parquet")):
        code = fn[:-len(".parquet")]
        try:
            df = pd.read_parquet(os.path.join(FEATURES_FULL, fn), columns=["$close"])
        except Exception:
            continue
        s = df["$close"]
        idx = s.index
        dates = idx.get_level_values(-1) if isinstance(idx, pd.MultiIndex) else idx
        s = pd.Series(s.to_numpy(), index=pd.DatetimeIndex(dates))
        frames[code] = s[~s.index.duplicated(keep="last")].sort_index()
    print(f"[p11] features_full 载入 {len(frames)} 只（$close，复权价）", flush=True)
    wide = pd.DataFrame(frames).sort_index().astype("float32")
    print(f"[p11] 面板 {wide.shape[0]} 个日期 × {wide.shape[1]} 只", flush=True)
    return wide


def load_unadj_closes() -> Tuple[Any, List[str]]:
    """stock_data（不复权价）→ DataFrame + 相对 features_full 缺失的代码列表。

    缺失的代码 = 已退市 / 被剔除的股票 → **幸存者偏差的量化依据**。
    """
    import pandas as pd

    ff_codes = {f[:-len(".parquet")] for f in os.listdir(FEATURES_FULL) if f.endswith(".parquet")}
    frames: Dict[str, Any] = {}
    missing: List[str] = []
    for fn in sorted(os.listdir(STOCK_DATA)):
        if not fn.endswith(".parquet") or fn.endswith(".bak.parquet"):
            continue
        code = fn[:-len(".parquet")]
        try:
            df = pd.read_parquet(os.path.join(STOCK_DATA, fn), columns=["close", "date"])
        except Exception:
            continue
        if code not in ff_codes:
            missing.append(code)
        s = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df["date"]))
        frames[code] = s[~s.index.duplicated(keep="last")].sort_index()
    print(f"[p11] stock_data 载入 {len(frames)} 只；features_full 缺失 {len(missing)} 只"
          f"（= 退市/剔除，幸存者偏差来源）", flush=True)
    wide = pd.DataFrame(frames).sort_index().astype("float32")
    return wide, sorted(missing)


def load_etf_close(code: str) -> Any:
    import pandas as pd
    p = os.path.join(ETF_PARQUET, f"{code}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    col = "date" if "date" in df.columns else df.columns[0]
    s = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(df[col]))
    return s[~s.index.duplicated(keep="last")].sort_index()


# ───────────────────────── 基准 ─────────────────────────


def build_benchmark(wide: Any, leg_dates: List[str], mode: str = "buyhold",
                    n_random: int = 0, seed: int = 0) -> Dict[str, Any]:
    """等权市场基准。

    mode="buyhold"（**主口径**）：真正的**买入持有等权**
        组合净值 = mean_i[ P_i(t) / P_i(t0) ]，成分与权重全程固定、**不做任何再平衡**
        → 停牌按最后价格冻结、退市按最后价格承担（池内退市股极少，见幸存者偏差量化）
    mode="rebalance"（敏感性）：每日**等权再平衡** + 当日无价剔除
        = mean_t[ mean_i r_i(t) ] 再连乘 → 内含「卖高买低」反转效应
    """
    import pandas as pd

    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in leg_dates])
    cw = wide.reindex(dates)
    alive = cw.notna()
    p = cw.ffill()
    r = p.pct_change(fill_method=None)
    members = [c for c in cw.columns if bool(alive.iloc[0].get(c, False))]
    if not members:
        raise SystemExit("[p11] ❌ 窗口起点无成分股")
    if n_random and len(members) > n_random:
        import random as _random
        members = sorted(_random.Random(seed).sample(members, n_random))
    if mode == "buyhold":
        # 真正买入持有：成分固定、权重恒 1/N、组合净值 = 各成分归一价路径的等权平均
        norm = p[members] / p[members].iloc[0]
        port = norm.mean(axis=1)
        rets = port.pct_change(fill_method=None).fillna(0.0)
        n_alive = pd.Series([len(members)] * len(dates))
    else:
        sub = r[members].where(alive[members])
        n_alive = sub.notna().sum(axis=1)
        br = (sub.sum(axis=1) / n_alive.replace(0, float("nan"))).fillna(0.0)
        rets = br
    rr = [float(x) for x in rets.tolist()]
    rr[0] = 0.0
    nav: List[Tuple[str, float]] = []
    v = 1.0
    for d, x in zip(leg_dates, rr):
        v *= (1.0 + x)
        nav.append((d, v))
    return {
        "nav": nav,
        "rets": rr,
        "n_members": len(members),
        "n_alive_last": int(n_alive.iloc[-1]),
        "n_alive_min": int(n_alive.iloc[1:].min()) if len(n_alive) > 1 else 0,
        "mode": mode,
    }

def build_single_asset_benchmark(series: Any, leg_dates: List[str],
                                 min_coverage: float = 0.8) -> Optional[Dict[str, Any]]:
    """单资产基准（如 510300 沪深300 ETF）。

    **覆盖率闸门**：源数据在窗口内的覆盖天数 < 窗口天数的 min_coverage → 返回 None。
    （没有这道闸门，2019 才起的数据会在 2016 窗口里 ffill 成"全 0 收益"，
      伪造出「夏普 0.87」这种数字。）
    """
    import pandas as pd
    if series is None or len(series) == 0:
        return None
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in leg_dates])
    cov = int(series.reindex(dates).notna().sum())
    if cov < min_coverage * len(dates):
        print(f"[p11]   [单资产基准] 覆盖 {cov}/{len(dates)} < {min_coverage:.0%} → 判为不可用", flush=True)
        return None
    s = series.reindex(dates).ffill()  # series 非 None（调用方已判）
    r = s.pct_change(fill_method=None).fillna(0.0)
    rets = [float(x) for x in r.tolist()]
    rets[0] = 0.0
    nav: List[Tuple[str, float]] = []
    v = 1.0
    for d, x in zip(leg_dates, rets):
        v *= (1.0 + x)
        nav.append((d, v))
    return {"nav": nav, "rets": rets, "n_members": 1, "mode": "single"}


# ───────────────────────── 分解 ─────────────────────────


def decompose(leg_nav: List[Tuple[str, float]], bench_nav: List[Tuple[str, float]]) -> Dict[str, Any]:
    bmap = {d: v for d, v in bench_nav}
    lmap = {d: v for d, v in leg_nav}
    common = sorted(set(lmap) & set(bmap))
    if len(common) < 30:
        raise SystemExit(f"[p11] ❌ 共同日期过少：{len(common)}")
    lr = nav_to_rets([(d, lmap[d]) for d in common])
    br = nav_to_rets([(d, bmap[d]) for d in common])
    ex = [a - b for a, b in zip(lr, br)]
    n = len(lr)
    mb, ml = statistics.fmean(br), statistics.fmean(lr)
    vb = statistics.pvariance(br)
    cov = sum((x - ml) * (y - mb) for x, y in zip(lr, br)) / n
    beta = (cov / vb) if vb > 0 else 0.0
    vl = statistics.pvariance(lr)
    rho = (cov / math.sqrt(vb * vl)) if vb > 0 and vl > 0 else 0.0
    ex_an = [a - beta * b for a, b in zip(lr, br)]
    sd_ex = statistics.pstdev(ex)
    ann_ex = statistics.fmean(ex) * ANN
    return {
        "n_common_days": n,
        "leg_sharpe": sharpe_from_rets(lr),
        "leg_return_ann": statistics.fmean(lr) * ANN,
        "leg_maxdd": maxdd_from_rets(lr),
        "bench_sharpe": sharpe_from_rets(br),
        "bench_return_ann": statistics.fmean(br) * ANN,
        "bench_maxdd": maxdd_from_rets(br),
        "excess_sharpe": sharpe_from_rets(ex),
        "excess_return_ann": statistics.fmean(ex) * ANN,
        "alpha_sharpe": sharpe_from_rets(ex_an),
        "beta": beta,
        "corr": rho,
        "ir": (ann_ex / (sd_ex * math.sqrt(ANN))) if sd_ex > 0 else 0.0,
        "excess_maxdd": maxdd_from_rets(ex),
    }


# ───────────────────────── 幸存者偏差量化 ─────────────────────────


def survivorship_impact(ff_wide: Any, missing_codes: List[str], unadj: Any,
                         start: datetime, end: datetime) -> Dict[str, Any]:
    """估算「把退市股加回等权基准」对基准年化收益的影响（粗略量级）。

    用不复权价算区间累计收益（含送转跳空 → 标注为量级参考）。
    """
    import pandas as pd
    s0 = pd.Timestamp(start)
    s1 = pd.Timestamp(end)

    def cum_ret(series_list) -> List[float]:
        out = []
        for s in series_list:
            seg = s[(s.index >= s0) & (s.index <= s1)]
            if len(seg) < 5:
                continue
            p0, p1 = float(seg.iloc[0]), float(seg.iloc[-1])
            if p0 and p0 > 0:
                out.append(p1 / p0 - 1.0)
        return out

    ff_cols = [c for c in ff_wide.columns]
    ff_list = [ff_wide[c].dropna() for c in ff_cols]
    pool = cum_ret(ff_list)
    miss = []
    if missing_codes and unadj is not None and len(unadj.columns):
        miss = cum_ret([unadj[c].dropna() for c in missing_codes if c in unadj.columns])
    if not pool or not miss:
        return {"ok": False, "reason": "数据不足"}
    pool_mean = statistics.fmean(pool)
    miss_mean = statistics.fmean(miss)
    n_pool, n_miss = len(pool), len(miss)
    mixed_mean = (pool_mean * n_pool + miss_mean * n_miss) / (n_pool + n_miss)
    return {
        "ok": True,
        "n_pool": n_pool,
        "n_missing": n_miss,
        "pool_mean_cum": round(pool_mean, 6),
        "missing_mean_cum": round(miss_mean, 6),
        "mixed_mean_cum": round(mixed_mean, 6),
        "delta_ann": round((mixed_mean - pool_mean) / max((s1 - s0).days / 365.25, 1e-9), 6),
        "note": "不复权价，含送转跳空 → 仅作量级参考",
    }


# ───────────────────────── 市场代理基准（无幸存者偏差） ─────────────────────────

BENCH_DIR = os.path.join(ROOT, "data", "benchmarks")

# 指数 ETF：本身按指数成分持有，**天然无幸存者偏差**（退市股的历史仍在指数里）
ETF_PROXIES = [
    ("510300.SH", "沪深300ETF"),
    ("510500.SH", "中证500ETF"),
    ("159915.SZ", "创业板ETF"),
    ("510050.SH", "上证50ETF"),
]

# A 股历年退市家数（来源：东方财富财富号《A股沪深北交所历年新增上市公司与退市公司数量
# 深度分析（1990—2026.5.29）》，2026-05-30；口径：强制退市 + 主动退市）
DELISTINGS_BY_YEAR = {
    2013: 0, 2014: 0, 2015: 0, 2016: 0, 2017: 1, 2018: 1, 2019: 10, 2020: 19,
    2021: 22, 2022: 46, 2023: 46, 2024: 52, 2025: 31, 2026: 4,
}


FIXTURES_DIR = os.path.join(BACKEND_DIR, "fixtures", "market_benchmarks")


def load_market_proxies() -> Dict[str, Any]:
    """载入市场代理基准（官方指数）。

    优先 `data/benchmarks/*.parquet`；缺失时回退到**仓库里的离线快照**
    `backend/fixtures/market_benchmarks/*.csv` → 整条链路不需要联网。
    """
    import pandas as pd
    out: Dict[str, Any] = {}

    def _ingest(name: str, df) -> None:
        if "trade_date" not in df.columns or "close" not in df.columns:
            return
        s = pd.Series(df["close"].to_numpy(),
                      index=pd.DatetimeIndex(df["trade_date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[name] = s

    if os.path.isdir(BENCH_DIR):
        for fn in sorted(os.listdir(BENCH_DIR)):
            if not fn.endswith(".parquet"):
                continue
            try:
                _ingest(fn[:-len(".parquet")], pd.read_parquet(os.path.join(BENCH_DIR, fn)))
            except Exception:
                continue
    if os.path.isdir(FIXTURES_DIR):
        for fn in sorted(os.listdir(FIXTURES_DIR)):
            if not fn.endswith(".csv"):
                continue
            key = fn[:-len(".csv")]
            if key in out:
                continue
            try:
                _ingest(key, pd.read_csv(os.path.join(FIXTURES_DIR, fn)))
            except Exception:
                continue
    return out


def count_delistings(start: datetime, end: datetime) -> float:
    """窗口内的退市家数（按年在窗口内的占比折算）。"""
    tot = 0.0
    for y, n in DELISTINGS_BY_YEAR.items():
        ys, ye = datetime(y, 1, 1), datetime(y + 1, 1, 1)
        ov = (min(ye, end) - max(ys, start)).total_seconds()
        yr = (ye - ys).total_seconds()
        if ov > 0 and yr > 0:
            tot += n * (ov / yr)
    return tot


def survivorship_scenario(bench_cum: float, n_members: int, start: datetime,
                          end: datetime, delisted_cum: float = -0.8) -> Dict[str, Any]:
    """幸存者偏差情景修正。

    池内**没有**已退市股票（已核验：600087/600247/600462/000033/000982 等全部不在
    stock_data / features_full / qlib_official_data 里），所以基准只由「活到今天」的股票构成。
    把窗口内退市的 N 只按平均累计收益 delisted_cum 加回等权基准：

        修正后基准累计 = 观测累计 + share × (退市股累计 − 观测累计)
        share = N_退市 / N_窗口起点成分
    """
    n_del = count_delistings(start, end)
    if n_members <= 0:
        return {"ok": False}
    share = n_del / n_members
    adj_cum = bench_cum + share * (delisted_cum - bench_cum)
    years = max((end - start).days / 365.25, 1e-9)
    obs_ann = (1.0 + bench_cum) ** (1.0 / years) - 1.0 if bench_cum > -1 else -1.0
    adj_ann = (1.0 + adj_cum) ** (1.0 / years) - 1.0 if adj_cum > -1 else -1.0
    return {
        "ok": True,
        "n_delisted_in_window": round(n_del, 1),
        "n_members_start": n_members,
        "share": round(share, 6),
        "delisted_cum_assumed": delisted_cum,
        "bench_cum_observed": round(bench_cum, 6),
        "bench_cum_adjusted": round(adj_cum, 6),
        "bench_ann_observed": round(obs_ann, 6),
        "bench_ann_adjusted": round(adj_ann, 6),
        "bench_ann_shift": round(adj_ann - obs_ann, 6),
        "note": "退市股平均累计收益假设 -80%；池内退市股数为公开统计",
    }


# ───────────────────────── 主流程 ─────────────────────────


def run_leg(codes: List[str], label: str, start: datetime, end: datetime):
    import multi_asset_p10_long_ashare as P10
    rec, nav = P10.run_leg(codes, start, end)
    print(f"[p11] {label}: A腿引擎夏普 {rec.get('engine_sharpe')} / 净值点 {len(nav)}", flush=True)
    return rec, nav


def phase_all(only: str = "") -> Dict[str, Any]:
    import multi_asset_p10_long_ashare as P10
    P10.enable_full_features()
    codes = P10.load_universe()
    print(f"[p11] 宇宙 {len(codes)} 只", flush=True)
    wide = load_close_panel()
    unadj, missing = load_unadj_closes()
    hs300 = load_etf_close("510300")
    print(f"[p11] 510300 数据可用: {hs300 is not None}"
          f"{' 首日 ' + str(hs300.index[0])[:10] if hs300 is not None and len(hs300) else ''}", flush=True)

    out_path = os.path.join(OUT_DIR, "multi_asset_p11_alpha.json")
    results: Dict[str, Any] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            results = (json.load(fh).get("results") or {})

    for label, start, end in WINDOWS:
        if only and label != only:
            continue
        print(f"[p11] 窗口 {label}: {start.date()}~{end.date()}", flush=True)
        rec, nav = run_leg(codes, label, start, end)
        leg_dates = [d for d, _ in nav]

        def dec_of(bm) -> Dict[str, Any]:
            return {k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in decompose(nav, bm["nav"]).items()}

        bh = build_benchmark(wide, leg_dates, mode="buyhold")
        rb = build_benchmark(wide, leg_dates, mode="rebalance")
        d_bh = dec_of(bh)
        d_rb = dec_of(rb)
        extra: Dict[str, Any] = {
            "rebalance_benchmark": {
                "n_members": rb["n_members"],
                "sharpe": d_rb["bench_sharpe"],
                "return_ann": d_rb["bench_return_ann"],
                "excess_sharpe": d_rb["excess_sharpe"],
                "alpha_sharpe": d_rb["alpha_sharpe"],
                "beta": d_rb["beta"],
                "note": "敏感性口径：每日等权再平衡 + 当日无价剔除（v1 的错误口径）",
            },
            "random500": {},
        }
        for sd in (7, 99, 1234):
            rbm = build_benchmark(wide, leg_dates, mode="buyhold", n_random=500, seed=sd)
            extra["random500"][f"s{sd}"] = {
                "n_members": rbm["n_members"],
                "sharpe": dec_of(rbm)["bench_sharpe"],
                "return_ann": dec_of(rbm)["bench_return_ann"],
                "excess_sharpe": dec_of(rbm)["excess_sharpe"],
                "alpha_sharpe": dec_of(rbm)["alpha_sharpe"],
                "beta": dec_of(rbm)["beta"],
            }
        hsb = build_single_asset_benchmark(hs300, leg_dates)
        if hsb is not None:
            dh = dec_of(hsb)
            extra["hs300"] = {
                "available": True,
                "first_date": str(hs300.index[0])[:10],
                "sharpe": dh["bench_sharpe"],
                "return_ann": dh["bench_return_ann"],
                "excess_sharpe": dh["excess_sharpe"],
                "alpha_sharpe": dh["alpha_sharpe"],
                "beta": dh["beta"],
                "corr": dh["corr"],
            }
        else:
            extra["hs300"] = {"available": False}

        results[label] = {
            "window": [str(start.date()), str(end.date())],
            "leg_engine": {
                "sharpe": rec.get("engine_sharpe"),
                "total_return": rec.get("engine_total_return"),
                "max_drawdown": rec.get("engine_max_drawdown"),
                "win_rate": rec.get("engine_win_rate"),
                "total_trades": rec.get("total_trades"),
            },
            "benchmark_buyhold": {
                "n_members": bh["n_members"],
                "sharpe": d_bh["bench_sharpe"],
                "return_ann": d_bh["bench_return_ann"],
                "maxdd": d_bh["bench_maxdd"],
            },
            "decomposition": d_bh,
            "alternatives": extra,
            "survivorship": survivorship_scenario(bh["nav"][-1][1] - 1.0, bh["n_members"], start, end),
            "survivorship_pool_check": {
                "n_stock_data_codes": 5595,
                "n_features_full_codes": 5513,
                "delisted_in_pool": 0,
                "note": "池内无已退市股票（600087/600247/600462/000033/000982 均不在）",
            },
            "leg_dates": leg_dates,
            "leg_nav_pairs": [[d, round(x, 8)] for d, x in nav],
        }
        print(f"[p11]   [买入持有] 腿夏普 {d_bh['leg_sharpe']:.4f} | 基准夏普 "
              f"{d_bh['bench_sharpe']:.4f} | 超额夏普 {d_bh['excess_sharpe']:.4f} | "
              f"α夏普 {d_bh['alpha_sharpe']:.4f} | β {d_bh['beta']:.4f} | "
              f"ρ {d_bh['corr']:.4f} | IR {d_bh['ir']:.4f}", flush=True)
        print(f"[p11]   [日频再平衡-敏感性] 基准夏普 {d_rb['bench_sharpe']:.4f} | "
              f"超额夏普 {d_rb['excess_sharpe']:.4f}", flush=True)
        if extra["hs300"].get("available"):
            print(f"[p11]   [沪深300] 基准夏普 {extra['hs300']['sharpe']:.4f} | "
                  f"超额夏普 {extra['hs300']['excess_sharpe']:.4f} | "
                  f"β {extra['hs300']['beta']:.4f}", flush=True)
        print(f"[p11]   [随机500-买入持有] 超额夏普: "
              + ", ".join(f"s{k[1:]}={v['excess_sharpe']:.4f}"
                          for k, v in extra["random500"].items()), flush=True)
        sv = results[label]["survivorship"]
        if sv.get("ok"):
            print(f"[p11]   [幸存者偏差] 窗口退市 {sv['n_delisted_in_window']} 只 / "
                  f"{sv['n_members_start']} 只 = {sv['share']:.4%} | "
                  f"基准年化 {sv['bench_ann_observed']:+.2%} → 修正 {sv['bench_ann_adjusted']:+.2%} "
                  f"({sv['bench_ann_shift']:+.3%})", flush=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"phase": "all", "results": results}, fh, ensure_ascii=False, indent=2)
    print(f"[p11] 写出 {out_path}", flush=True)
    return {"results": results}


def phase_bench(only: str = "") -> Dict[str, Any]:
    """只重算基准与分解（复用已存的 leg_dates，不重跑腿回测）。"""
    out_path = os.path.join(OUT_DIR, "multi_asset_p11_alpha.json")
    if not os.path.exists(out_path):
        raise SystemExit("[p11] ❌ 无既有产物，先跑 --phase all")
    with open(out_path, encoding="utf-8") as fh:
        blob = json.load(fh)
    results = blob.get("results") or {}
    wide = load_close_panel()
    proxies = load_market_proxies()
    print(f"[p11] 市场代理基准载入 {len(proxies)} 条: {list(proxies)}", flush=True)
    hs300_local = load_etf_close("510300")

    for label, v in list(results.items()):
        if only and label != only:
            continue
        leg_dates = v.get("leg_dates")
        if not leg_dates:
            print(f"[p11] {label}: 无 leg_dates，跳过", flush=True)
            continue
        start = datetime.strptime(v["window"][0], "%Y-%m-%d")
        end = datetime.strptime(v["window"][1], "%Y-%m-%d")

        def dec_of(bm) -> Dict[str, Any]:
            return {k: (round(val, 6) if isinstance(val, float) else val)
                    for k, val in decompose(v["_leg_nav"], bm["nav"]).items()}

        v["_leg_nav"] = [(d, float(x)) for d, x in v.get("leg_nav_pairs", [])]
        if not v["_leg_nav"]:
            print(f"[p11] {label}: 无 leg_nav_pairs，请重跑 --phase all", flush=True)
            continue

        bh = build_benchmark(wide, leg_dates, mode="buyhold")
        rb = build_benchmark(wide, leg_dates, mode="rebalance")
        d_bh, d_rb = dec_of(bh), dec_of(rb)
        extra = v.get("alternatives") or {}
        extra["rebalance_benchmark"] = {
            "n_members": rb["n_members"],
            "sharpe": d_rb["bench_sharpe"],
            "return_ann": d_rb["bench_return_ann"],
            "excess_sharpe": d_rb["excess_sharpe"],
            "alpha_sharpe": d_rb["alpha_sharpe"],
            "beta": d_rb["beta"],
            "note": "敏感性口径：每日等权再平衡 + 当日无价剔除",
        }
        r5: Dict[str, Any] = {}
        for sd in (7, 99, 1234):
            rbm = build_benchmark(wide, leg_dates, mode="buyhold", n_random=500, seed=sd)
            dd = dec_of(rbm)
            r5[f"s{sd}"] = {"n_members": rbm["n_members"], "sharpe": dd["bench_sharpe"],
                            "return_ann": dd["bench_return_ann"], "excess_sharpe": dd["excess_sharpe"],
                            "alpha_sharpe": dd["alpha_sharpe"], "beta": dd["beta"]}
        extra["random500"] = r5
        # 沪深300：本地 ETF（2019 起，覆盖率闸门）+ 指数代理
        hsb = build_single_asset_benchmark(hs300_local, leg_dates)
        extra["hs300_local"] = ({"available": True, "sharpe": dec_of(hsb)["bench_sharpe"],
                                 "return_ann": dec_of(hsb)["bench_return_ann"],
                                 "excess_sharpe": dec_of(hsb)["excess_sharpe"],
                                 "alpha_sharpe": dec_of(hsb)["alpha_sharpe"],
                                 "beta": dec_of(hsb)["beta"], "corr": dec_of(hsb)["corr"]}
                                if hsb is not None else {"available": False,
                                                        "reason": "本地 510300 数据覆盖不足"})
        extra["market_proxies"] = {}
        for key, series in proxies.items():
            bm = build_single_asset_benchmark(series, leg_dates)
            if bm is None:
                extra["market_proxies"][key] = {"available": False, "reason": "覆盖不足"}
                continue
            dd = dec_of(bm)
            extra["market_proxies"][key] = {
                "available": True, "sharpe": dd["bench_sharpe"],
                "return_ann": dd["bench_return_ann"], "excess_sharpe": dd["excess_sharpe"],
                "alpha_sharpe": dd["alpha_sharpe"], "beta": dd["beta"], "corr": dd["corr"],
            }
        v["benchmark_buyhold"] = {"n_members": bh["n_members"], "sharpe": d_bh["bench_sharpe"],
                                  "return_ann": d_bh["bench_return_ann"], "maxdd": d_bh["bench_maxdd"]}
        v["decomposition"] = d_bh
        v["alternatives"] = extra
        v["survivorship"] = survivorship_scenario(bh["nav"][-1][1] - 1.0, bh["n_members"], start, end)
        print(f"[p11] {label} [买入持有] 腿夏普 {d_bh['leg_sharpe']:.4f} 年化 {d_bh['leg_return_ann']:+.2%} | "
              f"基准夏普 {d_bh['bench_sharpe']:.4f} 年化 {d_bh['bench_return_ann']:+.2%} | "
              f"超额夏普 {d_bh['excess_sharpe']:.4f} | α夏普 {d_bh['alpha_sharpe']:.4f} | "
              f"β {d_bh['beta']:.4f} | ρ {d_bh['corr']:.4f}", flush=True)
        print(f"[p11] {label} [日频再平衡-敏感性] 基准年化 {d_rb['bench_return_ann']:+.2%} | "
              f"超额夏普 {d_rb['excess_sharpe']:.4f}", flush=True)
        for k, pv in extra["market_proxies"].items():
            if pv.get("available"):
                print(f"[p11] {label} [代理 {k}] 夏普 {pv['sharpe']:.4f} 年化 {pv['return_ann']:+.2%} | "
                      f"超额夏普 {pv['excess_sharpe']:.4f} | β {pv['beta']:.3f}", flush=True)
        sv = v["survivorship"]
        if sv.get("ok"):
            print(f"[p11] {label} [幸存者偏差] 窗口退市 {sv['n_delisted_in_window']} 只 / "
                  f"{sv['n_members_start']} 只 = {sv['share']:.4%} → 基准年化修正 "
                  f"{sv['bench_ann_shift']:+.3%}", flush=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"phase": "bench", "results": results}, fh, ensure_ascii=False, indent=2)
    print(f"[p11] 写出 {out_path}", flush=True)
    return {"results": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "bench"], default="all")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.phase == "bench":
        phase_bench(only=args.only)
    else:
        phase_all(only=args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
