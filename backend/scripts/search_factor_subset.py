#!/usr/bin/env python3
"""因子子集搜索（honest_composite 42 因子 → 减法搜索）

## 为什么这样跑（核心设计）

`HonestCompositeStrategy.prepare_backtest_data` 对每个 Alpha158 面板因子都把
**每只股票的 parquet 重读一遍**（17 因子 × 5238 只 ≈ 89k 次 parquet 读），
所以 42 因子版光 prepare 就要 440s（25 因子版只要 11s）。做子集搜索要跑
上千次回测，这条路走不通。

关键性质：**每个因子的 z 分数只依赖它自己**。复合分是
`acc/wsum`，acc = 各因子带符号 z 之和、wsum = 该 (日,股) 上有效的因子个数。
于是：

    42 个 z 矩阵 → 磁盘缓存（一次性，~2 分钟）
    任意子集 S → composite(S) = Σ_{f∈S} z_f / |{f∈S : 有效}|
                  = 逐因子累加，毫秒级

回测本体（引擎 / 可成交性 / TopK-Dropout / 成本 / 限涨掩码 / 调仓日掩码 /
lag）**完全复用官方引擎与官方策略代码**，只把打分那一步换成"从缓存重组"。
基线一致性有硬校验：42 因子子集必须复现官方 0.7992，25 因子子集必须复现
官方 0.8796，否则脚本直接退出。

## 口径（与 verify_10y_pit.py / real42_lag1.json 完全一致）

- 窗口 2023-05-08 → 2026-05-08（3 年真实盘口径）
- lag=1（收盘出信号、下一交易日收盘成交）
- enable_tradability=True（涨停买不进 / 跌停卖不出 / 流动性门槛）
- 股票池 = qlib_official_data/instruments/all.txt（5514 → 有效 ~5238，PIT）
- topk=60, n_drop=5, 周频调仓（周五）, 手续费 0.03% + 印花滑点 0.01%

## OOS 协议

- **IS** = 2023-05-08 → 2025-05-08（贪心选择只看这段）
- **OOS** = 2025-05-08 → 2026-05-08（选完之后才看，从不参与选择）
- 另跑独立窗口（is2y / oos1y）做稳健性交叉验证

## 用法

    cd ~/Projects/stock-platform
    P="env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3"

    $P backend/scripts/search_factor_subset.py --stage cache   --window full3y
    $P backend/scripts/search_factor_subset.py --stage verify  --window full3y
    $P backend/scripts/search_factor_subset.py --stage ablate  --window full3y
    $P backend/scripts/search_factor_subset.py --stage greedy  --window full3y
    $P backend/scripts/search_factor_subset.py --stage weights --window full3y
    $P backend/scripts/search_factor_subset.py --stage report

结果落 `data/backtest/factor_subset_*.json`，z 矩阵缓存落
`data/backtest/factor_subset_cache/<window>/`。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

QLIB_OFFICIAL = os.path.join(ROOT, "data", "qlib_official_data")
CACHE_ROOT = os.path.join(ROOT, "data", "backtest", "factor_subset_cache")
OUT_DIR = os.path.join(ROOT, "data", "backtest")

WINDOW_FULL = ("2023-05-08", "2026-05-08")
WINDOW_IS = ("2023-05-08", "2025-05-08")
WINDOW_OOS = ("2025-05-08", "2026-05-08")
WINDOW_KEYS = {
    "full3y": WINDOW_FULL,
    "is2y": WINDOW_IS,
    "oos1y": WINDOW_OOS,
}

# 官方基线（real42_lag1.json / real_lag1_neutral.json，lag=1，周频）
BASELINE_42F = {
    "sharpe_ratio": 0.7991795250472001,
    "total_return": 0.34779248262737783,
    "max_drawdown": -0.16460289642093642,
    "total_trades": 1080,
}
BASELINE_25F = {
    "sharpe_ratio": 0.8795668740640868,
    "total_return": 0.35505797455182114,
    "max_drawdown": -0.1337121157482497,
    "total_trades": 1190,
}

HC_CFG = {"topk": 60, "n_drop": 5, "rebalance_days": "weekly", "rebalance_weekday": 4}
HC_MPS = 0.015833
OHLCV = ["open", "high", "low", "close", "volume"]

# ─────────────────────────── 因子清单 ───────────────────────────
FAMILY_DESC = {
    "MIN": "诚实公式｜位置：收盘/N日最低价-1（正号，越接近区间最低越好）",
    "MAX": "诚实公式｜位置：收盘/N日最高价-1（负号，离区间最高越远越好）",
    "STD": "诚实公式｜波动：N日收盘标准差/收盘（负号，低波动优先）",
    "CORR": "诚实公式｜量价：close 与 volume 的 N 日滚动相关（负号，价升量缩好）",
    "CORD": "诚实公式｜量变：收益 与 成交量变化 的 N 日滚动相关（负号）",
    "VSUMP": "Alpha158｜量加权价格变化（负号，量价齐升为负贡献）",
    "VSUMN": "Alpha158｜成交量 N 日均值 / 短窗均值（正号，量能放大）",
    "VSUMD": "Alpha158｜成交量 N 日和与短窗和之差（负号）",
    "QTLD": "Alpha158｜收盘在 N 日高低区间中的位置（正号，靠近高点）",
    "IMAX": "Alpha158｜N 日内最高价出现位置（负号，越早越好）",
    "IMIN": "Alpha158｜N 日内最低价出现位置（负号，越早越好）",
    "WVMA": "Alpha158｜成交量加权均价 / 收盘（负号）",
    "VMA": "Alpha158｜N 日平均成交量 / 短窗均量（正号）",
    "VSTD": "Alpha158｜成交量 N 日标准差 / 均量（负号）",
}
FACTOR_META: Dict[str, Dict[str, Any]] = {}
ALL_FACTORS: List[str] = []
HONEST_FACTORS: List[str] = []
PANEL_FACTORS: List[str] = []


def _factor_sets() -> Tuple[List[str], List[str], List[str]]:
    global ALL_FACTORS, HONEST_FACTORS, PANEL_FACTORS, FACTOR_META
    if ALL_FACTORS:
        return ALL_FACTORS, HONEST_FACTORS, PANEL_FACTORS
    from app.services.backtest.strategies import honest_composite_strategy as hcs

    FACTOR_META = {}
    for name, sign in hcs._HONEST_FORMULAS.items():
        fam = "".join(ch for ch in name if not ch.isdigit())
        win = int("".join(ch for ch in name if ch.isdigit()) or 0)
        FACTOR_META[name] = {
            "name": name,
            "group": "honest_formula",
            "family": fam,
            "window": win,
            "sign": sign,
            "desc": f"{FAMILY_DESC[fam]}　窗口={win}日",
        }
    for name, sign in hcs._ALPHAPANEL_FACTORS.items():
        fam = "".join(ch for ch in name if not ch.isdigit())
        win = int("".join(ch for ch in name if ch.isdigit()) or 0)
        FACTOR_META[name] = {
            "name": name,
            "group": "alphapanel",
            "family": fam,
            "window": win,
            "sign": sign,
            "desc": f"{FAMILY_DESC[fam]}　窗口={win}日",
        }
    HONEST_FACTORS = list(hcs._HONEST_FORMULAS.keys())
    PANEL_FACTORS = list(hcs._ALPHAPANEL_FACTORS.keys())
    ALL_FACTORS = HONEST_FACTORS + PANEL_FACTORS
    return ALL_FACTORS, HONEST_FACTORS, PANEL_FACTORS


# ─────────────────────────── 基础设施 ───────────────────────────
def universe_from_instruments(instruments_dir: str) -> List[str]:
    """与 verify_10y_pit.py 完全一致的 PIT 股票池（带上市/退市日）。"""
    path = os.path.join(instruments_dir, "instruments", "all.txt")
    codes: List[str] = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if not parts:
                    continue
                name = parts[0]
                if "_" not in name:
                    continue
                code, _, mkt = name.partition("_")
                if mkt.upper() in ("SH", "SZ", "BJ") and code.isdigit():
                    codes.append(f"{code}.{mkt.upper()}")
    if codes:
        return sorted(codes)
    fdir = os.path.join(instruments_dir, "features")
    for f in sorted(os.listdir(fdir)):
        if not f.endswith(".parquet"):
            continue
        stem = f[: -len(".parquet")]
        if "_" not in stem:
            continue
        code, _, mkt = stem.partition("_")
        if mkt.upper() in ("SH", "SZ", "BJ") and code.isdigit():
            codes.append(f"{code}.{mkt.upper()}")
    return sorted(codes)


def window_dates(key: str) -> Tuple[datetime, datetime]:
    s, e = WINDOW_KEYS[key]
    return datetime.strptime(s, "%Y-%m-%d"), datetime.strptime(e, "%Y-%m-%d")


def cache_dir_for(window_key: str) -> str:
    return os.path.join(CACHE_ROOT, window_key)


def _ordered_stock_data(
    stock_data: Dict[str, pd.DataFrame], codes_all: Sequence[str]
) -> Dict[str, pd.DataFrame]:
    """按 sorted(codes_all) 顺序重建 stock_data（与回测引擎里的顺序一致）。"""
    return {c: stock_data[c] for c in codes_all if c in stock_data}


def _build_matrices(stock_data: Dict[str, pd.DataFrame]):
    """复刻官方 prepare_backtest_data 的 master 索引与 close/high/low/vol 矩阵。"""
    codes = [
        code
        for code, df in stock_data.items()
        if df is not None
        and not df.empty
        and "close" in df.columns
        and "high" in df.columns
        and "low" in df.columns
        and "volume" in df.columns
    ]
    indexes = [pd.DatetimeIndex(stock_data[c].index) for c in codes]
    master = indexes[0].append(indexes[1:]) if len(indexes) > 1 else indexes[0]
    master = master.unique().sort_values()
    n_dates, n_stocks = len(master), len(codes)

    def to_matrix(column: str) -> np.ndarray:
        out = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
        for j, code in enumerate(codes):
            df = stock_data[code]
            positions = master.get_indexer(pd.DatetimeIndex(df.index))
            ok = positions >= 0
            values = pd.to_numeric(df[column], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            out[positions[ok], j] = values[ok]
        return out

    frame_opts = {"index": master, "dtype": np.float64}
    close = pd.DataFrame(to_matrix("close"), **frame_opts)
    high = pd.DataFrame(to_matrix("high"), **frame_opts)
    low = pd.DataFrame(to_matrix("low"), **frame_opts)
    vol = pd.DataFrame(to_matrix("volume"), **frame_opts)
    return codes, master, close, high, low, vol


# ── 进程内缓存：master/close/high/low/vol 矩阵与可成交性矩阵都与"因子子集"无关，
#    同一 worker 内复用，可把每次回测的固定开销从 ~45s 压到 ~20s。
#
# 注意：引擎每次 run_backtest 都会重建 stock_data 这个 dict（值对象相同），
# 且代码字符串每次调用都是新对象 → 不能用 id() 做 key，
# 一律用"内容 key"（数量 + 首尾代码 + 窗口端点）。
_MATRIX_CACHE: Dict[Tuple, Tuple] = {}
_TRADABILITY_CACHE: Dict[Tuple, Any] = {}


def _content_key(items: Sequence[Any]) -> Tuple:
    n = len(items)
    if n == 0:
        return (0,)
    return (n, items[0], items[-1])


def _cached_matrices(stock_data: Dict[str, pd.DataFrame]):
    codes = list(stock_data.keys())
    key = _content_key(codes)
    hit = _MATRIX_CACHE.get(key)
    if hit is None:
        hit = _build_matrices(stock_data)
        _MATRIX_CACHE.clear()
        _MATRIX_CACHE[key] = hit
    return hit


def _patch_tradability_cache() -> None:
    """把 build_tradability 换成带缓存的版本（输入相同 → 输出必然相同）。"""
    from app.services.backtest.execution import tradability as _td

    if getattr(_td.build_tradability, "_cached", False):
        return
    _orig = _td.build_tradability

    def cached_build(
        stock_data,
        trading_dates,
        codes=None,
        participation_rate=_td.DEFAULT_PARTICIPATION_RATE,
        min_turnover=_td.DEFAULT_MIN_TURNOVER,
        st_codes=None,
        prev_close=None,
    ):
        codelist = list(codes) if codes is not None else list(stock_data.keys())
        key = (
            _content_key(codelist),
            len(trading_dates),
            trading_dates[0] if len(trading_dates) else None,
            trading_dates[-1] if len(trading_dates) else None,
            float(participation_rate),
            float(min_turnover),
            id(st_codes),
            id(prev_close),
        )
        hit = _TRADABILITY_CACHE.get(key)
        if hit is None:
            hit = _orig(
                stock_data,
                trading_dates,
                codes=codes,
                participation_rate=participation_rate,
                min_turnover=min_turnover,
                st_codes=st_codes,
                prev_close=prev_close,
            )
            _TRADABILITY_CACHE.clear()
            _TRADABILITY_CACHE[key] = hit
        return hit

    cached_build._cached = True  # type: ignore[attr-defined]
    _td.build_tradability = cached_build


# ─────────────────────────── 缓存构建 ───────────────────────────
def _panel_factor_matrices(
    codes: Sequence[str],
    master: pd.DatetimeIndex,
    start: datetime,
    end: datetime,
    panel_factors: Sequence[str],
) -> Dict[str, np.ndarray]:
    """每个面板因子一个 (n_dates, n_stocks) 矩阵。

    与官方实现唯一的差异：**每只股票的 parquet 只读一次**（一次拿全部面板列），
    官方是每个因子重读一次（17 倍冗余 I/O）。数值完全一致。
    """
    from app.services.backtest.strategies.honest_composite_strategy import (
        DEFAULT_PANEL_DIR,
    )
    from concurrent.futures import ThreadPoolExecutor

    n_dates, n_stocks = len(master), len(codes)
    mats = {
        f: np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
        for f in panel_factors
    }
    cols = list(panel_factors)

    def one(j: int, code: str) -> None:
        path = os.path.join(DEFAULT_PANEL_DIR, code.replace(".", "_") + ".parquet")
        try:
            pf = pd.read_parquet(path, columns=cols)
        except Exception:
            try:
                pf = pd.read_parquet(path)
            except Exception:
                return
        pf = pf.reset_index()
        if "date" not in pf.columns:
            return
        pf = pf.set_index("date").sort_index()
        pf = pf.loc[pd.Timestamp(start):pd.Timestamp(end)]
        pf = pf[~pf.index.duplicated(keep="first")]
        pos = master.get_indexer(pd.DatetimeIndex(pf.index))
        ok = pos >= 0
        for f in panel_factors:
            if f not in pf.columns:
                continue
            vals = pd.to_numeric(pf[f], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            mats[f][pos[ok], j] = vals[ok]

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda t: one(*t), enumerate(codes)))
    return mats


def _rank_ic_series(
    z_mats: Dict[str, np.ndarray], close: np.ndarray
) -> Dict[str, np.ndarray]:
    """逐因子逐日 RankIC（Spearman）：z 分数 vs 次日收益。

    z 矩阵已含官方方向（sign 已乘进去），所以 IC>0 表示该因子当期方向有效。
    """
    fwd = np.full_like(close, np.nan)
    fwd[:-1] = close[1:] / close[:-1] - 1.0
    fwd_df = pd.DataFrame(fwd).rank(axis=1, na_option="keep")
    out: Dict[str, np.ndarray] = {}
    for name, z in z_mats.items():
        zr = pd.DataFrame(z).rank(axis=1, na_option="keep")
        a = zr.to_numpy(dtype=np.float64)
        b = fwd_df.to_numpy(dtype=np.float64)
        m = np.isfinite(a) & np.isfinite(b)
        n = m.sum(axis=1)
        a0 = np.where(m, a, 0.0)
        b0 = np.where(m, b, 0.0)
        am = a0.sum(axis=1)
        bm = b0.sum(axis=1)
        cov = (a0 * b0).sum(axis=1) - am * bm / np.maximum(n, 1)
        va = (a0 * a0).sum(axis=1) - am * am / np.maximum(n, 1)
        vb = (b0 * b0).sum(axis=1) - bm * bm / np.maximum(n, 1)
        denom = np.sqrt(np.maximum(va, 0.0) * np.maximum(vb, 0.0))
        ic = np.where((n >= 30) & (denom > 1e-12), cov / np.maximum(denom, 1e-12), np.nan)
        out[name] = ic
    return out


def build_cache(window_key: str, force: bool = False) -> str:
    from app.services.backtest.execution.data_loader import DataLoader
    from app.services.backtest.strategies import honest_composite_strategy as hcs
    from app.services.trading.execution_rules import zscore_with_industry

    out = cache_dir_for(window_key)
    os.makedirs(out, exist_ok=True)
    meta_path = os.path.join(out, "meta.json")
    if os.path.exists(meta_path) and not force:
        print(f"[cache] 已存在，跳过: {out}")
        return out

    all_f, honest_f, panel_f = _factor_sets()
    start, end = window_dates(window_key)
    codes_all = universe_from_instruments(QLIB_OFFICIAL)
    print(f"[cache] 窗口 {start.date()} → {end.date()}，PIT 股票池 {len(codes_all)}")

    t0 = time.perf_counter()
    loader = DataLoader(os.path.join(ROOT, "data"), max_workers=8)
    stock_data = loader.load_multiple_stocks(
        codes_all, start, end, required_columns=list(OHLCV)
    )
    stock_data = _ordered_stock_data(stock_data, codes_all)
    codes, master, close, high, low, vol = _build_matrices(stock_data)
    print(
        f"[cache] 载入 {len(codes)} 只有效股票, {len(master)} 个交易日 "
        f"({time.perf_counter()-t0:.0f}s)"
    )

    n_stocks = len(codes)
    min_cross = max(2, min(30, n_stocks // 10))
    # 基线口径不含行业中性（verify_10y_pit.py 的 honest_composite 未开 industry_neutral）
    ind_ids, n_ind, industry_neutral = None, 0, False

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close.pct_change(fill_method=None)
        vchg = vol.pct_change(fill_method=None)
        cands: Dict[str, pd.DataFrame] = {}
        for n in (5, 10, 20, 30, 60):
            cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
            cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
            cands[f"STD{n}"] = close.rolling(n).std() / close
            cands[f"CORR{n}"] = close.rolling(n).corr(vol)
            cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)

    z_mats: Dict[str, np.ndarray] = {}
    for name, sign in hcs._HONEST_FORMULAS.items():
        z_mats[name] = zscore_with_industry(
            cands[name].to_numpy(dtype=np.float64),
            sign,
            min_cross,
            ind_ids,
            n_ind,
            industry_neutral,
        )
        cands.pop(name, None)

    t1 = time.perf_counter()
    pmats = _panel_factor_matrices(
        codes, master, start, end, list(hcs._ALPHAPANEL_FACTORS)
    )
    print(f"[cache] 面板因子读取完成 ({time.perf_counter()-t1:.0f}s)")
    for name, sign in hcs._ALPHAPANEL_FACTORS.items():
        z_mats[name] = zscore_with_industry(
            pmats[name], sign, min_cross, ind_ids, n_ind, industry_neutral
        )
        pmats[name] = None

    t2 = time.perf_counter()
    close_np = close.to_numpy(dtype=np.float64)
    ic = _rank_ic_series(z_mats, close_np)
    print(f"[cache] RankIC 计算完成 ({time.perf_counter()-t2:.0f}s)")

    for name, z in z_mats.items():
        np.save(os.path.join(out, f"z_{name}.npy"), z)
    np.save(os.path.join(out, "close.npy"), close_np)
    np.savez(os.path.join(out, "ic.npz"), **ic)
    meta = {
        "window": window_key,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in master],
        "codes": list(codes),
        "n_dates": int(len(master)),
        "n_stocks": int(n_stocks),
        "min_cross": int(min_cross),
        "industry_neutral": False,
        "factors": all_f,
        "honest_factors": honest_f,
        "panel_factors": panel_f,
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    print(
        f"[cache] 写入 {len(z_mats)} 个 z 矩阵 + IC → {out}"
        f"（总耗时 {time.perf_counter()-t0:.0f}s）"
    )
    return out


# ─────────────────────────── 变体策略 ───────────────────────────
def _register_variant_strategy() -> None:
    """注册 honest_variant：官方策略的子类，只替换打分那一步。"""
    from app.services.backtest.strategies import honest_composite_strategy as hcs
    from app.services.backtest.strategies.strategy_factory import StrategyFactory

    class HonestVariantStrategy(hcs.HonestCompositeStrategy):
        """official honest_composite，composite 改为从 z 矩阵缓存重组。"""

        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.factors: List[str] = list(config.get("factors") or [])
            self.weight_mode: str = str(config.get("weight_mode") or "equal")
            self.weight_window: int = int(config.get("weight_window") or 60)
            self.cache_key: str = str(config.get("cache_key") or "full3y")
            self._meta: Optional[Dict[str, Any]] = None
            self._ic: Optional[Dict[str, np.ndarray]] = None

        # ---------- 缓存 ----------
        def _meta_cache(self) -> Dict[str, Any]:
            if self._meta is None:
                with open(os.path.join(cache_dir_for(self.cache_key), "meta.json")) as fh:
                    self._meta = json.load(fh)
            return self._meta

        def _ic_cache(self) -> Dict[str, np.ndarray]:
            if self._ic is None:
                arr = np.load(
                    os.path.join(cache_dir_for(self.cache_key), "ic.npz"),
                    allow_pickle=True,
                )
                self._ic = {k: arr[k] for k in arr.files}
            return self._ic

        # ---------- 权重 ----------
        def _weight_vectors(self, n_dates: int) -> Dict[str, np.ndarray]:
            """每个因子的逐日权重向量（长度 n_dates）。"""
            if self.weight_mode == "equal":
                return {f: np.ones(n_dates) for f in self.factors}
            meta = self._meta_cache()
            ic = self._ic_cache()
            dates = pd.to_datetime(pd.Index(meta["dates"]))
            if self.weight_mode == "icir":
                # 静态 ICIR 权重：只用 IS 段（窗口结束前 2 年）的 IC 估计
                cut = pd.Timestamp(WINDOW_IS[1])
                is_mask = dates < cut
                if is_mask.sum() < 30:  # 窗口内没有 IS 段 → 退回等权
                    is_mask = np.ones(len(dates), dtype=bool)
                raw = {}
                for f in self.factors:
                    v = ic[f][is_mask]
                    v = v[np.isfinite(v)]
                    if len(v) < 20:
                        raw[f] = 1.0
                        continue
                    sd = float(np.std(v))
                    raw[f] = max(float(np.mean(v)) / sd, 0.0) if sd > 1e-9 else 1.0
                pos = [w for w in raw.values() if w > 0]
                scale = float(np.mean(pos)) if pos else 1.0
                if scale <= 0:
                    return {f: np.ones(n_dates) for f in self.factors}
                return {
                    f: np.full(n_dates, (raw[f] / scale) if raw[f] > 0 else 0.0)
                    for f in self.factors
                }
            if self.weight_mode == "rolling_ic":
                # 滚动 IC 权重（严格因果）：w_f(t) = mean(IC_f[s], s∈[t-W+1, t-1])
                # IC[s] 用 s→s+1 的收益，s+1 时刻才可知，故 s ≤ t-1。
                out = {}
                for f in self.factors:
                    series = ic[f]
                    valid = np.isfinite(series)
                    cs = np.concatenate([[0.0], np.cumsum(np.where(valid, series, 0.0))])
                    cc = np.concatenate([[0.0], np.cumsum(valid.astype(np.float64))])
                    w = np.zeros(n_dates)
                    for t in range(n_dates):
                        lo = max(0, t - self.weight_window + 1)
                        hi = t - 1
                        if hi < lo:
                            continue
                        c = cc[hi] - cc[lo]
                        if c < 20:
                            continue
                        w[t] = max((cs[hi] - cs[lo]) / c, 0.0)
                    out[f] = w
                return out
            raise ValueError(f"unknown weight_mode {self.weight_mode}")

        # ---------- 打分（唯一改动点）----------
        async def prepare_backtest_data(self, stock_data, start_date, end_date) -> None:
            from app.services.trading.execution_rules import (
                limit_up_mask_matrix,
                load_st_flags,
            )

            codes, master, close, high, low, vol = _cached_matrices(stock_data)
            if not codes:
                return
            n_dates, n_stocks = len(master), len(codes)
            cdir = cache_dir_for(self.cache_key)
            meta = self._meta_cache()

            # 缓存与本次回测必须逐行逐列对齐，否则直接报错（不做静默重排）
            if list(meta["codes"]) != list(codes):
                raise RuntimeError(
                    "缓存股票池与回测不一致："
                    f"cache={len(meta['codes'])} run={len(codes)}"
                )
            if list(meta["dates"]) != [pd.Timestamp(d).strftime("%Y-%m-%d") for d in master]:
                raise RuntimeError("缓存交易日与回测不一致")

            wvecs = self._weight_vectors(n_dates)
            acc = np.zeros((n_dates, n_stocks), dtype=np.float64)
            wsum = np.zeros((n_dates, n_stocks), dtype=np.float64)
            for name in self.factors:
                z = np.load(os.path.join(cdir, f"z_{name}.npy"), mmap_mode="r")
                w = wvecs[name][:, None]
                good = np.isfinite(z)
                # 与官方 acc[good] += w*z[good] 逐位等价，但避免布尔花式索引的
                # 临时拷贝（42 个因子 × 727×5238 的布尔索引是主要的 CPU 开销）
                acc += np.where(good, w * z, 0.0)
                wsum += np.where(good, w, 0.0)
                del z, good

            composite = np.where(
                wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan
            )

            # 涨停掩码（与官方一致）
            day_ret = close.pct_change(fill_method=None).to_numpy()
            limit_up_mask = limit_up_mask_matrix(
                day_ret, close.columns, is_st=load_st_flags(list(close.columns))
            )
            composite[limit_up_mask] = np.nan

            # 调仓日掩码
            if self.rebalance_days == "weekly":
                composite[master.dayofweek != self.rebalance_weekday] = np.nan
            elif self.rebalance_days == "monthly":
                composite[master.day != 1] = np.nan

            # lag
            if self.lag > 0:
                lagged = np.full_like(composite, np.nan)
                if self.lag < n_dates:
                    lagged[self.lag:] = composite[: n_dates - self.lag]
                composite = lagged

            for j, code in enumerate(codes):
                column = composite[:, j]
                score_map: Dict[datetime, float] = {}
                for k in range(n_dates):
                    value = column[k]
                    if np.isfinite(value):
                        score_map[master[k].to_pydatetime()] = float(value)
                stock_data[code].attrs[hcs._SCORES_ATTR] = score_map

    StrategyFactory.register_strategy("honest_variant", HonestVariantStrategy)


# ─────────────────────────── 回测执行 ───────────────────────────
def _make_executor():
    from app.services.backtest.execution.backtest_executor import BacktestExecutor
    from app.services.backtest.execution.data_loader import DataLoader

    class _CachedLoader(DataLoader):
        """worker 内缓存已载入行情（避免每次回测重读 parquet）。"""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._cache: Dict[Tuple, Dict[str, pd.DataFrame]] = {}

        def load_multiple_stocks(self, codes, start_date, end_date, **kwargs):
            key = (
                tuple(codes),
                start_date,
                end_date,
                tuple(kwargs.get("required_columns") or ()),
            )
            if key not in self._cache:
                self._cache[key] = super().load_multiple_stocks(
                    codes, start_date, end_date, **kwargs
                )
            return self._cache[key]

    class _FastExecutor(BacktestExecutor):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.data_loader = _CachedLoader(self.data_loader.data_dir, max_workers=8)
            self._prev_close_cache: Dict[Tuple, Dict[str, float]] = {}
            _patch_tradability_cache()

        def _get_required_data_columns(self, strategy_name, strategy_config):
            return list(OHLCV)

        async def _load_prev_close_map(self, stock_codes, start_date):
            key = (len(stock_codes), stock_codes[0], stock_codes[-1], start_date)
            if key not in self._prev_close_cache:
                self._prev_close_cache[key] = await super()._load_prev_close_map(
                    stock_codes, start_date
                )
            return self._prev_close_cache[key]

    return _FastExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )


# 每个 worker 进程一个 executor（行情加载缓存跨变体复用）
_EXECUTOR: Dict[str, Any] = {}


def _get_executor():
    key = f"{os.getpid()}"
    ex = _EXECUTOR.get(key)
    if ex is None:
        ex = _make_executor()
        _EXECUTOR[key] = ex
    return ex


def _bootstrap_sharpe_se(
    hist: List[Dict[str, Any]], n_boot: int = 2000, block: int = 10, seed: int = 7
) -> Dict[str, Any]:
    """块自举（block bootstrap）估计夏普标准误 + 置信区间。

    日频收益有自相关，普通 iid 自举会低估 SE → 用 block=10 日的移动块自举。
    """
    df = pd.DataFrame(hist)
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    vals = df["portfolio_value"].astype(float).to_numpy()
    rets = np.diff(vals) / vals[:-1]
    n = len(rets)
    if n < block * 3:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(starts), size=n_blocks)
        series = np.concatenate([rets[s : s + block] for s in starts[pick]])[:n]
        sd = series.std()
        sharpes[b] = series.mean() / sd * math.sqrt(252) if sd > 0 else 0.0
    point = rets.mean() / rets.std() * math.sqrt(252) if rets.std() > 0 else 0.0
    return {
        "sharpe_point": float(point),
        "sharpe_se": float(sharpes.std()),
        "sharpe_ci95_lo": float(np.percentile(sharpes, 2.5)),
        "sharpe_ci95_hi": float(np.percentile(sharpes, 97.5)),
        "n_boot": int(n_boot),
        "block": int(block),
    }


def _slice_metrics(
    hist: List[Dict[str, Any]], start: str, end: str
) -> Dict[str, Any]:
    """从日频组合净值序列切出子区间，算 Sharpe/回撤/收益。"""
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    df = df[(df["date"] >= lo) & (df["date"] <= hi)]
    if len(df) < 5:
        return {}
    vals = df["portfolio_value"].astype(float).to_numpy()
    rets = pd.Series(vals).pct_change().dropna()
    if rets.empty or rets.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return {
        "sharpe_ratio": sharpe,
        "total_return": float(vals[-1] / vals[0] - 1.0),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "n_days": int(len(df)),
        "start": str(df["date"].iloc[0].date()),
        "end": str(df["date"].iloc[-1].date()),
    }


def run_variant(
    factors: Sequence[str],
    window_key: str = "full3y",
    weight_mode: str = "equal",
    weight_window: int = 60,
    label: str = "",
    subwindow: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    """跑一次真实盘口径回测（lag=1 + 可成交性 + PIT 股票池 + 周频）。

    subwindow 非空时，在 3 年缓存内只回测 [start, end] 这一段
    （用于滚动窗口稳定性检验）；打分仍用 3 年缓存的 z 矩阵，
    因此因子计算口径与整段回测完全一致。
    """
    from app.services.backtest.models import BacktestConfig

    _register_variant_strategy()
    start, end = window_dates(window_key)
    codes = universe_from_instruments(QLIB_OFFICIAL)
    executor = _get_executor()

    cfg = dict(HC_CFG)
    cfg.update(
        {
            "lag": 1,
            "factors": list(factors),
            "weight_mode": weight_mode,
            "weight_window": weight_window,
            "cache_key": window_key,
        }
    )

    t0 = time.perf_counter()
    res = asyncio.run(
        executor.run_backtest(
            strategy_name="honest_variant",
            stock_codes=codes,
            start_date=start,
            end_date=end,
            strategy_config=cfg,
            backtest_config=BacktestConfig(
                enable_tradability=True,
                initial_cash=1_000_000.0,
                commission_rate=0.0003,
                slippage_rate=0.0001,
                max_position_size=HC_MPS,
                cash_reserve_ratio=0.05,
                board_lot_size=100,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                rebalance_frequency="daily",
                open_cost=0.0,
                close_cost=0.0,
                min_cost=0.0,
            ),
        )
    )
    wall = time.perf_counter() - t0

    out: Dict[str, Any] = {
        "label": label,
        "factors": list(factors),
        "n_factors": len(factors),
        "window": window_key,
        "weight_mode": weight_mode,
        "weight_window": weight_window if weight_mode == "rolling_ic" else None,
        "wall_s": round(wall, 1),
    }
    for k in (
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "total_trades",
        "win_rate",
        "volatility",
        "profit_factor",
    ):
        if k in res:
            out[k] = res[k]

    hist = res.get("portfolio_history") or []
    if hist:
        out["is"] = _slice_metrics(hist, WINDOW_IS[0], WINDOW_IS[1])
        out["oos"] = _slice_metrics(hist, WINDOW_OOS[0], WINDOW_OOS[1])
        out["bootstrap"] = _bootstrap_sharpe_se(hist)
        out["bootstrap_is"] = _bootstrap_sharpe_se(
            [h for h in hist if WINDOW_IS[0] <= str(h["date"])[:10] <= WINDOW_IS[1]]
        )
        out["bootstrap_oos"] = _bootstrap_sharpe_se(
            [h for h in hist if WINDOW_OOS[0] <= str(h["date"])[:10] <= WINDOW_OOS[1]]
        )
        df = pd.DataFrame(hist)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date", keep="last")
        df = df.set_index("date")
        yv = df["portfolio_value"].resample("YE").last()
        yr = yv.pct_change().dropna()
        out["yearly"] = {str(int(d.year)): round(float(v), 6) for d, v in yr.items()}
        first_year = df["portfolio_value"].resample("YE").last()
        if len(first_year):
            out["yearly_first_partial"] = round(
                float(first_year.iloc[0] / df["portfolio_value"].iloc[0] - 1.0), 6
            )
        vals = df["portfolio_value"].astype(float)
        rets = vals.pct_change().dropna()
        out["ret_stats"] = {
            "skew": float(rets.skew()),
            "kurtosis": float(rets.kurtosis()),
            "n_days": int(len(rets)),
        }
    md = res.get("monthly_returns_detail")
    if isinstance(md, list) and md:
        out["monthly"] = {str(x["month"]): round(float(x["return"]), 6) for x in md}
    # 保留日频净值序列（供滚动窗口稳定性检验切片；体积可控）
    if hist:
        out["_history"] = [
            {
                "date": str(h["date"])[:10],
                "portfolio_value": float(h["portfolio_value"]),
            }
            for h in hist
        ]
    return out


# ─────────────────────────── 并行 driver ───────────────────────────
def _worker_init(window_key: str) -> None:
    _register_variant_strategy()
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _worker_run(task: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return run_variant(
            factors=task["factors"],
            window_key=task.get("window", "full3y"),
            weight_mode=task.get("weight_mode", "equal"),
            weight_window=task.get("weight_window", 60),
            label=task.get("label", ""),
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        return {
            "label": task.get("label", ""),
            "factors": task.get("factors"),
            "n_factors": len(task.get("factors", [])),
            "window": task.get("window"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }


def run_tasks(
    tasks: List[Dict[str, Any]], workers: int, window_key: str
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if workers <= 1:
        _worker_init(window_key)
        for t in tasks:
            out.append(_worker_run(t))
            print(
                f"  [{len(out)}/{len(tasks)}] {out[-1].get('label')} "
                f"sharpe={out[-1].get('sharpe_ratio')}",
                flush=True,
            )
        return out
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(window_key,),
    ) as ex:
        futs = {ex.submit(_worker_run, t): i for i, t in enumerate(tasks)}
        collected: Dict[int, Dict[str, Any]] = {}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            collected[futs[fut]] = r
            print(
                f"  [{i}/{len(tasks)}] {r.get('label')} "
                f"sharpe={r.get('sharpe_ratio')} ({r.get('wall_s')}s)",
                flush=True,
            )
    # 关键：必须按提交顺序返回，否则调用方的 zip(tasks, outs) 会错位
    out = [collected[i] for i in range(len(tasks))]
    return out


def variant_key(
    factors: Sequence[str], weight_mode: str = "equal", weight_window: int = 60
) -> str:
    h = hashlib.sha1(
        (
            "|".join(sorted(factors))
            + f"#{weight_mode}#{weight_window if weight_mode == 'rolling_ic' else 0}"
        ).encode()
    ).hexdigest()[:12]
    return h


def _load_store(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def _save_store(path: str, store: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


# ─────────────────────────── 阶段实现 ───────────────────────────
def stage_cache(args) -> None:
    build_cache(args.window, force=args.force)


def stage_verify(args) -> None:
    """硬校验：42 因子子集必须复现官方 0.7992，25 因子子集必须复现官方 0.8796。"""
    all_f, honest_f, panel_f = _factor_sets()
    print(
        f"[verify] 因子总数 {len(all_f)} = 诚实公式 {len(honest_f)} + Alpha158 面板 {len(panel_f)}"
    )
    tasks = [
        {"factors": all_f, "label": "42f_all", "window": args.window},
        {"factors": honest_f, "label": "25f_honest_only", "window": args.window},
    ]
    res = run_tasks(tasks, 1, args.window)
    checks = {"42f_all": BASELINE_42F, "25f_honest_only": BASELINE_25F}
    ok_all = True
    for r in res:
        exp = checks.get(r["label"], {})
        print(f"\n[verify] {r['label']}")
        for k in ("sharpe_ratio", "total_return", "max_drawdown", "total_trades"):
            got, want = r.get(k), exp.get(k)
            ok = got is not None and abs(float(got) - float(want)) < 1e-6
            ok_all = ok_all and ok
            print(
                f"    {k:>16}: got={got}  official={want}  "
                f"{'OK' if ok else '*** MISMATCH ***'}"
            )
        print(f"    IS  ={r.get('is')}")
        print(f"    OOS ={r.get('oos')}")
        print(f"    yearly={r.get('yearly')}")
    print(f"\n[verify] 全部一致: {ok_all}")
    _save_store(
        os.path.join(OUT_DIR, "factor_subset_verify.json"),
        {"results": res, "match": bool(ok_all), "window": args.window},
    )
    if not ok_all:
        sys.exit(1)


def stage_ablate(args) -> None:
    """单因子消融：42 个因子各去掉一个。"""
    all_f, _, _ = _factor_sets()
    path = os.path.join(OUT_DIR, f"factor_subset_ablation_{args.window}.json")
    store = _load_store(path)
    results: Dict[str, Any] = store.get("results", {})

    tasks = []
    bkey = variant_key(all_f)
    if bkey not in results:
        tasks.append(
            {"factors": all_f, "label": "baseline_42f", "window": args.window, "_key": bkey}
        )
    for f in all_f:
        sub = [x for x in all_f if x != f]
        k = variant_key(sub)
        if k in results:
            continue
        tasks.append(
            {
                "factors": sub,
                "label": f"drop_{f}",
                "window": args.window,
                "_key": k,
            }
        )
    print(f"[ablate] 待跑 {len(tasks)} 个变体")
    if args.max_variants > 0:
        tasks = tasks[: args.max_variants]
        print(f"[ablate] --max-variants 截断为 {len(tasks)} 个（冒烟测试）")
    outs = run_tasks(
        [{k: v for k, v in t.items() if k != "_key"} for t in tasks],
        args.workers,
        args.window,
    )
    for t, r in zip(tasks, outs):
        results[t["_key"]] = r
    store = {
        "stage": "ablation",
        "window": args.window,
        "baseline_factors": all_f,
        "results": results,
        "n_variants": len(results),
    }
    _save_store(path, store)
    _print_ablation(results, all_f)


def _print_ablation(results: Dict[str, Any], all_f: List[str]) -> None:
    base = results.get(variant_key(all_f))
    if not base or "sharpe_ratio" not in base:
        return
    b_sh = base["sharpe_ratio"]
    b_is = (base.get("is") or {}).get("sharpe_ratio")
    rows = []
    for f in all_f:
        sub = [x for x in all_f if x != f]
        r = results.get(variant_key(sub))
        if not r or "sharpe_ratio" not in r:
            continue
        is_sh = (r.get("is") or {}).get("sharpe_ratio")
        rows.append(
            {
                "factor": f,
                "group": FACTOR_META.get(f, {}).get("group"),
                "family": FACTOR_META.get(f, {}).get("family"),
                "sharpe": r["sharpe_ratio"],
                "d_sharpe": r["sharpe_ratio"] - b_sh,
                "is_sharpe": is_sh,
                "d_is_sharpe": (is_sh - b_is)
                if (is_sh is not None and b_is is not None)
                else None,
                "ret": r.get("total_return"),
                "mdd": r.get("max_drawdown"),
                "trades": r.get("total_trades"),
                "oos_sharpe": (r.get("oos") or {}).get("sharpe_ratio"),
            }
        )
    rows.sort(key=lambda x: -x["d_sharpe"])
    print(f"\n[ablate] 基线 42f: 全窗口 sharpe={b_sh:.4f}  IS={b_is}")
    print("[ablate] 去掉后夏普提升 Top10（全窗口）:")
    for r in rows[:10]:
        dis = "n/a" if r["d_is_sharpe"] is None else f"{r['d_is_sharpe']:+.4f}"
        print(
            f"    {r['factor']:<8} Δ={r['d_sharpe']:+.4f} sharpe={r['sharpe']:.4f} "
            f"ΔIS={dis} ret={r['ret']:.4f} mdd={r['mdd']:.4f} trades={r['trades']}"
        )
    print("[ablate] 去掉后夏普下降 Top10:")
    for r in rows[-10:][::-1]:
        print(f"    {r['factor']:<8} Δ={r['d_sharpe']:+.4f} sharpe={r['sharpe']:.4f}")
    by_group: Dict[str, List[float]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r["d_sharpe"])
    print("[ablate] 按组平均 Δsharpe:")
    for g, v in by_group.items():
        print(f"    {g:<16} mean={np.mean(v):+.4f}  n={len(v)}")


def stage_greedy(args) -> None:
    """贪心后向剔除：每轮去掉"去掉后 IS 夏普最好"的因子。

    目标函数 = IS 段夏普（2023-05 → 2025-05），OOS 段从不参与选择。
    """
    all_f, _, _ = _factor_sets()
    path = os.path.join(OUT_DIR, f"factor_subset_greedy_{args.window}.json")
    store = _load_store(path)
    results: Dict[str, Any] = store.get("results", {})
    trace: List[Dict[str, Any]] = store.get("trace", [])

    # 第 1 轮（从全集去掉一个）与消融实验完全重合 → 直接复用消融结果
    ab_path = os.path.join(OUT_DIR, f"factor_subset_ablation_{args.window}.json")
    ab = _load_store(ab_path)
    seeded = 0
    for k, v in (ab.get("results") or {}).items():
        if k not in results:
            results[k] = v
            seeded += 1
    if seeded:
        print(f"[greedy] 复用消融实验结果 {seeded} 个")

    current: List[str] = list(all_f)

    def is_sharpe(r: Dict[str, Any]) -> float:
        v = (r.get("is") or {}).get("sharpe_ratio")
        if v is None:
            v = r.get("sharpe_ratio", -99.0)
        return float(v)

    while len(current) > args.min_factors:
        cands = list(current)
        tasks = []
        for f in cands:
            sub = [x for x in current if x != f]
            k = variant_key(sub)
            if k in results:
                continue
            tasks.append((f, k, sub))
        if tasks:
            print(
                f"[greedy] 第 {len(trace)+1} 轮：{len(tasks)} 候选待跑"
                f"（当前 {len(current)} 因子）",
                flush=True,
            )
            outs = run_tasks(
                [
                    {"factors": sub, "label": f"try_drop_{f}", "window": args.window}
                    for f, k, sub in tasks
                ],
                args.workers,
                args.window,
            )
            for (f, k, sub), r in zip(tasks, outs):
                results[k] = r
            _save_store(
                path,
                {
                    "stage": "greedy",
                    "window": args.window,
                    "results": results,
                    "trace": trace,
                },
            )

        scored = []
        for f in cands:
            sub = [x for x in current if x != f]
            r = results.get(variant_key(sub))
            if not r or "sharpe_ratio" not in r:
                continue
            scored.append((f, is_sharpe(r), r))
        if not scored:
            break
        scored.sort(key=lambda x: -x[1])
        best_f, best_sc, best_r = scored[0]
        cur_sc = is_sharpe(results[variant_key(current)])
        if best_sc <= cur_sc + args.min_gain:
            print(
                f"[greedy] 无改进（{best_sc:.4f} <= {cur_sc:.4f}+{args.min_gain}），停止"
            )
            break
        current = [x for x in current if x != best_f]
        trace.append(
            {
                "step": len(trace) + 1,
                "dropped": best_f,
                "is_sharpe": best_sc,
                "prev_is_sharpe": cur_sc,
                "n_factors": len(current),
                "full_sharpe": best_r.get("sharpe_ratio"),
                "full_return": best_r.get("total_return"),
                "full_mdd": best_r.get("max_drawdown"),
                "oos_sharpe": (best_r.get("oos") or {}).get("sharpe_ratio"),
                "factors": list(current),
            }
        )
        print(
            f"[greedy] 去掉 {best_f:<8} → {len(current)} 因子，"
            f"IS夏普 {cur_sc:.4f} → {best_sc:.4f}"
            f"（全窗口 {best_r.get('sharpe_ratio'):.4f}, "
            f"OOS {(best_r.get('oos') or {}).get('sharpe_ratio')})",
            flush=True,
        )
        _save_store(
            path,
            {"stage": "greedy", "window": args.window, "results": results, "trace": trace},
        )

    _save_store(
        path,
        {"stage": "greedy", "window": args.window, "results": results, "trace": trace},
    )
    print(f"\n[greedy] 最终子集 {len(current)} 因子: {current}")
    final = results.get(variant_key(current))
    if final:
        print(f"[greedy] 全窗口 sharpe={final.get('sharpe_ratio')}")
        print(f"[greedy] IS ={final.get('is')}")
        print(f"[greedy] OOS={final.get('oos')}")
        print(f"[greedy] yearly={final.get('yearly')}")


def stage_weights(args) -> None:
    """权重方案对比：等权 / 静态 ICIR / 滚动 IC(60,120)。"""
    path = os.path.join(OUT_DIR, f"factor_subset_weights_{args.window}.json")
    store = _load_store(path)
    results: Dict[str, Any] = store.get("results", {})

    gpath = os.path.join(OUT_DIR, f"factor_subset_greedy_{args.window}.json")
    gstore = _load_store(gpath)
    gtrace = gstore.get("trace", [])

    all_f, honest_f, _ = _factor_sets()
    subsets: Dict[str, List[str]] = {
        "42f_all": list(all_f),
        "25f_honest_only": list(honest_f),
    }
    if gtrace:
        subsets["greedy_best"] = list(gtrace[-1]["factors"])
    # 多起点贪心的各条路径终点也要试权重
    g2 = _load_store(os.path.join(OUT_DIR, f"factor_subset_greedy2_{args.window}.json"))
    for sname, trace in (g2.get("traces") or {}).items():
        if trace:
            subsets[f"g2_{sname}"] = list(trace[-1]["factors"])
    # 族级消融里 IS 夏普最好的几个子集也一起试权重方案
    fam_path = os.path.join(OUT_DIR, f"factor_subset_family_{args.window}.json")
    fam = _load_store(fam_path)
    fam_subsets = fam.get("subsets", {})
    fam_res = fam.get("results", {})
    ranked = sorted(
        fam_subsets.items(),
        key=lambda x: -(
            (fam_res.get(variant_key(x[1])) or {}).get("is") or {}
        ).get("sharpe_ratio", -9),
    )
    for sname, sub in ranked[:4]:
        subsets[f"fam_{sname}"] = list(sub)

    tasks = []
    for sname, sub in subsets.items():
        for mode, ww in (
            ("equal", 60),
            ("icir", 60),
            ("rolling_ic", 60),
            ("rolling_ic", 120),
        ):
            k = variant_key(sub, mode, ww)
            if k in results:
                continue
            tasks.append(
                {
                    "factors": sub,
                    "label": f"{sname}|{mode}|{ww}",
                    "window": args.window,
                    "weight_mode": mode,
                    "weight_window": ww,
                    "_key": k,
                }
            )
    print(f"[weights] 待跑 {len(tasks)} 个变体")
    outs = run_tasks(
        [{k: v for k, v in t.items() if k != "_key"} for t in tasks],
        args.workers,
        args.window,
    )
    for t, r in zip(tasks, outs):
        results[t["_key"]] = r
    _save_store(
        path,
        {
            "stage": "weights",
            "window": args.window,
            "results": results,
            "n_variants": len(results),
        },
    )
    for t in tasks:
        r = results.get(t["_key"], {})
        if not r or "sharpe_ratio" not in r:
            print(f"  {t['label']:<36} FAILED {r.get('error','')}")
            continue
        is_ = (r.get("is") or {}).get("sharpe_ratio")
        oos_ = (r.get("oos") or {}).get("sharpe_ratio")
        print(
            f"  {t['label']:<36} sharpe={r['sharpe_ratio']:.4f} "
            f"IS={is_ if is_ is None else round(is_,4)} "
            f"OOS={oos_ if oos_ is None else round(oos_,4)} "
            f"ret={r['total_return']:.4f} mdd={r['max_drawdown']:.4f} "
            f"trades={r['total_trades']}"
        )


def _dsr(sr: float, n_trials: int, years: float, skew: float, kurt: float) -> Dict[str, Any]:
    """Deflated Sharpe Ratio（Bailey & López de Prado 2014）。

    SR0 = 期望最大夏普（零假设下 N 次试验），
    DSR = P(SR > SR0)（含偏度/峰度修正）。
    """
    from scipy.stats import norm

    if n_trials < 2 or years <= 0:
        return {}
    euler = 0.5772156649015329
    sr0_var = (1.0 + 0.5 * sr * sr) / years  # Lo (2002) 年化夏普标准误平方
    sigma = math.sqrt(sr0_var)
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr0 = sigma * ((1.0 - euler) * z1 + euler * z2)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 3.0) / 4.0 * sr * sr, 1e-9))
    dsr = float(norm.cdf((sr - sr0) / denom))
    return {
        "n_trials": n_trials,
        "sr_expected_max_under_null": float(sr0),
        "dsr": dsr,
        "prob_sharpe_gt_expected_max": dsr,
    }


def stage_family(args) -> None:
    """族级消融：一次性去掉整个因子族（同族因子高度相关，单因子消融噪声大）。

    族：MIN / MAX / STD / CORR / CORD（诚实公式，5 个窗口）
        VSUMP / VSUMN / VSUMD / QTLD / IMAX / IMIN / WVMA / VMA / VSTD（Alpha158）
    另外跑几个"手挑子集"（基于单因子消融 + IC 排序的先验）。
    """
    all_f, honest_f, panel_f = _factor_sets()
    path = os.path.join(OUT_DIR, f"factor_subset_family_{args.window}.json")
    store = _load_store(path)
    results: Dict[str, Any] = store.get("results", {})

    fams: Dict[str, List[str]] = {}
    for f in all_f:
        fams.setdefault(FACTOR_META[f]["family"], []).append(f)

    subsets: Dict[str, List[str]] = {}
    for fam, members in fams.items():
        subsets[f"minus_{fam}"] = [f for f in all_f if f not in members]
    subsets["minus_honest_all"] = list(panel_f)
    subsets["minus_panel_all"] = list(honest_f)
    # 手挑：MAX 族 IC≈0/负 + IMIN5 IC 为负 → 一起去掉
    bad = [f for f in all_f if FACTOR_META[f]["family"] == "MAX"] + ["IMIN5"]
    subsets["minus_MAX_IMIN5"] = [f for f in all_f if f not in bad]
    # 只保留 ICIR 最高的因子族
    keep_fams = {"CORR", "CORD", "MIN", "VSUMD", "VSUMP", "VMA", "VSUMN", "QTLD"}
    subsets["only_high_ic_fams"] = [
        f for f in all_f if FACTOR_META[f]["family"] in keep_fams
    ]
    subsets["only_honest_high_ic"] = [
        f
        for f in all_f
        if FACTOR_META[f]["group"] == "honest_formula"
        and FACTOR_META[f]["family"] in {"CORR", "CORD", "MIN", "STD"}
    ]

    tasks = []
    for sname, sub in subsets.items():
        k = variant_key(sub)
        if k in results:
            continue
        tasks.append(
            {
                "factors": sub,
                "label": f"{sname}(n={len(sub)})",
                "window": args.window,
                "_key": k,
            }
        )
    print(f"[family] 待跑 {len(tasks)} 个变体")
    if args.max_variants > 0:
        tasks = tasks[: args.max_variants]
    outs = run_tasks(
        [{k: v for k, v in t.items() if k != "_key"} for t in tasks],
        args.workers,
        args.window,
    )
    for t, r in zip(tasks, outs):
        results[t["_key"]] = r
    _save_store(
        path,
        {
            "stage": "family",
            "window": args.window,
            "subsets": {k: v for k, v in subsets.items()},
            "results": results,
            "n_variants": len(results),
        },
    )
    base = results.get(variant_key(all_f))
    print(f"\n[family] {'子集':<32}{'因子数':>5}{'全窗口':>9}{'IS':>9}{'OOS':>9}{'收益':>9}{'回撤':>9}")
    if base and "sharpe_ratio" in base:
        print(
            f"[family] {'[基线 42f]':<32}{42:>5}{base['sharpe_ratio']:>9.4f}"
            f"{base['is']['sharpe_ratio']:>9.4f}{base['oos']['sharpe_ratio']:>9.4f}"
            f"{base['total_return']:>9.4f}{base['max_drawdown']:>9.4f}"
        )
    for sname, sub in sorted(
        subsets.items(), key=lambda x: -(results.get(variant_key(x[1])) or {}).get(
            "sharpe_ratio", -9
        )
    ):
        r = results.get(variant_key(sub))
        if not r or "sharpe_ratio" not in r:
            print(f"[family] {sname:<32}{len(sub):>5}  FAILED {r.get('error','')[:50] if r else ''}")
            continue
        print(
            f"[family] {sname:<32}{len(sub):>5}{r['sharpe_ratio']:>9.4f}"
            f"{r['is']['sharpe_ratio']:>9.4f}{r['oos']['sharpe_ratio']:>9.4f}"
            f"{r['total_return']:>9.4f}{r['max_drawdown']:>9.4f}"
        )


def stage_greedy2(args) -> None:
    """多起点贪心后向剔除。

    单起点贪心会陷局部最优（第一次实验：从 42 因子出发停在 39 因子、IS 0.8058，
    而族级消融的 minus_VSUMN 直接就 IS 0.8274）。这里从若干"种子子集"分别
    继续往后剔，取 IS 夏普最高的那条路径。

    选择目标仍然是 **IS 段夏普**，OOS 段不参与任何选择。
    """
    all_f, _, _ = _factor_sets()
    path = os.path.join(OUT_DIR, f"factor_subset_greedy2_{args.window}.json")
    store = _load_store(path)
    results: Dict[str, Any] = store.get("results", {})
    traces: Dict[str, Any] = store.get("traces", {})

    def is_sharpe(r: Dict[str, Any]) -> float:
        v = (r.get("is") or {}).get("sharpe_ratio")
        if v is None:
            v = r.get("sharpe_ratio", -99.0)
        return float(v)

    # ── 种子子集 ──
    seeds: Dict[str, List[str]] = {"all42": list(all_f)}
    fam_path = os.path.join(OUT_DIR, f"factor_subset_family_{args.window}.json")
    fam = _load_store(fam_path)
    for sname, sub in (fam.get("subsets") or {}).items():
        seeds[f"fam_{sname}"] = list(sub)
    g_path = os.path.join(OUT_DIR, f"factor_subset_greedy_{args.window}.json")
    g = _load_store(g_path)
    for t in g.get("trace", []):
        seeds[f"greedy{t['step']}"] = list(t["factors"])

    # 先把已知结果灌进 results（省掉重复回测）
    for src in (fam.get("results") or {}, g.get("results") or {}):
        for k, v in src.items():
            results.setdefault(k, v)

    seed_names = list(seeds)
    if args.seeds:
        seed_names = [s for s in seed_names if s in args.seeds.split(",")]
    print(f"[greedy2] 种子: {seed_names}")

    for sname in seed_names:
        current = list(seeds[sname])
        trace: List[Dict[str, Any]] = []
        while len(current) > args.min_factors:
            cands = list(current)
            tasks = []
            for f in cands:
                sub = [x for x in current if x != f]
                k = variant_key(sub)
                if k in results:
                    continue
                tasks.append((f, k, sub))
            if tasks:
                print(
                    f"[greedy2] seed={sname} 第 {len(trace)+1} 轮："
                    f"{len(tasks)} 候选（当前 {len(current)} 因子）",
                    flush=True,
                )
                outs = run_tasks(
                    [
                        {"factors": sub, "label": f"{sname}|try_drop_{f}", "window": args.window}
                        for f, k, sub in tasks
                    ],
                    args.workers,
                    args.window,
                )
                for (f, k, sub), r in zip(tasks, outs):
                    results[k] = r
                _save_store(
                    path,
                    {"stage": "greedy2", "window": args.window, "results": results, "traces": traces},
                )
            scored = []
            for f in cands:
                sub = [x for x in current if x != f]
                r = results.get(variant_key(sub))
                if not r or "sharpe_ratio" not in r:
                    continue
                scored.append((f, is_sharpe(r), r))
            if not scored:
                break
            scored.sort(key=lambda x: -x[1])
            best_f, best_sc, best_r = scored[0]
            cur_sc = is_sharpe(results[variant_key(current)])
            if best_sc <= cur_sc + args.min_gain:
                print(f"[greedy2] seed={sname} 无改进，停止")
                break
            current = [x for x in current if x != best_f]
            trace.append(
                {
                    "step": len(trace) + 1,
                    "dropped": best_f,
                    "is_sharpe": best_sc,
                    "prev_is_sharpe": cur_sc,
                    "n_factors": len(current),
                    "full_sharpe": best_r.get("sharpe_ratio"),
                    "full_return": best_r.get("total_return"),
                    "full_mdd": best_r.get("max_drawdown"),
                    "oos_sharpe": (best_r.get("oos") or {}).get("sharpe_ratio"),
                    "factors": list(current),
                }
            )
            print(
                f"[greedy2] seed={sname} 去掉 {best_f:<9} → {len(current)} 因子, "
                f"IS {cur_sc:.4f} → {best_sc:.4f} "
                f"(全窗口 {best_r.get('sharpe_ratio'):.4f}, "
                f"OOS {(best_r.get('oos') or {}).get('sharpe_ratio')})",
                flush=True,
            )
        traces[sname] = trace
        _save_store(
            path, {"stage": "greedy2", "window": args.window, "results": results, "traces": traces}
        )

    # 汇总：每条路径的终点 + 全局最优
    print("\n[greedy2] 各路径终点（按 IS 夏普排序）:")
    finals = []
    for sname, trace in traces.items():
        if not trace:
            continue
        t = trace[-1]
        finals.append((t["is_sharpe"], sname, t))
    finals.sort(key=lambda x: -x[0])
    for sc, sname, t in finals:
        print(
            f"  {sname:<34} n={t['n_factors']:>3} IS={sc:.4f} "
            f"全窗口={t.get('full_sharpe'):.4f} OOS={t.get('oos_sharpe'):.4f} "
            f"回撤={t.get('full_mdd'):.4f}"
        )
    if finals:
        best = finals[0]
        out = {
            "seed": best[1],
            "is_sharpe": best[0],
            "factors": best[2]["factors"],
            "n_factors": best[2]["n_factors"],
            "full_sharpe": best[2].get("full_sharpe"),
            "oos_sharpe": best[2].get("oos_sharpe"),
        }
        fr = results.get(variant_key(best[2]["factors"]))
        if fr:
            out["detail"] = fr
        _save_store(
            os.path.join(OUT_DIR, f"factor_subset_best_{args.window}.json"),
            out,
        )
        print(f"\n[greedy2] 最优路径 seed={best[1]} → {out['factors']}")


def stage_stability(args) -> None:
    """滚动窗口稳定性检验。

    动机：族级消融发现 `only_high_ic_fams` 的 OOS 夏普 1.82 但 IS 只有 0.11，
    说明 IS/OOS 两段是**不同市场状态**（regime），单一 IS 段选出来的子集
    未必稳。本阶段把 3 年切成多个不重叠子窗口，每个子窗口单独回测，
    看候选子集在每个窗口的表现是否一致（而不是只在某一段好）。
    """
    all_f, honest_f, _ = _factor_sets()
    path = os.path.join(OUT_DIR, f"factor_subset_stability_{args.window}.json")
    store = _load_store(path)
    results: Dict[str, Any] = store.get("results", {})

    # 候选子集：基线 + 各阶段最优
    cands: Dict[str, List[str]] = {
        "42f_all": list(all_f),
        "25f_honest_only": list(honest_f),
    }
    for src_file, key in (
        (f"factor_subset_family_{args.window}.json", "subsets"),
        (f"factor_subset_best_{args.window}.json", None),
    ):
        p = os.path.join(OUT_DIR, src_file)
        if not os.path.exists(p):
            continue
        st = _load_store(p)
        if key:
            for sname, sub in (st.get(key) or {}).items():
                cands[f"{sname}"] = list(sub)
        elif st.get("factors"):
            cands["greedy2_best"] = list(st["factors"])
    g2 = _load_store(os.path.join(OUT_DIR, f"factor_subset_greedy2_{args.window}.json"))
    for sname, trace in (g2.get("traces") or {}).items():
        if trace:
            cands[f"g2_{sname}"] = list(trace[-1]["factors"])

    # 子窗口定义（3 年窗口内的不重叠切片）
    subs = [
        ("w1_2023H2", "2023-05-08", "2023-11-08"),
        ("w2_2023Q4", "2023-11-08", "2024-05-08"),
        ("w3_2024H1", "2024-05-08", "2024-11-08"),
        ("w4_2024Q4", "2024-11-08", "2025-05-08"),
        ("w5_2025H1", "2025-05-08", "2025-11-08"),
        ("w6_2025Q4", "2025-11-08", "2026-05-08"),
    ]

    tasks = []
    for cname, sub in cands.items():
        k = variant_key(sub)
        if k in results:
            continue
        tasks.append(
            {
                "factors": sub,
                "label": cname,
                "window": args.window,
                "_key": k,
            }
        )
    print(
        f"[stability] 待跑 {len(tasks)} 个子集"
        f"（每个跑 1 次整段回测，再切成 {len(subs)} 个子窗口看分窗表现）"
    )
    outs = run_tasks(
        [{k: v for k, v in t.items() if k != "_key"} for t in tasks],
        args.workers,
        args.window,
    )
    for t, r in zip(tasks, outs):
        results[t["_key"]] = r
    _save_store(
        path,
        {
            "stage": "stability",
            "window": args.window,
            "candidates": {k: v for k, v in cands.items()},
            "subwindows": subs,
            "results": results,
            "n_variants": len(results),
        },
    )
    # 打印
    print(
        f"\n[stability] {'子集':<28}"
        + "".join(f"{w[0]:>11}" for w in subs)
        + f"{'均值':>9}{'最差':>9}{'整段':>9}"
    )
    table = []
    for cname, sub in cands.items():
        r = results.get(variant_key(sub))
        if not r or "sharpe_ratio" not in r:
            print(f"[stability] {cname:<28}  FAILED {r.get('error', '')[:60] if r else ''}")
            continue
        hist = r.get("_history")
        cells, vals = [], []
        for wname, s, e in subs:
            m = _slice_metrics(hist, s, e) if hist else {}
            v = m.get("sharpe_ratio")
            vals.append(v)
            cells.append(f"{v:>11.3f}" if v is not None else f"{'n/a':>11}")
        ok = [v for v in vals if v is not None]
        avg = float(np.mean(ok)) if ok else float("nan")
        worst = float(np.min(ok)) if ok else float("nan")
        print(
            f"[stability] {cname:<28}"
            + "".join(cells)
            + f"{avg:>9.3f}{worst:>9.3f}{r['sharpe_ratio']:>9.3f}"
        )
        table.append(
            {
                "subset": cname,
                "n_factors": len(sub),
                "per_window_sharpe": {w[0]: v for w, v in zip(subs, vals)},
                "mean": avg,
                "worst": worst,
                "full_sharpe": r["sharpe_ratio"],
                "full_return": r.get("total_return"),
                "full_mdd": r.get("max_drawdown"),
            }
        )
    _save_store(
        os.path.join(OUT_DIR, f"factor_subset_stability_table_{args.window}.json"),
        {"table": table},
    )


def stage_report(args) -> None:
    """汇总：因子清单 + 各阶段结果 + 尝试次数 + DSR。"""
    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "windows": {},
    }
    total_trials = 0
    for win in ("full3y", "is2y", "oos1y"):
        blocks = {}
        for stage in ("ablation", "greedy", "weights"):
            p = os.path.join(OUT_DIR, f"factor_subset_{stage}_{win}.json")
            if os.path.exists(p):
                st = _load_store(p)
                blocks[stage] = st
                total_trials += len(st.get("results", {}))
        summary["windows"][win] = blocks
    all_f, honest_f, panel_f = _factor_sets()
    summary["factors"] = [FACTOR_META[f] for f in all_f]
    summary["n_factors"] = len(all_f)
    summary["n_honest"] = len(honest_f)
    summary["n_panel"] = len(panel_f)
    summary["total_variants_tried"] = total_trials
    summary["baselines_official"] = {"42f": BASELINE_42F, "25f": BASELINE_25F}
    out = os.path.join(OUT_DIR, "factor_subset_summary.json")
    _save_store(out, summary)
    print(f"[report] 写入 {out}（累计 {total_trials} 个变体结果）")


def _fmt(r: Dict[str, Any]) -> str:
    if not r or "sharpe_ratio" not in r:
        return "FAILED"
    is_ = (r.get("is") or {}).get("sharpe_ratio")
    oos_ = (r.get("oos") or {}).get("sharpe_ratio")
    s = f"sharpe={r['sharpe_ratio']:.4f}"
    if is_ is not None:
        s += f" IS={is_:.4f}"
    if oos_ is not None:
        s += f" OOS={oos_:.4f}"
    s += (
        f" ret={r.get('total_return', 0):.4f} mdd={r.get('max_drawdown', 0):.4f}"
        f" win={r.get('win_rate', 0):.3f} trades={r.get('total_trades', 0)}"
    )
    return s


def stage_summarize(args) -> None:
    """把三个阶段的结果汇总成一份可直接阅读的报告（含 DSR 与尝试次数）。"""
    all_f, honest_f, panel_f = _factor_sets()
    out: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "n_factors_total": len(all_f),
        "n_honest": len(honest_f),
        "n_panel": len(panel_f),
        "factor_catalogue": [FACTOR_META[f] for f in all_f],
    }
    trials = 0
    for stage in ("ablation", "greedy", "weights"):
        p = os.path.join(OUT_DIR, f"factor_subset_{stage}_{args.window}.json")
        st = _load_store(p) if os.path.exists(p) else {}
        out[stage] = st
        trials += len(st.get("results", {}))
    out["total_variants_tried"] = trials

    lines: List[str] = []
    A = lines.append
    A("=" * 78)
    A("因子子集搜索 · 汇总报告")
    A(f"窗口 {args.window}　　累计尝试变体数 = {trials}")
    A("=" * 78)

    # ── 1. 因子清单 ──
    A("")
    A("【1】42 因子清单与分类")
    A(f"{'#':>3} {'因子':<10}{'类别':<15}{'族':<7}{'窗口':>4}{'符号':>5}  含义")
    for i, f in enumerate(all_f, 1):
        m = FACTOR_META[f]
        A(
            f"{i:>3} {f:<10}{m['group']:<15}{m['family']:<7}{m['window']:>4}"
            f"{m['sign']:>+5.0f}  {m['desc']}"
        )

    # ── 2. 消融排名 ──
    A("")
    A("【2】单因子消融（基线 = 42 因子全窗口夏普 0.7992）")
    ab = out.get("ablation", {})
    ab_res = ab.get("results", {})
    base = ab_res.get(variant_key(all_f))
    rows = []
    for f in all_f:
        r = ab_res.get(variant_key([x for x in all_f if x != f]))
        if not r or "sharpe_ratio" not in r:
            continue
        rows.append((f, r))
    if base and "sharpe_ratio" in base:
        A(
            f"基线: 夏普={base['sharpe_ratio']:.4f} 收益={base.get('total_return', 0):.4f} "
            f"回撤={base.get('max_drawdown', 0):.4f} 笔数={base.get('total_trades')}"
        )
    rows.sort(key=lambda x: -x[1]["sharpe_ratio"])
    A("")
    A("  去掉后夏普提升 Top10（Δ>0 = 该因子是净负面，去掉更好）:")
    for f, r in rows[:10]:
        d = r["sharpe_ratio"] - base["sharpe_ratio"] if base else float("nan")
        A(f"    {f:<10} Δ={d:+.4f}  {_fmt(r)}")
    A("")
    A("  去掉后夏普下降 Top10（Δ<0 = 该因子有正贡献）:")
    for f, r in rows[-10:][::-1]:
        d = r["sharpe_ratio"] - base["sharpe_ratio"] if base else float("nan")
        A(f"    {f:<10} Δ={d:+.4f}  {_fmt(r)}")

    # ── 3. 贪心路径 ──
    A("")
    A("【3】贪心后向剔除路径（选择目标 = IS 段夏普，OOS 段从未参与选择）")
    g = out.get("greedy", {})
    trace = g.get("trace", [])
    if trace:
        A(f"{'步':>3} {'去掉':<10}{'因子数':>6}{'IS夏普':>9}{'全窗口夏普':>11}{'OOS夏普':>9}")
        for t in trace:
            A(
                f"{t['step']:>3} {t['dropped']:<10}{t['n_factors']:>6}"
                f"{t['is_sharpe']:>9.4f}{t.get('full_sharpe', float('nan')):>11.4f}"
                f"{(t.get('oos_sharpe') if t.get('oos_sharpe') is not None else float('nan')):>9.4f}"
            )
        best = trace[-1]
        A("")
        A(f"  最终子集 ({len(best['factors'])} 因子): {best['factors']}")
        fr = g.get("results", {}).get(variant_key(best["factors"]))
        if fr:
            A(f"  {_fmt(fr)}")
            A(f"  分年收益: {fr.get('yearly')}  （首年残段 {fr.get('yearly_first_partial')}）")
    else:
        A("  （无贪心结果）")

    # ── 4. 权重方案 ──
    A("")
    A("【4】因子权重方案对比")
    w = out.get("weights", {})
    wres = w.get("results", {})
    if wres:
        A(f"  {'方案':<34}{'全窗口':>9}{'IS':>9}{'OOS':>9}{'收益':>9}{'回撤':>9}{'胜率':>8}")
        for k, r in sorted(wres.items(), key=lambda x: -x[1].get("sharpe_ratio", -9)):
            if "sharpe_ratio" not in r:
                A(f"  {str(r.get('label'))[:33]:<34}  FAILED {r.get('error', '')[:40]}")
                continue
            is_ = (r.get("is") or {}).get("sharpe_ratio")
            oos_ = (r.get("oos") or {}).get("sharpe_ratio")
            A(
                f"  {str(r.get('label'))[:33]:<34}{r['sharpe_ratio']:>9.4f}"
                f"{(is_ if is_ is not None else float('nan')):>9.4f}"
                f"{(oos_ if oos_ is not None else float('nan')):>9.4f}"
                f"{r.get('total_return', 0):>9.4f}{r.get('max_drawdown', 0):>9.4f}"
                f"{r.get('win_rate', 0):>8.3f}"
            )
    else:
        A("  （无权重结果）")

    # ── 5. DSR ──
    A("")
    A("【5】过拟合校正（Deflated Sharpe Ratio, Bailey & López de Prado 2014）")
    A(f"  本次搜索共尝试 {trials} 个变体。")
    cands = []
    for label, r in [("42f 基线", base)] + [
        (f"drop {f}", r) for f, r in rows[:10]
    ]:
        if r and "sharpe_ratio" in r:
            cands.append((label, r))
    if trace:
        fr = g.get("results", {}).get(variant_key(trace[-1]["factors"]))
        if fr:
            cands.append(("贪心最终子集", fr))
    for label, r in cands:
        rs = r.get("ret_stats") or {}
        d = _dsr(
            r["sharpe_ratio"],
            max(trials, 2),
            3.0,
            float(rs.get("skew", 0.0)),
            float(rs.get("kurtosis", 0.0)),
        )
        if d:
            A(
                f"  {label:<20} 夏普={r['sharpe_ratio']:.4f}  "
                f"零假设期望最大夏普={d['sr_expected_max_under_null']:.4f}  "
                f"DSR={d['dsr']:.4f}"
            )

    out["report_text"] = "\n".join(lines)
    p = os.path.join(OUT_DIR, f"factor_subset_report_{args.window}.json")
    _save_store(p, out)
    txt = os.path.join(OUT_DIR, f"factor_subset_report_{args.window}.txt")
    with open(txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[report] 写入 {p} / {txt}")


# ─────────────────────────── CLI ───────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        required=True,
        choices=[
            "cache",
            "verify",
            "ablate",
            "family",
            "greedy",
            "greedy2",
            "weights",
            "stability",
            "report",
            "summarize",
        ],
    )
    ap.add_argument("--window", default="full3y", choices=list(WINDOW_KEYS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-factors", type=int, default=10)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--seeds", default="", help="greedy2: 只跑指定种子（逗号分隔）")
    ap.add_argument("--max-variants", type=int, default=0, help=">0 则只跑前 N 个（冒烟测试）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    {
        "cache": stage_cache,
        "verify": stage_verify,
        "ablate": stage_ablate,
        "family": stage_family,
        "greedy": stage_greedy,
        "greedy2": stage_greedy2,
        "weights": stage_weights,
        "stability": stage_stability,
        "report": stage_report,
        "summarize": stage_summarize,
    }[args.stage](args)


if __name__ == "__main__":
    main()
