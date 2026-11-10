#!/usr/bin/env python3
"""组合合成研究实验室（真实盘口径）。

## 目的
在 **不修改回测引擎** 的前提下，批量测试「多策略分数合成 / 因子层改进 / 组合构建」
三类改动，产出可复现的对比数据。

## 口径（与 verify_10y_pit.py --lag 1 完全一致）
- 窗口: 2023-05-08 → 2026-05-08
- 股票池: data/qlib_official_data/instruments/all.txt（PIT 上市/退市日）
- lag=1（周五收盘出信号，下一交易日成交）
- enable_tradability=True（涨停买不进/跌停卖不出/停牌/流动性上限）
- TopK-Dropout（topk=60, n_drop=5, 等权, max_position_size=(1-0.05)/60）
- 成本: commission 0.03% + slippage 0.01%（单边）, 整手 100 股, 现金预留 5%

## 为什么需要"快速模拟器"
真实引擎一次 3 年全市场回测 ≈ 70-95s（其中 50s 是 parquet 加载）。
本研究要跑几十个变体 → 直接用引擎会花掉数小时且无法迭代。
本模块把「引擎的执行语义」抽成一个纯 numpy 模拟器：
  * 用同一份 DataLoader 加载的数据 → 同一份 (date × stock) 矩阵
  * 同一套 tradability 矩阵（复用 tradability.build_tradability）
  * 同一套 TopK-Dropout 撮合 + 成本 + 整手 + 现金预留规则
  * 同一套绩效指标口径（sharpe = μ/σ·√252, max_drawdown, 分年）

**保真度验收**：`--selftest` 会把模拟器结果与真实引擎
（verify_10y_pit.py --lag 1）在同一配置上对比。夏普/收益/回撤/笔数
必须落在容差内，否则拒绝后续实验。

## 交付
- backend/scripts/verify_composite_ensemble.py  ← 命令行入口
- data/backtest/ensemble_*.json                  ← 每个变体结果
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
QLIB_OFFICIAL = os.path.join(ROOT, "data", "qlib_official_data")
QLIB_LIVE = os.path.join(ROOT, "data", "qlib_data")
OUT_DIR = os.path.join(ROOT, "data", "backtest")

if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

START = datetime(2023, 5, 8)
END = datetime(2026, 5, 8)

# ── 引擎参数（与 verify_10y_pit.py 逐项对齐）────────────────────────
INITIAL_CASH = 1_000_000.0
COMMISSION = 0.0003
SLIPPAGE = 0.0001
CASH_RESERVE = 0.05
BOARD_LOT = 100
PARTICIPATION = 0.005
MIN_TURNOVER = 5_000_000.0
TRADABILITY = True


# ══════════════════════════════════════════════════════════════════
# 1. 数据加载（与引擎 DataLoader 完全同源）
# ══════════════════════════════════════════════════════════════════
def universe_from_instruments(instruments_dir: str = QLIB_OFFICIAL) -> List[str]:
    path = os.path.join(instruments_dir, "instruments", "all.txt")
    codes: List[str] = []
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
    return sorted(codes)


def load_matrices(
    codes: Sequence[str],
    start: datetime = START,
    end: datetime = END,
) -> Dict[str, Any]:
    """加载全市场 → (n_dates × n_stocks) 的 close/high/low/volume 矩阵。

    关键：与引擎一致 —— 用 DataLoader 加载（自带 PIT + 有效性过滤），
    然后取并集日历作为 master。
    """
    from app.services.backtest.execution.data_loader import DataLoader

    dl = DataLoader(ROOT + "/data", max_workers=8)
    stock_data = dl.load_multiple_stocks(list(codes), start, end)
    # 与引擎一致：按请求顺序重建，保证确定性
    stock_data = {c: stock_data[c] for c in codes if c in stock_data}

    kept = [c for c, df in stock_data.items()
            if df is not None and not df.empty and "close" in df.columns]
    print(f"[load] 有效股票 {len(kept)}/{len(codes)}", flush=True)

    master = pd.DatetimeIndex(sorted(set().union(*[
        set(pd.DatetimeIndex(stock_data[c].index)) for c in kept
    ])))
    n_d, n_s = len(master), len(kept)
    print(f"[load] master {n_d} 交易日 {master[0].date()} → {master[-1].date()}",
          flush=True)

    mats = {k: np.full((n_d, n_s), np.nan) for k in ("close", "high", "low", "volume")}
    for j, c in enumerate(kept):
        df = stock_data[c]
        pos = master.get_indexer(pd.DatetimeIndex(df.index))
        ok = pos >= 0
        for k in mats:
            vals = pd.to_numeric(df[k], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan)
            mats[k][pos[ok], j] = vals[ok]

    out = {
        "codes": kept,
        "master": master,
        "close": mats["close"],
        "high": mats["high"],
        "low": mats["low"],
        "volume": mats["volume"],
        "stock_data": stock_data,
    }
    return out


# ══════════════════════════════════════════════════════════════════
# 2. 因子计算
# ══════════════════════════════════════════════════════════════════
def cs_zscore(mat: np.ndarray, min_cross: int = 30) -> np.ndarray:
    """逐日截面 z-score（nan-aware），与 execution_rules.zscore_with_industry 一致。"""
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan)
    z = np.where(cnt >= min_cross, z, np.nan)
    return z


def limit_up_mask(mats: Dict[str, np.ndarray]) -> np.ndarray:
    """逐票涨停掩码（按板块 + ST），与 execution_rules.limit_up_mask_matrix 一致。"""
    from app.services.trading.execution_rules import (
        limit_up_mask_matrix,
        load_st_flags,
    )
    close = mats["close"]
    day_ret = (close / np.roll(close, 1, axis=0) - 1.0)
    day_ret[0] = np.nan
    codes = mats["codes"]
    return limit_up_mask_matrix(day_ret, codes, is_st=load_st_flags(list(codes)))


def honest_formula_factors(mats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """25 个诚实公式（MIN/MAX/STD/CORR/CORD × 5 窗）。"""
    close = mats["close"]
    vol = mats["volume"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close[1:] / close[:-1] - 1.0
        vchg = vol[1:] / vol[:-1] - 1.0
        ret = np.vstack([np.full((1, close.shape[1]), np.nan), ret])
        vchg = np.vstack([np.full((1, close.shape[1]), np.nan), vchg])
        out: Dict[str, np.ndarray] = {}
        for n in (5, 10, 20, 30, 60):
            def roll(fn):
                res = np.full_like(close, np.nan)
                s = pd.DataFrame(close)
                res[:] = fn(s).to_numpy()
                return res
            s = pd.DataFrame(close)
            v = pd.DataFrame(vol)
            out[f"MIN{n}"] = (s.rolling(n).min() / s - 1.0).to_numpy()
            out[f"MAX{n}"] = (s.rolling(n).max() / s - 1.0).to_numpy()
            out[f"STD{n}"] = (s.rolling(n).std() / s).to_numpy()
            out[f"CORR{n}"] = s.rolling(n).corr(v).to_numpy()
            out[f"CORD{n}"] = pd.DataFrame(ret).rolling(n).corr(
                pd.DataFrame(vchg)).to_numpy()
    return out


# 诚实公式的符号（与 honest_composite_strategy._HONEST_FORMULAS 一致）
HONEST_SIGNS: Dict[str, float] = {}
for _n in (5, 10, 20, 30, 60):
    HONEST_SIGNS[f"MIN{_n}"] = 1.0
    HONEST_SIGNS[f"MAX{_n}"] = -1.0
    HONEST_SIGNS[f"STD{_n}"] = -1.0
    HONEST_SIGNS[f"CORR{_n}"] = -1.0
    HONEST_SIGNS[f"CORD{_n}"] = -1.0

# 17 个 Alpha158 面板因子（与 honest_composite_strategy._ALPHAPANEL_FACTORS 一致）
ALPHAPANEL_SIGNS: Dict[str, float] = {
    "VSUMP5": -1.0, "VSUMP10": -1.0, "VSUMP20": -1.0,
    "VSUMN5": 1.0, "VSUMN10": 1.0, "VSUMN20": 1.0,
    "VSUMD5": -1.0, "VSUMD10": -1.0, "VSUMD20": -1.0,
    "QTLD5": 1.0, "QTLD10": 1.0, "QTLD20": 1.0,
    "IMAX5": -1.0, "IMIN5": -1.0, "WVMA5": -1.0,
    "VMA5": 1.0, "VSTD5": -1.0,
}


def load_panel_matrices(
    factors: Sequence[str],
    codes: Sequence[str],
    master: pd.DatetimeIndex,
    panel_dir: str = os.path.join(QLIB_LIVE, "features", "day"),
) -> Dict[str, np.ndarray]:
    """读 Alpha158 面板因子 → {factor: (n_dates × n_stocks)}。

    ## parquet 结构（实测，见 data/backtest/full_history_reconcile.json）
    `data/qlib_data/features/day/` 里 5241/5513 只是**两段拼接**：
      block1 = 交易日格（有 $close，无面板因子）
      block2 = 业务日格（含节假日行，有面板因子，无 $close）
    策略原读法 `sort_index() -> duplicated(keep='first')` 会让同一天只留
    其中一段 → 面板因子只有 ~51% 的交易日有值，且值是「跨节假日窗口」
    算出来的口径。

    本函数按**日期**取面板值：同一天有多个行时，优先取**带面板因子**的那行
    （block2），从而 100% 覆盖交易日历。

    `data/qlib_data/features_full/` 是单段文件（交易日格 + 面板因子齐全），
    同一段代码直接适用。

    性能：每个 parquet 只读一次，一次性抽出所有需要的因子
    （原策略是「每因子 × 每股票」重复读盘，5513 只 × 17 因子 = 9.4 万次读）。
    """
    n_d, n_s = len(master), len(codes)
    out = {f: np.full((n_d, n_s), np.nan) for f in factors}
    pos_of = {d: i for i, d in enumerate(master)}
    wanted = [f for f in factors if f]

    for j, code in enumerate(codes):
        path = os.path.join(panel_dir, code.replace(".", "_") + ".parquet")
        if not os.path.exists(path):
            continue
        try:
            # 只读需要的列（188 列 → 18 列，快 10 倍）
            df = pd.read_parquet(path, columns=wanted)
        except Exception:
            continue
        if not wanted or df.empty:
            continue
        idx = pd.DatetimeIndex(df.index.get_level_values("date"))
        df = df.reset_index()
        if "date" not in df.columns:
            continue
        df = df.loc[df["date"].notna()]
        if df.empty:
            continue
        # 同一日期多行时，优先保留「有面板因子」的行
        has = df[wanted].notna().any(axis=1).to_numpy()
        df = df.assign(_has=has)
        df = df.sort_values(["date", "_has"], ascending=[True, False],
                            kind="mergesort")
        df = df.drop_duplicates("date", keep="first")
        dates = pd.DatetimeIndex(df["date"])
        pos = np.array([pos_of.get(d, -1) for d in dates], dtype=np.int64)
        ok = pos >= 0
        if not ok.any():
            continue
        rows = np.where(ok)[0]
        cols = pos[ok]
        for f in factors:
            vals = pd.to_numeric(df[f].iloc[rows], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan)
            out[f][cols, j] = vals
    return out


def cached_panel_matrices(
    factors: Sequence[str],
    codes: Sequence[str],
    master: pd.DatetimeIndex,
    panel_dir: str,
    tag: str,
    cache_dir: str = "/Users/rone/.hermes/cache/scratch/stock_ens/panel_cache",
) -> Dict[str, np.ndarray]:
    """带磁盘缓存的面板加载（按 (tag, panel_dir, 因子集, 股票池, 日历) 键）。"""
    import hashlib

    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(
        f"{tag}|{panel_dir}|{sorted(factors)}|{len(codes)}|"
        f"{master[0]}|{master[-1]}|{len(master)}|{codes[0]}|{codes[-1]}"
        .encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{tag}_{key}.npz")
    if os.path.exists(path):
        try:
            z = np.load(path)
            if set(factors) <= set(z.files):
                return {f: z[f] for f in factors}
        except Exception:
            pass
    mats = load_panel_matrices(factors, codes, master, panel_dir)
    try:
        np.savez_compressed(path, **mats)
    except Exception:
        pass
    return mats

def load_panel_matrix(
    factor: str,
    codes: Sequence[str],
    master: pd.DatetimeIndex,
    panel_dir: str = os.path.join(QLIB_LIVE, "features", "day"),
) -> np.ndarray:
    return load_panel_matrices([factor], codes, master, panel_dir)[factor]


# features_full = 全历史 + 交易日格单段文件（面板因子与价格同日对齐）
PANEL_FULL = os.path.join(QLIB_LIVE, "features_full")


def build_42f_score(
    mats: Dict[str, Any],
    panel_aligned: bool = True,
    panel_dir: str = os.path.join(QLIB_LIVE, "features", "day"),
    panel_subset: Optional[Sequence[str]] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """42 因子等权复合 → (n_dates × n_stocks) 分数。返回 (分数, 各因子 z)。

    panel_aligned=False → 复现「引擎在 CWD=仓库根」时的行为：
        panel_dir 解析失败 → 17 个面板因子全 NaN → 实际只有 25 个公式因子。
    """
    min_cross = 30
    forms = honest_formula_factors(mats)
    zsum = np.zeros_like(mats["close"])
    wsum = np.zeros_like(mats["close"])
    z_by_factor: Dict[str, np.ndarray] = {}

    def add(name: str, z: np.ndarray) -> None:
        good = np.isfinite(z)
        zsum[good] += np.nan_to_num(z)[good]
        wsum[good] += 1.0
        z_by_factor[name] = z

    for name, sign in HONEST_SIGNS.items():
        add(name, cs_zscore(forms[name], min_cross) * sign)

    if panel_aligned:
        wanted = [f for f in ALPHAPANEL_SIGNS
                  if panel_subset is None or f in panel_subset]
        if wanted:
            tag = os.path.basename(os.path.dirname(panel_dir)) + "_" + \
                os.path.basename(panel_dir)
            if use_cache:
                raw_mats = cached_panel_matrices(
                    wanted, mats["codes"], mats["master"], panel_dir, tag)
            else:
                raw_mats = load_panel_matrices(
                    wanted, mats["codes"], mats["master"], panel_dir)
            for name in wanted:
                add(name, cs_zscore(raw_mats[name], min_cross)
                    * ALPHAPANEL_SIGNS[name])

    score = np.where(wsum > 0, zsum / np.where(wsum > 0, wsum, 1.0), np.nan)
    return score, z_by_factor


# ══════════════════════════════════════════════════════════════════
# 3. 可成交性矩阵（复用引擎实现）
# ══════════════════════════════════════════════════════════════════
def build_tradability_mats(mats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """复用 tradability.build_tradability（同一份 stock_data → 同一套规则）。"""
    from app.services.backtest.execution.tradability import build_tradability

    tm = build_tradability(
        mats["stock_data"],
        list(mats["master"]),
        list(mats["codes"]),
        participation_rate=PARTICIPATION,
        min_turnover=MIN_TURNOVER,
    )
    return {
        "can_buy": tm.can_buy,
        "can_sell": tm.can_sell,
        "max_buy_amount": tm.max_buy_amount,
    }


# ══════════════════════════════════════════════════════════════════
# 4. 快速模拟器（复刻引擎 TopK-Dropout 撮合语义）
# ══════════════════════════════════════════════════════════════════
class Selector:
    """选股 + 权重策略接口。

    引擎原生行为 = TopKEqualWeightSelector。
    其余实现用于「组合构建层」实验（逆波动率加权、分层选股）。
    """

    def select(self, order: np.ndarray, t: int, holdings: np.ndarray,
               n_hold: int) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        """返回 (目标持仓, 目标权重, 买入候选池)。

        目标持仓  : 期望持有的股票（按优先级排序）→ 决定卖谁、目标权重
        目标权重  : {股票: 目标市值占组合比例}；空 dict = 用引擎的
                    max_position_size 上限（引擎原生等权）
        买入候选池: 买入时按序取「不在原持仓集里」的第一只（引擎语义：
                    从完整排名列表取，不是只从 topk 取）
        """
        raise NotImplementedError


class TopKEqualWeightSelector(Selector):
    """引擎原生：全局 TopK，等权（每只受 max_position_size 约束）。"""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg

    def select(self, order: np.ndarray, t: int, holdings: np.ndarray,
               n_hold: int) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        topk = self.cfg.topk
        keep = order[:topk]
        return keep, {}, order          # 空权重 → 用引擎 max_position_size


class TopKInverseVolSelector(Selector):
    """TopK 选股 + 逆波动率加权（相对波动率加权 → 会改变夏普）。

    权重 ∝ 1/σ_i，σ_i 为个股过去 window 日收益标准差。
    目标权重归一到 Σw = 1 - cash_reserve（与等权组合总仓位一致）。
    """

    def __init__(self, cfg: SimConfig, inv_vol: np.ndarray) -> None:
        self.cfg = cfg
        self.inv_vol = inv_vol          # (n_dates × n_stocks)
        self.rebalance_to_targets = True

    def select(self, order: np.ndarray, t: int, holdings: np.ndarray,
               n_hold: int) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        topk = self.cfg.topk
        keep = order[:topk]
        iv = self.inv_vol[t, keep]
        iv = np.where(np.isfinite(iv) & (iv > 0), iv, np.nan)
        if not np.isfinite(iv).any():
            iv = np.ones(len(keep))
        iv = np.where(np.isfinite(iv), iv, np.nanmedian(iv))
        s = float(iv.sum())
        if s <= 0:
            w = {int(c): (1 - self.cfg.cash_reserve) / max(len(keep), 1)
                 for c in keep}
        else:
            w = {int(c): float(iv[i]) / s * (1 - self.cfg.cash_reserve)
                 for i, c in enumerate(keep)}
        return keep, w, order


class ScoreWeightSelector(Selector):
    """TopK 选股 + 分数比例加权（weight ∝ max(score,0)^power）。

    经典做法：等权假设「top-N 里每只都一样好」；分数比例加权让组合
    暴露直接对准信号的强弱。对 z-score 复合分尤其有效（分数单调对应
    预期收益时，等权会浪费信息）。
    """

    def __init__(self, cfg: SimConfig, power: float = 1.0,
                 floor: float = 0.0) -> None:
        self.cfg = cfg
        self.power = power
        self.floor = floor
        self.rebalance_to_targets = True
        self._score = None

    def select(self, order: np.ndarray, t: int, holdings: np.ndarray,
               n_hold: int) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        # 需要原始分数 → 由 simulate 注入
        score = getattr(self, "_score", None)
        if score is None:
            return (order[: self.cfg.topk], {},
                    order)
        topk = self.cfg.topk
        keep = order[:topk]
        raw = score[t, keep]
        w = np.clip(np.nan_to_num(raw, nan=0.0) - self.floor, 0.0, None)
        w = np.power(w, self.power)
        if not np.isfinite(w).any() or w.sum() <= 0:
            w = np.ones(len(keep))
        w = w / w.sum() * (1 - self.cfg.cash_reserve)
        return keep, {int(c): float(w[i]) for i, c in enumerate(keep)}, order


class LayeredSelector(Selector):
    """分层选股：先按某个分组变量分 n_layers 层，每层取 top per_layer 只。

    分组变量由外部提供（每只股票一个 layer id）。
    买入候选池 = 目标层内集合（与引擎「从完整排名取」不同，这是本变体的定义）。
    """

    def __init__(self, cfg: SimConfig, layer_ids: np.ndarray,
                 per_layer: int) -> None:
        self.cfg = cfg
        self.layer_ids = layer_ids
        self.per_layer = per_layer

    def select(self, order: np.ndarray, t: int, holdings: np.ndarray,
               n_hold: int) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        keep: List[int] = []
        counts: Dict[int, int] = {}
        for c in order:
            lid = int(self.layer_ids[c])
            if counts.get(lid, 0) >= self.per_layer:
                continue
            counts[lid] = counts.get(lid, 0) + 1
            keep.append(int(c))
            if len(keep) >= self.cfg.topk:
                break
        arr = np.array(keep, dtype=int)
        return arr, {}, arr


@dataclass
class SimConfig:
    topk: int = 60
    n_drop: int = 5
    hold_thresh: int = 0
    buffer: int = 0            # >0 → qlib topk_buffer 语义（缓冲区内的持仓不卖）
    lag: int = 1
    rebalance_days: str = "weekly"      # weekly | daily | monthly
    rebalance_weekday: int = 4          # 4=周五
    max_position_size: float = 0.015833
    cash_reserve: float = CASH_RESERVE
    commission: float = COMMISSION
    slippage: float = SLIPPAGE
    board_lot: int = BOARD_LOT
    initial_cash: float = INITIAL_CASH
    use_tradability: bool = TRADABILITY


def _rebalance_mask(master: pd.DatetimeIndex, cfg: SimConfig) -> np.ndarray:
    """调仓日掩码：非调仓日分数置 NaN → 执行器无分可依，不换仓。"""
    if cfg.rebalance_days == "weekly":
        return master.dayofweek == cfg.rebalance_weekday
    if cfg.rebalance_days == "monthly":
        return master.day == 1
    return np.ones(len(master), dtype=bool)


def simulate(
    score: np.ndarray,
    mats: Dict[str, Any],
    trad: Dict[str, np.ndarray],
    cfg: SimConfig,
    selector: Optional["Selector"] = None,
) -> Dict[str, object]:
    """复刻引擎执行：TopK-Dropout + 可成交性 + 成本 + 整手 + 现金预留。

    score: (n_dates × n_stocks) 原始分数（未做周频/lag 处理）
    selector: 可选的选股/权重策略（默认 = 引擎原生 TopK 等权）
    """
    master: pd.DatetimeIndex = mats["master"]
    close = mats["close"]
    n_d, n_s = close.shape
    codes = mats["codes"]
    sel = selector or TopKEqualWeightSelector(cfg)

    # ---- 分数预处理：涨停掩码 → 调仓日掩码 → lag ----
    sc = score.copy()
    sc[~np.isfinite(sc)] = np.nan
    lu = limit_up_mask(mats)
    sc[lu] = np.nan

    reb = _rebalance_mask(master, cfg)
    sc[~reb] = np.nan

    if cfg.lag > 0:
        lagged = np.full_like(sc, np.nan)
        lagged[cfg.lag:] = sc[:-cfg.lag]
        sc = lagged

    # 分数比例加权需要**排名用的那份分数**（掩码+lag 之后）
    if hasattr(sel, "_score"):
        sel._score = sc

    # ---- 组合状态 ----
    cash = cfg.initial_cash
    qty = np.zeros(n_s, dtype=np.float64)
    avg_cost = np.zeros(n_s, dtype=np.float64)
    last_px = np.zeros(n_s, dtype=np.float64)
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    n_trades = 0
    turnover = 0.0

    can_buy = trad["can_buy"] if cfg.use_tradability else np.ones((n_d, n_s), bool)
    can_sell = trad["can_sell"] if cfg.use_tradability else np.ones((n_d, n_s), bool)
    max_buy = trad["max_buy_amount"] if cfg.use_tradability else None

    # 引擎主循环有 `if current_idx >= 20` 预热门槛：个股在窗口内累计行数
    # 不足 21 之前不产生信号。窗口起点附近的次新股因此不参与排名。
    n_rows_seen = np.zeros(n_s, dtype=np.int64)

    for t in range(n_d):
        px = close[t]
        valid = np.isfinite(px) & (px > 0)
        if not valid.any():
            continue
        last_px[valid] = px[valid]
        n_rows_seen += valid.astype(np.int64)
        warm = n_rows_seen >= 21

        row = sc[t]
        have = valid & np.isfinite(row) & warm
        if have.any():
            # 排名：分数降序（同分按代码字典序，与引擎一致）
            idx = np.where(have)[0]
            order = idx[np.lexsort((np.array([codes[i] for i in idx]),
                                     -row[idx]))]
            holdings = np.where(qty > 0)[0]
            keep, tgt_w, buy_pool = sel.select(order, t, holdings, len(holdings))
            keep_set = set(int(c) for c in keep)
            holdings_set = set(int(c) for c in holdings)

            def _portfolio_value() -> float:
                return cash + float(np.sum(qty * np.where(
                    np.isfinite(last_px), last_px, 0.0)))

            def _buy(c: int, pv: float) -> None:
                nonlocal cash, n_trades
                if not can_buy[t, c]:
                    return
                px_c = px[c]
                # 目标市值 = 组合市值 × 目标权重（空权重 → 引擎的 max_position_size）
                w_target = float(tgt_w.get(int(c), cfg.max_position_size))
                max_pos_val = pv * w_target
                avail = min(max_pos_val, cash * (1 - cfg.cash_reserve))
                if max_buy is not None:
                    mb = max_buy[t, c]
                    if np.isfinite(mb) and mb > 0:
                        avail = min(avail, mb)
                if avail <= 0:
                    return
                q = int(avail / px_c / cfg.board_lot) * cfg.board_lot
                if q <= 0:
                    return
                cost = q * px_c * (1 + cfg.slippage)
                comm = max(cost * cfg.commission, 0.0)
                if cost + comm > cash:
                    return
                cash -= cost + comm
                old_q = qty[c]
                new_q = old_q + q
                avg_cost[c] = (avg_cost[c] * old_q + cost) / new_q if new_q else 0.0
                qty[c] = new_q
                n_trades += 1

            if len(holdings) < cfg.topk:
                # 初始建仓：按目标持仓买满（引擎：从完整排名取未持有的）
                buy = [int(c) for c in buy_pool if int(c) not in holdings_set
                       ][: cfg.topk - len(holdings)]
                for c in buy:
                    _buy(c, _portfolio_value())
            else:
                # Dropout：卖排名最差的 n_drop（引擎语义：排名 >= topk+hold_thresh）
                # buffer > 0 → qlib topk_buffer 语义：排名 < topk+buffer 的持仓
                #   留在缓冲区不卖，只有掉出缓冲区才卖（换手更慢）。
                rank = {int(c): i for i, c in enumerate(order)}
                held = [c for c in holdings if int(c) in rank]
                held_sorted = sorted(held, key=lambda c: rank[int(c)])
                exit_rank = cfg.topk + cfg.hold_thresh + max(0, cfg.buffer)
                sell = [c for c in reversed(held_sorted)
                        if rank[int(c)] >= exit_rank][: cfg.n_drop]
                # 分层/逆波动率变体：额外强制清掉不在目标持仓里的
                if tgt_w or len(keep_set) < cfg.topk:
                    extra = [c for c in holdings
                             if int(c) not in keep_set and int(c) in rank]
                    extra.sort(key=lambda c: -rank[int(c)])
                    for c in extra:
                        if c not in sell:
                            sell.append(c)
                    sell = sell[: max(cfg.n_drop, len(extra))]
                sold = 0
                for c in sell:
                    if not can_sell[t, c]:
                        continue
                    px_c = px[c]
                    proceeds = qty[c] * px_c * (1 - cfg.slippage)
                    comm = max(proceeds * cfg.commission, 0.0)
                    cash += proceeds - comm
                    qty[c] = 0.0
                    avg_cost[c] = 0.0
                    sold += 1
                    n_trades += 1
                buy = [int(c) for c in buy_pool
                       if int(c) not in holdings_set][:sold]
                for c in buy:
                    _buy(c, _portfolio_value())

                # ── 目标权重再平衡（分数比例加权 / 逆波动率加权）────────
                # 引擎的 topk_dropout 只在换仓时按 max_position_size 下单，
                # 不做「已持仓向目标权重调平」。这里显式做一次调平，
                # 用来检验「相对权重加权是否真能改变夏普」。
                if getattr(sel, "rebalance_to_targets", False) and tgt_w:
                    pv = _portfolio_value()
                    for c in sorted(keep_set):
                        if c not in range(n_s):
                            continue
                        w_target = float(tgt_w.get(int(c), 0.0))
                        if w_target <= 0:
                            continue
                        px_c = px[c]
                        if not np.isfinite(px_c) or px_c <= 0:
                            continue
                        cur_val = qty[c] * px_c
                        tgt_val = pv * w_target
                        if tgt_val > cur_val * 1.02:      # 买入差额
                            need = tgt_val - cur_val
                            if not can_buy[t, c]:
                                continue
                            q = int(need / px_c / cfg.board_lot) * cfg.board_lot
                            if q <= 0:
                                continue
                            cost = q * px_c * (1 + cfg.slippage)
                            comm = max(cost * cfg.commission, 0.0)
                            if cost + comm > cash:
                                continue
                            cash -= cost + comm
                            qty[c] += q
                            avg_cost[c] = ((avg_cost[c] * (qty[c] - q) + cost) / qty[c]
                                           if qty[c] else 0.0)
                            n_trades += 1
                        elif cur_val > tgt_val * 1.02:     # 卖出超额
                            if not can_sell[t, c]:
                                continue
                            excess = cur_val - tgt_val
                            q_sell = int(excess / px_c / cfg.board_lot) * cfg.board_lot
                            q_sell = min(q_sell, int(qty[c]))
                            if q_sell <= 0:
                                continue
                            proceeds = q_sell * px_c * (1 - cfg.slippage)
                            comm = max(proceeds * cfg.commission, 0.0)
                            cash += proceeds - comm
                            qty[c] -= q_sell
                            n_trades += 1

        # ---- 估值快照 ----
        val_px = np.where(np.isfinite(px) & (px > 0), px, last_px)
        equity = cash + float(np.sum(qty * val_px))
        equity_curve.append((master[t], equity))

    return _metrics(equity_curve, n_trades, master)


def _metrics(
    equity_curve: List[Tuple[pd.Timestamp, float]],
    n_trades: int,
    master: pd.DatetimeIndex,
) -> Dict[str, object]:
    if not equity_curve:
        return {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
                "total_trades": 0, "yearly": {}}
    s = pd.Series([v for _, v in equity_curve], index=pd.DatetimeIndex(
        [d for d, _ in equity_curve]))
    rets = s.pct_change().dropna()
    total_return = float(s.iloc[-1] / INITIAL_CASH - 1.0)
    vol = float(rets.std() * np.sqrt(252)) if len(rets) else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    mdd = float(dd.min()) if len(dd) else 0.0
    yearly_vals = s.resample("YE").last()
    yr = yearly_vals.pct_change().dropna()
    yearly = {str(d.year): round(float(v), 6) for d, v in yr.items()}
    return {
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "annualized_return": float((1 + total_return) ** (365 / max(
            (s.index[-1] - s.index[0]).days, 1)) - 1) if total_return > -1 else -1.0,
        "volatility": vol,
        "max_drawdown": mdd,
        "total_trades": int(n_trades),
        "yearly": yearly,
    }


# ══════════════════════════════════════════════════════════════════
# 5. 其它策略的分数（用于合成）
# ══════════════════════════════════════════════════════════════════
def rsi_71421_score(mats: Dict[str, np.ndarray]) -> np.ndarray:
    """rsi_71421 策略的日频分数（不含周频/lag 掩码）。"""
    close = mats["close"]
    vol = mats["volume"]
    min_cross = 30
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close[1:] / close[:-1] - 1.0
        ret = np.vstack([np.full((1, close.shape[1]), np.nan), ret])
        vchg = vol[1:] / vol[:-1] - 1.0
        vchg = np.vstack([np.full((1, close.shape[1]), np.nan), vchg])
        s = pd.DataFrame(close)
        v = pd.DataFrame(vol)
        out: Dict[str, np.ndarray] = {}
        for n in (7, 14, 21):
            delta = s.diff()
            gain = delta.clip(lower=0).rolling(n).mean()
            loss = (-delta.clip(upper=0)).rolling(n).mean()
            rs = gain / loss
            rsi = 100 - 100 / (1 + rs)
            out[f"RSI{n}"] = (-rsi).to_numpy()
            out[f"STD{n}"] = (-(s.rolling(n).std() / s)).to_numpy()
            out[f"CORR{n}"] = (-s.rolling(n).corr(v)).to_numpy()
            out[f"CORD{n}"] = (-pd.DataFrame(ret).rolling(n).corr(
                pd.DataFrame(vchg))).to_numpy()
    zsum = np.zeros_like(close)
    wsum = np.zeros_like(close)
    for name, mat in out.items():
        z = cs_zscore(mat, min_cross)
        good = np.isfinite(z)
        zsum[good] += np.nan_to_num(z)[good]
        wsum[good] += 1.0
    return np.where(wsum > 0, zsum / np.where(wsum > 0, wsum, 1.0), np.nan)


def factor_ranking_score(mats: Dict[str, Any]) -> np.ndarray:
    """factor_ranking：动量(126d 跳 20d) + 低波(60d) + 低振幅(10d) 等权 z。"""
    close = mats["close"]
    high = mats["high"]
    low = mats["low"]
    min_cross = 30
    s = pd.DataFrame(close)
    momentum = (s.shift(20) / s.shift(146) - 1.0).to_numpy()
    low_vol = (-s.pct_change().rolling(60, min_periods=60).std()).to_numpy()
    amp = ((high - low) / np.where(close == 0, np.nan, close))
    low_amp = (-pd.DataFrame(amp).rolling(10).mean()).to_numpy()
    zsum = np.zeros_like(close)
    wsum = np.zeros_like(close)
    for mat in (momentum, low_vol, low_amp):
        z = cs_zscore(mat, min_cross)
        good = np.isfinite(z)
        zsum[good] += np.nan_to_num(z)[good]
        wsum[good] += 1.0
    return np.where(wsum > 0, zsum / np.where(wsum > 0, wsum, 1.0), np.nan)


# ══════════════════════════════════════════════════════════════════
# 6. 策略分数矩阵的横截面相关分析
# ══════════════════════════════════════════════════════════════════
def cs_rank_ic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐日截面 RankIC（斯皮尔曼），返回 (n_dates,) 序列。"""
    n_d = a.shape[0]
    out = np.full(n_d, np.nan)
    for t in range(n_d):
        x, y = a[t], b[t]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30:
            continue
        rx = pd.Series(x[ok]).rank().to_numpy()
        ry = pd.Series(y[ok]).rank().to_numpy()
        sx, sy = rx.std(), ry.std()
        if sx > 0 and sy > 0:
            out[t] = float(np.corrcoef(rx, ry)[0, 1])
    return out


def score_correlation_matrix(
    scores: Dict[str, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, object]:
    """策略分数矩阵两两之间的平均横截面 RankIC 矩阵。"""
    names = list(scores)
    n = len(names)
    mat = np.full((n, n), np.nan)
    ic_series: Dict[str, List[float]] = {}
    for i, a in enumerate(names):
        for j in range(i, n):
            b = names[j]
            ic = cs_rank_ic(scores[a], scores[b])
            if dates is not None:
                mask = np.array([d in dates for d in range(len(ic))])
                ic = ic[mask]
            m = float(np.nanmean(ic)) if np.isfinite(ic).any() else np.nan
            mat[i, j] = mat[j, i] = m
            if i == j:
                ic_series[a] = [round(float(v), 6) for v in ic[np.isfinite(ic)]]
    return {
        "names": names,
        "mean_rank_ic_matrix": [[None if not np.isfinite(v) else round(float(v), 4)
                                 for v in row] for row in mat],
        "ic_std": {k: (round(float(np.std(v)), 4) if len(v) > 1 else None)
                   for k, v in ic_series.items()},
        "ic_mean": {k: (round(float(np.mean(v)), 5) if len(v) > 1 else None)
                    for k, v in ic_series.items()},
        "n_days": {k: len(v) for k, v in ic_series.items()},
    }


# ══════════════════════════════════════════════════════════════════
# 7. OOS 切分工具
# ══════════════════════════════════════════════════════════════════
def split_index(master: pd.DatetimeIndex, split: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (IS 行索引, OOS 行索引)。"""
    is_idx = np.where(master < split)[0]
    oos_idx = np.where(master >= split)[0]
    return is_idx, oos_idx


def slice_mats(mats: Dict[str, Any], rows: np.ndarray) -> Dict[str, Any]:
    out = {k: v for k, v in mats.items() if k not in ("master", "codes", "stock_data")}
    out["master"] = mats["master"][rows]
    out["codes"] = mats["codes"]
    out["stock_data"] = mats["stock_data"]
    for k in ("close", "high", "low", "volume"):
        out[k] = mats[k][rows]
    return out


def simulate_subset(
    score: np.ndarray,
    mats: Dict[str, Any],
    trad: Dict[str, np.ndarray],
    cfg: SimConfig,
    rows: np.ndarray,
) -> Dict[str, object]:
    """在 master 的子区间上跑模拟（用于 OOS 评估）。"""
    sub_mats = slice_mats(mats, rows)
    sub_trad = {k: v[rows] for k, v in trad.items()}
    return simulate(score[rows], sub_mats, sub_trad, cfg)
