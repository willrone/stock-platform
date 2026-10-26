#!/usr/bin/env python3
"""多策略合成实验 · 第 0 步：抽取各注册策略的**真实**分数矩阵 + 缓存基础数据。

## 为什么用策略类本身抽分数
`ensemble_lab.py` 里的手写重实现（build_42f_score / rsi_71421_score / ...）
只覆盖了 3 条腿，且 honest_composite 的面板因子读法与策略类不同
（策略类用 `duplicated(keep='first')`，features/day 是两段拼接 → 面板因子
只有约一半交易日有值）。要拿"引擎真实分数"做合成，必须**直接调用策略类的
`prepare_backtest_data`**，再从 `stock_data[code].attrs[_SCORES_ATTR]` 取回分数。

## 缓存（可断点续跑）
  cache/mats.npz      close/high/low/volume 矩阵 + codes + master 日历
  cache/trad.npz      can_buy / can_sell / max_buy_amount
  cache/legs/*.npz    各策略的 (n_dates × n_stocks) 日频分数矩阵

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/ensemble_legs.py --legs all
  ... --legs 42f rsi     # 只抽指定腿（已抽过的自动跳过）
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402

CACHE = "/Users/rone/.hermes/cache/scratch/stock_ens/cache"
LEG_DIR = os.path.join(CACHE, "legs")
os.makedirs(LEG_DIR, exist_ok=True)

# 腿定义：(key, 策略类导入路径, 类名, prepare 配置, 分数 attr)
# prepare 配置一律 lag=0 + rebalance_days='daily' → 掩码与滞后交给模拟器，
# 与 ensemble_lab.simulate 的语义一致（策略内 lag / 模拟器 lag 等价）。
LEG_DEFS: Dict[str, Dict[str, Any]] = {
    # ⚠️ 42f 有两个因子集，必须**显式钉死**，否则主线改默认会让历史实验不可比：
    #   use_panel=False → 25 个公式因子（2026-10-27 02:13 之后的主线默认）
    #   use_panel=True  → 25 公式 + 17 个 Alpha158 面板因子 = 42 因子
    # 2026-10-27 实测：主线从 42 因子切到 25 因子后，同一合成配置的引擎夏普
    # 从 1.2817 掉到 1.1081（同一配置、同一数据、只差因子集）→ 必须钉死。
    "42f": {
        "cls": ("app.services.backtest.strategies.honest_composite_strategy",
                "HonestCompositeStrategy"),
        "attr": "_honest_composite_scores",
        # 默认 = 当前主线（25 公式因子），显式写出 use_panel 避免默认值漂移
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "daily", "lag": 0,
                "use_panel": False},
    },
    "42f42": {
        "cls": ("app.services.backtest.strategies.honest_composite_strategy",
                "HonestCompositeStrategy"),
        "attr": "_honest_composite_scores",
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "daily", "lag": 0,
                "use_panel": True},
    },
    "rsi": {
        "cls": ("app.services.backtest.strategies.rsi_71421_strategy",
                "Rsi71421Strategy"),
        "attr": "_rsi_71421_scores",
        "cfg": {"topk": 30, "n_drop": 3, "rebalance_days": "daily", "lag": 0},
    },
    "fr": {
        "cls": ("app.services.backtest.strategies.factor_ranking_strategy",
                "FactorRankingStrategy"),
        "attr": "_factor_ranking_scores",
        "cfg": {"topk": 50, "n_drop": 5, "rebalance_days": "daily", "lag": 0},
    },
    "gbm": {
        "cls": ("app.services.backtest.strategies.gbm_voltgt_strategy",
                "GbmVoltgtStrategy"),
        "attr": "_gbm_voltgt_scores",
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "daily", "lag": 0,
                "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
                "warmup_months": 2, "target_vol": 0.20, "vt_window": 20},
    },
    "gbm_raw": {
        # target_vol=10 → scale=min(1, 10/市场波动率) 恒为 1 → 不做波动率目标截断
        # → 保留 LightGBM 原始预测分（未截断），作为「未加仓位层」的腿
        "cls": ("app.services.backtest.strategies.gbm_voltgt_strategy",
                "GbmVoltgtStrategy"),
        "attr": "_gbm_voltgt_scores",
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "daily", "lag": 0,
                "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
                "warmup_months": 2, "target_vol": 10.0, "vt_window": 20},
    },
    "mf": {
        "cls": ("app.services.backtest.strategies.strategies",
                "MultiFactorStrategy"),
        "attr": "_multi_factor_scores",
        "cfg": {"topk": 30, "n_drop": 3, "rebalance_days": "daily", "lag": 0},
    },
}


def import_cls(path: str, name: str):
    mod = __import__(path, fromlist=[name])
    return getattr(mod, name)


# ── 缓存：矩阵 ──────────────────────────────────────────────────────
def cache_mats() -> Dict[str, Any]:
    mpath = os.path.join(CACHE, "mats.npz")
    if os.path.exists(mpath):
        z = np.load(mpath, allow_pickle=False)
        out = {k: z[k] for k in z.files}
        out["master"] = pd.DatetimeIndex(out["master"])
        out["codes"] = [str(c) for c in out["codes"]]
        print(f"[cache] mats 命中 ({len(out['master'])} 日 × {len(out['codes'])} 股)",
              flush=True)
        return out
    codes = L.universe_from_instruments()
    mats = L.load_matrices(codes)
    np.savez_compressed(
        mpath,
        close=mats["close"], high=mats["high"], low=mats["low"],
        volume=mats["volume"],
        master=mats["master"].to_numpy(dtype="datetime64[ns]"),
        codes=np.array(mats["codes"], dtype="U16"),
    )
    print(f"[cache] mats 写入 {mpath}", flush=True)
    return mats


def load_stock_data(mats: Dict[str, Any]) -> Dict[str, Any]:
    """加载原始 DataFrame 字典（策略 prepare 需要往 attrs 里写分数）。"""
    from app.services.backtest.execution.data_loader import DataLoader
    dl = DataLoader(ROOT + "/data", max_workers=8)
    codes = list(mats["codes"])
    sd = dl.load_multiple_stocks(codes, L.START, L.END)
    return {c: sd[c] for c in codes if c in sd}


def cache_trad(mats: Dict[str, Any], stock_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    tpath = os.path.join(CACHE, "trad.npz")
    if os.path.exists(tpath):
        z = np.load(tpath, allow_pickle=False)
        out = {k: z[k] for k in z.files}
        print("[cache] trad 命中", flush=True)
        return out
    # 窗口前一日收盘价：引擎的 BacktestExecutor._load_prev_close_map 会加载
    # 窗口前 10 个日历日 → 保证窗口首日的涨跌停也能判定。
    # 不传 prev_close → 首日涨停股漏拦 → 与引擎不一致（实测 5 只 BJ 股分歧）。
    prev_close: Dict[str, float] = {}
    try:
        from app.services.backtest.execution.data_loader import DataLoader
        dl2 = DataLoader(ROOT + "/data", max_workers=8)
        pre_s = L.START - timedelta(days=10)
        pre_e = L.START - timedelta(days=1)
        for code in list(mats["codes"]):
            try:
                df = dl2.load_stock_data(code, pre_s, pre_e)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            col = None
            for cand in ("$close", "close"):
                if cand in df.columns:
                    col = cand
                    break
            if col is None:
                continue
            ser = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(ser):
                prev_close[str(code)] = float(ser.iloc[-1])
    except Exception as e:  # noqa: BLE001
        print(f"[cache] prev_close 加载失败: {e}", flush=True)

    tm = __import__("app.services.backtest.execution.tradability",
                    fromlist=["build_tradability"]).build_tradability(
        stock_data, list(mats["master"]), list(mats["codes"]),
        participation_rate=L.PARTICIPATION, min_turnover=L.MIN_TURNOVER,
        prev_close=prev_close)
    out = {"can_buy": tm.can_buy, "can_sell": tm.can_sell,
           "max_buy_amount": tm.max_buy_amount}
    np.savez_compressed(tpath, **out)
    print(f"[cache] trad 写入 {tpath}", flush=True)
    return out


# ── 抽腿 ────────────────────────────────────────────────────────────
def extract_leg(
    key: str,
    mats: Dict[str, Any],
    stock_data: Dict[str, Any],
) -> np.ndarray:
    path = os.path.join(LEG_DIR, f"{key}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        print(f"[leg] {key} 命中缓存", flush=True)
        return z["score"]

    spec = LEG_DEFS[key]
    cls = import_cls(*spec["cls"])
    t0 = time.perf_counter()
    print(f"[leg] {key} 调 prepare_backtest_data ...", flush=True)

    strat = cls(dict(spec["cfg"]))
    asyncio.run(strat.prepare_backtest_data(stock_data, L.START, L.END))

    master: pd.DatetimeIndex = mats["master"]
    codes = list(mats["codes"])
    pos_of = {d: i for i, d in enumerate(master)}
    out = np.full((len(master), len(codes)), np.nan)
    n_set = 0
    for j, code in enumerate(codes):
        sm = stock_data[code].attrs.get(spec["attr"]) or {}
        for d, v in sm.items():
            i = pos_of.get(pd.Timestamp(d))
            if i is not None and np.isfinite(v):
                out[i, j] = v
                n_set += 1
    print(f"[leg] {key} 完成: {n_set} 个有效分数 "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    np.savez_compressed(path, score=out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", nargs="*", default=["all"])
    args = ap.parse_args()
    todo = list(LEG_DEFS) if args.legs == ["all"] else args.legs
    mats = cache_mats()
    stock_data = load_stock_data(mats)
    print(f"[setup] stock_data {len(stock_data)} 只", flush=True)
    cache_trad(mats, stock_data)
    for key in todo:
        extract_leg(key, mats, stock_data)
    print("[done]")


if __name__ == "__main__":
    main()
