#!/usr/bin/env python3
"""模型类策略（GBM）超参搜索 — 真实盘口径 + OOS 验证

## 口径（硬性）
- lag=1（收盘出信号，下一交易日收盘成交）
- 可成交性检查开（涨跌停 / ST / 成交量 / 参与率）
- 股票池：data/qlib_official_data/instruments/all.txt（带上市/退市日 → PIT）
- 3 年窗口 2023-05-08 → 2026-05-08（qlib 官方数据末日）

## OOS 切分
- IS（选参）：2023-05-08 → 2025-05-08
- OOS（验证）：2025-05-08 → 2026-05-08
- 变体在**完整 3 年窗口**上跑一次真实引擎回测，然后从日度净值曲线
  （equity_curve）切出 IS / OOS 两段各自算指标。
  → 一次回测同时给出 IS 和 OOS，不需要跑两遍。
  → 切分口径与引擎一致：夏普 = (mean(r) - rf/252) * sqrt(252) / std(r)，
     rf = 0.03（EnhancedMetricsCalculator 默认）。

## 断点续跑
每个变体一个 JSON：data/backtest/gbm_search_runs/<variant_id>.json
已存在且 status=ok 就跳过。中断后重跑同样命令即可续跑。

## 并行
--jobs N：N 个进程各跑一个变体（每个变体内部 LightGBM 用 n_jobs 线程）。
注意内存：42 因子 z 矩阵 (727×5238×42) float32 = 640MB + presence 640MB，
每个 worker 约 2-3GB → jobs=4 时约 12GB，安全。

## 用法
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/search_gbm_params.py --stage all --jobs 4
  ... --stage horizon
  ... --only v_id_h1 v_id_h5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies.strategy_factory import (  # noqa: E402
    StrategyFactory,
)
from app.services.backtest.strategies.gbm_lab_strategy import (  # noqa: E402
    GbmLabStrategy,
    DEFAULT_PANEL_DIR,
    LEGACY_PANEL_DIR,
)

QLIB_OFFICIAL = os.path.join(ROOT, "data", "qlib_official_data")
OUT_DIR = os.path.join(ROOT, "data", "backtest")
RUN_DIR = os.path.join(OUT_DIR, "gbm_search_runs")

# 真实盘口径常量
LAG = 1
TOPK = 60
N_DROP = 5
MPS = 0.015833
IS_END = pd.Timestamp("2025-05-08")
WIN_START = pd.Timestamp("2023-05-08")
WIN_END = pd.Timestamp("2026-05-08")

# 运行时注册（不改 strategy_factory.py —— 那是正式注册表）
StrategyFactory.register_strategy("gbm_lab", GbmLabStrategy)


# ── 指标工具（口径完全对齐 portfolio_manager_array.get_performance_metrics）──
INITIAL_CASH = 1_000_000.0


def _metrics_from_equity(
    curve: List[List[Any]], base_value: Optional[float] = None
) -> Dict[str, float]:
    """从日度净值曲线算 sharpe / return / mdd。

    **与引擎逐位一致**（app/services/backtest/core/portfolio_manager_array.py
    → get_performance_metrics）：
      values  = equity_curve 的 portfolio_value 序列
      returns = values.pct_change().dropna()
      sharpe  = returns.mean() / returns.std() * sqrt(252)   ← **不减无风险利率**
      total_return = (values[-1] - initial_cash) / initial_cash
      max_drawdown = min((cumprod(1+r) / expanding_max) - 1)
    注意：引擎的夏普**没有扣无风险利率**（EnhancedMetricsCalculator 里有扣 rf，
    但回测主路径走的是 portfolio_manager_array 的版本）。为与基线可比，
    这里严格复刻引擎口径。

    base_value: 段起始净值（IS 段=initial_cash；OOS 段=IS 段末净值）。
                None → 用 INITIAL_CASH。
    """
    if not curve or len(curve) < 3:
        return {"sharpe": 0.0, "return": 0.0, "mdd": 0.0, "days": 0}
    vals = np.array([float(v) for _, v in curve], dtype=np.float64)
    if not np.isfinite(vals).all():
        return {"sharpe": 0.0, "return": 0.0, "mdd": 0.0, "days": len(vals)}
    rets = np.diff(vals) / vals[:-1]
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    base = INITIAL_CASH if base_value is None else float(base_value)
    total_return = float(vals[-1] / base - 1.0) if base else 0.0
    cum = np.cumprod(1.0 + rets)
    run_max = np.maximum.accumulate(cum)
    mdd = float(np.min((cum - run_max) / run_max))
    return {
        "sharpe": sharpe,
        "return": total_return,
        "mdd": mdd,
        "days": int(len(vals)),
    }


def _split_metrics(
    curve: List[List[Any]], trades: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """把日度净值曲线切 IS / OOS 两段，分别算指标。

    IS  = [窗口起, 2025-05-08)
    OOS = [2025-05-08, 窗口末]
    OOS 段的收益以 **IS 段末净值** 为基数（复利衔接，不是重新归零）。
    胜率口径对齐引擎：只统计 SELL 交易（引擎 winning/losing 都只取 SELL）。
    """
    out: Dict[str, Any] = {}

    def seg_rows(lo: Optional[pd.Timestamp], hi: Optional[pd.Timestamp]):
        rows = []
        for row in curve:
            d = pd.Timestamp(row[0])
            if lo is not None and d < lo:
                continue
            if hi is not None and d >= hi:
                continue
            rows.append(row)
        return rows

    is_rows = seg_rows(None, IS_END)
    oos_rows = seg_rows(IS_END, None)
    all_rows = seg_rows(None, None)

    # OOS 基数 = IS 段最后一根 K 线的净值
    oos_base = float(is_rows[-1][1]) if is_rows else INITIAL_CASH

    for label, rows, base in (
        ("is", is_rows, INITIAL_CASH),
        ("oos", oos_rows, oos_base),
        ("full", all_rows, INITIAL_CASH),
    ):
        m = _metrics_from_equity(rows, base)
        # 段内交易数与胜率（引擎口径：只算 SELL 的 pnl）
        sells = 0
        win = 0
        lose = 0
        for t in trades:
            ts = t.get("timestamp")
            if not ts:
                continue
            d = pd.Timestamp(ts).normalize()
            if label == "is" and d >= IS_END:
                continue
            if label == "oos" and d < IS_END:
                continue
            if t.get("action") != "SELL":
                continue
            sells += 1
            pnl = t.get("pnl")
            if pnl is None:
                continue
            if float(pnl) > 0:
                win += 1
            elif float(pnl) < 0:
                lose += 1
        m["sells"] = sells
        m["win_rate"] = round(win / (win + lose), 4) if (win + lose) else 0.0
        m["n_trades"] = sum(
            1 for t in trades
            if t.get("timestamp") and (
                (label == "is" and pd.Timestamp(t["timestamp"]).normalize() < IS_END)
                or (label == "oos" and pd.Timestamp(t["timestamp"]).normalize() >= IS_END)
                or label == "full"
            )
        )
        out[label] = m
    return out


def _yearly_from_equity(curve: List[List[Any]]) -> Dict[str, float]:
    if not curve:
        return {}
    df = pd.DataFrame(curve, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    yearly = df["value"].resample("YE").last()
    r = yearly.pct_change().dropna()
    return {str(int(ts.year)): round(float(v), 6) for ts, v in r.items()}


# ── 股票池 ────────────────────────────────────────────────────────
def universe_from_instruments(instruments_dir: str) -> List[str]:
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


# ── 单变体回测 ────────────────────────────────────────────────────
def run_variant(
    variant_id: str,
    cfg: Dict[str, Any],
    codes: List[str],
    start: datetime,
    end: datetime,
    out_path: str,
) -> Dict[str, Any]:
    """跑一个变体（真实引擎），写 JSON。已存在则直接返回（断点续跑）。"""
    if os.path.exists(out_path):
        try:
            with open(out_path) as fh:
                cached = json.load(fh)
            if cached.get("status") == "ok":
                print(f"  [skip] {variant_id} (已有结果)", flush=True)
                return cached
        except Exception:
            pass

    executor = BacktestExecutor(
        data_dir=ROOT + "/data",
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    try:
        res = asyncio.run(
            executor.run_backtest(
                strategy_name="gbm_lab",
                stock_codes=codes,
                start_date=start,
                end_date=end,
                strategy_config={**cfg, "lag": LAG},
                backtest_config=BacktestConfig(
                    enable_tradability=True,
                    initial_cash=1_000_000.0,
                    commission_rate=0.0003,
                    slippage_rate=0.0001,
                    max_position_size=MPS,
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
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        payload = {
            "variant_id": variant_id,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "config": cfg,
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        with open(out_path, "w") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)
        return payload

    curve = res.get("equity_curve") or []
    trades = res.get("trade_history") or []
    seg = _split_metrics(curve, trades)
    yearly = _yearly_from_equity(curve)

    payload: Dict[str, Any] = {
        "variant_id": variant_id,
        "status": "ok",
        "config": cfg,
        "wall_s": round(time.perf_counter() - t0, 1),
        "engine": {
            k: res.get(k)
            for k in (
                "sharpe_ratio", "total_return", "max_drawdown", "total_trades",
                "win_rate", "final_value", "volatility", "annualized_return",
            )
            if k in res
        },
        "is": seg["is"],
        "oos": seg["oos"],
        "full": seg["full"],
        "yearly": yearly,
        "loaded_stocks": res.get("stock_count") or res.get("actual_stock_count"),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)
    print(
        f"  [done] {variant_id}: IS sharpe={seg['is']['sharpe']:.3f} "
        f"ret={seg['is']['return']:.2%} mdd={seg['is']['mdd']:.2%} | "
        f"OOS sharpe={seg['oos']['sharpe']:.3f} ret={seg['oos']['return']:.2%} "
        f"mdd={seg['oos']['mdd']:.2%} ({payload['wall_s']}s)",
        flush=True,
    )
    return payload


# ── 变体定义 ──────────────────────────────────────────────────────
BASE: Dict[str, Any] = {
    "topk": TOPK,
    "n_drop": N_DROP,
    "rebalance_days": "weekly",
    "rebalance_weekday": 4,
    # 基线选择依据（backend/scripts/diag_gbm_ic.py 实测 1500 只 / 3 年 rank-IC）：
    #   目标变量 rank 的模型 IC 最高（h1 0.0616 / h5 0.0758 / h20 0.0779），
    #   raw 目标被市场 beta 污染（IC 0.020~0.033），vol_std 最差（0.007~0.009）。
    #   训练 horizon 越长，模型在长周期评估上的 IC 越好。
    #   → 基线 = horizon 5 + target rank（IC 最优组合），各维度在此基线上单因子扫描。
    "horizon": 5,
    "label_anchor": "trade",
    "target": "rank",
    "model": "lgbm",
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "retrain": "monthly",
    "train_window": "expand",
    "warmup_months": 2,
    "feature_set": "f42",
    "use_presence": False,
    "target_vol": 0.20,
    "blend": 1.0,
    "purge": True,
    "n_jobs": 2,
}


def _v(base: Dict[str, Any], **kw) -> Dict[str, Any]:
    out = dict(base)
    out.update(kw)
    return out


def build_variants(stage: str) -> List[Tuple[str, Dict[str, Any]]]:
    """返回 [(variant_id, strategy_config), ...]"""
    B = BASE
    vs: List[Tuple[str, Dict[str, Any]]] = []

    def add(suffix: str, **kw):
        vs.append((f"v_{suffix}", _v(B, **kw)))

    # ── Stage 0: 关键修复的 A/B（先确认修复本身有没有用）──────────
    if stage in ("all", "fix"):
        add("base")
        # 面板源 A/B：features_full（干净） vs features/day（旧的，日期重复）
        add("panel_legacy", panel_dir=LEGACY_PANEL_DIR)
        # 涨停掩码 A/B（修 vs 旧口径）
        add("mask_legacy", limit_mask_mode="legacy")
        # purge A/B
        add("nopurge", purge=False)
        # 标签锚点 A/B
        add("anchor_signal", label_anchor="signal")
        # presence 哑变量 A/B
        add("presence", use_presence=True)
        # subsample_freq A/B（D3）
        add("subsample_off", subsample_freq=0)
        # 等权因子基线（blend=0 → 完全不看模型分）
        add("eq_only", blend=0.0)

    # ── Stage 1: 预测标签 horizon ──────────────────────────────────
    if stage in ("all", "horizon"):
        for h in (1, 3, 5, 10, 20, 60):
            add(f"h{h}", horizon=h)

    # ── Stage 2: 目标变量 ──────────────────────────────────────────
    if stage in ("all", "target"):
        for t in ("raw", "rank", "cs_z", "ind_neutral", "vol_std"):
            add(f"t_{t}", target=t)

    # ── Stage 3: 模型超参 ──────────────────────────────────────────
    if stage in ("all", "hparam"):
        add("hp_n300", n_estimators=300)
        add("hp_d5", max_depth=5)
        add("hp_d2", max_depth=2)
        add("hp_lr002", learning_rate=0.02)
        add("hp_lr010", learning_rate=0.10)
        add("hp_mcs100", min_child_samples=100)
        add("hp_mcs500", min_child_samples=500)
        add("hp_sub06", subsample=0.6)
        add("hp_sub100", subsample=1.0)
        add("hp_col06", colsample_bytree=0.6)
        add("hp_col100", colsample_bytree=1.0)
        add("hp_l1_01", reg_alpha=0.1)
        add("hp_l2_10", reg_lambda=10.0)
        add("hp_leaves31", num_leaves=31)

    # ── Stage 4: 预测频率 + 重训窗口 ───────────────────────────────
    if stage in ("all", "retrain"):
        add("rt_weekly", retrain="weekly")
        add("rt_quarterly", retrain="quarterly")
        add("tw_rolling252", train_window="rolling", train_window_days=252)
        add("tw_rolling504", train_window="rolling", train_window_days=504)
        add("tw_roll_weekly", retrain="weekly", train_window="rolling",
            train_window_days=504)

    # ── Stage 5: 组合层 target_vol ────────────────────────────────
    if stage in ("all", "vol"):
        for tv in (0.10, 0.15, 0.20, 0.25):
            add(f"vt{int(tv * 100)}", target_vol=tv)
        add("vt_off", target_vol=0.0)

    # ── Stage 6: 特征集 ───────────────────────────────────────────
    if stage in ("all", "feature"):
        add("fs_f25", feature_set="f25")
        add("fs_f158", feature_set="alpha158")
        add("fs_ic20", top_n_factors=20)
        add("fs_ic40", top_n_factors=40)

    # ── Stage 7: 模型族 ───────────────────────────────────────────
    if stage in ("all", "model"):
        add("md_xgb", model="xgb")
        add("md_ridge", model="ridge")
        add("md_enet", model="elasticnet")

    # ── Stage 8: 与等权因子分集成 ────────────────────────────────
    if stage in ("all", "blend"):
        for b in (0.0, 0.3, 0.5, 0.7, 1.0):
            add(f"bl{int(b * 100)}", blend=b)

    return vs


# ── 并行执行 ──────────────────────────────────────────────────────
def _worker(args) -> Dict[str, Any]:
    variant_id, cfg, codes, start, end, out_path = args
    try:
        return run_variant(variant_id, cfg, codes, start, end, out_path)
    except Exception as e:  # noqa: BLE001
        return {
            "variant_id": variant_id,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "config": cfg,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="fix/horizon/target/hparam/retrain/vol/feature/model/blend/all")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=[],
                    help="只跑指定 variant_id")
    ap.add_argument("--out", default="gbm_search_results.json")
    ap.add_argument("--list", action="store_true", help="只列变体不跑")
    args = ap.parse_args()

    os.makedirs(RUN_DIR, exist_ok=True)
    variants = build_variants(args.stage)
    if args.only:
        keep = set(args.only)
        variants = [v for v in variants if v[0] in keep]

    print(f"[search] stage={args.stage} 变体数={len(variants)} "
          f"jobs={args.jobs}", flush=True)
    if args.list:
        for vid, cfg in variants:
            print(f"  {vid}: {json.dumps(cfg, ensure_ascii=False)}")
        return

    codes = universe_from_instruments(QLIB_OFFICIAL)
    print(f"[search] PIT 股票池: {len(codes)} 只", flush=True)
    print(f"[search] 窗口 {WIN_START.date()} → {WIN_END.date()} "
          f"(IS < {IS_END.date()}, OOS >= {IS_END.date()})", flush=True)
    print(f"[search] 口径: lag={LAG}, 可成交性=开, topk={TOPK}, n_drop={N_DROP}",
          flush=True)

    start = datetime(WIN_START.year, WIN_START.month, WIN_START.day)
    end = datetime(WIN_END.year, WIN_END.month, WIN_END.day)

    jobs = [
        (vid, cfg, codes, start, end, os.path.join(RUN_DIR, f"{vid}.json"))
        for vid, cfg in variants
    ]

    results: Dict[str, Any] = {}
    if args.jobs <= 1:
        for job in jobs:
            r = _worker(job)
            results[r["variant_id"]] = r
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.jobs) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                results[r["variant_id"]] = r

    # ── 汇总 ──────────────────────────────────────────────────────
    rows = []
    for vid, cfg in variants:
        r = results.get(vid) or {}
        if r.get("status") != "ok":
            rows.append({
                "variant_id": vid, "status": r.get("status", "missing"),
                "error": r.get("error", ""), "config": cfg,
            })
            continue
        rows.append({
            "variant_id": vid, "status": "ok", "config": cfg,
            "wall_s": r.get("wall_s"),
            "is_sharpe": r["is"]["sharpe"], "is_return": r["is"]["return"],
            "is_mdd": r["is"]["mdd"], "is_trades": r["is"]["n_trades"],
            "is_win_rate": r["is"]["win_rate"],
            "oos_sharpe": r["oos"]["sharpe"], "oos_return": r["oos"]["return"],
            "oos_mdd": r["oos"]["mdd"], "oos_trades": r["oos"]["n_trades"],
            "oos_win_rate": r["oos"]["win_rate"],
            "full_sharpe": r["full"]["sharpe"], "full_return": r["full"]["return"],
            "full_mdd": r["full"]["mdd"], "full_trades": r["full"]["n_trades"],
            "full_win_rate": r["full"]["win_rate"],
            "yearly": r.get("yearly", {}),
        })

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: -(r.get("is_sharpe") or -9))

    payload = {
        "methodology": {
            "lag": LAG,
            "enable_tradability": True,
            "realistic": True,
            "window": f"{WIN_START.date()}→{WIN_END.date()}",
            "is_window": f"{WIN_START.date()}→{IS_END.date()}",
            "oos_window": f"{IS_END.date()}→{WIN_END.date()}",
            "universe": "qlib_official_data/instruments/all.txt (PIT)",
            "n_universe": len(codes),
            "sharpe_formula": "mean(daily_ret)/std(daily_ret)*sqrt(252) "
                              "(引擎口径，不减无风险利率)",
            "selection_rule": "按 IS 夏普排序选参；OOS 仅作验证，不参与选择",
        },
        "stage": args.stage,
        "total_attempts": len(rows),
        "n_ok": len(ok_rows),
        "n_error": len(rows) - len(ok_rows),
        "results": rows,
        "is_ranking": [
            {
                "variant_id": r["variant_id"],
                "is_sharpe": r.get("is_sharpe"),
                "oos_sharpe": r.get("oos_sharpe"),
                "is_return": r.get("is_return"),
                "oos_return": r.get("oos_return"),
                "is_mdd": r.get("is_mdd"),
                "oos_mdd": r.get("oos_mdd"),
                "is_trades": r.get("is_trades"),
                "oos_trades": r.get("oos_trades"),
                "config": r.get("config"),
            }
            for r in ok_rows
        ],
    }
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)

    print(f"\n[search] 总尝试 {len(rows)}，成功 {len(ok_rows)}，"
          f"失败 {len(rows) - len(ok_rows)}", flush=True)
    print(f"[search] 已写入 {path}", flush=True)
    print(f"\n{'variant':<22}{'IS夏普':>9}{'OOS夏普':>9}{'IS收益':>10}"
          f"{'OOS收益':>10}{'IS回撤':>9}{'OOS回撤':>9}{'IS胜率':>8}{'OOS胜率':>8}"
          f"{'IS笔':>7}{'OOS笔':>7}")
    for r in ok_rows:
        print(f"{r['variant_id']:<22}{r['is_sharpe']:>9.3f}{r['oos_sharpe']:>9.3f}"
              f"{r['is_return']:>10.2%}{r['oos_return']:>10.2%}"
              f"{r['is_mdd']:>9.2%}{r['oos_mdd']:>9.2%}"
              f"{r['is_win_rate']:>8.1%}{r['oos_win_rate']:>8.1%}"
              f"{r['is_trades']:>7d}{r['oos_trades']:>7d}")


if __name__ == "__main__":
    main()
