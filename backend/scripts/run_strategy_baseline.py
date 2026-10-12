#!/usr/bin/env python3
"""全策略统一基线回测 runner（2026-10-12 立项）。

统一口径：
- 窗口   : 2023-05-12 → 2026-05-08（本地 qlib 数据最近 3 年，723 交易日）
- 宇宙   : data/qlib_data/features/day/*.parquet 全量（5513 只，沪深北）
           统计套利类（O(N^2) 配对扫描）限 --universe-size（默认 500，取排序尾部，清单落盘）
- 资金   : 1,000,000 初始（与 10-08 全市场回测同口径）
- 费率   : commission 0.0003 / slippage 0.0001 / max_position 0.2 /
           cash_reserve 0.05 / 整数倍 100 / 止亏 5% / 止盈 15% / 日频调仓
- 基准   : 全市场 5513 只等权买入持有（只算一次，全部策略共用同一基准序列）
- 策略参数: 全部用代码默认值，不传 strategy_config

用法:
  cd backend
  ./run_strategy_baseline.py --smoke              # 20 只冒烟（默认 A 批列表）
  ./run_strategy_baseline.py --batch a            # 12 策略 × 全量 5513
  ./run_strategy_baseline.py --batch b            # factor_ranking × 全量 5513
  ./run_strategy_baseline.py --batch c            # 3 统计套利 × 500
  ./run_strategy_baseline.py --strategies rsi macd --universe-size 200

产出:
  data/strategy_baseline/baseline_full_market_3y.json   （增量写，可断点重跑）
  data/strategy_baseline/baseline_universe_500.txt      （C 批宇宙清单，可复现）

断点续跑: 已存在 status=ok 的策略默认跳过；--rerun 强制重跑。
ML 类 4 策略（lightgbm/ml_ensemble/model_signal/model_topk_dropout）不跑：
official-baseline pkl joblib.load 段错误 + csi300/50 域 + 前视 → 记录为 unavailable。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
if not os.path.isdir(BACKEND_ROOT):
    BACKEND_ROOT = "/Users/rone/projects/stock-platform/backend"
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

# 降低 loguru 噪音，只留 WARNING+
try:
    from loguru import logger as _logger

    _logger.remove()
    _logger.add(sys.stderr, level=os.environ.get("BASELINE_LOG_LEVEL", "WARNING"))
except Exception:
    pass

DATA_ROOT = str(settings.DATA_ROOT_PATH)
QLIB_DAY = os.path.join(DATA_ROOT, "qlib_data", "features", "day")
OUT_DIR = os.path.join(DATA_ROOT, "strategy_baseline")

START = datetime(2023, 5, 12)
END = datetime(2026, 5, 8)
START_S = START.strftime("%Y-%m-%d")
END_S = END.strftime("%Y-%m-%d")

BATCH_A = [
    "moving_average",
    "rsi",
    "macd",
    "bollinger",
    "stochastic",
    "cci",
    "kdj",
    "obv",
    "value_factor",
    "momentum_factor",
    "low_volatility",
    "multi_factor",
]
BATCH_B = ["factor_ranking"]
# 统计套利: find_pairs O(N^2)，全量 5513 会爆 → 限定宇宙（见 --universe-size）
STATARB_BATCH_C = ["pairs_trading", "mean_reversion", "cointegration"]
# ML 类: 不参与基线（原因记录到输出 JSON）
ML_UNAVAILABLE = {
    "ml_ensemble_lgb_xgb_riskctl": "official-baseline pkl joblib.load 段错误(rc=139)；模型 csi300/50 只域 + 训练窗口覆盖回测期（前视）",
    "lightgbm": "official-baseline pkl joblib.load 段错误(rc=139)；模型 csi300/50 只域 + 前视",
    "model_signal": "依赖 official-baseline 预测引擎，pkl 加载段错误 + 域不匹配 + 前视",
    "model_topk_dropout": "依赖 official-baseline 预测引擎，pkl 加载段错误 + 域不匹配 + 前视",
}

PERF_KEYS = [
    "data_loading_s",
    "precompute_signals_s",
    "align_arrays_s",
    "main_loop_s",
    "metrics_s",
    "report_generation_s",
]

# 策略级回测配置覆盖（默认 0 处 → 用全局 0.2/5-15% 口径）
# factor_ranking 例外: TopK 等权组合的单仓上限必须与 topk 匹配
# (0.019 × 50 ≈ 95% 仓位)，否则 100 股/手约束下整组合建仓被拒；
# 且换仓由排名驱动，止盈止损必须关闭。来源: docs/backtest/FACTOR_RANKING_STRATEGY.md §3
STRATEGY_BACKTEST_OVERRIDES = {
    "factor_ranking": {"max_position_size": 0.019, "stop_loss_pct": 0.0, "take_profit_pct": 0.0},
    "model_topk_dropout": {"max_position_size": 0.019, "stop_loss_pct": 0.0, "take_profit_pct": 0.0},
}


def _f(x: Any, nd: int = 6) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, nd)


def conv(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, dict):
        return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    return o


def list_full_universe() -> List[str]:
    files = sorted(
        f[: -len(".parquet")] for f in os.listdir(QLIB_DAY) if f.endswith(".parquet")
    )
    return sorted(c.replace("_", ".") for c in files if c)


def pick_statarb_universe(full: List[str], size: int, out_file: str) -> List[str]:
    uni = sorted(full)[-size:] if len(full) > size else sorted(full)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as fh:
        fh.write("# stat-arb baseline universe (fixed for reproducibility)\n")
        fh.write("\n".join(uni) + "\n")
    print(f"[universe] stat-arb {size} 清单已落盘 {out_file} (实际 {len(uni)} 只)", flush=True)
    return uni


def load_closes(codes: List[str]) -> "pd.DataFrame":
    """全宇宙收盘价 → DataFrame(index=date, columns=code)，窗口内。

    用 pyarrow.dataset 一次读全目录（5513 只 ≈ 13M 行，~1.5s），
    再按代码过滤 + pandas 日期切片（避免逐文件 glob 慢路径）。
    """
    import pyarrow.dataset as ds

    # 注意: to_pandas 还原 MultiIndex 依赖显式选出索引列（仅选 $close 会丢索引）
    t0 = time.perf_counter()
    data = ds.dataset(QLIB_DAY, format="parquet")
    t = data.to_table(columns=["$close", "stock_code", "date"])
    df = t.to_pandas()
    if df.index.nlevels == 2 and df.index.names[0] is None:
        df.index.names = ["stock_code", "date"]
    df = df.reset_index()
    df = df[df["stock_code"].isin(set(codes))]
    df = df[(df["date"] >= START_S) & (df["date"] <= END_S)]
    df = df.set_index(["stock_code", "date"]).sort_index()
    out = df["$close"].unstack("stock_code").sort_index()
    print(
        f"[closes] shape={out.shape} read={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out


def compute_equal_weight_benchmark(closes: "pd.DataFrame") -> Dict[str, Any]:
    """全市场等权买入持有基准：日频等权收益复利。"""
    import numpy as np

    daily_ret = closes.pct_change(fill_method=None).fillna(0.0)
    ew_ret = daily_ret.mean(axis=1).fillna(0.0)
    value = (1.0 + ew_ret).cumprod()

    total = float(value.iloc[-1] - 1.0)
    n = len(value)
    annual = float(value.iloc[-1] ** (252.0 / n) - 1.0)
    std = float(ew_ret.std(ddof=1))
    vol = std * float(np.sqrt(252.0))
    sharpe = float(ew_ret.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0
    running_max = value.cummax()
    mdd = float(((value - running_max) / running_max).min())

    return {
        "name": "全市场等权买入持有",
        "symbol": "EQUI-WEIGHT-FULL-MKT",
        "n_stocks": int(closes.shape[1]),
        "n_days": int(n),
        "total_return": round(total, 6),
        "annualized_return": round(annual, 6),
        "volatility": round(vol, 6),
        "sharpe_ratio": round(sharpe, 6),
        "max_drawdown": round(mdd, 6),
        "daily_value": {
            d.strftime("%Y-%m-%d"): round(float(v), 6) for d, v in value.items()
        },
    }


def compare_to_benchmark(
    res: Dict[str, Any], benchmark: Dict[str, Any]
) -> Dict[str, Any]:
    """策略 vs 等权基准（策略净值 = portfolio_history.portfolio_value，对齐基准日期轴）。"""
    import numpy as np

    bench_dates = sorted(benchmark["daily_value"].keys())
    bench_vals = np.array([benchmark["daily_value"][d] for d in bench_dates])
    if len(bench_vals) < 2 or bench_vals[0] <= 0:
        return {"note": "benchmark series unusable"}
    bench_ret = bench_vals[1:] / bench_vals[:-1] - 1.0

    by_date = {
        str(h.get("date", ""))[:10]: h.get("portfolio_value")
        for h in (res.get("portfolio_history") or [])
    }
    series: List[float] = []
    last: Optional[float] = None
    for d in bench_dates:
        v = by_date.get(d)
        if v is not None:
            last = float(v)
        if last is not None:
            series.append(last)
    if len(series) < 2:
        return {"note": "strategy has no portfolio_value series"}

    s = np.array(series, dtype=float)
    s_norm = s / s[0]
    n = min(len(s_norm), len(bench_vals))
    s_n, b_n = s_norm[:n], bench_vals[:n]
    s_ret = s_n[1:] / s_n[:-1] - 1.0
    b_ret = bench_ret[: len(s_ret)]
    m = min(len(s_ret), len(b_ret))
    s_ret, b_ret = s_ret[:m], b_ret[:m]

    s_ann = float((1 + s_ret).prod() ** (252.0 / m) - 1.0)
    b_ann = float((1 + b_ret).prod() ** (252.0 / m) - 1.0)
    diff = s_ret - b_ret
    dstd = float(np.std(diff, ddof=1))
    te = dstd * float(np.sqrt(252.0))
    ir = float(diff.mean() / dstd * np.sqrt(252.0)) if dstd > 0 else 0.0
    corr = float(np.corrcoef(s_ret, b_ret)[0, 1]) if m > 2 else None

    return {
        "benchmark": benchmark["symbol"],
        "common_days": int(m),
        "strategy_ann_return": _f(s_ann),
        "benchmark_ann_return": _f(b_ann),
        "excess_ann_return": _f(s_ann - b_ann),
        "tracking_error": _f(te),
        "information_ratio": _f(ir),
        "correlation": _f(corr, 4),
        "strategy_total_return": _f(float((1 + s_ret).prod() - 1.0)),
        "benchmark_total_return": _f(float((1 + b_ret).prod() - 1.0)),
    }


def extract_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    g = res.get
    return {
        "total_return": _f(g("total_return")),
        "annualized_return": _f(g("annualized_return")),
        "volatility": _f(g("volatility")),
        "sharpe_ratio": _f(g("sharpe_ratio")),
        "max_drawdown": _f(g("max_drawdown")),
        "win_rate": _f(g("win_rate")),
        "profit_factor": _f(g("profit_factor")),
        "total_trades": int(g("total_trades") or 0),
        "stocks_traded": int(g("stocks_traded") or 0),
        "final_value": _f(g("final_value")),
        "trading_days": int(g("trading_days") or 0),
    }


def _run_one(
    strategy: str,
    codes: List[str],
    benchmark: Dict[str, Any],
) -> Dict[str, Any]:
    executor = BacktestExecutor(
        data_dir=DATA_ROOT,
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    try:
        # 策略级口径覆盖（如 factor_ranking 的 TopK 等权口径），未命中用全局口径
        ov = STRATEGY_BACKTEST_OVERRIDES.get(strategy, {})
        res = asyncio.run(
            executor.run_backtest(
                strategy_name=strategy,
                stock_codes=codes,
                start_date=START,
                end_date=END,
                strategy_config={},
                backtest_config=BacktestConfig(
                    initial_cash=1_000_000.0,
                    commission_rate=0.0003,
                    slippage_rate=0.0001,
                    max_position_size=ov.get("max_position_size", 0.2),
                    cash_reserve_ratio=0.05,
                    board_lot_size=100,
                    stop_loss_pct=ov.get("stop_loss_pct", 0.05),
                    take_profit_pct=ov.get("take_profit_pct", 0.15),
                    rebalance_frequency="daily",
                    open_cost=0.0,
                    close_cost=0.0,
                    min_cost=0.0,
                ),
            )
        )
    except Exception as e:
        return {
            "strategy": strategy,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "wall_s": round(time.perf_counter() - t0, 1),
        }
    wall = time.perf_counter() - t0
    vs_bench = compare_to_benchmark(res, benchmark) if benchmark else None
    pb = res.get("perf_breakdown") or {}
    rec: Dict[str, Any] = {
        "strategy": strategy,
        "status": "ok",
        "start_date": START.isoformat(),
        "end_date": END.isoformat(),
        "universe_size": len(codes),
        "initial_cash": res.get("initial_cash"),
        "metrics": extract_metrics(res),
        "vs_benchmark": vs_bench,
        "perf": {k: _f(pb.get(k), 3) for k in PERF_KEYS if pb.get(k) is not None},
        "wall_s": round(wall, 1),
    }
    return conv(rec)


def load_out(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_out(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=["a", "b", "c", "all"], default="a")
    ap.add_argument("--strategies", nargs="*", default=None, help="覆盖默认批次列表")
    ap.add_argument("--smoke", action="store_true", help="20 只宇宙冒烟")
    ap.add_argument("--universe-size", type=int, default=500, help="统计套利批次宇宙大小")
    ap.add_argument("--rerun", action="store_true", help="不跳过已完成策略")
    ap.add_argument(
        "--out",
        default=os.path.join(OUT_DIR, "baseline_full_market_3y.json"),
    )
    ap.add_argument("--skip-benchmark", action="store_true")
    args = ap.parse_args()

    full = list_full_universe()
    print(f"[universe] qlib 全量 = {len(full)} 只", flush=True)

    if args.smoke:
        uni_full, uni_stat = full[:20], full[:20]
        strategies = args.strategies or BATCH_A
    elif args.strategies:
        uni_full, uni_stat = full, None
        strategies = args.strategies
    else:
        uni_full = full
        uni_stat = (
            pick_statarb_universe(
                full, args.universe_size,
                os.path.join(OUT_DIR, "baseline_universe_500.txt"),
            )
            if args.batch in ("c", "all")
            else None
        )
        if args.batch == "a":
            strategies = BATCH_A
        elif args.batch == "b":
            strategies = BATCH_B
        elif args.batch == "c":
            strategies = STATARB_BATCH_C
        else:
            strategies = BATCH_A + BATCH_B + STATARB_BATCH_C

    plan: List[tuple] = []
    for s in strategies:
        uni = uni_stat if (s in STATARB_BATCH_C and uni_stat is not None) else uni_full
        plan.append((s, uni))

    # ML 不可用记录（不跑，只登记原因）
    ml = ML_UNAVAILABLE
    if args.smoke or args.strategies:
        ml = {k: v for k, v in ml.items() if k in strategies}

    benchmark: Dict[str, Any] = {}
    if not args.skip_benchmark:
        print("[benchmark] 计算全市场等权基准 …", flush=True)
        closes = load_closes(full)
        benchmark = compute_equal_weight_benchmark(closes)
        del closes
        print(
            f"[benchmark] 等权: ret={benchmark['total_return']:.1%} "
            f"sharpe={benchmark['sharpe_ratio']:.2f} mdd={benchmark['max_drawdown']:.1%}",
            flush=True,
        )

    out = load_out(args.out)
    out.setdefault("meta", {}).update(
        {
            "generated_at": datetime.now().isoformat(),
            "window": [START.isoformat(), END.isoformat()],
            "universe_full": len(full),
            "universe_stat_arb": len(uni_stat) if uni_stat else 0,
            "initial_cash": 1_000_000,
            "fees": {
                "commission": 0.0003,
                "slippage": 0.0001,
                "max_position": 0.2,
                "cash_reserve": 0.05,
                "stop_loss": 0.05,
                "take_profit": 0.15,
                "rebalance": "daily",
            },
            "ml_unavailable": ml,
        }
    )
    out.setdefault("strategies", {})
    if benchmark:
        out["benchmark"] = benchmark

    pending = []
    for s, uni in plan:
        rec = out["strategies"].get(s, {})
        if rec.get("status") == "ok" and rec.get("vs_benchmark") and not args.rerun:
            print(f"[skip] {s} 已完成且带基准对比（{rec.get('wall_s')}s），保留", flush=True)
            continue
        if rec.get("status") == "ok" and not args.rerun:
            print(
                f"[rerun-implicit] {s} 已有结果但缺 vs_benchmark，重跑补基准",
                flush=True,
            )
        pending.append((s, uni))

    t_batch = time.perf_counter()
    for s, uni in pending:
        print(f"\n===== {s} (universe={len(uni)}) 开始 {datetime.now().strftime('%H:%M:%S')} =====", flush=True)
        rec = _run_one(s, uni, benchmark)
        out["strategies"][s] = rec
        save_out(args.out, out)
        m = rec.get("metrics", {})
        vb = rec.get("vs_benchmark") or {}
        print(
            f"===== {s} 完成: ret={m.get('total_return')} sharpe={m.get('sharpe_ratio')} "
            f"mdd={m.get('max_drawdown')} trades={m.get('total_trades')} "
            f"excess_ann={vb.get('excess_ann_return')} ir={vb.get('information_ratio')} "
            f"wall={rec.get('wall_s')}s =====",
            flush=True,
        )
    print(f"\n[batch] 本批耗时 {time.perf_counter() - t_batch:.0f}s", flush=True)

    # 排行榜
    rows = []
    for s, rec in out["strategies"].items():
        if rec.get("status") != "ok":
            continue
        m = rec.get("metrics", {})
        vb = rec.get("vs_benchmark") or {}
        rows.append(
            (s, m.get("total_return"), m.get("sharpe_ratio"),
             m.get("max_drawdown"), m.get("total_trades"),
             vb.get("excess_ann_return"), vb.get("information_ratio"))
        )
    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0)))
    b = out.get("benchmark") or {}
    print("\n===== 排行榜 (按总收益, 统一窗口/费率) =====", flush=True)
    print(
        f"{'strategy':<18} {'total_ret':>10} {'sharpe':>7} {'max_dd':>9} "
        f"{'trades':>7} {'excess_ann':>11} {'IR':>6}",
        flush=True,
    )
    for s, tr, sh, mdd, trd, ex, ir in rows:
        print(
            f"{s:<18} {tr if tr is not None else float('nan'):>10.2%} "
            f"{sh if sh is not None else float('nan'):>7.2f} "
            f"{mdd if mdd is not None else float('nan'):>9.2%} {trd:>7} "
            f"{ex if ex is not None else float('nan'):>11.2%} "
            f"{ir if ir is not None else float('nan'):>6.2f}",
            flush=True,
        )
    if b:
        print(
            f"{'[基准]等权全市场':<18} {b.get('total_return', float('nan')):>10.2%} "
            f"{b.get('sharpe_ratio', float('nan')):>7.2f} "
            f"{b.get('max_drawdown', float('nan')):>9.2%} {'-':>7} {'-':>11} {'-':>6}",
            flush=True,
        )
    for k, v in (out.get("meta", {}).get("ml_unavailable") or {}).items():
        print(f"[unavailable] {k}: {v}", flush=True)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
