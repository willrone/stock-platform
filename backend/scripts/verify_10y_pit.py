#!/usr/bin/env python3
"""10 年 PIT 验证：42 因子基线 vs 行业中性（+ 其余策略线基线重跑）

## 为什么做
1. **行业中性只验过 3 年**（2023-05 → 2026-05），必须延长窗口才能确认
   那 +0.32 夏普不是过拟合。
2. **其余 4 条线的回测基线还是旧口径**（涨停阈值修正只重跑了 42f），
   对外数字口径不一致。

## PIT 语义（关键）
用 `data/qlib_official_data`：
  - 日历 1990-12-19 → 2026-05-08（8637 个交易日）
  - `instruments/all.txt` 带每只股票的**上市日 / 退市日**
    （688665_sh 上市 2021-02-09；000004_sz 退市 2026-04-27）
  - 引擎按回测区间加载每只股票 → **区间内没有数据 = 当时已退市**
    → 自动排除，即 point-in-time 宇宙，不是"现在还活着的股票"

## 用法
  env -u PYTHONPATH backend/.venv-py313/bin/python3 \
    backend/scripts/verify_10y_pit.py --years 10
  ... --years 3 --only honest_composite
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

QLIB_OFFICIAL = os.path.join(ROOT, "data", "qlib_official_data")
QLIB_LIVE = os.path.join(ROOT, "data", "qlib_data")
OUT_DIR = os.path.join(ROOT, "data", "backtest")

# 各策略线的回测口径（与审计报告 FULL_STRATEGY_AUDIT_2026-10-25.md 一致）
STRATEGIES: Dict[str, Dict[str, Any]] = {
    "honest_composite": {
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4},
        "mps": 0.015833,
        "label": "42 因子等权",
    },
    "rsi_71421": {
        "cfg": {"topk": 30, "n_drop": 3, "rebalance_days": "weekly",
                "rebalance_weekday": 4},
        "mps": 0.031667,
        "label": "RSI 7/14/21 均值回归",
    },
    "factor_ranking": {
        "cfg": {"topk": 50, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4},
        "mps": 0.019,
        "label": "动量/低波动/低振幅",
    },
    "gbm_voltgt": {
        "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                "rebalance_weekday": 4, "n_estimators": 100, "max_depth": 3,
                "learning_rate": 0.05, "warmup_months": 2,
                "target_vol": 0.20, "vt_window": 20},
        "mps": 0.015833,
        "label": "LightGBM 元模型 + 波动率目标",
    },
    "multi_factor": {
        "cfg": {"topk": 30, "n_drop": 3},
        "mps": 0.031667,
        "label": "多因子（对照）",
    },
}


def universe_from_instruments(instruments_dir: str) -> List[str]:
    """从 qlib instruments/all.txt 读股票池（带上市/退市日 → PIT 语义）。"""
    path = os.path.join(instruments_dir, "instruments", "all.txt")
    codes = []
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
    # 兜底：扫 features 目录
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


def run_one(
    data_root: str,
    codes: List[str],
    strategy: str,
    cfg: Dict[str, Any],
    mps: float,
    start: datetime,
    end: datetime,
    lag: Optional[int] = None,
    enable_tradability: bool = True,
) -> Dict[str, Any]:
    executor = BacktestExecutor(
        data_dir=data_root,
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    try:
        res = asyncio.run(
            executor.run_backtest(
                strategy_name=strategy,
                stock_codes=codes,
                start_date=start,
                end_date=end,
                strategy_config=(
                    {**dict(cfg), "lag": lag} if lag is not None else dict(cfg)
                ),
                backtest_config=BacktestConfig(
                    enable_tradability=enable_tradability,
                    initial_cash=1_000_000.0,
                    commission_rate=0.0003,
                    slippage_rate=0.0001,
                    max_position_size=mps,
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
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"strategy": strategy, "error": f"{type(e).__name__}: {e}",
                "wall_s": round(time.perf_counter() - t0, 1)}

    out: Dict[str, Any] = {
        "strategy": strategy,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    _n_actual = (res.get("stock_count") or res.get("actual_stock_count")
                 or (res.get("perf_breakdown") or {}).get("loaded_stocks"))
    out["requested_stocks"] = len(codes)
    out["loaded_stocks"] = _n_actual
    if _n_actual is not None and len(codes) and (_n_actual / len(codes)) < 0.80:
        out["SAMPLE_TOO_SMALL"] = (
            f"实际加载 {_n_actual}/{len(codes)} = {_n_actual/len(codes)*100:.1f}% < 80%，"
            f"结论不可信")
    if "SAMPLE_TOO_SMALL" in out:
        print(f"  ❌ {out['SAMPLE_TOO_SMALL']}", flush=True)
    elif _n_actual:
        print(f"  [样本] 加载 {_n_actual}/{len(codes)} "
              f"({_n_actual/len(codes)*100:.1f}%)", flush=True)
    for k in ("total_return", "annual_return", "sharpe_ratio", "max_drawdown",
              "total_trades", "win_rate", "volatility", "final_value",
              "avg_positions", "calmar_ratio", "annualized_return"):
        if k in res:
            out[k] = res[k]
    # 分年收益：引擎的 additional_metrics 里有 yearly_returns_detail
    yearly = res.get("yearly_returns") or res.get("yearly") or {}
    if isinstance(yearly, dict) and yearly:
        out["yearly"] = {str(k): round(float(v), 6) for k, v in yearly.items()}
    yd = res.get("yearly_returns_detail")
    if isinstance(yd, list) and yd:
        out["yearly"] = {
            str(int(x["year"])): round(float(x["return"]), 6) for x in yd
        }
    md = res.get("monthly_returns_detail")
    if isinstance(md, list) and md:
        out["monthly"] = {
            str(x["month"]): round(float(x["return"]), 6) for x in md
        }
    pb = res.get("perf_breakdown") or {}
    if isinstance(pb, dict):
        out["perf_breakdown"] = {k: v for k, v in pb.items()
                                 if not isinstance(v, (list, dict))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--only", nargs="*", default=[],
                    help="只跑指定策略（空 = 全部）")
    ap.add_argument("--neutral", action="store_true",
                    help="额外跑一遍 industry_neutral=True 的对照")
    ap.add_argument("--out", default="verify_10y_pit.json")
    ap.add_argument("--lag", type=int, default=None,
                    help="信号滞后交易日数。0=当日收盘信号当日收盘成交（实盘做不到）；"
                         "1=收盘出信号、下一交易日收盘成交（可实现）")
    ap.add_argument("--no-tradability", action="store_true",
                    help="关闭回测可成交性检查（回到『能买涨停能卖跌停』的旧口径）")
    args = ap.parse_args()

    # 10 年：结束对齐 2026-05-08（官方数据的最后一天）
    end = datetime(2026, 5, 8)
    start = datetime(end.year - args.years, end.month, end.day)

    # 数据目录：特征仍用 data/qlib_data（features/day/*.parquet 实际覆盖 1991 起），
    # 股票池用 data/qlib_official_data/instruments/all.txt（带上市日/退市日 → PIT 语义）
    data_root = ROOT + "/data"
    codes = universe_from_instruments(QLIB_OFFICIAL)
    print(f"[verify] 窗口 {start.date()} → {end.date()}", flush=True)
    print(f"[verify] PIT 股票池（instruments/all.txt）: {len(codes)} 只", flush=True)
    print(
        "[verify] 注：instruments 里 2016 年后才上市的 2799 只在该窗口内无数据，"
        "引擎会自动排除 → point-in-time 宇宙",
        flush=True,
    )

    todo = args.only or list(STRATEGIES)
    results: Dict[str, Any] = {}

    for name in todo:
        spec = STRATEGIES.get(name)
        if spec is None:
            print(f"[verify] 跳过未知策略 {name}", flush=True)
            continue
        print(f"\n[verify] === {name} ({spec['label']}) ===", flush=True)
        r = run_one(data_root, codes, name, spec["cfg"], spec["mps"], start, end,
                    lag=args.lag, enable_tradability=not args.no_tradability)
        results[name] = r
        print(f"  基线: sharpe={r.get('sharpe_ratio')} ret={r.get('total_return')} "
              f"mdd={r.get('max_drawdown')} trades={r.get('total_trades')} "
              f"({r.get('wall_s')}s)", flush=True)

        if args.neutral and name == "honest_composite":
            cfg_neu = dict(spec["cfg"])
            cfg_neu["industry_neutral"] = True
            r2 = run_one(data_root, codes, name, cfg_neu, spec["mps"], start, end,
                         lag=args.lag, enable_tradability=not args.no_tradability)
            results["honest_composite_neutral"] = r2
            print(f"  行业中性: sharpe={r2.get('sharpe_ratio')} "
                  f"ret={r2.get('total_return')} mdd={r2.get('max_drawdown')} "
                  f"trades={r2.get('total_trades')} ({r2.get('wall_s')}s)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    # 口径标记（2026-10-27）：下游汇总脚本只接受真实盘口径的结果，
    # 避免旧口径（lag0 + 无可成交性）的数字混进排名。
    payload = {
        "window": f"{start.date()}→{end.date()}",
        "universe_size": len(codes),
        "methodology": {
            "lag": args.lag if args.lag is not None else 0,
            "enable_tradability": not args.no_tradability,
            "realistic": (args.lag == 1) and (not args.no_tradability),
            "note": "真实盘口径 = lag 1 + 可成交性检查开",
        },
        "results": results,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[verify] 已写入 {path}", flush=True)


if __name__ == "__main__":
    main()
