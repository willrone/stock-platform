#!/usr/bin/env python3
"""TopK 策略「日频 vs 周频」对照（统一口径, 3 年全市场）。

为什么必须比:
  记忆库实证: 换手是赛马的决定性差异。日频线 2354-2471 笔 vs 周频线 306 笔,
  差 8 倍; 多出来的交易全要付佣金+印花税+滑点, 把因子优势吃光。
  引擎不消费 BacktestConfig.rebalance_frequency → 周频靠策略侧分数掩码实现
  (rebalance_days="weekly", 见 honest/rsi_71421/gbm_voltgt/factor_ranking)。

本脚本对每条 TopK 策略跑 daily / weekly 两格, 唯一变量是调仓频率,
其余(窗口/宇宙/费率/单仓/执行模型)全同。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
      backend/scripts/strategy_sweep.py --weekly-compare
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

try:
    from loguru import logger as _logger

    _logger.remove()
    _logger.add(sys.stderr, level=os.environ.get("BASELINE_LOG_LEVEL", "ERROR"))
except Exception:
    pass

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

QLIB_DAY = os.path.join(ROOT, "data", "qlib_data", "features", "day")
OUT_DIR = os.path.join(ROOT, "data", "strategy_baseline")

# TopK 类策略及其参数（与 seed_strategy_configs.py 的 benchmark 预设一致）
TOPK_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "honest_composite": {"topk": 60, "n_drop": 5},
    "rsi_71421": {"topk": 30, "n_drop": 3},
    "factor_ranking": {"topk": 50, "n_drop": 5},
    "gbm_voltgt": {"topk": 60, "n_drop": 5, "horizon": 1},
    "low_volatility": {"topk": 30, "n_drop": 3},
    "multi_factor": {"topk": 30, "n_drop": 3},
}


def list_full_universe() -> List[str]:
    files = sorted(
        f[: -len(".parquet")] for f in os.listdir(QLIB_DAY) if f.endswith(".parquet")
    )
    return sorted(c.replace("_", ".") for c in files if c)


def _f(x: Any, nd: int = 6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, nd)


async def run_one(strategy: str, sc: Dict[str, Any], codes, start, end, capital):
    ex = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    try:
        res = await ex.run_backtest(
            strategy_name=strategy, stock_codes=codes,
            start_date=start, end_date=end, strategy_config=dict(sc),
            backtest_config=BacktestConfig(
                initial_cash=capital,
                commission_rate=0.0003, slippage_rate=0.0001,
                cash_reserve_ratio=0.05, board_lot_size=100,
                stop_loss_pct=0.0, take_profit_pct=0.0,
                rebalance_frequency="daily",
            ),
        )
    except Exception as e:
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"[:400],
                "wall_s": round(time.perf_counter() - t0, 1)}
    g = res.get
    hist = res.get("portfolio_history") or []
    return {
        "status": "ok",
        "total_return": _f(g("total_return")),
        "annualized_return": _f(g("annualized_return")),
        "sharpe_ratio": _f(g("sharpe_ratio")),
        "max_drawdown": _f(g("max_drawdown")),
        "total_trades": int(g("total_trades") or 0),
        "final_positions": len((hist[-1].get("positions") or {})) if hist else 0,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-05-12")
    ap.add_argument("--end", default="2026-05-08")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--strategies", nargs="*", default=list(TOPK_STRATEGIES))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "frequency_compare_3y.json"))
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    codes = list_full_universe()
    out: Dict[str, Any] = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            try:
                out = json.load(fh)
            except Exception:
                out = {}
    out.setdefault("meta", {}).update({
        "window": [args.start, args.end],
        "universe": len(codes),
        "capital": args.capital,
        "fees": {"commission": 0.0003, "slippage": 0.0001, "cash_reserve": 0.05},
        "generated_at": datetime.now().isoformat(),
    })
    out.setdefault("cells", {})

    for s in args.strategies:
        base = TOPK_STRATEGIES[s]
        for freq in ("daily", "weekly"):
            key = f"{s}_{freq}"
            if out["cells"].get(key, {}).get("status") == "ok":
                print(f"[skip] {key}", flush=True)
                continue
            sc = dict(base)
            sc["rebalance_days"] = freq
            print(f"\n===== {key} {datetime.now().strftime('%H:%M:%S')} =====", flush=True)
            rec = await_run = asyncio.run(run_one(s, sc, codes, start, end, args.capital))
            rec["strategy"] = s
            rec["freq"] = freq
            rec["strategy_config"] = sc
            out["cells"][key] = rec
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            tmp = args.out + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, args.out)
            print(f"===== {key}: ret={rec.get('total_return')} "
                  f"sharpe={rec.get('sharpe_ratio')} mdd={rec.get('max_drawdown')} "
                  f"trades={rec.get('total_trades')} status={rec.get('status')} "
                  f"wall={rec.get('wall_s')}s =====", flush=True)

    print("\n===== 日频 vs 周频 =====", flush=True)
    print(f"{'strategy':<20}{'freq':<9}{'ret':>10}{'sharpe':>9}{'mdd':>9}{'trades':>8}")
    for s in args.strategies:
        for freq in ("daily", "weekly"):
            r = out["cells"].get(f"{s}_{freq}") or {}
            if r.get("status") != "ok":
                print(f"{s:<20}{freq:<9}  FAILED: {r.get('error','')[:60]}")
                continue
            print(f"{s:<20}{freq:<9}{(r.get('total_return') or 0):>10.2%}"
                  f"{(r.get('sharpe_ratio') or 0):>9.4f}"
                  f"{(r.get('max_drawdown') or 0):>9.2%}"
                  f"{r.get('total_trades') or 0:>8}")
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
