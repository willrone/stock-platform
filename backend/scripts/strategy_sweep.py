#!/usr/bin/env python3
"""全策略统一口径复验 runner（2026-10-25）。

目的: 把平台上每一条注册策略在**同一数据、同一窗口、同一费率、同一单仓口径**
下跑一遍，产出可比的收益/夏普/回撤/交易数/持仓数，用来回答
"每条策略的代码是否正常、实际回测数据如何、夏普多少"。

口径:
- 窗口   : --start → --end（默认 2023-05-12 → 2026-05-08，qlib 数据最近 3 年）
- 宇宙   : qlib 全量（默认 5513 只）；--universe-size 可限
- 资金   : 1,000,000
- 费率   : commission 0.0003 / slippage 0.0001 / cash_reserve 0.05 / board_lot 100
- 单仓   : **不传** max_position_size → 走策略自带的等权上限（TopK 类）；
           非 TopK 策略用引擎默认 0.2
- 调仓   : daily
- 止盈止损: 传 0（引擎零消费，传非 0 是误导）

每条策略跑完立即写 JSON（增量落盘），后台进程崩了也保留已完成部分。
ML 类 4 条若加载 pkl 段错误 → 记为 unavailable 并附原因。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
      backend/scripts/strategy_sweep.py --strategies honest_composite rsi_71421
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

# 引擎内部会再次 add sink（模块级默认 handler），这里在 import 引擎前
# 把 loguru 的全局 handler 压到 ERROR，避免上万行调试日志淹没输出。
try:
    from loguru import logger as _logger

    _logger.remove()
    _logger.add(sys.stderr, level=os.environ.get("BASELINE_LOG_LEVEL", "ERROR"))
    _logger.configure(extra={"_quiet": True})
except Exception:
    pass

from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

QLIB_DAY = os.path.join(ROOT, "data", "qlib_data", "features", "day")
OUT_DIR = os.path.join(ROOT, "data", "strategy_baseline")

# 全部注册策略（strategy_factory 的键），别名已展开到其代表实现
ALL_STRATEGIES = [
    # 技术分析
    "moving_average", "rsi", "macd", "bollinger", "stochastic", "cci", "kdj", "obv",
    # 统计套利（O(N^2) 配对，需限宇宙）
    "pairs_trading", "mean_reversion", "cointegration",
    # 因子
    "value_factor", "momentum_factor", "low_volatility", "multi_factor",
    # 横截面 / ML
    "factor_ranking", "honest_composite", "rsi_71421", "gbm_voltgt",
    "ml_ensemble_lgb_xgb_riskctl", "lightgbm", "model_signal", "model_topk_dropout",
]

STATARB = {"pairs_trading", "mean_reversion", "cointegration"}

# 策略默认参数（与 seed_strategy_configs.py 的 benchmark 预设对齐）
STRATEGY_CONFIG: Dict[str, Dict[str, Any]] = {
    "factor_ranking": {"topk": 50, "n_drop": 5},
    "honest_composite": {"topk": 60, "n_drop": 5},
    "rsi_71421": {"topk": 30, "n_drop": 3},
    "gbm_voltgt": {"topk": 60, "n_drop": 5, "horizon": 1},
    "low_volatility": {"topk": 30, "n_drop": 3},
    "multi_factor": {"topk": 30, "n_drop": 3},
    # ML 类需要 model_id（official-baseline 模型，numpy 2.x 兼容加载器已修）
    "lightgbm": {"model_id": "official-baseline-alpha158-b08dac04"},
    "model_signal": {"model_id": "official-baseline-alpha158-b08dac04"},
    "model_topk_dropout": {"model_id": "official-baseline-alpha158-b08dac04",
                           "topk": 50, "n_drop": 5},
}


def list_full_universe() -> List[str]:
    files = sorted(
        f[: -len(".parquet")] for f in os.listdir(QLIB_DAY) if f.endswith(".parquet")
    )
    return sorted(c.replace("_", ".") for c in files if c)


def _f(x: Any, nd: int = 6) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, nd)


def _sharpe_from_curve(portfolio_history: List[Dict[str, Any]]) -> Optional[float]:
    """从逐日组合净值算标准夏普 μ/σ·√252（引擎口径一致，缺 portfolio_value 时返回 None）。"""
    import numpy as np

    vals = []
    for h in portfolio_history or []:
        v = h.get("portfolio_value")
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv and fv == fv:
            vals.append(fv)
    if len(vals) < 3:
        return None
    s = np.array(vals, dtype=float)
    r = s[1:] / s[:-1] - 1.0
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return None
    return float(r.mean() / sd * np.sqrt(252.0))


async def run_one(
    strategy: str,
    codes: List[str],
    start: datetime,
    end: datetime,
    capital: float,
    strategy_config: Dict[str, Any],
) -> Dict[str, Any]:
    executor = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    try:
        res = await executor.run_backtest(
            strategy_name=strategy,
            stock_codes=codes,
            start_date=start,
            end_date=end,
            strategy_config=dict(strategy_config),
            backtest_config=BacktestConfig(
                initial_cash=capital,
                commission_rate=0.0003,
                slippage_rate=0.0001,
                cash_reserve_ratio=0.05,
                board_lot_size=100,
                stop_loss_pct=0.0,   # 引擎零消费，传 0 避免误导
                take_profit_pct=0.0,
                rebalance_frequency="daily",
            ),
        )
    except Exception as e:
        return {
            "strategy": strategy,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}"[:500],
            "wall_s": round(time.perf_counter() - t0, 1),
        }
    wall = time.perf_counter() - t0
    g = res.get
    hist = res.get("portfolio_history") or []
    last = hist[-1] if hist else {}
    positions = last.get("positions") or {}
    return {
        "strategy": strategy,
        "status": "ok",
        "strategy_config": strategy_config,
        "universe_size": len(codes),
        "total_return": _f(g("total_return")),
        "annualized_return": _f(g("annualized_return")),
        "volatility": _f(g("volatility")),
        "sharpe_ratio": _f(g("sharpe_ratio")),
        "sharpe_recomputed": _f(_sharpe_from_curve(hist)),
        "max_drawdown": _f(g("max_drawdown")),
        "win_rate": _f(g("win_rate")),
        "profit_factor": _f(g("profit_factor")),
        "total_trades": int(g("total_trades") or 0),
        "stocks_traded": int(g("stocks_traded") or 0),
        "final_positions": len(positions),
        "final_value": _f(g("final_value")),
        "trading_days": int(g("trading_days") or 0),
        "max_position_size": _f(getattr(g("config"), "max_position_size", None), 8)
        if isinstance(g("config"), object) else None,
        "wall_s": round(wall, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--start", default="2023-05-12")
    ap.add_argument("--end", default="2026-05-08")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--universe-size", type=int, default=0, help="0=全量")
    ap.add_argument("--statarb-universe-size", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "strategy_sweep_3y.json"))
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    full = list_full_universe()
    statarb_uni = sorted(full)[-args.statarb_universe_size:]

    strategies = args.strategies or ALL_STRATEGIES
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out: Dict[str, Any] = {}
    if os.path.exists(args.out) and not args.rerun:
        with open(args.out) as fh:
            try:
                out = json.load(fh)
            except Exception:
                out = {}
    out.setdefault("meta", {}).update({
        "generated_at": datetime.now().isoformat(),
        "window": [args.start, args.end],
        "universe_full": len(full),
        "capital": args.capital,
        "fees": {"commission": 0.0003, "slippage": 0.0001, "cash_reserve": 0.05,
                 "board_lot": 100, "rebalance": "daily", "stoploss": "引擎不消费"},
        "mps_policy": "TopK 类走策略等权上限 (1-cash_reserve)/topk；其余引擎默认 0.2",
    })
    out.setdefault("strategies", {})

    for s in strategies:
        if out["strategies"].get(s, {}).get("status") == "ok" and not args.rerun:
            print(f"[skip] {s} 已完成", flush=True)
            continue
        uni = statarb_uni if s in STATARB else full
        if args.universe_size:
            uni = uni[-args.universe_size:]
        print(f"\n===== {s} (uni={len(uni)}) {datetime.now().strftime('%H:%M:%S')} =====", flush=True)
        rec = asyncio.run(run_one(
            s, uni, start, end, args.capital, STRATEGY_CONFIG.get(s, {})
        ))
        out["strategies"][s] = rec
        tmp = args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, args.out)
        print(
            f"===== {s}: ret={rec.get('total_return')} sharpe={rec.get('sharpe_ratio')} "
            f"mdd={rec.get('max_drawdown')} trades={rec.get('total_trades')} "
            f"pos={rec.get('final_positions')} status={rec.get('status')} "
            f"wall={rec.get('wall_s')}s =====",
            flush=True,
        )

    print("\n===== 排行榜 =====", flush=True)
    rows = [(s, r) for s, r in out["strategies"].items() if r.get("status") == "ok"]
    rows.sort(key=lambda x: -(x[1].get("sharpe_ratio") or -9))
    print(f"{'strategy':<28}{'ret':>10}{'sharpe':>8}{'mdd':>9}{'trades':>8}{'pos':>6}")
    for s, r in rows:
        print(
            f"{s:<28}{(r.get('total_return') or 0):>10.2%}"
            f"{(r.get('sharpe_ratio') or 0):>8.3f}{(r.get('max_drawdown') or 0):>9.2%}"
            f"{r.get('total_trades') or 0:>8}{r.get('final_positions') or 0:>6}"
        )
    for s, r in out["strategies"].items():
        if r.get("status") != "ok":
            print(f"[unavailable] {s}: {r.get('error')}", flush=True)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
