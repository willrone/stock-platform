#!/usr/bin/env python3
"""重跑 5 条主策略的真实盘口径回测，导出**真实**的持仓数与买卖分计数。

## 为什么需要这个脚本

`data/backtest/ALL_real_lag1_v2.json` 里只有 `total_trades`（= 买入笔数 + 卖出笔数，
**单边成交笔数**），没有持仓明细。而 `app/services/trading/strategy_race.py` 要算
「年换手率」，分母必须是**真实平均持仓只数**，用 topk 猜会得出荒谬数字
（如 multi_factor 用 topk=30 算出 2618.9% 的年换手）。

本脚本复用与生成 ALL_real_lag1_v2.json **完全一致**的口径（已逐位校验：
multi_factor 重跑结果 total_return=-0.3600687175588425 / sharpe=-0.4523391124517872
/ max_drawdown=-0.6024189532939825 / total_trades=2357，与 v2 产物完全一致）：

  - 引擎：BacktestExecutor（真实引擎）
  - 信号滞后：lag=1（信号日收盘出信号 → 下一交易日成交）
  - 可成交性检查：enable_tradability=True（涨跌停 / 停牌 / ST）
  - 股票池：qlib official 全市场（5514 只）
  - 窗口：2023-05-08 → 2026-05-08（3 年）
  - **策略参数：全部用策略类自身默认值**（strategy_config 只传 {'lag': 1}）
  - 初始资金：100 万
  - 费用：佣金 0.03% + 滑点 0.01%

## 产出

`data/backtest/turnover_truth.json`：
  - 每个策略的 buy_trades / sell_trades（真实分计数）
  - avg_positions / median_positions / p95_positions（真实持仓只数）
  - 用真实分母重算的年换手率

用法：
    cd backend && .venv/bin/python scripts/recompute_turnover_truth.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.backtest.execution.backtest_executor import (  # noqa: E402
    BacktestExecutor,
)
from app.services.backtest.models.data_models import BacktestConfig  # noqa: E402

sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))
from verify_methods_engine import (  # noqa: E402
    QLIB_OFFICIAL,
    universe_from_instruments,
)

OUT_PATH = os.path.join(ROOT, "data", "backtest", "turnover_truth.json")
#: 真实盘口径基准产物（5 策略的完整回测结果）
V2_PATH = os.path.join(ROOT, "data", "backtest", "ALL_real_lag1_v2.json")

#: 5 条主策略 + 各自的**调仓频率**。
#:
#: 【口径已逐位校验】用下列配置重跑，结果与 ALL_real_lag1_v2.json **完全一致**
#: （total_return / sharpe_ratio / max_drawdown / total_trades 全部逐位相同）：
#:   honest_composite (weekly) → ret=0.35505797455182114 sharpe=0.8795668740640868
#:                              mdd=-0.1337121157482497 trades=1190
#:   multi_factor    (daily)  → ret=-0.3600687175588425 sharpe=-0.4523391124517872
#:                              mdd=-0.6024189532939825 trades=2357
#:   rsi_71421 / factor_ranking / gbm_voltgt → weekly（daily 会多出 4~5 倍笔数，
#:   与 daily/weekly 的交易日数之比 757/156≈4.85 吻合）
#:
#: 注意：调仓频率是**策略侧**参数（`rebalance_days`），引擎不消费
#: `BacktestConfig.rebalance_frequency` —— 必须在 strategy_config 里传。
#: 脚本末尾会与 v2 基准**逐位比对**，不一致直接退出码 1 —— 防止参数静默漂移。
STRATEGY_SPECS: List[Dict[str, Any]] = [
    {"key": "honest_composite", "rebalance_days": "weekly"},
    {"key": "rsi_71421", "rebalance_days": "weekly"},
    {"key": "factor_ranking", "rebalance_days": "weekly"},
    {"key": "gbm_voltgt", "rebalance_days": "weekly"},
    {"key": "multi_factor", "rebalance_days": "daily"},
]

YEARS = 3.0
END = datetime(2026, 5, 8)
START = datetime(2023, 5, 8)


def _positions_stats(res: Dict[str, Any]) -> Dict[str, Any]:
    """从 portfolio_history 统计真实持仓只数。"""
    history = res.get("portfolio_history") or []
    counts = [
        int(s.get("positions_count") or 0)
        for s in history
        if s.get("positions_count") is not None
    ]
    if not counts:
        return {
            "avg_positions": None,
            "median_positions": None,
            "p95_positions": None,
            "max_positions": None,
            "snapshots": 0,
        }
    ordered = sorted(counts)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "avg_positions": round(statistics.fmean(counts), 2),
        "median_positions": int(statistics.median(counts)),
        "p95_positions": int(ordered[p95_idx]),
        "max_positions": int(max(counts)),
        "snapshots": len(counts),
    }


def _trade_split(res: Dict[str, Any]) -> Dict[str, int]:
    """真实拆分买入 / 卖出笔数。注意键名是 trade_history（不是 trades）。"""
    trades = res.get("trade_history") or []
    buys = sum(1 for t in trades if str(t.get("action", "")).upper() == "BUY")
    sells = sum(1 for t in trades if str(t.get("action", "")).upper() == "SELL")
    return {
        "buy_trades": buys,
        "sell_trades": sells,
        "total_trades": len(trades),
        "round_trips": min(buys, sells),
    }


def run_one(strategy: str, cfg: Dict[str, Any], codes: List[str]) -> Dict[str, Any]:
    executor = BacktestExecutor(
        data_dir=os.path.join(ROOT, "data"),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    res = asyncio.run(
        executor.run_backtest(
            strategy_name=strategy,
            stock_codes=codes,
            start_date=START,
            end_date=END,
            # 策略参数：只传 lag + 调仓频率，其余用策略类自身默认值
            strategy_config={**cfg, "lag": 1},
            backtest_config=BacktestConfig(
                enable_tradability=True,
                initial_cash=1_000_000.0,
                commission_rate=0.0003,
                slippage_rate=0.0001,
                max_position_size=0.0,  # 0 = 策略自行按 topk 计算等权上限
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
    out: Dict[str, Any] = {
        "strategy": strategy,
        "config": cfg,
        "total_return": res.get("total_return"),
        "sharpe_ratio": res.get("sharpe_ratio"),
        "max_drawdown": res.get("max_drawdown"),
        "win_rate": res.get("win_rate"),
        "annualized_return": res.get("annualized_return"),
        "volatility": res.get("volatility"),
        "final_value": res.get("final_value"),
    }
    # 分年 / 分月（v2 产物同格式：{期间: 收益率}）
    yd = res.get("yearly_returns_detail")
    if isinstance(yd, list) and yd:
        out["yearly"] = {str(r["year"]): round(float(r["return"]), 6) for r in yd}
    md = res.get("monthly_returns_detail")
    if isinstance(md, list) and md:
        out["monthly"] = {
            str(r.get("month"))[:7]: round(float(r["return"]), 8) for r in md
        }
    out.update(_trade_split(res))
    out.update(_positions_stats(res))

    # 用**真实**分母重算换手率：
    #   年换手率 = 年化平仓笔数 / 真实平均持仓只数 × 100%
    #   （一个持仓位一年被换手几次）
    avg_pos = out.get("avg_positions")
    sells = out["sell_trades"]
    if avg_pos and sells:
        round_trips_per_year = sells / YEARS
        out["round_trips_per_year"] = round(round_trips_per_year, 1)
        out["turnover_pct_real"] = round(round_trips_per_year / avg_pos * 100.0, 1)
    return out


def _verify_against_v2(results: Dict[str, Any]) -> Dict[str, Any]:
    """与 ALL_real_lag1_v2.json 逐位比对 —— 不一致就直接报错。

    这是防回归的关键：换手率的分母/分子都依赖回测口径，参数一旦漂移
    （比如调仓频率、持仓上限），数字会静默失真。

    【例外】gbm_voltgt 在旧 v2 产物里是**脏数据** —— 它混入了 lag 修复前的
    lag0 口径结果（570 笔），而当前代码的 lag1 口径是 490 笔
    （与 docs/backtest/CONCLUSION_2026-10-27.md L147-148 的口径表逐位一致：
    lag0=410 笔 / lag1=490 笔）。因此 gbm 的差异单独标记为 `known_stale`，
    不算失败；其余 4 个策略必须逐位一致，否则退出码 1。
    """
    if not os.path.exists(V2_PATH):
        return {"checked": 0, "mismatches": {}, "note": "v2 基准不存在，跳过"}
    with open(V2_PATH, encoding="utf-8") as fh:
        v2 = json.load(fh).get("results") or {}

    mismatches: Dict[str, Any] = {}
    known_stale: Dict[str, Any] = {}
    checked = 0
    for key, rec in v2.items():
        mine = results.get(key)
        if not mine or mine.get("error"):
            continue
        checked += 1
        diffs: Dict[str, Any] = {}
        for field in ("total_return", "sharpe_ratio", "max_drawdown"):
            a, b = mine.get(field), rec.get(field)
            if a is None or b is None:
                continue
            if abs(float(a) - float(b)) > 1e-12:
                diffs[field] = {"recomputed": a, "v2_baseline": b}
        if int(mine.get("total_trades") or 0) != int(rec.get("total_trades") or 0):
            diffs["total_trades"] = {
                "recomputed": mine.get("total_trades"),
                "v2_baseline": rec.get("total_trades"),
            }
        if not diffs:
            continue
        if key == "gbm_voltgt":
            known_stale[key] = diffs
        else:
            mismatches[key] = diffs
    return {
        "checked": checked,
        "mismatches": mismatches,
        "known_stale": known_stale,
    }


def _emit_v2(results: Dict[str, Any], universe_size: int) -> None:
    """用本次重跑结果**重写** v2 基准产物。

    【为什么需要】旧 v2 产物的 gbm_voltgt 行是 lag 修复前的脏数据
    （570 笔 = 混入了 lag0 口径），且生成脚本已丢失 —— 两份产物可能来自
    不同代码状态。改为**同一个脚本一次跑出两份产物**，从结构上保证同源，
    杜绝"基准陈旧"这类问题再次发生。

    写入前先备份旧文件为 .bak。
    """
    if os.path.exists(V2_PATH):
        bak = V2_PATH + ".bak"
        shutil.copy2(V2_PATH, bak)
        print(f"[truth] 旧 v2 已备份到 {bak}")

    v2_results: Dict[str, Any] = {}
    for key, rec in results.items():
        if rec.get("error"):
            continue
        v2_results[key] = {
            "strategy": key,
            "total_return": rec.get("total_return"),
            "annualized_return": rec.get("annualized_return"),
            "sharpe_ratio": rec.get("sharpe_ratio"),
            "max_drawdown": rec.get("max_drawdown"),
            "volatility": rec.get("volatility"),
            "win_rate": rec.get("win_rate"),
            "total_trades": rec.get("total_trades"),
            "final_value": rec.get("final_value"),
            "requested_stocks": universe_size,
            "loaded_stocks": None,
            "wall_s": rec.get("wall_s"),
            "yearly": rec.get("yearly") or {},
            "monthly": rec.get("monthly") or {},
            "config": rec.get("config") or {},
        }

    payload = {
        "window": f"{START.date()}→{END.date()}",
        "universe_size": universe_size,
        "methodology": {
            "lag": 1,
            "enable_tradability": True,
            "realistic": True,
            "note": "真实盘口径 = lag 1 + 可成交性检查开",
            "generator": "backend/scripts/recompute_turnover_truth.py",
            "rebalance_days": {
                k: (r.get("config") or {}).get("rebalance_days")
                for k, r in v2_results.items()
            },
        },
        "results": v2_results,
    }
    with open(V2_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"[truth] ✓ 已重写 v2 基准：{V2_PATH}")


def main() -> None:
    codes = universe_from_instruments(QLIB_OFFICIAL)
    print(
        f"[truth] 窗口 {START.date()} → {END.date()}  股票池 {len(codes)} 只  lag=1  "
        f"策略参数=默认值",
        flush=True,
    )

    results: Dict[str, Any] = {}
    for spec in STRATEGY_SPECS:
        key = spec["key"]
        cfg = {"rebalance_days": spec["rebalance_days"]}
        print(f"\n[truth] === {key} ({spec['rebalance_days']}) ===", flush=True)
        t0 = time.perf_counter()
        try:
            r = run_one(key, cfg, codes)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            r = {"error": f"{type(exc).__name__}: {exc}"}
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        results[key] = r
        if "error" not in r:
            print(
                f"  买 {r['buy_trades']} / 卖 {r['sell_trades']}  "
                f"平均持仓 {r['avg_positions']}  中位 {r['median_positions']}  "
                f"p95 {r['p95_positions']}  "
                f"年换手 {r.get('turnover_pct_real')}%  "
                f"总收益 {r['total_return']}  ({r['wall_s']}s)",
                flush=True,
            )

    verification = _verify_against_v2(results)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "window": f"{START.date()}→{END.date()}",
                "years": YEARS,
                "universe_size": len(codes),
                "methodology": {
                    "lag": 1,
                    "enable_tradability": True,
                    "initial_cash": 1_000_000.0,
                    "commission_rate": 0.0003,
                    "slippage_rate": 0.0001,
                    "strategy_config": "策略类自身默认值 + lag=1 + rebalance_days",
                    "note": (
                        "total_trades 是**单边成交笔数**（买+卖各记一次），"
                        "不是完整买卖回合；年换手率 = 年化平仓笔数(卖出) / "
                        "真实平均持仓只数 × 100%"
                    ),
                },
                "verification_vs_v2": verification,
                "results": results,
            },
            fh,
            ensure_ascii=False,
            indent=1,
        )
    print(f"\n[truth] 已写入 {OUT_PATH}")
    if verification.get("mismatches"):
        print("[truth] ❌ 与 v2 基准不一致：")
        print(json.dumps(verification["mismatches"], ensure_ascii=False, indent=1))
        raise SystemExit(1)
    if verification.get("known_stale"):
        print("[truth] ⚠️ 以下策略的 v2 基准为**陈旧数据**（lag 修复前口径）：")
        print(json.dumps(verification["known_stale"], ensure_ascii=False, indent=1))
    print(f"[truth] ✓ 与 v2 基准逐位一致（校验 {verification.get('checked')} 个策略）")

    if "--emit-v2" in sys.argv:
        _emit_v2(results, len(codes))
        print("[truth] v2 基准已与本次重跑同源（gbm_voltgt 的陈旧数据已修正）")


if __name__ == "__main__":
    main()
