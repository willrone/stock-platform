#!/usr/bin/env python3
"""把策略基线结果注册进 app.db（backtest_results + backtest_benchmarks + tasks）。

背景: 基线 runner 只落 JSON; 平台 DB 的 backtest_results 表 0 行、
backtest_benchmarks 全是 test_backtest_001 假数据、benchmark_curve 恒空。
本脚本把基线 JSON 灌进三张表, 让"每个策略有回测基准数据"在平台层成立:

- backtest_results   : 每策略一行, 统一窗口/宇宙/费率, benchmark=等权序列对齐
- backtest_benchmarks: 真实回测 id 对应的等权基准对比(替代 test 假数据)
- tasks              : 登记一条 baseline 任务(可追溯)

幂等: 先按 source_tag 删旧 baseline 行再重插。
用法:
  cd backend && ./venv/bin/python scripts/register_strategy_baseline.py [--db data/app.db]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not os.path.isdir(BACKEND_ROOT):
    BACKEND_ROOT = "/Users/rone/projects/stock-platform/backend"
sys.path.insert(0, BACKEND_ROOT)

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
BASELINE_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "baseline_full_market_3y.json")
SOURCE_TAG = "baseline_full_market_3y"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(BACKEND_ROOT, "data", "app.db"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(BASELINE_JSON) as f:
        baseline = json.load(f)

    meta = baseline["meta"]
    benchmark = baseline.get("benchmark") or {}
    now = datetime.now().isoformat()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # 1) 清理旧的 baseline 行（幂等）
    cur.execute("DELETE FROM backtest_results WHERE backtest_id LIKE ?", (f"bt_{SOURCE_TAG}_%",))
    cur.execute("DELETE FROM backtest_benchmarks WHERE backtest_id LIKE ?", (f"bt_{SOURCE_TAG}_%",))
    cur.execute("DELETE FROM tasks WHERE task_id = ?", (f"baseline_{SOURCE_TAG}",))
    print(f"[cleanup] 旧 baseline 行已清: {cur.rowcount} tasks rows")

    bench_daily = benchmark.get("daily_value") or {}
    inserted = 0
    for sname, rec in baseline.get("strategies", {}).items():
        if rec.get("status") != "ok":
            continue
        m = rec.get("metrics", {})
        task_id = f"baseline_{SOURCE_TAG}"
        backtest_id = f"bt_{SOURCE_TAG}_{sname}"

        cur.execute(
            """INSERT OR REPLACE INTO backtest_results
               (task_id, backtest_id, strategy_name, start_date, end_date,
                initial_cash, final_value, total_return, annualized_return,
                volatility, sharpe_ratio, max_drawdown, win_rate, profit_factor,
                total_trades, trade_history, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                task_id,
                backtest_id,
                sname,
                rec.get("start_date") or meta["window"][0],
                rec.get("end_date") or meta["window"][1],
                float(rec.get("initial_cash") or 1_000_000),
                m.get("final_value") or 0.0,
                m.get("total_return") or 0.0,
                m.get("annualized_return") or 0.0,
                m.get("volatility") or 0.0,
                m.get("sharpe_ratio") or 0.0,
                m.get("max_drawdown") or 0.0,
                m.get("win_rate") or 0.0,
                m.get("profit_factor") or 0.0,  # 表 NOT NULL; 无盈利笔时记 0
                m.get("total_trades") or 0,
                None,  # trade_history 太大不落库; 基线以 JSON 为准
                now,
            ),
        )

        # benchmark 对比行（真实等权序列; 策略侧指标取 vs_benchmark）
        vb = rec.get("vs_benchmark") or {}
        bench_metrics = json.dumps(
            {
                "total_return": benchmark.get("total_return"),
                "annualized_return": benchmark.get("annualized_return"),
                "sharpe_ratio": benchmark.get("sharpe_ratio"),
                "max_drawdown": benchmark.get("max_drawdown"),
                "n_stocks": benchmark.get("n_stocks"),
                "n_days": benchmark.get("n_days"),
            },
            ensure_ascii=False,
        )
        cur.execute(
            """INSERT OR REPLACE INTO backtest_benchmarks
               (task_id, backtest_id, benchmark_symbol, benchmark_name, benchmark_data,
                correlation, beta, alpha, tracking_error, information_ratio,
                excess_return, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                task_id,
                backtest_id,
                benchmark.get("symbol", "EQUI-WEIGHT-FULL-MKT"),
                benchmark.get("name", "全市场等权买入持有"),
                # 基准序列只存首尾 + 聚合, 全序列在基线 JSON (避免 DB 膨胀)
                json.dumps(
                    {
                        "note": "full daily series in data/strategy_baseline/baseline_full_market_3y.json",
                        "first": bench_daily.get(sorted(bench_daily)[0]) if bench_daily else None,
                        "last": bench_daily.get(sorted(bench_daily)[-1]) if bench_daily else None,
                        "aggregates": json.loads(bench_metrics),
                    },
                    ensure_ascii=False,
                ),
                vb.get("correlation"),
                None,
                vb.get("excess_ann_return"),
                vb.get("tracking_error"),
                vb.get("information_ratio"),
                vb.get("excess_ann_return"),
                now,
                now,
            ),
        )
        inserted += 1

    # 2) 登记 baseline 任务(追溯用)
    n_ok = sum(1 for r in baseline.get("strategies", {}).values() if r.get("status") == "ok")
    cur.execute(
        """INSERT OR REPLACE INTO tasks
           (task_id, task_name, task_type, status, user_id, config, created_at,
            started_at, completed_at, progress, result, error_message, estimated_duration, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            f"baseline_{SOURCE_TAG}",
            "全策略统一基线回测（全市场×3年）",
            "backtest",
            "completed",
            None,
            json.dumps(
                {
                    "strategy_name": None,
                    "baseline": SOURCE_TAG,
                    "window": meta["window"],
                    "universe_full": meta["universe_full"],
                    "strategies_ok": n_ok,
                    "benchmark": benchmark.get("symbol"),
                    "json": BASELINE_JSON,
                },
                ensure_ascii=False,
            ),
            now,
            now,
            now,
            100.0,
            json.dumps(
                {
                    "strategies": {
                        s: (r.get("metrics") or {})
                        for s, r in baseline.get("strategies", {}).items()
                    },
                    "benchmark": benchmark,
                },
                ensure_ascii=False,
            ),
            None,
            None,
            now,
        ),
    )

    conn.commit()
    if not args.dry_run:
        pass
    # 验证
    n_res = cur.execute("SELECT COUNT(*) FROM backtest_results WHERE backtest_id LIKE ?", (f"bt_{SOURCE_TAG}_%",)).fetchone()[0]
    n_bench = cur.execute("SELECT COUNT(*) FROM backtest_benchmarks WHERE backtest_id LIKE ?", (f"bt_{SOURCE_TAG}_%",)).fetchone()[0]
    print(f"[done] backtest_results={n_res} backtest_benchmarks={n_bench} (期望 {inserted})")
    for row in cur.execute(
        "SELECT strategy_name, total_return, sharpe_ratio, max_drawdown FROM backtest_results WHERE backtest_id LIKE ? ORDER BY sharpe_ratio DESC",
        (f"bt_{SOURCE_TAG}_%",),
    ).fetchall():
        print(f"  {row[0]:<16} ret={row[1]} sharpe={row[2]} mdd={row[3]}")
    conn.close()


if __name__ == "__main__":
    main()
