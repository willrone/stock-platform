#!/usr/bin/env python3
"""
Minimal diagnostic script - uses only stdlib sqlite3 and json.
Run: python3 scripts/diagnose_minimal.py be715f16-c3f1-43e0-acb7-d6ec0bfd4ef6
"""
import json
import sqlite3
import sys
from pathlib import Path

# Resolve database path (same logic as config)
BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_REL = BACKEND_DIR / "data" / "app.db"
DB_PATH = DB_REL.resolve() if DB_REL.exists() else DB_REL

def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else "be715f16-c3f1-43e0-acb7-d6ec0bfd4ef6"
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 1. Check task result
    cur = conn.execute(
        "SELECT task_id, status, result FROM tasks WHERE task_id = ?",
        (task_id,)
    )
    row = cur.fetchone()
    if not row:
        print(f"Task {task_id} not found")
        conn.close()
        sys.exit(1)

    print(f"Task: {row['task_id']}, status: {row['status']}")
    result_raw = row["result"]
    if not result_raw:
        print("task.result is NULL/empty")
        conn.close()
        sys.exit(0)

    try:
        result = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
    except json.JSONDecodeError:
        print("task.result JSON parse failed")
        conn.close()
        sys.exit(1)

    total_trades = result.get("total_trades", 0)
    trade_history = result.get("trade_history", [])
    portfolio_history = result.get("portfolio_history", [])
    print(f"total_trades: {total_trades}")
    print(f"trade_history length: {len(trade_history)}")
    print(f"portfolio_history length: {len(portfolio_history)}")
    if trade_history:
        print(f"First trade keys: {list(trade_history[0].keys())}")

    # 2. Check backtest_detailed_results
    cur = conn.execute(
        "SELECT task_id, position_analysis IS NOT NULL as has_pos FROM backtest_detailed_results WHERE task_id = ?",
        (task_id,)
    )
    dr = cur.fetchone()
    if dr:
        print(f"backtest_detailed_results: exists, has_position_analysis={bool(dr['has_pos'])}")
        if dr["has_pos"]:
            cur2 = conn.execute("SELECT position_analysis FROM backtest_detailed_results WHERE task_id = ?", (task_id,))
            pa = cur2.fetchone()
            if pa:
                pa_val = json.loads(pa[0]) if isinstance(pa[0], str) else pa[0]
                if isinstance(pa_val, dict) and "stock_performance" in pa_val:
                    print(f"  stock_performance count: {len(pa_val.get('stock_performance', []))}")
    else:
        print("backtest_detailed_results: NOT FOUND")

    # 3. Check trade_records
    cur = conn.execute("SELECT COUNT(*) as c FROM trade_records WHERE task_id = ?", (task_id,))
    tr_count = cur.fetchone()["c"]
    print(f"trade_records count: {tr_count}")

    conn.close()
    print("\nDiagnosis done.")

if __name__ == "__main__":
    main()
