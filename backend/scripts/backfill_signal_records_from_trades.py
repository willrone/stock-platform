#!/usr/bin/env python3
"""
从任务的 trade_history（结果中的交易记录）回填 signal_records 表。
用于：任务有交易记录但 signal_records 表无记录的情况（例如保存信号时曾失败）。

用法:
  cd backend && python3 scripts/backfill_signal_records_from_trades.py <task_id>
  python3 scripts/backfill_signal_records_from_trades.py 13cab251-228e-4c80-b462-7764cccbe7ee
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone


def main():
    if len(sys.argv) < 2:
        print("用法: python3 backfill_signal_records_from_trades.py <task_id>")
        sys.exit(1)
    task_id = sys.argv[1].strip()

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(backend_dir, "data", "app.db")
    if not os.path.isfile(db_path):
        print(f"数据库不存在: {db_path}")
        sys.exit(2)

    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT task_id, result FROM tasks WHERE task_id = ?", (task_id,))
    row = cur.fetchone()
    if not row:
        print(f"未找到任务: {task_id}")
        conn.close()
        sys.exit(3)

    result_raw = row["result"]
    if not result_raw:
        print("任务无 result，无法回填")
        conn.close()
        sys.exit(4)

    result = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
    trade_history = result.get("trade_history") or []
    if not trade_history:
        print("result 中无 trade_history，无需回填")
        conn.close()
        sys.exit(0)

    backtest_id = result.get("backtest_id") or f"bt_{task_id[:8]}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    cur.execute(
        "SELECT COUNT(*) FROM signal_records WHERE task_id = ?", (task_id,)
    )
    already = cur.fetchone()[0]
    if already > 0:
        print(f"该任务已有 {already} 条信号记录，跳过回填（避免重复）。若需强制回填请先删除再运行。")
        conn.close()
        sys.exit(0)

    # 插入：用交易记录当作“已执行的信号”
    inserted = 0
    for t in trade_history:
        ts = t.get("timestamp") or ""
        if isinstance(ts, str) and "T" in ts:
            ts = ts.replace("Z", "+00:00")[:19].replace("T", " ")
        signal_id = f"sig_{uuid.uuid4().hex[:12]}"
        stock_code = t.get("stock_code") or ""
        action = t.get("action", "BUY").upper()
        if action not in ("BUY", "SELL"):
            action = "BUY"
        price = float(t.get("price") or 0)
        cur.execute(
            """
            INSERT INTO signal_records (
                task_id, backtest_id, signal_id, stock_code, stock_name,
                signal_type, timestamp, price, strength, reason, signal_metadata,
                executed, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                backtest_id,
                signal_id,
                stock_code,
                None,
                action,
                ts or now,
                price,
                1.0,
                "从交易记录回填",
                json.dumps({"trade_id": t.get("trade_id"), "from_backfill": True}),
                1,
                now,
            ),
        )
        inserted += 1

    conn.commit()
    conn.close()
    print(f"已从 trade_history 回填 {inserted} 条信号记录到 signal_records，task_id={task_id}")


if __name__ == "__main__":
    main()
