#!/usr/bin/env python3
"""
迁移交易记录 - 仅使用标准库 (sqlite3, json, datetime)
将 tasks.result.trade_history 同步到 trade_records 表
用法: python3 scripts/migrate_trade_history_stdlib.py <task_id>
"""
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = (BACKEND_DIR / "data" / "app.db").resolve()


def parse_timestamp(val):
    if val is None:
        return datetime.utcnow().isoformat()
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return datetime.utcnow().isoformat()
    return val.isoformat() if hasattr(val, "isoformat") else str(val)


def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not task_id:
        print("用法: python3 migrate_trade_history_stdlib.py <task_id>")
        sys.exit(1)
    if not DB_PATH.exists():
        print(f"数据库不存在: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT result FROM tasks WHERE task_id = ? AND task_type = 'backtest' AND result IS NOT NULL",
        (task_id,),
    )
    row = cur.fetchone()
    if not row:
        print(f"未找到任务或任务无结果: {task_id}")
        conn.close()
        sys.exit(1)

    result = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    trade_history = result.get("trade_history", [])
    if not trade_history:
        print(f"任务无交易历史: {task_id}")
        conn.close()
        sys.exit(0)

    cur = conn.execute("SELECT COUNT(*) FROM trade_records WHERE task_id = ?", (task_id,))
    if cur.fetchone()[0] > 0:
        print(f"任务已有交易记录，跳过: {task_id}")
        conn.close()
        sys.exit(0)

    backtest_id = f"bt_{task_id[:8]}"
    inserts = []
    for i, t in enumerate(trade_history):
        ts = parse_timestamp(t.get("timestamp"))
        action = str(t.get("action", ""))
        qty = int(t.get("quantity", 0))
        price = float(t.get("price", 0))
        comm = float(t.get("commission", 0))
        pnl_val = t.get("pnl")
        pnl = float(pnl_val) if pnl_val is not None else None
        trade_id = t.get("trade_id", f"trade_{task_id[:8]}_{i:06d}")
        stock_code = t.get("stock_code", "")
        inserts.append((task_id, backtest_id, trade_id, stock_code, stock_code, action, qty, price, ts, comm, pnl, None, "{}"))

    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """INSERT INTO trade_records (task_id, backtest_id, trade_id, stock_code, stock_name, action, quantity, price, timestamp, commission, pnl, holding_days, technical_indicators, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(*row, now) for row in inserts],
    )
    conn.commit()
    print(f"✓ 成功插入 {len(inserts)} 条交易记录: {task_id}")
    conn.close()


if __name__ == "__main__":
    main()
