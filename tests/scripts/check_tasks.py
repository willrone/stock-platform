#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('backend/data/app.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT task_id, task_name, task_type, status, created_at, completed_at
    FROM tasks
    WHERE task_type = 'backtest'
    ORDER BY created_at DESC
    LIMIT 10
""")

tasks = cursor.fetchall()
print(f"找到 {len(tasks)} 个回测任务:\n")
for task in tasks:
    print(f"ID: {task[0][:8]}...")
    print(f"  名称: {task[1]}")
    print(f"  状态: {task[3]}")
    print(f"  创建: {task[4]}")
    print(f"  完成: {task[5]}")
    print()

conn.close()
