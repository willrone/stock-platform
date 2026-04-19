from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.services.tasks.task_monitor import TaskMonitor


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _db_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")


def _create_tasks_table(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            task_name TEXT NOT NULL,
            task_type TEXT NOT NULL,
            status TEXT NOT NULL,
            user_id TEXT,
            config TEXT,
            created_at DATETIME NOT NULL,
            started_at DATETIME,
            completed_at DATETIME,
            progress FLOAT NOT NULL,
            result TEXT,
            error_message TEXT,
            estimated_duration INTEGER,
            updated_at DATETIME
        )
        """
    )
    conn.commit()
    conn.close()


def test_get_stuck_tasks_uses_updated_at_heartbeat(tmp_path: Path) -> None:
    db_path = tmp_path / "tasks.db"
    _create_tasks_table(db_path)

    now = _utcnow()
    started_at = now - timedelta(minutes=95)
    created_at = now - timedelta(minutes=100)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO tasks (
            task_id, task_name, task_type, status, user_id, config,
            created_at, started_at, completed_at, progress, result,
            error_message, estimated_duration, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "active-task",
            "active formal task",
            "backtest",
            "running",
            "tester",
            "{}",
            _db_timestamp(created_at),
            _db_timestamp(started_at),
            None,
            45.0,
            None,
            None,
            None,
            _db_timestamp(now - timedelta(minutes=5)),
        ),
    )
    conn.execute(
        """
        INSERT INTO tasks (
            task_id, task_name, task_type, status, user_id, config,
            created_at, started_at, completed_at, progress, result,
            error_message, estimated_duration, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "stale-task",
            "stale formal task",
            "backtest",
            "running",
            "tester",
            "{}",
            _db_timestamp(created_at),
            _db_timestamp(started_at),
            None,
            45.0,
            None,
            None,
            None,
            _db_timestamp(now - timedelta(minutes=95)),
        ),
    )
    conn.commit()
    conn.close()

    monitor = TaskMonitor(db_path=str(db_path))
    stuck_tasks = monitor.get_stuck_tasks(timeout_minutes=60)
    stuck_ids = {task["task_id"] for task in stuck_tasks}

    assert stuck_ids == {"stale-task"}
