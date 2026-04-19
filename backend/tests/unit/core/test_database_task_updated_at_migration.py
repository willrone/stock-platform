from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import create_engine

from app.core.database import ensure_sqlite_task_updated_at_column_sync


def test_task_updated_at_column_migration_adds_and_backfills(tmp_path: Path) -> None:
    db_path = tmp_path / "migration.db"
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
            estimated_duration INTEGER
        )
        """
    )
    conn.execute(
        """
        INSERT INTO tasks (
            task_id, task_name, task_type, status, user_id, config,
            created_at, started_at, completed_at, progress, result,
            error_message, estimated_duration
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-task",
            "legacy formal task",
            "backtest",
            "running",
            "tester",
            "{}",
            "2026-04-15T09:00:00",
            "2026-04-15T09:05:00",
            None,
            30.0,
            None,
            None,
            None,
        ),
    )
    conn.commit()
    conn.close()

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.begin() as connection:
        ensure_sqlite_task_updated_at_column_sync(connection)

    conn = sqlite3.connect(db_path)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()]
    updated_at = conn.execute(
        "SELECT updated_at FROM tasks WHERE task_id = 'legacy-task'"
    ).fetchone()[0]
    conn.close()

    assert "updated_at" in columns
    assert updated_at == "2026-04-15T09:05:00"
