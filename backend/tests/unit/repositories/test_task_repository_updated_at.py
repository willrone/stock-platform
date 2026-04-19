from __future__ import annotations

import time

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.task_models import Task, TaskType
from app.repositories.task_repository import TaskRepository


def _make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Task.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    return Session()


def test_update_task_progress_bumps_updated_at() -> None:
    session = _make_session()
    repo = TaskRepository(session)
    task = repo.create_task(
        task_name="demo-task",
        task_type=TaskType.BACKTEST,
        user_id="tester",
        config={"stock_codes": ["600036.SH"]},
    )

    assert task.updated_at is not None
    first_updated_at = task.updated_at

    time.sleep(0.01)
    refreshed = repo.update_task_progress(task.task_id, 12.5)

    assert refreshed.updated_at > first_updated_at
    assert refreshed.progress == 12.5
