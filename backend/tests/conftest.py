"""
简化的Pytest配置
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# CI runs the full historical `tests/` tree. Keep this hook explicit so
# `scripts/check_ci_tail_cleanup_sync.py` can detect any future temporary
# isolations and require matching ledger entries.
if os.getenv("GITHUB_ACTIONS") == "true":
    collect_ignore = []


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def pytest_configure(config: pytest.Config) -> None:
    """Prepare the legacy full-test CI database before modules import app.main.

    Many historical integration tests instantiate ``TestClient(app)`` without a
    context manager, so FastAPI lifespan startup does not run and tables are not
    created. CI executes that full legacy tree from a clean checkout, therefore
    create the app schema here once before test collection imports routes.
    """
    if os.getenv("GITHUB_ACTIONS") != "true":
        return

    from app.core.database import (
        Base,
        _seed_ci_smoke_models_sync,
        ensure_sqlite_task_updated_at_column_sync,
        sync_engine,
    )
    from app.models import backtest_detailed_models  # noqa: F401
    from app.models import strategy_config_models  # noqa: F401
    from app.models import task_models  # noqa: F401

    with sync_engine.begin() as connection:
        Base.metadata.create_all(bind=connection)
        ensure_sqlite_task_updated_at_column_sync(connection)
        _seed_ci_smoke_models_sync(connection)
