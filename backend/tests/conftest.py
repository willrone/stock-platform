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


def install_auth_overrides(app: object) -> None:
    """为存量契约/属性测试提供认证旁路。

    重构后业务路由要求 JWT + 真实 User（``require_current_user`` 会查库），旧测试
    只 stub 了 ``get_current_user``。这里通过 FastAPI 官方 ``dependency_overrides``
    把两个依赖都替换为轻量替身，不落真实 users 表；仅对显式传入的测试 app 生效，
    不改变生产行为。调用方应在用例结束后 ``pop`` 还原（共享 app 时）。
    """
    from types import SimpleNamespace

    from app.api.v1.dependencies import get_current_user, require_current_user

    app.dependency_overrides[get_current_user] = lambda: "test-user"  # type: ignore[attr-defined]
    app.dependency_overrides[require_current_user] = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        id="test-user",
        is_admin=False,
        is_active=True,
        subscription_tier="free",
        username="test-user",
        email="test-user@example.com",
    )


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
