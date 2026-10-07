"""Integration 测试的认证旁路。

分支把受保护路由的认证从 X-User-ID 后门改成真 JWT：``protected_router`` 挂
``Depends(get_current_user)``，``require_current_user`` 还要查库。存量集成测试
（``test_integration.py`` / ``test_integration_simple.py`` /
``test_backtest_portfolio.py`` …）直接用共享的 ``app.main.app`` 建 ``TestClient``，
既没有合法 token 也没有 users 表记录，因此全部 401。

这里用 autouse fixture 通过 FastAPI 官方 ``dependency_overrides`` 装上与单元测试
相同的替身（见 ``tests/conftest.py::install_auth_overrides``），用例结束后把被触碰
的两个 key 还原成用例前的状态：用例前不存在则 pop 掉，若已存在（例如共享 app 上
其他测试先装了自己的 override）则恢复原值，避免影响同一 session 里的其他测试。
"""

from collections.abc import Generator
from typing import Any, Dict

import pytest

from app.api.v1.dependencies import get_current_user, require_current_user
from app.main import app
from tests.conftest import install_auth_overrides

_AUTH_DEPENDENCIES = (get_current_user, require_current_user)
_MISSING = object()


@pytest.fixture(autouse=True)
def _auth_overrides() -> Generator[None, None, None]:
    """用例内安装认证替身，结束后仅还原本 fixture 触碰的 override key。"""
    previous: Dict[Any, Any] = {
        key: app.dependency_overrides.get(key, _MISSING) for key in _AUTH_DEPENDENCIES
    }
    install_auth_overrides(app)
    yield
    for key, old_value in previous.items():
        if old_value is _MISSING:
            app.dependency_overrides.pop(key, None)
        else:
            app.dependency_overrides[key] = old_value
