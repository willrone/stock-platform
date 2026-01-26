"""
回测详细 API 单元测试
测试 /backtest-detailed 相关端点，使用 mock 避免数据库依赖
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.backtest_detailed import router
from app.core.database import get_async_session


async def _mock_get_async_session():
    """用于覆盖的 mock 异步 session 生成器"""
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    yield session


@pytest.fixture
def app():
    """创建仅包含回测详细路由的测试应用"""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_async_session] = _mock_get_async_session
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_detailed_result():
    """模拟 BacktestDetailedResult"""
    m = MagicMock()
    m.to_dict.return_value = {
        "id": 1,
        "task_id": "task-1",
        "backtest_id": "bt-1",
        "extended_risk_metrics": {"sortino_ratio": 1.0, "calmar_ratio": 0.5},
        "drawdown_analysis": None,
        "monthly_returns": None,
        "position_analysis": {"stock_performance": []},
        "benchmark_comparison": None,
        "rolling_metrics": None,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
    }
    return m


@pytest.fixture
def mock_repository(mock_detailed_result):
    """模拟 BacktestDetailedRepository"""
    repo = MagicMock()
    repo.get_detailed_result_by_task_id = AsyncMock(return_value=mock_detailed_result)
    repo.get_portfolio_snapshots = AsyncMock(return_value=[])
    repo.get_trade_records = AsyncMock(return_value=[])
    repo.get_trade_records_count = AsyncMock(return_value=0)
    repo.get_trade_statistics = AsyncMock(return_value={"total_trades": 0, "win_rate": 0.0})
    repo.get_signal_records = AsyncMock(return_value=[])
    repo.get_signal_records_count = AsyncMock(return_value=0)
    repo.get_signal_statistics = AsyncMock(return_value={"total_signals": 0})
    repo.get_benchmark_data = AsyncMock(return_value=None)
    repo.delete_task_data = AsyncMock(return_value=True)
    return repo


class TestBacktestDetailedAPI:
    """回测详细 API 测试类"""

    @patch("app.api.v1.backtest_detailed.BacktestDetailedRepository")
    def test_get_detailed_result_200(
        self, mock_repo_cls, client, mock_repository, mock_detailed_result
    ):
        """GET /{task_id}/detailed-result 返回 200"""
        mock_repo_cls.return_value = mock_repository
        resp = client.get("/backtest-detailed/task-1/detailed-result")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["task_id"] == "task-1"

    @patch("app.api.v1.backtest_detailed.BacktestDetailedRepository")
    def test_get_detailed_result_404(self, mock_repo_cls, client):
        """GET /{task_id}/detailed-result 无数据时 404"""
        repo_404 = MagicMock()
        repo_404.get_detailed_result_by_task_id = AsyncMock(return_value=None)
        mock_repo_cls.return_value = repo_404
        resp = client.get("/backtest-detailed/nonexistent/detailed-result")
        assert resp.status_code == 404
        assert "未找到" in resp.json().get("detail", "")

    @patch("app.api.v1.backtest_detailed.BacktestDetailedRepository")
    def test_get_trade_statistics_200(self, mock_repo_cls, client, mock_repository):
        """GET /{task_id}/trade-statistics 返回 200"""
        mock_repo_cls.return_value = mock_repository
        resp = client.get("/backtest-detailed/task-1/trade-statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["total_trades"] == 0

    @patch("app.api.v1.backtest_detailed.BacktestDetailedRepository")
    def test_get_signal_statistics_200(self, mock_repo_cls, client, mock_repository):
        """GET /{task_id}/signal-statistics 返回 200"""
        mock_repo_cls.return_value = mock_repository
        resp = client.get("/backtest-detailed/task-1/signal-statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["total_signals"] == 0

    @patch("app.api.v1.backtest_detailed.BacktestDetailedRepository")
    def test_get_trade_records_200(self, mock_repo_cls, client, mock_repository):
        """GET /{task_id}/trade-records 返回 200"""
        mock_repo_cls.return_value = mock_repository
        resp = client.get("/backtest-detailed/task-1/trade-records")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "trades" in data["data"]
        assert "pagination" in data["data"]

    @patch("app.api.v1.backtest_detailed.BacktestDetailedRepository")
    def test_get_portfolio_snapshots_200(self, mock_repo_cls, client, mock_repository):
        """GET /{task_id}/portfolio-snapshots 返回 200"""
        mock_repo_cls.return_value = mock_repository
        resp = client.get("/backtest-detailed/task-1/portfolio-snapshots")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "snapshots" in data["data"]

    @patch("app.api.v1.backtest_detailed.ChartCacheService")
    def test_cache_chart_200(self, mock_cache_cls, client):
        """POST /{task_id}/cache-chart 成功时返回 200"""
        svc = MagicMock()
        svc.cache_chart_data = AsyncMock(return_value=True)
        mock_cache_cls.return_value = svc
        resp = client.post(
            "/backtest-detailed/task-1/cache-chart",
            json={
                "chart_type": "equity_curve",
                "chart_data": {"dates": ["2024-01-01"], "values": [100]},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["chart_type"] == "equity_curve"

    @patch("app.api.v1.backtest_detailed.ChartCacheService")
    def test_get_cached_chart_200(self, mock_cache_cls, client):
        """GET /{task_id}/cached-chart/{chart_type} 有缓存时返回 200"""
        svc = MagicMock()
        svc.get_cached_chart_data = AsyncMock(
            return_value={"dates": ["2024-01-01"], "values": [100]}
        )
        mock_cache_cls.return_value = svc
        resp = client.get("/backtest-detailed/task-1/cached-chart/equity_curve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "data" in data

    @patch("app.api.v1.backtest_detailed.ChartCacheService")
    def test_get_cached_chart_404(self, mock_cache_cls, client):
        """GET /{task_id}/cached-chart/{chart_type} 无缓存时 404"""
        svc = MagicMock()
        svc.get_cached_chart_data = AsyncMock(return_value=None)
        mock_cache_cls.return_value = svc
        resp = client.get("/backtest-detailed/task-1/cached-chart/equity_curve")
        assert resp.status_code == 404

    @patch("app.api.v1.backtest_detailed.ChartCacheService")
    def test_get_cache_statistics_200(self, mock_cache_cls, client):
        """GET /cache/statistics 返回 200"""
        svc = MagicMock()
        svc.get_cache_statistics = AsyncMock(return_value={"total_caches": 0})
        mock_cache_cls.return_value = svc
        resp = client.get("/backtest-detailed/cache/statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    @patch("app.api.v1.backtest_detailed.ChartCacheService")
    def test_cleanup_expired_cache_200(self, mock_cache_cls, client):
        """DELETE /cache/cleanup 返回 200"""
        svc = MagicMock()
        svc.cleanup_expired_cache = AsyncMock(return_value=3)
        mock_cache_cls.return_value = svc
        resp = client.delete("/backtest-detailed/cache/cleanup")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["deleted_count"] == 3

    @patch("app.api.v1.backtest_detailed.ChartCacheService")
    def test_invalidate_cache_200(self, mock_cache_cls, client):
        """DELETE /{task_id}/cache 返回 200"""
        svc = MagicMock()
        svc.invalidate_cache = AsyncMock(return_value=True)
        mock_cache_cls.return_value = svc
        resp = client.delete("/backtest-detailed/task-1/cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    @patch("app.api.v1.backtest_detailed.BacktestDetailedRepository")
    def test_delete_task_data_200(self, mock_repo_cls, client, mock_repository):
        """DELETE /{task_id}/data 返回 200"""
        mock_repo_cls.return_value = mock_repository
        resp = client.delete("/backtest-detailed/task-1/data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == "task-1"
