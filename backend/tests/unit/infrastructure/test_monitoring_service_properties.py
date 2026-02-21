"""
监控服务属性测试
验证监控数据准确性属性
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.services.infrastructure.monitoring_service import (
    DataMonitoringService,
    ErrorStatistics,
    PerformanceMetrics,
    ServiceHealthStatus,
)
from app.services.data import SimpleDataService
from app.services.data.parquet_manager import ParquetManager
from app.services.prediction import TechnicalIndicatorCalculator


@composite
def service_names(draw):
    """生成服务名称"""
    return draw(st.sampled_from(["data_service", "indicators_service", "parquet_manager"]))


@composite
def performance_data(draw):
    """生成性能数据"""
    return {
        "response_time": draw(st.floats(min_value=10.0, max_value=5000.0)),
        "success": draw(st.booleans()),
    }


@composite
def error_data(draw):
    """生成错误数据"""
    return {
        "service_name": draw(service_names()),
        "error_type": draw(
            st.sampled_from(["ConnectionError", "TimeoutError", "ValidationError"])
        ),
        "error_message": draw(st.text(min_size=10, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N")))),
    }


def _make_monitoring_service():
    """创建带 mock 依赖的监控服务"""
    mock_data = AsyncMock(spec=SimpleDataService)
    mock_ind = MagicMock(spec=TechnicalIndicatorCalculator)
    mock_pq = MagicMock(spec=ParquetManager)
    svc = DataMonitoringService(
        data_service=mock_data,
        indicators_service=mock_ind,
        parquet_manager=mock_pq,
    )
    return svc, mock_data, mock_ind, mock_pq


class TestMonitoringServiceProperties:
    """监控服务属性测试类"""

    def setup_method(self):
        self.svc, self.mock_data, self.mock_ind, self.mock_pq = _make_monitoring_service()

    @pytest.mark.asyncio
    @given(service_names())
    @settings(max_examples=6, deadline=10000)
    async def test_health_check_accuracy(self, service_name):
        """健康检查应准确反映服务状态"""
        svc, mock_data, mock_ind, mock_pq = _make_monitoring_service()

        if service_name == "data_service":
            from app.models.stock import DataServiceStatus
            mock_data.check_remote_service_status.return_value = DataServiceStatus(
                service_url="http://test",
                is_available=True,
                response_time_ms=100.0,
                last_check=datetime.now(),
                error_message=None,
            )
        elif service_name == "indicators_service":
            mock_ind.calculate_moving_average.return_value = [None, None, 2.0]
        elif service_name == "parquet_manager":
            mock_pq.get_storage_stats.return_value = {"total_files": 10}

        status = await svc.check_service_health(service_name)

        assert isinstance(status, ServiceHealthStatus)
        assert status.service_name == service_name
        assert isinstance(status.is_healthy, bool)
        assert status.response_time_ms >= 0
        assert isinstance(status.last_check, datetime)

    @pytest.mark.asyncio
    @given(service_names())
    @settings(max_examples=6, deadline=10000)
    async def test_health_check_failure_detection(self, service_name):
        """健康检查应检测服务故障"""
        svc, mock_data, mock_ind, mock_pq = _make_monitoring_service()

        if service_name == "data_service":
            from app.models.stock import DataServiceStatus
            mock_data.check_remote_service_status.return_value = DataServiceStatus(
                service_url="http://test",
                is_available=False,
                response_time_ms=0.0,
                last_check=datetime.now(),
                error_message="连接失败",
            )
        elif service_name == "indicators_service":
            mock_ind.calculate_moving_average.side_effect = Exception("计算失败")
        elif service_name == "parquet_manager":
            mock_pq.get_storage_stats.side_effect = Exception("存储访问失败")

        status = await svc.check_service_health(service_name)
        assert status.is_healthy is False
        assert status.error_message is not None

    @pytest.mark.asyncio
    @given(st.lists(performance_data(), min_size=5, max_size=20))
    @settings(max_examples=5, deadline=10000)
    async def test_performance_metrics_accuracy(self, perf_data_list):
        """性能指标应准确计算"""
        svc, _, _, _ = _make_monitoring_service()
        service_name = "test_service"

        success_count = 0
        response_times = []
        for p in perf_data_list:
            svc.record_request(service_name, p["response_time"], p["success"])
            response_times.append(p["response_time"])
            if p["success"]:
                success_count += 1

        metrics = svc.get_performance_metrics(service_name)
        assert metrics is not None
        assert metrics.service_name == service_name
        assert metrics.request_count == len(perf_data_list)
        assert metrics.error_count == len(perf_data_list) - success_count

        expected_avg = sum(response_times) / len(response_times)
        assert abs(metrics.avg_response_time - expected_avg) < 0.01
        assert metrics.max_response_time == max(response_times)
        assert metrics.min_response_time == min(response_times)

    @pytest.mark.asyncio
    @given(st.lists(error_data(), min_size=1, max_size=10))
    @settings(max_examples=5, deadline=10000)
    async def test_error_statistics_accuracy(self, error_data_list):
        """错误统计应准确记录"""
        svc, _, _, _ = _make_monitoring_service()

        for ed in error_data_list:
            svc.record_error(ed["service_name"], ed["error_type"], ed["error_message"])

        stats = svc.get_error_statistics(24)
        assert isinstance(stats, list)

        # 每个统计项应有正确结构
        for s in stats:
            assert isinstance(s, ErrorStatistics)
            assert s.count > 0

    @pytest.mark.asyncio
    async def test_data_quality_check(self):
        """数据质量检查应返回完整报告"""
        self.mock_pq.get_storage_stats.return_value = {
            "total_files": 50,
            "total_size": 10240000,
            "total_records": 100000,
            "stock_count": 25,
            "date_range": {
                "start": "2023-01-01",
                "end": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
            },
        }

        report = self.svc.check_data_quality()
        assert "overall_score" in report
        assert 0.0 <= report["overall_score"] <= 1.0
        assert "checks" in report
        assert "issues" in report
        assert "recommendations" in report

    @pytest.mark.asyncio
    async def test_anomaly_detection(self):
        """异常检测应识别高错误率和慢响应"""
        # 高错误率
        for i in range(20):
            self.svc.record_request("err_svc", 100.0, i < 5)

        # 慢响应
        for i in range(10):
            self.svc.record_request("slow_svc", 8000.0 if i < 7 else 1000.0, True)

        # 不健康服务
        self.svc._health_cache["dead_svc"] = ServiceHealthStatus(
            service_name="dead_svc",
            is_healthy=False,
            response_time_ms=0.0,
            last_check=datetime.now(),
            error_message="服务不可用",
        )

        anomalies = self.svc.detect_anomalies()
        assert isinstance(anomalies, list)
        assert len(anomalies) > 0

        types = [a["type"] for a in anomalies]
        assert "high_error_rate" in types
        assert "slow_response" in types
        assert "service_unhealthy" in types

        for a in anomalies:
            assert "type" in a
            assert "service" in a
            assert "severity" in a
            assert "message" in a

    @pytest.mark.asyncio
    async def test_system_overview_completeness(self):
        """系统概览应包含完整信息"""
        self.mock_pq.get_storage_stats.return_value = {
            "total_files": 10,
            "total_size": 1024000,
            "stock_count": 5,
        }
        self.svc.record_request("data_service", 150.0, True)
        self.svc.record_request("data_service", 200.0, False)

        self.svc._health_cache["data_service"] = ServiceHealthStatus(
            service_name="data_service",
            is_healthy=True,
            response_time_ms=150.0,
            last_check=datetime.now(),
            error_message=None,
        )

        overview = self.svc.get_system_overview()
        assert "timestamp" in overview
        assert "services" in overview
        assert "overall_health" in overview
        assert "total_requests" in overview
        assert "total_errors" in overview
        assert overview["total_requests"] == 2
        assert overview["total_errors"] == 1


@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
