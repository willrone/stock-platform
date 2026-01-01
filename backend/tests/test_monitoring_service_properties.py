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

from app.services.infrastructure import DataMonitoringService, ServiceHealthStatus, PerformanceMetrics, ErrorStatistics
from app.services.data import DataService as StockDataService
from app.services.prediction import TechnicalIndicatorCalculator
from app.services.data import ParquetManager
from app.services.data import DataSyncEngine


@composite
def service_names(draw):
    """生成服务名称"""
    return draw(st.sampled_from(['data_service', 'indicators_service', 'parquet_manager', 'sync_engine']))


@composite
def performance_data(draw):
    """生成性能数据"""
    return {
        'response_time': draw(st.floats(min_value=10.0, max_value=5000.0)),
        'success': draw(st.booleans())
    }


@composite
def error_data(draw):
    """生成错误数据"""
    return {
        'service_name': draw(service_names()),
        'error_type': draw(st.sampled_from(['ConnectionError', 'TimeoutError', 'ValidationError', 'DataError'])),
        'error_message': draw(st.text(min_size=10, max_size=100))
    }


class TestMonitoringServiceProperties:
    """监控服务属性测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟服务
        self.mock_data_service = AsyncMock(spec=StockDataService)
        self.mock_indicators_service = MagicMock(spec=TechnicalIndicatorCalculator)
        self.mock_parquet_manager = MagicMock(spec=ParquetManager)
        self.mock_sync_engine = AsyncMock(spec=DataSyncEngine)
        
        # 创建监控服务
        self.monitoring_service = DataMonitoringService(
            data_service=self.mock_data_service,
            indicators_service=self.mock_indicators_service,
            parquet_manager=self.mock_parquet_manager,
            sync_engine=self.mock_sync_engine
        )
        
        # 清理监控服务状态
        self.monitoring_service._response_times.clear()
        self.monitoring_service._error_counts.clear()
        self.monitoring_service._request_counts.clear()
        self.monitoring_service._error_logs.clear()
        self.monitoring_service._health_cache.clear()
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    @given(service_names())
    @settings(max_examples=10, deadline=5000)
    async def test_health_check_accuracy(self, service_name):
        """
        属性 5: 监控数据准确性
        健康检查应该准确反映服务的实际状态
        **功能: data-management-implementation, 属性 5: 监控数据准确性**
        **验证: 需求 5.1, 5.2**
        """
        # 配置模拟服务行为
        if service_name == "data_service":
            from app.models.stock import DataServiceStatus
            self.mock_data_service.check_remote_service_status.return_value = DataServiceStatus(
                service_url="http://test-service",
                is_available=True,
                response_time_ms=100.0,
                last_check=datetime.now(),
                error_message=None
            )
        elif service_name == "indicators_service":
            self.mock_indicators_service.calculate_moving_average.return_value = [None, None, 2.0]
        elif service_name == "parquet_manager":
            self.mock_parquet_manager.get_storage_stats.return_value = {"total_files": 10}
        elif service_name == "sync_engine":
            self.mock_sync_engine.get_sync_history.return_value = []
        
        # 执行健康检查
        health_status = await self.monitoring_service.check_service_health(service_name)
        
        # 验证健康状态
        assert isinstance(health_status, ServiceHealthStatus)
        assert health_status.service_name == service_name
        assert isinstance(health_status.is_healthy, bool)
        assert isinstance(health_status.response_time_ms, float)
        assert health_status.response_time_ms >= 0
        assert isinstance(health_status.last_check, datetime)
        
        # 验证健康状态与服务实际状态一致
        if service_name == "data_service":
            self.mock_data_service.check_remote_service_status.assert_called_once()
        elif service_name == "indicators_service":
            self.mock_indicators_service.calculate_moving_average.assert_called_once()
        elif service_name == "parquet_manager":
            self.mock_parquet_manager.get_storage_stats.assert_called_once()
        elif service_name == "sync_engine":
            self.mock_sync_engine.get_sync_history.assert_called_once()
    
    @pytest.mark.asyncio
    @given(service_names())
    async def test_health_check_failure_detection(self, service_name):
        """
        属性 5: 监控数据准确性
        健康检查应该准确检测服务故障
        **功能: data-management-implementation, 属性 5: 监控数据准确性**
        **验证: 需求 5.1, 5.2**
        """
        # 配置模拟服务故障
        if service_name == "data_service":
            from app.models.stock import DataServiceStatus
            self.mock_data_service.check_remote_service_status.return_value = DataServiceStatus(
                service_url="http://test-service",
                is_available=False,
                response_time_ms=0.0,
                last_check=datetime.now(),
                error_message="连接失败"
            )
        elif service_name == "indicators_service":
            self.mock_indicators_service.calculate_moving_average.side_effect = Exception("计算失败")
        elif service_name == "parquet_manager":
            self.mock_parquet_manager.get_storage_stats.side_effect = Exception("存储访问失败")
        elif service_name == "sync_engine":
            self.mock_sync_engine.get_sync_history.side_effect = Exception("同步引擎故障")
        
        # 执行健康检查
        health_status = await self.monitoring_service.check_service_health(service_name)
        
        # 验证故障检测
        assert health_status.service_name == service_name
        assert health_status.is_healthy is False
        assert health_status.error_message is not None
        assert len(health_status.error_message) > 0
    
    @pytest.mark.asyncio
    @given(st.lists(performance_data(), min_size=5, max_size=20))
    @settings(max_examples=5, deadline=5000)
    async def test_performance_metrics_accuracy(self, perf_data_list):
        """
        属性 5: 监控数据准确性
        性能指标应该准确计算和统计请求数据
        **功能: data-management-implementation, 属性 5: 监控数据准确性**
        **验证: 需求 5.1, 5.3**
        """
        # 为此测试创建独立的监控服务实例
        test_monitoring_service = DataMonitoringService(
            data_service=self.mock_data_service,
            indicators_service=self.mock_indicators_service,
            parquet_manager=self.mock_parquet_manager,
            sync_engine=self.mock_sync_engine
        )
        
        service_name = "test_service"
        
        # 记录性能数据
        success_count = 0
        response_times = []
        
        for perf_data in perf_data_list:
            test_monitoring_service.record_request(
                service_name, 
                perf_data['response_time'], 
                perf_data['success']
            )
            response_times.append(perf_data['response_time'])
            if perf_data['success']:
                success_count += 1
        
        # 获取性能指标
        metrics = test_monitoring_service.get_performance_metrics(service_name)
        
        # 验证指标准确性
        assert metrics is not None
        assert metrics.service_name == service_name
        assert metrics.request_count == len(perf_data_list)
        assert metrics.error_count == len(perf_data_list) - success_count
        
        # 验证统计计算准确性
        expected_avg = sum(response_times) / len(response_times)
        expected_max = max(response_times)
        expected_min = min(response_times)
        expected_success_rate = success_count / len(perf_data_list)
        
        assert abs(metrics.avg_response_time - expected_avg) < 0.01
        assert metrics.max_response_time == expected_max
        assert metrics.min_response_time == expected_min
        assert abs(metrics.success_rate - expected_success_rate) < 0.01
    
    @pytest.mark.asyncio
    @given(st.lists(error_data(), min_size=1, max_size=10))
    @settings(max_examples=5, deadline=5000)
    async def test_error_statistics_accuracy(self, error_data_list):
        """
        属性 5: 监控数据准确性
        错误统计应该准确记录和分类错误信息
        **功能: data-management-implementation, 属性 5: 监控数据准确性**
        **验证: 需求 5.4, 5.5**
        """
        # 为此测试创建独立的监控服务实例
        test_monitoring_service = DataMonitoringService(
            data_service=self.mock_data_service,
            indicators_service=self.mock_indicators_service,
            parquet_manager=self.mock_parquet_manager,
            sync_engine=self.mock_sync_engine
        )
        
        # 记录错误数据
        for error_data in error_data_list:
            test_monitoring_service.record_error(
                error_data['service_name'],
                error_data['error_type'],
                error_data['error_message']
            )
        
        # 获取错误统计
        error_stats = test_monitoring_service.get_error_statistics(24)
        
        # 验证错误统计准确性
        assert isinstance(error_stats, list)
        
        # 按错误类型分组验证
        error_counts = {}
        for error_data in error_data_list:
            key = f"{error_data['service_name']} - {error_data['error_type']}"
            error_counts[key] = error_counts.get(key, 0) + 1
        
        # 验证统计结果
        stats_by_type = {stat.error_type: stat.count for stat in error_stats}
        
        for expected_type, expected_count in error_counts.items():
            assert expected_type in stats_by_type
            assert stats_by_type[expected_type] == expected_count
    
    @pytest.mark.asyncio
    async def test_data_quality_check_accuracy(self):
        """
        属性 5: 监控数据准确性
        数据质量检查应该准确评估数据状态
        **功能: data-management-implementation, 属性 5: 监控数据准确性**
        **验证: 需求 5.5, 6.4**
        """
        # 配置存储统计模拟数据
        mock_storage_stats = {
            'total_files': 50,
            'total_size': 10240000,  # 10MB
            'total_records': 100000,
            'stock_count': 25,
            'date_range': {
                'start': '2023-01-01',
                'end': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            }
        }
        
        self.mock_parquet_manager.get_storage_stats.return_value = mock_storage_stats
        
        # 执行数据质量检查
        quality_report = self.monitoring_service.check_data_quality()
        
        # 验证质量报告结构
        assert 'overall_score' in quality_report
        assert 'checks' in quality_report
        assert 'issues' in quality_report
        assert 'recommendations' in quality_report
        
        # 验证评分范围
        assert 0.0 <= quality_report['overall_score'] <= 1.0
        
        # 验证检查项目
        checks = quality_report['checks']
        assert 'data_completeness' in checks
        assert 'data_freshness' in checks
        assert 'storage_efficiency' in checks
        assert 'stock_coverage' in checks
        
        # 验证数据完整性检查
        completeness_check = checks['data_completeness']
        assert completeness_check['score'] == 1.0  # 有文件，应该通过
        assert completeness_check['status'] == 'pass'
        
        # 验证股票覆盖度检查
        coverage_check = checks['stock_coverage']
        assert coverage_check['stock_count'] == 25
        # 25只股票应该是fair级别
        assert coverage_check['status'] == 'fair'
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_accuracy(self):
        """
        属性 5: 监控数据准确性
        异常检测应该准确识别系统异常
        **功能: data-management-implementation, 属性 5: 监控数据准确性**
        **验证: 需求 5.2, 5.4**
        """
        service_name = "test_service"
        
        # 模拟高错误率场景
        for i in range(20):
            success = i < 5  # 只有前5个成功，错误率75%
            self.monitoring_service.record_request(service_name, 100.0, success)
        
        # 模拟慢响应场景
        slow_service = "slow_service"
        for i in range(10):
            response_time = 8000.0 if i < 7 else 1000.0  # 前7个响应很慢，平均值会超过5000ms
            self.monitoring_service.record_request(slow_service, response_time, True)
        
        # 模拟不健康服务
        unhealthy_service = "unhealthy_service"
        unhealthy_status = ServiceHealthStatus(
            service_name=unhealthy_service,
            is_healthy=False,
            response_time_ms=0.0,
            last_check=datetime.now(),
            error_message="服务不可用"
        )
        self.monitoring_service._health_cache[unhealthy_service] = unhealthy_status
        
        # 执行异常检测
        anomalies = self.monitoring_service.detect_anomalies()
        
        # 验证异常检测结果
        assert isinstance(anomalies, list)
        assert len(anomalies) > 0
        
        # 验证检测到的异常类型
        anomaly_types = [anomaly['type'] for anomaly in anomalies]
        
        # 应该检测到高错误率异常
        assert 'high_error_rate' in anomaly_types
        
        # 应该检测到慢响应异常
        assert 'slow_response' in anomaly_types
        
        # 应该检测到服务不健康异常
        assert 'service_unhealthy' in anomaly_types
        
        # 验证异常详细信息
        for anomaly in anomalies:
            assert 'type' in anomaly
            assert 'service' in anomaly
            assert 'severity' in anomaly
            assert 'message' in anomaly
            assert 'detected_at' in anomaly
            assert anomaly['severity'] in ['high', 'medium', 'low']
    
    @pytest.mark.asyncio
    async def test_system_overview_completeness(self):
        """
        属性 5: 监控数据准确性
        系统概览应该包含完整和准确的系统状态信息
        **功能: data-management-implementation, 属性 5: 监控数据准确性**
        **验证: 需求 5.1, 5.2, 5.3**
        """
        # 配置模拟数据
        self.mock_parquet_manager.get_storage_stats.return_value = {
            'total_files': 10,
            'total_size': 1024000,
            'stock_count': 5
        }
        
        # 添加一些性能数据
        self.monitoring_service.record_request("data_service", 150.0, True)
        self.monitoring_service.record_request("data_service", 200.0, False)
        
        # 添加健康状态
        health_status = ServiceHealthStatus(
            service_name="data_service",
            is_healthy=True,
            response_time_ms=150.0,
            last_check=datetime.now(),
            error_message=None
        )
        self.monitoring_service._health_cache["data_service"] = health_status
        
        # 获取系统概览
        overview = self.monitoring_service.get_system_overview()
        
        # 验证概览结构完整性
        required_fields = [
            'timestamp', 'services', 'overall_health', 
            'total_requests', 'total_errors', 'storage_stats', 'data_quality'
        ]
        
        for field in required_fields:
            assert field in overview, f"缺少必需字段: {field}"
        
        # 验证服务信息
        assert 'data_service' in overview['services']
        service_info = overview['services']['data_service']
        
        assert 'healthy' in service_info
        assert 'response_time' in service_info
        assert 'last_check' in service_info
        assert 'metrics' in service_info
        
        # 验证统计数据准确性
        assert overview['total_requests'] == 2
        assert overview['total_errors'] == 1
        
        # 验证时间戳格式
        timestamp = datetime.fromisoformat(overview['timestamp'])
        assert isinstance(timestamp, datetime)


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
    # 清理代码可以在这里添加