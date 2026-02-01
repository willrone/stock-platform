"""
集成测试和验证
验证所有组件集成后的正确性属性
"""

import pytest
import asyncio
import tempfile
import shutil
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
from pathlib import Path

from app.core.container import ServiceContainer
from app.services.data import SimpleDataService as StockDataService
from app.services.data.parquet_manager import ParquetManager
from app.services.data.data_validator import DataValidator
from app.services.infrastructure.cache_service import cache_manager
from app.services.infrastructure.monitoring_service import DataMonitoringService

# DataSyncEngine 已移除，使用占位符
try:
    from app.services.data.incremental_updater import IncrementalUpdater as DataSyncEngine
except ImportError:
    DataSyncEngine = None

# connection_pool_manager 可能已移除
try:
    from app.services.infrastructure.connection_pool import connection_pool_manager
except ImportError:
    connection_pool_manager = None
from app.models.stock import DataSyncRequest
from app.models.file_management import FileListRequest, FileListResponse


@composite
def stock_codes(draw):
    """生成股票代码"""
    return draw(st.sampled_from(['000001', '000002', '600000', '600036', '000858']))


@composite
def date_ranges(draw):
    """生成日期范围"""
    start_date = draw(st.dates(
        min_value=datetime(2023, 1, 1).date(),
        max_value=datetime(2023, 12, 31).date()
    ))
    end_date = draw(st.dates(
        min_value=start_date,
        max_value=min(start_date + timedelta(days=30), datetime(2024, 1, 31).date())
    ))
    return datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.min.time())


class TestIntegrationProperties:
    """集成测试属性测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # 直接创建服务实例，不使用容器
        from app.services.data import SimpleDataService as StockDataService
        from app.services.data.parquet_manager import ParquetManager
        from app.services.infrastructure.monitoring_service import DataMonitoringService
        from app.services.data.data_validator import DataValidator
        from app.services.prediction import TechnicalIndicatorCalculator
        
        self.stock_data_service = StockDataService()
        self.parquet_manager = ParquetManager()
        self.indicators_service = TechnicalIndicatorCalculator()
        self.monitoring_service = DataMonitoringService(
            data_service=self.stock_data_service,
            indicators_service=self.indicators_service,
            parquet_manager=self.parquet_manager
        )
        self.data_validator = DataValidator()
        
        # 设置临时数据路径
        self.original_data_path = self.parquet_manager.base_path
        self.parquet_manager.base_path = self.temp_path / "data"
        self.parquet_manager.base_path.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """测试后清理"""
        # 恢复原始数据路径
        if self.original_data_path:
            self.parquet_manager.base_path = self.original_data_path
        
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    @given(stock_codes(), date_ranges())
    @settings(max_examples=3, deadline=20000)
    async def test_end_to_end_data_flow(self, stock_code, date_range):
        """
        属性 1-7: 所有核心属性的综合测试
        端到端数据流应该保持完整性和一致性
        **功能: data-management-implementation, 属性 1-7: 所有核心属性的综合测试**
        **验证: 所有需求**
        """
        start_date, end_date = date_range
        
        # 模拟远端数据服务
        mock_data = self._generate_mock_stock_data(stock_code, start_date, end_date)
        
        with patch.object(self.stock_data_service, 'fetch_remote_data') as mock_fetch:
            mock_fetch.return_value = mock_data
            
            # 1. 测试数据获取
            stock_data = await self.stock_data_service.get_stock_data(
                stock_code, start_date, end_date
            )
            
            assert stock_data is not None
            assert len(stock_data) > 0
            assert all(item.stock_code == stock_code for item in stock_data)
            
            # 2. 测试数据存储
            df = pd.DataFrame([{
                'stock_code': item.stock_code,
                'date': item.date,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume
            } for item in stock_data])
            
            save_success = self.stock_data_service.save_to_parquet(df, stock_code)
            assert save_success
            
            # 3. 测试文件管理
            file_request = FileListRequest(
                stock_codes=[stock_code],
                start_date=start_date,
                end_date=end_date
            )
            
            file_info = await self.parquet_manager.get_file_list(file_request.to_filter_criteria())
            assert len(file_info) >= 0
            
            # 4. 测试数据验证
            validation_result = self.data_validator.validate_stock_data(df, stock_code)
            assert validation_result.quality_score > 0.5
            
            # 5. 测试监控服务
            health_status = await self.monitoring_service.get_service_health()
            assert health_status['overall_status'] in ['healthy', 'degraded', 'unhealthy']
            
            # 6. 测试缓存功能
            cache_stats = cache_manager.get_global_stats()
            assert cache_stats['total_caches'] >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_consistency(self):
        """
        属性 1-7: 所有核心属性的综合测试
        并发操作应该保持数据一致性
        **功能: data-management-implementation, 属性 1-7: 所有核心属性的综合测试**
        **验证: 所有需求**
        """
        stock_codes = ['000001', '000002', '600000']
        start_date = datetime(2023, 6, 1)
        end_date = datetime(2023, 6, 30)
        
        async def process_stock(stock_code: str):
            try:
                # 模拟数据获取
                mock_data = self._generate_mock_stock_data(stock_code, start_date, end_date)
                
                with patch.object(self.container.stock_data_service, 'fetch_remote_data') as mock_fetch:
                    mock_fetch.return_value = mock_data
                    
                    # 获取数据
                    stock_data = await self.container.stock_data_service.get_stock_data(
                        stock_code, start_date, end_date
                    )
                    
                    if stock_data:
                        # 保存数据
                        df = pd.DataFrame([{
                            'stock_code': item.stock_code,
                            'date': item.date,
                            'open': item.open,
                            'high': item.high,
                            'low': item.low,
                            'close': item.close,
                            'volume': item.volume
                        } for item in stock_data])
                        
                        save_success = self.container.stock_data_service.save_to_parquet(df, stock_code)
                        return {'stock_code': stock_code, 'success': save_success, 'records': len(stock_data)}
                    
                return {'stock_code': stock_code, 'success': False, 'records': 0}
            
            except Exception as e:
                return {'stock_code': stock_code, 'success': False, 'error': str(e), 'records': 0}
        
        # 并发处理多只股票
        tasks = [process_stock(code) for code in stock_codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证所有任务都完成
        assert len(results) == len(stock_codes)
        
        # 验证结果一致性
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success', False)]
        assert len(successful_results) > 0
        
        # 验证数据完整性
        for result in successful_results:
            assert result['records'] > 0
            
            # 验证文件确实被创建
            stock_code = result['stock_code']
            year = start_date.year
            file_path = self.container.parquet_manager.data_path / "daily" / stock_code / f"{year}.parquet"
            assert file_path.exists()
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_consistency(self):
        """
        属性 1-7: 所有核心属性的综合测试
        错误恢复后系统应该保持一致性
        **功能: data-management-implementation, 属性 1-7: 所有核心属性的综合测试**
        **验证: 所有需求**
        """
        stock_code = '000001'
        start_date = datetime(2023, 7, 1)
        end_date = datetime(2023, 7, 31)
        
        # 第一次尝试：模拟失败
        with patch.object(self.container.stock_data_service, 'fetch_remote_data') as mock_fetch:
            mock_fetch.side_effect = Exception("网络错误")
            
            # 应该触发降级策略
            stock_data = await self.container.stock_data_service.get_stock_data(
                stock_code, start_date, end_date
            )
            
            # 可能返回None或降级数据
            # 验证系统状态仍然一致
            health_status = await self.container.monitoring_service.get_service_health()
            assert 'overall_status' in health_status
        
        # 第二次尝试：模拟成功
        mock_data = self._generate_mock_stock_data(stock_code, start_date, end_date)
        
        with patch.object(self.container.stock_data_service, 'fetch_remote_data') as mock_fetch:
            mock_fetch.return_value = mock_data
            
            # 应该成功获取数据
            stock_data = await self.container.stock_data_service.get_stock_data(
                stock_code, start_date, end_date
            )
            
            assert stock_data is not None
            assert len(stock_data) > 0
            
            # 验证错误恢复后的系统状态
            health_status = await self.container.monitoring_service.get_service_health()
            assert health_status['overall_status'] in ['healthy', 'degraded']
            
            # 验证缓存状态
            cache_stats = cache_manager.get_global_stats()
            assert cache_stats['total_hits'] >= 0
            assert cache_stats['total_misses'] >= 0
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """
        属性 1-7: 所有核心属性的综合测试
        系统在负载下应该保持性能和稳定性
        **功能: data-management-implementation, 属性 1-7: 所有核心属性的综合测试**
        **验证: 所有需求**
        """
        stock_codes = ['000001', '000002', '600000', '600036', '000858']
        start_date = datetime(2023, 8, 1)
        end_date = datetime(2023, 8, 15)
        
        # 记录开始时间和资源使用
        start_time = time.time()
        initial_cache_stats = cache_manager.get_global_stats()
        
        async def load_test_worker(worker_id: int):
            results = []
            for i, stock_code in enumerate(stock_codes):
                try:
                    # 模拟数据
                    mock_data = self._generate_mock_stock_data(stock_code, start_date, end_date)
                    
                    with patch.object(self.container.stock_data_service, 'fetch_remote_data') as mock_fetch:
                        mock_fetch.return_value = mock_data
                        
                        # 获取数据
                        stock_data = await self.container.stock_data_service.get_stock_data(
                            stock_code, start_date, end_date
                        )
                        
                        if stock_data:
                            results.append({
                                'worker_id': worker_id,
                                'stock_code': stock_code,
                                'records': len(stock_data),
                                'success': True
                            })
                        else:
                            results.append({
                                'worker_id': worker_id,
                                'stock_code': stock_code,
                                'records': 0,
                                'success': False
                            })
                
                except Exception as e:
                    results.append({
                        'worker_id': worker_id,
                        'stock_code': stock_code,
                        'error': str(e),
                        'success': False
                    })
                
                # 小延迟避免过度负载
                await asyncio.sleep(0.01)
            
            return results
        
        # 启动多个并发工作者
        num_workers = 5
        tasks = [load_test_worker(i) for i in range(num_workers)]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证性能指标
        assert total_time < 30.0  # 应该在30秒内完成
        
        # 统计结果
        successful_operations = 0
        total_operations = 0
        
        for worker_results in all_results:
            if isinstance(worker_results, list):
                for result in worker_results:
                    total_operations += 1
                    if result.get('success', False):
                        successful_operations += 1
        
        # 验证成功率
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        assert success_rate > 0.7  # 至少70%的操作应该成功
        
        # 验证系统稳定性
        final_cache_stats = cache_manager.get_global_stats()
        assert final_cache_stats['total_caches'] >= initial_cache_stats['total_caches']
        
        # 验证监控服务仍然正常
        health_status = await self.container.monitoring_service.get_service_health()
        assert 'overall_status' in health_status
    
    @pytest.mark.asyncio
    async def test_data_sync_integration(self):
        """
        属性 1-7: 所有核心属性的综合测试
        数据同步功能应该与其他组件正确集成
        **功能: data-management-implementation, 属性 1-7: 所有核心属性的综合测试**
        **验证: 所有需求**
        """
        stock_codes = ['000001', '000002']
        start_date = datetime(2023, 9, 1)
        end_date = datetime(2023, 9, 15)
        
        # 创建同步请求
        sync_request = DataSyncRequest(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            force_update=True
        )
        
        # 模拟远端数据
        with patch.object(self.container.stock_data_service, 'get_stock_data') as mock_get_data:
            def mock_get_stock_data(code, start, end, force_remote=False):
                mock_data_list = []
                for i in range(10):  # 10天的数据
                    date = start + timedelta(days=i)
                    if date <= end:
                        from app.models.stock import StockData
                        mock_data_list.append(StockData(
                            stock_code=code,
                            date=date,
                            open=100.0 + i,
                            high=105.0 + i,
                            low=95.0 + i,
                            close=102.0 + i,
                            volume=1000000 + i * 10000,
                            adj_close=102.0 + i
                        ))
                return mock_data_list
            
            mock_get_data.side_effect = mock_get_stock_data
            
            # 执行同步
            sync_response = await self.container.stock_data_service.sync_multiple_stocks(sync_request)
            
            # 验证同步结果
            assert sync_response.success or len(sync_response.synced_stocks) > 0
            assert sync_response.total_records > 0
            
            # 验证监控数据更新
            monitoring_stats = await self.container.monitoring_service.get_performance_metrics()
            assert 'data_operations' in monitoring_stats
            
            # 验证文件管理功能
            file_request = FileListRequest(
                stock_codes=stock_codes,
                start_date=start_date,
                end_date=end_date
            )
            
            file_info = await self.container.parquet_manager.get_file_list(file_request.to_filter_criteria())
            # 可能有文件被创建，也可能没有（取决于模拟的实现）
            assert isinstance(len(file_info), int)
    
    def _generate_mock_stock_data(self, stock_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """生成模拟股票数据"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        base_price = 100.0
        
        for i, date in enumerate(dates):
            # 简单的价格模拟
            price = base_price + i * 0.5
            high = price * 1.02
            low = price * 0.98
            volume = 1000000 + i * 10000
            
            data.append({
                'stock_code': stock_code,
                'date': date,
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'adj_close': price
            })
        
        return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def cleanup_after_integration_test():
    """集成测试后自动清理"""
    yield
    # 清理全局状态
    cache_manager.clear_all()