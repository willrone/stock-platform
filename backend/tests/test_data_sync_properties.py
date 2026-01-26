"""
数据同步引擎属性测试
验证数据同步完整性属性
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.services.data import SimpleDataService as StockDataService
from app.services.data.data_validator import DataValidator
from app.services.data.parquet_manager import ParquetManager

# DataSyncEngine 和 SyncProgressTracker 已移除，使用占位符
# 相关测试可能需要跳过或重构
try:
    from app.services.data.incremental_updater import IncrementalUpdater as DataSyncEngine
    SyncProgressTracker = None  # 占位符
except ImportError:
    DataSyncEngine = None
    SyncProgressTracker = None
from app.models.stock_simple import StockData
from app.models.sync_models import (
    BatchSyncRequest, SyncOptions, SyncMode, SyncStatus
)


@composite
def stock_codes(draw):
    """生成股票代码"""
    market = draw(st.sampled_from(['SZ', 'SH']))
    code = draw(st.integers(min_value=1, max_value=999999))
    return f"{code:06d}.{market}"


@composite
def batch_sync_requests(draw):
    """生成批量同步请求"""
    codes = draw(st.lists(stock_codes(), min_size=1, max_size=5))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=draw(st.integers(min_value=1, max_value=365)))
    
    return BatchSyncRequest(
        stock_codes=codes,
        start_date=start_date,
        end_date=end_date,
        force_update=draw(st.booleans()),
        sync_mode=draw(st.sampled_from(list(SyncMode))),
        max_concurrent=draw(st.integers(min_value=1, max_value=5)),
        retry_count=draw(st.integers(min_value=1, max_value=3))
    )


@composite
def stock_data_lists(draw):
    """生成股票数据列表"""
    stock_code = draw(stock_codes())
    size = draw(st.integers(min_value=1, max_value=20))
    
    base_date = datetime(2023, 1, 1)
    data_list = []
    
    for i in range(size):
        open_price = draw(st.floats(min_value=1.0, max_value=100.0))
        high_price = draw(st.floats(min_value=open_price, max_value=open_price * 1.2))
        low_price = draw(st.floats(min_value=open_price * 0.8, max_value=open_price))
        close_price = draw(st.floats(min_value=low_price, max_value=high_price))
        
        data = StockData(
            stock_code=stock_code,
            date=base_date + timedelta(days=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=draw(st.integers(min_value=1000, max_value=1000000)),
            adj_close=close_price
        )
        data_list.append(data)
    
    return data_list


class TestDataSyncEngineProperties:
    """数据同步引擎属性测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟服务
        self.mock_data_service = AsyncMock(spec=StockDataService)
        self.parquet_manager = ParquetManager(self.temp_dir)
        
        # 创建同步引擎
        self.sync_engine = DataSyncEngine(
            data_service=self.mock_data_service,
            parquet_manager=self.parquet_manager
        )
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    @given(batch_sync_requests())
    @settings(max_examples=15, deadline=20000)
    async def test_batch_sync_completeness(self, request):
        """
        属性 4: 数据同步完整性
        批量同步应该处理所有股票代码，结果应该准确反映每只股票的同步状态
        **功能: data-management-implementation, 属性 4: 数据同步完整性**
        **验证: 需求 4.1, 4.2, 4.3, 4.4, 4.5**
        """
        # 模拟数据服务返回
        async def mock_get_stock_data(stock_code, start_date, end_date, force_remote=False):
            # 模拟部分成功，部分失败
            if stock_code.endswith('1.SZ'):
                return None  # 模拟失败
            else:
                return [
                    StockData(
                        stock_code=stock_code,
                        date=start_date,
                        open=10.0,
                        high=11.0,
                        low=9.0,
                        close=10.5,
                        volume=1000000,
                        adj_close=10.5
                    )
                ]
        
        self.mock_data_service.get_stock_data.side_effect = mock_get_stock_data
        
        # 执行批量同步
        result = await self.sync_engine.sync_stocks_batch(request)
        
        # 验证同步完整性
        assert result.total_stocks == len(request.stock_codes)
        assert result.success_count + result.failure_count == len(request.stock_codes)
        
        # 验证每只股票都有结果
        all_results = result.successful_syncs + result.failed_syncs
        result_stock_codes = {r.stock_code for r in all_results}
        request_stock_codes = set(request.stock_codes)
        assert result_stock_codes == request_stock_codes
        
        # 验证同步状态一致性
        for sync_result in result.successful_syncs:
            assert sync_result.success is True
            assert sync_result.records_synced > 0
        
        for sync_result in result.failed_syncs:
            assert sync_result.success is False
            assert sync_result.error_message is not None
    
    @pytest.mark.asyncio
    @given(st.lists(stock_codes(), min_size=1, max_size=3))
    @settings(max_examples=10, deadline=15000)
    async def test_sync_progress_tracking_accuracy(self, stock_codes_list):
        """
        属性: 同步进度跟踪准确性
        进度跟踪应该准确反映同步状态
        **功能: data-management-implementation, 属性 4: 数据同步完整性**
        """
        # 创建同步请求
        request = BatchSyncRequest(
            stock_codes=stock_codes_list,
            max_concurrent=1  # 串行执行便于测试
        )
        
        # 模拟数据服务
        call_count = 0
        async def mock_get_stock_data(stock_code, start_date, end_date, force_remote=False):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # 模拟处理时间
            return [
                StockData(
                    stock_code=stock_code,
                    date=datetime.now(),
                    open=10.0,
                    high=11.0,
                    low=9.0,
                    close=10.5,
                    volume=1000000,
                    adj_close=10.5
                )
            ]
        
        self.mock_data_service.get_stock_data.side_effect = mock_get_stock_data
        
        # 启动同步任务
        sync_task = asyncio.create_task(self.sync_engine.sync_stocks_batch(request))
        
        # 等待一段时间让同步开始
        await asyncio.sleep(0.05)
        
        # 检查进度
        progress_checks = []
        for _ in range(3):
            await asyncio.sleep(0.1)
            # 获取所有进度记录
            for sync_id in self.sync_engine.progress_tracker._progress_data:
                progress = await self.sync_engine.get_sync_progress(sync_id)
                if progress:
                    progress_checks.append(progress)
        
        # 等待同步完成
        result = await sync_task
        
        # 验证进度跟踪
        if progress_checks:
            # 验证进度递增
            progress_percentages = [p.progress_percentage for p in progress_checks]
            assert all(p >= 0 for p in progress_percentages)
            assert all(p <= 100 for p in progress_percentages)
            
            # 验证最终状态
            final_progress = progress_checks[-1] if progress_checks else None
            if final_progress:
                assert final_progress.total_stocks == len(stock_codes_list)
    
    @pytest.mark.asyncio
    @given(stock_data_lists())
    @settings(max_examples=10, deadline=12000)
    async def test_single_stock_sync_consistency(self, stock_data):
        """
        属性: 单股票同步一致性
        单股票同步结果应该与实际操作一致
        **功能: data-management-implementation, 属性 4: 数据同步完整性**
        """
        if not stock_data:
            return
        
        stock_code = stock_data[0].stock_code
        
        # 模拟数据服务返回
        self.mock_data_service.get_stock_data.return_value = stock_data
        
        # 执行单股票同步
        options = SyncOptions(force_update=True, sync_mode=SyncMode.FULL)
        result = await self.sync_engine.sync_single_stock(stock_code, options)
        
        # 验证同步结果
        assert result.stock_code == stock_code
        assert result.success is True
        assert result.records_synced == len(stock_data)
        assert result.start_time <= result.end_time
        
        # 验证数据确实被保存
        saved_data = self.parquet_manager.load_stock_data(
            stock_code,
            stock_data[0].date,
            stock_data[-1].date
        )
        assert len(saved_data) == len(stock_data)
    
    @pytest.mark.asyncio
    @given(st.lists(stock_codes(), min_size=2, max_size=4))
    @settings(max_examples=8, deadline=15000)
    async def test_concurrent_sync_safety(self, stock_codes_list):
        """
        属性: 并发同步安全性
        并发同步不应该导致数据竞争或不一致状态
        **功能: data-management-implementation, 属性 4: 数据同步完整性**
        """
        # 模拟数据服务
        async def mock_get_stock_data(stock_code, start_date, end_date, force_remote=False):
            await asyncio.sleep(0.1)  # 模拟网络延迟
            return [
                StockData(
                    stock_code=stock_code,
                    date=datetime.now(),
                    open=10.0,
                    high=11.0,
                    low=9.0,
                    close=10.5,
                    volume=1000000,
                    adj_close=10.5
                )
            ]
        
        self.mock_data_service.get_stock_data.side_effect = mock_get_stock_data
        
        # 创建多个并发同步请求
        requests = [
            BatchSyncRequest(
                stock_codes=[code],
                max_concurrent=1
            )
            for code in stock_codes_list
        ]
        
        # 并发执行同步
        tasks = [self.sync_engine.sync_stocks_batch(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        # 验证所有同步都成功
        for result in results:
            assert result.success is True
            assert result.success_count == 1
            assert result.failure_count == 0
        
        # 验证没有数据丢失或重复
        all_synced_codes = []
        for result in results:
            for sync_result in result.successful_syncs:
                all_synced_codes.append(sync_result.stock_code)
        
        assert len(all_synced_codes) == len(stock_codes_list)
        assert set(all_synced_codes) == set(stock_codes_list)
    
    @pytest.mark.asyncio
    @given(batch_sync_requests())
    @settings(max_examples=8, deadline=12000)
    async def test_sync_history_consistency(self, request):
        """
        属性: 同步历史一致性
        同步历史应该准确记录所有同步操作
        **功能: data-management-implementation, 属性 4: 数据同步完整性**
        """
        # 记录初始历史数量
        initial_history_count = len(self.sync_engine.get_sync_history())
        
        # 模拟数据服务
        self.mock_data_service.get_stock_data.return_value = [
            StockData(
                stock_code="test",
                date=datetime.now(),
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.5,
                volume=1000000,
                adj_close=10.5
            )
        ]
        
        # 执行同步
        result = await self.sync_engine.sync_stocks_batch(request)
        
        # 验证历史记录
        history = self.sync_engine.get_sync_history()
        assert len(history) == initial_history_count + 1
        
        # 验证最新记录
        latest_entry = history[-1]
        assert latest_entry.sync_id == result.sync_id
        assert latest_entry.request.stock_codes == request.stock_codes
        assert latest_entry.result.total_stocks == result.total_stocks
    
    @pytest.mark.asyncio
    async def test_progress_tracker_thread_safety(self):
        """
        属性: 进度跟踪器线程安全
        并发访问进度跟踪器应该是安全的
        **功能: data-management-implementation, 属性 4: 数据同步完整性**
        """
        tracker = SyncProgressTracker()
        sync_id = "test_sync"
        
        # 创建进度
        await tracker.create_progress(sync_id, 10)
        
        # 并发更新进度
        async def update_progress(completed):
            await tracker.update_progress(sync_id, completed=completed)
        
        # 并发执行更新
        tasks = [update_progress(i) for i in range(1, 6)]
        await asyncio.gather(*tasks)
        
        # 验证最终状态
        progress = await tracker.get_progress(sync_id)
        assert progress is not None
        assert progress.completed_stocks >= 1  # 至少有一个更新生效
        assert progress.completed_stocks <= 5   # 不超过最大值


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
    # 清理代码可以在这里添加
    # 清理临时文件和资源
    pass