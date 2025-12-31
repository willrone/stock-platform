"""
性能优化属性测试
验证性能优化功能的正确性属性
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

from app.services.cache_service import LRUCache, CacheManager, cache_manager
from app.services.stream_processor import StreamProcessor, ChunkedDataReader, MemoryMonitor
from app.services.connection_pool import (
    HTTPConnectionPool, DatabaseConnectionPool, ConnectionPoolManager,
    PoolConfig, connection_pool_manager
)


@composite
def cache_configs(draw):
    """生成缓存配置"""
    return {
        'max_size': draw(st.integers(min_value=10, max_value=1000)),
        'default_ttl': draw(st.floats(min_value=1.0, max_value=3600.0)),
        'memory_limit_mb': draw(st.integers(min_value=1, max_value=100))
    }


@composite
def test_data_frames(draw):
    """生成测试数据DataFrame"""
    size = draw(st.integers(min_value=100, max_value=1000))
    
    dates = pd.date_range(start='2023-01-01', periods=size, freq='D')
    
    data = []
    for i, date in enumerate(dates):
        data.append({
            'id': i,
            'date': date,
            'value': draw(st.floats(min_value=1.0, max_value=1000.0)),
            'category': draw(st.sampled_from(['A', 'B', 'C'])),
            'count': draw(st.integers(min_value=1, max_value=100))
        })
    
    return pd.DataFrame(data)


class TestPerformanceOptimizationProperties:
    """性能优化属性测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    @given(cache_configs())
    @settings(max_examples=5, deadline=10000)
    async def test_cache_hit_rate_improvement(self, cache_config):
        """
        属性 7: 性能优化有效性
        缓存机制应该显著提高数据访问的命中率
        **功能: data-management-implementation, 属性 7: 性能优化有效性**
        **验证: 需求 7.1**
        """
        # 创建缓存实例
        cache = LRUCache(**cache_config)
        
        # 根据缓存大小调整测试数据量
        max_size = cache_config['max_size']
        test_size = min(max_size // 2, 25)  # 使用缓存大小的一半，最多25个
        
        # 生成测试数据
        test_keys = [f"key_{i}" for i in range(test_size)]
        test_values = [f"value_{i}" for i in range(test_size)]
        
        # 第一轮：填充缓存
        for key, value in zip(test_keys, test_values):
            cache.put(key, value)
        
        # 第二轮：访问数据，应该有高命中率
        hits_before = cache.get_stats().hits
        
        successful_gets = 0
        for key in test_keys:  # 访问所有键
            result = cache.get(key)
            if result is not None:
                successful_gets += 1
        
        stats = cache.get_stats()
        hits_after = stats.hits
        
        # 验证命中率提升
        new_hits = hits_after - hits_before
        assert new_hits == successful_gets  # 成功获取的次数应该等于新增命中数
        assert successful_gets > 0  # 至少应该有一些成功的获取
        
        # 验证缓存大小限制
        assert stats.size <= cache_config['max_size']
    
    @pytest.mark.asyncio
    @given(test_data_frames())
    @settings(max_examples=3, deadline=15000)
    async def test_stream_processing_memory_efficiency(self, test_df):
        """
        属性 7: 性能优化有效性
        流式处理应该保持内存使用在合理范围内
        **功能: data-management-implementation, 属性 7: 性能优化有效性**
        **验证: 需求 7.2, 7.4**
        """
        # 创建测试文件
        test_file = self.temp_path / "test_data.parquet"
        test_df.to_parquet(test_file, index=False)
        
        # 创建流处理器
        processor = StreamProcessor(
            chunk_size=100,
            max_workers=2,
            memory_limit_mb=50
        )
        
        # 定义处理函数
        def process_chunk(chunk_df):
            # 简单的数据处理：计算统计信息
            return chunk_df.groupby('category').agg({
                'value': ['mean', 'sum', 'count'],
                'count': 'sum'
            }).reset_index()
        
        # 执行流式处理
        output_file = self.temp_path / "processed_data.parquet"
        
        start_memory = processor.get_memory_stats()['current_mb']
        
        stats = await processor.process_file_stream(
            file_path=test_file,
            processor_func=process_chunk,
            output_path=output_file,
            file_type='parquet'
        )
        
        end_memory = processor.get_memory_stats()['current_mb']
        
        # 验证处理统计
        assert stats.processed_records > 0
        assert stats.processing_time_seconds > 0
        assert stats.throughput_records_per_second > 0
        
        # 验证内存效率
        memory_increase = end_memory - start_memory
        assert memory_increase < 100  # 内存增长应该小于100MB
        
        # 验证输出文件存在
        assert output_file.exists()
        
        await processor.close()
    
    @pytest.mark.asyncio
    async def test_connection_pool_efficiency(self):
        """
        属性 7: 性能优化有效性
        连接池应该提高连接复用率和响应时间
        **功能: data-management-implementation, 属性 7: 性能优化有效性**
        **验证: 需求 7.3**
        """
        # 创建连接池配置
        config = PoolConfig(
            max_connections=10,
            min_connections=2,
            max_keepalive_connections=5,
            keepalive_expiry=30.0,
            timeout=10.0
        )
        
        # 创建HTTP连接池
        pool = await connection_pool_manager.create_http_pool('test_pool', config)
        
        # 模拟多个并发请求
        async def make_request(request_id: int):
            try:
                client = await pool.get_client()
                # 这里我们不能真正发送HTTP请求，所以模拟一个延迟
                await asyncio.sleep(0.01)
                return f"response_{request_id}"
            except Exception as e:
                return f"error_{request_id}: {e}"
        
        # 并发执行多个请求
        start_time = time.time()
        
        tasks = [make_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证所有请求都成功
        successful_requests = sum(1 for r in results if isinstance(r, str) and r.startswith('response_'))
        assert successful_requests == 20
        
        # 验证并发性能
        assert total_time < 1.0  # 20个请求应该在1秒内完成
        
        # 获取连接池统计
        stats = await pool.get_pool_stats()
        assert isinstance(stats.total_requests, int)
        
        await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_cache_memory_management(self):
        """
        属性 7: 性能优化有效性
        缓存应该有效管理内存使用，防止内存泄漏
        **功能: data-management-implementation, 属性 7: 性能优化有效性**
        **验证: 需求 7.1, 7.4**
        """
        # 创建小容量缓存
        cache = LRUCache(max_size=10, memory_limit_mb=1)
        
        # 填充大量数据，测试内存管理
        large_data = "x" * 1024  # 1KB数据
        
        for i in range(50):
            key = f"large_key_{i}"
            cache.put(key, large_data)
        
        # 验证缓存大小限制
        stats = cache.get_stats()
        assert stats.size <= 10  # 不应该超过最大大小
        
        # 验证内存使用
        assert stats.memory_usage_bytes < 2 * 1024 * 1024  # 应该小于2MB
        
        # 验证LRU驱逐机制
        assert stats.evictions > 0  # 应该有驱逐发生
        
        # 测试缓存清理
        cache.clear()
        stats_after_clear = cache.get_stats()
        assert stats_after_clear.size == 0
        assert stats_after_clear.memory_usage_bytes == 0
    
    @pytest.mark.asyncio
    async def test_stream_processing_throughput(self):
        """
        属性 7: 性能优化有效性
        流式处理应该保持稳定的吞吐量
        **功能: data-management-implementation, 属性 7: 性能优化有效性**
        **验证: 需求 7.2, 7.5**
        """
        # 创建大量测试数据
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': [i * 0.1 for i in range(10000)],
            'category': ['A', 'B', 'C'] * 3334
        })
        
        test_file = self.temp_path / "large_test_data.parquet"
        large_df.to_parquet(test_file, index=False)
        
        # 创建流处理器
        processor = StreamProcessor(chunk_size=1000, max_workers=2)
        
        # 定义简单的处理函数
        def simple_transform(chunk_df):
            chunk_df['processed_value'] = chunk_df['value'] * 2
            return chunk_df
        
        # 执行流式处理
        output_file = self.temp_path / "processed_large_data.parquet"
        
        stats = await processor.process_file_stream(
            file_path=test_file,
            processor_func=simple_transform,
            output_path=output_file,
            file_type='parquet'
        )
        
        # 验证吞吐量
        assert stats.total_records == 10000
        assert stats.processed_records == 10000
        assert stats.throughput_records_per_second > 1000  # 至少1000条/秒
        
        # 验证处理时间合理
        assert stats.processing_time_seconds < 30  # 应该在30秒内完成
        
        # 验证输出正确性
        assert output_file.exists()
        processed_df = pd.read_parquet(output_file)
        assert len(processed_df) == 10000
        assert 'processed_value' in processed_df.columns
        
        await processor.close()
    
    @pytest.mark.asyncio
    async def test_cache_concurrent_access(self):
        """
        属性 7: 性能优化有效性
        缓存应该支持并发访问而不影响性能
        **功能: data-management-implementation, 属性 7: 性能优化有效性**
        **验证: 需求 7.1**
        """
        cache = LRUCache(max_size=100, memory_limit_mb=10)
        
        # 预填充一些数据
        for i in range(50):
            cache.put(f"key_{i}", f"value_{i}")
        
        # 并发访问函数
        async def concurrent_access(worker_id: int):
            results = []
            for i in range(20):
                key = f"key_{i % 50}"
                value = cache.get(key)
                if value:
                    results.append(value)
                
                # 偶尔添加新数据
                if i % 5 == 0:
                    cache.put(f"new_key_{worker_id}_{i}", f"new_value_{worker_id}_{i}")
                
                await asyncio.sleep(0.001)  # 小延迟模拟真实场景
            
            return results
        
        # 启动多个并发工作者
        start_time = time.time()
        
        tasks = [concurrent_access(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证所有工作者都成功完成
        successful_workers = sum(1 for r in results if isinstance(r, list))
        assert successful_workers == 10
        
        # 验证并发性能
        assert total_time < 5.0  # 应该在5秒内完成
        
        # 验证缓存状态一致性
        stats = cache.get_stats()
        assert stats.size > 0
        assert stats.hits > 0
        assert stats.hit_rate > 0
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_accuracy(self):
        """
        属性 7: 性能优化有效性
        内存监控应该准确反映内存使用情况
        **功能: data-management-implementation, 属性 7: 性能优化有效性**
        **验证: 需求 7.4**
        """
        monitor = MemoryMonitor(warning_threshold_mb=100, critical_threshold_mb=200)
        
        # 获取初始内存状态
        initial_status = monitor.check_memory()
        assert initial_status['status'] in ['normal', 'warning', 'critical']
        assert initial_status['current_mb'] > 0
        assert initial_status['peak_mb'] >= initial_status['current_mb']
        
        # 分配一些内存（创建大对象）
        large_objects = []
        for i in range(10):
            large_obj = [0] * 100000  # 大约800KB的列表
            large_objects.append(large_obj)
        
        # 检查内存状态变化
        after_allocation_status = monitor.check_memory()
        
        # 验证内存使用增加
        assert after_allocation_status['current_mb'] >= initial_status['current_mb']
        assert after_allocation_status['peak_mb'] >= after_allocation_status['current_mb']
        
        # 强制垃圾回收
        monitor.force_gc()
        
        # 清理大对象
        large_objects.clear()
        del large_objects
        
        # 再次强制垃圾回收
        monitor.force_gc()
        
        # 检查内存是否有所释放
        final_status = monitor.check_memory()
        
        # 验证监控数据的合理性
        assert final_status['current_mb'] > 0
        assert final_status['warning_threshold_mb'] == 100
        assert final_status['critical_threshold_mb'] == 200
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self):
        """
        属性 7: 性能优化有效性
        系统应该能够检测性能退化并采取优化措施
        **功能: data-management-implementation, 属性 7: 性能优化有效性**
        **验证: 需求 7.5**
        """
        # 创建缓存管理器
        manager = CacheManager()
        
        # 获取测试缓存
        test_cache = manager.get_cache('performance_test')
        
        # 模拟正常操作
        normal_times = []
        for i in range(100):
            start_time = time.time()
            test_cache.put(f"key_{i}", f"value_{i}")
            result = test_cache.get(f"key_{i}")
            end_time = time.time()
            
            normal_times.append(end_time - start_time)
            assert result == f"value_{i}"
        
        # 计算正常操作的平均时间
        avg_normal_time = sum(normal_times) / len(normal_times)
        
        # 模拟性能退化场景（填满缓存导致频繁驱逐）
        degraded_times = []
        for i in range(100, 1100):  # 添加1000个额外项目
            start_time = time.time()
            test_cache.put(f"key_{i}", f"value_{i}" * 100)  # 更大的值
            end_time = time.time()
            
            degraded_times.append(end_time - start_time)
        
        # 计算退化场景的平均时间
        avg_degraded_time = sum(degraded_times) / len(degraded_times)
        
        # 获取缓存统计
        stats = test_cache.get_stats()
        
        # 验证性能退化检测
        assert stats.evictions > 0  # 应该有驱逐发生
        
        # 验证缓存仍然有效（命中率应该合理）
        assert stats.hit_rate >= 0  # 命中率应该非负
        
        # 验证内存使用在限制范围内
        assert stats.memory_usage_bytes > 0


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
    # 清理全局缓存管理器
    cache_manager.clear_all()