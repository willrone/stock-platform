"""
服务容器属性测试
验证服务容器的正确性属性
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.core.container import ServiceContainer, get_container, cleanup_container
from app.services.data import DataService as StockDataService
from app.services.prediction import TechnicalIndicatorCalculator
from app.services.data import ParquetManager


@composite
def container_operations(draw):
    """生成容器操作序列"""
    operations = draw(st.lists(
        st.sampled_from(['initialize', 'get_service', 'cleanup']),
        min_size=1,
        max_size=10
    ))
    return operations


class TestServiceContainerProperties:
    """服务容器属性测试类"""
    
    @pytest.mark.asyncio
    async def test_container_initialization_idempotent(self):
        """
        属性: 容器初始化是幂等的
        多次初始化应该产生相同的结果
        """
        container = ServiceContainer()
        
        # 多次初始化
        await container.initialize()
        await container.initialize()
        await container.initialize()
        
        # 验证服务可用
        assert container.data_service is not None
        assert container.indicators_service is not None
        assert container.parquet_manager is not None
        
        await container.cleanup()
    
    @pytest.mark.asyncio
    @given(container_operations())
    @settings(max_examples=50, deadline=10000)
    async def test_container_state_consistency(self, operations):
        """
        属性 1: API路由真实服务调用
        对于任何操作序列，容器状态应该保持一致
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        **验证: 需求 1.1, 1.2, 1.3, 1.4, 1.5**
        """
        container = ServiceContainer()
        initialized = False
        
        try:
            for operation in operations:
                if operation == 'initialize' and not initialized:
                    await container.initialize()
                    initialized = True
                    
                    # 验证所有服务都已正确初始化
                    assert isinstance(container.data_service, StockDataService)
                    assert isinstance(container.indicators_service, TechnicalIndicatorCalculator)
                    assert isinstance(container.parquet_manager, ParquetManager)
                    
                elif operation == 'get_service' and initialized:
                    # 验证服务获取的一致性
                    service1 = container.data_service
                    service2 = container.data_service
                    assert service1 is service2  # 应该是同一个实例
                    
                elif operation == 'cleanup' and initialized:
                    await container.cleanup()
                    initialized = False
                    
                    # 验证清理后的状态
                    with pytest.raises(RuntimeError):
                        _ = container.data_service
        
        finally:
            if initialized:
                await container.cleanup()
    
    @pytest.mark.asyncio
    async def test_global_container_singleton(self):
        """
        属性: 全局容器是单例的
        多次获取全局容器应该返回同一个实例
        """
        # 清理可能存在的容器
        await cleanup_container()
        
        # 获取多个容器实例
        container1 = await get_container()
        container2 = await get_container()
        container3 = await get_container()
        
        # 验证是同一个实例
        assert container1 is container2
        assert container2 is container3
        
        # 验证服务也是同一个实例
        assert container1.data_service is container2.data_service
        assert container1.indicators_service is container2.indicators_service
        
        await cleanup_container()
    
    @pytest.mark.asyncio
    async def test_service_dependencies_correct(self):
        """
        属性: 服务依赖关系正确
        复合服务应该正确依赖基础服务
        """
        container = ServiceContainer()
        await container.initialize()
        
        # 验证基础服务存在
        assert container.data_service is not None
        assert container.parquet_manager is not None
        
        # 验证复合服务正确依赖基础服务
        sync_engine = container.data_sync_engine
        monitoring_service = container.monitoring_service
        
        assert sync_engine is not None
        assert monitoring_service is not None
        
        # 验证依赖关系（通过检查内部属性）
        assert hasattr(sync_engine, 'data_service')
        assert hasattr(sync_engine, 'parquet_manager')
        assert hasattr(monitoring_service, 'data_service')
        
        await container.cleanup()
    
    @pytest.mark.asyncio
    async def test_container_error_handling(self):
        """
        属性: 容器错误处理正确
        未初始化的容器应该抛出适当的错误
        """
        container = ServiceContainer()
        
        # 验证未初始化时的错误
        with pytest.raises(RuntimeError, match="服务容器未初始化"):
            _ = container.data_service
        
        with pytest.raises(RuntimeError, match="服务容器未初始化"):
            _ = container.indicators_service
        
        with pytest.raises(RuntimeError, match="服务容器未初始化"):
            _ = container.parquet_manager
        
        # 初始化后应该正常工作
        await container.initialize()
        
        # 现在应该可以正常访问
        assert container.data_service is not None
        assert container.indicators_service is not None
        assert container.parquet_manager is not None
        
        await container.cleanup()
    
    @pytest.mark.asyncio
    async def test_container_cleanup_safety(self):
        """
        属性: 容器清理是安全的
        多次清理不应该导致错误
        """
        container = ServiceContainer()
        await container.initialize()
        
        # 验证初始化成功
        assert container.data_service is not None
        
        # 多次清理应该是安全的
        await container.cleanup()
        await container.cleanup()
        await container.cleanup()
        
        # 清理后访问应该抛出错误
        with pytest.raises(RuntimeError):
            _ = container.data_service
    
    @pytest.mark.asyncio
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=15000)
    async def test_concurrent_container_access(self, num_tasks):
        """
        属性: 并发访问容器是安全的
        多个并发任务访问容器应该是安全的
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        """
        await cleanup_container()
        
        async def access_container():
            container = await get_container()
            # 验证服务可用
            assert container.data_service is not None
            assert container.indicators_service is not None
            return container
        
        # 并发访问容器
        tasks = [access_container() for _ in range(num_tasks)]
        containers = await asyncio.gather(*tasks)
        
        # 验证所有容器都是同一个实例
        first_container = containers[0]
        for container in containers[1:]:
            assert container is first_container
        
        await cleanup_container()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
    # 清理代码可以在这里添加
    # 注意：这里不能使用await，因为这是同步fixture