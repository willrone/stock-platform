"""
服务容器属性测试
验证服务容器的正确性属性
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.core.container import ServiceContainer, get_container, cleanup_container
from app.services.data import SimpleDataService
from app.services.data.sftp_sync_service import SFTPSyncService
from app.services.prediction import TechnicalIndicatorCalculator


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
        assert container.sftp_sync_service is not None

        await container.cleanup()

    @pytest.mark.asyncio
    @given(container_operations())
    @settings(max_examples=20, deadline=15000)
    async def test_container_state_consistency(self, operations):
        """
        属性: 对于任何操作序列，容器状态应该保持一致
        """
        container = ServiceContainer()
        initialized = False

        try:
            for operation in operations:
                if operation == 'initialize' and not initialized:
                    await container.initialize()
                    initialized = True

                    assert isinstance(container.data_service, SimpleDataService)
                    assert isinstance(container.indicators_service, TechnicalIndicatorCalculator)
                    assert isinstance(container.sftp_sync_service, SFTPSyncService)

                elif operation == 'get_service' and initialized:
                    service1 = container.data_service
                    service2 = container.data_service
                    assert service1 is service2

                elif operation == 'cleanup' and initialized:
                    await container.cleanup()
                    initialized = False

                    with pytest.raises(RuntimeError):
                        _ = container.data_service

        finally:
            if initialized:
                await container.cleanup()

    @pytest.mark.asyncio
    async def test_global_container_singleton(self):
        """
        属性: 全局容器是单例的
        """
        await cleanup_container()

        container1 = await get_container()
        container2 = await get_container()
        container3 = await get_container()

        assert container1 is container2
        assert container2 is container3

        assert container1.data_service is container2.data_service
        assert container1.indicators_service is container2.indicators_service

        await cleanup_container()

    @pytest.mark.asyncio
    async def test_container_error_handling(self):
        """
        属性: 容器错误处理正确
        未初始化的容器应该抛出适当的错误
        """
        container = ServiceContainer()

        with pytest.raises(RuntimeError, match="服务容器未初始化"):
            _ = container.data_service

        with pytest.raises(RuntimeError, match="服务容器未初始化"):
            _ = container.indicators_service

        with pytest.raises(RuntimeError, match="服务容器未初始化"):
            _ = container.sftp_sync_service

        await container.initialize()

        assert container.data_service is not None
        assert container.indicators_service is not None
        assert container.sftp_sync_service is not None

        await container.cleanup()

    @pytest.mark.asyncio
    async def test_container_cleanup_safety(self):
        """
        属性: 容器清理是安全的
        多次清理不应该导致错误
        """
        container = ServiceContainer()
        await container.initialize()

        assert container.data_service is not None

        await container.cleanup()
        await container.cleanup()
        await container.cleanup()

        with pytest.raises(RuntimeError):
            _ = container.data_service

    @pytest.mark.asyncio
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, deadline=15000)
    async def test_concurrent_container_access(self, num_tasks):
        """
        属性: 并发访问容器是安全的
        """
        await cleanup_container()

        async def access_container():
            container = await get_container()
            assert container.data_service is not None
            assert container.indicators_service is not None
            return container

        tasks = [access_container() for _ in range(num_tasks)]
        containers = await asyncio.gather(*tasks)

        first_container = containers[0]
        for container in containers[1:]:
            assert container is first_container

        await cleanup_container()


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
