"""
错误处理和降级机制属性测试
Feature: stock-prediction-platform, 需求 1.4: 数据服务错误处理
"""

import asyncio
import tempfile
from datetime import datetime, timedelta

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.models.stock_simple import StockData
from app.services.data_service_simple import SimpleStockDataService


@composite
def stock_code_strategy(draw):
    """生成有效的股票代码"""
    number = draw(st.integers(min_value=1, max_value=999999))
    market = draw(st.sampled_from(['SH', 'SZ']))
    return f"{number:06d}.{market}"


def test_service_status_tracking():
    """测试服务状态跟踪功能"""
    async def run_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            data_service = SimpleStockDataService(
                data_path=temp_dir,
                remote_url="http://definitely-invalid-host-name-12345.invalid"  # 使用.invalid域名确保无效
            )
            
            # 执行健康检查
            health_status = await data_service.health_check()
            
            # 验证健康检查结果结构
            assert "timestamp" in health_status, "健康检查应该包含时间戳"
            assert "remote_service" in health_status, "健康检查应该包含远端服务状态"
            assert "local_storage" in health_status, "健康检查应该包含本地存储状态"
            assert "service_status" in health_status, "健康检查应该包含服务状态摘要"
            
            # 验证远端服务状态
            remote_status = health_status["remote_service"]
            assert remote_status["is_available"] == False, "无效URL的服务应该不可用"
            assert remote_status["error_message"] is not None, "应该有错误信息"
            
            # 验证本地存储状态
            local_status = health_status["local_storage"]
            assert local_status["is_available"] == True, "本地存储应该可用"
            assert local_status["stocks_count"] >= 0, "股票文件数量应该非负"
            
            # 验证服务状态摘要
            service_status = health_status["service_status"]
            assert service_status["current_status"] in ["正常", "不可用", "降级"], "服务状态应该是有效值"
            assert service_status["consecutive_failures"] >= 0, "连续失败次数应该非负"
    
    asyncio.run(run_test())


def test_service_degradation():
    """测试服务降级机制"""
    async def run_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            data_service = SimpleStockDataService(
                data_path=temp_dir,
                remote_url="http://definitely-invalid-host-name-12345.invalid"
            )
            
            stock_code = "000001.SZ"
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 10)
            
            # 先创建一些本地测试数据
            test_data = data_service.generate_mock_data(stock_code, start_date, end_date)
            data_service.save_to_local(test_data, stock_code, merge_with_existing=False)
            
            # 多次尝试获取数据，触发服务降级
            for i in range(5):
                data = await data_service.get_stock_data(stock_code, start_date, end_date)
                print(f"第{i+1}次尝试，连续失败次数: {data_service.status_tracker.consecutive_failures}")
            
            # 验证服务是否进入降级状态
            assert data_service.status_tracker.is_service_degraded(), "服务应该进入降级状态"
            
            # 验证降级状态下仍能获取本地数据
            data = await data_service.get_stock_data(stock_code, start_date, end_date)
            assert data is not None, "降级状态下应该能获取本地数据"
            assert len(data) > 0, "应该返回本地数据"
            
            # 验证服务状态摘要
            status_summary = data_service.get_service_status_summary()
            assert status_summary["is_degraded"] == True, "状态摘要应该显示服务降级"
            assert status_summary["current_status"] == "降级", "当前状态应该是降级"
    
    asyncio.run(run_test())


def test_fallback_to_local_data():
    """测试降级到本地数据的机制"""
    async def run_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            # 使用有效URL创建服务
            data_service = SimpleStockDataService(
                data_path=temp_dir,
                remote_url="http://192.168.3.62"
            )
            
            stock_code = "000001.SZ"
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 10)
            
            # 先创建本地数据
            test_data = data_service.generate_mock_data(stock_code, start_date, end_date)
            data_service.save_to_local(test_data, stock_code, merge_with_existing=False)
            
            # 验证本地数据存在
            local_file = data_service.get_local_data_path(stock_code)
            assert local_file.exists(), "本地文件应该存在"
            
            # 修改URL为无效地址模拟服务不可用
            original_url = data_service.remote_url
            data_service.remote_url = "http://definitely-invalid-host-name-12345.invalid"
            
            try:
                # 强制从远端获取（应该失败并降级到本地）
                data = await data_service.get_stock_data(
                    stock_code, start_date, end_date, force_remote=True
                )
                
                # 应该能够获取到本地数据
                assert data is not None, "应该能降级到本地数据"
                assert len(data) > 0, "本地数据应该有内容"
                
                # 验证数据的正确性
                for item in data:
                    assert isinstance(item, StockData), "每个项目应该是StockData类型"
                    assert item.stock_code == stock_code, "股票代码应该正确"
            
            finally:
                # 恢复原始URL
                data_service.remote_url = original_url
    
    asyncio.run(run_test())


@given(stock_code=stock_code_strategy())
@settings(max_examples=5, deadline=None)
def test_error_recovery_property(stock_code):
    """
    属性测试：错误恢复机制
    
    当服务从错误状态恢复时，应该能够正常工作
    
    验证：需求 1.4
    """
    def run_test():
        async def async_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                data_service = SimpleStockDataService(
                    data_path=temp_dir,
                    remote_url="http://definitely-invalid-host-name-12345.invalid"
                )
                
                start_date = datetime(2023, 1, 1)
                end_date = datetime(2023, 1, 10)
                
                # 创建本地数据
                test_data = data_service.generate_mock_data(stock_code, start_date, end_date)
                data_service.save_to_local(test_data, stock_code, merge_with_existing=False)
                
                # 触发多次失败
                for _ in range(5):
                    await data_service.get_stock_data(stock_code, start_date, end_date)
                
                # 验证服务降级
                assert data_service.status_tracker.is_service_degraded(), "服务应该降级"
                
                # 重置服务状态（模拟服务恢复）
                data_service.reset_service_status()
                
                # 验证状态重置
                assert not data_service.status_tracker.is_service_degraded(), "服务状态应该重置"
                assert data_service.status_tracker.consecutive_failures == 0, "连续失败次数应该重置"
                
                # 验证仍能获取本地数据
                data = await data_service.get_stock_data(stock_code, start_date, end_date)
                assert data is not None, "重置后应该能获取数据"
        
        asyncio.run(async_test())
    
    run_test()


def test_concurrent_error_handling():
    """测试并发情况下的错误处理"""
    async def run_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            data_service = SimpleStockDataService(
                data_path=temp_dir,
                remote_url="http://definitely-invalid-host-name-12345.invalid"
            )
            
            # 创建多个股票的本地数据
            stock_codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 10)
            
            for stock_code in stock_codes:
                test_data = data_service.generate_mock_data(stock_code, start_date, end_date)
                data_service.save_to_local(test_data, stock_code, merge_with_existing=False)
            
            # 并发获取数据
            tasks = [
                data_service.get_stock_data(stock_code, start_date, end_date)
                for stock_code in stock_codes
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 验证所有请求都能处理（即使远端服务不可用）
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"股票 {stock_codes[i]} 处理异常: {result}")
                else:
                    assert result is not None, f"股票 {stock_codes[i]} 应该返回数据"
            
            # 验证服务状态跟踪正常工作
            status_summary = data_service.get_service_status_summary()
            assert status_summary["consecutive_failures"] > 0, "应该记录失败次数"
    
    asyncio.run(run_test())


def test_service_status_history():
    """测试服务状态历史记录"""
    async def run_test():
        with tempfile.TemporaryDirectory() as temp_dir:
            data_service = SimpleStockDataService(
                data_path=temp_dir,
                remote_url="http://definitely-invalid-host-name-12345.invalid"
            )
            
            # 执行多次健康检查
            for i in range(5):
                await data_service.health_check()
            
            # 获取状态历史
            status_summary = data_service.get_service_status_summary()
            recent_statuses = status_summary["recent_statuses"]
            
            # 验证历史记录
            assert len(recent_statuses) > 0, "应该有状态历史记录"
            assert len(recent_statuses) <= 10, "最多保留10条记录"
            
            # 验证每条记录的结构
            for status in recent_statuses:
                assert "timestamp" in status, "每条记录应该有时间戳"
                assert "is_available" in status, "每条记录应该有可用性状态"
                assert "response_time_ms" in status, "每条记录应该有响应时间"
                assert isinstance(status["is_available"], bool), "可用性应该是布尔值"
            
            # 验证成功率计算
            success_rate = status_summary["recent_success_rate"]
            assert 0.0 <= success_rate <= 1.0, "成功率应该在0-1之间"
    
    asyncio.run(run_test())