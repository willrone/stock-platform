"""
数据管理API属性测试
验证数据管理API端点调用真实服务的正确性属性
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
from fastapi.testclient import TestClient

from app.main import create_application
from app.services.data import ParquetManager
from app.services.data import DataSyncEngine
from app.models.file_management import FileFilters, IntegrityStatus
from app.models.sync_models import BatchSyncRequest, SyncMode


@composite
def stock_codes(draw):
    """生成股票代码"""
    market = draw(st.sampled_from(['SZ', 'SH']))
    code = draw(st.integers(min_value=1, max_value=999999))
    return f"{code:06d}.{market}"


@composite
def file_filters(draw):
    """生成文件过滤条件"""
    return {
        "stock_code": draw(st.one_of(st.none(), stock_codes())),
        "start_date": draw(st.one_of(st.none(), st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime.now()
        ))),
        "end_date": draw(st.one_of(st.none(), st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime.now()
        ))),
        "min_size": draw(st.one_of(st.none(), st.integers(min_value=1000, max_value=1000000))),
        "max_size": draw(st.one_of(st.none(), st.integers(min_value=1000000, max_value=10000000))),
        "limit": draw(st.integers(min_value=1, max_value=100)),
        "offset": draw(st.integers(min_value=0, max_value=50))
    }


@composite
def sync_requests(draw):
    """生成同步请求"""
    codes = draw(st.lists(stock_codes(), min_size=1, max_size=5))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=draw(st.integers(min_value=1, max_value=365)))
    
    return {
        "stock_codes": codes,
        "start_date": start_date,
        "end_date": end_date,
        "force_update": draw(st.booleans()),
        "sync_mode": draw(st.sampled_from(["incremental", "full"])),
        "max_concurrent": draw(st.integers(min_value=1, max_value=5)),
        "retry_count": draw(st.integers(min_value=1, max_value=3))
    }


class TestDataManagementAPIProperties:
    """数据管理API属性测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试应用，禁用限流中间件
        self.test_app = create_application()
        self.test_app.user_middleware = [
            middleware for middleware in self.test_app.user_middleware
            if 'RateLimitMiddleware' not in str(middleware.cls)
        ]
        
        self.client = TestClient(self.test_app)
        
        # 创建模拟服务
        self.mock_parquet_manager = MagicMock(spec=ParquetManager)
        self.mock_sync_engine = AsyncMock(spec=DataSyncEngine)
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    @given(file_filters())
    @settings(max_examples=10, deadline=5000)
    async def test_data_files_api_calls_real_parquet_manager(self, filters):
        """
        属性 1: API路由真实服务调用
        数据文件列表API应该调用真实的Parquet管理器而不是返回模拟数据
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        **验证: 需求 1.4, 2.1**
        """
        from app.models.file_management import DetailedFileInfo, IntegrityStatus
        from app.core.container import get_parquet_manager
        
        # 模拟Parquet管理器返回
        mock_file_info = DetailedFileInfo(
            file_path=f"/data/{filters.get('stock_code', '000001.SZ')}/test.parquet",
            stock_code=filters.get('stock_code', '000001.SZ'),
            date_range=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
            record_count=1000,
            file_size=1024000,
            last_modified=datetime.now(),
            integrity_status=IntegrityStatus.VALID,
            compression_ratio=0.3,
            created_at=datetime.now()
        )
        
        # 配置模拟管理器
        self.mock_parquet_manager.get_detailed_file_list.return_value = [mock_file_info]
        
        # 使用FastAPI依赖覆盖
        self.test_app.dependency_overrides[get_parquet_manager] = lambda: self.mock_parquet_manager
        
        try:
            # 构建查询参数
            params = {}
            for key, value in filters.items():
                if value is not None:
                    if isinstance(value, datetime):
                        params[key] = value.isoformat()
                    else:
                        params[key] = value
            
            response = self.client.get("/api/v1/data/files", params=params)
            
            # 验证响应
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "files" in data["data"]
            
            # 验证调用了真实服务
            self.mock_parquet_manager.get_detailed_file_list.assert_called_once()
            
            # 验证过滤条件传递正确
            call_args = self.mock_parquet_manager.get_detailed_file_list.call_args[0][0]
            assert call_args.stock_code == filters.get('stock_code')
            assert call_args.limit == filters.get('limit', 100)
            assert call_args.offset == filters.get('offset', 0)
            
        finally:
            # 清理依赖覆盖
            self.test_app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    async def test_data_stats_api_calls_real_parquet_manager(self):
        """
        属性 1: API路由真实服务调用
        数据统计API应该调用真实的Parquet管理器获取统计信息
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        **验证: 需求 5.1, 2.2**
        """
        from app.models.file_management import ComprehensiveStats
        from app.core.container import get_parquet_manager
        
        # 模拟统计数据
        mock_stats = ComprehensiveStats(
            total_files=10,
            total_size_bytes=10240000,
            total_records=50000,
            stock_count=5,
            date_range=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
            average_file_size=1024000.0,
            storage_efficiency=4.88,
            last_sync_time=datetime.now(),
            stocks_by_size=[("000001.SZ", 2048000), ("000002.SZ", 1536000)],
            monthly_distribution={"2023-01": 1000, "2023-02": 1200}
        )
        
        # 配置模拟管理器
        self.mock_parquet_manager.get_comprehensive_stats.return_value = mock_stats
        
        # 使用FastAPI依赖覆盖
        self.test_app.dependency_overrides[get_parquet_manager] = lambda: self.mock_parquet_manager
        
        try:
            response = self.client.get("/api/v1/data/stats")
            
            # 验证响应
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "total_files" in data["data"]
            assert data["data"]["total_files"] == 10
            assert data["data"]["stock_count"] == 5
            
            # 验证调用了真实服务
            self.mock_parquet_manager.get_comprehensive_stats.assert_called_once()
            
        finally:
            # 清理依赖覆盖
            self.test_app.dependency_overrides.clear()
    
    @pytest.mark.asyncio
    @given(sync_requests())
    @settings(max_examples=5, deadline=10000)
    async def test_data_sync_api_calls_real_sync_engine(self, sync_request):
        """
        属性 1: API路由真实服务调用
        数据同步API应该调用真实的数据同步引擎而不是返回模拟结果
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        **验证: 需求 1.5, 4.1**
        """
        from app.models.sync_models import BatchSyncResult, SyncResult
        
        # 模拟同步结果
        mock_sync_results = [
            SyncResult(
                stock_code=code,
                success=True,
                records_synced=1000,
                start_time=datetime.now(),
                end_time=datetime.now(),
                data_range=(sync_request["start_date"], sync_request["end_date"])
            )
            for code in sync_request["stock_codes"]
        ]
        
        mock_batch_result = BatchSyncResult(
            sync_id="test-sync-123",
            success=True,
            total_stocks=len(sync_request["stock_codes"]),
            successful_syncs=mock_sync_results,
            failed_syncs=[],
            start_time=datetime.now(),
            end_time=datetime.now(),
            message="同步完成"
        )
        
        with patch('app.core.container.get_data_sync_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.sync_stocks_batch.return_value = mock_batch_result
            mock_get_engine.return_value = mock_engine
            
            # 构建请求数据
            request_data = {
                "stock_codes": sync_request["stock_codes"],
                "start_date": sync_request["start_date"].isoformat(),
                "end_date": sync_request["end_date"].isoformat(),
                "force_update": sync_request["force_update"],
                "sync_mode": sync_request["sync_mode"],
                "max_concurrent": sync_request["max_concurrent"],
                "retry_count": sync_request["retry_count"]
            }
            
            response = self.client.post("/api/v1/data/sync", json=request_data)
            
            # 验证响应
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "sync_id" in data["data"]
            assert data["data"]["total_stocks"] == len(sync_request["stock_codes"])
            
            # 验证调用了真实服务
            mock_get_engine.assert_called_once()
            mock_engine.sync_stocks_batch.assert_called_once()
            
            # 验证请求参数传递正确
            call_args = mock_engine.sync_stocks_batch.call_args[0][0]
            assert call_args.stock_codes == sync_request["stock_codes"]
            assert call_args.force_update == sync_request["force_update"]
            assert call_args.sync_mode.value == sync_request["sync_mode"]
    
    @pytest.mark.asyncio
    async def test_sync_progress_api_calls_real_sync_engine(self):
        """
        属性 1: API路由真实服务调用
        同步进度API应该调用真实的数据同步引擎获取进度信息
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        **验证: 需求 4.2, 5.3**
        """
        from app.models.sync_models import SyncProgress, SyncStatus
        
        sync_id = "test-sync-123"
        mock_progress = SyncProgress(
            sync_id=sync_id,
            total_stocks=5,
            completed_stocks=3,
            failed_stocks=1,
            current_stock="000001.SZ",
            progress_percentage=80.0,
            estimated_remaining_time=timedelta(minutes=2),
            start_time=datetime.now(),
            status=SyncStatus.RUNNING,
            last_update=datetime.now()
        )
        
        with patch('app.core.container.get_data_sync_engine') as mock_get_engine:
            mock_engine = AsyncMock()
            mock_engine.get_sync_progress.return_value = mock_progress
            mock_get_engine.return_value = mock_engine
            
            response = self.client.get(f"/api/v1/data/sync/{sync_id}/progress")
            
            # 验证响应
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["sync_id"] == sync_id
            assert data["data"]["progress_percentage"] == 80.0
            
            # 验证调用了真实服务
            mock_get_engine.assert_called_once()
            mock_engine.get_sync_progress.assert_called_once_with(sync_id)
    
    @pytest.mark.asyncio
    async def test_delete_files_api_calls_real_parquet_manager(self):
        """
        属性 1: API路由真实服务调用
        删除文件API应该调用真实的Parquet管理器进行安全删除
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        **验证: 需求 2.3**
        """
        from app.models.file_management import DeletionResult
        
        file_paths = ["/data/000001.SZ/test1.parquet", "/data/000002.SZ/test2.parquet"]
        
        mock_deletion_result = DeletionResult(
            success=True,
            deleted_files=file_paths,
            failed_files=[],
            total_deleted=2,
            freed_space_bytes=2048000,
            message="删除成功"
        )
        
        with patch('app.core.container.get_parquet_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.delete_files_safely.return_value = mock_deletion_result
            mock_get_manager.return_value = mock_manager
            
            response = self.client.delete(
                "/api/v1/data/files",
                params={"file_paths": file_paths}
            )
            
            # 验证响应
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["total_deleted"] == 2
            assert data["data"]["freed_space_bytes"] == 2048000
            
            # 验证调用了真实服务
            mock_get_manager.assert_called_once()
            mock_manager.delete_files_safely.assert_called_once_with(file_paths)
    
    @pytest.mark.asyncio
    async def test_api_error_handling_consistency(self):
        """
        属性 1: API路由真实服务调用
        API错误处理应该一致地处理服务层异常
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        **验证: 需求 6.1, 6.3**
        """
        with patch('app.core.container.get_parquet_manager') as mock_get_manager:
            # 模拟服务异常
            mock_manager = MagicMock()
            mock_manager.get_comprehensive_stats.side_effect = Exception("服务异常")
            mock_get_manager.return_value = mock_manager
            
            response = self.client.get("/api/v1/data/stats")
            
            # 验证错误处理
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "服务异常" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_api_response_format_consistency(self):
        """
        属性 1: API路由真实服务调用
        所有数据管理API应该返回一致的响应格式
        **功能: data-management-implementation, 属性 1: API路由真实服务调用**
        **验证: 需求 1.4, 1.5**
        """
        from app.models.file_management import ComprehensiveStats
        
        mock_stats = ComprehensiveStats(
            total_files=1,
            total_size_bytes=1024,
            total_records=100,
            stock_count=1,
            date_range=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
            average_file_size=1024.0,
            storage_efficiency=0.1,
            last_sync_time=datetime.now(),
            stocks_by_size=[],
            monthly_distribution={}
        )
        
        with patch('app.core.container.get_parquet_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_comprehensive_stats.return_value = mock_stats
            mock_get_manager.return_value = mock_manager
            
            response = self.client.get("/api/v1/data/stats")
            
            # 验证标准响应格式
            assert response.status_code == 200
            data = response.json()
            
            # 检查标准字段
            assert "success" in data
            assert "message" in data
            assert "data" in data
            assert "timestamp" in data
            
            # 检查数据类型
            assert isinstance(data["success"], bool)
            assert isinstance(data["message"], str)
            assert data["data"] is not None


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
    # 清理代码可以在这里添加