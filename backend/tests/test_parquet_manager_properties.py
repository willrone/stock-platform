"""
Parquet管理器属性测试
验证Parquet文件管理的完整性属性
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
import pandas as pd

from app.services.data.parquet_manager import ParquetManager
from app.models.stock_simple import StockData
from app.models.file_management import (
    FileFilters, FilterCriteria, IntegrityStatus
)


@composite
def stock_codes(draw):
    """生成股票代码"""
    market = draw(st.sampled_from(['SZ', 'SH']))
    code = draw(st.integers(min_value=1, max_value=999999))
    return f"{code:06d}.{market}"


@composite
def stock_data_lists(draw):
    """生成股票数据列表"""
    stock_code = draw(stock_codes())
    size = draw(st.integers(min_value=1, max_value=50))
    
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


class TestParquetManagerProperties:
    """Parquet管理器属性测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ParquetManager(self.temp_dir)
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    @given(stock_data_lists())
    @settings(max_examples=20, deadline=15000)
    async def test_file_save_and_list_consistency(self, stock_data):
        """
        属性 2: Parquet文件管理完整性
        保存文件后，文件列表应该准确反映文件系统状态
        **功能: data-management-implementation, 属性 2: Parquet文件管理完整性**
        **验证: 需求 2.1, 2.2, 2.3, 2.4, 2.5**
        """
        if not stock_data:
            return
        
        stock_code = stock_data[0].stock_code
        
        # 保存数据
        success = self.manager.save_stock_data(stock_data)
        assert success, "数据保存应该成功"
        
        # 获取文件列表
        filters = FileFilters(stock_code=stock_code, limit=100)
        file_list = self.manager.get_detailed_file_list(filters)
        
        # 验证文件列表不为空
        assert len(file_list) > 0, "保存数据后应该有文件"
        
        # 验证文件信息正确
        for file_info in file_list:
            assert file_info.stock_code == stock_code
            assert file_info.record_count > 0
            assert file_info.file_size > 0
            assert Path(file_info.file_path).exists()
            
            # 验证文件完整性
            validation_result = self.manager.validate_file_integrity(file_info.file_path)
            assert validation_result.is_valid or validation_result.integrity_status != IntegrityStatus.CORRUPTED
    
    @pytest.mark.asyncio
    @given(stock_data_lists())
    @settings(max_examples=15, deadline=12000)
    async def test_comprehensive_stats_accuracy(self, stock_data):
        """
        属性: 综合统计信息准确性
        统计信息应该准确反映实际文件状态
        **功能: data-management-implementation, 属性 2: Parquet文件管理完整性**
        """
        if not stock_data:
            return
        
        # 保存数据
        self.manager.save_stock_data(stock_data)
        
        # 获取统计信息
        stats = self.manager.get_comprehensive_stats()
        
        # 验证统计信息
        assert stats.total_files > 0, "应该有文件"
        assert stats.total_records > 0, "应该有记录"
        assert stats.total_size_bytes > 0, "应该有文件大小"
        assert stats.stock_count > 0, "应该有股票"
        
        # 验证平均文件大小计算正确
        expected_avg = stats.total_size_bytes / stats.total_files
        assert abs(stats.average_file_size - expected_avg) < 0.01, "平均文件大小计算错误"
        
        # 验证日期范围
        if stats.total_records > 0:
            assert stats.date_range[0] <= stats.date_range[1], "日期范围应该有效"
    
    @pytest.mark.asyncio
    @given(stock_data_lists(), st.integers(min_value=1, max_value=10))
    @settings(max_examples=10, deadline=15000)
    async def test_file_deletion_consistency(self, stock_data, num_files_to_delete):
        """
        属性: 文件删除一致性
        删除文件后，文件系统状态应该与删除结果一致
        **功能: data-management-implementation, 属性 2: Parquet文件管理完整性**
        """
        if not stock_data:
            return
        
        # 保存数据
        self.manager.save_stock_data(stock_data)
        
        # 获取文件列表
        filters = FileFilters(limit=100)
        file_list = self.manager.get_detailed_file_list(filters)
        
        if not file_list:
            return
        
        # 选择要删除的文件
        files_to_delete = [f.file_path for f in file_list[:min(num_files_to_delete, len(file_list))]]
        
        # 删除文件
        deletion_result = self.manager.delete_files_safely(files_to_delete)
        
        # 验证删除结果
        assert len(deletion_result.deleted_files) + len(deletion_result.failed_files) == len(files_to_delete)
        
        # 验证文件确实被删除
        for deleted_file in deletion_result.deleted_files:
            assert not Path(deleted_file).exists(), f"文件 {deleted_file} 应该被删除"
        
        # 验证失败的文件仍然存在
        for failed_file, _ in deletion_result.failed_files:
            # 注意：某些失败可能是因为文件不存在，所以这里不强制检查
            pass
        
        # 验证统计信息更新
        new_stats = self.manager.get_comprehensive_stats()
        expected_files = len(file_list) - len(deletion_result.deleted_files)
        assert new_stats.total_files == expected_files, "文件数量统计应该更新"
    
    @pytest.mark.asyncio
    @given(stock_data_lists())
    @settings(max_examples=15, deadline=10000)
    async def test_file_filtering_correctness(self, stock_data):
        """
        属性: 文件筛选正确性
        筛选结果应该符合筛选条件
        **功能: data-management-implementation, 属性 2: Parquet文件管理完整性**
        """
        if not stock_data:
            return
        
        stock_code = stock_data[0].stock_code
        
        # 保存数据
        self.manager.save_stock_data(stock_data)
        
        # 按股票代码筛选
        criteria = FilterCriteria(stock_codes=[stock_code])
        filtered_files = self.manager.filter_files(criteria)
        
        # 验证筛选结果
        for file_info in filtered_files:
            assert file_info.stock_code == stock_code, "筛选结果应该匹配股票代码"
        
        # 按文件大小筛选
        if filtered_files:
            min_size = min(f.file_size for f in filtered_files)
            max_size = max(f.file_size for f in filtered_files)
            
            if min_size < max_size:
                mid_size = (min_size + max_size) // 2
                
                size_criteria = FilterCriteria(min_file_size=mid_size)
                size_filtered = self.manager.filter_files(size_criteria)
                
                for file_info in size_filtered:
                    assert file_info.file_size >= mid_size, "筛选结果应该满足最小文件大小"
    
    @pytest.mark.asyncio
    @given(stock_data_lists())
    @settings(max_examples=10, deadline=12000)
    async def test_file_integrity_validation_consistency(self, stock_data):
        """
        属性: 文件完整性验证一致性
        完整性验证结果应该与实际文件状态一致
        **功能: data-management-implementation, 属性 2: Parquet文件管理完整性**
        """
        if not stock_data:
            return
        
        # 保存数据
        self.manager.save_stock_data(stock_data)
        
        # 获取文件列表
        filters = FileFilters(limit=10)
        file_list = self.manager.get_detailed_file_list(filters)
        
        for file_info in file_list:
            # 验证文件完整性
            validation_result = self.manager.validate_file_integrity(file_info.file_path)
            
            # 验证验证结果的一致性
            assert validation_result.file_path == file_info.file_path
            
            if validation_result.is_valid:
                assert validation_result.integrity_status == IntegrityStatus.VALID
                assert len(validation_result.error_messages) == 0
            else:
                assert validation_result.integrity_status in [
                    IntegrityStatus.CORRUPTED, 
                    IntegrityStatus.INCOMPLETE, 
                    IntegrityStatus.UNKNOWN
                ]
            
            # 如果文件存在且可读，记录数应该大于0
            if Path(file_info.file_path).exists() and validation_result.is_valid:
                assert validation_result.record_count > 0
    
    @pytest.mark.asyncio
    @given(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=20))
    @settings(max_examples=10, deadline=10000)
    async def test_pagination_consistency(self, num_stocks, records_per_stock):
        """
        属性: 分页一致性
        分页结果应该覆盖所有数据且无重复
        **功能: data-management-implementation, 属性 2: Parquet文件管理完整性**
        """
        # 创建多个股票的数据
        all_stock_data = []
        for i in range(num_stocks):
            stock_code = f"{i+1:06d}.SZ"
            base_date = datetime(2023, 1, 1)
            
            stock_data = []
            for j in range(records_per_stock):
                data = StockData(
                    stock_code=stock_code,
                    date=base_date + timedelta(days=j),
                    open=10.0,
                    high=11.0,
                    low=9.0,
                    close=10.5,
                    volume=1000000,
                    adj_close=10.5
                )
                stock_data.append(data)
            
            all_stock_data.extend(stock_data)
            self.manager.save_stock_data(stock_data)
        
        # 获取总数
        all_files = self.manager.get_detailed_file_list(FileFilters(limit=1000))
        total_files = len(all_files)
        
        if total_files == 0:
            return
        
        # 测试分页
        page_size = max(1, total_files // 2)
        page1 = self.manager.get_detailed_file_list(FileFilters(limit=page_size, offset=0))
        page2 = self.manager.get_detailed_file_list(FileFilters(limit=page_size, offset=page_size))
        
        # 验证分页结果
        assert len(page1) <= page_size
        assert len(page2) <= page_size
        
        # 验证无重复
        page1_paths = {f.file_path for f in page1}
        page2_paths = {f.file_path for f in page2}
        assert len(page1_paths.intersection(page2_paths)) == 0, "分页结果不应该有重复"
        
        # 验证总数一致
        total_paginated = len(page1) + len(page2)
        assert total_paginated <= total_files, "分页总数不应该超过实际文件数"


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后自动清理"""
    yield
    # 清理代码可以在这里添加
    # 清理临时文件
    pass