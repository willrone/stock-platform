"""
数据存储格式统一性属性测试
Feature: stock-prediction-platform, Property 7: 数据存储格式统一性
"""

import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.models.stock_simple import StockData
from app.models.database import DatabaseManager, Task, TaskResult, ModelMetadata, SystemConfig, TaskStatus
from app.services.parquet_manager import ParquetManager


@composite
def stock_code_strategy(draw):
    """生成有效的股票代码"""
    number = draw(st.integers(min_value=1, max_value=999999))
    market = draw(st.sampled_from(['SH', 'SZ']))
    return f"{number:06d}.{market}"


@composite
def stock_data_list_strategy(draw):
    """生成股票数据列表"""
    stock_code = draw(stock_code_strategy())
    start_date = draw(st.dates(min_value=datetime(2020, 1, 1).date(), max_value=datetime(2023, 12, 31).date()))
    days_count = draw(st.integers(min_value=10, max_value=50))
    
    data_list = []
    base_price = draw(st.floats(min_value=10.0, max_value=1000.0))
    
    for i in range(days_count):
        date = datetime.combine(start_date, datetime.min.time()) + timedelta(days=i)
        
        # 生成合理的价格数据
        price_change = draw(st.floats(min_value=-0.05, max_value=0.05))
        close_price = max(1.0, base_price * (1 + price_change))
        
        open_price = draw(st.floats(min_value=close_price * 0.98, max_value=close_price * 1.02))
        high_price = max(open_price, close_price) * draw(st.floats(min_value=1.0, max_value=1.02))
        low_price = min(open_price, close_price) * draw(st.floats(min_value=0.98, max_value=1.0))
        volume = draw(st.integers(min_value=100000, max_value=10000000))
        
        data_list.append(StockData(
            stock_code=stock_code,
            date=date,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=volume,
            adj_close=round(close_price, 2)
        ))
        
        base_price = close_price
    
    return data_list


class TestStorageFormatConsistency:
    """数据存储格式统一性属性测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_manager = ParquetManager(base_path=f"{self.temp_dir}/parquet")
        self.db_manager = DatabaseManager(db_path=f"{self.temp_dir}/test.db")
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(stock_data_list=stock_data_list_strategy())
    @settings(max_examples=15, deadline=None)
    def test_parquet_storage_consistency_property(self, stock_data_list):
        """
        属性测试：Parquet存储格式一致性
        
        对于任何有效的股票数据，Parquet存储应该：
        1. 保存后能够完整读取
        2. 数据类型保持一致
        3. 数据顺序正确
        4. 支持增量更新
        
        验证：需求 7.1, 7.4
        """
        if not stock_data_list:
            return
        
        stock_code = stock_data_list[0].stock_code
        
        # 保存数据
        success = self.parquet_manager.save_stock_data(stock_data_list, merge_with_existing=False)
        assert success, "数据保存应该成功"
        
        # 读取数据
        start_date = min(data.date for data in stock_data_list)
        end_date = max(data.date for data in stock_data_list)
        
        loaded_data = self.parquet_manager.load_stock_data(stock_code, start_date, end_date)
        
        # 验证数据完整性
        assert len(loaded_data) == len(stock_data_list), "读取的数据数量应该与保存的一致"
        
        # 按日期排序进行比较
        original_sorted = sorted(stock_data_list, key=lambda x: x.date)
        loaded_sorted = sorted(loaded_data, key=lambda x: x.date)
        
        for original, loaded in zip(original_sorted, loaded_sorted):
            assert original.stock_code == loaded.stock_code, "股票代码应该一致"
            assert original.date == loaded.date, "日期应该一致"
            assert abs(original.open - loaded.open) < 0.01, "开盘价应该一致"
            assert abs(original.high - loaded.high) < 0.01, "最高价应该一致"
            assert abs(original.low - loaded.low) < 0.01, "最低价应该一致"
            assert abs(original.close - loaded.close) < 0.01, "收盘价应该一致"
            assert original.volume == loaded.volume, "成交量应该一致"
            assert abs(original.adj_close - loaded.adj_close) < 0.01, "复权价应该一致"
        
        # 测试增量更新
        # 添加新的数据点
        new_date = end_date + timedelta(days=1)
        new_data = StockData(
            stock_code=stock_code,
            date=new_date,
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
            adj_close=103.0
        )
        
        # 增量保存
        success = self.parquet_manager.save_stock_data([new_data], merge_with_existing=True)
        assert success, "增量保存应该成功"
        
        # 验证增量数据
        extended_end_date = new_date
        all_loaded_data = self.parquet_manager.load_stock_data(stock_code, start_date, extended_end_date)
        assert len(all_loaded_data) == len(stock_data_list) + 1, "增量更新后数据数量应该正确"
        
        # 验证新数据存在
        new_data_found = any(data.date == new_date for data in all_loaded_data)
        assert new_data_found, "新增的数据应该存在"
    
    @given(
        task_name=st.text(min_size=1, max_size=50),
        stock_codes=st.lists(stock_code_strategy(), min_size=1, max_size=3),
        indicators=st.lists(st.sampled_from(['MA5', 'RSI', 'MACD']), min_size=1, max_size=3, unique=True)
    )
    @settings(max_examples=10, deadline=None)
    def test_database_storage_consistency_property(self, task_name, stock_codes, indicators):
        """
        属性测试：数据库存储格式一致性
        
        对于任何有效的任务数据，数据库存储应该：
        1. 保存后能够完整读取
        2. 数据类型正确
        3. 关联关系正确
        4. 支持CRUD操作
        
        验证：需求 7.2
        """
        import json
        
        # 创建任务
        task = Task(
            name=task_name,
            description=f"测试任务: {task_name}",
            stock_codes=json.dumps(stock_codes),
            indicators=json.dumps(indicators),
            models=json.dumps(['LSTM', 'XGBoost']),
            parameters=json.dumps({'test_param': 'test_value'}),
            status=TaskStatus.PENDING,
            progress=0.0,
            created_at=datetime.now()
        )
        
        # 保存任务到数据库
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO tasks (name, description, stock_codes, indicators, models, parameters, status, progress, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.name, task.description, task.stock_codes, task.indicators,
                task.models, task.parameters, task.status.value, task.progress, task.created_at
            ))
            task_id = cursor.lastrowid
        
        # 读取任务
        row = self.db_manager.fetch_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        assert row is not None, "任务应该能够读取"
        
        # 验证数据一致性
        assert row['name'] == task.name, "任务名称应该一致"
        assert row['description'] == task.description, "任务描述应该一致"
        assert json.loads(row['stock_codes']) == stock_codes, "股票代码列表应该一致"
        assert json.loads(row['indicators']) == indicators, "指标列表应该一致"
        assert row['status'] == task.status.value, "任务状态应该一致"
        assert abs(row['progress'] - task.progress) < 0.01, "进度应该一致"
        
        # 测试任务结果存储
        for stock_code in stock_codes:
            task_result = TaskResult(
                task_id=task_id,
                stock_code=stock_code,
                prediction_date=datetime.now() + timedelta(days=1),
                prediction_value=100.5,
                confidence=0.85,
                model_name='LSTM',
                indicators_used=json.dumps(indicators),
                backtest_metrics=json.dumps({'accuracy': 0.75, 'sharpe_ratio': 1.2}),
                created_at=datetime.now()
            )
            
            # 保存任务结果
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO task_results (task_id, stock_code, prediction_date, prediction_value, 
                                            confidence, model_name, indicators_used, backtest_metrics, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_result.task_id, task_result.stock_code, task_result.prediction_date,
                    task_result.prediction_value, task_result.confidence, task_result.model_name,
                    task_result.indicators_used, task_result.backtest_metrics, task_result.created_at
                ))
                result_id = cursor.lastrowid
            
            # 验证任务结果
            result_row = self.db_manager.fetch_one("SELECT * FROM task_results WHERE id = ?", (result_id,))
            assert result_row is not None, "任务结果应该能够读取"
            assert result_row['task_id'] == task_id, "任务ID应该一致"
            assert result_row['stock_code'] == stock_code, "股票代码应该一致"
            assert abs(result_row['prediction_value'] - task_result.prediction_value) < 0.01, "预测值应该一致"
            assert abs(result_row['confidence'] - task_result.confidence) < 0.01, "置信度应该一致"
        
        # 验证关联查询
        results = self.db_manager.fetch_all("""
            SELECT tr.*, t.name as task_name 
            FROM task_results tr 
            JOIN tasks t ON tr.task_id = t.id 
            WHERE t.id = ?
        """, (task_id,))
        
        assert len(results) == len(stock_codes), "关联查询结果数量应该正确"
        for result in results:
            assert result['task_name'] == task.name, "关联查询的任务名称应该正确"
    
    def test_storage_format_compatibility(self):
        """测试存储格式兼容性"""
        # 创建测试数据
        stock_data = [
            StockData(
                stock_code="TEST.SZ",
                date=datetime(2023, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                adj_close=103.0
            ),
            StockData(
                stock_code="TEST.SZ",
                date=datetime(2023, 1, 2),
                open=103.0,
                high=108.0,
                low=102.0,
                close=106.0,
                volume=1200000,
                adj_close=106.0
            )
        ]
        
        # 测试Parquet存储
        success = self.parquet_manager.save_stock_data(stock_data)
        assert success, "Parquet存储应该成功"
        
        # 验证文件信息
        file_infos = self.parquet_manager.get_file_info("TEST.SZ")
        assert len(file_infos) > 0, "应该有文件信息"
        
        file_info = file_infos[0]
        assert file_info.stock_code == "TEST.SZ", "文件信息中的股票代码应该正确"
        assert file_info.record_count == 2, "记录数量应该正确"
        assert file_info.start_date <= file_info.end_date, "日期范围应该合理"
        
        # 测试日期范围查询
        date_range = self.parquet_manager.get_available_date_range("TEST.SZ")
        assert date_range is not None, "应该能获取日期范围"
        assert date_range[0] <= date_range[1], "日期范围应该合理"
        
        # 测试存储统计
        stats = self.parquet_manager.get_storage_stats()
        assert stats['total_files'] > 0, "应该有文件统计"
        assert stats['stock_count'] > 0, "应该有股票统计"
        assert 'TEST.SZ' in stats['stocks'], "应该包含测试股票"
    
    def test_data_lifecycle_management(self):
        """测试数据生命周期管理"""
        # 创建一些测试数据
        old_data = [
            StockData(
                stock_code="OLD.SZ",
                date=datetime(2020, 1, 1),
                open=50.0,
                high=55.0,
                low=49.0,
                close=53.0,
                volume=500000,
                adj_close=53.0
            )
        ]
        
        recent_data = [
            StockData(
                stock_code="RECENT.SZ",
                date=datetime.now() - timedelta(days=1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                adj_close=103.0
            )
        ]
        
        # 保存数据
        self.parquet_manager.save_stock_data(old_data)
        self.parquet_manager.save_stock_data(recent_data)
        
        # 获取初始统计
        initial_stats = self.parquet_manager.get_storage_stats()
        initial_file_count = initial_stats['total_files']
        
        # 清理旧文件（保留最近30天）
        deleted_count = self.parquet_manager.cleanup_old_files(days_to_keep=30)
        
        # 验证清理结果
        final_stats = self.parquet_manager.get_storage_stats()
        
        # 应该删除了一些文件
        if deleted_count > 0:
            assert final_stats['total_files'] < initial_file_count, "应该删除了一些文件"
        
        # 最近的数据应该还在
        recent_range = self.parquet_manager.get_available_date_range("RECENT.SZ")
        assert recent_range is not None, "最近的数据应该还存在"