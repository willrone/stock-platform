"""
数据库和存储层基础测试
测试数据库管理器和Parquet管理器的基本功能

注意：DatabaseManager 已废弃，迁移到 SQLAlchemy ORM。
整个模块跳过，待重写为基于 SQLAlchemy 的测试。
"""

import pytest

try:
    from app.models.database import (
        DatabaseManager,
        ModelMetadata,
        SystemConfig,
        Task,
        TaskResult,
        TaskStatus,
    )
except ImportError:
    pytest.skip(
        "DatabaseManager 已废弃（迁移到 SQLAlchemy），整个测试模块跳过",
        allow_module_level=True,
    )
from app.models.stock_simple import StockData
from app.services.data.data_lifecycle import DataLifecycleManager, RetentionPolicy
from app.services.data.parquet_manager import ParquetManager


class TestDatabaseManager:
    """数据库管理器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = DatabaseManager(db_path=f"{self.temp_dir}/test.db")
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """测试数据库初始化"""
        # 验证数据库文件存在
        assert Path(self.db_manager.db_path).exists()
        
        # 验证表结构
        tables = self.db_manager.fetch_all("""
            SELECT name FROM sqlite_master WHERE type='table'
        """)
        
        table_names = {row['name'] for row in tables}
        expected_tables = {'tasks', 'task_results', 'model_metadata', 'system_config'}
        
        assert expected_tables.issubset(table_names), "应该包含所有必需的表"
    
    def test_task_crud_operations(self):
        """测试任务的CRUD操作"""
        import json
        
        # 创建任务
        task_data = {
            'name': '测试任务',
            'description': '这是一个测试任务',
            'stock_codes': json.dumps(['000001.SZ', '000002.SZ']),
            'indicators': json.dumps(['MA5', 'RSI']),
            'models': json.dumps(['LSTM']),
            'parameters': json.dumps({'param1': 'value1'}),
            'status': TaskStatus.PENDING.value,
            'progress': 0.0,
            'created_at': datetime.now()
        }
        
        # 插入任务
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO tasks (name, description, stock_codes, indicators, models, parameters, status, progress, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(task_data.values()))
            task_id = cursor.lastrowid
        
        # 读取任务
        task_row = self.db_manager.fetch_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        assert task_row is not None
        assert task_row['name'] == task_data['name']
        assert task_row['status'] == TaskStatus.PENDING.value
        
        # 更新任务状态
        with self.db_manager.get_connection() as conn:
            conn.execute("""
                UPDATE tasks SET status = ?, progress = ? WHERE id = ?
            """, (TaskStatus.RUNNING.value, 50.0, task_id))
        
        # 验证更新
        updated_task = self.db_manager.fetch_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        assert updated_task['status'] == TaskStatus.RUNNING.value
        assert updated_task['progress'] == 50.0
        
        # 删除任务
        with self.db_manager.get_connection() as conn:
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        
        # 验证删除
        deleted_task = self.db_manager.fetch_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        assert deleted_task is None
    
    def test_task_result_operations(self):
        """测试任务结果操作"""
        import json
        
        # 先创建一个任务
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO tasks (name, description, stock_codes, indicators, models, parameters, status, progress, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ('测试任务', '描述', '[]', '[]', '[]', '{}', TaskStatus.PENDING.value, 0.0, datetime.now()))
            task_id = cursor.lastrowid
        
        # 创建任务结果
        result_data = {
            'task_id': task_id,
            'stock_code': '000001.SZ',
            'prediction_date': datetime.now() + timedelta(days=1),
            'prediction_value': 100.5,
            'confidence': 0.85,
            'model_name': 'LSTM',
            'indicators_used': json.dumps(['MA5', 'RSI']),
            'backtest_metrics': json.dumps({'accuracy': 0.75}),
            'created_at': datetime.now()
        }
        
        # 插入结果
        with self.db_manager.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO task_results (task_id, stock_code, prediction_date, prediction_value, 
                                        confidence, model_name, indicators_used, backtest_metrics, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(result_data.values()))
            result_id = cursor.lastrowid
        
        # 读取结果
        result_row = self.db_manager.fetch_one("SELECT * FROM task_results WHERE id = ?", (result_id,))
        assert result_row is not None
        assert result_row['task_id'] == task_id
        assert result_row['stock_code'] == '000001.SZ'
        assert abs(result_row['prediction_value'] - 100.5) < 0.01
        
        # 测试关联查询
        joined_results = self.db_manager.fetch_all("""
            SELECT tr.*, t.name as task_name 
            FROM task_results tr 
            JOIN tasks t ON tr.task_id = t.id 
            WHERE tr.task_id = ?
        """, (task_id,))
        
        assert len(joined_results) == 1
        assert joined_results[0]['task_name'] == '测试任务'


class TestParquetManager:
    """Parquet管理器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_manager = ParquetManager(base_path=f"{self.temp_dir}/parquet")
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parquet_file_path_generation(self):
        """测试Parquet文件路径生成"""
        stock_code = "000001.SZ"
        date = datetime(2023, 5, 15)
        
        file_path = self.parquet_manager.get_file_path(stock_code, date)
        expected_path = self.parquet_manager.base_path / "000001.SZ" / "2023" / "05" / "000001.SZ_2023-05.parquet"
        
        assert file_path == expected_path
    
    def test_stock_data_save_and_load(self):
        """测试股票数据保存和加载"""
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
        
        # 保存数据
        success = self.parquet_manager.save_stock_data(stock_data)
        assert success, "数据保存应该成功"
        
        # 加载数据
        loaded_data = self.parquet_manager.load_stock_data(
            "TEST.SZ",
            datetime(2023, 1, 1),
            datetime(2023, 1, 2)
        )
        
        assert len(loaded_data) == 2, "应该加载2条数据"
        
        # 验证数据内容
        loaded_data.sort(key=lambda x: x.date)
        assert loaded_data[0].stock_code == "TEST.SZ"
        assert loaded_data[0].close == 103.0
        assert loaded_data[1].close == 106.0
    
    def test_incremental_data_update(self):
        """测试增量数据更新"""
        # 初始数据
        initial_data = [
            StockData(
                stock_code="INC.SZ",
                date=datetime(2023, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                adj_close=103.0
            )
        ]
        
        # 保存初始数据
        self.parquet_manager.save_stock_data(initial_data, merge_with_existing=False)
        
        # 增量数据
        incremental_data = [
            StockData(
                stock_code="INC.SZ",
                date=datetime(2023, 1, 2),
                open=103.0,
                high=108.0,
                low=102.0,
                close=106.0,
                volume=1200000,
                adj_close=106.0
            )
        ]
        
        # 增量保存
        self.parquet_manager.save_stock_data(incremental_data, merge_with_existing=True)
        
        # 验证合并结果
        all_data = self.parquet_manager.load_stock_data(
            "INC.SZ",
            datetime(2023, 1, 1),
            datetime(2023, 1, 2)
        )
        
        assert len(all_data) == 2, "应该有2条数据"
    
    def test_file_info_and_stats(self):
        """测试文件信息和统计"""
        # 创建测试数据
        test_data = [
            StockData(
                stock_code="STATS.SZ",
                date=datetime(2023, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                adj_close=103.0
            )
        ]
        
        self.parquet_manager.save_stock_data(test_data)
        
        # 获取文件信息
        file_infos = self.parquet_manager.get_file_info("STATS.SZ")
        assert len(file_infos) > 0, "应该有文件信息"
        
        file_info = file_infos[0]
        assert file_info.stock_code == "STATS.SZ"
        assert file_info.record_count == 1
        
        # 获取日期范围
        date_range = self.parquet_manager.get_available_date_range("STATS.SZ")
        assert date_range is not None, "应该有日期范围"
        
        # 获取存储统计
        stats = self.parquet_manager.get_storage_stats()
        assert stats['total_files'] > 0, "应该有文件统计"
        assert stats['stock_count'] > 0, "应该有股票统计"


class TestDataLifecycleManager:
    """数据生命周期管理器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = DatabaseManager(db_path=f"{self.temp_dir}/test.db")
        self.parquet_manager = ParquetManager(base_path=f"{self.temp_dir}/parquet")
        
        # 使用短保留期进行测试
        retention_policy = RetentionPolicy(
            parquet_retention_days=1,
            task_retention_days=1,
            log_retention_days=1,
            temp_file_retention_days=1,
            model_retention_days=1
        )
        
        self.lifecycle_manager = DataLifecycleManager(
            db_manager=self.db_manager,
            parquet_manager=self.parquet_manager,
            retention_policy=retention_policy,
            base_path=self.temp_dir
        )
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_storage_usage_calculation(self):
        """测试存储使用情况计算"""
        # 创建一些测试数据
        test_data = [
            StockData(
                stock_code="USAGE.SZ",
                date=datetime(2023, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                adj_close=103.0
            )
        ]
        
        self.parquet_manager.save_stock_data(test_data)
        
        # 获取存储使用情况
        usage = self.lifecycle_manager.get_storage_usage()
        
        assert 'parquet' in usage
        assert 'database' in usage
        assert 'total' in usage
        assert usage['total']['size'] > 0, "总存储大小应该大于0"
    
    def test_cleanup_dry_run(self):
        """测试清理试运行"""
        import os
        
        # 创建一些旧文件
        old_temp_file = self.lifecycle_manager.temp_dir / "old_temp.txt"
        old_temp_file.write_text("test content")
        
        # 修改文件时间为2天前
        old_time = datetime.now() - timedelta(days=2)
        old_timestamp = old_time.timestamp()
        os.utime(old_temp_file, (old_timestamp, old_timestamp))
        
        # 运行试运行清理
        result = self.lifecycle_manager.run_cleanup(dry_run=True)
        
        assert result.deleted_files >= 0, "应该有删除文件统计"
        assert len(result.errors) == 0, "试运行不应该有错误"
        
        # 验证文件仍然存在（试运行不删除）
        assert old_temp_file.exists(), "试运行不应该删除文件"
    
    def test_cleanup_schedule_configuration(self):
        """测试清理调度配置"""
        success = self.lifecycle_manager.schedule_cleanup(interval_hours=12)
        assert success, "调度配置应该成功"
        
        # 验证配置保存到数据库
        config_row = self.db_manager.fetch_one("""
            SELECT * FROM system_config WHERE key = 'cleanup_schedule'
        """)
        
        assert config_row is not None, "应该保存调度配置"
        assert 'interval_hours' in config_row['value'], "配置应该包含间隔时间"