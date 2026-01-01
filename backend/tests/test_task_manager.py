"""
任务管理服务基础测试
测试任务管理器的基本功能
"""

import tempfile
import shutil
import json
from datetime import datetime, timedelta

import pytest

from app.models.database import DatabaseManager, TaskStatus
from app.models.stock_simple import StockData
from app.services.tasks import (
    TaskManager, TaskCreateRequest, TaskUpdateRequest, TaskQuery
)


class TestTaskManager:
    """任务管理器基础测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = DatabaseManager(db_path=f"{self.temp_dir}/test.db")
        self.task_manager = TaskManager(self.db_manager)
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_task_basic(self):
        """测试基本任务创建"""
        request = TaskCreateRequest(
            name="测试任务",
            description="这是一个测试任务",
            stock_codes=["000001.SZ", "000002.SZ"],
            indicators=["MA5", "RSI"],
            models=["LSTM"],
            parameters={"epochs": 100, "batch_size": 32}
        )
        
        task_id = self.task_manager.create_task(request)
        
        assert isinstance(task_id, int)
        assert task_id > 0
        
        # 验证任务详情
        task = self.task_manager.get_task(task_id)
        assert task is not None
        assert task.name == "测试任务"
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0.0
        
        # 验证JSON字段
        stock_codes = json.loads(task.stock_codes)
        indicators = json.loads(task.indicators)
        models = json.loads(task.models)
        parameters = json.loads(task.parameters)
        
        assert stock_codes == ["000001.SZ", "000002.SZ"]
        assert indicators == ["MA5", "RSI"]
        assert models == ["LSTM"]
        assert parameters == {"epochs": 100, "batch_size": 32}
    
    def test_create_task_validation(self):
        """测试任务创建验证"""
        # 空名称
        with pytest.raises(ValueError, match="任务名称不能为空"):
            self.task_manager.create_task(TaskCreateRequest(
                name="",
                description="",
                stock_codes=["000001.SZ"],
                indicators=["MA5"],
                models=["LSTM"],
                parameters={}
            ))
        
        # 空股票代码列表
        with pytest.raises(ValueError, match="股票代码列表不能为空"):
            self.task_manager.create_task(TaskCreateRequest(
                name="测试",
                description="",
                stock_codes=[],
                indicators=["MA5"],
                models=["LSTM"],
                parameters={}
            ))
        
        # 空指标列表
        with pytest.raises(ValueError, match="指标列表不能为空"):
            self.task_manager.create_task(TaskCreateRequest(
                name="测试",
                description="",
                stock_codes=["000001.SZ"],
                indicators=[],
                models=["LSTM"],
                parameters={}
            ))
        
        # 空模型列表
        with pytest.raises(ValueError, match="模型列表不能为空"):
            self.task_manager.create_task(TaskCreateRequest(
                name="测试",
                description="",
                stock_codes=["000001.SZ"],
                indicators=["MA5"],
                models=[],
                parameters={}
            ))
    
    def test_update_task_status(self):
        """测试任务状态更新"""
        # 创建任务
        request = TaskCreateRequest(
            name="状态测试任务",
            description="用于测试状态更新",
            stock_codes=["000001.SZ"],
            indicators=["MA5"],
            models=["LSTM"],
            parameters={}
        )
        
        task_id = self.task_manager.create_task(request)
        
        # 更新为运行中
        success = self.task_manager.update_task(TaskUpdateRequest(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            progress=25.0
        ))
        
        assert success
        
        task = self.task_manager.get_task(task_id)
        assert task.status == TaskStatus.RUNNING
        assert task.progress == 25.0
        assert task.started_at is not None
        
        # 更新为完成
        success = self.task_manager.update_task(TaskUpdateRequest(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0
        ))
        
        assert success
        
        task = self.task_manager.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.progress == 100.0
        assert task.completed_at is not None
    
    def test_update_task_error(self):
        """测试任务错误更新"""
        # 创建任务
        request = TaskCreateRequest(
            name="错误测试任务",
            description="用于测试错误处理",
            stock_codes=["000001.SZ"],
            indicators=["MA5"],
            models=["LSTM"],
            parameters={}
        )
        
        task_id = self.task_manager.create_task(request)
        
        # 更新为失败状态
        error_message = "模型训练失败：数据不足"
        success = self.task_manager.update_task(TaskUpdateRequest(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error_message=error_message
        ))
        
        assert success
        
        task = self.task_manager.get_task(task_id)
        assert task.status == TaskStatus.FAILED
        assert task.error_message == error_message
        assert task.completed_at is not None
    
    def test_query_tasks(self):
        """测试任务查询"""
        # 创建多个任务
        requests = [
            TaskCreateRequest(
                name=f"查询测试任务{i}",
                description="用于查询测试",
                stock_codes=[f"{i:06d}.SZ"],
                indicators=["MA5"],
                models=["LSTM"],
                parameters={}
            )
            for i in range(1, 6)
        ]
        
        task_ids = []
        for request in requests:
            task_id = self.task_manager.create_task(request)
            task_ids.append(task_id)
        
        # 更新部分任务状态
        self.task_manager.update_task(TaskUpdateRequest(task_ids[0], status=TaskStatus.RUNNING))
        self.task_manager.update_task(TaskUpdateRequest(task_ids[1], status=TaskStatus.COMPLETED))
        
        # 查询所有任务
        all_tasks = self.task_manager.query_tasks(TaskQuery())
        assert len(all_tasks) == 5
        
        # 查询运行中的任务
        running_tasks = self.task_manager.query_tasks(TaskQuery(status=TaskStatus.RUNNING))
        assert len(running_tasks) == 1
        assert running_tasks[0].status == TaskStatus.RUNNING
        
        # 查询已完成的任务
        completed_tasks = self.task_manager.query_tasks(TaskQuery(status=TaskStatus.COMPLETED))
        assert len(completed_tasks) == 1
        assert completed_tasks[0].status == TaskStatus.COMPLETED
        
        # 查询待处理的任务
        pending_tasks = self.task_manager.query_tasks(TaskQuery(status=TaskStatus.PENDING))
        assert len(pending_tasks) == 3
    
    def test_query_tasks_with_stock_filter(self):
        """测试按股票代码过滤查询"""
        # 创建包含不同股票的任务
        requests = [
            TaskCreateRequest(
                name="股票过滤测试1",
                description="",
                stock_codes=["000001.SZ", "000002.SZ"],
                indicators=["MA5"],
                models=["LSTM"],
                parameters={}
            ),
            TaskCreateRequest(
                name="股票过滤测试2",
                description="",
                stock_codes=["000003.SZ"],
                indicators=["MA5"],
                models=["LSTM"],
                parameters={}
            )
        ]
        
        for request in requests:
            self.task_manager.create_task(request)
        
        # 按股票代码过滤
        filtered_tasks = self.task_manager.query_tasks(TaskQuery(stock_code="000001.SZ"))
        assert len(filtered_tasks) == 1
        assert "000001.SZ" in filtered_tasks[0].name or "股票过滤测试1" == filtered_tasks[0].name
    
    def test_delete_task(self):
        """测试任务删除"""
        # 创建任务
        request = TaskCreateRequest(
            name="删除测试任务",
            description="用于测试删除",
            stock_codes=["000001.SZ"],
            indicators=["MA5"],
            models=["LSTM"],
            parameters={}
        )
        
        task_id = self.task_manager.create_task(request)
        
        # 验证任务存在
        task = self.task_manager.get_task(task_id)
        assert task is not None
        
        # 删除任务
        success = self.task_manager.delete_task(task_id)
        assert success
        
        # 验证任务不存在
        deleted_task = self.task_manager.get_task(task_id)
        assert deleted_task is None
        
        # 重复删除应该返回False
        success_again = self.task_manager.delete_task(task_id)
        assert not success_again
    
    def test_task_statistics(self):
        """测试任务统计"""
        # 创建不同状态的任务
        requests = [
            TaskCreateRequest(
                name=f"统计测试任务{i}",
                description="用于统计测试",
                stock_codes=["000001.SZ"],
                indicators=["MA5"],
                models=["LSTM"],
                parameters={}
            )
            for i in range(4)
        ]
        
        task_ids = []
        for request in requests:
            task_id = self.task_manager.create_task(request)
            task_ids.append(task_id)
        
        # 更新任务状态
        self.task_manager.update_task(TaskUpdateRequest(task_ids[0], status=TaskStatus.RUNNING))
        self.task_manager.update_task(TaskUpdateRequest(task_ids[1], status=TaskStatus.COMPLETED))
        self.task_manager.update_task(TaskUpdateRequest(task_ids[2], status=TaskStatus.FAILED))
        # task_ids[3] 保持 PENDING 状态
        
        # 获取统计信息
        stats = self.task_manager.get_task_statistics()
        
        # 验证统计
        assert stats['total_tasks'] == 4
        assert stats['pending_tasks'] == 1
        assert stats['running_tasks'] == 1
        assert stats['completed_tasks'] == 1
        assert stats['failed_tasks'] == 1
        assert len(stats['recent_tasks']) <= 4
    
    def test_task_callbacks(self):
        """测试任务回调"""
        status_changes = []
        progress_updates = []
        
        def status_callback(task_id, status):
            status_changes.append((task_id, status))
        
        def progress_callback(task_id, progress):
            progress_updates.append((task_id, progress))
        
        # 注册回调
        self.task_manager.add_status_change_callback(status_callback)
        self.task_manager.add_progress_callback(progress_callback)
        
        # 创建任务
        request = TaskCreateRequest(
            name="回调测试任务",
            description="用于测试回调",
            stock_codes=["000001.SZ"],
            indicators=["MA5"],
            models=["LSTM"],
            parameters={}
        )
        
        task_id = self.task_manager.create_task(request)
        
        # 验证创建时的状态回调
        assert len(status_changes) == 1
        assert status_changes[0] == (task_id, TaskStatus.PENDING)
        
        # 更新状态和进度
        self.task_manager.update_task(TaskUpdateRequest(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            progress=50.0
        ))
        
        # 验证回调
        assert len(status_changes) == 2
        assert status_changes[1] == (task_id, TaskStatus.RUNNING)
        assert len(progress_updates) == 1
        assert progress_updates[0] == (task_id, 50.0)
    
    def test_nonexistent_task_operations(self):
        """测试对不存在任务的操作"""
        nonexistent_id = 99999
        
        # 获取不存在的任务
        task = self.task_manager.get_task(nonexistent_id)
        assert task is None
        
        # 更新不存在的任务
        success = self.task_manager.update_task(TaskUpdateRequest(
            task_id=nonexistent_id,
            status=TaskStatus.RUNNING
        ))
        assert not success
        
        # 删除不存在的任务
        success = self.task_manager.delete_task(nonexistent_id)
        assert not success
        
        # 获取不存在任务的结果
        results = self.task_manager.get_task_results(nonexistent_id)
        assert len(results) == 0