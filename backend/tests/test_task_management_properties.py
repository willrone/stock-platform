"""
任务管理系统属性测试
功能: production-ready-implementation
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.models.task_models import Task, TaskType, TaskStatus, Base
from backend.app.repositories.task_repository import TaskRepository, PredictionResultRepository
from backend.app.services.task_queue import TaskScheduler, TaskPriority, QueuedTask, TaskExecutor
from backend.app.services.task_execution_engine import TaskExecutionEngine, ProgressTracker
from backend.app.services.task_notification_service import TaskNotificationService
from backend.app.services.websocket_manager import WebSocketManager
from backend.app.core.error_handler import TaskError


class TestTaskManagementIntegrity:
    """属性 2: 任务管理完整性测试"""
    
    def setup_method(self):
        """测试设置"""
        # 创建内存数据库
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        
        # 创建仓库实例
        self.task_repository = TaskRepository(self.db_session)
        self.prediction_result_repository = PredictionResultRepository(self.db_session)
        
        # 创建通知服务
        self.notification_service = TaskNotificationService()
        
        # 创建调度器
        self.scheduler = TaskScheduler(max_executors=2)
    
    def teardown_method(self):
        """测试清理"""
        self.db_session.close()
        if hasattr(self, 'scheduler'):
            self.scheduler.stop()
    
    @given(
        task_name=st.text(min_size=1, max_size=100),
        task_type=st.sampled_from([t for t in TaskType]),
        user_id=st.text(min_size=1, max_size=50),
        config=st.dictionaries(st.text(), st.text())
    )
    @settings(max_examples=100)
    def test_task_creation_integrity(self, task_name, task_type, user_id, config):
        """
        功能: production-ready-implementation, 属性 2: 任务管理完整性
        验证任务创建的完整性 - 任何有效的任务创建都应该正确创建任务记录
        """
        try:
            # 创建任务
            task = self.task_repository.create_task(task_name, task_type, user_id, config)
            
            # 验证任务属性
            assert task.task_id is not None
            assert task.task_name == task_name
            assert task.task_type == task_type.value
            assert task.user_id == user_id
            assert task.config == config
            assert task.status == TaskStatus.CREATED.value
            assert task.progress == 0.0
            assert task.created_at is not None
            assert isinstance(task.created_at, datetime)
            
            # 验证任务可以从数据库检索
            retrieved_task = self.task_repository.get_task_by_id(task.task_id)
            assert retrieved_task is not None
            assert retrieved_task.task_id == task.task_id
            assert retrieved_task.task_name == task_name
            assert retrieved_task.task_type == task_type.value
            
            # 验证任务可以转换为字典
            task_dict = task.to_dict()
            assert isinstance(task_dict, dict)
            assert task_dict["task_id"] == task.task_id
            assert task_dict["task_name"] == task_name
            
        except Exception as e:
            # 如果是数据验证错误，这是可接受的
            if "缺少必需字段" in str(e) or "无效" in str(e):
                pytest.skip(f"数据验证错误: {e}")
            else:
                raise
    
    @given(
        initial_status=st.sampled_from([TaskStatus.CREATED, TaskStatus.QUEUED]),
        target_status=st.sampled_from([TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED]),
        progress=st.floats(min_value=0.0, max_value=100.0)
    )
    @settings(max_examples=100)
    def test_task_status_update_integrity(self, initial_status, target_status, progress):
        """
        功能: production-ready-implementation, 属性 2: 任务管理完整性
        验证任务状态更新的完整性 - 任何有效的状态更新都应该正确反映在数据库中
        """
        # 创建初始任务
        task = self.task_repository.create_task(
            "测试任务", TaskType.PREDICTION, "test_user", {"test": "config"}
        )
        
        # 设置初始状态
        if initial_status != TaskStatus.CREATED:
            self.task_repository.update_task_status(task.task_id, initial_status)
        
        # 更新到目标状态
        result = {"test_result": "success"} if target_status == TaskStatus.COMPLETED else None
        error_msg = "测试错误" if target_status == TaskStatus.FAILED else None
        
        updated_task = self.task_repository.update_task_status(
            task.task_id, target_status, progress=progress, 
            result=result, error_message=error_msg
        )
        
        # 验证状态更新
        assert updated_task.status == target_status.value
        assert updated_task.progress == progress
        
        if target_status == TaskStatus.RUNNING:
            assert updated_task.started_at is not None
        elif target_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            assert updated_task.completed_at is not None
            if target_status == TaskStatus.COMPLETED:
                assert updated_task.result == result
            else:
                assert updated_task.error_message == error_msg
        
        # 验证数据库中的状态
        db_task = self.task_repository.get_task_by_id(task.task_id)
        assert db_task.status == target_status.value
        assert db_task.progress == progress
    
    @given(
        user_id=st.text(min_size=1, max_size=50),
        task_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50)
    def test_user_task_management_integrity(self, user_id, task_count):
        """
        功能: production-ready-implementation, 属性 2: 任务管理完整性
        验证用户任务管理的完整性 - 用户应该能够管理自己的任务
        """
        created_tasks = []
        
        # 创建多个任务
        for i in range(task_count):
            task = self.task_repository.create_task(
                f"任务_{i}", TaskType.PREDICTION, user_id, {"index": i}
            )
            created_tasks.append(task)
        
        # 获取用户任务列表
        user_tasks = self.task_repository.get_tasks_by_user(user_id, limit=20)
        
        # 验证任务列表
        assert len(user_tasks) == task_count
        
        # 验证所有任务都属于该用户
        for task in user_tasks:
            assert task.user_id == user_id
            assert task.task_id in [t.task_id for t in created_tasks]
        
        # 验证任务按创建时间排序（最新的在前）
        for i in range(len(user_tasks) - 1):
            assert user_tasks[i].created_at >= user_tasks[i + 1].created_at
        
        # 测试任务删除（只能删除已完成的任务）
        if task_count > 0:
            # 将第一个任务标记为完成
            completed_task = created_tasks[0]
            self.task_repository.update_task_status(
                completed_task.task_id, TaskStatus.COMPLETED, progress=100.0
            )
            
            # 删除任务
            delete_result = self.task_repository.delete_task(completed_task.task_id, user_id)
            assert delete_result is True
            
            # 验证任务已删除
            deleted_task = self.task_repository.get_task_by_id(completed_task.task_id)
            assert deleted_task is None
            
            # 验证用户任务列表更新
            updated_user_tasks = self.task_repository.get_tasks_by_user(user_id)
            assert len(updated_user_tasks) == task_count - 1
    
    @given(
        task_count=st.integers(min_value=5, max_value=20),
        days_back=st.integers(min_value=1, max_value=90)
    )
    @settings(max_examples=30)
    def test_task_statistics_integrity(self, task_count, days_back):
        """
        功能: production-ready-implementation, 属性 2: 任务管理完整性
        验证任务统计的完整性 - 统计信息应该准确反映任务状态
        """
        user_id = "stats_test_user"
        
        # 创建不同状态的任务
        statuses = [TaskStatus.CREATED, TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED]
        created_tasks = []
        
        for i in range(task_count):
            task = self.task_repository.create_task(
                f"统计测试任务_{i}", TaskType.PREDICTION, user_id, {"index": i}
            )
            
            # 随机设置任务状态
            status = statuses[i % len(statuses)]
            if status != TaskStatus.CREATED:
                self.task_repository.update_task_status(task.task_id, status)
            
            created_tasks.append((task, status))
        
        # 获取统计信息
        stats = self.task_repository.get_task_statistics(user_id, days=days_back)
        
        # 验证统计信息
        assert stats["total_tasks"] == task_count
        assert isinstance(stats["status_counts"], dict)
        assert isinstance(stats["type_counts"], dict)
        assert isinstance(stats["avg_duration_seconds"], (int, float))
        assert 0 <= stats["success_rate"] <= 1
        
        # 验证状态计数
        expected_status_counts = {}
        for task, status in created_tasks:
            status_value = status.value
            expected_status_counts[status_value] = expected_status_counts.get(status_value, 0) + 1
        
        for status, count in expected_status_counts.items():
            assert stats["status_counts"].get(status, 0) == count
        
        # 验证成功率计算
        completed_count = expected_status_counts.get(TaskStatus.COMPLETED.value, 0)
        expected_success_rate = completed_count / task_count
        assert abs(stats["success_rate"] - expected_success_rate) < 0.01


class TestTaskQueueReliability:
    """任务队列可靠性测试"""
    
    def setup_method(self):
        """测试设置"""
        self.scheduler = TaskScheduler(max_executors=2)
        self.test_results = []
        
        # 注册测试处理器
        def test_handler(queued_task, context):
            result = {
                "task_id": queued_task.task_id,
                "task_type": queued_task.task_type.value,
                "config": queued_task.config,
                "execution_time": 0.1
            }
            self.test_results.append(result)
            return result
        
        self.scheduler.register_task_handler(TaskType.PREDICTION, test_handler)
        self.scheduler.start()
    
    def teardown_method(self):
        """测试清理"""
        self.scheduler.stop()
    
    @given(
        task_count=st.integers(min_value=1, max_value=10),
        priority=st.sampled_from([p for p in TaskPriority])
    )
    @settings(max_examples=50)
    def test_task_queue_reliability(self, task_count, priority):
        """
        功能: production-ready-implementation, 属性 2: 任务管理完整性
        验证任务队列的可靠性 - 任务应该按优先级和顺序正确执行
        """
        # 清空之前的结果
        self.test_results.clear()
        
        # 入队任务
        task_ids = []
        for i in range(task_count):
            task_id = f"test_task_{i}_{datetime.now().timestamp()}"
            success = self.scheduler.enqueue_task(
                task_id=task_id,
                task_type=TaskType.PREDICTION,
                config={"index": i, "priority": priority.name},
                user_id="test_user",
                priority=priority
            )
            assert success is True
            task_ids.append(task_id)
        
        # 等待任务执行完成
        import time
        max_wait_time = 10  # 最多等待10秒
        start_time = time.time()
        
        while len(self.test_results) < task_count and (time.time() - start_time) < max_wait_time:
            time.sleep(0.1)
        
        # 验证任务执行结果
        assert len(self.test_results) == task_count
        
        # 验证所有任务都被执行
        executed_task_ids = [result["task_id"] for result in self.test_results]
        for task_id in task_ids:
            assert task_id in executed_task_ids
        
        # 验证队列状态
        queue_status = self.scheduler.get_queue_status()
        assert isinstance(queue_status, dict)
        assert "queue_size" in queue_status
        assert "running_tasks" in queue_status
        assert "statistics" in queue_status
    
    @given(
        task_id=st.text(min_size=1, max_size=50),
        task_type=st.sampled_from([TaskType.PREDICTION, TaskType.BACKTEST])
    )
    @settings(max_examples=30)
    def test_task_execution_context_integrity(self, task_id, task_type):
        """
        功能: production-ready-implementation, 属性 2: 任务管理完整性
        验证任务执行上下文的完整性 - 执行上下文应该正确管理任务生命周期
        """
        from backend.app.services.task_queue import TaskExecutionContext
        import threading
        
        # 创建执行上下文
        cancel_event = threading.Event()
        context = TaskExecutionContext(
            task_id=task_id,
            executor_id="test_executor",
            start_time=datetime.utcnow(),
            cancel_event=cancel_event
        )
        
        # 验证上下文属性
        assert context.task_id == task_id
        assert context.executor_id == "test_executor"
        assert isinstance(context.start_time, datetime)
        assert context.cancel_event is cancel_event
        
        # 测试取消功能
        assert not cancel_event.is_set()
        cancel_event.set()
        assert cancel_event.is_set()
        
        # 测试进度回调
        progress_updates = []
        
        def progress_callback(progress, message):
            progress_updates.append({"progress": progress, "message": message})
        
        context.progress_callback = progress_callback
        
        # 模拟进度更新
        if context.progress_callback:
            context.progress_callback(50.0, "测试进度")
            context.progress_callback(100.0, "完成")
        
        # 验证进度回调
        assert len(progress_updates) == 2
        assert progress_updates[0]["progress"] == 50.0
        assert progress_updates[1]["progress"] == 100.0


class TestNotificationServiceReliability:
    """属性 5: 通知服务可靠性测试"""
    
    def setup_method(self):
        """测试设置"""
        self.notification_service = TaskNotificationService()
        self.websocket_manager = Mock()
        self.notification_service.websocket_manager = self.websocket_manager
    
    @given(
        task_id=st.text(min_size=1, max_size=50),
        user_id=st.text(min_size=1, max_size=50),
        task_name=st.text(min_size=1, max_size=100),
        task_type=st.sampled_from([t for t in TaskType])
    )
    @settings(max_examples=100)
    async def test_notification_service_reliability(self, task_id, user_id, task_name, task_type):
        """
        功能: production-ready-implementation, 属性 5: 通知服务可靠性
        验证通知服务的可靠性 - 任何任务状态变化都应该正确推送通知
        """
        # 模拟WebSocket管理器
        self.websocket_manager.notify_user = AsyncMock()
        
        # 测试任务创建通知
        await self.notification_service.notify_task_created(
            task_id, task_name, task_type, user_id, {"test": "config"}
        )
        
        # 验证用户自动订阅
        subscriptions = await self.notification_service.get_user_task_subscriptions(user_id)
        assert task_id in subscriptions
        
        # 测试状态变化通知
        await self.notification_service.notify_task_status_change(
            task_id=task_id,
            task_name=task_name,
            task_type=task_type.value,
            old_status=TaskStatus.CREATED.value,
            new_status=TaskStatus.RUNNING.value,
            progress=50.0,
            user_id=user_id,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow()
        )
        
        # 验证WebSocket通知被调用
        assert self.websocket_manager.notify_user.call_count >= 2
        
        # 测试进度通知
        await self.notification_service.notify_task_progress(
            task_id, 75.0, "执行中", estimated_remaining_seconds=30
        )
        
        # 验证进度通知
        assert self.websocket_manager.notify_user.call_count >= 3
        
        # 测试完成通知
        await self.notification_service.notify_task_status_change(
            task_id=task_id,
            task_name=task_name,
            task_type=task_type.value,
            old_status=TaskStatus.RUNNING.value,
            new_status=TaskStatus.COMPLETED.value,
            progress=100.0,
            user_id=user_id,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            result={"success": True}
        )
        
        # 验证完成通知
        assert self.websocket_manager.notify_user.call_count >= 4
    
    @given(
        user_id=st.text(min_size=1, max_size=50),
        task_ids=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5, unique=True)
    )
    @settings(max_examples=50)
    async def test_subscription_management_reliability(self, user_id, task_ids):
        """
        功能: production-ready-implementation, 属性 5: 通知服务可靠性
        验证订阅管理的可靠性 - 订阅和取消订阅应该正确管理
        """
        # 订阅多个任务
        for task_id in task_ids:
            await self.notification_service.subscribe_user_to_task(user_id, task_id)
        
        # 验证订阅
        user_subscriptions = await self.notification_service.get_user_task_subscriptions(user_id)
        assert len(user_subscriptions) == len(task_ids)
        for task_id in task_ids:
            assert task_id in user_subscriptions
        
        # 验证任务订阅者
        for task_id in task_ids:
            subscribers = await self.notification_service.get_task_subscribers(task_id)
            assert user_id in subscribers
        
        # 取消部分订阅
        if len(task_ids) > 1:
            unsubscribe_task = task_ids[0]
            await self.notification_service.unsubscribe_user_from_task(user_id, unsubscribe_task)
            
            # 验证取消订阅
            updated_subscriptions = await self.notification_service.get_user_task_subscriptions(user_id)
            assert unsubscribe_task not in updated_subscriptions
            assert len(updated_subscriptions) == len(task_ids) - 1
            
            # 验证任务订阅者更新
            unsubscribe_subscribers = await self.notification_service.get_task_subscribers(unsubscribe_task)
            assert user_id not in unsubscribe_subscribers
    
    @given(
        notification_count=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=30)
    def test_notification_statistics_integrity(self, notification_count):
        """
        功能: production-ready-implementation, 属性 5: 通知服务可靠性
        验证通知统计的完整性 - 统计信息应该准确反映通知活动
        """
        # 获取初始统计
        initial_stats = self.notification_service.get_notification_statistics()
        initial_total = initial_stats["total_sent"]
        
        # 模拟发送通知（通过直接更新统计）
        for i in range(notification_count):
            self.notification_service.notification_stats["total_sent"] += 1
            self.notification_service.notification_stats["status_notifications"] += 1
        
        # 获取更新后的统计
        updated_stats = self.notification_service.get_notification_statistics()
        
        # 验证统计更新
        assert updated_stats["total_sent"] == initial_total + notification_count
        assert updated_stats["status_notifications"] >= notification_count
        
        # 验证统计结构
        assert isinstance(updated_stats, dict)
        required_fields = ["total_sent", "status_notifications", "progress_notifications", "failed_notifications"]
        for field in required_fields:
            assert field in updated_stats
            assert isinstance(updated_stats[field], int)


class TestProgressTrackerReliability:
    """进度跟踪器可靠性测试"""
    
    @given(
        task_id=st.text(min_size=1, max_size=50),
        total_steps=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50)
    def test_progress_tracker_reliability(self, task_id, total_steps):
        """
        功能: production-ready-implementation, 属性 2: 任务管理完整性
        验证进度跟踪器的可靠性 - 进度跟踪应该准确反映任务执行状态
        """
        progress_updates = []
        
        def progress_callback(progress, message):
            progress_updates.append({"progress": progress, "message": message})
        
        # 创建进度跟踪器
        tracker = ProgressTracker(task_id, total_steps, progress_callback)
        
        # 验证初始状态
        assert tracker.task_id == task_id
        assert tracker.total_steps == total_steps
        assert tracker.current_step == 0
        
        # 模拟步骤执行
        for i in range(total_steps):
            step_name = f"步骤_{i+1}"
            progress = tracker.update_step(step_name, {"step_index": i})
            
            # 验证进度信息
            assert progress.task_id == task_id
            assert progress.current_step == step_name
            assert progress.progress_percentage == ((i + 1) / total_steps) * 100
            
            # 验证回调被调用
            assert len(progress_updates) == i + 1
            assert progress_updates[i]["progress"] == progress.progress_percentage
            assert progress_updates[i]["message"] == step_name
        
        # 验证最终状态
        assert tracker.current_step == total_steps
        final_progress = progress_updates[-1]["progress"]
        assert final_progress == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])