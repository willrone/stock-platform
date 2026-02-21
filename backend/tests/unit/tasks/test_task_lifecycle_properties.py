"""
任务生命周期管理属性测试
Feature: stock-prediction-platform, Property 5: 任务生命周期管理

注意：DatabaseManager 已废弃，迁移到 SQLAlchemy ORM。
整个模块跳过，待重写为基于 SQLAlchemy 的测试。
"""

import pytest

try:
    from app.models.database import DatabaseManager, TaskStatus
except ImportError:
    pytest.skip(
        "DatabaseManager 已废弃（迁移到 SQLAlchemy），整个测试模块跳过",
        allow_module_level=True,
    )

import tempfile
import shutil
import json
from datetime import datetime, timedelta

from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.services.tasks import TaskManager, TaskCreateRequest, TaskUpdateRequest, TaskQuery


@composite
def stock_code_strategy(draw):
    """生成有效的股票代码"""
    number = draw(st.integers(min_value=1, max_value=999999))
    market = draw(st.sampled_from(['SH', 'SZ']))
    return f"{number:06d}.{market}"


@composite
def task_create_request_strategy(draw):
    """生成任务创建请求"""
    name = draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    description = draw(st.text(min_size=0, max_size=200))
    stock_codes = draw(st.lists(stock_code_strategy(), min_size=1, max_size=5, unique=True))
    indicators = draw(st.lists(st.sampled_from(['MA5', 'MA10', 'RSI', 'MACD', 'BOLLINGER']), min_size=1, max_size=3, unique=True))
    models = draw(st.lists(st.sampled_from(['LSTM', 'XGBoost', 'Transformer']), min_size=1, max_size=2, unique=True))
    parameters = draw(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(max_size=50), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
        min_size=0,
        max_size=5
    ))
    
    return TaskCreateRequest(
        name=name,
        description=description,
        stock_codes=stock_codes,
        indicators=indicators,
        models=models,
        parameters=parameters
    )


class TestTaskLifecycleProperties:
    """任务生命周期管理属性测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = DatabaseManager(db_path=f"{self.temp_dir}/test.db")
        self.task_manager = TaskManager(self.db_manager)
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(request=task_create_request_strategy())
    @settings(max_examples=20, deadline=None)
    def test_task_creation_and_retrieval_property(self, request):
        """
        属性测试：任务创建和检索一致性
        
        对于任何有效的任务创建请求，系统应该：
        1. 成功创建任务并返回有效ID
        2. 能够通过ID检索到完整的任务信息
        3. 检索到的信息与创建时一致
        4. 任务初始状态为PENDING
        
        验证：需求 4.1, 4.2
        """
        # 创建任务
        task_id = self.task_manager.create_task(request)
        
        assert isinstance(task_id, int), "任务ID应该是整数"
        assert task_id > 0, "任务ID应该大于0"
        
        # 检索任务
        task = self.task_manager.get_task(task_id)
        
        assert task is not None, "应该能够检索到任务"
        assert task.id == task_id, "任务ID应该一致"
        assert task.name == request.name.strip(), "任务名称应该一致（去除空格）"
        assert task.description == request.description, "任务描述应该一致"
        assert task.status == TaskStatus.PENDING, "初始状态应该是PENDING"
        assert task.progress == 0.0, "初始进度应该是0"
        assert task.created_at is not None, "创建时间应该存在"
        assert task.started_at is None, "开始时间应该为空"
        assert task.completed_at is None, "完成时间应该为空"
        
        # 验证JSON字段解析
        parsed_stock_codes = json.loads(task.stock_codes)
        parsed_indicators = json.loads(task.indicators)
        parsed_models = json.loads(task.models)
        parsed_parameters = json.loads(task.parameters)
        
        assert parsed_stock_codes == request.stock_codes, "股票代码列表应该一致"
        assert parsed_indicators == request.indicators, "指标列表应该一致"
        assert parsed_models == request.models, "模型列表应该一致"
        assert parsed_parameters == request.parameters, "参数应该一致"
    
    @given(
        request=task_create_request_strategy(),
        status_sequence=st.lists(
            st.sampled_from([TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED]),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=15, deadline=None)
    def test_task_status_transition_property(self, request, status_sequence):
        """
        属性测试：任务状态转换一致性
        
        对于任何任务状态转换序列，系统应该：
        1. 正确更新任务状态
        2. 自动设置相应的时间戳
        3. 状态转换历史可追踪
        4. 进度值在合理范围内
        
        验证：需求 4.2, 4.3
        """
        # 创建任务
        task_id = self.task_manager.create_task(request)
        
        previous_status = TaskStatus.PENDING
        
        for i, new_status in enumerate(status_sequence):
            # 更新状态
            progress = min(100.0, (i + 1) * 30.0)  # 递增进度
            
            update_request = TaskUpdateRequest(
                task_id=task_id,
                status=new_status,
                progress=progress
            )
            
            success = self.task_manager.update_task(update_request)
            assert success, f"状态更新应该成功: {previous_status} -> {new_status}"
            
            # 验证更新结果
            updated_task = self.task_manager.get_task(task_id)
            assert updated_task is not None, "更新后应该能检索到任务"
            assert updated_task.status == new_status, f"状态应该更新为 {new_status}"
            assert updated_task.progress == progress, f"进度应该更新为 {progress}"
            
            # 验证时间戳设置
            if new_status == TaskStatus.RUNNING and previous_status == TaskStatus.PENDING:
                assert updated_task.started_at is not None, "运行状态应该设置开始时间"
            
            if new_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                assert updated_task.completed_at is not None, "完成状态应该设置完成时间"
            
            previous_status = new_status
    
    @given(
        requests=st.lists(task_create_request_strategy(), min_size=1, max_size=10),
        query_status=st.one_of(st.none(), st.sampled_from(list(TaskStatus)))
    )
    @settings(max_examples=10, deadline=None)
    def test_task_query_consistency_property(self, requests, query_status):
        """
        属性测试：任务查询一致性
        
        对于任何任务集合和查询条件，系统应该：
        1. 返回符合条件的任务
        2. 查询结果与实际存储一致
        3. 分页功能正确工作
        4. 统计信息准确
        
        验证：需求 4.4, 4.5
        """
        # 创建多个任务
        created_task_ids = []
        for request in requests:
            task_id = self.task_manager.create_task(request)
            created_task_ids.append(task_id)
        
        # 随机更新一些任务状态
        if query_status and created_task_ids:
            # 将一些任务更新为查询状态
            for task_id in created_task_ids[:len(created_task_ids)//2]:
                self.task_manager.update_task(TaskUpdateRequest(
                    task_id=task_id,
                    status=query_status
                ))
        
        # 执行查询
        query = TaskQuery(
            status=query_status,
            limit=50,
            offset=0
        )
        
        results = self.task_manager.query_tasks(query)
        
        # 验证查询结果
        assert isinstance(results, list), "查询结果应该是列表"
        assert len(results) <= query.limit, "结果数量不应超过限制"
        
        # 验证每个结果的完整性
        for result in results:
            assert result.id > 0, "任务ID应该有效"
            assert isinstance(result.name, str), "任务名称应该是字符串"
            assert isinstance(result.status, TaskStatus), "状态应该是TaskStatus枚举"
            assert 0 <= result.progress <= 100, "进度应该在0-100之间"
            assert result.created_at is not None, "创建时间应该存在"
            assert result.stock_count >= 0, "股票数量应该非负"
            assert result.result_count >= 0, "结果数量应该非负"
            
            # 如果指定了状态过滤，验证结果状态
            if query_status:
                assert result.status == query_status, f"查询结果状态应该匹配过滤条件: {result.status} == {query_status}"
        
        # 验证排序（按创建时间降序）
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].created_at >= results[i + 1].created_at, "结果应该按创建时间降序排列"
    
    def test_task_deletion_consistency(self):
        """测试任务删除一致性"""
        # 创建任务
        request = TaskCreateRequest(
            name="删除测试任务",
            description="用于测试删除功能",
            stock_codes=["000001.SZ"],
            indicators=["MA5"],
            models=["LSTM"],
            parameters={}
        )
        
        task_id = self.task_manager.create_task(request)
        
        # 验证任务存在
        task = self.task_manager.get_task(task_id)
        assert task is not None, "任务应该存在"
        
        # 删除任务
        success = self.task_manager.delete_task(task_id)
        assert success, "删除应该成功"
        
        # 验证任务不存在
        deleted_task = self.task_manager.get_task(task_id)
        assert deleted_task is None, "删除后任务应该不存在"
        
        # 验证重复删除
        success_again = self.task_manager.delete_task(task_id)
        assert not success_again, "重复删除应该返回False"
    
    def test_task_statistics_accuracy(self):
        """测试任务统计准确性"""
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
            for i in range(5)
        ]
        
        task_ids = []
        for request in requests:
            task_id = self.task_manager.create_task(request)
            task_ids.append(task_id)
        
        # 更新任务状态
        self.task_manager.update_task(TaskUpdateRequest(task_ids[0], status=TaskStatus.RUNNING))
        self.task_manager.update_task(TaskUpdateRequest(task_ids[1], status=TaskStatus.COMPLETED))
        self.task_manager.update_task(TaskUpdateRequest(task_ids[2], status=TaskStatus.FAILED))
        # task_ids[3] 和 task_ids[4] 保持 PENDING 状态
        
        # 获取统计信息
        stats = self.task_manager.get_task_statistics()
        
        # 验证统计准确性
        assert stats['total_tasks'] == 5, "总任务数应该正确"
        assert stats['pending_tasks'] == 2, "待处理任务数应该正确"
        assert stats['running_tasks'] == 1, "运行中任务数应该正确"
        assert stats['completed_tasks'] == 1, "已完成任务数应该正确"
        assert stats['failed_tasks'] == 1, "失败任务数应该正确"
        assert len(stats['recent_tasks']) <= 5, "最近任务数量应该合理"
    
    def test_concurrent_task_operations(self):
        """测试并发任务操作"""
        import threading
        import time
        
        results = []
        errors = []
        
        def create_task_worker(worker_id):
            try:
                request = TaskCreateRequest(
                    name=f"并发测试任务{worker_id}",
                    description="并发创建测试",
                    stock_codes=[f"{worker_id:06d}.SZ"],
                    indicators=["MA5"],
                    models=["LSTM"],
                    parameters={"worker_id": worker_id}
                )
                
                task_id = self.task_manager.create_task(request)
                results.append(task_id)
                
                # 立即更新状态
                self.task_manager.update_task(TaskUpdateRequest(
                    task_id=task_id,
                    status=TaskStatus.RUNNING,
                    progress=50.0
                ))
                
            except Exception as e:
                errors.append(str(e))
        
        # 创建多个线程并发执行
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_task_worker, args=(i,))
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(errors) == 0, f"并发操作不应该有错误: {errors}"
        assert len(results) == 5, "应该创建5个任务"
        assert len(set(results)) == 5, "任务ID应该唯一"
        
        # 验证所有任务都能正确检索
        for task_id in results:
            task = self.task_manager.get_task(task_id)
            assert task is not None, f"任务 {task_id} 应该存在"
            assert task.status == TaskStatus.RUNNING, "任务状态应该是RUNNING"
            assert task.progress == 50.0, "任务进度应该是50%"