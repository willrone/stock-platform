"""
基础设施属性测试
功能: production-ready-implementation
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, AsyncMock

from app.core.error_handler import (
    BaseError,
    ErrorRecoveryManager,
    ErrorSeverity,
    ErrorType,
    ModelError,
    PredictionError,
    TaskError,
)
from app.core.logging_config import AuditLogger, LogContext, PerformanceLogger
from app.models.task_models import Task, TaskStatus, TaskType
from app.services.infrastructure.websocket_manager import (
    ClientConnection,
    WebSocketManager,
    WebSocketMessage,
)


class TestTaskManagementIntegrity:
    """属性 2: 任务管理完整性测试"""

    @given(
        task_name=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N"))),
        task_type=st.sampled_from([t.value for t in TaskType]),
        user_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=50)
    def test_task_creation_integrity(self, task_name, task_type, user_id):
        """验证任务创建的完整性 - ORM 模型属性赋值"""
        # Task 是 ORM 模型，task_id/status/progress/created_at 由数据库 server_default 生成
        # 在内存中创建时这些字段为 None，所以我们手动设置
        task = Task(
            task_id=uuid.uuid4(),
            task_name=task_name,
            task_type=task_type,
            user_id=user_id,
            status=TaskStatus.CREATED.value,
            progress=0.0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            config={},
        )

        assert task.task_id is not None
        assert task.task_name == task_name
        assert task.task_type == task_type
        assert task.user_id == user_id
        assert task.status == TaskStatus.CREATED.value
        assert task.progress == 0.0
        assert task.created_at is not None

        task_dict = task.to_dict()
        assert isinstance(task_dict, dict)
        assert task_dict["task_name"] == task_name
        assert task_dict["task_type"] == task_type

    @given(
        progress=st.floats(min_value=0.0, max_value=1.0),
        status=st.sampled_from([s.value for s in TaskStatus]),
    )
    @settings(max_examples=50)
    def test_task_status_update_integrity(self, progress, status):
        """验证任务状态更新的完整性"""
        task = Task(
            task_id=uuid.uuid4(),
            task_name="测试任务",
            task_type=TaskType.PREDICTION.value,
            user_id="test_user",
            status=TaskStatus.CREATED.value,
            progress=0.0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        task.status = status
        task.progress = progress

        if status in [TaskStatus.RUNNING.value, TaskStatus.COMPLETED.value]:
            task.started_at = datetime.now()

        if status == TaskStatus.COMPLETED.value:
            task.completed_at = datetime.now()

        assert task.status == status
        assert task.progress == progress

        if status in [TaskStatus.RUNNING.value, TaskStatus.COMPLETED.value]:
            assert task.started_at is not None

        if status == TaskStatus.COMPLETED.value:
            assert task.completed_at is not None
            assert task.completed_at >= task.started_at


class TestNotificationServiceReliability:
    """属性 5: 通知服务可靠性测试"""

    def setup_method(self):
        self.websocket_manager = WebSocketManager()

    @pytest.mark.asyncio
    @given(
        connection_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        user_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        task_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=20, deadline=10000)
    async def test_websocket_connection_management(self, connection_id, user_id, task_id):
        """验证WebSocket连接管理的可靠性"""
        ws_mgr = WebSocketManager()
        mock_websocket = AsyncMock()

        connection = ClientConnection(websocket=mock_websocket, user_id=user_id)

        assert connection.websocket == mock_websocket
        assert connection.user_id == user_id
        assert isinstance(connection.subscribed_tasks, set)
        assert isinstance(connection.connected_at, datetime)

        ws_mgr.active_connections[connection_id] = connection
        if user_id not in ws_mgr.user_connections:
            ws_mgr.user_connections[user_id] = set()
        ws_mgr.user_connections[user_id].add(connection_id)

        assert connection_id in ws_mgr.active_connections

        await ws_mgr.subscribe_task(connection_id, task_id)
        assert task_id in connection.subscribed_tasks

        await ws_mgr.disconnect(connection_id)
        assert connection_id not in ws_mgr.active_connections

    @pytest.mark.asyncio
    @given(
        task_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        status=st.sampled_from([s.value for s in TaskStatus]),
        progress=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=20, deadline=10000)
    async def test_task_status_notification_reliability(self, task_id, status, progress):
        """验证任务状态通知的可靠性"""
        message_data = {
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "timestamp": datetime.now().isoformat(),
        }

        message = WebSocketMessage(type="task_status", data=message_data)

        assert message.type == "task_status"
        assert message.data["task_id"] == task_id
        assert message.data["status"] == status

        json_message = message.to_json()
        assert isinstance(json_message, str)

        parsed = json.loads(json_message)
        assert parsed["type"] == "task_status"
        assert parsed["data"]["task_id"] == task_id


class TestLoggingSystemIntegrity:
    """属性 10: 日志系统完整性测试"""

    @given(
        action=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        user_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        resource=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        success=st.booleans(),
    )
    @settings(max_examples=20)
    def test_audit_logging_integrity(self, action, user_id, resource, success):
        """验证审计日志的完整性"""
        request_id = f"req_{datetime.now().timestamp()}"

        with LogContext(request_id=request_id, user_id=user_id):
            AuditLogger.log_user_action(
                action=action,
                user_id=user_id,
                resource=resource,
                success=success,
            )
            from app.core.logging_config import request_id_var, user_id_var

            assert request_id_var.get() == request_id
            assert user_id_var.get() == user_id

    @given(
        endpoint=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        method=st.sampled_from(["GET", "POST", "PUT", "DELETE"]),
        duration_ms=st.floats(min_value=0.1, max_value=10000.0),
        status_code=st.integers(min_value=200, max_value=599),
    )
    @settings(max_examples=20)
    def test_performance_logging_integrity(self, endpoint, method, duration_ms, status_code):
        """验证性能日志的完整性"""
        PerformanceLogger.log_api_performance(
            endpoint=endpoint,
            method=method,
            duration_ms=duration_ms,
            status_code=status_code,
            user_id="test_user",
        )

    @given(
        table=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
        operation=st.sampled_from(["INSERT", "UPDATE", "DELETE"]),
        record_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N"))),
    )
    @settings(max_examples=20)
    def test_data_change_logging_integrity(self, table, operation, record_id):
        """验证数据变更日志的完整性"""
        AuditLogger.log_data_change(
            table=table,
            operation=operation,
            record_id=record_id,
            old_values={"field1": "old"},
            new_values={"field1": "new"},
            user_id="test_user",
        )


class TestErrorHandlingReliability:
    """错误处理可靠性测试"""

    def setup_method(self):
        self.error_manager = ErrorRecoveryManager()

    @given(
        error_message=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N"))),
        error_type=st.sampled_from([t for t in ErrorType]),
        severity=st.sampled_from([s for s in ErrorSeverity]),
    )
    @settings(max_examples=50)
    def test_error_handling_integrity(self, error_message, error_type, severity):
        """验证错误处理的完整性"""
        mgr = ErrorRecoveryManager()
        error = BaseError(message=error_message, error_type=error_type, severity=severity)

        assert error.message == error_message
        assert error.error_type == error_type
        assert error.severity == severity
        assert error.error_id is not None
        assert isinstance(error.timestamp, datetime)

        recovery_actions = mgr.handle_error(error)
        assert isinstance(recovery_actions, list)
        assert error in mgr.error_history

        error_dict = error.to_dict()
        assert isinstance(error_dict, dict)
        assert error_dict["message"] == error_message
        assert error_dict["error_type"] == error_type.value
        assert error_dict["severity"] == severity.value

    @given(prediction_error_msg=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N"))))
    @settings(max_examples=50)
    def test_prediction_error_recovery(self, prediction_error_msg):
        """验证预测错误的恢复策略"""
        mgr = ErrorRecoveryManager()
        error = PredictionError(message=prediction_error_msg, severity=ErrorSeverity.MEDIUM)

        recovery_actions = mgr.handle_error(error)
        assert len(recovery_actions) > 0

        for action in recovery_actions:
            assert hasattr(action, "action_type")
            assert hasattr(action, "parameters")
            assert hasattr(action, "description")

    @given(task_error_msg=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N"))))
    @settings(max_examples=50)
    def test_task_error_recovery(self, task_error_msg):
        """验证任务错误的恢复策略"""
        mgr = ErrorRecoveryManager()
        error = TaskError(message=task_error_msg, severity=ErrorSeverity.HIGH)

        recovery_actions = mgr.handle_error(error)
        # TaskError handler 可能返回空列表（如果 message 不匹配任何关键词）
        assert isinstance(recovery_actions, list)

        stats = mgr.get_error_statistics(hours=1)
        assert isinstance(stats, dict)
        assert "total_errors" in stats
        assert "error_counts_by_type" in stats
        assert "error_counts_by_severity" in stats
        assert stats["total_errors"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
