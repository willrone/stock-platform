"""
任务 API contract tests

锁定 /tasks 相关核心接口的返回结构，避免 task/backtest DTO 再次漂移。
"""

import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

api_package = ModuleType("app.api")
api_package.__path__ = [str(BACKEND_ROOT / "app" / "api")]
v1_package = ModuleType("app.api.v1")
v1_package.__path__ = [str(BACKEND_ROOT / "app" / "api" / "v1")]
fake_torch = ModuleType("torch")
fake_torch.Tensor = type("Tensor", (), {})
sys.modules.setdefault("app.api", api_package)
sys.modules.setdefault("app.api.v1", v1_package)

with patch.dict(
    "sys.modules",
    {
        "qlib": MagicMock(),
        "qlib.config": MagicMock(),
        "qlib.data": MagicMock(),
        "qlib.model": MagicMock(),
        "vectorbt": MagicMock(),
        "vectorbt.portfolio": MagicMock(),
        "paramiko": MagicMock(),
        "torch": fake_torch,
    },
):
    from app.api.v1.dependencies import get_current_user
    from app.api.v1.tasks import router


FIXED_NOW = datetime(2026, 4, 8, 12, 0, 0)


@pytest.fixture
def app():
    """创建仅包含 tasks 路由的测试应用。"""

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: "test-user"
    return app


@pytest.fixture
def client(app):
    """创建测试客户端。"""

    return TestClient(app)


@pytest.fixture
def mock_session():
    """提供假的数据库 session。"""

    session = MagicMock()
    session.close = MagicMock()
    session.rollback = MagicMock()
    return session


@pytest.fixture
def prediction_task():
    """提供预测任务样本。"""

    return SimpleNamespace(
        task_id="task-pred-1",
        task_name="预测任务",
        task_type="prediction",
        status="created",
        progress=0.0,
        created_at=FIXED_NOW,
        completed_at=None,
        error_message=None,
        config={"stock_codes": ["000001.SZ"], "model_id": "model-v1"},
        result=None,
    )


@pytest.fixture
def backtest_task():
    """提供回测任务样本。"""

    return SimpleNamespace(
        task_id="task-bt-1",
        task_name="回测任务",
        task_type="backtest",
        status="completed",
        progress=100.0,
        created_at=FIXED_NOW,
        completed_at=FIXED_NOW,
        error_message=None,
        config={
            "stock_codes": ["000001.SZ", "000002.SZ"],
            "strategy_name": "multi_factor",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "initial_cash": 100000,
        },
        result={
            "summary": {"annual_return": 0.12},
            "metrics": {"sharpe_ratio": 1.5},
        },
    )


@pytest.fixture
def prediction_row():
    """提供预测结果样本。"""

    return SimpleNamespace(
        stock_code="000001.SZ",
        predicted_direction=1,
        predicted_price=10.32,
        prediction_date=FIXED_NOW,
        predicted_return=0.032,
        confidence_score=0.91,
        confidence_interval_lower=0.01,
        confidence_interval_upper=0.05,
        risk_metrics={"value_at_risk": 0.02, "volatility": 0.18},
    )


class TestTaskContractAPI:
    """任务 API contract tests。"""

    @patch("app.api.v1.tasks.get_process_executor")
    @patch("app.api.v1.tasks.TaskRepository")
    @patch("app.api.v1.tasks.SessionLocal")
    def test_create_task_contract(
        self,
        mock_session_local,
        mock_repository_cls,
        mock_executor_factory,
        client,
        mock_session,
        prediction_task,
    ):
        """创建任务接口返回统一任务 DTO。"""

        mock_session_local.return_value = mock_session
        mock_repository = MagicMock()
        mock_repository.create_task.return_value = prediction_task
        mock_repository_cls.return_value = mock_repository
        mock_executor_factory.return_value = MagicMock(submit=MagicMock())

        response = client.post(
            "/tasks",
            json={
                "task_name": "预测任务",
                "task_type": "prediction",
                "stock_codes": ["000001.SZ"],
                "model_id": "model-v1",
                "prediction_config": {"horizon": "short_term"},
            },
        )

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["task_id"] == "task-pred-1"
        assert payload["task_name"] == "预测任务"
        assert payload["task_type"] == "prediction"
        assert payload["status"] == "created"
        assert payload["stock_codes"] == ["000001.SZ"]
        assert payload["model_id"] == "model-v1"
        assert payload["config"]["model_id"] == "model-v1"
        assert payload["original_task_id"] is None

    @patch("app.api.v1.tasks.log_structured_exception")
    @patch("app.api.v1.tasks.get_process_executor")
    @patch("app.api.v1.tasks.TaskRepository")
    @patch("app.api.v1.tasks.SessionLocal")
    def test_create_task_submit_failure_marks_task_failed(
        self,
        mock_session_local,
        mock_repository_cls,
        mock_executor_factory,
        mock_log_structured_exception,
        client,
        mock_session,
        prediction_task,
    ):
        """任务提交失败时应显式回写 FAILED，而不是静默吞错。"""

        mock_session_local.return_value = mock_session
        mock_repository = MagicMock()
        mock_repository.create_task.return_value = prediction_task
        mock_repository_cls.return_value = mock_repository

        executor = MagicMock()
        executor.submit.side_effect = RuntimeError("pool down")
        mock_executor_factory.return_value = executor

        response = client.post(
            "/tasks",
            json={
                "task_name": "预测任务",
                "task_type": "prediction",
                "stock_codes": ["000001.SZ"],
                "model_id": "model-v1",
                "prediction_config": {"horizon": "short_term"},
            },
        )

        assert response.status_code == 200
        mock_log_structured_exception.assert_called_once()
        assert mock_repository.update_task_status.call_count == 1
        update_kwargs = mock_repository.update_task_status.call_args.kwargs
        assert update_kwargs["task_id"] == "task-pred-1"
        assert "任务提交失败: pool down" == update_kwargs["error_message"]

    @patch("app.api.v1.tasks.TaskRepository")
    @patch("app.api.v1.tasks.SessionLocal")
    def test_list_tasks_contract(
        self,
        mock_session_local,
        mock_repository_cls,
        client,
        mock_session,
        prediction_task,
        backtest_task,
    ):
        """任务列表接口返回统一列表 DTO。"""

        mock_session_local.return_value = mock_session
        mock_repository = MagicMock()
        mock_repository.get_tasks_by_user.side_effect = [
            [prediction_task, backtest_task],
            [prediction_task, backtest_task],
        ]
        mock_repository_cls.return_value = mock_repository

        response = client.get("/tasks?limit=20&offset=0")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["total"] == 2
        assert payload["limit"] == 20
        assert payload["offset"] == 0
        assert len(payload["tasks"]) == 2
        assert payload["tasks"][0]["task_type"] == "prediction"
        assert payload["tasks"][1]["task_type"] == "backtest"
        assert payload["tasks"][1]["config"]["strategy_name"] == "multi_factor"

    @patch("app.api.v1.tasks.PredictionResultRepository")
    @patch("app.api.v1.tasks.TaskRepository")
    @patch("app.api.v1.tasks.SessionLocal")
    def test_get_task_detail_contract(
        self,
        mock_session_local,
        mock_repository_cls,
        mock_prediction_repository_cls,
        client,
        mock_session,
        backtest_task,
        prediction_row,
    ):
        """任务详情接口返回统一详情 DTO。"""

        mock_session_local.return_value = mock_session
        mock_repository = MagicMock()
        mock_repository.get_task_by_id.return_value = backtest_task
        mock_repository_cls.return_value = mock_repository

        mock_prediction_repository = MagicMock()
        mock_prediction_repository.get_prediction_results_by_task.return_value = [
            prediction_row
        ]
        mock_prediction_repository_cls.return_value = mock_prediction_repository

        fake_stock_loader_module = ModuleType("app.services.data.stock_data_loader")

        class FakeStockDataLoader:
            def __init__(self, data_root: str):
                self.data_root = data_root

            def load_stock_data(self, stock_code: str, end_date=None):
                return SimpleNamespace(empty=True, columns=[])

        fake_stock_loader_module.StockDataLoader = FakeStockDataLoader

        with patch.dict(
            sys.modules,
            {"app.services.data.stock_data_loader": fake_stock_loader_module},
        ):
            response = client.get("/tasks/task-bt-1")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["task_id"] == "task-bt-1"
        assert payload["task_type"] == "backtest"
        assert payload["config"]["strategy_name"] == "multi_factor"
        assert payload["results"]["total_stocks"] == 2
        assert payload["results"]["successful_predictions"] == 1
        assert payload["results"]["predictions"][0]["stock_code"] == "000001.SZ"
        assert payload["results"]["backtest_results"] == backtest_task.result
        assert payload["backtest_results"] == backtest_task.result
        assert payload["result"] == backtest_task.result

    @patch("app.api.v1.tasks.log_best_effort_failure")
    @patch("app.api.v1.tasks.PredictionResultRepository")
    @patch("app.api.v1.tasks.TaskRepository")
    @patch("app.api.v1.tasks.SessionLocal")
    def test_get_task_detail_logs_latest_price_load_failure(
        self,
        mock_session_local,
        mock_repository_cls,
        mock_prediction_repository_cls,
        mock_log_best_effort_failure,
        client,
        mock_session,
        backtest_task,
        prediction_row,
    ):
        """最新价格读取失败时应记录告警并继续返回任务详情。"""

        mock_session_local.return_value = mock_session
        mock_repository = MagicMock()
        mock_repository.get_task_by_id.return_value = backtest_task
        mock_repository_cls.return_value = mock_repository

        mock_prediction_repository = MagicMock()
        mock_prediction_repository.get_prediction_results_by_task.return_value = [
            prediction_row
        ]
        mock_prediction_repository_cls.return_value = mock_prediction_repository

        fake_stock_loader_module = ModuleType("app.services.data.stock_data_loader")

        class FakeStockDataLoader:
            def __init__(self, data_root: str):
                self.data_root = data_root

            def load_stock_data(self, stock_code: str, end_date=None):
                raise RuntimeError("data file missing")

        fake_stock_loader_module.StockDataLoader = FakeStockDataLoader

        with patch.dict(
            sys.modules,
            {"app.services.data.stock_data_loader": fake_stock_loader_module},
        ):
            response = client.get("/tasks/task-bt-1")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["results"]["predictions"][0]["stock_code"] == "000001.SZ"
        mock_log_best_effort_failure.assert_called_once()
        assert (
            mock_log_best_effort_failure.call_args.kwargs["context"]["stock_code"]
            == "000001.SZ"
        )

    @patch("app.api.v1.tasks.get_process_executor")
    @patch("app.api.v1.tasks.deep_merge")
    @patch("app.api.v1.tasks.TaskRepository")
    @patch("app.api.v1.tasks.SessionLocal")
    def test_rebuild_task_contract(
        self,
        mock_session_local,
        mock_repository_cls,
        mock_deep_merge,
        mock_executor_factory,
        client,
        mock_session,
        backtest_task,
    ):
        """任务重建接口返回统一任务 DTO。"""

        rebuilt_task = SimpleNamespace(
            task_id="task-bt-2",
            task_name="[重建] 回测任务",
            task_type="backtest",
            status="created",
            progress=0.0,
            created_at=FIXED_NOW,
            completed_at=None,
            error_message=None,
            config=backtest_task.config,
            result=None,
        )
        merged_config = {**backtest_task.config, "initial_cash": 200000}

        mock_session_local.return_value = mock_session
        mock_repository = MagicMock()
        mock_repository.get_task_by_id.return_value = backtest_task
        mock_repository.create_task.return_value = rebuilt_task
        mock_repository_cls.return_value = mock_repository
        mock_deep_merge.return_value = merged_config
        mock_executor_factory.return_value = MagicMock(submit=MagicMock())

        response = client.post(
            "/tasks/task-bt-1/rebuild",
            json={
                "task_name": "[重建] 回测任务",
                "config_override": {"initial_cash": 200000},
            },
        )

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["task_id"] == "task-bt-2"
        assert payload["task_type"] == "backtest"
        assert payload["original_task_id"] == "task-bt-1"
        assert payload["config"]["initial_cash"] == 200000

    @patch("app.api.v1.tasks.TaskRepository")
    @patch("app.api.v1.tasks.SessionLocal")
    def test_stop_task_contract(
        self,
        mock_session_local,
        mock_repository_cls,
        client,
        mock_session,
    ):
        """停止任务接口返回统一任务 DTO。"""

        running_task = SimpleNamespace(
            task_id="task-run-1",
            task_name="运行中任务",
            task_type="prediction",
            status="running",
            progress=50.0,
            created_at=FIXED_NOW,
            completed_at=None,
            error_message=None,
            config={"stock_codes": ["000001.SZ"], "model_id": "model-v1"},
            result=None,
        )
        cancelled_task = SimpleNamespace(
            task_id="task-run-1",
            task_name="运行中任务",
            task_type="prediction",
            status="cancelled",
            progress=50.0,
            created_at=FIXED_NOW,
            completed_at=None,
            error_message=None,
            config={"stock_codes": ["000001.SZ"], "model_id": "model-v1"},
            result=None,
        )

        mock_session_local.return_value = mock_session
        mock_repository = MagicMock()
        mock_repository.get_task_by_id.return_value = running_task
        mock_repository.update_task_status.return_value = cancelled_task
        mock_repository_cls.return_value = mock_repository

        response = client.post("/tasks/task-run-1/stop")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["task_id"] == "task-run-1"
        assert payload["status"] == "cancelled"
        assert payload["task_type"] == "prediction"

    @patch("app.api.v1.tasks.get_process_executor")
    @patch("app.api.v1.tasks.TaskRepository")
    @patch("app.api.v1.tasks.SessionLocal")
    def test_retry_task_contract(
        self,
        mock_session_local,
        mock_repository_cls,
        mock_executor_factory,
        client,
        mock_session,
    ):
        """重试任务接口返回统一任务 DTO。"""

        failed_task = SimpleNamespace(
            task_id="task-failed-1",
            task_name="失败任务",
            task_type="prediction",
            status="failed",
            progress=0.0,
            created_at=FIXED_NOW,
            completed_at=None,
            error_message="boom",
            config={"stock_codes": ["000001.SZ"], "model_id": "model-v1"},
            result=None,
        )
        recreated_task = SimpleNamespace(
            task_id="task-failed-1",
            task_name="失败任务",
            task_type="prediction",
            status="created",
            progress=0.0,
            created_at=FIXED_NOW,
            completed_at=None,
            error_message=None,
            config={"stock_codes": ["000001.SZ"], "model_id": "model-v1"},
            result=None,
        )

        mock_session_local.return_value = mock_session
        mock_repository = MagicMock()
        mock_repository.get_task_by_id.return_value = failed_task
        mock_repository.update_task_status.return_value = recreated_task
        mock_repository_cls.return_value = mock_repository
        mock_executor_factory.return_value = MagicMock(submit=MagicMock())

        response = client.post("/tasks/task-failed-1/retry")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["task_id"] == "task-failed-1"
        assert payload["status"] == "created"
        assert payload["task_type"] == "prediction"
        assert payload["config"]["model_id"] == "model-v1"
