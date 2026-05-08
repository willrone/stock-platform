"""模型/训练接口 contract tests。"""

import sys
from concurrent.futures import Future
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
    from app.api.v1.models import router as models_router
    from app.api.v1.training_progress import router as training_router


FIXED_NOW = datetime(2026, 4, 8, 12, 0, 0)


@pytest.fixture
def app() -> FastAPI:
    """创建仅包含模型/训练路由的测试应用。"""
    app = FastAPI()
    app.include_router(models_router)
    app.include_router(training_router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """创建测试客户端。"""
    return TestClient(app)


@pytest.fixture
def model_record() -> SimpleNamespace:
    """构造模型记录样本。"""
    return SimpleNamespace(
        model_id="model-1",
        model_name="统一契约模型",
        model_type="lightgbm",
        version="1.0.0",
        status="training",
        training_progress=42.5,
        training_stage="training",
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        training_data_start=FIXED_NOW,
        training_data_end=FIXED_NOW,
        hyperparameters={"learning_rate": 0.1},
        performance_metrics={"accuracy": {"value": 0.88}, "mse": 0.12},
        evaluation_report={
            "training_data_info": {"stock_codes": ["000001.SZ", "000002.SZ"]},
            "performance_metrics": {"accuracy": 0.88},
        },
    )


class TestModelAndTrainingContractAPI:
    """模型/训练 contract tests。"""

    @patch("app.api.v1.models.SessionLocal")
    def test_list_models_contract(self, mock_session_local, client, model_record):
        """/models 列表返回稳定 DTO 字段。"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_session.query.return_value.order_by.return_value.all.return_value = [
            model_record
        ]

        response = client.get("/models")

        assert response.status_code == 200
        payload = response.json()["data"]["models"]
        assert len(payload) == 1
        first = payload[0]
        assert first["model_id"] == "model-1"
        assert first["model_name"] == "统一契约模型"
        assert first["status"] == "training"
        assert first["training_progress"] == 42.5
        assert first["training_stage"] == "training"
        assert first["accuracy"] == 0.88

    @patch("app.api.v1.models.ModelInfoRepository")
    @patch("app.api.v1.models.SessionLocal")
    def test_model_detail_contract(
        self,
        mock_session_local,
        mock_repository_cls,
        client,
        model_record,
    ):
        """/models/{id} 详情返回稳定字段和 training_info。"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_repository = MagicMock()
        mock_repository.get_model_info.return_value = model_record
        mock_repository_cls.return_value = mock_repository

        response = client.get("/models/model-1")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["model_id"] == "model-1"
        assert payload["accuracy"] == 0.88
        assert payload["training_info"]["stock_codes"] == ["000001.SZ", "000002.SZ"]
        assert payload["training_info"]["hyperparameters"]["learning_rate"] == 0.1
        assert payload["status"] == "training"

    @patch("app.api.v1.models.SessionLocal")
    def test_evaluation_report_contract_supports_json_string(
        self,
        mock_session_local,
        client,
        model_record,
    ):
        """/models/{id}/evaluation-report 兼容字符串 JSON 报告。"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        model_with_string_report = SimpleNamespace(**model_record.__dict__)
        model_with_string_report.evaluation_report = (
            "{"
            '"performance_metrics":{"accuracy":0.9},'
            '"training_data_info":{"stock_codes":["000001.SZ"]}'
            "}"
        )

        mock_session.query.return_value.filter.return_value.first.return_value = (
            model_with_string_report
        )

        response = client.get("/models/model-1/evaluation-report")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["performance_metrics"]["accuracy"] == 0.9
        assert payload["training_data_info"]["stock_codes"] == ["000001.SZ"]

    @patch("app.api.v1.models.SessionLocal")
    def test_evaluation_report_contract_backfills_legacy_fields(
        self,
        mock_session_local,
        client,
        model_record,
    ):
        """/models/{id}/evaluation-report 会为旧报告补齐 early stopping 和样本统计。"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        legacy_model = SimpleNamespace(**model_record.__dict__)
        legacy_model.evaluation_report = {
            "performance_metrics": {"accuracy": 0.9},
            "training_summary": {
                "total_samples": 386,
                "train_samples": 193,
                "validation_samples": 193,
                "test_samples": 0,
            },
            "training_data_info": {
                "stock_codes": ["000001.SZ"],
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-12-31T00:00:00",
            },
        }

        mock_session.query.return_value.filter.return_value.first.return_value = (
            legacy_model
        )

        response = client.get("/models/model-1/evaluation-report")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["training_data_info"] == {
            "stock_codes": ["000001.SZ"],
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-12-31T00:00:00",
            "total_samples": 386,
            "train_samples": 193,
            "validation_samples": 193,
            "test_samples": 0,
        }
        assert payload["early_stopping_info"] == {
            "early_stopped": False,
            "stopped_epoch": 0,
            "best_epoch": 0,
            "early_stopping_reason": None,
        }
        assert payload["signal_quality"] == {
            "ic": None,
            "icir": None,
            "rank_ic": None,
            "rank_icir": None,
            "long_short_ann_return": None,
            "long_short_ann_sharpe": None,
            "long_avg_ann_return": None,
            "long_avg_ann_sharpe": None,
            "sample_count": 0,
            "analysis_scope": None,
        }
        assert payload["segment_evaluation"]["train"]["dataset_samples"] == 193
        assert payload["segment_evaluation"]["validation"]["dataset_samples"] == 193
        assert payload["segment_evaluation"]["test"]["dataset_samples"] == 0

    @patch("app.api.v1.training_progress.SessionLocal")
    @patch("app.api.v1.training_progress._get_task_manager")
    def test_training_progress_contract_fallback_to_model(
        self,
        mock_get_task_manager,
        mock_session_local,
        client,
        model_record,
    ):
        """旧 /training/tasks/{id}/progress 在 task_manager 不可用时可回退到 model。"""
        mock_get_task_manager.return_value = None
        model_record.status = "ready"

        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = (
            model_record
        )

        response = client.get("/training/tasks/model-1/progress")

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["task_id"] == "model-1"
        assert payload["status"] == "completed"
        assert payload["progress_percentage"] == 42.5
        assert payload["stage"] == "training"

    @patch("app.api.v1.models.get_train_executor")
    @patch("app.api.v1.models.SessionLocal")
    def test_train_contract_propagates_official_workflow_preset(
        self,
        mock_session_local,
        mock_get_train_executor,
        client,
    ):
        """/models/train 应把 official replication 预设透传到后台训练配置。"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        captured = {}

        class DummyExecutor:
            def submit(self, fn, **kwargs):
                captured["fn"] = fn
                captured["kwargs"] = kwargs
                future = Future()
                future.set_result(None)
                return future

        mock_get_train_executor.return_value = DummyExecutor()

        response = client.post(
            "/models/train",
            json={
                "model_name": "official-alpha158-smoke",
                "model_type": "lightgbm",
                "stock_codes": ["600036.SH", "601288.SH", "601398.SH"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "hyperparameters": {"learning_rate": 0.1},
                "workflow_mode": "official_replication",
                "official_dataset": "alpha158",
                "official_market": "csi300",
            },
        )

        assert response.status_code == 200
        submit_kwargs = captured["kwargs"]
        assert submit_kwargs["workflow_mode"] == "official_replication"
        assert submit_kwargs["official_dataset"] == "alpha158"
        assert submit_kwargs["official_market"] == "csi300"
        added_model = mock_session.add.call_args.args[0]
        assert added_model.hyperparameters["workflow_mode"] == "official_replication"
        assert added_model.hyperparameters["official_dataset"] == "alpha158"
        assert added_model.hyperparameters["official_market"] == "csi300"

    @patch("app.api.v1.models.get_train_executor")
    @patch("app.api.v1.models.SessionLocal")
    def test_train_contract_persists_alpha360_official_market_defaults(
        self,
        mock_session_local,
        mock_get_train_executor,
        client,
    ):
        """alpha360 official preset 应写入 benchmark 与 segments，防止只锁住 alpha158。"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        captured = {}

        class DummyExecutor:
            def submit(self, fn, **kwargs):
                captured["fn"] = fn
                captured["kwargs"] = kwargs
                future = Future()
                future.set_result(None)
                return future

        mock_get_train_executor.return_value = DummyExecutor()

        response = client.post(
            "/models/train",
            json={
                "model_name": "official-alpha360-smoke",
                "model_type": "lightgbm",
                "stock_codes": ["600036.SH", "601288.SH", "601398.SH"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "hyperparameters": {"learning_rate": 0.05},
                "workflow_mode": "official_replication",
                "official_dataset": "alpha360",
                "official_market": "csi500",
            },
        )

        assert response.status_code == 200
        submit_kwargs = captured["kwargs"]
        assert submit_kwargs["workflow_mode"] == "official_replication"
        assert submit_kwargs["official_dataset"] == "alpha360"
        assert submit_kwargs["official_market"] == "csi500"

        added_model = mock_session.add.call_args.args[0]
        assert added_model.hyperparameters["workflow_mode"] == "official_replication"
        assert added_model.hyperparameters["official_dataset"] == "alpha360"
        assert added_model.hyperparameters["official_market"] == "csi500"
        assert added_model.hyperparameters["official_benchmark"] == "SH000905"
        assert added_model.hyperparameters["official_segments"] == {
            "train": ["2008-01-01", "2014-12-31"],
            "valid": ["2015-01-01", "2016-12-31"],
            "test": ["2017-01-01", "2020-08-01"],
        }

    @patch("app.api.v1.models.notify_model_training_progress", new_callable=MagicMock)
    @patch("app.api.v1.models.SessionLocal")
    def test_cancel_training_contract_marks_model_cancelled(
        self,
        mock_session_local,
        mock_notify_progress,
        client,
        model_record,
    ):
        """/models/{id}/cancel-training 应标记取消请求并回写模型状态。"""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        model_record.status = "training"
        model_record.training_stage = "training"

        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = model_record
        mock_session.query.return_value = mock_query

        pending_future = Future()

        from app.api.v1 import models as models_module

        models_module._active_training_jobs[model_record.model_id] = {
            "future": pending_future,
            "cancel_event": models_module.threading.Event(),
        }

        try:
            response = client.post(f"/models/{model_record.model_id}/cancel-training")
        finally:
            models_module._active_training_jobs.clear()

        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["model_id"] == model_record.model_id
        assert payload["status"] == "cancelled"
        assert payload["cancel_requested"] is True
        assert model_record.status == "cancelled"
        assert model_record.training_stage == "cancelled"
        assert model_record.performance_metrics["status"] == "cancelled"
        mock_session.commit.assert_called()
        assert pending_future.cancelled() is True
