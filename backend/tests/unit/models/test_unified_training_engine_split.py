"""unified_qlib_training_engine 第一刀拆分回归测试。"""

from datetime import datetime

import pandas as pd
import pytest

from app.services.qlib.training_engine.pipeline import QlibTrainingPipeline
from app.services.qlib.training_engine.result_assembler import QlibTrainingResultAssembler
from app.services.qlib import unified_qlib_training_engine as training_engine_module
from app.services.qlib.unified_qlib_training_engine import (
    QlibModelType,
    QlibTrainingConfig,
    QlibTrainingResult,
    UnifiedQlibTrainingEngine,
)


class TestUnifiedEngineOrchestratorDelegation:
    """验证 UnifiedQlibTrainingEngine 已退化为协调入口。"""

    @pytest.mark.asyncio
    async def test_train_model_delegates_to_orchestrator(self, monkeypatch) -> None:
        """train_model 应将请求交给 orchestrator 执行。"""
        engine = UnifiedQlibTrainingEngine()
        config = QlibTrainingConfig(
            model_type=QlibModelType.LINEAR,
            hyperparameters={},
        )
        captured = {}

        async def fake_execute(request):
            captured["request"] = request
            return "delegated-result"

        monkeypatch.setattr(engine.training_orchestrator, "execute", fake_execute)

        result = await engine.train_model(
            model_id="model-1",
            model_name="delegation-check",
            stock_codes=["000001.SZ"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            config=config,
        )

        assert result == "delegated-result"
        assert captured["request"].model_name == "delegation-check"
        assert captured["request"].stock_codes == ["000001.SZ"]


class TestPipelineAvailabilityGuard:
    """验证 pipeline 承接 Qlib 可用性校验职责。"""

    def test_ensure_qlib_available_raises_install_guidance(self) -> None:
        """Qlib 不可用时应返回可操作的安装提示。"""
        with pytest.raises(RuntimeError) as exc_info:
            QlibTrainingPipeline.ensure_qlib_available(False)

        assert "Qlib库未安装" in str(exc_info.value)
        assert "pip install git+https://github.com/microsoft/qlib.git" in str(
            exc_info.value
        )


class TestResultAssembler:
    """验证 result assembly 逻辑已抽离并保持兼容。"""

    def test_legacy_output_is_normalized_and_accuracy_is_filled(self) -> None:
        """两段式旧训练输出应补齐 early stopping 并回填准确率。"""
        assembler = QlibTrainingResultAssembler(QlibTrainingResult)
        model, history, early_stopping = assembler.normalize_training_output(
            ("mock-model", [{"epoch": 1}],)
        )

        assert model == "mock-model"
        assert early_stopping["early_stopped"] is False
        assembler.fill_accuracy_into_history(history, 0.81234, 0.73456)
        assert history[0]["train_accuracy"] == 0.8123
        assert history[0]["val_accuracy"] == 0.7346

        result = assembler.assemble(
            model_path="/tmp/model.pkl",
            model_config={"model": "linear"},
            training_metrics={"accuracy": 0.8123},
            validation_metrics={"accuracy": 0.7346},
            feature_importance={"f1": 0.5},
            training_history=history,
            training_duration=12.3,
            train_samples=100,
            validation_samples=25,
            feature_correlation={"summary": "ok"},
            early_stopping_info=early_stopping,
        )

        assert isinstance(result, QlibTrainingResult)
        assert result.train_samples == 100
        assert result.validation_samples == 25
        assert result.early_stopped is False


class TestPrepareTrainingDatasetsAdapter:
    """验证训练/验证数据适配器不会把验证集伪装成训练集。"""

    @pytest.mark.asyncio
    async def test_validation_adapter_exposes_valid_segment_lengths(self, monkeypatch) -> None:
        monkeypatch.setattr(training_engine_module, "QLIB_AVAILABLE", True)
        engine = UnifiedQlibTrainingEngine()
        config = QlibTrainingConfig(
            model_type=QlibModelType.LINEAR,
            hyperparameters={},
            validation_split=0.2,
        )

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        index = pd.MultiIndex.from_product(
            [["000001.SZ"], dates], names=["instrument", "datetime"]
        )
        dataset = pd.DataFrame(
            {
                "$close": [10 + i for i in range(10)],
                "$volume": [1000 + 10 * i for i in range(10)],
            },
            index=index,
        )

        train_dataset, val_dataset = await engine._prepare_training_datasets(
            dataset,
            0.2,
            config,
        )

        assert train_dataset is not val_dataset
        assert len(train_dataset) == 8
        assert len(val_dataset) == 2
        assert train_dataset.data.shape[0] == 8
        assert val_dataset.data.shape[0] == 2
        assert train_dataset.segments["train"].shape[0] == 8
        assert train_dataset.segments["valid"].shape[0] == 2
        assert val_dataset.segments["train"].shape[0] == 8
        assert val_dataset.segments["valid"].shape[0] == 2
