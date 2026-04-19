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
            test_samples=12,
            feature_correlation={"summary": "ok"},
            signal_quality={"rank_ic": 0.12},
            segment_evaluation={"test": {"dataset_samples": 12}},
            early_stopping_info=early_stopping,
        )

        assert isinstance(result, QlibTrainingResult)
        assert result.train_samples == 100
        assert result.validation_samples == 25
        assert result.test_samples == 12
        assert result.segment_evaluation == {"test": {"dataset_samples": 12}}
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

    @pytest.mark.asyncio
    async def test_label_normalization_can_follow_qlib_csranknorm(self, monkeypatch) -> None:
        monkeypatch.setattr(training_engine_module, "QLIB_AVAILABLE", True)
        engine = UnifiedQlibTrainingEngine()
        config = QlibTrainingConfig(
            model_type=QlibModelType.LINEAR,
            hyperparameters={},
            validation_split=0.5,
            label_normalization="cs_rank_norm",
        )

        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        index = pd.MultiIndex.from_product(
            [["000001.SZ", "000002.SZ", "000003.SZ"], dates],
            names=["instrument", "datetime"],
        )
        close_values = []
        base_prices = {"000001.SZ": 10.0, "000002.SZ": 20.0, "000003.SZ": 30.0}
        for code in ["000001.SZ", "000002.SZ", "000003.SZ"]:
            for step in range(4):
                close_values.append(base_prices[code] + step * (1 if code == "000001.SZ" else 2 if code == "000002.SZ" else 3))
        dataset = pd.DataFrame(
            {
                "$close": close_values,
                "$volume": [1000 + i for i in range(len(index))],
            },
            index=index,
        )

        train_dataset, _ = await engine._prepare_training_datasets(dataset, 0.5, config)
        train_labels = train_dataset.segments["train"]["label"]
        expected = train_labels.groupby(level="datetime", group_keys=False).rank(pct=True)
        expected = (expected - 0.5) * 3.46

        pd.testing.assert_series_equal(train_labels, expected, check_names=False)

    @pytest.mark.asyncio
    async def test_label_definition_can_use_cross_sectional_excess_return(self, monkeypatch) -> None:
        monkeypatch.setattr(training_engine_module, "QLIB_AVAILABLE", True)
        engine = UnifiedQlibTrainingEngine()
        config = QlibTrainingConfig(
            model_type=QlibModelType.LINEAR,
            hyperparameters={},
            prediction_horizon=1,
            validation_split=0.5,
            label_definition="future_excess_return_cs",
        )

        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        stock_series = {
            "000001.SZ": [10.0, 11.0, 12.0, 13.0],
            "000002.SZ": [10.0, 12.0, 14.0, 16.0],
            "000003.SZ": [10.0, 15.0, 20.0, 25.0],
        }
        rows = []
        index = []
        for code, closes in stock_series.items():
            for dt, close in zip(dates, closes):
                rows.append({"$close": close, "$volume": 1000.0})
                index.append((code, dt))
        dataset = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(index, names=["instrument", "datetime"]),
        )

        train_dataset, _ = await engine._prepare_training_datasets(dataset, 0.5, config)
        train_labels = train_dataset.segments["train"]["label"].sort_index()

        raw = dataset.copy()
        raw["label"] = raw.groupby(level="instrument")["$close"].shift(-1) / raw["$close"] - 1
        raw = raw.fillna(0.0)
        train_raw = raw[raw.index.get_level_values("datetime").isin(dates[:2])].copy()
        lower = train_raw["label"].quantile(0.01)
        upper = train_raw["label"].quantile(0.99)
        train_raw["label"] = train_raw["label"].clip(lower=lower, upper=upper)
        expected = train_raw["label"] - train_raw.groupby(level="datetime")["label"].transform("mean")
        expected = expected.sort_index()

        pd.testing.assert_series_equal(train_labels, expected, check_names=False)


class _FakeOfficialDataset:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def prepare(self, segment, col_set="__all", data_key="infer", **kwargs):
        if col_set == "label":
            return self.frame[[self.frame.columns[-1]]]
        return self.frame


class _FakeOfficialAdapter:
    def __init__(self, frame: pd.DataFrame, primary_segment: str = "valid"):
        self.dataset = _FakeOfficialDataset(frame)
        self.primary_segment = primary_segment


class TestOfficialEvaluationInputs:
    def test_extract_evaluation_inputs_reads_single_label_column_from_official_adapter(self) -> None:
        engine = UnifiedQlibTrainingEngine()
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2015-01-05"), "000001_sz"),
                (pd.Timestamp("2015-01-05"), "600036_sh"),
                (pd.Timestamp("2015-01-06"), "000001_sz"),
            ],
            names=["datetime", "instrument"],
        )
        frame = pd.DataFrame(
            {
                ("feature", "KMID"): [0.1, 0.2, 0.3],
                ("label", "Ref($close, -2) / Ref($close, -1) - 1"): [0.01, -0.02, 0.03],
            },
            index=index,
        )
        adapter = _FakeOfficialAdapter(frame)

        evaluation_inputs = engine._extract_evaluation_inputs(
            adapter,
            predictions=pd.Series([0.02, -0.01, 0.01], index=index),
            dataset_name="验证集",
        )

        assert evaluation_inputs is not None
        assert evaluation_inputs["y_true"].tolist() == [0.01, -0.02, 0.03]
        assert list(evaluation_inputs["y_index"]) == list(index)

    def test_calculate_signal_quality_uses_official_adapter_labels(self) -> None:
        engine = UnifiedQlibTrainingEngine()
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2015-01-05"), "000001_sz"),
                (pd.Timestamp("2015-01-05"), "600036_sh"),
                (pd.Timestamp("2015-01-06"), "000001_sz"),
                (pd.Timestamp("2015-01-06"), "600036_sh"),
            ],
            names=["datetime", "instrument"],
        )
        frame = pd.DataFrame(
            {
                ("feature", "KMID"): [0.1, 0.2, 0.3, 0.4],
                ("label", "Ref($close, -2) / Ref($close, -1) - 1"): [0.01, -0.02, 0.03, -0.01],
            },
            index=index,
        )
        adapter = _FakeOfficialAdapter(frame)
        predictions = pd.Series([0.02, -0.01, 0.01, -0.03], index=index)

        signal_quality = engine._calculate_signal_quality(adapter, predictions, "验证集")

        assert signal_quality["sample_count"] == 4
        assert signal_quality["rank_ic"] is not None



def test_official_signal_quality_marks_test_scope() -> None:
    engine = UnifiedQlibTrainingEngine()
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2017-01-03"), "000001_sz"),
            (pd.Timestamp("2017-01-03"), "600036_sh"),
        ],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame(
        {
            ("feature", "KMID"): [0.1, 0.2],
            ("label", "Ref($close, -2) / Ref($close, -1) - 1"): [0.01, -0.02],
        },
        index=index,
    )
    adapter = _FakeOfficialAdapter(frame, primary_segment="test")
    signal_quality = engine._calculate_signal_quality(
        adapter,
        predictions=pd.Series([0.02, -0.01], index=index),
        dataset_name="测试集",
    )
    assert signal_quality["analysis_scope"] == "test"
