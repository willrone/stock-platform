from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.qlib.official_workflow import (
    OfficialDataset,
    OfficialMarket,
    build_official_dataset_config,
    build_official_lightgbm_workflow_config,
    create_official_dataset_adapter,
)
from app.services.qlib.training_engine.pipeline import QlibTrainingPipeline, TrainingRequest


class _FakeOfficialDataset:
    def __init__(self):
        self.calls = []

    def prepare(self, segment, col_set="__all", data_key="infer", **kwargs):
        self.calls.append((segment, col_set, data_key))
        rows = {
            "train": [1] * 10,
            "valid": [1] * 4,
            "test": [1] * 6,
        }[segment]
        return rows


def test_build_official_dataset_config_uses_dataseth_and_handler_defaults() -> None:
    workflow = build_official_lightgbm_workflow_config(
        dataset=OfficialDataset.ALPHA158,
        market=OfficialMarket.CSI300,
    )

    config = build_official_dataset_config(workflow)

    assert config["class"] == "DatasetH"
    assert config["module_path"] == "qlib.data.dataset"
    assert config["kwargs"]["handler"]["class"] == "Alpha158"
    assert config["kwargs"]["handler"]["module_path"] == "qlib.contrib.data.handler"
    assert config["kwargs"]["handler"]["kwargs"]["instruments"] == "csi300"
    assert config["kwargs"]["segments"]["train"] == ("2008-01-01", "2014-12-31")
    assert config["kwargs"]["segments"]["valid"] == ("2015-01-01", "2016-12-31")
    assert config["kwargs"]["segments"]["test"] == ("2017-01-01", "2020-08-01")


def test_build_official_dataset_config_can_override_instruments() -> None:
    workflow = build_official_lightgbm_workflow_config(
        dataset=OfficialDataset.ALPHA158,
        market=OfficialMarket.CSI300,
    )

    config = build_official_dataset_config(workflow, instruments_override=["600036.SH", "601288.SH"])

    assert config["kwargs"]["handler"]["kwargs"]["instruments"] == ["600036_sh", "601288_sh"]


def test_build_official_dataset_config_normalizes_mixed_instrument_codes() -> None:
    workflow = build_official_lightgbm_workflow_config(
        dataset=OfficialDataset.ALPHA158,
        market=OfficialMarket.CSI300,
    )

    config = build_official_dataset_config(
        workflow,
        instruments_override=[" 600036.SH ", "000001_sz", "600519.SH"],
    )

    assert config["kwargs"]["handler"]["kwargs"]["instruments"] == [
        "600036_sh",
        "000001_sz",
        "600519_sh",
    ]


def test_create_official_dataset_adapter_counts_segment_lengths() -> None:
    workflow = build_official_lightgbm_workflow_config(
        dataset=OfficialDataset.ALPHA360,
        market=OfficialMarket.CSI300,
    )
    fake_dataset = _FakeOfficialDataset()

    adapter = create_official_dataset_adapter(workflow, dataset_factory=lambda _cfg: fake_dataset)

    assert adapter.shape == (20, 360)
    assert adapter.empty is False
    assert adapter.segment_lengths == {"train": 10, "valid": 4, "test": 6}
    assert len(adapter) == 10

    valid_adapter = adapter.for_segment("valid")
    assert len(valid_adapter) == 4
    assert valid_adapter.primary_segment == "valid"
    assert valid_adapter.dataset is fake_dataset


@pytest.mark.asyncio
async def test_pipeline_prepare_dataset_uses_official_replication_path(monkeypatch) -> None:
    engine = SimpleNamespace(data_provider=SimpleNamespace(prepare_qlib_dataset=AsyncMock()))
    pipeline = QlibTrainingPipeline(engine)
    request = TrainingRequest(
        model_id="model-1",
        model_name="official-smoke",
        stock_codes=["600036.SH"],
        start_date=None,
        end_date=None,
        config=SimpleNamespace(
            workflow_mode="official_replication",
            official_dataset="alpha158",
            official_market="csi300",
            use_alpha_factors=True,
            cache_features=True,
        ),
    )

    sentinel = MagicMock(name="official-dataset-adapter")
    built = {}

    async def _unexpected(*args, **kwargs):
        raise AssertionError("local enhanced data path should not be used")

    engine.data_provider.prepare_qlib_dataset = _unexpected

    from app.services.qlib.training_engine import pipeline as pipeline_module

    class DummyBuilder:
        def __init__(self):
            self.official_qlib_data_path = "/tmp/qlib-official"

        def prepare_stocks(self, stock_codes):
            built["stock_codes"] = stock_codes
            return {"success": stock_codes, "failed": []}

    monkeypatch.setattr(
        pipeline_module,
        "OfficialQlibDataBuilder",
        DummyBuilder,
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_official_dataset_adapter",
        lambda workflow, stock_codes=None, provider_uri=None: built.update({
            "provider_uri": provider_uri,
            "adapter_stock_codes": stock_codes,
        }) or sentinel,
    )

    dataset = await pipeline.prepare_dataset(request)

    assert dataset is sentinel
    assert built["stock_codes"] == ["600036.SH"]
    assert built["adapter_stock_codes"] == ["600036.SH"]
    assert built["provider_uri"] == "/tmp/qlib-official"


@pytest.mark.asyncio
async def test_pipeline_prepare_training_datasets_returns_official_segments() -> None:
    engine = SimpleNamespace()
    pipeline = QlibTrainingPipeline(engine)
    workflow = build_official_lightgbm_workflow_config(
        dataset=OfficialDataset.ALPHA158,
        market=OfficialMarket.CSI300,
    )
    adapter = create_official_dataset_adapter(
        workflow,
        dataset_factory=lambda _cfg: _FakeOfficialDataset(),
    )

    train_dataset, val_dataset = await pipeline.prepare_training_datasets(
        adapter,
        validation_split=0.2,
        config=SimpleNamespace(workflow_mode="official_replication"),
    )
    test_dataset = pipeline.prepare_test_dataset(adapter)

    assert train_dataset.primary_segment == "train"
    assert val_dataset.primary_segment == "valid"
    assert test_dataset.primary_segment == "test"
    assert len(train_dataset) == 10
    assert len(val_dataset) == 4
    assert len(test_dataset) == 6
