"""训练报告与 DTO 链路回归测试。"""

from __future__ import annotations

from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

_EVALUATION_REPORT_PATH = (
    Path(__file__).resolve().parents[3]
    / "app"
    / "services"
    / "models"
    / "evaluation_report.py"
)
_EVALUATION_REPORT_SPEC = spec_from_file_location(
    "stock_platform_evaluation_report", _EVALUATION_REPORT_PATH
)
assert _EVALUATION_REPORT_SPEC and _EVALUATION_REPORT_SPEC.loader
_evaluation_report = module_from_spec(_EVALUATION_REPORT_SPEC)
_EVALUATION_REPORT_SPEC.loader.exec_module(_evaluation_report)
EvaluationReportGenerator = _evaluation_report.EvaluationReportGenerator
normalize_report_payload = _evaluation_report.normalize_report_payload
build_official_record_summary = _evaluation_report.build_official_record_summary

_MODEL_DTO_PATH = (
    Path(__file__).resolve().parents[3] / "app" / "api" / "v1" / "model_dto.py"
)
_MODEL_DTO_SPEC = spec_from_file_location("stock_platform_model_dto", _MODEL_DTO_PATH)
assert _MODEL_DTO_SPEC and _MODEL_DTO_SPEC.loader
_model_dto = module_from_spec(_MODEL_DTO_SPEC)
_MODEL_DTO_SPEC.loader.exec_module(_model_dto)
build_model_detail_dto = _model_dto.build_model_detail_dto


def test_evaluation_report_preserves_sample_breakdown_and_early_stopping() -> None:
    generator = EvaluationReportGenerator()

    report = generator.generate_report(
        model_id="model-1",
        model_name="demo-model",
        model_type="lightgbm",
        version="v1",
        training_summary={
            "duration": 12.5,
            "total_samples": 485,
            "train_samples": 388,
            "validation_samples": 97,
            "test_samples": 0,
            "epochs": 37,
            "batch_size": 32,
            "learning_rate": 0.05,
        },
        performance_metrics={"accuracy": 0.61, "rmse": 0.12, "mae": 0.08},
        feature_importance={"alpha_1": 0.4, "alpha_2": 0.2},
        training_history=[
            {
                "epoch": 1,
                "train_loss": 0.45,
                "val_loss": 0.51,
                "train_accuracy": 0.55,
                "val_accuracy": 0.53,
                "timestamp": "2026-04-12T00:00:00",
            },
            {
                "epoch": 2,
                "train_loss": None,
                "val_loss": None,
                "train_accuracy": 0.58,
                "val_accuracy": 0.56,
                "timestamp": "2026-04-12T00:01:00",
            },
        ],
        hyperparameters={"learning_rate": 0.05},
        training_data_info={
            "stock_codes": ["600036.SH"],
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2025-12-31T00:00:00",
        },
        early_stopping_info={
            "early_stopped": True,
            "stopped_epoch": 37,
            "best_epoch": 27,
            "early_stopping_reason": "Qlib/LightGBM官方早停",
        },
        signal_quality={
            "ic": 0.0123,
            "icir": 0.87,
            "rank_ic": 0.0189,
            "rank_icir": 1.02,
            "long_short_ann_return": 0.15,
            "long_short_ann_sharpe": 1.1,
            "long_avg_ann_return": 0.08,
            "long_avg_ann_sharpe": 0.72,
            "sample_count": 97,
            "analysis_scope": "validation",
        },
        segment_evaluation={
            "train": {
                "dataset_samples": 388,
                "evaluated_samples": 388,
                "performance_metrics": {"accuracy": 0.63},
                "signal_quality": {
                    "rank_ic": 0.022,
                    "sample_count": 388,
                    "analysis_scope": "train",
                },
            },
            "validation": {
                "dataset_samples": 97,
                "evaluated_samples": 97,
                "performance_metrics": {"accuracy": 0.61},
                "signal_quality": {
                    "rank_ic": 0.0189,
                    "sample_count": 97,
                    "analysis_scope": "validation",
                },
            },
            "test": {
                "dataset_samples": 44,
                "evaluated_samples": 40,
                "performance_metrics": {"accuracy": 0.59},
                "signal_quality": {
                    "rank_ic": 0.015,
                    "sample_count": 40,
                    "analysis_scope": "test",
                },
            },
        },
    )

    payload = generator.to_dict(report)

    assert payload["training_summary"]["train_samples"] == 388
    assert payload["training_summary"]["validation_samples"] == 97
    assert payload["training_summary"]["total_samples"] == 485
    assert payload["signal_quality"] == {
        "ic": 0.0123,
        "icir": 0.87,
        "rank_ic": 0.0189,
        "rank_icir": 1.02,
        "long_short_ann_return": 0.15,
        "long_short_ann_sharpe": 1.1,
        "long_avg_ann_return": 0.08,
        "long_avg_ann_sharpe": 0.72,
        "sample_count": 97,
        "analysis_scope": "validation",
    }
    assert payload["training_data_info"]["train_samples"] == 388
    assert payload["training_data_info"]["validation_samples"] == 97
    assert payload["training_data_info"]["total_samples"] == 485
    assert payload["training_data_info"]["test_samples"] == 0
    assert payload["segment_evaluation"]["train"]["dataset_samples"] == 388
    assert payload["segment_evaluation"]["validation"]["evaluated_samples"] == 97
    assert (
        payload["segment_evaluation"]["test"]["signal_quality"]["analysis_scope"]
        == "test"
    )
    assert payload["portfolio_bridge_summary"] == {
        "model_id": None,
        "task_count": 0,
        "tasks": [],
        "best_by_total_return": None,
        "best_by_sharpe": None,
        "smallest_drawdown": None,
    }
    assert payload["official_record_summary"] == {
        "signal_record": {
            "train": {
                "dataset_samples": 0,
                "evaluated_samples": 0,
                "has_signal_quality": False,
            },
            "validation": {
                "dataset_samples": 0,
                "evaluated_samples": 0,
                "has_signal_quality": False,
            },
            "test": {
                "dataset_samples": 0,
                "evaluated_samples": 0,
                "has_signal_quality": False,
            },
        },
        "sig_ana_record": {
            "train": {
                "ic": None,
                "icir": None,
                "rank_ic": None,
                "rank_icir": None,
                "long_short_ann_return": None,
                "long_short_ann_sharpe": None,
                "long_avg_ann_return": None,
                "long_avg_ann_sharpe": None,
                "sample_count": 0,
                "analysis_scope": "train",
            },
            "validation": {
                "ic": None,
                "icir": None,
                "rank_ic": None,
                "rank_icir": None,
                "long_short_ann_return": None,
                "long_short_ann_sharpe": None,
                "long_avg_ann_return": None,
                "long_avg_ann_sharpe": None,
                "sample_count": 0,
                "analysis_scope": "validation",
            },
            "test": {
                "ic": None,
                "icir": None,
                "rank_ic": None,
                "rank_icir": None,
                "long_short_ann_return": None,
                "long_short_ann_sharpe": None,
                "long_avg_ann_return": None,
                "long_avg_ann_sharpe": None,
                "sample_count": 0,
                "analysis_scope": "test",
            },
        },
        "port_ana_record": {
            "task_count": 0,
            "best_by_total_return": None,
            "best_by_sharpe": None,
            "smallest_drawdown": None,
            "tasks": [],
        },
    }
    assert payload["early_stopping_info"] == {
        "early_stopped": True,
        "stopped_epoch": 37,
        "best_epoch": 27,
        "early_stopping_reason": "Qlib/LightGBM官方早停",
    }
    assert payload["training_history"][1]["train_loss"] is None
    assert payload["training_history"][1]["val_loss"] is None


def test_build_model_detail_dto_extracts_early_stopping_info_from_evaluation_report() -> (
    None
):
    model = SimpleNamespace(
        model_id="model-1",
        model_name="demo-model",
        model_type="lightgbm",
        version="v1",
        performance_metrics={"accuracy": 0.61},
        hyperparameters={"learning_rate": 0.05},
        evaluation_report={
            "training_data_info": {"stock_codes": ["600036.SH"]},
            "early_stopping_info": {
                "early_stopped": True,
                "stopped_epoch": 37,
                "best_epoch": 27,
                "early_stopping_reason": "Qlib/LightGBM官方早停",
            },
        },
        training_data_start=datetime(2024, 1, 1),
        training_data_end=datetime(2025, 12, 31),
        created_at=datetime(2026, 4, 12, 12, 0, 0),
        status="ready",
    )

    dto = build_model_detail_dto(model)

    assert dto["training_info"]["stock_codes"] == ["600036.SH"]
    assert dto["training_info"]["early_stopping_info"] == {
        "early_stopped": True,
        "stopped_epoch": 37,
        "best_epoch": 27,
        "early_stopping_reason": "Qlib/LightGBM官方早停",
    }


def test_normalize_report_payload_backfills_legacy_fields() -> None:
    payload = normalize_report_payload(
        {
            "model_id": "legacy-model",
            "training_summary": {
                "total_samples": 200,
                "train_samples": 160,
                "validation_samples": 40,
                "test_samples": 0,
            },
            "training_data_info": {
                "stock_codes": ["000001.SZ"],
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-12-31T00:00:00",
            },
        }
    )

    assert payload["training_data_info"] == {
        "stock_codes": ["000001.SZ"],
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-12-31T00:00:00",
        "total_samples": 200,
        "train_samples": 160,
        "validation_samples": 40,
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
    assert payload["segment_evaluation"]["train"]["dataset_samples"] == 160
    assert payload["segment_evaluation"]["validation"]["dataset_samples"] == 40
    assert (
        payload["segment_evaluation"]["validation"]["signal_quality"]["analysis_scope"]
        is None
    )
    assert payload["segment_evaluation"]["test"]["dataset_samples"] == 0
    assert payload["portfolio_bridge_summary"] == {
        "model_id": None,
        "task_count": 0,
        "tasks": [],
        "best_by_total_return": None,
        "best_by_sharpe": None,
        "smallest_drawdown": None,
    }
    assert payload["official_record_summary"] == {
        "signal_record": {
            "train": {
                "dataset_samples": 0,
                "evaluated_samples": 0,
                "has_signal_quality": False,
            },
            "validation": {
                "dataset_samples": 0,
                "evaluated_samples": 0,
                "has_signal_quality": False,
            },
            "test": {
                "dataset_samples": 0,
                "evaluated_samples": 0,
                "has_signal_quality": False,
            },
        },
        "sig_ana_record": {
            "train": {
                "ic": None,
                "icir": None,
                "rank_ic": None,
                "rank_icir": None,
                "long_short_ann_return": None,
                "long_short_ann_sharpe": None,
                "long_avg_ann_return": None,
                "long_avg_ann_sharpe": None,
                "sample_count": 0,
                "analysis_scope": "train",
            },
            "validation": {
                "ic": None,
                "icir": None,
                "rank_ic": None,
                "rank_icir": None,
                "long_short_ann_return": None,
                "long_short_ann_sharpe": None,
                "long_avg_ann_return": None,
                "long_avg_ann_sharpe": None,
                "sample_count": 0,
                "analysis_scope": "validation",
            },
            "test": {
                "ic": None,
                "icir": None,
                "rank_ic": None,
                "rank_icir": None,
                "long_short_ann_return": None,
                "long_short_ann_sharpe": None,
                "long_avg_ann_return": None,
                "long_avg_ann_sharpe": None,
                "sample_count": 0,
                "analysis_scope": "test",
            },
        },
        "port_ana_record": {
            "task_count": 0,
            "best_by_total_return": None,
            "best_by_sharpe": None,
            "smallest_drawdown": None,
            "tasks": [],
        },
    }


def test_build_official_record_summary_reuses_segment_and_portfolio_bridge_data() -> (
    None
):
    summary = build_official_record_summary(
        {
            "training_summary": {
                "train_samples": 388,
                "validation_samples": 97,
                "test_samples": 44,
            },
            "signal_quality": {
                "rank_ic": 0.0189,
                "sample_count": 97,
                "analysis_scope": "validation",
            },
            "segment_evaluation": {
                "train": {
                    "dataset_samples": 388,
                    "evaluated_samples": 388,
                    "signal_quality": {
                        "rank_ic": 0.022,
                        "sample_count": 388,
                        "analysis_scope": "train",
                    },
                },
                "validation": {
                    "dataset_samples": 97,
                    "evaluated_samples": 97,
                    "signal_quality": {
                        "rank_ic": 0.0189,
                        "sample_count": 97,
                        "analysis_scope": "validation",
                    },
                },
                "test": {
                    "dataset_samples": 44,
                    "evaluated_samples": 40,
                    "signal_quality": {
                        "rank_ic": 0.015,
                        "sample_count": 40,
                        "analysis_scope": "test",
                    },
                },
            },
            "portfolio_bridge_summary": {
                "task_count": 2,
                "best_by_total_return": {"task_id": "task-a", "total_return": 0.11},
                "best_by_sharpe": {"task_id": "task-b", "sharpe_ratio": 1.8},
                "smallest_drawdown": {"task_id": "task-c", "max_drawdown": -0.03},
                "tasks": [{"task_id": "task-a"}],
            },
        }
    )

    assert summary["signal_record"]["train"] == {
        "dataset_samples": 388,
        "evaluated_samples": 388,
        "has_signal_quality": True,
        "analysis_scope": "train",
    }
    assert summary["sig_ana_record"]["validation"]["rank_ic"] == 0.0189
    assert summary["sig_ana_record"]["test"]["sample_count"] == 40
    assert summary["port_ana_record"] == {
        "task_count": 2,
        "best_by_total_return": {"task_id": "task-a", "total_return": 0.11},
        "best_by_sharpe": {"task_id": "task-b", "sharpe_ratio": 1.8},
        "smallest_drawdown": {"task_id": "task-c", "max_drawdown": -0.03},
        "tasks": [{"task_id": "task-a"}],
    }
