from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.models import router as models_router

FIXED_NOW = datetime(2026, 4, 14, 12, 0, 0)


def _build_app() -> TestClient:
    app = FastAPI()
    app.include_router(models_router)
    return TestClient(app)


def test_evaluation_report_includes_portfolio_bridge_summary() -> None:
    client = _build_app()
    model_record = SimpleNamespace(
        model_id="model-bridge",
        model_name="bridge-model",
        model_type="lightgbm",
        version="1.0.0",
        status="ready",
        created_at=FIXED_NOW,
        evaluation_report={
            "training_summary": {
                "total_samples": 100,
                "train_samples": 60,
                "validation_samples": 20,
                "test_samples": 20,
            },
            "training_data_info": {
                "stock_codes": ["600036.SH"],
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-12-31T00:00:00",
            },
            "performance_metrics": {"accuracy": 0.61},
        },
    )

    bridge_summary = {
        "model_id": "model-bridge",
        "task_count": 1,
        "tasks": [
            {
                "task_id": "task-1",
                "task_name": "official-2024-full",
                "window_label": "2024-full",
                "portfolio_metrics": {"total_return": 0.01},
                "signal_summary": {"raw_signal_count": 10},
                "cost_metrics": {
                    "total_cost": 123.4,
                    "final_value_without_cost": 10123.4,
                    "gross_minus_net_value_gap": 123.4,
                    "total_return_without_cost": 0.022,
                    "gross_minus_net_return_gap": 0.012,
                },
                "stock_contribution_summary": {
                    "best_stock": {"stock_code": "600036.SH", "total_pnl": 345.6},
                    "worst_stock": {"stock_code": "601288.SH", "total_pnl": -120.0},
                    "top_contributors": [
                        {"stock_code": "600036.SH", "total_pnl": 345.6}
                    ],
                    "bottom_contributors": [
                        {"stock_code": "601288.SH", "total_pnl": -120.0}
                    ],
                },
            }
        ],
        "best_by_total_return": {"task_id": "task-1", "total_return": 0.01},
        "best_by_sharpe": None,
        "smallest_drawdown": None,
        "cost_vs_gross_gap_rollup": {
            "task_count": 1,
            "tasks": [
                {
                    "task_id": "task-1",
                    "window_label": "2024-full",
                    "total_cost": 123.4,
                    "gross_minus_net_value_gap": 123.4,
                    "total_return": 0.01,
                    "total_return_without_cost": 0.022,
                }
            ],
            "largest_cost_gap": {
                "task_id": "task-1",
                "gross_minus_net_value_gap": 123.4,
            },
            "best_gross_return": {
                "task_id": "task-1",
                "total_return_without_cost": 0.022,
            },
            "best_net_return": {"task_id": "task-1", "total_return": 0.01},
        },
        "per_stock_contribution_rollup": {
            "stocks": [
                {
                    "stock_code": "600036.SH",
                    "task_mentions": 1,
                    "positive_task_count": 1,
                    "negative_task_count": 0,
                    "total_pnl": 345.6,
                    "signal_count": 7,
                }
            ],
            "best_overall": {"stock_code": "600036.SH", "total_pnl": 345.6},
            "worst_overall": {"stock_code": "601288.SH", "total_pnl": -120.0},
        },
    }

    official_record_summary = {
        "signal_record": {
            "train": {
                "dataset_samples": 60,
                "evaluated_samples": 60,
                "has_signal_quality": True,
                "analysis_scope": "train",
            },
            "validation": {
                "dataset_samples": 20,
                "evaluated_samples": 20,
                "has_signal_quality": True,
                "analysis_scope": "validation",
            },
            "test": {
                "dataset_samples": 20,
                "evaluated_samples": 18,
                "has_signal_quality": True,
                "analysis_scope": "test",
            },
        },
        "sig_ana_record": {
            "train": {"rank_ic": 0.11},
            "validation": {"rank_ic": 0.09},
            "test": {"rank_ic": 0.05},
        },
        "port_ana_record": {
            "task_count": 1,
            "tasks": [{"task_id": "task-1"}],
            "best_by_total_return": None,
            "best_by_sharpe": None,
            "smallest_drawdown": None,
        },
    }

    with (
        patch("app.api.v1.models.SessionLocal") as mock_session_local,
        patch(
            "app.api.v1.models.build_portfolio_bridge_summary",
            return_value=bridge_summary,
        ),
        patch(
            "app.api.v1.models.build_official_record_summary",
            return_value=official_record_summary,
        ),
    ):
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = (
            model_record
        )

        response = client.get("/models/model-bridge/evaluation-report")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["portfolio_bridge_summary"] == bridge_summary
    assert payload["official_record_summary"] == official_record_summary
    assert (
        payload["cost_vs_gross_gap_summary"]
        == bridge_summary["cost_vs_gross_gap_rollup"]
    )
    assert (
        payload["per_stock_ranking_preference"]
        == bridge_summary["per_stock_contribution_rollup"]
    )
    assert payload["ranking_overlap_summary"]["available"] is False
    assert payload["event_replay_summary"]["available"] is False
    assert payload["training_summary"]["test_samples"] == 20
