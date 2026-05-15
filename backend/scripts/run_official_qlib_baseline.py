#!/usr/bin/env python3
"""Run stock-platform's official-style Qlib LightGBM baseline.

This intentionally goes through the same UnifiedQlibTrainingEngine path used by
/models/train, but runs synchronously so failures produce a direct traceback.
"""
# isort: skip_file

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.qlib import native_env  # noqa: E402

native_env.ensure_libomp_env_before_lightgbm_import()

from app.services.data import official_qlib_data_builder  # noqa: E402
from app.services.qlib.official_workflow import (  # noqa: E402
    OfficialDataset,
    OfficialMarket,
    build_official_lightgbm_workflow_config,
)
from app.services.qlib.unified_qlib_training_engine import (  # noqa: E402
    QlibModelType,
    QlibTrainingConfig,
    UnifiedQlibTrainingEngine,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["alpha158", "alpha360"], default="alpha158"
    )
    parser.add_argument("--market", choices=["csi300", "csi500"], default="csi300")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=50,
        help="Limit local discovered stocks for smoke/baseline runs. Use 0 for all.",
    )
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--early-stopping-rounds", type=int, default=10)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


async def _run() -> dict[str, Any]:
    args = _parse_args()
    dataset = OfficialDataset(args.dataset)
    market = OfficialMarket(args.market)
    workflow = build_official_lightgbm_workflow_config(dataset=dataset, market=market)

    builder = official_qlib_data_builder.OfficialQlibDataBuilder()
    max_stocks = None if args.max_stocks == 0 else args.max_stocks
    stock_codes = builder.discover_available_stock_codes(max_stocks)
    if not stock_codes:
        raise RuntimeError("No local parquet stock data found for official baseline")

    model_id = f"official-baseline-{args.dataset}-{uuid.uuid4().hex[:8]}"
    model_name = args.model_name or f"official-{args.dataset}-{args.market}-baseline"
    hyperparameters = {
        "workflow_mode": "official_replication",
        "official_dataset": workflow.dataset.value,
        "official_market": workflow.market,
        "official_benchmark": workflow.benchmark,
        "official_segments": {
            "train": list(workflow.segments.train),
            "valid": list(workflow.segments.valid),
            "test": list(workflow.segments.test),
        },
        "official_max_stocks": max_stocks,
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_iterations,
        "early_stopping_rounds": args.early_stopping_rounds,
        "open_cost": workflow.open_cost,
        "close_cost": workflow.close_cost,
        "min_cost": workflow.min_cost,
    }

    config = QlibTrainingConfig(
        model_type=QlibModelType.LIGHTGBM,
        hyperparameters=hyperparameters,
        validation_split=0.2,
        early_stopping_patience=args.early_stopping_rounds,
        use_alpha_factors=True,
        cache_features=True,
    )
    config.workflow_mode = "official_replication"
    config.official_dataset = workflow.dataset.value
    config.official_market = workflow.market
    config.official_max_stocks = max_stocks

    engine = UnifiedQlibTrainingEngine()
    result = await engine.train_model(
        model_id=model_id,
        model_name=model_name,
        stock_codes=stock_codes,
        start_date=datetime.fromisoformat(workflow.segments.train[0]),
        end_date=datetime.fromisoformat(workflow.segments.test[1]),
        config=config,
    )

    payload = {
        "model_id": model_id,
        "model_name": model_name,
        "dataset": workflow.dataset.value,
        "market": workflow.market,
        "benchmark": workflow.benchmark,
        "stock_count": len(getattr(config, "resolved_stock_codes", stock_codes)),
        "stock_codes_preview": getattr(
            config, "resolved_stock_codes", stock_codes
        )[:20],
        "model_path": result.model_path,
        "training_summary": {
            "train_samples": result.train_samples,
            "validation_samples": result.validation_samples,
            "test_samples": result.test_samples,
            "epochs": len(result.training_history),
            "best_epoch": result.best_epoch,
            "early_stopped": result.early_stopped,
        },
        "validation_metrics": result.validation_metrics,
        "signal_quality": result.signal_quality,
        "segment_evaluation": result.segment_evaluation,
    }
    payload = _json_safe(payload)

    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT
        / "backend"
        / "reports"
        / "official_qlib_baseline"
        / f"{model_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output_path), **payload}, ensure_ascii=False, indent=2
        )
    )
    return payload


if __name__ == "__main__":
    asyncio.run(_run())
