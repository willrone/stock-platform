"""Strict baseline golden artifacts shared helpers.

提供 strict-baseline golden files 的导出、读取与比对公共逻辑。
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "app.db"
DEFAULT_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "strict_baseline"
DEFAULT_MANIFEST_PATH = DEFAULT_GOLDEN_DIR / "manifest.json"
GOLDEN_SCHEMA_VERSION = "strict-baseline-golden.v1"
MANIFEST_SCHEMA_VERSION = "strict-baseline-manifest.v1"
RATIO_ABS_TOLERANCE = 1e-9
MONEY_ABS_TOLERANCE = 1e-6
MONEY_FIELDS = {
    "final_value",
    "initial_cash",
    "total_commission",
    "total_slippage",
    "total_cost",
}

SOURCE_TASK_IDS = {
    "stochastic": "87dca06e-84c4-4d09-b4d8-3c8cf22e75a1",
    "cci": "5650fcfd-b429-4d83-b641-5fe0e14447ef",
    "cointegration": "1846214a-881d-464d-af0c-5864f82771a5",
    "multi_factor": "1c88160e-2b4d-47b3-a74d-cbf3dc20bc20",
    "obv": "ca493750-1107-4b6f-9e38-6f8126c1bb55",
    "low_volatility": "8ec31dac-3a89-4587-8e6e-957f5abe2f0a",
    "momentum_factor": "86e9aa33-ae67-4889-8ab7-b439f4904c8c",
    "rsi": "34ab0a39-d54b-4127-ab09-77ef03619dc1",
    "bollinger": "eebddc55-1734-4b38-a072-fa2682af994d",
    "pairs_trading": "63b92d02-130b-49e8-936a-fbb68f9e9597",
    "kdj": "e78a5ab8-5b0b-4bb6-8e30-26b31c26fd54",
    "value_factor": "77b6111d-706e-490b-99a7-a079da46c1b9",
    "mean_reversion": "dcb1b357-26b7-4d63-ab2a-cef8176ae977",
    "moving_average": "d6408834-1c25-4815-b5b8-253be25ebd1a",
    "macd": "149c1268-7893-41d2-ab5a-6134cf8c9c3e",
}

TOP_LEVEL_METRICS = [
    "final_value",
    "total_return",
    "annualized_return",
    "sharpe_ratio",
    "max_drawdown",
    "volatility",
    "win_rate",
    "profit_factor",
    "total_trades",
    "total_signals",
    "winning_trades",
    "losing_trades",
    "trading_days",
    "best_month",
    "worst_month",
    "monthly_return_mean",
    "monthly_return_std",
    "excess_return_with_cost",
    "excess_return_without_cost",
]

SIGNAL_SUMMARY_KEYS = [
    "execution_rate",
    "execution_rate_actionable",
    "raw_signal_count",
    "actionable_signal_count",
    "executed_signal_count",
    "top_rejection_reasons",
]

FINGERPRINT_KEYS = [
    "portfolio_history",
    "trade_history",
    "monthly_returns_detail",
    "performance_analysis",
    "perf_breakdown",
    "backtest_config",
    "metrics",
]


def stable_json_dumps(value: Any) -> str:
    """Return a stable JSON string for hashing and storage."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    """Hash a JSON-serializable value with stable formatting."""
    payload = stable_json_dumps(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_task_row(connection: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    """Load a completed task row by task_id."""
    row = connection.execute(
        """
        SELECT task_id, task_name, created_at, completed_at, config, result
        FROM tasks
        WHERE task_id = ?
        """,
        (task_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"任务不存在: {task_id}")
    return row


def parse_task_row(row: sqlite3.Row) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse config/result JSON from a sqlite row."""
    config = json.loads(row["config"] or "{}")
    result = json.loads(row["result"] or "{}")
    return config, result


def build_signal_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the comparable signal execution summary."""
    summary = result.get("signal_execution_summary") or {}
    return {key: summary.get(key) for key in SIGNAL_SUMMARY_KEYS}


def build_metric_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """Build the comparable metric snapshot payload."""
    top_level = {key: result.get(key) for key in TOP_LEVEL_METRICS}
    return {
        "top_level": top_level,
        "metrics": result.get("metrics") or {},
        "cost_statistics": result.get("cost_statistics") or {},
        "signal_execution_summary": build_signal_summary(result),
    }


def build_config_snapshot(
    config: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build the comparable config snapshot payload."""
    stock_codes = config.get("stock_codes") or []
    capital_config = {
        "initial_cash": config.get("initial_cash"),
        "commission_rate": config.get("commission_rate"),
        "slippage_rate": config.get("slippage_rate"),
    }
    return {
        "strategy_name": result.get("strategy_name") or config.get("strategy_name"),
        "date_range": {
            "start_date": config.get("start_date"),
            "end_date": config.get("end_date"),
        },
        "capital": capital_config,
        "stock_universe": {
            "count": len(stock_codes),
            "sha256": sha256_json(stock_codes),
        },
        "strategy_config": config.get("strategy_config") or {},
        "config_sha256": sha256_json(config),
        "backtest_config_sha256": sha256_json(result.get("backtest_config") or {}),
    }


def build_fingerprints(result: dict[str, Any]) -> dict[str, Any]:
    """Build hashes for large or noisy result sections."""
    fingerprints: dict[str, Any] = {}
    for key in FINGERPRINT_KEYS:
        value = result.get(key)
        if isinstance(value, list):
            fingerprints[f"{key}_length"] = len(value)
        fingerprints[f"{key}_sha256"] = sha256_json(value)
    return fingerprints


def build_golden_document(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a task row into one strict-baseline golden document."""
    config, result = parse_task_row(row)
    strategy_name = result.get("strategy_name") or config.get("strategy_name")
    return {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "strategy_name": strategy_name,
        "source_task": {
            "task_id": row["task_id"],
            "task_name": row["task_name"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        },
        "config_snapshot": build_config_snapshot(config, result),
        "metric_snapshot": build_metric_snapshot(result),
        "fingerprints": build_fingerprints(result),
        "tolerance_policy": build_tolerance_policy(),
    }


def build_tolerance_policy() -> dict[str, Any]:
    """Describe the tolerance rules used by the verifier."""
    return {
        "ratio_abs_tolerance": RATIO_ABS_TOLERANCE,
        "money_abs_tolerance": MONEY_ABS_TOLERANCE,
        "integer_fields": "exact",
        "hash_fields": "exact",
        "ignored_sections": ["source_task"],
    }


def build_manifest(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the manifest document for all strict baselines."""
    entries = []
    for document in documents:
        metric_top = document["metric_snapshot"]["top_level"]
        entries.append(
            {
                "strategy_name": document["strategy_name"],
                "task_id": document["source_task"]["task_id"],
                "task_name": document["source_task"]["task_name"],
                "file": f"strategies/{document['strategy_name']}.json",
                "date_range": document["config_snapshot"]["date_range"],
                "stock_count": document["config_snapshot"]["stock_universe"]["count"],
                "final_value": metric_top["final_value"],
                "total_return": metric_top["total_return"],
                "annualized_return": metric_top["annualized_return"],
                "sharpe_ratio": metric_top["sharpe_ratio"],
                "max_drawdown": metric_top["max_drawdown"],
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "strategy_count": len(documents),
        "golden_dir": "strategies",
        "tolerance_policy": build_tolerance_policy(),
        "task_id_mapping": entries,
        "strategy_names": [document["strategy_name"] for document in documents],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{content}\n", encoding="utf-8")


def flatten_payload(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict/list payloads into comparable dot paths."""
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flat.update(flatten_payload(child, child_prefix))
        return flat
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            flat.update(flatten_payload(child, child_prefix))
        return flat
    flat[prefix] = value
    return flat


def build_comparable_map(document: dict[str, Any]) -> dict[str, Any]:
    """Build a flattened map for compare-only sections."""
    comparable: dict[str, Any] = {}
    for section in (
        "strategy_name",
        "config_snapshot",
        "metric_snapshot",
        "fingerprints",
    ):
        comparable.update(flatten_payload(document[section], section))
    return comparable


def tolerance_for_field(path: str, value: Any) -> float | None:
    """Return the numeric tolerance for one flattened field path."""
    if not isinstance(value, float):
        return None
    field_name = path.rsplit(".", maxsplit=1)[-1]
    if field_name in MONEY_FIELDS:
        return MONEY_ABS_TOLERANCE
    return RATIO_ABS_TOLERANCE


def compare_documents(
    golden_document: dict[str, Any],
    candidate_document: dict[str, Any],
    strict_hashes: bool = True,
) -> list[str]:
    """Compare a candidate document against a golden document."""
    expected_map = build_comparable_map(golden_document)
    actual_map = build_comparable_map(candidate_document)
    if not strict_hashes:
        expected_map = {k: v for k, v in expected_map.items() if "sha256" not in k}
        actual_map = {k: v for k, v in actual_map.items() if "sha256" not in k}

    mismatches: list[str] = []
    missing = sorted(set(expected_map) - set(actual_map))
    extra = sorted(set(actual_map) - set(expected_map))
    mismatches.extend(f"缺少字段: {path}" for path in missing)
    mismatches.extend(f"多出字段: {path}" for path in extra)

    for path in sorted(set(expected_map) & set(actual_map)):
        expected_value = expected_map[path]
        actual_value = actual_map[path]
        tolerance = tolerance_for_field(path, expected_value)
        if tolerance is None:
            if expected_value != actual_value:
                mismatches.append(
                    f"{path}: 期望 {expected_value!r}，实际 {actual_value!r}"
                )
            continue
        if not math.isclose(expected_value, actual_value, abs_tol=tolerance):
            mismatches.append(
                f"{path}: 期望 {expected_value}，实际 {actual_value}，容忍 {tolerance}"
            )
    return mismatches


def open_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Open the sqlite database with Row factory enabled."""
    connection = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection
