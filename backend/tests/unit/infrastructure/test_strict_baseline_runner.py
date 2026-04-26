"""Integration tests for the strict-baseline regression runner."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = BACKEND_ROOT / "tests" / "scripts"
RUNNER_PATH = SCRIPTS_DIR / "run_strict_baseline_regression.py"
COMMON_PATH = SCRIPTS_DIR / "strict_baseline_common.py"


def load_module(module_path: Path, module_name: str):
    """Load one Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRICT_COMMON = load_module(COMMON_PATH, "strict_baseline_common_test")


def create_tasks_table(db_path: Path) -> None:
    """Create the minimal tasks table used by strict-baseline scripts."""
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            task_name TEXT,
            created_at TEXT,
            completed_at TEXT,
            config TEXT,
            result TEXT
        )
        """
    )
    connection.commit()
    connection.close()


def insert_task(
    db_path: Path, task_id: str, result_overrides: dict | None = None
) -> None:
    """Insert one deterministic task row for runner tests."""
    config = {
        "strategy_name": "rsi",
        "start_date": "2021-01-01",
        "end_date": "2021-01-31",
        "initial_cash": 100000.0,
        "commission_rate": 0.0003,
        "slippage_rate": 0.0001,
        "stock_codes": ["000001.SZ", "000002.SZ"],
        "strategy_config": {"period": 14},
    }
    result = {
        "strategy_name": "rsi",
        "final_value": 101234.5678,
        "total_return": 0.012345678,
        "annualized_return": 0.103456789,
        "sharpe_ratio": 1.23456789,
        "max_drawdown": -0.023456789,
        "volatility": 0.145678901,
        "win_rate": 0.61,
        "profit_factor": 1.45,
        "total_trades": 12,
        "total_signals": 18,
        "winning_trades": 7,
        "losing_trades": 5,
        "trading_days": 20,
        "best_month": 0.02,
        "worst_month": -0.01,
        "monthly_return_mean": 0.01,
        "monthly_return_std": 0.005,
        "excess_return_with_cost": 0.011,
        "excess_return_without_cost": 0.012,
        "metrics": {"alpha": 0.1},
        "cost_statistics": {
            "initial_cash": 100000.0,
            "total_commission": 12.34,
            "total_slippage": 3.21,
            "total_cost": 15.55,
        },
        "signal_execution_summary": {
            "execution_rate": 0.66,
            "execution_rate_actionable": 0.73,
            "raw_signal_count": 20,
            "actionable_signal_count": 18,
            "executed_signal_count": 12,
            "top_rejection_reasons": ["limit_up"],
        },
        "portfolio_history": [{"date": "2021-01-01", "value": 100000.0}],
        "trade_history": [{"date": "2021-01-05", "action": "BUY"}],
        "monthly_returns_detail": [{"month": "2021-01", "return": 0.01}],
        "performance_analysis": {"score": 0.9},
        "perf_breakdown": {"daily": [0.01, -0.01]},
        "backtest_config": {"engine": "baseline"},
    }
    if result_overrides:
        result.update(result_overrides)

    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        INSERT INTO tasks (task_id, task_name, created_at, completed_at, config, result)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            f"task-{task_id}",
            "2026-04-10T08:00:00Z",
            "2026-04-10T08:05:00Z",
            json.dumps(config, ensure_ascii=False),
            json.dumps(result, ensure_ascii=False),
        ),
    )
    connection.commit()
    connection.close()


def build_manifest_fixture(base_dir: Path, db_path: Path, task_id: str) -> Path:
    """Create one temporary manifest + strategy golden fixture."""
    manifest_dir = base_dir / "strict_baseline"
    strategies_dir = manifest_dir / "strategies"
    strategies_dir.mkdir(parents=True, exist_ok=True)

    with STRICT_COMMON.open_db(db_path) as connection:
        row = STRICT_COMMON.load_task_row(connection, task_id)
        document = STRICT_COMMON.build_golden_document(row)

    STRICT_COMMON.write_json(strategies_dir / "rsi.json", document)
    manifest = {
        "schema_version": "strict-baseline-manifest.v1",
        "generated_at": "2026-04-10T08:05:00Z",
        "strategy_count": 1,
        "golden_dir": "strategies",
        "tolerance_policy": STRICT_COMMON.build_tolerance_policy(),
        "task_id_mapping": [
            {
                "strategy_name": "rsi",
                "task_id": task_id,
                "task_name": f"task-{task_id}",
                "file": "strategies/rsi.json",
                "date_range": document["config_snapshot"]["date_range"],
                "stock_count": document["config_snapshot"]["stock_universe"]["count"],
                "final_value": document["metric_snapshot"]["top_level"]["final_value"],
                "total_return": document["metric_snapshot"]["top_level"][
                    "total_return"
                ],
                "annualized_return": document["metric_snapshot"]["top_level"][
                    "annualized_return"
                ],
                "sharpe_ratio": document["metric_snapshot"]["top_level"][
                    "sharpe_ratio"
                ],
                "max_drawdown": document["metric_snapshot"]["top_level"][
                    "max_drawdown"
                ],
            }
        ],
        "strategy_names": ["rsi"],
    }
    manifest_path = manifest_dir / "manifest.json"
    STRICT_COMMON.write_json(manifest_path, manifest)
    return manifest_path


def run_runner(
    db_path: Path, manifest_path: Path, output_dir: Path
) -> subprocess.CompletedProcess:
    """Execute the regression runner script."""
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--db-path",
        str(db_path),
        "--manifest-path",
        str(manifest_path),
        "--summary-json",
        str(output_dir / "summary.json"),
        "--summary-md",
        str(output_dir / "summary.md"),
        "--junit-xml",
        str(output_dir / "junit.xml"),
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_runner_outputs_summary_and_junit(tmp_path: Path) -> None:
    """Runner should emit CI-friendly artifacts when regression passes."""
    db_path = tmp_path / "app.db"
    create_tasks_table(db_path)
    insert_task(db_path, "task-pass")
    manifest_path = build_manifest_fixture(tmp_path, db_path, "task-pass")

    result = run_runner(db_path, manifest_path, tmp_path / "out-pass")

    assert result.returncode == 0
    assert "[PASS] strict baseline regression: 1/1 passed" in result.stdout
    summary = json.loads((tmp_path / "out-pass" / "summary.json").read_text())
    assert summary["failed"] == 0
    assert summary["results"][0]["status"] == "passed"
    junit_xml = (tmp_path / "out-pass" / "junit.xml").read_text()
    assert 'testsuite name="strict-baseline-regression"' in junit_xml
    assert 'testcase classname="strict_baseline" name="rsi"' in junit_xml


def test_runner_returns_non_zero_when_drift_detected(tmp_path: Path) -> None:
    """Runner should fail and explain mismatches when rerun drifts."""
    db_path = tmp_path / "app.db"
    create_tasks_table(db_path)
    insert_task(db_path, "task-fail")
    manifest_path = build_manifest_fixture(tmp_path, db_path, "task-fail")
    insert_task(db_path, "task-rerun", {"final_value": 99999.0})

    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--db-path",
        str(db_path),
        "--manifest-path",
        str(manifest_path),
        "--task-id",
        "task-rerun",
        "--strategy",
        "rsi",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 1
    assert "[FAIL] strict baseline regression: 0/1 passed" in result.stdout
    assert "metric_snapshot.top_level.final_value" in result.stdout
