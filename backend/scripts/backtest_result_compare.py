#!/usr/bin/env python
"""Compare two normalized backtest golden result files.

This is a correctness guard for performance optimization work. It intentionally
ignores runtime-only fields and focuses on trading semantics: scalar metrics,
trade ledger, equity curve, signal counters, and rejection reason distribution.

Usage:
  cd backend
  ./venv/bin/python scripts/backtest_result_compare.py baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ABS_TOL = 1e-6
REL_TOL = 1e-9

SCALAR_FIELDS = [
    "initial_cash",
    "final_value",
    "total_return",
    "annualized_return",
    "volatility",
    "sharpe_ratio",
    "max_drawdown",
    "total_trades",
    "win_rate",
    "profit_factor",
    "winning_trades",
    "losing_trades",
    "total_signals",
    "trading_days",
]

METRIC_FIELDS = [
    "sharpe_ratio",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "volatility",
    "win_rate",
    "profit_factor",
    "total_trades",
]

TRADE_FIELDS_EXACT = ["stock_code", "action", "quantity", "timestamp"]
TRADE_FIELDS_FLOAT = ["price", "commission", "slippage_cost", "pnl"]

PORTFOLIO_FIELDS_EXACT = ["date", "positions_count"]
PORTFOLIO_FIELDS_FLOAT = [
    "portfolio_value",
    "portfolio_value_without_cost",
    "cash",
    "total_return",
    "total_return_without_cost",
]

COST_FIELDS = ["total_commission", "total_slippage", "total_cost", "cost_ratio"]

SIGNAL_SUMMARY_FLOAT_FIELDS = ["execution_rate", "execution_rate_actionable"]
SIGNAL_SUMMARY_EXACT_FIELDS = [
    "raw_signal_count",
    "actionable_signal_count",
    "executed_signal_count",
]


@dataclass
class Difference:
    path: str
    expected: Any
    actual: Any
    message: str

    def format(self) -> str:
        return (
            f"{self.path}: {self.message}\n"
            f"  expected={self.expected!r}\n"
            f"  actual  ={self.actual!r}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return data


def _is_missing(value: Any) -> bool:
    return value is None


def _numbers_close(
    expected: Any, actual: Any, *, abs_tol: float, rel_tol: float
) -> bool:
    if _is_missing(expected) and _is_missing(actual):
        return True
    try:
        e = float(expected)
        a = float(actual)
    except (TypeError, ValueError):
        return expected == actual
    if math.isinf(e) or math.isinf(a):
        return math.isinf(e) and math.isinf(a) and (e > 0) == (a > 0)
    if math.isnan(e) or math.isnan(a):
        return math.isnan(e) and math.isnan(a)
    return math.isclose(e, a, abs_tol=abs_tol, rel_tol=rel_tol)


def _compare_float(
    diffs: list[Difference], path: str, expected: Any, actual: Any
) -> None:
    if not _numbers_close(expected, actual, abs_tol=ABS_TOL, rel_tol=REL_TOL):
        diffs.append(Difference(path, expected, actual, "float mismatch"))


def _compare_exact(
    diffs: list[Difference], path: str, expected: Any, actual: Any
) -> None:
    if expected != actual:
        diffs.append(Difference(path, expected, actual, "exact mismatch"))


def _compare_field_block(
    diffs: list[Difference],
    expected: dict[str, Any],
    actual: dict[str, Any],
    fields: list[str],
    prefix: str,
) -> None:
    for field in fields:
        if field in expected or field in actual:
            _compare_float(
                diffs, f"{prefix}.{field}", expected.get(field), actual.get(field)
            )


def _compare_trades(
    diffs: list[Difference], expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    e_trades = expected.get("trade_history") or []
    a_trades = actual.get("trade_history") or []
    _compare_exact(diffs, "trade_history.length", len(e_trades), len(a_trades))
    for i, (e_trade, a_trade) in enumerate(zip(e_trades, a_trades)):
        for field in TRADE_FIELDS_EXACT:
            _compare_exact(
                diffs,
                f"trade_history[{i}].{field}",
                e_trade.get(field),
                a_trade.get(field),
            )
        for field in TRADE_FIELDS_FLOAT:
            _compare_float(
                diffs,
                f"trade_history[{i}].{field}",
                e_trade.get(field),
                a_trade.get(field),
            )


def _normalize_position(pos: dict[str, Any]) -> dict[str, Any]:
    return {
        "quantity": pos.get("quantity"),
        "avg_cost": pos.get("avg_cost"),
        "current_price": pos.get("current_price"),
        "market_value": pos.get("market_value"),
        "unrealized_pnl": pos.get("unrealized_pnl"),
    }


def _compare_positions(
    diffs: list[Difference], path: str, expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    e_positions = expected or {}
    a_positions = actual or {}
    _compare_exact(diffs, f"{path}.codes", sorted(e_positions), sorted(a_positions))
    for code in sorted(set(e_positions) & set(a_positions)):
        e_pos = _normalize_position(e_positions[code])
        a_pos = _normalize_position(a_positions[code])
        _compare_exact(
            diffs, f"{path}.{code}.quantity", e_pos["quantity"], a_pos["quantity"]
        )
        for field in ["avg_cost", "current_price", "market_value", "unrealized_pnl"]:
            _compare_float(diffs, f"{path}.{code}.{field}", e_pos[field], a_pos[field])


def _compare_portfolio_history(
    diffs: list[Difference], expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    e_hist = expected.get("portfolio_history") or []
    a_hist = actual.get("portfolio_history") or []
    _compare_exact(diffs, "portfolio_history.length", len(e_hist), len(a_hist))
    for i, (e_snap, a_snap) in enumerate(zip(e_hist, a_hist)):
        for field in PORTFOLIO_FIELDS_EXACT:
            _compare_exact(
                diffs,
                f"portfolio_history[{i}].{field}",
                e_snap.get(field),
                a_snap.get(field),
            )
        for field in PORTFOLIO_FIELDS_FLOAT:
            _compare_float(
                diffs,
                f"portfolio_history[{i}].{field}",
                e_snap.get(field),
                a_snap.get(field),
            )
        _compare_positions(
            diffs,
            f"portfolio_history[{i}].positions",
            e_snap.get("positions") or {},
            a_snap.get("positions") or {},
        )


def _reason_counts(summary: dict[str, Any]) -> dict[str, int]:
    reasons = summary.get("top_rejection_reasons") or []
    out: dict[str, int] = {}
    for item in reasons:
        if isinstance(item, dict):
            reason = str(item.get("reason") or item.get("execution_reason") or "")
            count = int(item.get("count") or 0)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            reason = str(item[0])
            count = int(item[1])
        else:
            continue
        out[reason] = out.get(reason, 0) + count
    return out


def _compare_signal_summary(
    diffs: list[Difference], expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    e_summary = expected.get("signal_execution_summary") or {}
    a_summary = actual.get("signal_execution_summary") or {}
    for field in SIGNAL_SUMMARY_EXACT_FIELDS:
        if field in e_summary or field in a_summary:
            _compare_exact(
                diffs,
                f"signal_execution_summary.{field}",
                e_summary.get(field),
                a_summary.get(field),
            )
    for field in SIGNAL_SUMMARY_FLOAT_FIELDS:
        if field in e_summary or field in a_summary:
            _compare_float(
                diffs,
                f"signal_execution_summary.{field}",
                e_summary.get(field),
                a_summary.get(field),
            )
    if e_summary or a_summary:
        _compare_exact(
            diffs,
            "signal_execution_summary.rejection_reason_counts",
            _reason_counts(e_summary),
            _reason_counts(a_summary),
        )


def compare_results(
    expected: dict[str, Any], actual: dict[str, Any]
) -> list[Difference]:
    diffs: list[Difference] = []

    for field in ["strategy_name", "stock_codes", "start_date", "end_date"]:
        _compare_exact(diffs, field, expected.get(field), actual.get(field))

    _compare_field_block(diffs, expected, actual, SCALAR_FIELDS, "root")
    _compare_field_block(
        diffs,
        expected.get("metrics") or {},
        actual.get("metrics") or {},
        METRIC_FIELDS,
        "metrics",
    )
    _compare_field_block(
        diffs,
        expected.get("cost_statistics") or {},
        actual.get("cost_statistics") or {},
        COST_FIELDS,
        "cost_statistics",
    )

    _compare_trades(diffs, expected, actual)
    _compare_portfolio_history(diffs, expected, actual)
    _compare_signal_summary(diffs, expected, actual)

    return diffs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--max-diffs", type=int, default=50)
    args = parser.parse_args()

    expected = _load_json(args.expected)
    actual = _load_json(args.actual)
    diffs = compare_results(expected, actual)

    if not diffs:
        print("PASS: backtest results match golden baseline")
        return 0

    print(f"FAIL: {len(diffs)} differences found")
    for diff in diffs[: args.max_diffs]:
        print(diff.format())
    if len(diffs) > args.max_diffs:
        print(f"... truncated {len(diffs) - args.max_diffs} more differences")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
