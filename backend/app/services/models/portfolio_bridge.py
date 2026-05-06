"""Bridge model-level signal quality with formal-task portfolio outcomes."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Optional

from loguru import logger
from sqlalchemy import text


def _safe_json_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_model_id_from_task(task_config: Dict[str, Any]) -> Optional[str]:
    if not isinstance(task_config, dict):
        return None
    candidates = [
        task_config.get("model_id"),
        (task_config.get("backtest_config") or {}).get("model_id"),
        ((task_config.get("backtest_config") or {}).get("strategy_config") or {}).get(
            "model_id"
        ),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _extract_backtest_result(task_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(task_result, dict):
        return {}
    nested = task_result.get("backtest_results")
    if isinstance(nested, dict):
        return nested
    return task_result


def _infer_window_label(task_name: str, backtest_result: Dict[str, Any]) -> str:
    lowered = (task_name or "").lower()
    for label in ["2024-2025-full", "2025-full", "2024-full", "2026-ytd"]:
        if label in lowered:
            return label

    quarter_match = re.search(r"(20\d{2})[-_]?q([1-4])", lowered)
    if quarter_match:
        return f"{quarter_match.group(1)}-q{quarter_match.group(2)}"

    bimonth_match = re.search(r"(20\d{2})[-_](\d{2})_(\d{2})", lowered)
    if bimonth_match:
        return f"{bimonth_match.group(1)}-{bimonth_match.group(2)}_{bimonth_match.group(3)}"

    start_date = backtest_result.get("start_date")
    end_date = backtest_result.get("end_date")
    if isinstance(start_date, str) and isinstance(end_date, str):
        return f"{start_date[:10]}→{end_date[:10]}"

    return "unclassified"


def _query_signal_summary(session: Any, task_id: str) -> Dict[str, Any]:
    rows = session.execute(
        text(
            """
            SELECT stock_code, signal_type, executed, execution_reason
            FROM signal_records
            WHERE task_id = :task_id
            """
        ),
        {"task_id": task_id},
    ).fetchall()

    raw_signal_count = len(rows)
    executed_signal_count = sum(1 for row in rows if bool(row.executed))
    rejected_signal_count = raw_signal_count - executed_signal_count
    reason_counts = Counter(
        (
            (row.execution_reason or "EXECUTED")
            if bool(row.executed)
            else (row.execution_reason or "UNEXECUTED_NO_REASON")
        )
        for row in rows
    )
    stock_counts = Counter(row.stock_code for row in rows)
    signal_type_counts = Counter(row.signal_type for row in rows)

    return {
        "raw_signal_count": raw_signal_count,
        "executed_signal_count": executed_signal_count,
        "rejected_signal_count": rejected_signal_count,
        "signal_type_counts": dict(signal_type_counts),
        "stock_signal_counts": dict(stock_counts),
        "top_execution_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reason_counts.most_common(3)
        ],
        "top_signal_stocks": [
            {"stock_code": stock_code, "count": count}
            for stock_code, count in stock_counts.most_common(3)
        ],
    }


def _normalize_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)


def _extract_cost_metrics(task_result: Dict[str, Any]) -> Dict[str, Any]:
    cost_statistics = _safe_json_dict(task_result.get("cost_statistics"))
    portfolio_history = task_result.get("portfolio_history")
    last_snapshot = (
        portfolio_history[-1]
        if isinstance(portfolio_history, list) and portfolio_history
        else {}
    )
    final_value = _safe_number(task_result.get("final_value")) or _safe_number(
        last_snapshot.get("portfolio_value")
    )
    final_value_without_cost = _safe_number(
        task_result.get("final_value_without_cost")
    ) or _safe_number(last_snapshot.get("portfolio_value_without_cost"))
    total_return = _safe_number(task_result.get("total_return")) or _safe_number(
        last_snapshot.get("total_return")
    )
    total_return_without_cost = _safe_number(
        task_result.get("total_return_without_cost")
    ) or _safe_number(last_snapshot.get("total_return_without_cost"))
    total_cost = _safe_number(cost_statistics.get("total_cost"))
    gross_minus_net_value_gap = (
        final_value_without_cost - final_value
        if final_value_without_cost is not None and final_value is not None
        else total_cost
    )
    gross_minus_net_return_gap = (
        total_return_without_cost - total_return
        if total_return_without_cost is not None and total_return is not None
        else None
    )
    return {
        "total_cost": total_cost,
        "total_commission": _safe_number(cost_statistics.get("total_commission")),
        "total_slippage": _safe_number(cost_statistics.get("total_slippage")),
        "cost_ratio": _safe_number(cost_statistics.get("cost_ratio")),
        "final_value_without_cost": final_value_without_cost,
        "total_return_without_cost": total_return_without_cost,
        "gross_minus_net_value_gap": gross_minus_net_value_gap,
        "gross_minus_net_return_gap": gross_minus_net_return_gap,
    }


def _extract_monthly_return_summary(task_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mean": _safe_number(task_result.get("monthly_return_mean")),
        "std": _safe_number(task_result.get("monthly_return_std")),
        "best_month": _safe_number(task_result.get("best_month")),
        "worst_month": _safe_number(task_result.get("worst_month")),
        "positive_months": int(task_result.get("positive_months") or 0),
        "negative_months": int(task_result.get("negative_months") or 0),
    }


def _extract_stock_contribution_summary(task_result: Dict[str, Any]) -> Dict[str, Any]:
    stock_details = task_result.get("stock_performance_detail")
    stock_details = stock_details if isinstance(stock_details, list) else []
    normalized_details = []
    for item in stock_details:
        if not isinstance(item, dict):
            continue
        normalized_details.append(
            {
                "stock_code": item.get("stock_code"),
                "total_pnl": _safe_number(item.get("total_pnl")) or 0.0,
                "trade_count": int(item.get("trade_count") or 0),
                "avg_pnl_per_trade": _safe_number(item.get("avg_pnl_per_trade")),
            }
        )
    normalized_details.sort(key=lambda item: item["total_pnl"], reverse=True)
    best_stock = task_result.get("best_performing_stock")
    best_stock = (
        best_stock
        if isinstance(best_stock, dict)
        else (normalized_details[0] if normalized_details else None)
    )
    worst_stock = task_result.get("worst_performing_stock")
    worst_stock = (
        worst_stock
        if isinstance(worst_stock, dict)
        else (normalized_details[-1] if normalized_details else None)
    )
    return {
        "best_stock": best_stock,
        "worst_stock": worst_stock,
        "top_contributors": normalized_details[:3],
        "bottom_contributors": sorted(
            normalized_details, key=lambda item: item["total_pnl"]
        )[:3],
    }


def _default_bridge_summary(model_id: str) -> Dict[str, Any]:
    return {
        "model_id": model_id,
        "task_count": 0,
        "tasks": [],
        "best_by_total_return": None,
        "best_by_sharpe": None,
        "smallest_drawdown": None,
        "cost_vs_gross_gap_rollup": {
            "task_count": 0,
            "tasks": [],
            "largest_cost_gap": None,
            "best_gross_return": None,
            "best_net_return": None,
        },
        "per_stock_contribution_rollup": {
            "stocks": [],
            "best_overall": None,
            "worst_overall": None,
        },
    }


def build_portfolio_bridge_summary(
    session: Any, model_id: str, max_tasks: int = 12
) -> Dict[str, Any]:
    """Return compact formal-task portfolio summaries for a model."""
    if not model_id:
        return _default_bridge_summary(model_id)

    try:
        task_rows = session.execute(
            text(
                """
                SELECT task_id, task_name, status, created_at, config, result
                FROM tasks
                WHERE task_type = 'backtest' AND status = 'completed'
                ORDER BY created_at DESC
                """
            )
        ).fetchall()
    except Exception as exc:
        logger.warning(f"查询模型 {model_id} 的 bridge tasks 失败: {exc}")
        return _default_bridge_summary(model_id)

    matched_tasks = []
    stock_rollup: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        task_config = _safe_json_dict(row.config)
        task_model_id = _extract_model_id_from_task(task_config)
        if task_model_id != model_id:
            continue

        task_result = _extract_backtest_result(_safe_json_dict(row.result))
        signal_summary = _query_signal_summary(session, row.task_id)
        portfolio_metrics = {
            "final_value": _safe_number(task_result.get("final_value")),
            "total_return": _safe_number(task_result.get("total_return")),
            "annualized_return": _safe_number(task_result.get("annualized_return")),
            "volatility": _safe_number(task_result.get("volatility")),
            "sharpe_ratio": _safe_number(task_result.get("sharpe_ratio")),
            "max_drawdown": _safe_number(task_result.get("max_drawdown")),
            "total_trades": int(task_result.get("total_trades") or 0),
            "win_rate": _safe_number(task_result.get("win_rate")),
            "profit_factor": _safe_number(task_result.get("profit_factor")),
        }
        cost_metrics = _extract_cost_metrics(task_result)
        monthly_return_summary = _extract_monthly_return_summary(task_result)
        stock_contribution_summary = _extract_stock_contribution_summary(task_result)
        stock_signal_counts = signal_summary.get("stock_signal_counts") or {}
        for item in task_result.get("stock_performance_detail") or []:
            if not isinstance(item, dict) or not item.get("stock_code"):
                continue
            stock_code = item["stock_code"]
            pnl = _safe_number(item.get("total_pnl")) or 0.0
            bucket = stock_rollup.setdefault(
                stock_code,
                {
                    "stock_code": stock_code,
                    "task_mentions": 0,
                    "positive_task_count": 0,
                    "negative_task_count": 0,
                    "total_pnl": 0.0,
                    "signal_count": 0,
                },
            )
            bucket["task_mentions"] += 1
            bucket["total_pnl"] += pnl
            if pnl > 0:
                bucket["positive_task_count"] += 1
            elif pnl < 0:
                bucket["negative_task_count"] += 1
            bucket["signal_count"] += int(stock_signal_counts.get(stock_code) or 0)
        matched_tasks.append(
            {
                "task_id": row.task_id,
                "task_name": row.task_name,
                "status": row.status,
                "created_at": _normalize_datetime(getattr(row, "created_at", None)),
                "window_label": _infer_window_label(row.task_name, task_result),
                "strategy_name": task_result.get("strategy_name")
                or (task_config.get("backtest_config") or {}).get("strategy_name"),
                "period": {
                    "start_date": task_result.get("start_date"),
                    "end_date": task_result.get("end_date"),
                },
                "portfolio_metrics": portfolio_metrics,
                "signal_summary": signal_summary,
                "cost_metrics": cost_metrics,
                "monthly_return_summary": monthly_return_summary,
                "stock_contribution_summary": stock_contribution_summary,
            }
        )

    summary = _default_bridge_summary(model_id)
    if not matched_tasks:
        return summary

    summary["task_count"] = len(matched_tasks)
    summary["tasks"] = matched_tasks[:max_tasks]

    valid_returns = [
        task
        for task in matched_tasks
        if task["portfolio_metrics"].get("total_return") is not None
    ]
    valid_sharpes = [
        task
        for task in matched_tasks
        if task["portfolio_metrics"].get("sharpe_ratio") is not None
    ]
    valid_drawdowns = [
        task
        for task in matched_tasks
        if task["portfolio_metrics"].get("max_drawdown") is not None
    ]

    if valid_returns:
        best_return = max(
            valid_returns, key=lambda task: task["portfolio_metrics"]["total_return"]
        )
        summary["best_by_total_return"] = {
            "task_id": best_return["task_id"],
            "task_name": best_return["task_name"],
            "window_label": best_return["window_label"],
            "total_return": best_return["portfolio_metrics"]["total_return"],
        }
    if valid_sharpes:
        best_sharpe = max(
            valid_sharpes, key=lambda task: task["portfolio_metrics"]["sharpe_ratio"]
        )
        summary["best_by_sharpe"] = {
            "task_id": best_sharpe["task_id"],
            "task_name": best_sharpe["task_name"],
            "window_label": best_sharpe["window_label"],
            "sharpe_ratio": best_sharpe["portfolio_metrics"]["sharpe_ratio"],
        }
    if valid_drawdowns:
        smallest_drawdown = max(
            valid_drawdowns, key=lambda task: task["portfolio_metrics"]["max_drawdown"]
        )
        summary["smallest_drawdown"] = {
            "task_id": smallest_drawdown["task_id"],
            "task_name": smallest_drawdown["task_name"],
            "window_label": smallest_drawdown["window_label"],
            "max_drawdown": smallest_drawdown["portfolio_metrics"]["max_drawdown"],
        }

    valid_cost_tasks = [
        task
        for task in matched_tasks
        if (task.get("cost_metrics") or {}).get("gross_minus_net_value_gap") is not None
    ]
    summary["cost_vs_gross_gap_rollup"]["task_count"] = len(valid_cost_tasks)
    summary["cost_vs_gross_gap_rollup"]["tasks"] = [
        {
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "window_label": task["window_label"],
            "total_cost": task["cost_metrics"].get("total_cost"),
            "gross_minus_net_value_gap": task["cost_metrics"].get(
                "gross_minus_net_value_gap"
            ),
            "total_return": task["portfolio_metrics"].get("total_return"),
            "total_return_without_cost": task["cost_metrics"].get(
                "total_return_without_cost"
            ),
        }
        for task in valid_cost_tasks[:max_tasks]
    ]
    if valid_cost_tasks:
        largest_cost_gap = max(
            valid_cost_tasks,
            key=lambda task: task["cost_metrics"].get("gross_minus_net_value_gap")
            or float("-inf"),
        )
        summary["cost_vs_gross_gap_rollup"]["largest_cost_gap"] = {
            "task_id": largest_cost_gap["task_id"],
            "task_name": largest_cost_gap["task_name"],
            "window_label": largest_cost_gap["window_label"],
            "gross_minus_net_value_gap": largest_cost_gap["cost_metrics"].get(
                "gross_minus_net_value_gap"
            ),
        }
        best_gross = max(
            valid_cost_tasks,
            key=lambda task: task["cost_metrics"].get("total_return_without_cost")
            or float("-inf"),
        )
        summary["cost_vs_gross_gap_rollup"]["best_gross_return"] = {
            "task_id": best_gross["task_id"],
            "task_name": best_gross["task_name"],
            "window_label": best_gross["window_label"],
            "total_return_without_cost": best_gross["cost_metrics"].get(
                "total_return_without_cost"
            ),
        }
        if valid_returns:
            best_net = max(
                valid_returns,
                key=lambda task: task["portfolio_metrics"]["total_return"],
            )
            summary["cost_vs_gross_gap_rollup"]["best_net_return"] = {
                "task_id": best_net["task_id"],
                "task_name": best_net["task_name"],
                "window_label": best_net["window_label"],
                "total_return": best_net["portfolio_metrics"].get("total_return"),
            }

    aggregated_stocks = sorted(
        stock_rollup.values(), key=lambda item: item["total_pnl"], reverse=True
    )
    summary["per_stock_contribution_rollup"]["stocks"] = aggregated_stocks
    if aggregated_stocks:
        summary["per_stock_contribution_rollup"]["best_overall"] = aggregated_stocks[0]
        summary["per_stock_contribution_rollup"]["worst_overall"] = min(
            aggregated_stocks, key=lambda item: item["total_pnl"]
        )

    return summary
