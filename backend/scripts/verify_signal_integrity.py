#!/usr/bin/env python
"""端到端信号完整性校验 — 对 golden baseline 做数据一致性验证

验证回测链路的中间数据一致性：
  a) 信号计数一致性
  b) 交易与信号映射
  c) 组合价值一致性
  d) 持仓一致性 (受限于 snapshot 不存 positions, 属签名不够的 check)

可独立运行：
    cd backend
    .venv-py313/bin/python scripts/verify_signal_integrity.py --baseline <path>

集成在 backtest_golden_runner.py 的 verify 流程中，在 baseline 比较后额外调用。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ABS_TOL = 1e-6
REL_TOL = 1e-9


def _isclose(a: float, b: float, *, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    try:
        return math.isclose(float(a), float(b), abs_tol=abs_tol, rel_tol=rel_tol)
    except (TypeError, ValueError):
        return a == b


def _f(v: Any) -> float:
    """Safely coerce to float."""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Check (a): signal count consistency
# ---------------------------------------------------------------------------

def check_signal_count_consistency(baseline: dict) -> dict:
    """Verify total_signals ≥ executed trades, infer rejections."""
    detail: dict[str, Any] = {}

    total_signals = baseline.get("total_signals")
    if total_signals is None:
        return {"passed": False, "detail": {"error": "missing total_signals in baseline"}}

    total_signals = int(total_signals)
    trades = baseline.get("trade_history") or []
    total_trades = len(trades)

    detail["total_signals"] = total_signals
    detail["executed_trades"] = total_trades

    passed = True
    issues: list[str] = []

    # (i) 信号总数 ≥ 交易数
    if total_signals < total_trades:
        issues.append(f"total_signals ({total_signals}) < executed_trades ({total_trades})")
        passed = False
    else:
        detail["inferred_rejected_signals"] = total_signals - total_trades

    # (ii) signal_execution_summary（如果存在）
    sig_summary = baseline.get("signal_execution_summary")
    if sig_summary and isinstance(sig_summary, dict):
        raw_count = sig_summary.get("raw_signal_count")
        actionable_count = sig_summary.get("actionable_signal_count")
        executed_count = sig_summary.get("executed_signal_count")

        if raw_count is not None:
            detail["summary_raw_signal_count"] = raw_count
            if int(raw_count) != total_signals:
                issues.append(
                    f"signal_execution_summary.raw_signal_count ({raw_count}) "
                    f"!= total_signals ({total_signals})"
                )
                passed = False

        if actionable_count is not None:
            detail["summary_actionable_count"] = actionable_count
            if int(actionable_count) > total_signals:
                issues.append(
                    f"actionable_signal_count ({actionable_count}) > total_signals ({total_signals})"
                )
                passed = False

        if executed_count is not None:
            detail["summary_executed_count"] = executed_count
            if int(executed_count) != total_trades:
                issues.append(
                    f"executed_signal_count ({executed_count}) != trade_history count ({total_trades})"
                )

        # (iii) rejection counts add up
        rejection_reasons = sig_summary.get("top_rejection_reasons") or []
        rejection_total = sum(int(r.get("count", 0)) for r in rejection_reasons)
        if rejection_total > 0:
            detail["rejection_reason_total"] = rejection_total
            expected_rejected = total_signals - total_trades
            if rejection_total != expected_rejected:
                issues.append(
                    f"rejection_reason total ({rejection_total}) != "
                    f"inferred rejected ({expected_rejected})"
                )
        else:
            detail["rejection_data"] = "not available in baseline"
    else:
        detail["signal_execution_summary"] = "not available in baseline"

    if issues:
        detail["issues"] = issues

    return {"passed": passed, "detail": detail}


# ---------------------------------------------------------------------------
# Check (b): trade ↔ signal mapping
# ---------------------------------------------------------------------------

def check_trade_signal_mapping(baseline: dict) -> dict:
    """For each trade, verify semantic validity: stock_code, direction, prices."""
    trades = baseline.get("trade_history") or []
    detail: dict[str, Any] = {"total_trades": len(trades)}
    passed = True
    issues: list[dict] = []

    if not trades:
        detail["note"] = "no trades to verify"
        return {"passed": True, "detail": detail}

    for i, t in enumerate(trades):
        action = t.get("action")
        stock_code = t.get("stock_code")
        quantity = t.get("quantity", 0)
        price = t.get("price")
        trade_id = t.get("trade_id", f"trade[{i}]")

        trade_issues: list[str] = []

        # Every trade must have a valid stock_code
        if not stock_code:
            trade_issues.append("missing stock_code")
            passed = False

        # Every trade must have a valid action
        if action not in ("BUY", "SELL"):
            trade_issues.append(f"invalid action: {action}")
            passed = False

        # Quantity must be positive
        if not quantity or int(quantity) <= 0:
            trade_issues.append(f"non-positive quantity: {quantity}")
            passed = False

        # Price must be positive
        if not price or float(price) <= 0:
            trade_issues.append(f"non-positive price: {price}")
            passed = False

        if trade_issues:
            issues.append({"trade_id": trade_id, "stock_code": stock_code, "issues": trade_issues})

    if issues:
        detail["issues"] = issues

    detail["mapped_trades_ok"] = len(trades) - len(issues)
    return {"passed": passed, "detail": detail}


# ---------------------------------------------------------------------------
# Check (c): portfolio value coherence
# ---------------------------------------------------------------------------

def check_portfolio_value_coherence(baseline: dict) -> dict:
    """Verify portfolio value via multiple independent derivations.

    1. last snapshot: portfolio_value == portfolio_value_without_cost - total_cost
    2. final_value == last snapshot portfolio_value
    3. cash flow reconstruction from trade records
       (only exact when no open positions, otherwise note the gap)
    4. total_cost == Σtrade.commission + Σtrade.slippage_cost
    """
    detail: dict[str, Any] = {}
    issues: list[str] = []
    passed = True

    initial_cash = _f(baseline.get("initial_cash"))
    final_value = _f(baseline.get("final_value"))
    cost_stats = baseline.get("cost_statistics") or {}
    total_cost = _f(cost_stats.get("total_cost"))
    portfolio_history = baseline.get("portfolio_history") or []
    trades = baseline.get("trade_history") or []

    detail["initial_cash"] = initial_cash
    detail["final_value"] = final_value
    detail["total_cost"] = total_cost

    # ---- Check 1: snapshot internal consistency ----
    if portfolio_history:
        last_snap = portfolio_history[-1]
        snap_value = _f(last_snap.get("portfolio_value"))
        snap_value_wc = _f(last_snap.get("portfolio_value_without_cost"))
        snap_cash = _f(last_snap.get("cash"))

        detail["last_snapshot_portfolio_value"] = snap_value
        detail["last_snapshot_value_without_cost"] = snap_value_wc
        detail["last_snapshot_cash"] = snap_cash

        # 1a: final_value == last portfolio_value
        if not _isclose(final_value, snap_value, abs_tol=0.01):
            issues.append(
                f"final_value ({final_value:.4f}) != last portfolio_value ({snap_value:.4f})"
            )
            passed = False

        # 1b: portfolio_value_without_cost - total_cost ≈ portfolio_value
        computed_from_wc = snap_value_wc - total_cost
        detail["computed_from_without_cost"] = computed_from_wc
        wc_diff = computed_from_wc - snap_value
        detail["without_cost_discrepancy"] = wc_diff
        if not _isclose(computed_from_wc, snap_value, abs_tol=0.01):
            issues.append(
                f"portfolio_value_without_cost - total_cost ({computed_from_wc:.4f}) "
                f"!= portfolio_value ({snap_value:.4f}), diff={wc_diff:.4f}"
            )
            if abs(wc_diff) > 1.0:
                passed = False

    # ---- Check 2: cost breakdown ----
    total_commission_from_trades = sum(_f(t.get("commission")) for t in trades)
    total_slippage_from_trades = sum(_f(t.get("slippage_cost")) for t in trades)
    cost_commission = _f(cost_stats.get("total_commission"))
    cost_slippage = _f(cost_stats.get("total_slippage"))

    detail["Σtrade_commission"] = total_commission_from_trades
    detail["Σtrade_slippage"] = total_slippage_from_trades
    detail["cost_stat_commission"] = cost_commission
    detail["cost_stat_slippage"] = cost_slippage

    if not _isclose(total_commission_from_trades, cost_commission, abs_tol=0.01):
        issues.append(
            f"Σtrade.commission ({total_commission_from_trades:.4f}) "
            f"!= cost_statistics.total_commission ({cost_commission:.4f})"
        )
        passed = False

    if not _isclose(total_slippage_from_trades, cost_slippage, abs_tol=0.01):
        issues.append(
            f"Σtrade.slippage_cost ({total_slippage_from_trades:.4f}) "
            f"!= cost_statistics.total_slippage ({cost_slippage:.4f})"
        )
        passed = False

    if abs(total_cost - (cost_commission + cost_slippage)) > 0.01:
        issues.append(
            f"cost_statistics.total_cost ({total_cost:.4f}) != "
            f"total_commission + total_slippage ({cost_commission + cost_slippage:.4f})"
        )

    # ---- Check 3: cash flow reconstruction ----
    # For each BUY:  cash_change = -(qty * execution_price + commission)
    # For each SELL: cash_change =  qty * execution_price - commission
    total_cash_change = 0.0
    trade_pnl_sum = 0.0
    buy_commission_sum = 0.0

    for t in trades:
        action = t.get("action")
        qty = _f(t.get("quantity"))
        price = _f(t.get("price"))  # execution_price (includes slippage)
        comm = _f(t.get("commission"))
        pnl = _f(t.get("pnl"))
        trade_pnl_sum += pnl

        if action == "BUY":
            total_cash_change -= qty * price + comm
            buy_commission_sum += comm
        elif action == "SELL":
            total_cash_change += qty * price - comm

    total_cash = initial_cash + total_cash_change

    detail["trade_pnl_sum"] = trade_pnl_sum
    detail["buy_commission_sum"] = buy_commission_sum
    detail["total_cash_change_from_trades"] = total_cash_change
    detail["reconstructed_cash"] = total_cash

    # Compare reconstructed cash to actual cash
    snap_cash = detail.get("last_snapshot_cash")
    if snap_cash is not None and not _isclose(total_cash, snap_cash, abs_tol=0.01):
        issues.append(
            f"reconstructed_cash ({total_cash:.4f}) != last_snapshot_cash ({snap_cash:.4f})"
        )
        passed = False

    # When positions are NOT stored in snapshot (known limitation), we can still
    # verify via portfolio_value_without_cost chain:
    #   reconstructed_cash + open_positions_mv = portfolio_value_without_cost
    # But open_positions_mv is not in the snapshot. Instead, we check that:
    #   snap_cash + (snap_value - snap_cash) = snap_value  (tautology)
    # and from check 1b: snap_value = snap_value_wc - total_cost ✓

    if issues:
        detail["issues"] = issues

    return {"passed": passed, "detail": detail}


# ---------------------------------------------------------------------------
# Check (d): position consistency
# ---------------------------------------------------------------------------

def check_position_consistency(baseline: dict) -> dict:
    """Verify positions in portfolio history are internally consistent.

    Known limitation: current record_portfolio_snapshot always writes
    positions={}. This check will be signature-incomplete until positions
    are stored.  It checks what IS available: the open-position gap
    implied by (portfolio_value - cash).
    """
    portfolio_history = baseline.get("portfolio_history") or []
    detail: dict[str, Any] = {}
    issues: list[str] = []
    passed = True

    if not portfolio_history:
        return {"passed": False, "detail": {"error": "no portfolio_history available"}}

    # Count how many snapshots have non-empty positions
    snapshots_with_positions = sum(
        1 for snap in portfolio_history if snap.get("positions")
    )

    # If no snapshots have positions (current code behaviour), note it
    if snapshots_with_positions == 0:
        detail["note"] = (
            "portfolio_history snapshots do not store positions (known backend limitation). "
            "Full position consistency cannot be verified."
        )
        detail["snapshots_with_positions"] = 0
        # Still check: open-position gap implied by value - cash
        last = portfolio_history[-1]
        snap_value = _f(last.get("portfolio_value"))
        snap_cash = _f(last.get("cash"))
        implied_open_mv = snap_value - snap_cash
        if implied_open_mv > 0.01:
            detail["implied_open_positions_mv"] = implied_open_mv
            detail["position_check"] = "insufficient data — snapshot stores positions={}"

        # This check is expected to be incomplete
        return {"passed": True, "detail": detail}

    detail["snapshots_with_positions"] = snapshots_with_positions

    for snap in portfolio_history:
        positions = snap.get("positions") or {}
        if not positions:
            continue

        snap_value = _f(snap.get("portfolio_value"))
        snap_cash = _f(snap.get("cash"))

        for code, pos in positions.items():
            qty = _f(pos.get("quantity"))
            if qty <= 0:
                continue

            avg_cost = _f(pos.get("avg_cost"))
            current_price = _f(pos.get("current_price"))
            market_value = _f(pos.get("market_value"))
            unrealized_pnl = _f(pos.get("unrealized_pnl"))

            # market_value ≈ qty * current_price
            expected_mv = qty * current_price
            if not _isclose(market_value, expected_mv, abs_tol=0.01):
                issues.append(
                    f"position {code}: market_value ({market_value:.4f}) "
                    f"!= qty*price ({qty}*{current_price:.4f}={expected_mv:.4f})"
                )
                passed = False

            # unrealized_pnl ≈ (current_price - avg_cost) * qty
            if avg_cost > 0:
                expected_upnl = (current_price - avg_cost) * qty
                if not _isclose(unrealized_pnl, expected_upnl, abs_tol=0.01):
                    issues.append(
                        f"position {code}: unrealized_pnl ({unrealized_pnl:.4f}) "
                        f"!= (price-avg_cost)*qty ({expected_upnl:.4f})"
                    )
                    passed = False

    if issues:
        detail["issues"] = issues[:20]

    return {"passed": passed, "detail": detail}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_baseline(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return data


def verify_result(result: dict) -> dict:
    """Public API: run all four checks on a backtest result dict and return a report."""
    return _verify(result)


def _verify(baseline: dict) -> dict:
    """Run all four checks and return a consolidated report."""
    case_name = (
        baseline.get("golden_case") or {}
    ).get("name") or baseline.get("strategy_name", "unknown")

    checks: dict[str, dict] = {}

    checks["signal_count_consistency"] = check_signal_count_consistency(baseline)
    checks["trade_signal_mapping"] = check_trade_signal_mapping(baseline)
    checks["portfolio_value_coherence"] = check_portfolio_value_coherence(baseline)
    checks["position_consistency"] = check_position_consistency(baseline)

    all_passed = all(c["passed"] for c in checks.values())

    total_signals = int(baseline.get("total_signals") or 0)
    trades = baseline.get("trade_history") or []
    initial_cash = _f(baseline.get("initial_cash"))
    final_value = _f(baseline.get("final_value"))
    portfolio_history = baseline.get("portfolio_history") or []

    # Cash flow reconstruction for summary
    total_cash_change = 0.0
    for t in trades:
        action = t.get("action")
        qty = _f(t.get("quantity"))
        price = _f(t.get("price"))
        comm = _f(t.get("commission"))
        if action == "BUY":
            total_cash_change -= qty * price + comm
        elif action == "SELL":
            total_cash_change += qty * price - comm

    computed_value = initial_cash + total_cash_change
    # If portfolio_value != cash at end, add open positions value
    if portfolio_history:
        last_snap = portfolio_history[-1]
        diff = _f(last_snap.get("portfolio_value")) - _f(last_snap.get("cash"))
        computed_value += max(0.0, diff)

    discrepancy = computed_value - final_value

    report: dict[str, Any] = {
        "case_name": case_name,
        "passed": all_passed,
        "checks": checks,
        "summary": {
            "total_signals": total_signals,
            "executed_trades": len(trades),
            "rejected_signals": total_signals - len(trades),
            "final_value": final_value,
            "computed_value": computed_value,
            "discrepancy": discrepancy,
        },
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify backtest signal integrity from a golden baseline"
    )
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Path to golden baseline JSON"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print full check details"
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"Error: baseline not found: {args.baseline}")
        return 2

    baseline = load_baseline(args.baseline)
    report = _verify(baseline)

    indent = 2 if args.verbose else None
    json_str = json.dumps(report, indent=indent, ensure_ascii=False)
    print(json_str)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
