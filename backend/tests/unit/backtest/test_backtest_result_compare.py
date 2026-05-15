from __future__ import annotations

from copy import deepcopy

from scripts.backtest_result_compare import compare_results


def _sample_result() -> dict:
    return {
        "strategy_name": "moving_average",
        "stock_codes": ["000001.SZ"],
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-02-01T00:00:00",
        "initial_cash": 100000.0,
        "final_value": 101000.0,
        "total_return": 0.01,
        "annualized_return": 0.12,
        "volatility": 0.2,
        "sharpe_ratio": 0.6,
        "max_drawdown": -0.03,
        "total_trades": 2,
        "win_rate": 1.0,
        "profit_factor": 2.0,
        "winning_trades": 1,
        "losing_trades": 0,
        "total_signals": 3,
        "trading_days": 20,
        "metrics": {
            "sharpe_ratio": 0.6,
            "total_return": 0.01,
            "annualized_return": 0.12,
            "max_drawdown": -0.03,
            "volatility": 0.2,
            "win_rate": 1.0,
            "profit_factor": 2.0,
            "total_trades": 2,
        },
        "cost_statistics": {
            "total_commission": 2.0,
            "total_slippage": 1.0,
            "total_cost": 3.0,
            "cost_ratio": 0.00003,
        },
        "trade_history": [
            {
                "trade_id": "T000001",
                "stock_code": "000001.SZ",
                "action": "BUY",
                "quantity": 100,
                "price": 10.0,
                "timestamp": "2024-01-05T00:00:00",
                "commission": 1.0,
                "slippage_cost": 0.1,
                "pnl": 0.0,
            }
        ],
        "portfolio_history": [
            {
                "date": "2024-01-05T00:00:00",
                "portfolio_value": 100000.0,
                "portfolio_value_without_cost": 100001.0,
                "cash": 99000.0,
                "positions_count": 1,
                "positions": {
                    "000001.SZ": {
                        "quantity": 100,
                        "avg_cost": 10.01,
                        "current_price": 10.0,
                        "market_value": 1000.0,
                        "unrealized_pnl": -1.0,
                    }
                },
                "total_return": 0.0,
                "total_return_without_cost": 0.00001,
            }
        ],
        "signal_execution_summary": {
            "execution_rate": 0.5,
            "execution_rate_actionable": 1.0,
            "raw_signal_count": 3,
            "actionable_signal_count": 1,
            "executed_signal_count": 1,
            "top_rejection_reasons": [{"reason": "no_position", "count": 2}],
        },
    }


def test_compare_results_accepts_identical_payloads() -> None:
    expected = _sample_result()
    actual = deepcopy(expected)

    assert compare_results(expected, actual) == []


def test_compare_results_rejects_trade_quantity_changes() -> None:
    expected = _sample_result()
    actual = deepcopy(expected)
    actual["trade_history"][0]["quantity"] = 200

    diffs = compare_results(expected, actual)

    assert any(diff.path == "trade_history[0].quantity" for diff in diffs)


def test_compare_results_rejects_rejection_reason_distribution_changes() -> None:
    expected = _sample_result()
    actual = deepcopy(expected)
    actual["signal_execution_summary"]["top_rejection_reasons"] = [
        {"reason": "cash_insufficient", "count": 2}
    ]

    diffs = compare_results(expected, actual)

    assert any(
        diff.path == "signal_execution_summary.rejection_reason_counts"
        for diff in diffs
    )
