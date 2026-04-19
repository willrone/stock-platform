from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.services.models.portfolio_bridge import build_portfolio_bridge_summary


def _seed_schema(session: Session) -> None:
    session.execute(
        text(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                config TEXT,
                created_at TEXT,
                result TEXT
            )
            """
        )
    )
    session.execute(
        text(
            """
            CREATE TABLE signal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                executed BOOLEAN NOT NULL,
                execution_reason TEXT
            )
            """
        )
    )
    session.commit()


def test_build_portfolio_bridge_summary_collects_task_and_signal_rollups() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with Session(engine) as session:
        _seed_schema(session)
        session.execute(
            text(
                """
                INSERT INTO tasks (task_id, task_name, task_type, status, config, created_at, result)
                VALUES
                (:task_id, :task_name, 'backtest', 'completed', :config, '2026-04-14T12:00:00', :result),
                (:other_task_id, :other_task_name, 'backtest', 'completed', :other_config, '2026-04-13T12:00:00', :other_result)
                """
            ),
            {
                "task_id": "task-official-2024",
                "task_name": "hermes-qlib-costmodel-official-2024-full",
                "config": '{"backtest_config": {"strategy_config": {"model_id": "model-official"}}}',
                "result": '{"strategy_name": "model_topk_dropout", "start_date": "2024-01-01T00:00:00", "end_date": "2024-12-31T00:00:00", "total_return": 0.031, "sharpe_ratio": 1.2, "max_drawdown": -0.08, "total_trades": 12, "final_value": 1031000.0, "cost_statistics": {"total_cost": 3200.0, "total_commission": 2100.0, "total_slippage": 1100.0, "cost_ratio": 0.0032}, "portfolio_history": [{"date": "2024-12-31T00:00:00", "portfolio_value": 1031000.0, "portfolio_value_without_cost": 1034200.0, "total_return": 0.031, "total_return_without_cost": 0.0342}], "monthly_return_mean": 0.01, "monthly_return_std": 0.02, "best_month": 0.05, "worst_month": -0.03, "positive_months": 8, "negative_months": 4, "stock_performance_detail": [{"stock_code": "600036.SH", "total_pnl": 5500.0, "trade_count": 8, "avg_pnl_per_trade": 687.5}, {"stock_code": "601288.SH", "total_pnl": -1200.0, "trade_count": 4, "avg_pnl_per_trade": -300.0}], "best_performing_stock": {"stock_code": "600036.SH", "total_pnl": 5500.0, "trade_count": 8, "avg_pnl_per_trade": 687.5}, "worst_performing_stock": {"stock_code": "601288.SH", "total_pnl": -1200.0, "trade_count": 4, "avg_pnl_per_trade": -300.0}}',
                "other_task_id": "task-other-model",
                "other_task_name": "hermes-other-model-2024-full",
                "other_config": '{"backtest_config": {"strategy_config": {"model_id": "model-other"}}}',
                "other_result": '{"strategy_name": "model_topk_dropout", "start_date": "2024-01-01T00:00:00", "end_date": "2024-12-31T00:00:00", "total_return": 0.999, "sharpe_ratio": 9.9, "max_drawdown": -0.01, "total_trades": 1}',
            },
        )
        session.execute(
            text(
                """
                INSERT INTO signal_records (task_id, stock_code, signal_type, executed, execution_reason)
                VALUES
                ('task-official-2024', '600036.SH', 'BUY', 1, NULL),
                ('task-official-2024', '600036.SH', 'SELL', 1, NULL),
                ('task-official-2024', '601288.SH', 'BUY', 0, '可买数量不足: 无法买入200股'),
                ('task-other-model', '000001.SZ', 'BUY', 1, NULL)
                """
            )
        )
        session.commit()

        summary = build_portfolio_bridge_summary(session, 'model-official')

    assert summary['model_id'] == 'model-official'
    assert summary['task_count'] == 1
    assert len(summary['tasks']) == 1
    task = summary['tasks'][0]
    assert task['task_id'] == 'task-official-2024'
    assert task['window_label'] == '2024-full'
    assert task['portfolio_metrics']['total_return'] == 0.031
    assert task['signal_summary']['raw_signal_count'] == 3
    assert task['signal_summary']['executed_signal_count'] == 2
    assert task['signal_summary']['rejected_signal_count'] == 1
    assert task['signal_summary']['top_execution_reasons'][0] == {
        'reason': 'EXECUTED',
        'count': 2,
    }
    assert task['signal_summary']['top_signal_stocks'][0] == {
        'stock_code': '600036.SH',
        'count': 2,
    }
    assert task['cost_metrics'] == {
        'total_cost': 3200.0,
        'total_commission': 2100.0,
        'total_slippage': 1100.0,
        'cost_ratio': 0.0032,
        'final_value_without_cost': 1034200.0,
        'total_return_without_cost': 0.0342,
        'gross_minus_net_value_gap': 3200.0,
        'gross_minus_net_return_gap': pytest.approx(0.0032),
    }
    assert task['monthly_return_summary'] == {
        'mean': 0.01,
        'std': 0.02,
        'best_month': 0.05,
        'worst_month': -0.03,
        'positive_months': 8,
        'negative_months': 4,
    }
    assert task['stock_contribution_summary']['best_stock']['stock_code'] == '600036.SH'
    assert task['stock_contribution_summary']['worst_stock']['stock_code'] == '601288.SH'
    assert summary['cost_vs_gross_gap_rollup']['largest_cost_gap'] == {
        'task_id': 'task-official-2024',
        'task_name': 'hermes-qlib-costmodel-official-2024-full',
        'window_label': '2024-full',
        'gross_minus_net_value_gap': 3200.0,
    }
    assert summary['cost_vs_gross_gap_rollup']['best_gross_return']['total_return_without_cost'] == 0.0342
    assert summary['per_stock_contribution_rollup']['best_overall']['stock_code'] == '600036.SH'
    assert summary['per_stock_contribution_rollup']['worst_overall']['stock_code'] == '601288.SH'
    assert summary['per_stock_contribution_rollup']['stocks'][0]['signal_count'] == 2
    assert summary['best_by_total_return'] == {
        'task_id': 'task-official-2024',
        'task_name': 'hermes-qlib-costmodel-official-2024-full',
        'window_label': '2024-full',
        'total_return': 0.031,
    }
    assert summary['best_by_sharpe']['sharpe_ratio'] == 1.2
    assert summary['smallest_drawdown']['max_drawdown'] == -0.08


def test_build_portfolio_bridge_summary_returns_empty_shape_when_no_tasks_match() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with Session(engine) as session:
        _seed_schema(session)
        summary = build_portfolio_bridge_summary(session, 'missing-model')

    assert summary == {
        'model_id': 'missing-model',
        'task_count': 0,
        'tasks': [],
        'best_by_total_return': None,
        'best_by_sharpe': None,
        'smallest_drawdown': None,
        'cost_vs_gross_gap_rollup': {
            'task_count': 0,
            'tasks': [],
            'largest_cost_gap': None,
            'best_gross_return': None,
            'best_net_return': None,
        },
        'per_stock_contribution_rollup': {
            'stocks': [],
            'best_overall': None,
            'worst_overall': None,
        },
    }
