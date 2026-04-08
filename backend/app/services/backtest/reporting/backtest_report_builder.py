"""Build normalized backtest result payloads for executor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..core.portfolio_manager import PortfolioManager
from ..models import BacktestConfig


@dataclass(frozen=True)
class BacktestReportBuildInput:
    """Input payload for assembling a backtest result report."""

    strategy_name: str
    stock_codes: list[str]
    start_date: datetime
    end_date: datetime
    config: BacktestConfig
    portfolio_manager: PortfolioManager
    performance_metrics: dict[str, float]
    strategy_config: Optional[dict[str, Any]] = None


class BacktestReportBuilder:
    """Assemble and normalize backtest result payloads."""

    def build_report(self, payload: BacktestReportBuildInput) -> dict[str, Any]:
        """Build the base backtest report without runtime diagnostics."""
        report = self._build_basic_report(payload)
        report["metrics"] = self._build_metrics(payload.performance_metrics)
        report["backtest_config"] = self._build_backtest_config(payload)
        report["trade_history"] = self._build_trade_history(payload.portfolio_manager)
        report["portfolio_history"] = self._build_portfolio_history(payload)
        report["cost_statistics"] = self._build_cost_statistics(payload)
        report.update(self._build_excess_return_metrics(payload))
        report.update(self._calculate_additional_metrics(payload.portfolio_manager))
        return report

    def attach_runtime_diagnostics(
        self,
        report: dict[str, Any],
        loop_result: dict[str, int],
        perf_breakdown: dict[str, float],
    ) -> None:
        """Attach executor runtime counters and timing diagnostics."""
        report["total_signals"] = loop_result.get("total_signals", 0)
        report["trading_days"] = loop_result.get("trading_days", 0)
        report["perf_breakdown"] = dict(perf_breakdown)

    def attach_signal_execution_summary(
        self,
        report: dict[str, Any],
        signal_stats: dict[str, Any],
    ) -> None:
        """Attach normalized signal execution summary to the report."""
        report["signal_execution_summary"] = {
            "execution_rate": signal_stats.get("execution_rate", 0.0),
            "execution_rate_actionable": signal_stats.get(
                "execution_rate_actionable", 0.0
            ),
            "raw_signal_count": signal_stats.get("raw_signal_count", 0),
            "actionable_signal_count": signal_stats.get(
                "actionable_signal_count", 0
            ),
            "executed_signal_count": signal_stats.get("executed_signal_count", 0),
            "top_rejection_reasons": signal_stats.get(
                "top_rejection_reasons", []
            ),
        }

    def _build_basic_report(
        self, payload: BacktestReportBuildInput
    ) -> dict[str, Any]:
        """Build strategy-level scalar fields for the report."""
        return {
            "strategy_name": payload.strategy_name,
            "stock_codes": payload.stock_codes,
            "start_date": payload.start_date.isoformat(),
            "end_date": payload.end_date.isoformat(),
            "initial_cash": payload.config.initial_cash,
            "final_value": self._resolve_final_value(payload.portfolio_manager),
            "total_return": payload.performance_metrics.get("total_return", 0),
            "annualized_return": payload.performance_metrics.get(
                "annualized_return", 0
            ),
            "volatility": payload.performance_metrics.get("volatility", 0),
            "sharpe_ratio": payload.performance_metrics.get("sharpe_ratio", 0),
            "max_drawdown": payload.performance_metrics.get("max_drawdown", 0),
            "total_trades": payload.performance_metrics.get("total_trades", 0),
            "win_rate": payload.performance_metrics.get("win_rate", 0),
            "profit_factor": payload.performance_metrics.get("profit_factor", 0),
            "winning_trades": payload.performance_metrics.get("winning_trades", 0),
            "losing_trades": payload.performance_metrics.get("losing_trades", 0),
        }

    def _resolve_final_value(self, portfolio_manager: PortfolioManager) -> float:
        """Resolve final portfolio value without regressing to cash-only valuation."""
        portfolio_history = getattr(portfolio_manager, "portfolio_history", None)
        if portfolio_history:
            return portfolio_history[-1]["portfolio_value"]
        return portfolio_manager.get_portfolio_value({})

    def _build_metrics(
        self, performance_metrics: dict[str, float]
    ) -> dict[str, float]:
        """Build optimizer-facing metrics block."""
        return {
            "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0),
            "total_return": performance_metrics.get("total_return", 0),
            "annualized_return": performance_metrics.get("annualized_return", 0),
            "max_drawdown": performance_metrics.get("max_drawdown", 0),
            "volatility": performance_metrics.get("volatility", 0),
            "win_rate": performance_metrics.get("win_rate", 0),
            "profit_factor": performance_metrics.get("profit_factor", 0),
            "total_trades": performance_metrics.get("total_trades", 0),
        }

    def _build_backtest_config(
        self, payload: BacktestReportBuildInput
    ) -> dict[str, Any]:
        """Build the serialized backtest config block."""
        config_dict: dict[str, Any] = {
            "strategy_name": payload.strategy_name,
            "start_date": payload.start_date.isoformat(),
            "end_date": payload.end_date.isoformat(),
            "initial_cash": payload.config.initial_cash,
            "commission_rate": payload.config.commission_rate,
            "slippage_rate": payload.config.slippage_rate,
            "max_position_size": payload.config.max_position_size,
        }
        if payload.strategy_config:
            config_dict["strategy_config"] = payload.strategy_config
        return config_dict

    def _build_trade_history(
        self, portfolio_manager: PortfolioManager
    ) -> list[dict[str, Any]]:
        """Serialize trade history for API responses."""
        return [
            self._serialize_trade(trade)
            for trade in getattr(portfolio_manager, "trades", [])
        ]

    def _serialize_trade(self, trade: Any) -> dict[str, Any]:
        """Serialize one trade object or dict into report schema."""
        timestamp = self._get_trade_attr(trade, "timestamp")
        return {
            "trade_id": self._get_trade_attr(trade, "trade_id"),
            "stock_code": self._get_trade_attr(trade, "stock_code"),
            "action": self._get_trade_attr(trade, "action"),
            "quantity": self._get_trade_attr(trade, "quantity"),
            "price": self._get_trade_attr(trade, "price"),
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
            "commission": self._get_trade_attr(trade, "commission"),
            "slippage_cost": self._get_trade_attr(trade, "slippage_cost", 0.0),
            "pnl": self._get_trade_attr(trade, "pnl"),
        }

    def _build_portfolio_history(
        self, payload: BacktestReportBuildInput
    ) -> list[dict[str, Any]]:
        """Serialize portfolio snapshots with returns and positions."""
        initial_cash = payload.config.initial_cash
        return [
            self._serialize_snapshot(snapshot, initial_cash)
            for snapshot in payload.portfolio_manager.portfolio_history
        ]

    def _serialize_snapshot(
        self, snapshot: dict[str, Any], initial_cash: float
    ) -> dict[str, Any]:
        """Serialize one portfolio snapshot."""
        portfolio_value = snapshot["portfolio_value"]
        value_without_cost = snapshot.get(
            "portfolio_value_without_cost", portfolio_value
        )
        return {
            "date": snapshot["date"].isoformat(),
            "portfolio_value": portfolio_value,
            "portfolio_value_without_cost": value_without_cost,
            "cash": snapshot["cash"],
            "positions_count": len(snapshot.get("positions", {})),
            "positions": snapshot.get("positions", {}),
            "total_return": self._calculate_return(portfolio_value, initial_cash),
            "total_return_without_cost": self._calculate_return(
                value_without_cost, initial_cash
            ),
        }

    def _calculate_return(self, portfolio_value: float, initial_cash: float) -> float:
        """Calculate cumulative return against initial cash."""
        if initial_cash <= 0:
            return 0
        return (portfolio_value - initial_cash) / initial_cash

    def _build_cost_statistics(
        self, payload: BacktestReportBuildInput
    ) -> dict[str, float]:
        """Build commission/slippage summary block."""
        portfolio_manager = payload.portfolio_manager
        total_cost = (
            portfolio_manager.total_commission + portfolio_manager.total_slippage
        )
        return {
            "total_commission": portfolio_manager.total_commission,
            "total_slippage": portfolio_manager.total_slippage,
            "total_cost": total_cost,
            "cost_ratio": (
                total_cost / payload.config.initial_cash
                if payload.config.initial_cash > 0
                else 0
            ),
        }

    def _build_excess_return_metrics(
        self, payload: BacktestReportBuildInput
    ) -> dict[str, Any]:
        """Build with-cost and without-cost excess return blocks."""
        metrics_without_cost = (
            payload.portfolio_manager.get_performance_metrics_without_cost()
        )
        return {
            "excess_return_without_cost": {
                "mean": metrics_without_cost.get("mean", 0),
                "std": metrics_without_cost.get("std", 0),
                "annualized_return": metrics_without_cost.get(
                    "annualized_return", 0
                ),
                "information_ratio": metrics_without_cost.get(
                    "information_ratio", 0
                ),
                "max_drawdown": metrics_without_cost.get("max_drawdown", 0),
            },
            "excess_return_with_cost": {
                "mean": self._estimate_mean_return(payload.performance_metrics),
                "std": payload.performance_metrics.get("volatility", 0),
                "annualized_return": payload.performance_metrics.get(
                    "annualized_return", 0
                ),
                "information_ratio": payload.performance_metrics.get(
                    "sharpe_ratio", 0
                ),
                "max_drawdown": payload.performance_metrics.get("max_drawdown", 0),
            },
        }

    def _estimate_mean_return(self, performance_metrics: dict[str, float]) -> float:
        """Estimate daily mean excess return from volatility for legacy schema."""
        volatility = performance_metrics.get("volatility", 0)
        if volatility <= 0:
            return 0
        return volatility / np.sqrt(252)

    def _calculate_additional_metrics(
        self, portfolio_manager: PortfolioManager
    ) -> dict[str, Any]:
        """Build optional analytics blocks while keeping failures non-fatal."""
        try:
            if not portfolio_manager.portfolio_history:
                return {}
            additional_metrics = self._calculate_periodic_metrics(portfolio_manager)
            additional_metrics.update(
                self._calculate_trade_distribution_metrics(portfolio_manager)
            )
            return additional_metrics
        except Exception as exc:
            logger.error(f"计算额外指标失败: {exc}")
            return {}

    def _calculate_periodic_metrics(
        self, portfolio_manager: PortfolioManager
    ) -> dict[str, Any]:
        """Calculate monthly/yearly return breakdowns."""
        portfolio_values = pd.Series(
            [
                snapshot["portfolio_value"]
                for snapshot in portfolio_manager.portfolio_history
            ],
            index=[snapshot["date"] for snapshot in portfolio_manager.portfolio_history],
        ).sort_index()

        additional_metrics: dict[str, Any] = {}
        monthly_values = portfolio_values.resample("ME").last()
        monthly_returns = monthly_values.pct_change().dropna()
        if not monthly_returns.empty:
            additional_metrics.update(
                {
                    "monthly_return_mean": float(monthly_returns.mean()),
                    "monthly_return_std": float(monthly_returns.std()),
                    "best_month": float(monthly_returns.max()),
                    "worst_month": float(monthly_returns.min()),
                    "positive_months": int((monthly_returns > 0).sum()),
                    "negative_months": int((monthly_returns < 0).sum()),
                    "monthly_returns_detail": [
                        {
                            "month": period.strftime("%Y-%m"),
                            "return": float(period_return),
                        }
                        for period, period_return in monthly_returns.items()
                    ],
                }
            )

        yearly_values = portfolio_values.resample("Y").last()
        yearly_returns = yearly_values.pct_change().dropna()
        if not yearly_returns.empty:
            additional_metrics["yearly_returns_detail"] = [
                {"year": period.year, "return": float(period_return)}
                for period, period_return in yearly_returns.items()
            ]
        return additional_metrics

    def _calculate_trade_distribution_metrics(
        self, portfolio_manager: PortfolioManager
    ) -> dict[str, Any]:
        """Calculate trade-level distribution and per-stock performance blocks."""
        trades = getattr(portfolio_manager, "trades", [])
        if not trades:
            return {}

        stock_performance = self._build_stock_performance(trades)
        metrics: dict[str, Any] = {
            "stock_performance_detail": list(stock_performance.values()),
            "stocks_traded": len(stock_performance),
        }
        if stock_performance:
            stock_items = list(stock_performance.values())
            metrics["best_performing_stock"] = max(
                stock_items, key=lambda item: item["total_pnl"]
            )
            metrics["worst_performing_stock"] = min(
                stock_items, key=lambda item: item["total_pnl"]
            )

        pnl_series = pd.Series(
            [float(self._get_trade_attr(trade, "pnl", 0.0) or 0.0) for trade in trades]
        )
        if not pnl_series.empty:
            metrics.update(
                {
                    "trade_pnl_mean": float(pnl_series.mean()),
                    "trade_pnl_median": float(pnl_series.median()),
                    "trade_pnl_std": float(pnl_series.std()),
                }
            )
        return metrics

    def _build_stock_performance(
        self, trades: list[Any]
    ) -> dict[str, dict[str, Any]]:
        """Aggregate trade performance by stock code."""
        stock_performance: dict[str, dict[str, Any]] = {}
        for trade in trades:
            stock_code = self._get_trade_attr(trade, "stock_code")
            action = self._get_trade_attr(trade, "action")
            pnl = float(self._get_trade_attr(trade, "pnl", 0.0) or 0.0)
            stock_stats = stock_performance.setdefault(
                stock_code,
                {"stock_code": stock_code, "total_pnl": 0.0, "trade_count": 0},
            )
            stock_stats["trade_count"] += 1
            if action == "SELL":
                stock_stats["total_pnl"] += pnl

        for stats in stock_performance.values():
            trade_count = max(stats["trade_count"], 1)
            stats["avg_pnl_per_trade"] = float(stats["total_pnl"]) / trade_count
        return stock_performance

    def _get_trade_attr(
        self, trade: Any, field_name: str, default: Any = None
    ) -> Any:
        """Read trade fields from either dict payloads or trade objects."""
        if isinstance(trade, dict):
            return trade.get(field_name, default)
        return getattr(trade, field_name, default)
