"""Build normalized backtest result payloads for executor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import settings
from app.services.data.simple_data_service import SimpleDataService
from app.services.data.stock_data_loader import StockDataLoader

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
        report["official_portfolio_analysis"] = self._build_official_portfolio_analysis(
            payload
        )
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
            "actionable_signal_count": signal_stats.get("actionable_signal_count", 0),
            "executed_signal_count": signal_stats.get("executed_signal_count", 0),
            "top_rejection_reasons": signal_stats.get("top_rejection_reasons", []),
        }

    def _build_basic_report(self, payload: BacktestReportBuildInput) -> dict[str, Any]:
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
            return float(portfolio_history[-1]["portfolio_value"])
        return float(portfolio_manager.get_portfolio_value({}))

    def _build_metrics(self, performance_metrics: dict[str, float]) -> dict[str, float]:
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
            "initial_cash": getattr(payload.config, "initial_cash", 0.0),
            "commission_rate": getattr(payload.config, "commission_rate", 0.0),
            "slippage_rate": getattr(payload.config, "slippage_rate", 0.0),
            "open_cost": getattr(payload.config, "open_cost", 0.0),
            "close_cost": getattr(payload.config, "close_cost", 0.0),
            "min_cost": getattr(payload.config, "min_cost", 0.0),
            "max_position_size": getattr(payload.config, "max_position_size", 0.0),
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
            "timestamp": (
                timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp
            ),
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
        benchmark = self._extract_benchmark(payload.strategy_config)
        official_metrics = self._calculate_official_excess_return_metrics(
            payload, benchmark
        )
        if official_metrics is not None:
            return official_metrics

        metrics_without_cost = (
            payload.portfolio_manager.get_performance_metrics_without_cost()
        )
        return {
            "excess_return_without_cost": {
                "mean": metrics_without_cost.get("mean", 0),
                "std": metrics_without_cost.get("std", 0),
                "annualized_return": metrics_without_cost.get("annualized_return", 0),
                "information_ratio": metrics_without_cost.get("information_ratio", 0),
                "max_drawdown": metrics_without_cost.get("max_drawdown", 0),
            },
            "excess_return_with_cost": {
                "mean": payload.performance_metrics.get("excess_return_mean_with_cost"),
                "std": payload.performance_metrics.get("excess_return_std_with_cost"),
                "annualized_return": payload.performance_metrics.get(
                    "annualized_return", 0
                ),
                "information_ratio": payload.performance_metrics.get(
                    "information_ratio_with_cost"
                ),
                "max_drawdown": payload.performance_metrics.get("max_drawdown", 0),
            },
        }

    def _build_official_portfolio_analysis(
        self, payload: BacktestReportBuildInput
    ) -> dict[str, Any]:
        """Build a Qlib-style benchmark-relative portfolio analysis block."""
        benchmark = self._extract_benchmark(payload.strategy_config)
        excess_metrics = self._build_excess_return_metrics(payload)
        return {
            "benchmark": benchmark,
            "excess_return_without_cost": excess_metrics["excess_return_without_cost"],
            "excess_return_with_cost": excess_metrics["excess_return_with_cost"],
        }

    def _extract_benchmark(self, strategy_config: Optional[dict[str, Any]]) -> Any:
        """Extract benchmark config if present."""
        if not isinstance(strategy_config, dict):
            return None
        return strategy_config.get("benchmark")

    def _calculate_official_excess_return_metrics(
        self,
        payload: BacktestReportBuildInput,
        benchmark: Any,
    ) -> Optional[dict[str, Any]]:
        """Calculate Qlib-style benchmark-relative excess return metrics."""
        if benchmark is None:
            return None

        returns_with_cost, returns_without_cost = self._build_portfolio_return_series(
            payload.portfolio_manager
        )
        if returns_with_cost is None or returns_without_cost is None:
            return None

        benchmark_returns = self._load_benchmark_return_series(
            benchmark,
            payload.start_date,
            payload.end_date,
        )
        if benchmark_returns is None or benchmark_returns.empty:
            logger.warning(f"无法为 official_portfolio_analysis 加载基准收益序列: {benchmark}")
            return None

        common_dates = (
            returns_with_cost.index.intersection(returns_without_cost.index)
            .intersection(benchmark_returns.index)
            .sort_values()
        )
        if len(common_dates) == 0:
            logger.warning(
                f"组合收益与基准收益无共同日期，无法计算 official_portfolio_analysis: {benchmark}"
            )
            return None

        excess_without_cost = (
            returns_without_cost.loc[common_dates] - benchmark_returns.loc[common_dates]
        )
        excess_with_cost = (
            returns_with_cost.loc[common_dates] - benchmark_returns.loc[common_dates]
        )

        return {
            "excess_return_without_cost": self._run_qlib_risk_analysis(
                excess_without_cost
            ),
            "excess_return_with_cost": self._run_qlib_risk_analysis(excess_with_cost),
        }

    def _build_portfolio_return_series(
        self, portfolio_manager: PortfolioManager
    ) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Build daily return series with-cost and without-cost from portfolio history."""
        portfolio_history = getattr(portfolio_manager, "portfolio_history", None)
        if not portfolio_history:
            return None, None

        frame = pd.DataFrame(portfolio_history)
        if (
            frame.empty
            or "date" not in frame.columns
            or "portfolio_value" not in frame.columns
        ):
            return None, None

        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame.sort_values("date").drop_duplicates(subset="date", keep="last")
        frame = frame.set_index("date")

        without_cost_column = (
            "portfolio_value_without_cost"
            if "portfolio_value_without_cost" in frame.columns
            else "portfolio_value"
        )
        returns_with_cost = frame["portfolio_value"].astype(float).pct_change().dropna()
        returns_without_cost = (
            frame[without_cost_column].astype(float).pct_change().dropna()
        )

        if returns_with_cost.empty or returns_without_cost.empty:
            return None, None

        return returns_with_cost, returns_without_cost

    def _load_benchmark_return_series(
        self,
        benchmark: Any,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.Series]:
        """Load benchmark daily return series supporting Qlib-style codes."""
        if isinstance(benchmark, pd.Series):
            series = benchmark.copy()
            series.index = pd.to_datetime(series.index).normalize()
            return series.sort_index().dropna()

        if isinstance(benchmark, (list, tuple)):
            benchmark_frames: list[pd.Series] = []
            for code in benchmark:
                single = self._load_benchmark_return_series(code, start_date, end_date)
                if single is not None and not single.empty:
                    benchmark_frames.append(single)
            if not benchmark_frames:
                return None
            combined = pd.concat(benchmark_frames, axis=1).dropna(how="all")
            if combined.empty:
                return None
            return combined.mean(axis=1).dropna()

        close_series = self._load_single_benchmark_close_series(
            str(benchmark), start_date, end_date
        )
        if close_series is None or close_series.empty:
            return None
        returns = close_series.pct_change().dropna()
        return returns.sort_index()

    def _load_single_benchmark_close_series(
        self,
        benchmark: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.Series]:
        """Load one benchmark close-price series from local parquet/JSON data."""
        loader = StockDataLoader(data_root=settings.DATA_ROOT_PATH)
        data_service = SimpleDataService(data_path=settings.DATA_ROOT_PATH)

        for candidate in self._benchmark_code_candidates(benchmark):
            try:
                benchmark_df = loader.load_stock_data(
                    candidate,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not benchmark_df.empty and "close" in benchmark_df.columns:
                    series = benchmark_df["close"].astype(float)
                    series.index = pd.to_datetime(series.index).normalize()
                    return series.sort_index().dropna()
            except Exception as exc:
                logger.debug(
                    f"使用 StockDataLoader 加载 benchmark 失败，尝试下一个候选: {candidate}, {exc}"
                )

            try:
                local_rows = data_service.load_from_local(
                    candidate, start_date, end_date
                )
                if local_rows:
                    benchmark_df = pd.DataFrame(local_rows)
                    if (
                        "date" in benchmark_df.columns
                        and "close" in benchmark_df.columns
                    ):
                        benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
                        benchmark_df = benchmark_df.sort_values("date")
                        series = benchmark_df.set_index("date")["close"].astype(float)
                        series.index = pd.to_datetime(series.index).normalize()
                        return series.sort_index().dropna()
            except Exception as exc:
                logger.debug(
                    f"使用 SimpleDataService 加载 benchmark 失败，尝试下一个候选: {candidate}, {exc}"
                )

        return None

    def _benchmark_code_candidates(self, benchmark: str) -> list[str]:
        """Generate likely local-code candidates from Qlib/API benchmark formats."""
        candidates = [benchmark]
        if (
            len(benchmark) == 8
            and benchmark[:2] in {"SH", "SZ"}
            and benchmark[2:].isdigit()
        ):
            candidates.append(f"{benchmark[2:]}.{benchmark[:2]}")
        elif len(benchmark) == 9 and benchmark[6] == "." and benchmark[:6].isdigit():
            exchange = benchmark[7:]
            if exchange in {"SH", "SZ"}:
                candidates.append(f"{exchange}{benchmark[:6]}")
        return list(dict.fromkeys(candidates))

    def _run_qlib_risk_analysis(self, excess_returns: pd.Series) -> dict[str, Any]:
        """Run Qlib official risk_analysis and flatten the result to API schema."""
        if excess_returns.empty:
            return {
                "mean": None,
                "std": None,
                "annualized_return": None,
                "information_ratio": None,
                "max_drawdown": None,
            }

        # isort: off
        from qlib.contrib.evaluate import risk_analysis

        # isort: on

        analysis_df = risk_analysis(excess_returns, freq="day")
        return {
            "mean": self._to_optional_float(analysis_df.loc["mean", "risk"]),
            "std": self._to_optional_float(analysis_df.loc["std", "risk"]),
            "annualized_return": self._to_optional_float(
                analysis_df.loc["annualized_return", "risk"]
            ),
            "information_ratio": self._to_optional_float(
                analysis_df.loc["information_ratio", "risk"]
            ),
            "max_drawdown": self._to_optional_float(
                analysis_df.loc["max_drawdown", "risk"]
            ),
        }

    def _to_optional_float(self, value: Any) -> Optional[float]:
        """Convert scalar values to optional floats while normalizing NaN/inf."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _estimate_mean_return(self, performance_metrics: dict[str, float]) -> float:
        """Estimate daily mean excess return from volatility for legacy schema."""
        volatility = performance_metrics.get("volatility", 0)
        if volatility <= 0:
            return 0.0
        return float(volatility) / float(np.sqrt(252))

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
            index=[
                snapshot["date"] for snapshot in portfolio_manager.portfolio_history
            ],
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

    def _build_stock_performance(self, trades: list[Any]) -> dict[str, dict[str, Any]]:
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

    def _get_trade_attr(self, trade: Any, field_name: str, default: Any = None) -> Any:
        """Read trade fields from either dict payloads or trade objects."""
        if isinstance(trade, dict):
            return trade.get(field_name, default)
        return getattr(trade, field_name, default)
