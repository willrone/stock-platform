"""
图表数据生成器 - 为回测结果生成各种图表所需的数据
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorSeverity, TaskError


class ChartDataGenerator:
    """图表数据生成器"""

    async def generate_chart_data(
        self, backtest_result: Dict[str, Any], chart_type: str
    ) -> Dict[str, Any]:
        """生成指定类型的图表数据"""
        try:
            if chart_type == "equity_curve":
                return await self._generate_equity_curve_data(backtest_result)
            elif chart_type == "drawdown_curve":
                return await self._generate_drawdown_curve_data(backtest_result)
            elif chart_type == "monthly_heatmap":
                return await self._generate_monthly_heatmap_data(backtest_result)
            elif chart_type == "trade_distribution":
                return await self._generate_trade_distribution_data(backtest_result)
            elif chart_type == "position_weights":
                return await self._generate_position_weights_data(backtest_result)
            elif chart_type == "risk_metrics":
                return await self._generate_risk_metrics_data(backtest_result)
            else:
                raise ValueError(f"不支持的图表类型: {chart_type}")

        except Exception as e:
            logger.error(f"生成图表数据失败: {chart_type}, {e}", exc_info=True)
            raise TaskError(
                message=f"生成图表数据失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    async def _generate_equity_curve_data(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成收益曲线图表数据"""
        portfolio_history = backtest_result.get("portfolio_history", [])
        if not portfolio_history:
            return {"equity_curve": [], "benchmark_curve": []}

        logger.info(f"生成收益曲线数据: portfolio_history长度={len(portfolio_history)}")

        # 转换为DataFrame
        df = pd.DataFrame(portfolio_history)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            # 按日期排序，确保数据顺序正确
            df = df.sort_values("date")
            logger.info(
                f"收益曲线数据排序后: 日期范围={df['date'].min()} 至 {df['date'].max()}, 数据量={len(df)}"
            )
        elif "snapshot_date" in df.columns:
            df["date"] = pd.to_datetime(df["snapshot_date"])
            df = df.sort_values("date")
            logger.info(
                f"收益曲线数据排序后: 日期范围={df['date'].min()} 至 {df['date'].max()}, 数据量={len(df)}"
            )

        # 计算累积收益率
        initial_value = df["portfolio_value"].iloc[0]
        df["cumulative_return"] = (df["portfolio_value"] / initial_value - 1) * 100

        # 构建收益曲线数据
        equity_curve = []
        for _, row in df.iterrows():
            equity_curve.append(
                {
                    "date": row["date"].isoformat() if pd.notna(row["date"]) else "",
                    "portfolio_value": float(row["portfolio_value"]),
                    "cumulative_return": float(row["cumulative_return"]),
                    "cash": float(row.get("cash", 0)),
                }
            )

        # TODO: 添加基准数据（如沪深300指数）
        benchmark_curve = []

        return {
            "equity_curve": equity_curve,
            "benchmark_curve": benchmark_curve,
            "initial_value": float(initial_value),
            "final_value": float(df["portfolio_value"].iloc[-1]),
            "total_return": float(df["cumulative_return"].iloc[-1]),
        }

    async def _generate_drawdown_curve_data(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成回撤曲线图表数据"""
        portfolio_history = backtest_result.get("portfolio_history", [])
        if not portfolio_history:
            return {"drawdown_curve": []}

        # 转换为DataFrame
        df = pd.DataFrame(portfolio_history)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # 计算回撤
        values = df["portfolio_value"]
        peak = values.expanding().max()
        drawdown = (values - peak) / peak * 100  # 转换为百分比

        # 构建回撤曲线数据
        drawdown_curve = []
        for i, (_, row) in enumerate(df.iterrows()):
            drawdown_curve.append(
                {
                    "date": row["date"].isoformat() if pd.notna(row["date"]) else "",
                    "drawdown": float(drawdown.iloc[i]),
                    "peak_value": float(peak.iloc[i]),
                }
            )

        # 找到最大回撤点
        max_dd_idx = drawdown.idxmin()
        max_drawdown_info = {
            "max_drawdown": float(drawdown.min()),
            "max_drawdown_date": df.loc[max_dd_idx, "date"].isoformat()
            if pd.notna(df.loc[max_dd_idx, "date"])
            else "",
            "max_drawdown_value": float(values.iloc[max_dd_idx]),
        }

        return {
            "drawdown_curve": drawdown_curve,
            "max_drawdown_info": max_drawdown_info,
        }

    async def _generate_monthly_heatmap_data(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成月度收益热力图数据"""
        portfolio_history = backtest_result.get("portfolio_history", [])
        if not portfolio_history:
            return {"monthly_data": [], "years": [], "months": []}

        # 转换为DataFrame
        df = pd.DataFrame(portfolio_history)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        # 按月重采样
        monthly_values = df["portfolio_value"].resample("ME").last()
        monthly_returns = monthly_values.pct_change().dropna() * 100  # 转换为百分比

        # 构建热力图数据
        monthly_data = []
        years = set()
        months = set()

        for date, return_val in monthly_returns.items():
            year = date.year
            month = date.month
            years.add(year)
            months.add(month)

            monthly_data.append(
                {
                    "year": year,
                    "month": month,
                    "return": float(return_val),
                    "date": date.strftime("%Y-%m"),
                }
            )

        return {
            "monthly_data": monthly_data,
            "years": sorted(list(years)),
            "months": sorted(list(months)),
        }

    async def _generate_trade_distribution_data(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成交易分布图表数据"""
        trade_history = backtest_result.get("trade_history", [])
        if not trade_history:
            return {"profit_distribution": [], "trade_stats": {}}

        # 分析交易盈亏分布
        profits = []
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0

        for trade in trade_history:
            if trade.get("action") == "SELL":  # 只统计卖出交易的盈亏
                pnl = trade.get("pnl", 0)
                profits.append(pnl)

                if pnl > 0:
                    winning_trades += 1
                    total_profit += pnl
                elif pnl < 0:
                    losing_trades += 1
                    total_loss += abs(pnl)

        # 创建盈亏分布直方图数据
        if profits:
            profit_array = np.array(profits)
            hist, bin_edges = np.histogram(profit_array, bins=20)

            profit_distribution = []
            for i in range(len(hist)):
                profit_distribution.append(
                    {
                        "bin_start": float(bin_edges[i]),
                        "bin_end": float(bin_edges[i + 1]),
                        "count": int(hist[i]),
                        "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                    }
                )
        else:
            profit_distribution = []

        # 交易统计
        trade_stats = {
            "total_trades": len(
                [t for t in trade_history if t.get("action") == "SELL"]
            ),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / (winning_trades + losing_trades)
            if (winning_trades + losing_trades) > 0
            else 0,
            "avg_profit": total_profit / winning_trades if winning_trades > 0 else 0,
            "avg_loss": total_loss / losing_trades if losing_trades > 0 else 0,
            "profit_factor": total_profit / total_loss
            if total_loss > 0
            else 0.0,
        }

        return {"profit_distribution": profit_distribution, "trade_stats": trade_stats}

    async def _generate_position_weights_data(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成持仓权重图表数据"""
        trade_history = backtest_result.get("trade_history", [])
        if not trade_history:
            return {"position_weights": [], "stock_performance": []}

        # 按股票统计持仓信息
        stock_stats = {}

        for trade in trade_history:
            stock_code = trade.get("stock_code", "")
            if stock_code not in stock_stats:
                stock_stats[stock_code] = {
                    "stock_code": stock_code,
                    "total_value": 0,
                    "total_pnl": 0,
                    "trade_count": 0,
                }

            # 累计交易金额（用于计算权重）
            trade_value = trade.get("quantity", 0) * trade.get("price", 0)
            stock_stats[stock_code]["total_value"] += trade_value
            stock_stats[stock_code]["total_pnl"] += trade.get("pnl", 0)
            stock_stats[stock_code]["trade_count"] += 1

        # 计算总交易金额
        total_value = sum(stats["total_value"] for stats in stock_stats.values())

        # 构建持仓权重数据
        position_weights = []
        stock_performance = []

        for stock_code, stats in stock_stats.items():
            weight = stats["total_value"] / total_value if total_value > 0 else 0

            position_weights.append(
                {
                    "stock_code": stock_code,
                    "weight": float(weight * 100),  # 转换为百分比
                    "total_value": float(stats["total_value"]),
                }
            )

            stock_performance.append(
                {
                    "stock_code": stock_code,
                    "total_pnl": float(stats["total_pnl"]),
                    "trade_count": stats["trade_count"],
                    "avg_pnl_per_trade": float(
                        stats["total_pnl"] / stats["trade_count"]
                    )
                    if stats["trade_count"] > 0
                    else 0,
                }
            )

        # 按权重排序
        position_weights.sort(key=lambda x: x["weight"], reverse=True)
        stock_performance.sort(key=lambda x: x["total_pnl"], reverse=True)

        return {
            "position_weights": position_weights,
            "stock_performance": stock_performance,
        }

    async def _generate_risk_metrics_data(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成风险指标图表数据"""
        portfolio_history = backtest_result.get("portfolio_history", [])
        if not portfolio_history:
            return {"risk_metrics": {}, "rolling_metrics": []}

        # 转换为DataFrame
        df = pd.DataFrame(portfolio_history)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        # 计算收益率
        returns = df["portfolio_value"].pct_change().dropna()

        # 基础风险指标
        risk_metrics = {
            "volatility": float(returns.std() * np.sqrt(252) * 100),  # 年化波动率（百分比）
            "sharpe_ratio": float(backtest_result.get("sharpe_ratio", 0)),
            "max_drawdown": float(
                backtest_result.get("max_drawdown", 0) * 100
            ),  # 转换为百分比
            "var_95": float(returns.quantile(0.05) * 100),  # 95% VaR（百分比）
            "skewness": float(returns.skew()),
            "kurtosis": float(returns.kurtosis()),
        }

        # 滚动指标（30天窗口）
        rolling_window = min(30, len(returns) // 4)  # 确保有足够的数据点
        if rolling_window >= 5:
            rolling_volatility = (
                returns.rolling(window=rolling_window).std() * np.sqrt(252) * 100
            )
            rolling_sharpe = (
                returns.rolling(window=rolling_window).mean()
                / returns.rolling(window=rolling_window).std()
                * np.sqrt(252)
            )

            rolling_metrics = []
            for date, vol in rolling_volatility.dropna().items():
                sharpe = (
                    rolling_sharpe.loc[date]
                    if pd.notna(rolling_sharpe.loc[date])
                    else 0
                )
                rolling_metrics.append(
                    {
                        "date": date.isoformat(),
                        "volatility": float(vol),
                        "sharpe_ratio": float(sharpe),
                    }
                )
        else:
            rolling_metrics = []

        return {"risk_metrics": risk_metrics, "rolling_metrics": rolling_metrics}
