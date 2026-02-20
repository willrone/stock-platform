"""
回测数据适配器 - 将现有回测结果转换为增强格式
用于支持完整的可视化功能
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorSeverity, TaskError

from ..analysis.position_analysis import PositionAnalyzer
from ..models.analysis_models import (
    DrawdownAnalysis,
    EnhancedBacktestResult,
    EnhancedPositionAnalysis,
    ExtendedRiskMetrics,
    MonthlyReturnsAnalysis,
)


class BacktestDataAdapter:
    """回测数据适配器 - 将现有回测结果转换为可视化所需格式"""

    def __init__(self):
        self.risk_free_rate = 0.03  # 无风险利率，用于计算夏普比率等指标

    async def adapt_backtest_result(
        self, task_result: Dict[str, Any]
    ) -> EnhancedBacktestResult:
        """适配现有回测结果为增强格式"""
        try:
            logger.info("开始适配回测结果数据")

            # 1. 提取现有数据
            base_data = self._extract_base_data(task_result)

            # 2. 计算扩展风险指标
            extended_risk_metrics = await self._calculate_extended_risk_metrics(
                task_result.get("portfolio_history", []),
                task_result.get("initial_cash", 100000),
            )

            # 3. 分析月度收益
            monthly_returns = await self._analyze_monthly_returns(
                task_result.get("portfolio_history", [])
            )

            # 4. 分析持仓表现（使用完整的PositionAnalyzer）
            position_analysis = await self._analyze_positions(
                task_result.get("trade_history", []),
                task_result.get("portfolio_history", []),
            )

            # 5. 计算回撤详细分析
            drawdown_analysis = await self._analyze_drawdowns(
                task_result.get("portfolio_history", [])
            )

            # 6. 计算基准对比
            from ..analysis.enhanced_metrics_calculator import EnhancedMetricsCalculator

            metrics_calculator = EnhancedMetricsCalculator()
            benchmark_data = await metrics_calculator.calculate_benchmark_comparison(
                task_result.get("portfolio_history", []), benchmark_symbol="000300.SH"
            )

            # 7. 计算滚动指标时间序列
            rolling_metrics = self._calculate_rolling_metrics_series(
                task_result.get("portfolio_history", [])
            )
            logger.info(
                f"滚动指标计算结果: type={type(rolling_metrics)}, "
                f"keys={list(rolling_metrics.keys()) if isinstance(rolling_metrics, dict) else 'N/A'}, "
                f"dates_len={len(rolling_metrics.get('dates', []))} if dict else 'N/A'"
            )

            # 8. 构建增强结果
            enhanced_result = EnhancedBacktestResult(
                # 现有字段直接映射
                strategy_name=base_data["strategy_name"],
                stock_codes=base_data["stock_codes"],
                start_date=base_data["start_date"],
                end_date=base_data["end_date"],
                initial_cash=base_data["initial_cash"],
                final_value=base_data["final_value"],
                total_return=base_data["total_return"],
                annualized_return=base_data["annualized_return"],
                volatility=base_data["volatility"],
                sharpe_ratio=base_data["sharpe_ratio"],
                max_drawdown=base_data["max_drawdown"],
                total_trades=base_data["total_trades"],
                win_rate=base_data["win_rate"],
                profit_factor=base_data["profit_factor"],
                winning_trades=base_data["winning_trades"],
                losing_trades=base_data["losing_trades"],
                backtest_config=base_data["backtest_config"],
                trade_history=base_data["trade_history"],
                portfolio_history=base_data["portfolio_history"],
                # 新增字段
                extended_risk_metrics=extended_risk_metrics,
                monthly_returns=monthly_returns,
                position_analysis=position_analysis,
                drawdown_analysis=drawdown_analysis,
                benchmark_data=benchmark_data if benchmark_data else None,
                rolling_metrics=rolling_metrics,
            )

            logger.info("回测结果数据适配完成")
            return enhanced_result

        except Exception as e:
            logger.error(f"适配回测结果失败: {e}", exc_info=True)
            raise TaskError(
                message=f"适配回测结果失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                original_exception=e,
            )

    def _extract_base_data(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取现有的基础数据"""
        return {
            "strategy_name": task_result.get("strategy_name", ""),
            "stock_codes": task_result.get("stock_codes", []),
            "start_date": task_result.get("start_date", ""),
            "end_date": task_result.get("end_date", ""),
            "initial_cash": task_result.get("initial_cash", 100000),
            "final_value": task_result.get("final_value", 100000),
            "total_return": task_result.get("total_return", 0),
            "annualized_return": task_result.get("annualized_return", 0),
            "volatility": task_result.get("volatility", 0),
            "sharpe_ratio": task_result.get("sharpe_ratio", 0),
            "max_drawdown": task_result.get("max_drawdown", 0),
            "total_trades": task_result.get("total_trades", 0),
            "win_rate": task_result.get("win_rate", 0),
            "profit_factor": task_result.get("profit_factor", 0),
            "winning_trades": task_result.get("winning_trades", 0),
            "losing_trades": task_result.get("losing_trades", 0),
            "backtest_config": task_result.get("backtest_config", {}),
            "trade_history": task_result.get("trade_history", []),
            "portfolio_history": task_result.get("portfolio_history", []),
        }

    async def _calculate_extended_risk_metrics(
        self, portfolio_history: List[Dict], initial_cash: float
    ) -> Optional[ExtendedRiskMetrics]:
        """基于现有组合历史计算扩展风险指标"""

        if not portfolio_history:
            logger.warning("组合历史数据为空，无法计算扩展风险指标")
            return None

        try:
            # 转换为pandas DataFrame
            df = pd.DataFrame(portfolio_history)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

            # 计算收益率序列
            returns = df["portfolio_value"].pct_change().dropna()

            if len(returns) == 0:
                logger.warning("收益率序列为空，无法计算扩展风险指标")
                return None

            # 计算扩展指标
            extended_metrics = {}

            # 获取基础指标
            volatility = returns.std() * np.sqrt(252)
            if pd.isna(volatility) or np.isnan(volatility):
                volatility = 0.0

            annualized_return = (df["portfolio_value"].iloc[-1] / initial_cash) ** (
                252 / len(returns)
            ) - 1
            if pd.isna(annualized_return) or np.isnan(annualized_return):
                annualized_return = 0.0

            # Sortino比率
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(252)
                # 确保下行偏差不是NaN
                if pd.isna(downside_deviation) or np.isnan(downside_deviation):
                    downside_deviation = 0.0
                extended_metrics["sortino_ratio"] = (
                    (annualized_return - self.risk_free_rate) / downside_deviation
                    if downside_deviation > 0
                    else 0
                )
                extended_metrics["downside_deviation"] = downside_deviation
            else:
                extended_metrics["sortino_ratio"] = 0
                extended_metrics["downside_deviation"] = 0

            # Calmar比率
            max_drawdown = self._calculate_max_drawdown(df["portfolio_value"])
            if max_drawdown < 0:
                extended_metrics["calmar_ratio"] = (
                    annualized_return - self.risk_free_rate
                ) / abs(max_drawdown)
            else:
                extended_metrics["calmar_ratio"] = 0

            # VaR 95%
            var_95 = returns.quantile(0.05)
            if pd.isna(var_95) or np.isnan(var_95):
                var_95 = 0.0
            extended_metrics["var_95"] = var_95

            # VaR 99%
            var_99 = returns.quantile(0.01)
            if pd.isna(var_99) or np.isnan(var_99):
                var_99 = 0.0
            extended_metrics["var_99"] = var_99

            # CVaR 95% (Expected Shortfall)
            cvar_95_series = returns[returns <= var_95]
            cvar_95 = float(cvar_95_series.mean()) if len(cvar_95_series) > 0 else float(var_95)
            if pd.isna(cvar_95) or np.isnan(cvar_95):
                cvar_95 = 0.0
            extended_metrics["cvar_95"] = cvar_95

            # CVaR 99%
            cvar_99_series = returns[returns <= var_99]
            cvar_99 = float(cvar_99_series.mean()) if len(cvar_99_series) > 0 else float(var_99)
            if pd.isna(cvar_99) or np.isnan(cvar_99):
                cvar_99 = 0.0
            extended_metrics["cvar_99"] = cvar_99

            # 最大回撤持续时间
            extended_metrics[
                "max_drawdown_duration"
            ] = self._calculate_max_drawdown_duration(df["portfolio_value"])

            # 夏普比率（重新计算，使用无风险利率）
            sharpe_ratio = (
                (annualized_return - self.risk_free_rate) / volatility
                if volatility > 0
                else 0
            )

            return ExtendedRiskMetrics(
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                sortino_ratio=extended_metrics["sortino_ratio"],
                calmar_ratio=extended_metrics["calmar_ratio"],
                max_drawdown_duration=extended_metrics["max_drawdown_duration"],
                var_95=extended_metrics["var_95"],
                var_99=extended_metrics.get("var_99", 0.0),
                cvar_95=extended_metrics.get("cvar_95", 0.0),
                cvar_99=extended_metrics.get("cvar_99", 0.0),
                downside_deviation=extended_metrics["downside_deviation"],
            )

        except Exception as e:
            logger.error(f"计算扩展风险指标失败: {e}", exc_info=True)
            return None

    async def _analyze_monthly_returns(
        self, portfolio_history: List[Dict]
    ) -> Optional[List[MonthlyReturnsAnalysis]]:
        """分析月度收益"""

        if not portfolio_history:
            logger.warning("组合历史数据为空，无法分析月度收益")
            return None

        try:
            df = pd.DataFrame(portfolio_history)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

            # 按月重采样
            monthly_values = df["portfolio_value"].resample("ME").last()
            monthly_returns = monthly_values.pct_change().dropna()

            # 获取初始价值用于计算累积收益
            initial_value = df["portfolio_value"].iloc[0]

            result = []

            for date, monthly_return in monthly_returns.items():
                # 计算从初始时点到当前月末的累积收益
                current_value = monthly_values[date]
                cumulative_return = (current_value - initial_value) / initial_value

                analysis = MonthlyReturnsAnalysis(
                    year=date.year,
                    month=date.month,
                    date=date.strftime("%Y-%m"),
                    monthly_return=float(monthly_return),
                    cumulative_return=float(cumulative_return),
                )
                result.append(analysis)

            logger.info(f"月度收益分析完成，共 {len(result)} 个月")
            return result

        except Exception as e:
            logger.error(f"分析月度收益失败: {e}", exc_info=True)
            return None

    async def _analyze_positions(
        self, trade_history: List[Dict], portfolio_history: List[Dict]
    ) -> Optional[EnhancedPositionAnalysis]:
        """分析持仓表现（使用完整的PositionAnalyzer服务）"""

        logger.info(
            f"开始分析持仓表现: trade_history长度={len(trade_history) if trade_history else 0}, portfolio_history长度={len(portfolio_history) if portfolio_history else 0}"
        )

        if not trade_history:
            logger.warning("交易历史数据为空，无法分析持仓表现")
            return None

        try:
            # 使用完整的PositionAnalyzer服务
            analyzer = PositionAnalyzer()
            analysis_result = await analyzer.analyze_position_performance(
                trade_history=trade_history, portfolio_history=portfolio_history or []
            )

            if not analysis_result:
                logger.warning("PositionAnalyzer返回空结果")
                return None

            logger.info(f"PositionAnalyzer返回结果: keys={list(analysis_result.keys())}")

            # 转换stock_performance为兼容格式（包含原有字段）
            stock_performance = analysis_result.get("stock_performance", [])
            logger.info(f"stock_performance 原始数据长度: {len(stock_performance)}")
            compatible_stock_performance = []

            for stock_data in stock_performance:
                # 转换为兼容原有前端格式的数据
                compatible_data = {
                    "stock_code": stock_data.get("stock_code", ""),
                    "stock_name": stock_data.get(
                        "stock_name", stock_data.get("stock_code", "")
                    ),
                    "total_return": stock_data.get("total_return", 0.0),
                    "trade_count": stock_data.get("total_trades", 0),
                    "win_rate": stock_data.get("win_rate", 0.0),
                    "avg_holding_period": stock_data.get("avg_holding_period", 0),
                    "winning_trades": stock_data.get("winning_trades", 0),
                    "losing_trades": stock_data.get("losing_trades", 0),
                    # 添加额外字段供前端使用
                    "avg_return_per_trade": stock_data.get("avg_return_per_trade", 0.0),
                    "return_ratio": stock_data.get("return_ratio", 0.0),
                    "trade_frequency": stock_data.get("trade_frequency", 0.0),
                    "avg_win": stock_data.get("avg_win", 0.0),
                    "avg_loss": stock_data.get("avg_loss", 0.0),
                    "largest_win": stock_data.get("largest_win", 0.0),
                    "largest_loss": stock_data.get("largest_loss", 0.0),
                    "profit_factor": stock_data.get("profit_factor", 0.0),
                    "max_holding_period": stock_data.get("max_holding_period", 0),
                    "min_holding_period": stock_data.get("min_holding_period", 0),
                    "avg_buy_price": stock_data.get("avg_buy_price", 0.0),
                    "avg_sell_price": stock_data.get("avg_sell_price", 0.0),
                    "price_improvement": stock_data.get("price_improvement", 0.0),
                    "total_volume": stock_data.get("total_volume", 0.0),
                    "total_commission": stock_data.get("total_commission", 0.0),
                    "commission_ratio": stock_data.get("commission_ratio", 0.0),
                }
                compatible_stock_performance.append(compatible_data)

            # 构建增强的持仓分析结果
            enhanced_analysis = EnhancedPositionAnalysis(
                stock_performance=compatible_stock_performance,
                position_weights=analysis_result.get("position_weights"),
                trading_patterns=analysis_result.get("trading_patterns"),
                holding_periods=analysis_result.get("holding_periods"),
                concentration_risk=analysis_result.get("concentration_risk"),
            )

            logger.info(f"持仓分析完成，共分析 {len(compatible_stock_performance)} 只股票")
            return enhanced_analysis

        except Exception as e:
            logger.error(f"分析持仓表现失败: {e}", exc_info=True)
            return None

    def _calculate_avg_holding_period(self, trades: List[Dict]) -> int:
        """计算平均持仓期"""
        try:
            # 简化实现：配对买卖交易计算持仓期
            buy_trades = [t for t in trades if t.get("action") == "BUY"]
            sell_trades = [t for t in trades if t.get("action") == "SELL"]

            if not buy_trades or not sell_trades:
                return 0

            # 简单配对最近的买卖交易
            holding_periods = []
            for sell_trade in sell_trades:
                sell_date = pd.to_datetime(sell_trade.get("timestamp", ""))
                # 找到最近的买入交易
                for buy_trade in reversed(buy_trades):
                    buy_date = pd.to_datetime(buy_trade.get("timestamp", ""))
                    if buy_date <= sell_date:
                        holding_period = (sell_date - buy_date).days
                        holding_periods.append(holding_period)
                        break

            return int(np.mean(holding_periods)) if holding_periods else 0

        except Exception as e:
            logger.error(f"计算平均持仓期失败: {e}")
            return 0

    async def _analyze_drawdowns(
        self, portfolio_history: List[Dict]
    ) -> Optional[DrawdownAnalysis]:
        """分析回撤详情"""

        if not portfolio_history:
            logger.warning("组合历史数据为空，无法分析回撤")
            return None

        try:
            df = pd.DataFrame(portfolio_history)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

            values = df["portfolio_value"]

            # 计算回撤序列
            peak = values.expanding().max()
            drawdown = (values - peak) / peak

            # 找到最大回撤期间
            max_dd_idx = drawdown.idxmin()
            max_dd_value = drawdown.min()

            # 找到最大回撤开始和结束时间
            max_dd_start = None
            max_dd_end = None

            # 向前找到峰值点
            for i in range(len(drawdown)):
                if drawdown.index[i] >= max_dd_idx:
                    break
                if drawdown.iloc[i] == 0:  # 新高点
                    max_dd_start = drawdown.index[i]

            # 向后找到恢复点
            for i in range(len(drawdown) - 1, -1, -1):
                if drawdown.index[i] <= max_dd_idx:
                    continue
                if drawdown.iloc[i] >= -0.001:  # 基本恢复
                    max_dd_end = drawdown.index[i]
                    break

            # 构建回撤曲线数据
            drawdown_curve = [
                {"date": date.isoformat(), "drawdown": float(dd)}
                for date, dd in drawdown.items()
            ]

            analysis = DrawdownAnalysis(
                max_drawdown=float(max_dd_value),
                max_drawdown_date=max_dd_idx.isoformat() if max_dd_idx else None,
                max_drawdown_start=max_dd_start.isoformat() if max_dd_start else None,
                max_drawdown_end=max_dd_end.isoformat() if max_dd_end else None,
                max_drawdown_duration=(max_dd_end - max_dd_start).days
                if max_dd_start and max_dd_end
                else 0,
                drawdown_curve=drawdown_curve,
            )

            logger.info(f"回撤分析完成，最大回撤: {max_dd_value:.2%}")
            return analysis

        except Exception as e:
            logger.error(f"分析回撤详情失败: {e}", exc_info=True)
            return None

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """计算最大回撤"""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return drawdown.min()

    def _calculate_max_drawdown_duration(self, values: pd.Series) -> int:
        """计算最大回撤持续时间"""
        try:
            peak = values.expanding().max()
            drawdown = (values - peak) / peak

            # 找到所有回撤期间
            is_drawdown = drawdown < -0.001  # 小于-0.1%认为是回撤
            drawdown_periods = []
            current_period = 0

            for in_dd in is_drawdown:
                if in_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0

            if current_period > 0:
                drawdown_periods.append(current_period)

            return max(drawdown_periods) if drawdown_periods else 0

        except Exception as e:
            logger.error(f"计算最大回撤持续时间失败: {e}")
            return 0

    def _calculate_rolling_metrics_series(
        self, portfolio_history: List[Dict], window: int = 60
    ) -> Dict[str, Any]:
        """计算滚动指标时间序列，用于前端图表展示"""
        if not portfolio_history:
            return {}

        try:
            df = pd.DataFrame(portfolio_history)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

            returns = df["portfolio_value"].pct_change().dropna()

            if len(returns) < window:
                logger.warning(
                    f"数据长度 {len(returns)} 小于滚动窗口 {window}，跳过滚动指标计算"
                )
                return {}

            # 滚动夏普比率
            # 年化收益 = 日均收益 × 252，年化波动率 = 日标准差 × √252
            rolling_annualized_return = returns.rolling(window).mean() * 252
            rolling_std = returns.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = (rolling_annualized_return - self.risk_free_rate) / rolling_std

            # 滚动波动率
            rolling_volatility = rolling_std

            # 滚动最大回撤
            rolling_values = df["portfolio_value"]
            rolling_drawdown_series = []
            dates = []
            for i in range(window, len(rolling_values)):
                window_values = rolling_values.iloc[i - window : i + 1]
                peak = window_values.expanding().max()
                dd = ((window_values - peak) / peak).min()
                rolling_drawdown_series.append(float(dd) if not pd.isna(dd) else 0)
                dates.append(rolling_values.index[i].isoformat())

            # 对齐数据：取有效部分（去掉 NaN）
            valid_sharpe = rolling_sharpe.dropna()
            valid_volatility = rolling_volatility.dropna()

            # 使用 sharpe 的有效日期作为基准
            result_dates = [d.isoformat() for d in valid_sharpe.index]

            result = {
                "dates": result_dates,
                "rolling_sharpe": [
                    float(v) if not pd.isna(v) else 0 for v in valid_sharpe.values
                ],
                "rolling_volatility": [
                    float(v) if not pd.isna(v) else 0 for v in valid_volatility.values
                ],
                "rolling_drawdown": rolling_drawdown_series[
                    : len(result_dates)
                ],
                "window_size": window,
            }

            logger.info(
                f"滚动指标计算完成，数据点数: {len(result_dates)}, 窗口: {window}"
            )
            return result

        except Exception as e:
            logger.error(f"计算滚动指标时间序列失败: {e}", exc_info=True)
            return {}
