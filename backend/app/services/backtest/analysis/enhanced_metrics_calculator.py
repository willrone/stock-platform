"""
增强的指标计算引擎
提供更多专业的金融风险和绩效指标计算
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorSeverity, TaskError


class EnhancedMetricsCalculator:
    """增强的指标计算器"""

    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化指标计算器

        Args:
            risk_free_rate: 无风险利率，默认3%
        """
        self.risk_free_rate = risk_free_rate

    async def calculate_performance_metrics(
        self,
        portfolio_history: List[Dict],
        trade_history: List[Dict],
        initial_cash: float,
    ) -> Dict[str, Any]:
        """
        计算完整的绩效指标

        Args:
            portfolio_history: 组合历史数据
            trade_history: 交易历史数据
            initial_cash: 初始资金

        Returns:
            完整的绩效指标字典
        """
        try:
            if not portfolio_history:
                logger.warning("组合历史数据为空，无法计算绩效指标")
                return {}

            # 转换为pandas DataFrame
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
            portfolio_df.set_index("date", inplace=True)

            # 计算收益率序列
            returns = portfolio_df["portfolio_value"].pct_change().dropna()

            if len(returns) == 0:
                logger.warning("收益率序列为空，无法计算绩效指标")
                return {}

            # 基础收益指标
            basic_metrics = self._calculate_basic_metrics(
                portfolio_df, returns, initial_cash
            )

            # 风险调整收益指标
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(
                returns, basic_metrics["annualized_return"]
            )

            # 回撤指标
            drawdown_metrics = self._calculate_drawdown_metrics(
                portfolio_df["portfolio_value"]
            )

            # 交易统计指标
            trading_metrics = self._calculate_trading_metrics(trade_history)

            # 分布统计指标
            distribution_metrics = self._calculate_distribution_metrics(returns)

            # 滚动指标
            rolling_metrics = self._calculate_rolling_metrics(returns)

            # 合并所有指标
            all_metrics = {
                **basic_metrics,
                **risk_adjusted_metrics,
                **drawdown_metrics,
                **trading_metrics,
                **distribution_metrics,
                **rolling_metrics,
            }

            logger.info("绩效指标计算完成")
            return all_metrics

        except Exception as e:
            logger.error(f"计算绩效指标失败: {e}", exc_info=True)
            raise TaskError(
                message=f"计算绩效指标失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    def _calculate_basic_metrics(
        self, portfolio_df: pd.DataFrame, returns: pd.Series, initial_cash: float
    ) -> Dict[str, float]:
        """计算基础收益指标"""

        # 总收益率
        total_return = (
            portfolio_df["portfolio_value"].iloc[-1] - initial_cash
        ) / initial_cash

        # 年化收益率
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        annualized_return = (
            (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
        )

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 日均收益率
        daily_return_mean = returns.mean()

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "volatility": float(volatility),
            "daily_return_mean": float(daily_return_mean),
            "trading_days": len(returns),
        }

    def _calculate_risk_adjusted_metrics(
        self, returns: pd.Series, annualized_return: float
    ) -> Dict[str, float]:
        """计算风险调整收益指标"""

        # 夏普比率
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (
            (annualized_return - self.risk_free_rate) / volatility
            if volatility > 0
            else 0
        )

        # Sortino比率
        # 标准 downside deviation：对所有收益日计算，正收益视为0
        target_return = self.risk_free_rate / 252  # 日无风险利率作为目标收益
        downside_diff = returns - target_return
        downside_diff = downside_diff.clip(upper=0)  # 正的部分归零
        downside_deviation = np.sqrt((downside_diff ** 2).mean()) * np.sqrt(252)
        sortino_ratio = (
            (annualized_return - self.risk_free_rate) / downside_deviation
            if downside_deviation > 1e-6
            else 0
        )

        # 信息比率（假设基准收益为无风险利率）
        excess_returns = returns - (self.risk_free_rate / 252)
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (
            excess_returns.mean() * np.sqrt(252) / tracking_error
            if tracking_error > 0
            else 0
        )

        # VaR和CVaR
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = (
            returns[returns <= var_95].mean()
            if len(returns[returns <= var_95]) > 0
            else var_95
        )
        cvar_99 = (
            returns[returns <= var_99].mean()
            if len(returns[returns <= var_99]) > 0
            else var_99
        )

        return {
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "information_ratio": float(information_ratio),
            "downside_deviation": float(downside_deviation),
            "tracking_error": float(tracking_error),
            "var_95": float(var_95),
            "var_99": float(var_99),
            "cvar_95": float(cvar_95),
            "cvar_99": float(cvar_99),
        }

    def _calculate_drawdown_metrics(
        self, portfolio_values: pd.Series
    ) -> Dict[str, float]:
        """计算回撤指标"""

        # 计算回撤序列
        cumulative_returns = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        # 最大回撤
        max_drawdown = drawdown.min()

        # 平均回撤
        negative_drawdowns = drawdown[drawdown < 0]
        avg_drawdown = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0

        # 回撤恢复时间
        max_dd_date = drawdown.idxmin()
        recovery_date = None

        # 寻找恢复日期
        for date in drawdown[max_dd_date:].index:
            if drawdown[date] >= -0.001:  # 基本恢复到峰值
                recovery_date = date
                break

        recovery_days = (recovery_date - max_dd_date).days if recovery_date else None

        # Calmar比率 = 年化收益 / |最大回撤|（不扣无风险利率）
        annualized_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (
            252 / len(portfolio_values)
        ) - 1
        calmar_ratio = (
            annualized_return / abs(max_drawdown)
            if max_drawdown < -0.01  # 最大回撤至少1%才有意义
            else 0
        )

        # 回撤持续时间统计
        drawdown_durations = self._calculate_drawdown_durations(drawdown)

        return {
            "max_drawdown": float(max_drawdown),
            "avg_drawdown": float(avg_drawdown),
            "calmar_ratio": float(calmar_ratio),
            "max_drawdown_recovery_days": recovery_days or 0,
            "avg_drawdown_duration": float(np.mean(drawdown_durations))
            if drawdown_durations
            else 0,
            "max_drawdown_duration": max(drawdown_durations)
            if drawdown_durations
            else 0,
        }

    def _calculate_drawdown_durations(self, drawdown: pd.Series) -> List[int]:
        """计算所有回撤期间的持续时间"""
        durations = []
        current_duration = 0
        in_drawdown = False

        for dd in drawdown:
            if dd < -0.001:  # 进入回撤
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:  # 退出回撤
                if in_drawdown:
                    durations.append(current_duration)
                    in_drawdown = False
                    current_duration = 0

        # 如果最后还在回撤中
        if in_drawdown:
            durations.append(current_duration)

        return durations

    def _calculate_trading_metrics(self, trade_history: List[Dict]) -> Dict[str, Any]:
        """计算交易统计指标"""

        if not trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "avg_trade_duration": 0,
            }

        # 分析交易
        winning_trades = [t for t in trade_history if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trade_history if t.get("pnl", 0) < 0]

        # 基础统计
        total_trades = len(trade_history)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # 盈亏统计
        wins = [t["pnl"] for t in winning_trades]
        losses = [t["pnl"] for t in losing_trades]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        # 盈亏比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # 交易持续时间分析（简化版）
        trade_durations = []
        buy_trades = {}

        for trade in trade_history:
            stock_code = trade.get("stock_code", "")
            action = trade.get("action", "")
            timestamp = pd.to_datetime(trade.get("timestamp", ""))

            if action == "BUY":
                buy_trades[stock_code] = timestamp
            elif action == "SELL" and stock_code in buy_trades:
                duration = (timestamp - buy_trades[stock_code]).days
                trade_durations.append(duration)
                del buy_trades[stock_code]

        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

        return {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "largest_win": float(largest_win),
            "largest_loss": float(largest_loss),
            "avg_trade_duration": float(avg_trade_duration),
        }

    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """计算收益分布统计指标"""

        # 基础统计量
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # 正收益和负收益统计
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        positive_days = len(positive_returns)
        negative_days = len(negative_returns)
        total_days = len(returns)

        positive_ratio = positive_days / total_days if total_days > 0 else 0

        # 最大连续盈利和亏损天数
        max_consecutive_wins = self._calculate_max_consecutive(returns > 0)
        max_consecutive_losses = self._calculate_max_consecutive(returns < 0)

        return {
            "return_mean": float(mean_return),
            "return_std": float(std_return),
            "return_skewness": float(skewness),
            "return_kurtosis": float(kurtosis),
            "positive_days": positive_days,
            "negative_days": negative_days,
            "positive_ratio": float(positive_ratio),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
        }

    def _calculate_max_consecutive(self, condition_series: pd.Series) -> int:
        """计算最大连续True的数量"""
        max_consecutive = 0
        current_consecutive = 0

        for value in condition_series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_rolling_metrics(
        self, returns: pd.Series, window: int = 252
    ) -> Dict[str, Any]:
        """计算滚动指标"""

        if len(returns) < window:
            logger.warning(f"数据长度 {len(returns)} 小于滚动窗口 {window}，跳过滚动指标计算")
            return {}

        # 滚动夏普比率
        rolling_sharpe = (
            returns.rolling(window).mean() * np.sqrt(252) - self.risk_free_rate
        ) / (returns.rolling(window).std() * np.sqrt(252))

        # 滚动波动率
        rolling_volatility = returns.rolling(window).std() * np.sqrt(252)

        # 滚动最大回撤（简化版）
        rolling_cumret = (
            (1 + returns).rolling(window).apply(lambda x: x.prod(), raw=False)
        )
        rolling_max = rolling_cumret.rolling(window).max()
        rolling_drawdown = (rolling_cumret - rolling_max) / rolling_max
        rolling_max_dd = rolling_drawdown.rolling(window).min()

        return {
            "rolling_sharpe_mean": float(rolling_sharpe.mean())
            if not rolling_sharpe.empty
            else 0,
            "rolling_sharpe_std": float(rolling_sharpe.std())
            if not rolling_sharpe.empty
            else 0,
            "rolling_volatility_mean": float(rolling_volatility.mean())
            if not rolling_volatility.empty
            else 0,
            "rolling_volatility_std": float(rolling_volatility.std())
            if not rolling_volatility.empty
            else 0,
            "rolling_max_drawdown_mean": float(rolling_max_dd.mean())
            if not rolling_max_dd.empty
            else 0,
        }

    async def calculate_benchmark_comparison(
        self,
        portfolio_history: List[Dict],
        benchmark_symbol: str = "000300.SH",  # 沪深300
    ) -> Dict[str, Any]:
        """
        计算与基准的对比指标

        Args:
            portfolio_history: 组合历史数据
            benchmark_symbol: 基准指数代码

        Returns:
            基准对比指标
        """
        try:
            logger.info(f"计算与基准 {benchmark_symbol} 的对比指标")

            if not portfolio_history:
                logger.warning("组合历史数据为空，无法计算基准对比")
                return {}

            # 获取基准数据
            from app.core.config import settings
            from app.services.data.simple_data_service import SimpleDataService
            from app.services.data.stock_data_loader import StockDataLoader

            # 确定日期范围
            dates = [pd.to_datetime(snapshot["date"]) for snapshot in portfolio_history]
            start_date = min(dates)
            end_date = max(dates)

            # 尝试从本地加载基准数据
            loader = StockDataLoader(data_root=settings.DATA_ROOT_PATH)
            benchmark_df = loader.load_stock_data(
                benchmark_symbol, start_date=start_date, end_date=end_date
            )

            if benchmark_df.empty or len(benchmark_df) == 0:
                # 尝试从远程服务获取
                logger.info(f"本地无基准数据，尝试从远程服务获取: {benchmark_symbol}")
                data_service = SimpleDataService()
                benchmark_data_list = await data_service.get_stock_data(
                    benchmark_symbol, start_date, end_date
                )

                if benchmark_data_list and len(benchmark_data_list) > 0:
                    benchmark_df = pd.DataFrame(
                        [
                            {"date": item.date, "close": item.close}
                            for item in benchmark_data_list
                        ]
                    )
                    benchmark_df = benchmark_df.set_index("date")
                    benchmark_df = benchmark_df.sort_index()

            if benchmark_df.empty or len(benchmark_df) == 0:
                logger.warning(f"无法获取基准数据: {benchmark_symbol}，返回空结果")
                return {
                    "benchmark_symbol": benchmark_symbol,
                    "benchmark_name": self._get_benchmark_name(benchmark_symbol),
                    "error": "无法获取基准数据",
                }

            # 准备组合数据
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
            portfolio_df = portfolio_df.set_index("date")
            portfolio_df = portfolio_df.sort_index()

            # 计算收益率
            portfolio_returns = portfolio_df["portfolio_value"].pct_change().dropna()
            benchmark_returns = benchmark_df["close"].pct_change().dropna()

            # 对齐日期
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) == 0:
                logger.warning("组合和基准数据没有共同的日期，无法计算对比指标")
                return {
                    "benchmark_symbol": benchmark_symbol,
                    "benchmark_name": self._get_benchmark_name(benchmark_symbol),
                    "error": "数据日期不匹配",
                }

            portfolio_returns_aligned = portfolio_returns.loc[common_dates]
            benchmark_returns_aligned = benchmark_returns.loc[common_dates]

            # 计算相关性
            correlation = float(
                portfolio_returns_aligned.corr(benchmark_returns_aligned)
            )

            # 计算Beta（组合收益对基准收益的敏感度）
            if (
                len(portfolio_returns_aligned) > 1
                and benchmark_returns_aligned.std() > 0
            ):
                covariance = portfolio_returns_aligned.cov(benchmark_returns_aligned)
                benchmark_variance = benchmark_returns_aligned.var()
                beta = (
                    float(covariance / benchmark_variance)
                    if benchmark_variance > 0
                    else 0.0
                )
            else:
                beta = 0.0

            # 计算Alpha（超额收益）
            portfolio_annual_return = portfolio_returns_aligned.mean() * 252
            benchmark_annual_return = benchmark_returns_aligned.mean() * 252
            alpha = float(portfolio_annual_return - (beta * benchmark_annual_return))

            # 计算跟踪误差（组合收益与基准收益的差异的标准差）
            excess_returns = portfolio_returns_aligned - benchmark_returns_aligned
            tracking_error = float(excess_returns.std() * np.sqrt(252))

            # 计算信息比率（超额收益/跟踪误差）
            excess_return_mean = float(excess_returns.mean() * np.sqrt(252))
            information_ratio = (
                float(excess_return_mean / tracking_error)
                if tracking_error > 0
                else 0.0
            )

            # 计算总超额收益
            total_excess_return = float(
                (
                    portfolio_df["portfolio_value"].iloc[-1]
                    / portfolio_df["portfolio_value"].iloc[0]
                )
                - (benchmark_df["close"].iloc[-1] / benchmark_df["close"].iloc[0])
            )

            # 计算上涨/下跌捕获率
            up_capture, down_capture = self._calculate_capture_ratios(
                portfolio_returns_aligned, benchmark_returns_aligned
            )

            return {
                "benchmark_symbol": benchmark_symbol,
                "benchmark_name": self._get_benchmark_name(benchmark_symbol),
                "correlation": correlation,
                "beta": beta,
                "alpha": alpha,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "excess_return": total_excess_return,
                "upside_capture": up_capture,
                "downside_capture": down_capture,
                "benchmark_data": benchmark_df["close"].to_dict(),  # 保存基准数据用于前端展示
            }

        except Exception as e:
            logger.error(f"计算基准对比指标失败: {e}", exc_info=True)
            return {
                "benchmark_symbol": benchmark_symbol,
                "benchmark_name": self._get_benchmark_name(benchmark_symbol),
                "error": str(e),
            }

    def _get_benchmark_name(self, symbol: str) -> str:
        """获取基准名称"""
        benchmark_names = {
            "000300.SH": "沪深300指数",
            "000905.SH": "中证500指数",
            "000852.SH": "中证1000指数",
            "399001.SZ": "深证成指",
            "399006.SZ": "创业板指",
        }
        return benchmark_names.get(symbol, f"{symbol}指数")

    def _calculate_capture_ratios(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """计算上涨/下跌捕获率"""
        try:
            # 上涨捕获率：基准上涨时，组合的平均收益 / 基准的平均收益
            up_periods = benchmark_returns > 0
            if up_periods.sum() > 0:
                portfolio_up_avg = portfolio_returns[up_periods].mean()
                benchmark_up_avg = benchmark_returns[up_periods].mean()
                up_capture = (
                    float(portfolio_up_avg / benchmark_up_avg)
                    if benchmark_up_avg > 0
                    else 0.0
                )
            else:
                up_capture = 0.0

            # 下跌捕获率：基准下跌时，组合的平均收益 / 基准的平均收益
            down_periods = benchmark_returns < 0
            if down_periods.sum() > 0:
                portfolio_down_avg = portfolio_returns[down_periods].mean()
                benchmark_down_avg = benchmark_returns[down_periods].mean()
                down_capture = (
                    float(portfolio_down_avg / benchmark_down_avg)
                    if benchmark_down_avg < 0
                    else 0.0
                )
            else:
                down_capture = 0.0

            return up_capture, down_capture
        except Exception as e:
            logger.error(f"计算捕获率失败: {e}")
            return 0.0, 0.0

    def calculate_sector_analysis(self, trade_history: List[Dict]) -> Dict[str, Any]:
        """
        计算行业分析指标

        Args:
            trade_history: 交易历史数据

        Returns:
            行业分析结果
        """
        try:
            # 按股票代码分组统计
            stock_performance = {}

            for trade in trade_history:
                stock_code = trade.get("stock_code", "")
                pnl = trade.get("pnl", 0)

                if stock_code not in stock_performance:
                    stock_performance[stock_code] = {
                        "total_pnl": 0,
                        "trade_count": 0,
                        "win_count": 0,
                    }

                stock_performance[stock_code]["total_pnl"] += pnl
                if trade.get("action") == "SELL":
                    stock_performance[stock_code]["trade_count"] += 1
                    if pnl > 0:
                        stock_performance[stock_code]["win_count"] += 1

            # 计算每只股票的胜率
            for stock_code, perf in stock_performance.items():
                perf["win_rate"] = (
                    perf["win_count"] / perf["trade_count"]
                    if perf["trade_count"] > 0
                    else 0
                )

            # 排序
            sorted_stocks = sorted(
                stock_performance.items(), key=lambda x: x[1]["total_pnl"], reverse=True
            )

            return {
                "stock_performance": dict(sorted_stocks),
                "best_performing_stock": sorted_stocks[0] if sorted_stocks else None,
                "worst_performing_stock": sorted_stocks[-1] if sorted_stocks else None,
                "total_stocks_traded": len(stock_performance),
            }

        except Exception as e:
            logger.error(f"计算行业分析失败: {e}", exc_info=True)
            return {}
