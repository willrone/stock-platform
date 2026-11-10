"""
回测数据适配器 - 将现有回测结果转换为增强格式
用于支持完整的可视化功能
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorSeverity, TaskError

from ..models.analysis_models import (
    DrawdownAnalysis,
    EnhancedBacktestResult,
    ExtendedRiskMetrics,
    MonthlyReturnsAnalysis,
    PositionAnalysis,
)


class BacktestDataAdapter:
    """回测数据适配器 - 将现有回测结果转换为可视化所需格式"""

    def __init__(self, include_benchmark: bool = False) -> None:
        self.risk_free_rate = 0.03  # 无风险利率，用于计算夏普比率等指标
        self.include_benchmark = include_benchmark

    async def adapt_backtest_result(
        self, task_result: Dict[str, Any]
    ) -> EnhancedBacktestResult:
        """适配现有回测结果为增强格式"""
        try:
            logger.info("开始适配回测结果数据")

            # 1. 提取现有数据
            base_data = self._extract_base_data(task_result)

            # 2. 计算扩展风险指标（引擎权威指标一并传入，避免重算出第二套数）
            engine_metrics = {
                key: task_result.get(key)
                for key in (
                    "annualized_return",
                    "volatility",
                    "sharpe_ratio",
                    "max_drawdown",
                )
                if task_result.get(key) is not None
            }
            extended_risk_metrics = await self._calculate_extended_risk_metrics(
                task_result.get("portfolio_history", []),
                task_result.get("initial_cash", 100000),
                engine_metrics=engine_metrics,
            )

            # 3. 分析月度收益（基准 = t0 初始资金；保留首月）
            monthly_returns = await self._analyze_monthly_returns(
                task_result.get("portfolio_history", []),
                task_result.get("initial_cash", 100000),
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

            # 6. 按需计算基准对比。该步骤可能触发本地/远程数据加载，默认跳过以保持适配器轻量稳定。
            benchmark_data = None
            if self.include_benchmark:
                from ..analysis.enhanced_metrics_calculator import (
                    EnhancedMetricsCalculator,
                )

                metrics_calculator = EnhancedMetricsCalculator()
                benchmark_data = (
                    await metrics_calculator.calculate_benchmark_comparison(
                        task_result.get("portfolio_history", []),
                        benchmark_symbol="000300.SH",
                    )
                )

            # 7. 构建增强结果
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
            )

            logger.info("回测结果数据适配完成")
            return enhanced_result

        except Exception as e:
            logger.opt(exception=True).error(f"适配回测结果失败: {e}")
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

    @staticmethod
    def _prefer_engine(engine_value: Any, local_value: float) -> float:
        """引擎权威值优先；引擎缺失/非有限时才用本地重算值。

        单一口径来源：引擎（portfolio_manager）算的指标与回测结果一起返回，
        适配器不再用另一套定义重算一遍（那会在同一响应里产生两个夏普）。
        """
        if engine_value is None:
            return float(local_value)
        try:
            value = float(engine_value)
        except (TypeError, ValueError):
            return float(local_value)
        if not np.isfinite(value):
            return float(local_value)
        return value

    async def _calculate_extended_risk_metrics(
        self,
        portfolio_history: List[Dict],
        initial_cash: float,
        engine_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExtendedRiskMetrics]:
        """基于现有组合历史计算扩展风险指标

        口径（2026-11-01 修正）：
        - **单一权威来源**：引擎已经算好 annualized_return / volatility /
          sharpe_ratio / max_drawdown 并随结果返回；适配器原先用**另一套定义**又
          重算一遍（夏普 = (年化 − rf) / 年化波动；引擎 = 日均 / 日σ × √252），
          同一份 API 响应因此出现**两个夏普**（实测 9.340288 vs 12.873346，差 3.53）。
          现在引擎值优先，缺失才本地重算。
        - **补回首段**：引擎的快照是每个交易日**收盘后**记录的，第一个快照 = 第一个
          交易日收盘，不是 t0。原先 pct_change().dropna() 把「t0 → 首个快照」整段丢掉：
          期间数少 1（年化指数变成 252/(N−1)，系统性偏高）且首日盈亏不进入波动率 / VaR。
        """
        engine_metrics = engine_metrics or {}

        if not portfolio_history:
            logger.opt(exception=True).warning("组合历史数据为空，无法计算扩展风险指标")
            return None

        try:
            # 转换为pandas DataFrame（按时间排序，保证 iloc[0]/iloc[-1] 是首/末快照）
            df = pd.DataFrame(portfolio_history)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            df = df.sort_index()
            values = df["portfolio_value"]

            # 计算收益率序列：**补上「t0 → 首个快照」这一段**
            # （原实现 pct_change().dropna() 把这一段整段丢掉）
            returns = values.pct_change()
            if len(returns) > 0 and initial_cash and float(initial_cash) > 0:
                returns.iloc[0] = (values.iloc[0] - float(initial_cash)) / float(
                    initial_cash
                )
            returns = returns.dropna()

            if len(returns) == 0:
                logger.opt(exception=True).warning(
                    "收益率序列为空，无法计算扩展风险指标"
                )
                return None

            # 计算扩展指标
            extended_metrics = {}

            # 获取基础指标（引擎权威值优先，避免同一响应里出现两套数）
            volatility = returns.std() * np.sqrt(252)
            if pd.isna(volatility) or np.isnan(volatility):
                volatility = 0.0
            volatility = self._prefer_engine(
                engine_metrics.get("volatility"), volatility
            )

            local_annualized = (values.iloc[-1] / initial_cash) ** (
                252 / len(returns)
            ) - 1
            if pd.isna(local_annualized) or np.isnan(local_annualized):
                local_annualized = 0.0
            annualized_return = self._prefer_engine(
                engine_metrics.get("annualized_return"), local_annualized
            )

            # Sortino比率
            # 口径说明：分子用**引擎的年化**（365 日历天 CAGR，与页面显示的年化同一个数），
            # 分母是下行波动的 √252 交易日年化 —— 这是行业常见搭配（分子 CAGR、
            # 分母交易日缩放），两者口径差异约 3~4%（样本里 242 交易日/年 ≠ 252）。
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
            max_drawdown = self._prefer_engine(
                engine_metrics.get("max_drawdown"),
                self._calculate_max_drawdown(values),
            )
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

            # 最大回撤持续时间
            extended_metrics["max_drawdown_duration"] = (
                self._calculate_max_drawdown_duration(df["portfolio_value"])
            )

            # 夏普比率：引擎权威值优先（引擎定义 = 日均 / 日σ × √252；
            # 本地兜底口径 = (年化 − rf) / 年化波动，与引擎不同，只在引擎缺失时用）
            local_sharpe = (
                (annualized_return - self.risk_free_rate) / volatility
                if volatility > 0
                else 0
            )
            sharpe_ratio = self._prefer_engine(
                engine_metrics.get("sharpe_ratio"), local_sharpe
            )

            return ExtendedRiskMetrics(
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                sortino_ratio=extended_metrics["sortino_ratio"],
                calmar_ratio=extended_metrics["calmar_ratio"],
                max_drawdown_duration=extended_metrics["max_drawdown_duration"],
                var_95=extended_metrics["var_95"],
                downside_deviation=extended_metrics["downside_deviation"],
            )

        except Exception as e:
            logger.opt(exception=True).error(f"计算扩展风险指标失败: {e}")
            return None

    async def _analyze_monthly_returns(
        self, portfolio_history: List[Dict], initial_cash: float
    ) -> Optional[List[MonthlyReturnsAnalysis]]:
        """分析月度收益

        口径（2026-11-01 修正）：
        - 基准是 **t0 初始资金**（initial_cash），不是第一个日度快照。引擎的快照是
          每个交易日**收盘后**记录的，第一个快照 = 第一个交易日收盘，用它当基准会把
          首日盈亏整段排除。
        - **不丢弃首月**：原实现 monthly_values.pct_change().dropna() 会把第一个月度桶
          丢掉，导致「月度序列复利 ≠ 累计收益」（实测缺口 10.29pp：3 个月 +10%/+5%/−2%
          时，月度复利 +2.90% 而累计收益 +13.19%），前端热力图也少首月。
        """

        if not portfolio_history:
            logger.opt(exception=True).warning("组合历史数据为空，无法分析月度收益")
            return None

        try:
            df = pd.DataFrame(portfolio_history)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            df = df.sort_index()

            # 基准：t0 初始资金（引擎权威口径的 total_return 也用它）
            if initial_cash and float(initial_cash) > 0:
                base = float(initial_cash)
            else:
                base = float(df["portfolio_value"].iloc[0])

            # 按月重采样
            monthly_values = df["portfolio_value"].resample("ME").last()
            monthly_returns = monthly_values.pct_change()
            if len(monthly_returns) > 0:
                # 首月的基准是 t0，而不是「不存在的上一个月末」
                monthly_returns.iloc[0] = (monthly_values.iloc[0] - base) / base
            monthly_returns = monthly_returns.dropna()

            result = []

            for date, monthly_return in monthly_returns.items():
                # 计算从初始时点到当前月末的累积收益
                current_value = monthly_values[date]
                cumulative_return = (current_value - base) / base

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
            logger.opt(exception=True).error(f"分析月度收益失败: {e}")
            return None

    async def _analyze_positions(
        self, trade_history: List[Dict], portfolio_history: List[Dict]
    ) -> Optional[List[PositionAnalysis]]:
        """轻量分析持仓表现，保持结果适配器快速稳定。"""

        if not trade_history:
            logger.opt(exception=True).warning("交易历史数据为空，无法分析持仓表现")
            return None

        try:
            stock_stats: Dict[str, Dict[str, Any]] = {}
            for trade in trade_history:
                stock_code = str(trade.get("stock_code", ""))
                if not stock_code:
                    continue
                stats = stock_stats.setdefault(
                    stock_code,
                    {
                        "stock_code": stock_code,
                        "stock_name": stock_code,
                        "trade_count": 0,
                        "raw_trade_count": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "total_pnl": 0.0,
                    },
                )
                pnl = float(trade.get("pnl", 0.0) or 0.0)
                action = str(trade.get("action", "")).upper()
                stats["raw_trade_count"] += 1
                stats["total_pnl"] += pnl
                if action == "SELL" or pnl != 0:
                    stats["trade_count"] += 1
                    if pnl > 0:
                        stats["winning_trades"] += 1
                    elif pnl < 0:
                        stats["losing_trades"] += 1

            position_analysis = []
            for stats in stock_stats.values():
                trade_count = int(stats["trade_count"] or stats["raw_trade_count"])
                winning_trades = int(stats["winning_trades"])
                losing_trades = int(stats["losing_trades"])
                win_rate = winning_trades / max(winning_trades + losing_trades, 1)
                position_analysis.append(
                    PositionAnalysis(
                        stock_code=str(stats["stock_code"]),
                        stock_name=str(stats["stock_name"]),
                        total_return=float(stats["total_pnl"]),
                        trade_count=trade_count,
                        win_rate=float(win_rate),
                        avg_holding_period=0,
                        winning_trades=winning_trades,
                        losing_trades=losing_trades,
                    )
                )

            logger.info(f"持仓分析完成，共分析 {len(position_analysis)} 只股票")
            return position_analysis

        except Exception as e:
            logger.opt(exception=True).error(f"分析持仓表现失败: {e}")
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
            logger.opt(exception=True).error(f"计算平均持仓期失败: {e}")
            return 0

    async def _analyze_drawdowns(
        self, portfolio_history: List[Dict]
    ) -> Optional[DrawdownAnalysis]:
        """分析回撤详情"""

        if not portfolio_history:
            logger.opt(exception=True).warning("组合历史数据为空，无法分析回撤")
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

            # 回撤序列全为 NaN（portfolio_value 整段缺失）时：
            #   - pandas 现在只发 FutureWarning，未来版本 idxmin 会**抛 ValueError**
            #     （且该异常被本函数的 except 吞掉 → 整个回撤分析丢失）；
            #   - max_dd_value 会是 NaN → 进 JSON 就是**非法字面量**。
            # 这里显式判掉：无可用回撤数据时返回全 0 的分析，绝不让 NaN 外泄。
            if len(drawdown) == 0 or drawdown.notna().sum() == 0:
                logger.opt(exception=True).warning(
                    "回撤序列全为 NaN（组合历史含缺失值），返回空回撤分析"
                )
                return DrawdownAnalysis(
                    max_drawdown=0.0,
                    max_drawdown_date=None,
                    max_drawdown_start=None,
                    max_drawdown_end=None,
                    max_drawdown_duration=0,
                    drawdown_curve=[
                        {"date": date.isoformat(), "drawdown": None}
                        for date in drawdown.index
                    ],
                )

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

            # 构建回撤曲线数据（缺失值用 None，不用 NaN —— NaN 不是合法 JSON）
            drawdown_curve = [
                {
                    "date": date.isoformat(),
                    "drawdown": float(dd) if pd.notna(dd) else None,
                }
                for date, dd in drawdown.items()
            ]

            analysis = DrawdownAnalysis(
                max_drawdown=float(max_dd_value),
                max_drawdown_date=max_dd_idx.isoformat() if max_dd_idx else None,
                max_drawdown_start=max_dd_start.isoformat() if max_dd_start else None,
                max_drawdown_end=max_dd_end.isoformat() if max_dd_end else None,
                max_drawdown_duration=(
                    (max_dd_end - max_dd_start).days
                    if max_dd_start and max_dd_end
                    else 0
                ),
                drawdown_curve=drawdown_curve,
            )

            logger.info(f"回撤分析完成，最大回撤: {max_dd_value:.2%}")
            return analysis

        except Exception as e:
            logger.opt(exception=True).error(f"分析回撤详情失败: {e}")
            return None

    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """计算最大回撤"""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return float(drawdown.min())

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
            logger.opt(exception=True).error(f"计算最大回撤持续时间失败: {e}")
            return 0
