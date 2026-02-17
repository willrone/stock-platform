"""
持仓分析模块
提供详细的持仓表现、股票分析和组合构成分析功能
"""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorSeverity, TaskError


class PositionAnalyzer:
    """持仓分析器"""

    def __init__(self):
        pass

    async def analyze_position_performance(
        self, trade_history: List[Dict], portfolio_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        分析持仓表现

        Args:
            trade_history: 交易历史数据
            portfolio_history: 组合历史数据

        Returns:
            持仓分析结果
        """
        try:
            if not trade_history:
                logger.warning("交易历史数据为空，无法进行持仓分析")
                return {}

            logger.info(f"开始分析持仓表现: trade_history长度={len(trade_history)}")

            # 股票级别分析
            stock_analysis = self._analyze_stock_performance(trade_history)
            logger.info(f"股票级别分析完成: 分析了{len(stock_analysis)}只股票")

            # 持仓权重分析
            weight_analysis = self._analyze_position_weights(portfolio_history)

            # 交易模式分析
            trading_pattern_analysis = self._analyze_trading_patterns(trade_history)

            # 持仓时间分析
            holding_period_analysis = self._analyze_holding_periods(trade_history)

            # 风险集中度分析
            concentration_analysis = self._analyze_concentration_risk(
                trade_history, portfolio_history
            )

            result = {
                "stock_performance": stock_analysis,
                "position_weights": weight_analysis,
                "trading_patterns": trading_pattern_analysis,
                "holding_periods": holding_period_analysis,
                "concentration_risk": concentration_analysis,
            }

            logger.info(f"持仓分析完成，分析了 {len(stock_analysis)} 只股票")
            return result

        except Exception as e:
            logger.error(f"持仓表现分析失败: {e}", exc_info=True)
            raise TaskError(
                message=f"持仓表现分析失败: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                original_exception=e,
            )

    def _analyze_stock_performance(
        self, trade_history: List[Dict]
    ) -> List[Dict[str, Any]]:
        """分析各股票表现"""

        # 按股票分组统计
        stock_stats = defaultdict(
            lambda: {
                "stock_code": "",
                "trades": [],
                "buy_trades": [],
                "sell_trades": [],
                "total_pnl": 0,
                "total_volume": 0,
                "total_commission": 0,
            }
        )

        # 收集交易数据
        for trade in trade_history:
            stock_code = trade.get("stock_code", "")
            action = trade.get("action", "")
            pnl = trade.get("pnl", 0)
            quantity = trade.get("quantity", 0)
            price = trade.get("price", 0)
            commission = trade.get("commission", 0)

            stock_stats[stock_code]["stock_code"] = stock_code
            stock_stats[stock_code]["trades"].append(trade)
            stock_stats[stock_code]["total_pnl"] += pnl
            stock_stats[stock_code]["total_volume"] += quantity * price
            stock_stats[stock_code]["total_commission"] += commission

            if action == "BUY":
                stock_stats[stock_code]["buy_trades"].append(trade)
            elif action == "SELL":
                stock_stats[stock_code]["sell_trades"].append(trade)

        # 计算每只股票的详细指标
        result = []
        for stock_code, stats in stock_stats.items():
            buy_trades = stats["buy_trades"]
            sell_trades = stats["sell_trades"]

            # 基础统计
            total_trades = len(sell_trades)  # 以卖出交易为准
            winning_trades = len([t for t in sell_trades if t.get("pnl", 0) > 0])
            losing_trades = len([t for t in sell_trades if t.get("pnl", 0) < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # 盈亏分析
            wins = [t["pnl"] for t in sell_trades if t.get("pnl", 0) > 0]
            losses = [t["pnl"] for t in sell_trades if t.get("pnl", 0) < 0]

            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0

            # 持仓期分析
            holding_periods = self._calculate_holding_periods_for_stock(
                buy_trades, sell_trades
            )
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            max_holding_period = max(holding_periods) if holding_periods else 0
            min_holding_period = min(holding_periods) if holding_periods else 0

            # 交易频率
            if buy_trades and sell_trades:
                first_trade_date = min(
                    pd.to_datetime(t["timestamp"]) for t in buy_trades + sell_trades
                )
                last_trade_date = max(
                    pd.to_datetime(t["timestamp"]) for t in buy_trades + sell_trades
                )
                trading_span_days = (last_trade_date - first_trade_date).days
                trade_frequency = total_trades / max(
                    trading_span_days / 30, 1
                )  # 每月交易次数
            else:
                trade_frequency = 0

            # 价格分析
            buy_prices = [t["price"] for t in buy_trades]
            sell_prices = [t["price"] for t in sell_trades]

            avg_buy_price = np.mean(buy_prices) if buy_prices else 0
            avg_sell_price = np.mean(sell_prices) if sell_prices else 0
            price_improvement = (
                (avg_sell_price - avg_buy_price) / avg_buy_price
                if avg_buy_price > 0
                else 0
            )

            stock_analysis = {
                "stock_code": stock_code,
                "stock_name": stock_code,  # 可以后续从股票信息服务获取
                # 收益指标
                "total_return": float(stats["total_pnl"]),
                "avg_return_per_trade": float(stats["total_pnl"] / total_trades)
                if total_trades > 0
                else 0,
                "return_ratio": float(stats["total_pnl"] / stats["total_volume"])
                if stats["total_volume"] > 0
                else 0,
                # 交易统计
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": float(win_rate),
                "trade_frequency": float(trade_frequency),
                # 盈亏分析
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "largest_win": float(largest_win),
                "largest_loss": float(largest_loss),
                "profit_factor": float(abs(avg_win / avg_loss))
                if avg_loss != 0
                else float("inf"),
                # 持仓期分析
                "avg_holding_period": int(avg_holding_period),
                "max_holding_period": int(max_holding_period),
                "min_holding_period": int(min_holding_period),
                # 价格分析
                "avg_buy_price": float(avg_buy_price),
                "avg_sell_price": float(avg_sell_price),
                "price_improvement": float(price_improvement),
                # 成本分析
                "total_volume": float(stats["total_volume"]),
                "total_commission": float(stats["total_commission"]),
                "commission_ratio": float(
                    stats["total_commission"] / stats["total_volume"]
                )
                if stats["total_volume"] > 0
                else 0,
            }

            result.append(stock_analysis)

        # 按总收益排序
        result.sort(key=lambda x: x["total_return"], reverse=True)

        return result

    def _calculate_holding_periods_for_stock(
        self, buy_trades: List[Dict], sell_trades: List[Dict]
    ) -> List[int]:
        """计算单只股票的持仓期"""

        holding_periods = []

        # 简化的FIFO配对算法
        buy_queue = sorted(buy_trades, key=lambda x: pd.to_datetime(x["timestamp"]))
        sell_queue = sorted(sell_trades, key=lambda x: pd.to_datetime(x["timestamp"]))

        buy_idx = 0
        for sell_trade in sell_queue:
            if buy_idx < len(buy_queue):
                buy_trade = buy_queue[buy_idx]
                buy_date = pd.to_datetime(buy_trade["timestamp"])
                sell_date = pd.to_datetime(sell_trade["timestamp"])

                if sell_date >= buy_date:
                    holding_period = (sell_date - buy_date).days
                    holding_periods.append(holding_period)
                    buy_idx += 1

        return holding_periods

    def _analyze_position_weights(
        self, portfolio_history: List[Dict]
    ) -> Dict[str, Any]:
        """分析持仓权重"""

        if not portfolio_history:
            return {}

        try:
            # 收集所有持仓数据
            all_positions = []
            weight_history = []

            for snapshot in portfolio_history:
                date = snapshot.get("date")
                portfolio_value = snapshot.get("portfolio_value", 0)
                positions = snapshot.get("positions", {})

                if portfolio_value > 0 and positions:
                    # 计算当前时点的权重
                    position_weights = {}
                    for stock_code, position in positions.items():
                        market_value = position.get("market_value", 0)
                        weight = market_value / portfolio_value
                        position_weights[stock_code] = weight

                        all_positions.append(
                            {
                                "date": date,
                                "stock_code": stock_code,
                                "weight": weight,
                                "market_value": market_value,
                                "quantity": position.get("quantity", 0),
                                "current_price": position.get("current_price", 0),
                            }
                        )

                    weight_history.append(
                        {
                            "date": date,
                            "weights": position_weights,
                            "portfolio_value": portfolio_value,
                        }
                    )

            if not all_positions:
                return {}

            # 分析权重统计
            weight_stats = self._calculate_weight_statistics(all_positions)

            # 分析权重变化
            weight_changes = self._analyze_weight_changes(weight_history)

            # 分析集中度
            concentration_metrics = self._calculate_concentration_metrics(
                weight_history
            )

            return {
                "weight_statistics": weight_stats,
                "weight_changes": weight_changes,
                "concentration_metrics": concentration_metrics,
                "position_history": all_positions[-50:],  # 最近50个持仓记录
                "current_weights": weight_history[-1]["weights"]
                if weight_history
                else {},
            }

        except Exception as e:
            logger.error(f"分析持仓权重失败: {e}", exc_info=True)
            return {}

    def _calculate_weight_statistics(self, all_positions: List[Dict]) -> Dict[str, Any]:
        """计算权重统计"""

        # 按股票分组
        stock_weights = defaultdict(list)
        for position in all_positions:
            stock_weights[position["stock_code"]].append(position["weight"])

        # 计算每只股票的权重统计
        weight_stats = []
        for stock_code, weights in stock_weights.items():
            stats = {
                "stock_code": stock_code,
                "avg_weight": float(np.mean(weights)),
                "max_weight": float(max(weights)),
                "min_weight": float(min(weights)),
                "weight_volatility": float(np.std(weights)),
                "observations": len(weights),
            }
            weight_stats.append(stats)

        # 按平均权重排序
        weight_stats.sort(key=lambda x: x["avg_weight"], reverse=True)

        return weight_stats

    def _analyze_weight_changes(
        self, weight_history: List[Dict]
    ) -> List[Dict[str, Any]]:
        """分析权重变化"""

        if len(weight_history) < 2:
            return []

        weight_changes = []

        for i in range(1, len(weight_history)):
            prev_weights = weight_history[i - 1]["weights"]
            curr_weights = weight_history[i]["weights"]
            date = weight_history[i]["date"]

            # 计算权重变化
            all_stocks = set(prev_weights.keys()) | set(curr_weights.keys())

            for stock_code in all_stocks:
                prev_weight = prev_weights.get(stock_code, 0)
                curr_weight = curr_weights.get(stock_code, 0)
                weight_change = curr_weight - prev_weight

                if abs(weight_change) > 0.001:  # 只记录显著变化
                    change_record = {
                        "date": date,
                        "stock_code": stock_code,
                        "prev_weight": float(prev_weight),
                        "curr_weight": float(curr_weight),
                        "weight_change": float(weight_change),
                        "change_type": "increase" if weight_change > 0 else "decrease",
                    }
                    weight_changes.append(change_record)

        # 按日期和变化幅度排序
        weight_changes.sort(
            key=lambda x: (x["date"], abs(x["weight_change"])), reverse=True
        )

        return weight_changes[-100:]  # 返回最近100个显著变化

    def _calculate_concentration_metrics(
        self, weight_history: List[Dict]
    ) -> Dict[str, Any]:
        """计算集中度指标"""

        if not weight_history:
            return {}

        concentration_metrics = []

        for snapshot in weight_history:
            weights = list(snapshot["weights"].values())

            if weights:
                # 赫芬达尔指数 (HHI)
                hhi = sum(w**2 for w in weights)

                # 前N大持仓集中度
                sorted_weights = sorted(weights, reverse=True)
                top_1_concentration = (
                    sorted_weights[0] if len(sorted_weights) >= 1 else 0
                )
                top_3_concentration = (
                    sum(sorted_weights[:3])
                    if len(sorted_weights) >= 3
                    else sum(sorted_weights)
                )
                top_5_concentration = (
                    sum(sorted_weights[:5])
                    if len(sorted_weights) >= 5
                    else sum(sorted_weights)
                )

                # 有效股票数量
                effective_stocks = 1 / hhi if hhi > 0 else 0

                concentration_metrics.append(
                    {
                        "date": snapshot["date"],
                        "hhi": float(hhi),
                        "effective_stocks": float(effective_stocks),
                        "top_1_concentration": float(top_1_concentration),
                        "top_3_concentration": float(top_3_concentration),
                        "top_5_concentration": float(top_5_concentration),
                        "total_positions": len(weights),
                    }
                )

        # 计算平均集中度指标
        if concentration_metrics:
            avg_metrics = {
                "avg_hhi": float(np.mean([m["hhi"] for m in concentration_metrics])),
                "avg_effective_stocks": float(
                    np.mean([m["effective_stocks"] for m in concentration_metrics])
                ),
                "avg_top_1_concentration": float(
                    np.mean([m["top_1_concentration"] for m in concentration_metrics])
                ),
                "avg_top_3_concentration": float(
                    np.mean([m["top_3_concentration"] for m in concentration_metrics])
                ),
                "avg_top_5_concentration": float(
                    np.mean([m["top_5_concentration"] for m in concentration_metrics])
                ),
                "avg_total_positions": float(
                    np.mean([m["total_positions"] for m in concentration_metrics])
                ),
            }
        else:
            avg_metrics = {}

        return {
            "time_series": concentration_metrics[-50:],  # 最近50个时点
            "averages": avg_metrics,
        }

    def _analyze_trading_patterns(self, trade_history: List[Dict]) -> Dict[str, Any]:
        """分析交易模式"""

        if not trade_history:
            return {}

        try:
            # 按时间排序
            sorted_trades = sorted(
                trade_history, key=lambda x: pd.to_datetime(x["timestamp"])
            )

            # 交易时间分析
            time_analysis = self._analyze_trading_times(sorted_trades)

            # 交易规模分析
            size_analysis = self._analyze_trade_sizes(sorted_trades)

            # 交易频率分析
            frequency_analysis = self._analyze_trading_frequency(sorted_trades)

            # 交易成功率分析
            success_analysis = self._analyze_trading_success(sorted_trades)

            return {
                "time_patterns": time_analysis,
                "size_patterns": size_analysis,
                "frequency_patterns": frequency_analysis,
                "success_patterns": success_analysis,
            }

        except Exception as e:
            logger.error(f"分析交易模式失败: {e}", exc_info=True)
            return {}

    def _analyze_trading_times(self, trades: List[Dict]) -> Dict[str, Any]:
        """分析交易时间模式"""

        # 按月份统计
        monthly_counts = defaultdict(int)
        # 按星期统计
        weekday_counts = defaultdict(int)

        for trade in trades:
            timestamp = pd.to_datetime(trade["timestamp"])
            monthly_counts[timestamp.month] += 1
            weekday_counts[timestamp.weekday()] += 1

        # 转换为列表格式
        monthly_distribution = [
            {"month": month, "count": count, "percentage": count / len(trades)}
            for month, count in monthly_counts.items()
        ]

        weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday_distribution = [
            {
                "weekday": weekday,
                "weekday_name": weekday_names[weekday],
                "count": count,
                "percentage": count / len(trades),
            }
            for weekday, count in weekday_counts.items()
        ]

        return {
            "monthly_distribution": monthly_distribution,
            "weekday_distribution": weekday_distribution,
        }

    def _analyze_trade_sizes(self, trades: List[Dict]) -> Dict[str, Any]:
        """分析交易规模"""

        trade_values = []
        for trade in trades:
            value = trade.get("quantity", 0) * trade.get("price", 0)
            trade_values.append(value)

        if not trade_values:
            return {}

        return {
            "avg_trade_size": float(np.mean(trade_values)),
            "median_trade_size": float(np.median(trade_values)),
            "max_trade_size": float(max(trade_values)),
            "min_trade_size": float(min(trade_values)),
            "trade_size_std": float(np.std(trade_values)),
            "total_volume": float(sum(trade_values)),
        }

    def _analyze_trading_frequency(self, trades: List[Dict]) -> Dict[str, Any]:
        """分析交易频率"""

        if len(trades) < 2:
            return {}

        # 计算交易间隔
        timestamps = [pd.to_datetime(t["timestamp"]) for t in trades]
        intervals = [
            (timestamps[i] - timestamps[i - 1]).days for i in range(1, len(timestamps))
        ]

        # 按月统计交易次数
        monthly_trades = defaultdict(int)
        for trade in trades:
            timestamp = pd.to_datetime(trade["timestamp"])
            month_key = timestamp.strftime("%Y-%m")
            monthly_trades[month_key] += 1

        monthly_frequency = list(monthly_trades.values())

        return {
            "avg_interval_days": float(np.mean(intervals)) if intervals else 0,
            "median_interval_days": float(np.median(intervals)) if intervals else 0,
            "min_interval_days": min(intervals) if intervals else 0,
            "max_interval_days": max(intervals) if intervals else 0,
            "avg_monthly_trades": float(np.mean(monthly_frequency))
            if monthly_frequency
            else 0,
            "max_monthly_trades": max(monthly_frequency) if monthly_frequency else 0,
            "total_trading_days": (timestamps[-1] - timestamps[0]).days
            if len(timestamps) >= 2
            else 0,
        }

    def _analyze_trading_success(self, trades: List[Dict]) -> Dict[str, Any]:
        """分析交易成功率"""

        sell_trades = [t for t in trades if t.get("action") == "SELL"]

        if not sell_trades:
            return {}

        # 按盈亏分类
        winning_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in sell_trades if t.get("pnl", 0) < 0]
        breakeven_trades = [t for t in sell_trades if t.get("pnl", 0) == 0]

        total_trades = len(sell_trades)

        return {
            "total_closed_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "breakeven_trades": len(breakeven_trades),
            "win_rate": len(winning_trades) / total_trades if total_trades > 0 else 0,
            "loss_rate": len(losing_trades) / total_trades if total_trades > 0 else 0,
            "avg_win_amount": float(np.mean([t["pnl"] for t in winning_trades]))
            if winning_trades
            else 0,
            "avg_loss_amount": float(np.mean([t["pnl"] for t in losing_trades]))
            if losing_trades
            else 0,
        }

    def _analyze_holding_periods(self, trade_history: List[Dict]) -> Dict[str, Any]:
        """分析持仓期"""

        # 按股票分组计算持仓期
        stock_holdings = defaultdict(lambda: {"buys": [], "sells": []})

        for trade in trade_history:
            stock_code = trade.get("stock_code", "")
            action = trade.get("action", "")

            if action == "BUY":
                stock_holdings[stock_code]["buys"].append(trade)
            elif action == "SELL":
                stock_holdings[stock_code]["sells"].append(trade)

        # 计算所有持仓期
        all_holding_periods = []

        for stock_code, holdings in stock_holdings.items():
            periods = self._calculate_holding_periods_for_stock(
                holdings["buys"], holdings["sells"]
            )
            all_holding_periods.extend(periods)

        if not all_holding_periods:
            return {}

        # 统计分析
        return {
            "avg_holding_period": float(np.mean(all_holding_periods)),
            "median_holding_period": float(np.median(all_holding_periods)),
            "max_holding_period": max(all_holding_periods),
            "min_holding_period": min(all_holding_periods),
            "holding_period_std": float(np.std(all_holding_periods)),
            "total_positions_closed": len(all_holding_periods),
            "short_term_positions": len(
                [p for p in all_holding_periods if p <= 7]
            ),  # 一周内
            "medium_term_positions": len(
                [p for p in all_holding_periods if 7 < p <= 30]
            ),  # 一周到一月
            "long_term_positions": len(
                [p for p in all_holding_periods if p > 30]
            ),  # 超过一月
        }

    def _analyze_concentration_risk(
        self, trade_history: List[Dict], portfolio_history: List[Dict]
    ) -> Dict[str, Any]:
        """分析风险集中度"""

        try:
            # 基于交易量的集中度
            trade_concentration = self._calculate_trade_concentration(trade_history)

            # 基于持仓的集中度
            position_concentration = self._calculate_position_concentration(
                portfolio_history
            )

            return {
                "trade_concentration": trade_concentration,
                "position_concentration": position_concentration,
            }

        except Exception as e:
            logger.error(f"分析风险集中度失败: {e}", exc_info=True)
            return {}

    def _calculate_trade_concentration(
        self, trade_history: List[Dict]
    ) -> Dict[str, Any]:
        """计算交易集中度"""

        # 按股票统计交易量
        stock_volumes = defaultdict(float)
        total_volume = 0

        for trade in trade_history:
            stock_code = trade.get("stock_code", "")
            volume = trade.get("quantity", 0) * trade.get("price", 0)
            stock_volumes[stock_code] += volume
            total_volume += volume

        if total_volume == 0:
            return {}

        # 计算权重
        stock_weights = {
            stock: volume / total_volume for stock, volume in stock_volumes.items()
        }

        # 计算集中度指标
        weights = list(stock_weights.values())
        hhi = sum(w**2 for w in weights)

        sorted_weights = sorted(weights, reverse=True)
        top_1 = sorted_weights[0] if len(sorted_weights) >= 1 else 0
        top_3 = (
            sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
        )
        top_5 = (
            sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights)
        )

        return {
            "hhi": float(hhi),
            "effective_stocks": float(1 / hhi) if hhi > 0 else 0,
            "top_1_weight": float(top_1),
            "top_3_weight": float(top_3),
            "top_5_weight": float(top_5),
            "total_stocks": len(stock_weights),
        }

    def _calculate_position_concentration(
        self, portfolio_history: List[Dict]
    ) -> Dict[str, Any]:
        """计算持仓集中度"""

        if not portfolio_history:
            return {}

        # 使用最新的持仓快照
        latest_snapshot = portfolio_history[-1]
        positions = latest_snapshot.get("positions", {})
        portfolio_value = latest_snapshot.get("portfolio_value", 0)

        if not positions or portfolio_value == 0:
            return {}

        # 计算权重
        weights = []
        for position in positions.values():
            market_value = position.get("market_value", 0)
            weight = market_value / portfolio_value
            weights.append(weight)

        if not weights:
            return {}

        # 计算集中度指标
        hhi = sum(w**2 for w in weights)
        sorted_weights = sorted(weights, reverse=True)

        top_1 = sorted_weights[0] if len(sorted_weights) >= 1 else 0
        top_3 = (
            sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
        )
        top_5 = (
            sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights)
        )

        return {
            "hhi": float(hhi),
            "effective_positions": float(1 / hhi) if hhi > 0 else 0,
            "top_1_weight": float(top_1),
            "top_3_weight": float(top_3),
            "top_5_weight": float(top_5),
            "total_positions": len(weights),
        }
