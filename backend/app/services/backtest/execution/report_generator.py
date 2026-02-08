"""
回测报告生成模块
负责生成回测报告、计算指标等
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from ..models.data_models import BacktestConfig
from ..core.portfolio_manager import PortfolioManager
from ..core.base_strategy import BaseStrategy


class BacktestReportGenerator:
    """回测报告生成器"""

    def __init__(self):
        """初始化报告生成器"""
        pass

    def generate_backtest_report(
        self,
        strategy_name: str,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        config: BacktestConfig,
        portfolio_manager: PortfolioManager,
        performance_metrics: Dict[str, float],
        strategy_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """生成回测报告"""

        # 基础信息
        report = {
            "strategy_name": strategy_name,
            "stock_codes": stock_codes,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_cash": config.initial_cash,
            # NOTE: Do NOT call get_portfolio_value({}) here - passing an empty price map
            # will value all positions at 0 and return cash-only, which makes final_value
            # inconsistent with total_return/portfolio_history.
            # Use the last recorded portfolio value (already computed with prices) when available.
            "final_value": (
                portfolio_manager.portfolio_history[-1]["portfolio_value"]
                if getattr(portfolio_manager, "portfolio_history", None)
                else portfolio_manager.get_portfolio_value({})
            ),
            # 收益指标
            "total_return": performance_metrics.get("total_return", 0),
            "annualized_return": performance_metrics.get("annualized_return", 0),
            # 风险指标
            "volatility": performance_metrics.get("volatility", 0),
            "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0),
            "max_drawdown": performance_metrics.get("max_drawdown", 0),
            # 交易统计
            "total_trades": performance_metrics.get("total_trades", 0),
            "win_rate": performance_metrics.get("win_rate", 0),
            "profit_factor": performance_metrics.get("profit_factor", 0),
            "winning_trades": performance_metrics.get("winning_trades", 0),
            "losing_trades": performance_metrics.get("losing_trades", 0),
            # 将指标也放在 metrics 字段中，方便优化器使用
            "metrics": {
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0),
                "total_return": performance_metrics.get("total_return", 0),
                "annualized_return": performance_metrics.get("annualized_return", 0),
                "max_drawdown": performance_metrics.get("max_drawdown", 0),
                "volatility": performance_metrics.get("volatility", 0),
                "win_rate": performance_metrics.get("win_rate", 0),
                "profit_factor": performance_metrics.get("profit_factor", 0),
                "total_trades": performance_metrics.get("total_trades", 0),
            },
            # 配置信息
            "backtest_config": {
                "strategy_name": strategy_name,  # 添加策略名称，方便前端获取
                "start_date": start_date.isoformat(),  # 添加开始日期
                "end_date": end_date.isoformat(),  # 添加结束日期
                "initial_cash": config.initial_cash,  # 添加初始资金
                "commission_rate": config.commission_rate,
                "slippage_rate": config.slippage_rate,
                "max_position_size": config.max_position_size,
                **(
                    {"strategy_config": strategy_config}
                    if strategy_config
                    and isinstance(strategy_config, dict)
                    and len(strategy_config) > 0
                    else {}
                ),
            },
            # 交易记录
            "trade_history": [
                {
                    "trade_id": trade.trade_id if hasattr(trade, 'trade_id') else trade['trade_id'],
                    "stock_code": trade.stock_code if hasattr(trade, 'stock_code') else trade['stock_code'],
                    "action": trade.action if hasattr(trade, 'action') else trade['action'],
                    "quantity": trade.quantity if hasattr(trade, 'quantity') else trade['quantity'],
                    "price": trade.price if hasattr(trade, 'price') else trade['price'],
                    "timestamp": (trade.timestamp if hasattr(trade, 'timestamp') else trade['timestamp']).isoformat(),
                    "commission": trade.commission if hasattr(trade, 'commission') else trade['commission'],
                    "slippage_cost": getattr(trade, "slippage_cost", 0.0) if hasattr(trade, 'slippage_cost') else trade.get('slippage_cost', 0.0),
                    "pnl": trade.pnl if hasattr(trade, 'pnl') else trade['pnl'],
                }
                for trade in portfolio_manager.trades
            ],
            # 组合历史（包含完整的positions信息）
            "portfolio_history": [
                {
                    "date": snapshot["date"].isoformat(),
                    "portfolio_value": snapshot["portfolio_value"],
                    "portfolio_value_without_cost": snapshot.get(
                        "portfolio_value_without_cost", snapshot["portfolio_value"]
                    ),
                    "cash": snapshot["cash"],
                    "positions_count": len(snapshot.get("positions", {})),
                    "positions": snapshot.get("positions", {}),  # 包含完整的持仓信息
                    "total_return": (snapshot["portfolio_value"] - config.initial_cash)
                    / config.initial_cash
                    if config.initial_cash > 0
                    else 0,
                    "total_return_without_cost": (
                        snapshot.get(
                            "portfolio_value_without_cost", snapshot["portfolio_value"]
                        )
                        - config.initial_cash
                    )
                    / config.initial_cash
                    if config.initial_cash > 0
                    else 0,
                }
                for snapshot in portfolio_manager.portfolio_history
            ],
            # 交易成本统计
            "cost_statistics": {
                "total_commission": portfolio_manager.total_commission,
                "total_slippage": portfolio_manager.total_slippage,
                "total_cost": portfolio_manager.total_commission
                + portfolio_manager.total_slippage,
                "cost_ratio": (
                    portfolio_manager.total_commission
                    + portfolio_manager.total_slippage
                )
                / config.initial_cash
                if config.initial_cash > 0
                else 0,
            },
        }

        # 添加无成本指标到报告
        metrics_without_cost = portfolio_manager.get_performance_metrics_without_cost()
        report["excess_return_without_cost"] = {
            "mean": metrics_without_cost.get("mean", 0),
            "std": metrics_without_cost.get("std", 0),
            "annualized_return": metrics_without_cost.get("annualized_return", 0),
            "information_ratio": metrics_without_cost.get("information_ratio", 0),
            "max_drawdown": metrics_without_cost.get("max_drawdown", 0),
        }

        report["excess_return_with_cost"] = {
            "mean": performance_metrics.get("volatility", 0) / np.sqrt(252)
            if performance_metrics.get("volatility", 0) > 0
            else 0,
            "std": performance_metrics.get("volatility", 0),
            "annualized_return": performance_metrics.get("annualized_return", 0),
            "information_ratio": performance_metrics.get(
                "sharpe_ratio", 0
            ),  # 使用夏普比率作为近似
            "max_drawdown": performance_metrics.get("max_drawdown", 0),
        }

        # 计算额外的分析指标
        report.update(self._calculate_additional_metrics(portfolio_manager))

        return report


    def rebalance_topk_buffer(
        self,
        portfolio_manager: PortfolioManager,
        current_prices: Dict[str, float],
        current_date: datetime,
        scores: Dict[str, float],
        topk: int = 10,
        buffer_n: int = 20,
        max_changes: int = 2,
        strategy: Optional[BaseStrategy] = None,
        debug: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        """每日 TopK 选股 + buffer 换仓 + 每天最多换 max_changes 只。

        规则（实盘对齐版）：
        - 目标持仓数量=topk
        - 若持仓仍在 Top(topk+buffer_n) 内，则尽量保留（减少换手）
        - 每天最多做 max_changes 个 "卖出+买入" 的替换

        Returns:
            executed_trade_signals, unexecuted_signals, trades_this_day
        """
        executed_trade_signals: List[Dict[str, Any]] = []
        unexecuted_signals: List[Dict[str, Any]] = []
        trades_this_day = 0

        if topk <= 0:
            return executed_trade_signals, unexecuted_signals, trades_this_day

        # P2 优化: 使用 NumPy 向量化排序，替代 Python sorted()
        # rank by score desc, tie-break by stock_code for determinism
        codes = list(scores.keys())
        score_values = np.array([scores[c] for c in codes])
        
        # argsort 默认升序，用负号实现降序
        sorted_indices = np.argsort(-score_values)
        ranked_codes = [codes[i] for i in sorted_indices]
        
        topk_list = ranked_codes[:topk]
        buffer_list = ranked_codes[: max(topk, topk + buffer_n)]
        buffer_set = set(buffer_list)

        holdings = list(portfolio_manager.positions.keys())
        holdings_set = set(holdings)

        # Keep holdings inside buffer zone
        kept = [c for c in holdings if c in buffer_set]

        # P2 优化: 使用 NumPy 向量化 rank_index 查找
        # If kept > topk, trim lowest-ranked among kept
        rank_index = {c: i for i, c in enumerate(ranked_codes)}
        if len(kept) > topk:
            # 使用 NumPy 数组操作替代 sorted
            kept_ranks = np.array([rank_index.get(c, 10**9) for c in kept])
            kept_sorted_indices = np.argsort(kept_ranks)
            kept = [kept[i] for i in kept_sorted_indices[:topk]]

        kept_set = set(kept)

        # Sell candidates: holdings outside buffer OR trimmed
        to_sell = [c for c in holdings if c not in kept_set]

        # Buy candidates: topk names not already kept
        to_buy = [c for c in topk_list if c not in kept_set]

        # Decide actions under max_changes
        # - If current holdings < topk: allow buys even without sells (build initial positions)
        # - Otherwise: do replacement pairs (sell+buy) up to max_changes
        current_n = len(holdings)
        if current_n < topk:
            # how many new names to buy today
            buy_quota = min(max_changes, topk - current_n, len(to_buy))
            to_sell = []
            to_buy = to_buy[:buy_quota]
        else:
            # replacement pairs
            n_pairs = min(max_changes, len(to_sell), len(to_buy))
            to_sell = to_sell[:n_pairs]
            to_buy = to_buy[:n_pairs]

        if debug:
            try:
                nonzero = sum(1 for _, s in scores.items() if isinstance(s, (int, float)) and s != 0)
                logger.info(
                    f"[topk_buffer] {current_date.date()} holdings={len(holdings)} nonzero_scores={nonzero} "
                    f"topk={topk} buffer={buffer_n} max_changes={max_changes} "
                    f"to_sell={len(to_sell)} to_buy={len(to_buy)}"
                )
                logger.info(
                    f"[topk_buffer] topk_list(head)={topk_list[:min(5,len(topk_list))]} "
                    f"holdings(head)={holdings[:min(5,len(holdings))]}"
                )
            except Exception:
                pass

        # Execute sells first
        successful_sells = 0
        for code in to_sell:
            sig = TradingSignal(
                timestamp=current_date,
                stock_code=code,
                signal_type=SignalType.SELL,
                strength=1.0,
                price=float(current_prices.get(code, 0.0) or 0.0),
                reason=f"topk_buffer rebalance sell (out of buffer/topk)",
                metadata={"trade_mode": "topk_buffer"},
            )
            if strategy is not None:
                is_valid, validation_reason = strategy.validate_signal(
                    sig,
                    portfolio_manager.get_portfolio_value(current_prices),
                    portfolio_manager.positions,
                )
                if not is_valid:
                    unexecuted_signals.append(
                        {
                            "stock_code": code,
                            "timestamp": current_date,
                            "signal_type": sig.signal_type.name,
                            "execution_reason": validation_reason or "信号验证失败",
                        }
                    )
                    continue

            trade, failure_reason = portfolio_manager.execute_signal(sig, current_prices)
            if trade:
                successful_sells += 1
                trades_this_day += 1
                executed_trade_signals.append(
                    {"stock_code": code, "timestamp": current_date, "signal_type": sig.signal_type.name}
                )
            else:
                unexecuted_signals.append(
                    {
                        "stock_code": code,
                        "timestamp": current_date,
                        "signal_type": sig.signal_type.name,
                        "execution_reason": failure_reason or "执行失败（未知原因）",
                    }
                )

        # Execute buys
        # Guardrails:
        # 1) replacement 模式下：只允许用「成功卖出」换入，避免卖失败仍买导致持仓膨胀
        # 2) 任何情况下都不允许持仓数超过 topk
        current_positions_n = len(portfolio_manager.positions)
        remaining_capacity = max(0, topk - current_positions_n)

        if current_n >= topk:
            # replacement mode: buys must be backed by successful sells
            buy_quota = min(len(to_buy), successful_sells, remaining_capacity)
        else:
            # build mode: still respect capacity
            buy_quota = min(len(to_buy), remaining_capacity)

        to_buy = to_buy[:buy_quota]

        for code in to_buy:
            # Hard cap: never allow positions to exceed topk (even if earlier logic misbehaves)
            if len(portfolio_manager.positions) >= topk:
                unexecuted_signals.append(
                    {
                        "stock_code": code,
                        "timestamp": current_date,
                        "signal_type": SignalType.BUY.name,
                        "execution_reason": f"超过topk持仓上限(topk={topk})，跳过买入",
                    }
                )
                break

            sig = TradingSignal(
                timestamp=current_date,
                stock_code=code,
                signal_type=SignalType.BUY,
                strength=1.0,
                price=float(current_prices.get(code, 0.0) or 0.0),
                reason=f"topk_buffer rebalance buy (enter top{topk})",
                metadata={"trade_mode": "topk_buffer"},
            )
            if strategy is not None:
                is_valid, validation_reason = strategy.validate_signal(
                    sig,
                    portfolio_manager.get_portfolio_value(current_prices),
                    portfolio_manager.positions,
                )
                if not is_valid:
                    unexecuted_signals.append(
                        {
                            "stock_code": code,
                            "timestamp": current_date,
                            "signal_type": sig.signal_type.name,
                            "execution_reason": validation_reason or "信号验证失败",
                        }
                    )
                    continue

            trade, failure_reason = portfolio_manager.execute_signal(sig, current_prices)
            if trade:
                trades_this_day += 1
                executed_trade_signals.append(
                    {"stock_code": code, "timestamp": current_date, "signal_type": sig.signal_type.name}
                )
            else:
                unexecuted_signals.append(
                    {
                        "stock_code": code,
                        "timestamp": current_date,
                        "signal_type": sig.signal_type.name,
                        "execution_reason": failure_reason or "执行失败（未知原因）",
                    }
                )

        return executed_trade_signals, unexecuted_signals, trades_this_day


    def calculate_additional_metrics(
        self, portfolio_manager: PortfolioManager
    ) -> Dict[str, Any]:
        """计算额外的分析指标（时间分段表现、个股表现等）"""
        additional_metrics: Dict[str, Any] = {}

        try:
            if not portfolio_manager.portfolio_history:
                return additional_metrics

            # --- 时间分段表现：按月 / 按年收益 ---
            portfolio_values = pd.Series(
                [
                    snapshot["portfolio_value"]
                    for snapshot in portfolio_manager.portfolio_history
                ],
                index=[
                    snapshot["date"] for snapshot in portfolio_manager.portfolio_history
                ],
            ).sort_index()

            # 月度收益（月末权益）
            # pandas>=3.0: 'M' deprecated, use month-end 'ME'
            monthly_values = portfolio_values.resample("ME").last()
            monthly_returns = monthly_values.pct_change().dropna()

            if len(monthly_returns) > 0:
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
                                "return": float(ret),
                            }
                            for period, ret in monthly_returns.items()
                        ],
                    }
                )

            # 年度收益（年末权益）
            yearly_values = portfolio_values.resample("Y").last()
            yearly_returns = yearly_values.pct_change().dropna()

            if len(yearly_returns) > 0:
                additional_metrics["yearly_returns_detail"] = [
                    {
                        "year": period.year,
                        "return": float(ret),
                    }
                    for period, ret in yearly_returns.items()
                ]

            # --- 交易行为与个股表现 ---
            if portfolio_manager.trades:
                stock_performance: Dict[str, Dict[str, Any]] = {}

                # 辅助函数：统一访问 trade 属性（支持 Trade 对象和字典）
                def get_trade_attr(trade, attr: str):
                    if isinstance(trade, dict):
                        return trade.get(attr)
                    return getattr(trade, attr, None)

                for trade in portfolio_manager.trades:
                    stock_code = get_trade_attr(trade, 'stock_code')
                    action = get_trade_attr(trade, 'action')
                    pnl = get_trade_attr(trade, 'pnl') or 0.0

                    stock_stats = stock_performance.setdefault(
                        stock_code,
                        {
                            "stock_code": stock_code,
                            "total_pnl": 0.0,
                            "trade_count": 0,
                        },
                    )
                    stock_stats["trade_count"] += 1
                    # 只有卖出交易才有实现盈亏
                    if action == "SELL":
                        stock_stats["total_pnl"] += float(pnl)

                # 计算每只股票的平均单笔盈亏
                for stats in stock_performance.values():
                    trades = max(stats["trade_count"], 1)
                    stats["avg_pnl_per_trade"] = float(stats["total_pnl"]) / trades

                # 个股表现汇总
                stock_perf_list = list(stock_performance.values())
                additional_metrics.update(
                    {
                        "stock_performance_detail": stock_perf_list,
                        "best_performing_stock": max(
                            stock_perf_list, key=lambda x: x["total_pnl"]
                        )
                        if stock_perf_list
                        else None,
                        "worst_performing_stock": min(
                            stock_perf_list, key=lambda x: x["total_pnl"]
                        )
                        if stock_perf_list
                        else None,
                        "stocks_traded": len(stock_perf_list),
                    }
                )

                # 单笔交易分布的整体特征（便于前端画直方图/统计）
                pnls = [float(get_trade_attr(t, 'pnl') or 0.0) for t in portfolio_manager.trades]
                if pnls:
                    pnl_series = pd.Series(pnls)
                    additional_metrics.update(
                        {
                            "trade_pnl_mean": float(pnl_series.mean()),
                            "trade_pnl_median": float(pnl_series.median()),
                            "trade_pnl_std": float(pnl_series.std()),
                        }
                    )

        except Exception as exc:
            logger.error(f"计算额外指标失败: {exc}")

        return additional_metrics


