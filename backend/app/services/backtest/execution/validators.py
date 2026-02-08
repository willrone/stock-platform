"""
参数验证和统计模块
"""

from typing import Any, Dict, List
from datetime import datetime
from loguru import logger

from ..strategies.factory import AdvancedStrategyFactory
from ....core.error_handler import TaskError
from ..models.data_models import TradingSignal
from ..core.portfolio_manager import PortfolioManager


def validate_backtest_parameters(
    self,
    strategy_name: str,
    stock_codes: List[str],
    start_date: datetime,
    end_date: datetime,
    strategy_config: Dict[str, Any],
) -> bool:
    """验证回测参数"""
    try:
        # 验证策略名称
        available_strategies = AdvancedStrategyFactory.get_available_strategies()
        if strategy_name.lower() not in available_strategies:
            raise TaskError(
                message=f"不支持的策略: {strategy_name}，可用策略: {available_strategies}",
                severity=ErrorSeverity.MEDIUM,
            )

        # 验证股票代码
        if not stock_codes or len(stock_codes) == 0:
            raise TaskError(message="股票代码列表不能为空", severity=ErrorSeverity.MEDIUM)

        if len(stock_codes) > 1000:
            raise TaskError(
                message=f"股票数量过多: {len(stock_codes)}，最多支持1000只股票",
                severity=ErrorSeverity.MEDIUM,
            )

        # 验证日期范围
        if start_date >= end_date:
            raise TaskError(message="开始日期必须早于结束日期", severity=ErrorSeverity.MEDIUM)

        date_range = (end_date - start_date).days
        if date_range < 30:
            raise TaskError(
                message=f"回测期间太短: {date_range}天，至少需要30天",
                severity=ErrorSeverity.MEDIUM,
            )

        if date_range > 3650:  # 10年
            raise TaskError(
                message=f"回测期间太长: {date_range}天，最多支持10年",
                severity=ErrorSeverity.MEDIUM,
            )

        # 验证策略配置
        if not isinstance(strategy_config, dict):
            raise TaskError(message="策略配置必须是字典格式", severity=ErrorSeverity.MEDIUM)

        return True

    except TaskError:
        raise
    except Exception as e:
        raise TaskError(
            message=f"参数验证失败: {str(e)}",
            severity=ErrorSeverity.MEDIUM,
            original_exception=e,
        )


def _get_execution_failure_reason(
    self,
    signal: TradingSignal,
    portfolio_manager: PortfolioManager,
    current_prices: Dict[str, float],
) -> str:
    """
    获取执行失败的原因

    Args:
        signal: 交易信号
        portfolio_manager: 组合管理器
        current_prices: 当前价格

    Returns:
        失败原因字符串
    """
    try:
        stock_code = signal.stock_code
        current_price = current_prices.get(stock_code, signal.price)

        if signal.signal_type == SignalType.BUY:
            # 买入失败的可能原因（逻辑与 _execute_buy 保持一致）
            # 计算组合价值（使用与 _execute_buy 相同的逻辑）
            portfolio_value = portfolio_manager.get_portfolio_value(
                {stock_code: current_price}
            )
            max_position_value = (
                portfolio_value * portfolio_manager.config.max_position_size
            )

            current_position = portfolio_manager.positions.get(stock_code)
            current_position_value = (
                current_position.market_value if current_position else 0
            )

            available_cash_for_stock = max_position_value - current_position_value
            available_cash_for_stock = min(
                available_cash_for_stock, portfolio_manager.cash * 0.95
            )  # 保留5%现金

            if available_cash_for_stock <= 0:
                if (
                    current_position_value > 0
                    and current_position_value >= max_position_value
                ):
                    return f"已达到最大持仓限制: 当前持仓 {current_position_value:.2f} >= 最大持仓 {max_position_value:.2f}"
                else:
                    return f"可用资金不足: 需要保留5%现金，可用资金 {portfolio_manager.cash:.2f}"

            # 计算购买数量（最小交易单位为100股）
            quantity = int(available_cash_for_stock / current_price / 100) * 100
            if quantity <= 0:
                return f"可买数量不足: 可用资金 {available_cash_for_stock:.2f}，价格 {current_price:.2f}，无法买入100股"

            # 计算实际成本（包含手续费和滑点）
            # 应用滑点（买入时价格上涨）
            execution_price = current_price * (
                1 + portfolio_manager.config.slippage_rate
            )
            slippage_cost_per_share = (
                current_price * portfolio_manager.config.slippage_rate
            )

            total_cost = quantity * execution_price
            commission = total_cost * portfolio_manager.config.commission_rate
            slippage_cost = quantity * slippage_cost_per_share
            total_cost_with_all_fees = total_cost + commission

            if total_cost_with_all_fees > portfolio_manager.cash:
                return f"资金不足: 需要 {total_cost_with_all_fees:.2f}（含手续费 {commission:.2f}），可用 {portfolio_manager.cash:.2f}"

            # 如果所有检查都通过但还是失败了，可能是其他原因
            return f"执行失败: 可能因滑点成本 {slippage_cost:.2f} 或其他限制"

        elif signal.signal_type == SignalType.SELL:
            # 卖出失败的可能原因
            if stock_code not in portfolio_manager.positions:
                return "无持仓"

            position = portfolio_manager.positions[stock_code]
            if position.quantity <= 0:
                return "持仓数量为0"

            # 如果所有检查都通过但还是失败了，可能是其他原因
            return "执行失败（未知原因）"

        return "未知信号类型"

    except Exception as e:
        logger.warning(f"获取执行失败原因时出错: {e}")
        return f"执行异常: {str(e)}"

def get_execution_statistics(execution_stats: dict) -> dict:
    """获取执行统计信息"""
    return {
        "total_backtests": execution_stats.get("total_backtests", 0),
        "successful_backtests": execution_stats.get("successful_backtests", 0),
        "failed_backtests": execution_stats.get("failed_backtests", 0),
    }
