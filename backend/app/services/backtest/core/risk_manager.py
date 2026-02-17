"""
风险管理器

负责止损止盈检查和最大回撤熔断逻辑。
独立模块，遵循单一职责原则。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from ..models import BacktestConfig, SignalType, TradingSignal

# 命名常量
CIRCUIT_BREAKER_RECOVERY_RATIO = 0.5  # 回撤缩小到阈值的 50% 时恢复开仓
FULL_RETURN_RATIO = 1.0  # 100% 浮动收益/亏损基准


@dataclass
class CircuitBreakerEvent:
    """熔断事件记录"""

    timestamp: datetime
    event_type: str  # "triggered" 或 "recovered"
    drawdown_pct: float
    portfolio_value: float
    peak_value: float


class RiskManager:
    """风险管理器 - 止损止盈 + 最大回撤熔断"""

    def __init__(self, config: BacktestConfig):
        self.config = config

        # 熔断状态
        self.peak_portfolio_value: float = config.initial_cash
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_events: List[CircuitBreakerEvent] = []

    def check_stop_loss_take_profit(
        self,
        positions_with_prices: Dict[str, "PositionPriceInfo"],
    ) -> List[TradingSignal]:
        """
        检查所有持仓的止损止盈条件

        Args:
            positions_with_prices: 持仓信息字典
                key=stock_code, value=PositionPriceInfo

        Returns:
            需要触发的卖出信号列表
        """
        stop_loss = self.config.stop_loss_pct
        take_profit = self.config.take_profit_pct
        sl_enabled = _is_threshold_enabled(stop_loss)
        tp_enabled = _is_threshold_enabled(take_profit)

        if not sl_enabled and not tp_enabled:
            return []

        signals: List[TradingSignal] = []
        for code, info in positions_with_prices.items():
            signal = self._check_single_position(
                code,
                info,
                sl_enabled,
                tp_enabled,
            )
            if signal is not None:
                signals.append(signal)

        return signals

    def _check_single_position(
        self,
        stock_code: str,
        info: "PositionPriceInfo",
        sl_enabled: bool,
        tp_enabled: bool,
    ) -> Optional[TradingSignal]:
        """检查单只股票的止损止盈"""
        if info.quantity <= 0 or info.avg_cost <= 0:
            return None

        pnl_ratio = (info.current_price - info.avg_cost) / info.avg_cost

        # 止损检查（浮亏达到阈值）
        if sl_enabled and pnl_ratio <= -self.config.stop_loss_pct:
            logger.info(
                f"触发止损: {stock_code}, "
                f"亏损={pnl_ratio:.2%}, 阈值={-self.config.stop_loss_pct:.2%}"
            )
            return _build_sell_signal(stock_code, info, f"止损触发(亏损{pnl_ratio:.2%})")

        # 止盈检查（浮盈达到阈值）
        if tp_enabled and pnl_ratio >= self.config.take_profit_pct:
            logger.info(
                f"触发止盈: {stock_code}, "
                f"盈利={pnl_ratio:.2%}, 阈值={self.config.take_profit_pct:.2%}"
            )
            return _build_sell_signal(stock_code, info, f"止盈触发(盈利{pnl_ratio:.2%})")

        return None

    def update_circuit_breaker(
        self,
        portfolio_value: float,
        current_date: datetime,
    ) -> None:
        """
        更新最大回撤熔断状态

        Args:
            portfolio_value: 当前组合净值
            current_date: 当前日期
        """
        max_dd = self.config.max_drawdown_pct
        if not _is_threshold_enabled(max_dd):
            return

        # 更新历史最高净值
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        drawdown = _calculate_drawdown(
            portfolio_value,
            self.peak_portfolio_value,
        )

        if self.circuit_breaker_active:
            self._check_recovery(drawdown, portfolio_value, current_date)
        else:
            self._check_trigger(drawdown, portfolio_value, current_date)

    def _check_trigger(
        self,
        drawdown: float,
        value: float,
        date: datetime,
    ) -> None:
        """检查是否触发熔断"""
        if drawdown >= self.config.max_drawdown_pct:
            self.circuit_breaker_active = True
            event = CircuitBreakerEvent(
                timestamp=date,
                event_type="triggered",
                drawdown_pct=drawdown,
                portfolio_value=value,
                peak_value=self.peak_portfolio_value,
            )
            self.circuit_breaker_events.append(event)
            logger.warning(
                f"熔断触发: 回撤={drawdown:.2%}, " f"阈值={self.config.max_drawdown_pct:.2%}"
            )

    def _check_recovery(
        self,
        drawdown: float,
        value: float,
        date: datetime,
    ) -> None:
        """检查是否恢复开仓"""
        recovery_threshold = (
            self.config.max_drawdown_pct * CIRCUIT_BREAKER_RECOVERY_RATIO
        )
        if drawdown < recovery_threshold:
            self.circuit_breaker_active = False
            event = CircuitBreakerEvent(
                timestamp=date,
                event_type="recovered",
                drawdown_pct=drawdown,
                portfolio_value=value,
                peak_value=self.peak_portfolio_value,
            )
            self.circuit_breaker_events.append(event)
            logger.info(f"熔断恢复: 回撤={drawdown:.2%}, " f"恢复阈值={recovery_threshold:.2%}")

    def filter_signals_by_circuit_breaker(
        self,
        signals: List[TradingSignal],
    ) -> List[TradingSignal]:
        """
        熔断时过滤掉 BUY 信号，保留 SELL 信号

        Args:
            signals: 原始信号列表

        Returns:
            过滤后的信号列表
        """
        if not self.circuit_breaker_active:
            return signals

        filtered = [s for s in signals if s.signal_type != SignalType.BUY]
        blocked = len(signals) - len(filtered)
        if blocked > 0:
            logger.info(f"熔断生效: 阻止 {blocked} 个买入信号")
        return filtered

    def get_circuit_breaker_summary(self) -> Dict:
        """获取熔断统计摘要"""
        triggered = [
            e for e in self.circuit_breaker_events if e.event_type == "triggered"
        ]
        recovered = [
            e for e in self.circuit_breaker_events if e.event_type == "recovered"
        ]
        return {
            "total_triggers": len(triggered),
            "total_recoveries": len(recovered),
            "currently_active": self.circuit_breaker_active,
            "peak_portfolio_value": self.peak_portfolio_value,
            "events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.event_type,
                    "drawdown_pct": round(e.drawdown_pct, 4),
                    "portfolio_value": round(e.portfolio_value, 2),
                    "peak_value": round(e.peak_value, 2),
                }
                for e in self.circuit_breaker_events
            ],
        }


@dataclass
class PositionPriceInfo:
    """持仓价格信息（用于止损止盈检查）"""

    stock_code: str
    quantity: int
    avg_cost: float
    current_price: float
    timestamp: datetime = field(default_factory=datetime.now)


def _is_threshold_enabled(value: Optional[float]) -> bool:
    """判断阈值是否启用（非 None 且 > 0）"""
    return value is not None and value > 0


def _calculate_drawdown(current: float, peak: float) -> float:
    """计算回撤比例"""
    if peak <= 0:
        return 0.0
    return (peak - current) / peak


def _build_sell_signal(
    stock_code: str,
    info: "PositionPriceInfo",
    reason: str,
) -> TradingSignal:
    """构建止损/止盈卖出信号"""
    return TradingSignal(
        timestamp=info.timestamp,
        stock_code=stock_code,
        signal_type=SignalType.SELL,
        strength=FULL_RETURN_RATIO,
        price=info.current_price,
        reason=reason,
        metadata={"source": "risk_manager"},
    )
