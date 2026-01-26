"""
信号整合器

负责整合多个策略的信号，通过加权投票、一致性增强等机制生成最终信号。
参考QuantConnect的信号融合算法。
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import SignalType, TradingSignal


class SignalIntegrator:
    """信号整合器 - 参考QuantConnect的信号融合算法"""

    def __init__(self, method: str = "weighted_voting"):
        """
        初始化信号整合器

        Args:
            method: 整合方法，支持 "weighted_voting" (加权投票)
        """
        self.method = method
        if method not in ["weighted_voting"]:
            raise ValueError(f"不支持的整合方法: {method}")

    def integrate(
        self,
        signals: List[TradingSignal],
        weights: Dict[str, float],
        consistency_threshold: float = 0.6,
    ) -> List[TradingSignal]:
        """
        整合多个策略的信号

        算法：
        1. 按股票分组信号
        2. 计算加权投票得分
        3. 应用一致性增强
        4. 解决冲突信号
        5. 生成最终信号

        Args:
            signals: 所有策略生成的信号列表
            weights: 策略权重字典 {strategy_name: weight}
            consistency_threshold: 一致性阈值，超过此阈值时增强信号强度

        Returns:
            整合后的信号列表
        """
        if not signals:
            return []

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("所有权重之和不能为0")
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # 按股票分组信号
        signals_by_stock = defaultdict(list)
        for signal in signals:
            signals_by_stock[signal.stock_code].append(signal)

        integrated_signals = []

        for stock_code, stock_signals in signals_by_stock.items():
            integrated_signal = self._integrate_stock_signals(
                stock_code, stock_signals, normalized_weights, consistency_threshold
            )
            if integrated_signal:
                integrated_signals.append(integrated_signal)

        return integrated_signals

    def _integrate_stock_signals(
        self,
        stock_code: str,
        signals: List[TradingSignal],
        weights: Dict[str, float],
        consistency_threshold: float,
    ) -> Optional[TradingSignal]:
        """
        整合单个股票的所有信号

        Args:
            stock_code: 股票代码
            signals: 该股票的所有信号
            weights: 归一化后的权重
            consistency_threshold: 一致性阈值

        Returns:
            整合后的信号，如果无法生成则返回None
        """
        if not signals:
            return None

        # 获取信号的时间戳（应该相同或接近）
        timestamps = [s.timestamp for s in signals]
        integrated_timestamp = max(timestamps)  # 使用最新的时间戳

        # 获取信号的价格（应该相同或接近）
        prices = [s.price for s in signals]
        integrated_price = prices[0]  # 使用第一个信号的价格

        # 计算加权投票得分和加权平均强度
        buy_score = 0.0
        sell_score = 0.0
        buy_weighted_strength_sum = 0.0  # 买入信号的加权强度总和
        sell_weighted_strength_sum = 0.0  # 卖出信号的加权强度总和
        buy_weight_sum = 0.0  # 买入信号的权重总和
        sell_weight_sum = 0.0  # 卖出信号的权重总和

        # 统计信号来源
        signal_sources = []

        for signal in signals:
            # 获取策略名称（从metadata或signal的reason中提取）
            strategy_name = self._extract_strategy_name(signal)

            # 获取权重（如果策略不在weights中，使用平均权重）
            weight = weights.get(strategy_name, 1.0 / len(weights) if weights else 1.0)

            # 计算加权强度
            weighted_strength = signal.strength * weight

            # 根据信号类型累加得分和强度
            if signal.signal_type == SignalType.BUY:
                buy_score += weighted_strength
                buy_weighted_strength_sum += weighted_strength
                buy_weight_sum += weight
            elif signal.signal_type == SignalType.SELL:
                sell_score += weighted_strength
                sell_weighted_strength_sum += weighted_strength
                sell_weight_sum += weight

            signal_sources.append(
                {
                    "strategy": strategy_name,
                    "type": signal.signal_type.name,
                    "strength": signal.strength,
                    "weight": weight,
                }
            )

        # 计算一致性（同向信号的比例）
        buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
        total_count = len(signals)

        consistency = (
            max(buy_count, sell_count) / total_count if total_count > 0 else 0.0
        )

        # 确定最终信号类型和强度
        if buy_score > sell_score:
            final_type = SignalType.BUY
            # 计算买入信号的加权平均强度
            final_strength = (
                buy_weighted_strength_sum / buy_weight_sum
                if buy_weight_sum > 0
                else 0.0
            )
        elif sell_score > buy_score:
            final_type = SignalType.SELL
            # 计算卖出信号的加权平均强度
            final_strength = (
                sell_weighted_strength_sum / sell_weight_sum
                if sell_weight_sum > 0
                else 0.0
            )
        else:
            # 得分相等或都为0，返回HOLD信号
            return None

        # 应用一致性增强
        if consistency >= consistency_threshold:
            # 一致性高时增强信号强度
            enhancement_factor = 1.0 + (consistency - consistency_threshold) * 0.5
            final_strength = min(1.0, final_strength * enhancement_factor)

        # 解决冲突：如果买入和卖出信号都存在且强度接近，降低最终信号强度
        if buy_count > 0 and sell_count > 0:
            conflict_ratio = min(buy_count, sell_count) / total_count
            final_strength *= 1.0 - conflict_ratio * 0.3  # 冲突时降低30%强度

        # 生成最终信号
        reasons = [s.reason for s in signals]
        integrated_reason = f"组合信号: {', '.join(set(reasons[:3]))}"  # 最多显示3个原因

        integrated_signal = TradingSignal(
            timestamp=integrated_timestamp,
            stock_code=stock_code,
            signal_type=final_type,
            strength=final_strength,
            price=integrated_price,
            reason=integrated_reason,
            metadata={
                "integration_method": self.method,
                "consistency": consistency,
                "buy_score": buy_score,
                "sell_score": sell_score,
                "source_signals": signal_sources,
                "total_strategies": len(signals),
            },
        )

        return integrated_signal

    def _extract_strategy_name(self, signal: TradingSignal) -> str:
        """
        从信号中提取策略名称

        Args:
            signal: 交易信号

        Returns:
            策略名称
        """
        # 优先从metadata中获取
        if signal.metadata and "strategy_name" in signal.metadata:
            return signal.metadata["strategy_name"]

        # 从reason中提取（如果格式为 "策略名: 原因"）
        if ":" in signal.reason:
            return signal.reason.split(":")[0].strip()

        # 默认返回"unknown"
        return "unknown"

    def normalize_signal_strength(
        self, signals: List[TradingSignal]
    ) -> List[TradingSignal]:
        """
        归一化信号强度到0-1范围

        Args:
            signals: 信号列表

        Returns:
            归一化后的信号列表
        """
        if not signals:
            return signals

        # 找到最大和最小强度
        strengths = [s.strength for s in signals]
        min_strength = min(strengths)
        max_strength = max(strengths)

        if max_strength == min_strength:
            # 所有强度相同，直接返回
            return signals

        # 归一化
        normalized_signals = []
        for signal in signals:
            normalized_strength = (signal.strength - min_strength) / (
                max_strength - min_strength
            )
            normalized_signal = TradingSignal(
                timestamp=signal.timestamp,
                stock_code=signal.stock_code,
                signal_type=signal.signal_type,
                strength=normalized_strength,
                price=signal.price,
                reason=signal.reason,
                metadata=signal.metadata,
            )
            normalized_signals.append(normalized_signal)

        return normalized_signals
