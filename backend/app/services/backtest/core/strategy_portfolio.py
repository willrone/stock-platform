"""
策略组合类

管理多个策略实例，整合各策略信号并生成组合信号。
参考Backtrader的Cerebro设计思想。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import SignalType, TradingSignal
from ..utils.signal_integrator import SignalIntegrator
from .base_strategy import BaseStrategy


class StrategyPortfolio(BaseStrategy):
    """策略组合类 - 参考Backtrader的Cerebro设计思想"""

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: Optional[Dict[str, float]] = None,
        integration_method: str = "weighted_voting",
        name: Optional[str] = None,
    ):
        """
        初始化策略组合

        Args:
            strategies: 策略实例列表
            weights: 策略权重字典 {strategy_name: weight}，如果为None则使用平均权重
            integration_method: 信号整合方法
            name: 组合名称，如果为None则自动生成
        """
        # 验证策略列表
        if not strategies:
            raise ValueError("策略列表不能为空")

        self.strategies = strategies
        self.weights = weights or self._default_weights()

        # 验证权重
        self._validate_weights()

        # 归一化权重
        self._normalize_weights()

        # 创建信号整合器
        self.integrator = SignalIntegrator(integration_method)

        # 设置组合名称
        if name is None:
            strategy_names = [s.name for s in strategies]
            name = f"Portfolio({', '.join(strategy_names)})"

        # 初始化基类
        super().__init__(
            name,
            {
                "strategies": [s.name for s in strategies],
                "weights": self.weights,
                "integration_method": integration_method,
            },
        )

    def _default_weights(self) -> Dict[str, float]:
        """生成默认权重（平均分配）"""
        weight = 1.0 / len(self.strategies)
        return {strategy.name: weight for strategy in self.strategies}

    def _validate_weights(self):
        """验证权重配置"""
        # 检查所有权重是否非负
        for weight in self.weights.values():
            if weight < 0:
                raise ValueError(f"权重不能为负: {weight}")

        # 检查策略名称是否匹配
        strategy_names = {s.name for s in self.strategies}
        weight_names = set(self.weights.keys())

        if not weight_names.issubset(strategy_names):
            missing = weight_names - strategy_names
            raise ValueError(f"权重配置中包含不存在的策略: {missing}")

    def _normalize_weights(self):
        """归一化权重，确保所有权重之和为1.0"""
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            raise ValueError("所有权重之和不能为0")

        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        [性能优化] 向量化预计算组合策略的整合信号
        
        对于组合策略，需要：
        1. 收集所有子策略的预计算信号
        2. 按日期进行向量化的加权投票整合
        3. 返回整合后的信号序列
        
        Args:
            data: 股票数据 DataFrame
            
        Returns:
            整合后的信号 Series，index 为日期，值为 SignalType 或 None
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # 1. 收集所有子策略的预计算信号
            sub_signals_map: Dict[str, pd.Series] = {}
            
            for strategy in self.strategies:
                try:
                    sig_series = strategy.precompute_all_signals(data)
                    if sig_series is not None and len(sig_series) > 0:
                        sub_signals_map[strategy.name] = sig_series
                except Exception as e:
                    logger.warning(f"子策略 {strategy.name} 预计算失败: {e}")
                    continue
            
            if not sub_signals_map:
                logger.warning("组合策略: 所有子策略预计算均失败，返回 None")
                return None
            
            # 2. 向量化整合信号（按日期加权投票）
            # 获取所有日期的并集
            all_dates = data.index
            
            # 初始化结果
            result = pd.Series([None] * len(all_dates), index=all_dates, dtype=object)
            
            # 归一化权重
            total_weight = sum(self.weights.get(name, 1.0) for name in sub_signals_map.keys())
            if total_weight == 0:
                total_weight = len(sub_signals_map)
            
            # 对每个日期进行加权投票
            for date in all_dates:
                buy_score = 0.0
                sell_score = 0.0
                
                for strategy_name, sig_series in sub_signals_map.items():
                    weight = self.weights.get(strategy_name, 1.0) / total_weight
                    
                    try:
                        if date in sig_series.index:
                            sig = sig_series.loc[date]
                            if sig == SignalType.BUY:
                                buy_score += weight
                            elif sig == SignalType.SELL:
                                sell_score += weight
                    except Exception:
                        continue
                
                # 根据加权投票结果确定最终信号
                # 使用一致性阈值 0.3（至少30%的权重同意）
                consistency_threshold = 0.3
                if buy_score > sell_score and buy_score >= consistency_threshold:
                    result.loc[date] = SignalType.BUY
                elif sell_score > buy_score and sell_score >= consistency_threshold:
                    result.loc[date] = SignalType.SELL
            
            # 统计信号数量
            buy_count = (result == SignalType.BUY).sum()
            sell_count = (result == SignalType.SELL).sum()
            logger.info(f"✅ 组合策略向量化预计算完成: BUY={buy_count}, SELL={sell_count}, 子策略数={len(sub_signals_map)}")
            
            return result
            
        except Exception as e:
            logger.error(f"组合策略向量化预计算失败: {e}")
            return None

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """
        生成组合信号

        流程：
        1. 优先使用预计算信号（如果存在）
        2. 否则收集所有策略的信号
        3. 为每个信号添加策略名称到metadata
        4. 使用SignalIntegrator整合信号
        5. 返回最终信号

        Args:
            data: 股票数据
            current_date: 当前日期

        Returns:
            整合后的信号列表
        """
        import time
        import logging
        logger = logging.getLogger(__name__)

        # 性能优化：优先使用预计算信号
        try:
            precomputed = data.attrs.get("_precomputed_signals", {}).get(self.name)
            if precomputed is not None:
                sig_type = None
                if isinstance(precomputed, pd.Series):
                    sig_type = precomputed.get(current_date)
                elif isinstance(precomputed, dict):
                    sig_type = precomputed.get(current_date)
                
                if isinstance(sig_type, SignalType):
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    # 获取当前价格
                    try:
                        current_price = float(data.loc[current_date, 'close'])
                    except Exception:
                        current_price = 0.0
                    
                    return [TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=sig_type,
                        strength=0.8,  # 组合策略默认强度
                        price=current_price,
                        reason=f"[向量化] 组合策略信号 ({len(self.strategies)} 子策略)",
                        metadata={
                            "strategy_name": self.name,
                            "source_strategy": self.name,
                            "sub_strategies": [s.name for s in self.strategies],
                        },
                    )]
                return []
        except Exception as e:
            logger.debug(f"组合策略预计算信号查找失败: {e}")

        all_signals = []

        # profiling: collect per-sub-strategy timings (seconds)
        sub_strategy_times: Dict[str, float] = {}

        # 收集所有策略的信号
        for strategy in self.strategies:
            try:
                t0 = time.perf_counter()
                signals = strategy.generate_signals(data, current_date)
                sub_strategy_times[strategy.name] = time.perf_counter() - t0

                # 为每个信号添加策略名称到metadata
                for signal in signals:
                    if signal.metadata is None:
                        signal.metadata = {}
                    signal.metadata["strategy_name"] = strategy.name
                    signal.metadata["source_strategy"] = strategy.name

                all_signals.extend(signals)
            except Exception as e:
                # 如果某个策略生成信号失败，记录错误但不影响其他策略
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"策略 {strategy.name} 生成信号失败: {e}")
                continue

        # 如果没有信号，返回空列表
        if not all_signals:
            return []

        # 使用信号整合器整合信号
        t_int = time.perf_counter()
        integrated_signals = self.integrator.integrate(
            all_signals, self.weights, consistency_threshold=0.6
        )
        integrate_time = time.perf_counter() - t_int

        # Attach lightweight perf summary to the first integrated signal (for backtest-level profiling)
        if integrated_signals:
            sig0 = integrated_signals[0]
            if sig0.metadata is None:
                sig0.metadata = {}
            sig0.metadata["portfolio_perf"] = {
                "sub_strategy_times": sub_strategy_times,
                "integrate_time": integrate_time,
            }

        return integrated_signals

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算技术指标

        对于组合策略，返回所有子策略的指标（合并）

        Args:
            data: 股票数据

        Returns:
            指标字典
        """
        all_indicators = {}

        for strategy in self.strategies:
            try:
                indicators = strategy.calculate_indicators(data)
                # 为指标添加策略前缀，避免冲突
                for key, value in indicators.items():
                    prefixed_key = f"{strategy.name}_{key}"
                    all_indicators[prefixed_key] = value
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"策略 {strategy.name} 计算指标失败: {e}")
                continue

        return all_indicators

    def add_strategy(self, strategy: BaseStrategy, weight: Optional[float] = None):
        """
        动态添加策略

        Args:
            strategy: 策略实例
            weight: 策略权重，如果为None则使用平均权重
        """
        if strategy.name in [s.name for s in self.strategies]:
            raise ValueError(f"策略 {strategy.name} 已存在于组合中")

        self.strategies.append(strategy)

        if weight is not None:
            self.weights[strategy.name] = weight
        else:
            # 重新平均分配权重
            self.weights = self._default_weights()

        self._normalize_weights()

    def remove_strategy(self, strategy_name: str):
        """
        移除策略

        Args:
            strategy_name: 策略名称
        """
        self.strategies = [s for s in self.strategies if s.name != strategy_name]

        if strategy_name in self.weights:
            del self.weights[strategy_name]

        if not self.strategies:
            raise ValueError("组合中至少需要保留一个策略")

        # 重新归一化权重
        self._normalize_weights()

    def update_weights(self, weights: Dict[str, float]):
        """
        更新策略权重

        Args:
            weights: 新的权重字典
        """
        self.weights.update(weights)
        self._validate_weights()
        self._normalize_weights()

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略组合信息

        Returns:
            组合信息字典
        """
        return {
            "name": self.name,
            "strategy_count": len(self.strategies),
            "strategies": [
                {
                    "name": s.name,
                    "weight": self.weights.get(s.name, 0.0),
                    "config": s.config,
                }
                for s in self.strategies
            ],
            "integration_method": self.integrator.method,
            "total_weight": sum(self.weights.values()),
        }
