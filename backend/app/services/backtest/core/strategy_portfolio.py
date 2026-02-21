"""
策略组合类

管理多个策略实例，整合各策略信号并生成组合信号。
参考Backtrader的Cerebro设计思想。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
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

        使用 score-based 逻辑：每个子策略对每个日期计算连续分值(-1.0~+1.0)，
        加权求和后判断是否触发交易信号。

        这解决了旧逻辑中 RSI precompute 返回全零导致被排除的问题。
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            all_dates = data.index
            result = pd.Series(0.0, index=all_dates, dtype=np.float64)

            # 检查所有子策略是否支持 compute_score
            has_score = all(
                hasattr(s, 'compute_score') and callable(getattr(s, 'compute_score'))
                for s in self.strategies
            )

            if not has_score:
                logger.info("部分子策略不支持 compute_score，回退到旧逻辑")
                return self._precompute_signal_based(data)

            # === Score-based 预计算 ===
            total_weight = sum(
                self.weights.get(s.name, 1.0 / len(self.strategies))
                for s in self.strategies
            )
            if total_weight == 0:
                total_weight = 1.0

            score_threshold = 0.3
            if hasattr(self, 'config') and isinstance(self.config, dict):
                score_threshold = self.config.get("score_threshold", 0.3)

            for date in all_dates:
                weighted_sum = 0.0
                for strategy in self.strategies:
                    try:
                        score = strategy.compute_score(data, date)
                        w = self.weights.get(strategy.name, 1.0 / len(self.strategies))
                        weighted_sum += score * (w / total_weight)
                    except Exception:
                        continue

                if abs(weighted_sum) >= score_threshold:
                    result.loc[date] = max(-1.0, min(1.0, weighted_sum))

            buy_count = (result > 0).sum()
            sell_count = (result < 0).sum()
            logger.info(
                "Score-based precompute done: BUY=%d, SELL=%d, threshold=%.2f, strategies=%d",
                buy_count, sell_count, score_threshold, len(self.strategies)
            )

            return result

        except Exception as e:
            logger.error("Score-based precompute failed, fallback: %s", e)
            return self._precompute_signal_based(data)

    def _precompute_signal_based(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """旧的 signal-based 预计算逻辑（作为 fallback）"""
        import logging

        logger = logging.getLogger(__name__)

        try:
            sub_signals_map = {}
            for strategy in self.strategies:
                try:
                    sig_series = strategy.precompute_all_signals(data)
                    if sig_series is not None and len(sig_series) > 0:
                        sub_signals_map[strategy.name] = sig_series
                except Exception as e:
                    logger.warning("Sub-strategy %s precompute failed: %s", strategy.name, e)
                    continue

            if not sub_signals_map:
                return None

            all_dates = data.index
            result = pd.Series(0.0, index=all_dates, dtype=np.float64)

            total_weight = sum(
                self.weights.get(name, 1.0) for name in sub_signals_map.keys()
            )
            if total_weight == 0:
                total_weight = len(sub_signals_map)

            for date in all_dates:
                buy_score = 0.0
                sell_score = 0.0
                for strategy_name, sig_series in sub_signals_map.items():
                    weight = self.weights.get(strategy_name, 1.0) / total_weight
                    try:
                        if date in sig_series.index:
                            sig = sig_series.loc[date]
                            if isinstance(sig, (int, float)) and sig != 0 and not pd.isna(sig):
                                if sig > 0:
                                    buy_score += abs(float(sig)) * weight
                                else:
                                    sell_score += abs(float(sig)) * weight
                            elif sig == SignalType.BUY:
                                buy_score += weight
                            elif sig == SignalType.SELL:
                                sell_score += weight
                    except Exception:
                        continue

                consistency_threshold = 0.5
                if buy_score > sell_score and buy_score >= consistency_threshold:
                    result.loc[date] = min(1.0, buy_score)
                elif sell_score > buy_score and sell_score >= consistency_threshold:
                    result.loc[date] = -min(1.0, sell_score)

            return result
        except Exception as e:
            logger.error("Signal-based precompute failed: %s", e)
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
        import logging
        import time

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

                # 支持浮点信号（正=买入，负=卖出）
                if isinstance(sig_type, (int, float)) and sig_type != 0 and not pd.isna(sig_type):
                    fv = float(sig_type)
                    final_sig_type = SignalType.BUY if fv > 0 else SignalType.SELL
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    try:
                        current_price = float(data.loc[current_date, "close"])
                    except Exception:
                        current_price = 0.0

                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=final_sig_type,
                            strength=min(1.0, abs(fv)),
                            price=current_price,
                            reason=f"[向量化] 组合策略信号 ({len(self.strategies)} 子策略)",
                            metadata={
                                "strategy_name": self.name,
                                "source_strategy": self.name,
                                "sub_strategies": [s.name for s in self.strategies],
                            },
                        )
                    ]

                if isinstance(sig_type, SignalType):
                    stock_code = data.attrs.get("stock_code", "UNKNOWN")
                    # 获取当前价格
                    try:
                        current_price = float(data.loc[current_date, "close"])
                    except Exception:
                        current_price = 0.0

                    return [
                        TradingSignal(
                            timestamp=current_date,
                            stock_code=stock_code,
                            signal_type=sig_type,
                            strength=0.8,  # 组合策略默认强度（枚举信号无浮点强度）
                            price=current_price,
                            reason=f"[向量化] 组合策略信号 ({len(self.strategies)} 子策略)",
                            metadata={
                                "strategy_name": self.name,
                                "source_strategy": self.name,
                                "sub_strategies": [s.name for s in self.strategies],
                            },
                        )
                    ]
                return []
        except Exception as e:
            logger.debug(f"组合策略预计算信号查找失败: {e}")

        # === Score-based 整合：每个子策略计算连续分值，加权求和 ===
        try:
            scores = {}
            for strategy in self.strategies:
                score = strategy.compute_score(data, current_date)
                scores[strategy.name] = score

            # 加权求和
            weighted_sum = 0.0
            total_weight = 0.0
            for strategy in self.strategies:
                w = self.weights.get(strategy.name, 1.0 / len(self.strategies))
                weighted_sum += scores[strategy.name] * w
                total_weight += w

            final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

            # 阈值判断（默认 0.3，可通过 config 配置）
            score_threshold = 0.3
            if hasattr(self, 'config') and isinstance(self.config, dict):
                score_threshold = self.config.get("score_threshold", 0.3)

            if abs(final_score) >= score_threshold:
                sig_type = SignalType.BUY if final_score > 0 else SignalType.SELL
                strength = min(1.0, abs(final_score))
                stock_code = data.attrs.get("stock_code", "UNKNOWN")
                try:
                    current_price = float(data.loc[current_date, "close"])
                except Exception:
                    current_price = 0.0

                return [
                    TradingSignal(
                        timestamp=current_date,
                        stock_code=stock_code,
                        signal_type=sig_type,
                        strength=strength,
                        price=current_price,
                        reason=f"[score-based] 组合评分 {final_score:.3f} (阈值 {score_threshold})",
                        metadata={
                            "strategy_name": self.name,
                            "source_strategy": self.name,
                            "integration_method": "score_based",
                            "scores": scores,
                            "weighted_score": final_score,
                            "score_threshold": score_threshold,
                            "sub_strategies": [s.name for s in self.strategies],
                        },
                    )
                ]
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Score-based 整合失败，回退到信号整合: {e}")

        # === Fallback: 原有信号收集 + SignalIntegrator 整合 ===
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
        
        # regime_aware_voting 需要传入市场数据用于 regime 检测
        integrate_kwargs: Dict[str, Any] = {}
        if self.integrator.method == "regime_aware_voting":
            integrate_kwargs["market_data"] = data
        
        integrated_signals = self.integrator.integrate(
            all_signals, self.weights, consistency_threshold=0.6,
            **integrate_kwargs,
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
