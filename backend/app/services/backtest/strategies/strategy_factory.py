"""
策略工厂

负责创建和管理所有策略实例，整合基础策略和高级策略
"""

from typing import Dict, Any, List

from ..core.base_strategy import BaseStrategy
from .technical.basic_strategies import MovingAverageStrategy, RSIStrategy, MACDStrategy
from .strategies import (
    # 技术分析策略
    BollingerBandStrategy,
    StochasticStrategy,
    CCIStrategy,
    
    # 统计套利策略
    PairsTradingStrategy,
    MeanReversionStrategy,
    CointegrationStrategy,
    
    # 因子投资策略
    ValueFactorStrategy,
    MomentumFactorStrategy,
    LowVolatilityStrategy,
    MultiFactorStrategy
)
from app.core.error_handler import TaskError, ErrorSeverity


class StrategyFactory:
    """统一的策略工厂，整合所有策略"""
    
    _strategies = {
        # 基础技术分析策略
        'moving_average': MovingAverageStrategy,
        'rsi': RSIStrategy,
        'macd': MACDStrategy,
        
        # 高级技术分析策略
        'bollinger': BollingerBandStrategy,
        'stochastic': StochasticStrategy,
        'cci': CCIStrategy,
        
        # 统计套利策略
        'pairs_trading': PairsTradingStrategy,
        'mean_reversion': MeanReversionStrategy,
        'cointegration': CointegrationStrategy,
        
        # 因子投资策略
        'value_factor': ValueFactorStrategy,
        'momentum_factor': MomentumFactorStrategy,
        'low_volatility': LowVolatilityStrategy,
        'multi_factor': MultiFactorStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> BaseStrategy:
        """创建策略实例"""
        strategy_name = strategy_name.lower()
        strategy_class = cls._strategies.get(strategy_name)
        
        if not strategy_class:
            available = list(cls._strategies.keys())
            raise TaskError(
                message=f"未知的策略类型: {strategy_name}，可用策略: {available}",
                severity=ErrorSeverity.MEDIUM
            )
        
        return strategy_class(config)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """获取可用策略列表"""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategies_by_category(cls) -> Dict[str, List[str]]:
        """按类别获取策略列表"""
        categories = {
            'technical': [
                'moving_average', 'rsi', 'macd',
                'bollinger', 'stochastic', 'cci'
            ],
            'statistical_arbitrage': [
                'pairs_trading', 'mean_reversion', 'cointegration'
            ],
            'factor_investment': [
                'value_factor', 'momentum_factor', 'low_volatility', 'multi_factor'
            ]
        }
        return categories
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """注册新策略"""
        cls._strategies[name.lower()] = strategy_class


# 为了向后兼容，保留 AdvancedStrategyFactory 作为别名
# 但推荐使用统一的 StrategyFactory
AdvancedStrategyFactory = StrategyFactory
