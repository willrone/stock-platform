"""
高级策略工厂
"""

from typing import Any, Dict, List

from app.core.error_handler import ErrorSeverity, TaskError

from ..core.base_strategy import BaseStrategy
from .technical.bollinger_band import BollingerBandStrategy
from .technical.stochastic import StochasticStrategy
from .technical.cci import CCIStrategy
from .statistical_arbitrage.pairs_trading import PairsTradingStrategy
from .statistical_arbitrage.mean_reversion import MeanReversionStrategy
from .statistical_arbitrage.cointegration import CointegrationStrategy
from .factor.value_factor import ValueFactorStrategy
from .factor.momentum_factor import MomentumFactorStrategy
from .factor.low_volatility import LowVolatilityStrategy
from .factor.multi_factor import MultiFactorStrategy


class AdvancedStrategyFactory:
    """高级策略工厂"""

    _strategies = {
        # 技术分析策略
        "bollinger": BollingerBandStrategy,
        "stochastic": StochasticStrategy,
        "cci": CCIStrategy,
        # 统计套利策略
        "pairs_trading": PairsTradingStrategy,
        "mean_reversion": MeanReversionStrategy,
        "cointegration": CointegrationStrategy,
        # 因子投资策略
        "value_factor": ValueFactorStrategy,
        "momentum_factor": MomentumFactorStrategy,
        "low_volatility": LowVolatilityStrategy,
        "multi_factor": MultiFactorStrategy,
    }

    @classmethod
    def create_strategy(
        cls, strategy_name: str, config: Dict[str, Any]
    ) -> BaseStrategy:
        """创建策略实例"""
        strategy_name = strategy_name.lower()

        if strategy_name not in cls._strategies:
            raise TaskError(
                message=f"未知的策略类型: {strategy_name}，可用策略: {list(cls._strategies.keys())}",
                severity=ErrorSeverity.MEDIUM,
            )

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config)

    @classmethod
    def get_available_strategies(cls) -> Dict[str, List[str]]:
        """获取可用策略分类列表"""
        categories = {
            "technical": [],
            "statistical_arbitrage": [],
            "factor_investment": [],
        }

        for name in cls._strategies.keys():
            if name in ["bollinger", "stochastic", "cci"]:
                categories["technical"].append(name)
            elif name in ["pairs_trading", "mean_reversion", "cointegration"]:
                categories["statistical_arbitrage"].append(name)
            elif name in [
                "value_factor",
                "momentum_factor",
                "low_volatility",
                "multi_factor",
            ]:
                categories["factor_investment"].append(name)

        return categories

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type, category: str):
        """注册新策略"""
        cls._strategies[name.lower()] = strategy_class
