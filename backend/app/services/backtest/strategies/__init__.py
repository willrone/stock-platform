"""
量化交易策略集合

包含以下策略类型：
1. 技术分析策略：布林带策略、随机指标策略、CCI策略
2. 统计套利策略：配对交易策略、均值回归策略、协整策略
3. 因子投资策略：价值因子策略、动量因子策略、低波动因子策略、多因子组合策略
"""

# 基类
from .base.statistical_arbitrage_base import StatisticalArbitrageStrategy
from .base.factor_base import FactorStrategy

# 技术分析策略
from .technical.bollinger_band import BollingerBandStrategy
from .technical.stochastic import StochasticStrategy
from .technical.cci import CCIStrategy

# 统计套利策略
from .statistical_arbitrage.pairs_trading import PairsTradingStrategy
from .statistical_arbitrage.mean_reversion import MeanReversionStrategy
from .statistical_arbitrage.cointegration import CointegrationStrategy

# 因子投资策略
from .factor.value_factor import ValueFactorStrategy
from .factor.momentum_factor import MomentumFactorStrategy
from .factor.low_volatility import LowVolatilityStrategy
from .factor.multi_factor import MultiFactorStrategy

# 工厂类
from .factory import AdvancedStrategyFactory

__all__ = [
    # 基类
    "StatisticalArbitrageStrategy",
    "FactorStrategy",
    # 技术分析策略
    "BollingerBandStrategy",
    "StochasticStrategy",
    "CCIStrategy",
    # 统计套利策略
    "PairsTradingStrategy",
    "MeanReversionStrategy",
    "CointegrationStrategy",
    # 因子投资策略
    "ValueFactorStrategy",
    "MomentumFactorStrategy",
    "LowVolatilityStrategy",
    "MultiFactorStrategy",
    # 工厂类
    "AdvancedStrategyFactory",
]
