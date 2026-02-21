"""
策略工厂

负责创建和管理所有策略实例，整合基础策略和高级策略
"""

from typing import Any, Dict, List

from app.core.error_handler import ErrorSeverity, TaskError

from ..core.base_strategy import BaseStrategy
from ..core.strategy_portfolio import StrategyPortfolio
from .ml_ensemble_strategy import MLEnsembleLgbXgbRiskCtlStrategy
from .strategies import (  # 技术分析策略; 统计套利策略; 因子投资策略
    BollingerBandStrategy,
    CCIStrategy,
    CointegrationStrategy,
    LowVolatilityStrategy,
    MeanReversionStrategy,
    MomentumFactorStrategy,
    MultiFactorStrategy,
    PairsTradingStrategy,
    StochasticStrategy,
    ValueFactorStrategy,
)
from .technical.basic_strategies import MACDStrategy, MovingAverageStrategy, RSIStrategy


class StrategyFactory:
    """统一的策略工厂，整合所有策略"""

    _strategies = {
        # 基础技术分析策略
        "moving_average": MovingAverageStrategy,
        "rsi": RSIStrategy,
        "macd": MACDStrategy,
        # 高级技术分析策略
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
        # ML 集成策略
        "ml_ensemble_lgb_xgb_riskctl": MLEnsembleLgbXgbRiskCtlStrategy,
    }

    @classmethod
    def create_strategy(
        cls, strategy_name: str, config: Dict[str, Any]
    ) -> BaseStrategy:
        """
        创建策略实例（支持单策略和组合策略）

        Args:
            strategy_name: 策略名称，如果是"portfolio"或config中包含"strategies"则创建组合策略
            config: 策略配置

        Returns:
            策略实例（单策略或组合策略）
        """
        strategy_name = strategy_name.lower()

        # 检测是否为组合策略
        if strategy_name == "portfolio" or "strategies" in config:
            return cls._create_portfolio_strategy(config)
        else:
            # 创建单策略
            return cls._create_single_strategy(strategy_name, config)

    @classmethod
    def _create_single_strategy(
        cls, strategy_name: str, config: Dict[str, Any]
    ) -> BaseStrategy:
        """创建单策略实例"""
        strategy_class = cls._strategies.get(strategy_name)

        if not strategy_class:
            available = list(cls._strategies.keys())
            raise TaskError(
                message=f"未知的策略类型: {strategy_name}，可用策略: {available}",
                severity=ErrorSeverity.MEDIUM,
            )

        return strategy_class(config)

    @classmethod
    def _create_portfolio_strategy(cls, config: Dict[str, Any]) -> StrategyPortfolio:
        """
        创建策略组合

        Args:
            config: 组合策略配置，格式：
                {
                    "strategies": [
                        {
                            "name": "rsi",
                            "weight": 0.4,
                            "config": {"rsi_period": 14}
                        },
                        ...
                    ],
                    "integration_method": "weighted_voting"  # 可选
                }

        Returns:
            策略组合实例
        """
        # 兼容 sub_strategies 格式（前端传入）转换为 strategies 格式
        if "sub_strategies" in config and config["sub_strategies"]:
            ext_weights = config.get("weights", {})
            converted = []
            # 前端类名 → 工厂注册名 映射
            _name_map = {
                "RSI": "rsi", "MACD": "macd",
                "MeanReversion": "mean_reversion",
                "MovingAverage": "moving_average",
                "Bollinger": "bollinger", "BollingerBand": "bollinger",
                "Stochastic": "stochastic", "CCI": "cci",
                "PairsTrading": "pairs_trading",
                "Cointegration": "cointegration",
                "ValueFactor": "value_factor",
                "MomentumFactor": "momentum_factor",
                "LowVolatility": "low_volatility",
                "MultiFactor": "multi_factor",
            }
            for sc in config["sub_strategies"]:
                stype = sc.get("type", "")
                # 规范化策略名：先查映射表，再尝试小写
                stype_norm = _name_map.get(stype, stype.lower())
                sparams = sc.get("params", {})
                w = ext_weights.get(sc.get("type", ""), ext_weights.get(stype_norm, 1.0))
                converted.append({"name": stype_norm, "weight": w, "config": sparams})
            config = dict(config)
            config["strategies"] = converted

        # 兼容前端未传或传空 strategies 的情况，使用默认组合
        if "strategies" not in config or not config["strategies"]:
            config = dict(config)
            config["strategies"] = [
                {
                    "name": "bollinger",
                    "weight": 1.0,
                    "config": {"period": 20, "std_dev": 2, "entry_threshold": 0.02},
                },
                {
                    "name": "cci",
                    "weight": 1.0,
                    "config": {"period": 20, "oversold": -100, "overbought": 100},
                },
                {
                    "name": "macd",
                    "weight": 1.0,
                    "config": {
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                    },
                },
            ]

        strategy_configs = config["strategies"]
        if not isinstance(strategy_configs, list) or len(strategy_configs) == 0:
            raise TaskError(
                message="'strategies'必须是非空列表", severity=ErrorSeverity.MEDIUM
            )

        strategies = []
        weights = {}

        for strat_config in strategy_configs:
            if not isinstance(strat_config, dict):
                raise TaskError(
                    message=f"策略配置必须是字典，当前类型: {type(strat_config)}",
                    severity=ErrorSeverity.MEDIUM,
                )

            if "name" not in strat_config:
                raise TaskError(
                    message="策略配置中必须包含'name'字段", severity=ErrorSeverity.MEDIUM
                )

            name = strat_config["name"]
            weight = strat_config.get("weight", 1.0)
            strat_config_dict = strat_config.get("config", {})

            # 创建子策略
            try:
                strategy = cls._create_single_strategy(name, strat_config_dict)
                strategies.append(strategy)
                weights[strategy.name] = weight
            except Exception as e:
                raise TaskError(
                    message=f"创建策略 {name} 失败: {str(e)}", severity=ErrorSeverity.MEDIUM
                )

        # 获取整合方法
        integration_method = config.get("integration_method", "weighted_voting")

        # 创建策略组合
        return StrategyPortfolio(
            strategies=strategies,
            weights=weights,
            integration_method=integration_method,
        )

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """获取可用策略列表"""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategies_by_category(cls) -> Dict[str, List[str]]:
        """按类别获取策略列表"""
        categories = {
            "technical": [
                "moving_average",
                "rsi",
                "macd",
                "bollinger",
                "stochastic",
                "cci",
            ],
            "statistical_arbitrage": [
                "pairs_trading",
                "mean_reversion",
                "cointegration",
            ],
            "factor_investment": [
                "value_factor",
                "momentum_factor",
                "low_volatility",
                "multi_factor",
            ],
        }
        return categories

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """注册新策略"""
        cls._strategies[name.lower()] = strategy_class


# 为了向后兼容，保留 AdvancedStrategyFactory 作为别名
# 但推荐使用统一的 StrategyFactory
AdvancedStrategyFactory = StrategyFactory
