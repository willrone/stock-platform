"""
指标注册机制
支持动态注册新指标，便于扩展
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from loguru import logger


class IndicatorCategory(Enum):
    """指标类别"""

    TECHNICAL = "technical"  # 技术指标
    ALPHA = "alpha"  # Alpha因子
    FUNDAMENTAL = "fundamental"  # 基本面特征
    BASE = "base"  # 基础统计指标


@dataclass
class IndicatorConfig:
    """指标配置"""

    name: str  # 指标名称
    category: IndicatorCategory  # 指标类别
    calculator_class: str  # 计算器类名
    calculator_method: str  # 计算方法名
    params: Dict[str, Any]  # 计算参数
    description: Optional[str] = None  # 指标描述
    version: str = "1.0.0"  # 指标版本
    enabled: bool = True  # 是否启用


class IndicatorRegistry:
    """指标注册表"""

    # 技术指标注册
    TECHNICAL_INDICATORS: Dict[str, IndicatorConfig] = {}

    # Alpha因子注册
    ALPHA_FACTORS: Dict[str, IndicatorConfig] = {}

    # 基本面特征注册
    FUNDAMENTAL_FEATURES: Dict[str, IndicatorConfig] = {}

    # 基础统计指标注册
    BASE_INDICATORS: Dict[str, IndicatorConfig] = {}

    @classmethod
    def register_indicator(
        cls,
        name: str,
        category: IndicatorCategory,
        calculator_class: str,
        calculator_method: str,
        params: Dict[str, Any] = None,
        description: Optional[str] = None,
        version: str = "1.0.0",
        enabled: bool = True,
    ):
        """
        注册新指标

        Args:
            name: 指标名称
            category: 指标类别
            calculator_class: 计算器类名（如 'TechnicalIndicatorCalculator'）
            calculator_method: 计算方法名（如 'calculate_rsi'）
            params: 计算参数
            description: 指标描述
            version: 指标版本
            enabled: 是否启用
        """
        config = IndicatorConfig(
            name=name,
            category=category,
            calculator_class=calculator_class,
            calculator_method=calculator_method,
            params=params or {},
            description=description,
            version=version,
            enabled=enabled,
        )

        if category == IndicatorCategory.TECHNICAL:
            cls.TECHNICAL_INDICATORS[name] = config
        elif category == IndicatorCategory.ALPHA:
            cls.ALPHA_FACTORS[name] = config
        elif category == IndicatorCategory.FUNDAMENTAL:
            cls.FUNDAMENTAL_FEATURES[name] = config
        elif category == IndicatorCategory.BASE:
            cls.BASE_INDICATORS[name] = config
        else:
            raise ValueError(f"不支持的指标类别: {category}")

        logger.info(f"注册指标: {name} ({category.value}), 版本: {version}")

    @classmethod
    def get_all_indicators(cls) -> Dict[str, IndicatorConfig]:
        """获取所有已注册的指标"""
        all_indicators = {}
        all_indicators.update(cls.TECHNICAL_INDICATORS)
        all_indicators.update(cls.ALPHA_FACTORS)
        all_indicators.update(cls.FUNDAMENTAL_FEATURES)
        all_indicators.update(cls.BASE_INDICATORS)
        return all_indicators

    @classmethod
    def get_indicators_by_category(
        cls, category: IndicatorCategory
    ) -> Dict[str, IndicatorConfig]:
        """按类别获取指标"""
        if category == IndicatorCategory.TECHNICAL:
            return cls.TECHNICAL_INDICATORS
        elif category == IndicatorCategory.ALPHA:
            return cls.ALPHA_FACTORS
        elif category == IndicatorCategory.FUNDAMENTAL:
            return cls.FUNDAMENTAL_FEATURES
        elif category == IndicatorCategory.BASE:
            return cls.BASE_INDICATORS
        else:
            return {}

    @classmethod
    def get_indicator_config(cls, name: str) -> Optional[IndicatorConfig]:
        """获取指定指标的配置"""
        all_indicators = cls.get_all_indicators()
        return all_indicators.get(name)

    @classmethod
    def is_indicator_registered(cls, name: str) -> bool:
        """检查指标是否已注册"""
        return name in cls.get_all_indicators()

    @classmethod
    def unregister_indicator(cls, name: str):
        """取消注册指标"""
        if name in cls.TECHNICAL_INDICATORS:
            del cls.TECHNICAL_INDICATORS[name]
        elif name in cls.ALPHA_FACTORS:
            del cls.ALPHA_FACTORS[name]
        elif name in cls.FUNDAMENTAL_FEATURES:
            del cls.FUNDAMENTAL_FEATURES[name]
        elif name in cls.BASE_INDICATORS:
            del cls.BASE_INDICATORS[name]
        else:
            logger.warning(f"指标 {name} 未注册，无法取消注册")

    @classmethod
    def enable_indicator(cls, name: str):
        """启用指标"""
        config = cls.get_indicator_config(name)
        if config:
            config.enabled = True
            logger.info(f"启用指标: {name}")
        else:
            logger.warning(f"指标 {name} 未注册，无法启用")

    @classmethod
    def disable_indicator(cls, name: str):
        """禁用指标"""
        config = cls.get_indicator_config(name)
        if config:
            config.enabled = False
            logger.info(f"禁用指标: {name}")
        else:
            logger.warning(f"指标 {name} 未注册，无法禁用")

    @classmethod
    def get_enabled_indicators(cls) -> Dict[str, IndicatorConfig]:
        """获取所有启用的指标"""
        all_indicators = cls.get_all_indicators()
        return {
            name: config for name, config in all_indicators.items() if config.enabled
        }


# 初始化默认指标注册
def _initialize_default_indicators():
    """初始化默认指标注册"""

    # 注册技术指标
    IndicatorRegistry.register_indicator(
        name="MA5",
        category=IndicatorCategory.TECHNICAL,
        calculator_class="TechnicalIndicatorCalculator",
        calculator_method="calculate_moving_average",
        params={"period": 5},
        description="5日移动平均线",
    )

    IndicatorRegistry.register_indicator(
        name="MA10",
        category=IndicatorCategory.TECHNICAL,
        calculator_class="TechnicalIndicatorCalculator",
        calculator_method="calculate_moving_average",
        params={"period": 10},
        description="10日移动平均线",
    )

    IndicatorRegistry.register_indicator(
        name="MA20",
        category=IndicatorCategory.TECHNICAL,
        calculator_class="TechnicalIndicatorCalculator",
        calculator_method="calculate_moving_average",
        params={"period": 20},
        description="20日移动平均线",
    )

    IndicatorRegistry.register_indicator(
        name="MA60",
        category=IndicatorCategory.TECHNICAL,
        calculator_class="TechnicalIndicatorCalculator",
        calculator_method="calculate_moving_average",
        params={"period": 60},
        description="60日移动平均线",
    )

    IndicatorRegistry.register_indicator(
        name="RSI14",
        category=IndicatorCategory.TECHNICAL,
        calculator_class="TechnicalIndicatorCalculator",
        calculator_method="calculate_rsi",
        params={"period": 14},
        description="14日相对强弱指数",
    )

    IndicatorRegistry.register_indicator(
        name="MACD",
        category=IndicatorCategory.TECHNICAL,
        calculator_class="TechnicalIndicatorCalculator",
        calculator_method="calculate_macd",
        params={},
        description="MACD指标",
    )

    IndicatorRegistry.register_indicator(
        name="BOLLINGER",
        category=IndicatorCategory.TECHNICAL,
        calculator_class="TechnicalIndicatorCalculator",
        calculator_method="calculate_bollinger_bands",
        params={},
        description="布林带指标",
    )

    # 注册基本面特征
    IndicatorRegistry.register_indicator(
        name="price_change",
        category=IndicatorCategory.FUNDAMENTAL,
        calculator_class="FeatureCalculator",
        calculator_method="calculate_fundamental_features",
        params={},
        description="价格变化率",
    )

    IndicatorRegistry.register_indicator(
        name="volatility_5d",
        category=IndicatorCategory.FUNDAMENTAL,
        calculator_class="FeatureCalculator",
        calculator_method="calculate_fundamental_features",
        params={},
        description="5日波动率",
    )

    # 注册Alpha158因子（批量注册）
    for i in range(1, 159):
        IndicatorRegistry.register_indicator(
            name=f"alpha_{i:03d}",
            category=IndicatorCategory.ALPHA,
            calculator_class="Alpha158Calculator",
            calculator_method="calculate_alpha_factors",
            params={},
            description=f"Alpha158因子 #{i}",
        )

    logger.info(
        f"默认指标注册完成: 技术指标 {len(IndicatorRegistry.TECHNICAL_INDICATORS)}, "
        f"Alpha因子 {len(IndicatorRegistry.ALPHA_FACTORS)}, "
        f"基本面特征 {len(IndicatorRegistry.FUNDAMENTAL_FEATURES)}"
    )


# 模块加载时初始化
_initialize_default_indicators()
