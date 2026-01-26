"""
预测引擎模块

该模块包含所有与股票预测、风险评估和技术分析相关的服务，包括：
- 预测引擎和预测回退机制
- 风险评估和置信区间计算
- 特征提取和技术指标计算
- 预测错误处理和重试机制

主要组件：
- PredictionEngine: 主预测引擎
- PredictionFallbackEngine: 预测回退机制
- RiskAssessmentService: 风险评估服务
- FeatureExtractor: 特征提取器
- TechnicalIndicatorCalculator: 技术指标计算器
"""

# 特征提取
from .feature_extractor import (
    FeatureCache,
    FeatureConfig,
    FeatureExtractor,
    StatisticalFeatures,
    TechnicalIndicators,
)

# 预测引擎
from .prediction_engine import (
    ModelLoader,
    PredictionConfig,
    PredictionEngine,
    PredictionOutput,
)
from .prediction_engine import RiskAssessment as EngineRiskAssessment

# 预测回退
from .prediction_fallback import (
    FallbackConfig,
    FallbackStrategy,
    PredictionErrorHandler,
    PredictionFallbackEngine,
    RetryConfig,
    RetryManager,
)

# 风险评估
from .risk_assessment import (
    ConfidenceInterval,
    ConfidenceIntervalCalculator,
    RiskAssessmentConfig,
    RiskAssessmentResult,
    RiskAssessmentService,
    RiskMetricsCalculator,
    ScenarioAnalysis,
)

# 技术指标
from .technical_indicators import (
    BatchIndicatorRequest,
    BatchIndicatorResponse,
    TechnicalIndicatorCalculator,
    TechnicalIndicatorResult,
)

__all__ = [
    # 预测引擎
    "PredictionEngine",
    "ModelLoader",
    "EngineRiskAssessment",
    "PredictionConfig",
    "PredictionOutput",
    # 预测回退
    "PredictionFallbackEngine",
    "RetryManager",
    "PredictionErrorHandler",
    "FallbackStrategy",
    "FallbackConfig",
    "RetryConfig",
    # 风险评估
    "RiskAssessmentService",
    "ConfidenceIntervalCalculator",
    "RiskMetricsCalculator",
    "ScenarioAnalysis",
    "RiskAssessmentConfig",
    "ConfidenceInterval",
    "RiskAssessmentResult",
    # 特征提取
    "FeatureExtractor",
    "TechnicalIndicators",
    "StatisticalFeatures",
    "FeatureCache",
    "FeatureConfig",
    # 技术指标
    "TechnicalIndicatorCalculator",
    "TechnicalIndicatorResult",
    "BatchIndicatorRequest",
    "BatchIndicatorResponse",
]
