"""
数据模型
"""

# 导入所有模型以确保它们在 SQLAlchemy 的 Base.metadata 中注册
from app.models.user_models import User
from app.models.strategy_config_models import StrategyConfig
from app.models.task_models import Task, PredictionResult, BacktestResult, ModelInfo, ModelLifecycleEvent
from app.models.subscription_models import SubscriptionPlan, UserSubscription, UsageRecord
from app.models.backtest_detailed_models import (
    BacktestDetailedResult,
    BacktestChartCache,
    PortfolioSnapshot,
    TradeRecord,
    SignalRecord,
    BacktestBenchmark,
    BacktestStatistics,
)
from app.models.commerce_models import (
    UserBalance,
    UsageEvent,
    BillingRecord,
    CommercePricingRule,
)

__all__ = [
    # User models
    "User",
    # Strategy config
    "StrategyConfig",
    # Task models
    "Task",
    "PredictionResult",
    "BacktestResult",
    "ModelInfo",
    "ModelLifecycleEvent",
    # Subscription models
    "SubscriptionPlan",
    "UserSubscription",
    "UsageRecord",
    # Backtest detailed models
    "BacktestDetailedResult",
    "BacktestChartCache",
    "PortfolioSnapshot",
    "TradeRecord",
    "SignalRecord",
    "BacktestBenchmark",
    "BacktestStatistics",
    # Commerce models
    "UserBalance",
    "UsageEvent",
    "BillingRecord",
    "CommercePricingRule",
]
