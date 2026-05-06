"""
API v1 模块。

保持包入口轻量，避免导入单个路由模块时连带加载所有 API 与可选重依赖。
"""

__all__ = [
    "health",
    "stocks",
    "predictions",
    "tasks",
    "models",
    "backtest",
    "backtest_detailed",
    "backtest_websocket",
    "data",
    "system",
    "qlib",
    "infrastructure",
    "data_versioning",
    "features",
    "training_progress",
    "monitoring",
    "files",
    "strategy_configs",
    "optimization",
    "signals",
]
