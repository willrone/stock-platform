"""
回测执行器 - 完整的回测流程执行和结果分析
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from app.core.error_handler import ErrorContext, ErrorSeverity, TaskError
from app.models.task_models import BacktestResult

from ..core.base_strategy import BaseStrategy
from ..core.portfolio_manager import PortfolioManager
from ..core.portfolio_manager_array import PortfolioManagerArray
from ..models import BacktestConfig, Position, SignalType, Trade, TradingSignal
from ..reporting import BacktestReportBuildInput, BacktestReportBuilder
from ..strategies.strategy_factory import AdvancedStrategyFactory, StrategyFactory
from .backtest_progress_monitor import backtest_progress_monitor
from .data_loader import DataLoader
from .trade_modes import TradeModeExecutionContext, get_trade_mode_executor

# 性能监控（可选导入，避免依赖问题）
try:
    from ..utils.performance_profiler import (
        BacktestPerformanceProfiler,
        PerformanceContext,
    )

    PERFORMANCE_PROFILING_AVAILABLE = True
except ImportError:
    PERFORMANCE_PROFILING_AVAILABLE = False
    BacktestPerformanceProfiler = None
    PerformanceContext = None


def _multiprocess_precompute_worker(task: Tuple) -> Tuple[bool, str, Optional[Dict], Optional[str]]:
    """
    多进程预计算 worker 函数（模块级，可被 pickle 序列化）。

    Args:
        task: (stock_code, data_dict, strategy_info) 元组

    Returns:
        (success, stock_code, signals_dict, error_message)
    """
    stock_code, data_dict, strategy_info = task

    try:
        # 重建 DataFrame
        df = pd.DataFrame(data_dict['values'], columns=data_dict['columns'])
        df.index = pd.to_datetime(data_dict['index'])
        df.attrs['stock_code'] = data_dict['stock_code']

        # 重建策略对象
        from ..strategies.strategy_factory import StrategyFactory, AdvancedStrategyFactory

        strategy_name = strategy_info['name']  # 使用策略名称（如 "MACD"）
        strategy_class_name = strategy_info['class_name']  # 类名（如 "MACDStrategy"）
        strategy_config = strategy_info['config']

        # 尝试从工厂创建策略（尝试多种名称格式）
        strategy = None
        names_to_try = [
            strategy_name,  # 原始名称
            strategy_name.lower(),  # 小写
            strategy_class_name,  # 类名
            strategy_class_name.replace('Strategy', ''),  # 去掉 Strategy 后缀
            strategy_class_name.replace('Strategy', '').lower(),  # 去掉后缀并小写
        ]

        for name in names_to_try:
            if strategy is not None:
                break
            try:
                strategy = StrategyFactory.create_strategy(name, strategy_config)
            except Exception:
                try:
                    strategy = AdvancedStrategyFactory.create_strategy(name, strategy_config)
                except Exception:
                    pass

        if strategy is None:
            return (False, stock_code, None, f"无法创建策略 {strategy_name} (尝试了: {names_to_try})")

        # 执行向量化预计算
        signals = strategy.precompute_all_signals(df)

        if signals is not None:
            # 将 Series 转换为可序列化格式
            signals_dict = {
                'values': signals.tolist(),
                'index': [str(idx) for idx in signals.index],
            }
            return (True, stock_code, signals_dict, None)
        else:
            return (False, stock_code, None, "precompute_all_signals 返回 None")

    except Exception as e:
        logger.error(f"业务逻辑错误：{e}", extra={"error_type": "DOMAIN", "error_code": "BUSINESS_ERROR"})
        from app.core.errors import DomainError
        raise DomainError(message=f"业务逻辑错误：{e}", context="unknown", details={"original_error": str(e)}) from e

            message=f"业务逻辑错误：{e}",


            context="unknown",


            details={"original_error": str(e)},


        ) from e