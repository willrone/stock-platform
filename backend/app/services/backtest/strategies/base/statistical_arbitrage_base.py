"""
统计套利策略基类
"""

from typing import Any, Dict

import pandas as pd

from ...core.base_strategy import BaseStrategy


class StatisticalArbitrageStrategy(BaseStrategy):
    """统计套利策略基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.pairs = config.get("pairs", [])
        self.entry_threshold = config.get("entry_threshold", 2.0)
        self.exit_threshold = config.get("exit_threshold", 0.5)
        self.max_holding_period = config.get("max_holding_period", 60)
        self.stop_loss_threshold = config.get("stop_loss_threshold", 3.0)

    def calculate_spread_zscore(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """计算价差的Z-score"""
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
        zscore = (spread - mean) / std
        return zscore

    def validate_pair_correlation(
        self, data1: pd.DataFrame, data2: pd.DataFrame, min_corr: float = 0.7
    ) -> float:
        """验证配对的相关性"""
        returns1 = data1["close"].pct_change().dropna()
        returns2 = data2["close"].pct_change().dropna()

        min_len = min(len(returns1), len(returns2))
        correlation = returns1.iloc[-min_len:].corr(returns2.iloc[-min_len:])

        return correlation
