"""
因子投资策略基类
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...core.base_strategy import BaseStrategy


class FactorStrategy(BaseStrategy):
    """因子投资策略基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.lookback_period = config.get("lookback_period", 252)
        self.rebalance_frequency = config.get("rebalance_frequency", "monthly")
        self.market_cap_neutral = config.get("market_cap_neutral", False)
        self.industry_neutral = config.get("industry_neutral", False)
        self.max_position_size = config.get("max_position_size", 0.05)
        self.min_position_size = config.get("min_position_size", 0.01)

    def normalize_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """因子标准化"""
        normalized = factors.copy()

        for col in normalized.columns:
            if normalized[col].std() > 0:
                normalized[col] = (
                    normalized[col] - normalized[col].mean()
                ) / normalized[col].std()

        return normalized

    def apply_neutralization(
        self,
        scores: pd.Series,
        market_caps: Optional[pd.Series] = None,
        industries: Optional[pd.Series] = None,
    ) -> pd.Series:
        """应用中性化处理"""
        if market_caps is None and industries is None:
            return scores

        neutralized = scores.copy()

        if market_caps is not None and self.market_cap_neutral:
            log_mcap = np.log(market_caps)
            valid_idx = neutralized.index.intersection(log_mcap.index)
            if len(valid_idx) > 10:
                X = np.column_stack(
                    [np.ones(len(valid_idx)), log_mcap.loc[valid_idx].values]
                )
                y = neutralized.loc[valid_idx].values
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    neutralized.loc[valid_idx] = y - X @ beta
                except:
                    pass

        if industries is not None and self.industry_neutral:
            valid_idx = neutralized.index.intersection(industries.index)
            if len(valid_idx) > 10:
                unique_industries = industries.loc[valid_idx].unique()
                for ind in unique_industries:
                    ind_mask = industries.loc[valid_idx] == ind
                    if ind_mask.sum() > 5:
                        neutralized.loc[valid_idx[ind_mask]] -= neutralized.loc[
                            valid_idx[ind_mask]
                        ].mean()

        return neutralized
