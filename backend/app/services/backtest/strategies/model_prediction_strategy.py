"""模型预测驱动策略。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import SignalType, TradingSignal
from .model_prediction_base import BaseModelPredictionStrategy


class ModelPredictionStrategy(BaseModelPredictionStrategy):
    """基于模型预测收益率序列生成阈值触发交易信号。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("ModelSignal", config)
        self.buy_threshold = float(config.get("buy_threshold", 0.01))
        self.sell_threshold = float(config.get("sell_threshold", -0.01))

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        prediction_series = self._get_prediction_series(data)
        if prediction_series is None:
            return None

        raw_states = prediction_series.apply(self._classify_signal_type)
        previous_states = raw_states.shift(1)
        transition_mask = raw_states.notna() & raw_states.ne(previous_states)
        return raw_states.where(transition_mask)

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        indicators = self.calculate_indicators(data)
        prediction_series = indicators["predicted_return"]
        if current_date not in prediction_series.index:
            return []

        predicted_return = prediction_series.loc[current_date]
        if pd.isna(predicted_return):
            return []

        signal_type = self._classify_signal_type(predicted_return)
        if signal_type is None:
            return []

        current_idx = self._get_current_idx(data, current_date)
        if current_idx < 0:
            current_idx = int(data.index.get_loc(current_date))
        if current_idx > 0:
            previous_return = prediction_series.iloc[current_idx - 1]
            previous_signal_type = self._classify_signal_type(previous_return)
            if previous_signal_type == signal_type:
                return []

        current_price = float(indicators["price"].iloc[current_idx])
        stock_code = data.attrs.get("stock_code", "UNKNOWN")

        return [
            TradingSignal(
                timestamp=current_date,
                stock_code=stock_code,
                signal_type=signal_type,
                strength=min(1.0, abs(float(predicted_return)) * 100),
                price=current_price,
                reason=f"模型 {self.model_id} 预测收益率 {float(predicted_return):.2%}",
                metadata={
                    "model_id": self.model_id,
                    "predicted_return": float(predicted_return),
                    "buy_threshold": self.buy_threshold,
                    "sell_threshold": self.sell_threshold,
                    "horizon": self.horizon,
                },
            )
        ]

    def _classify_signal_type(self, predicted_return: Any) -> Optional[SignalType]:
        if pd.isna(predicted_return):
            return None
        if predicted_return >= self.buy_threshold:
            return SignalType.BUY
        if predicted_return <= self.sell_threshold:
            return SignalType.SELL
        return None
