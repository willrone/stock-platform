"""Official-style model ranking strategy using TopK + Dropout portfolio execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from ..models import SignalType, TradingSignal
from .model_prediction_base import BaseModelPredictionStrategy


class ModelTopkDropoutStrategy(BaseModelPredictionStrategy):
    """Emit daily ranking-score signals and delegate execution to TopK/Dropout trade mode."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("ModelTopkDropout", config)
        self.topk = int(config.get("topk", 10))
        self.n_drop = int(config.get("n_drop", 2))
        self.hold_thresh = int(config.get("hold_thresh", config.get("buffer", 0) or 0))
        self.benchmark = config.get("benchmark")
        self.deal_price = str(config.get("deal_price", "close"))
        self.score_scale = float(config.get("score_scale", 100.0))

        if self.topk <= 0:
            raise ValueError("model_topk_dropout 策略要求 topk > 0")
        if self.n_drop <= 0:
            raise ValueError("model_topk_dropout 策略要求 n_drop > 0")

    def get_trade_mode(self) -> str:
        return "topk_dropout"

    def get_trade_mode_config(self) -> Dict[str, Any]:
        return {
            "topk": self.topk,
            "n_drop": self.n_drop,
            "hold_thresh": self.hold_thresh,
            "benchmark": self.benchmark,
            "deal_price": self.deal_price,
        }

    def precompute_all_signals(self, data: pd.DataFrame) -> None:
        """Ranking strategies need raw daily scores, not discrete threshold states."""
        return None

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        indicators = self.calculate_indicators(data)
        prediction_series = indicators["predicted_return"]
        if prediction_series is None or current_date not in prediction_series.index:
            return []

        predicted_return = prediction_series.loc[current_date]
        if pd.isna(predicted_return):
            return []

        current_idx = self._get_current_idx(data, current_date)
        if current_idx < 0:
            current_idx = int(data.index.get_loc(current_date))

        current_price = float(indicators["price"].iloc[current_idx])
        ranking_score = float(predicted_return)
        stock_code = data.attrs.get("stock_code", "UNKNOWN")

        return [
            TradingSignal(
                timestamp=current_date,
                stock_code=stock_code,
                signal_type=SignalType.BUY,
                strength=min(1.0, abs(ranking_score) * self.score_scale),
                price=current_price,
                reason=(
                    f"模型 {self.model_id} 排名分数 {ranking_score:.4%} "
                    f"(TopK={self.topk}, n_drop={self.n_drop})"
                ),
                metadata={
                    "model_id": self.model_id,
                    "predicted_return": ranking_score,
                    "ranking_score": ranking_score,
                    "signal_role": "ranking_score",
                    "trade_mode": self.get_trade_mode(),
                    "topk": self.topk,
                    "n_drop": self.n_drop,
                    "hold_thresh": self.hold_thresh,
                    "benchmark": self.benchmark,
                    "deal_price": self.deal_price,
                    "horizon": self.horizon,
                },
            )
        ]
