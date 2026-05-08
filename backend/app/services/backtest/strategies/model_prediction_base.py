"""Shared model-prediction backtest strategy helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, cast

import pandas as pd

from app.core.config import settings
from app.services.prediction.prediction_engine import PredictionConfig, PredictionEngine

from ..core.base_strategy import BaseStrategy


class BaseModelPredictionStrategy(BaseStrategy):
    """Common prediction-series loading/caching logic for model-driven strategies."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.model_id = cast(str, config.get("model_id"))
        if not self.model_id:
            raise ValueError(f"{name.lower()} 策略要求提供 model_id")

        self.horizon = config.get("horizon", "short_term")
        self.confidence_level = float(config.get("confidence_level", 0.95))
        self._prediction_cache_key = self.name

    async def prepare_backtest_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        engine = PredictionEngine(
            model_dir=str(settings.MODEL_STORAGE_PATH),
            data_dir=str(settings.DATA_ROOT_PATH),
        )
        prediction_config = PredictionConfig(
            model_id=self.model_id,
            horizon=self.horizon,
            confidence_level=self.confidence_level,
            risk_assessment=False,
        )

        for stock_code, data in stock_data.items():
            prediction_series = await engine.predict_return_series(
                stock_code=stock_code,
                config=prediction_config,
                start_date=start_date,
                end_date=end_date,
            )
            cache = data.attrs.setdefault("_model_prediction_returns", {})
            cache[self._prediction_cache_key] = prediction_series.sort_index()

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        prediction_series = self._get_prediction_series(data)
        return {
            "price": data["close"],
            "predicted_return": prediction_series,
        }

    def _get_prediction_series(self, data: pd.DataFrame) -> Optional[pd.Series]:
        cache = data.attrs.get("_model_prediction_returns", {})
        series = cache.get(self._prediction_cache_key)
        if series is None:
            return None

        if isinstance(series.index, pd.MultiIndex):
            stock_code = data.attrs.get("stock_code")
            if stock_code is not None:
                try:
                    if "instrument" in series.index.names:
                        series = series.xs(stock_code, level="instrument")
                    else:
                        series = series.xs(stock_code, level=0)
                except (KeyError, ValueError):
                    try:
                        series = series.droplevel(0)
                    except (KeyError, ValueError):
                        pass
            else:
                try:
                    series = series.droplevel(0)
                except (KeyError, ValueError):
                    pass

        return series.reindex(data.index)
