"""LightGBM 模型驱动回测策略。

直接加载保存的 LightGBM 模型，批量生成预测信号，通过回测框架验证夏普。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.backtest.models import SignalType, TradingSignal
from app.services.backtest.strategies.model_prediction_base import BaseModelPredictionStrategy


class LightGBMPredictionStrategy(BaseModelPredictionStrategy):
    """基于 LightGBM 模型的截面选股+多空信号回测策略。

    prepare_backtest_data 时加载 LightGBM 模型，对每只股票生成连续预测信号，
    最终回调测引擎的 generate_signals 做截面选股。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("LightGBM", config)

        # LightGBM 模型路径
        self.lgb_model_path = cast(str, config.get("lgb_model_path", ""))
        if not self.lgb_model_path:
            # 默认路径
            self.lgb_model_path = str(settings.MODEL_STORAGE_PATH / "lgb_model.txt")

        # 策略参数
        self.top_k = int(config.get("top_k", 5))
        self.buy_threshold = float(config.get("buy_threshold", 0.55))
        self.sell_threshold = float(config.get("sell_threshold", 0.45))

        # 模型缓存
        self._model: Any = None
        self._feature_cols: List[str] = []

    def _load_model(self) -> None:
        """加载 LightGBM 模型和元数据"""
        if self._model is not None:
            return

        model_path = Path(self.lgb_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"LightGBM 模型不存在: {self.lgb_model_path}")

        self._model = lgb.Booster(model_file=str(model_path))

        # 加载特征名
        meta_path = model_path.with_suffix(".pkl") if model_path.suffix == ".txt" else None
        if meta_path and meta_path.exists():
            meta = joblib.load(str(meta_path))
            self._feature_cols = meta.get("features", [])

        if not self._feature_cols:
            self._feature_cols = [
                f for f in _DEFAULT_FEATURES
            ]

    def _compute_features(self, data: pd.DataFrame) -> np.ndarray:
        """从原始数据计算 LightGBM 所需的全部特征。

        注意：这里和训练时的特征提取逻辑必须一致。
        """
        if data.empty:
            return np.zeros((0, len(self._feature_cols)))

        result = pd.DataFrame(index=data.index)
        c = data['close']
        v = data['volume']
        h = data['high']
        l = data['low']
        ret = c.pct_change()

        for w in [5, 10, 20, 60, 120]:
            ma = c.rolling(w).mean()
            result[f'ma_{w}_ratio'] = c / ma - 1

        for w in [1, 5, 10, 20, 60]:
            result[f'mom_{w}d'] = c.pct_change(w)

        for w in [5, 10, 20, 60]:
            result[f'vol_{w}d'] = ret.rolling(w).std()

        result['vol_ratio'] = v / v.rolling(5).mean()

        gains = ret.clip(0)
        losses = (-ret).clip(0)
        for per in [6, 14]:
            result[f'rsi_{per}'] = 100 - 100 / (1 + gains.rolling(per).mean() / losses.rolling(per).mean().clip(0.001))

        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        result['macd'] = ema12 - ema26
        macd_ema = result['macd'].ewm(span=9).mean()
        result['macd_hist'] = result['macd'] - macd_ema

        # 截面特征（单股票层面用当前行的值）
        for col in ['mom_5d', 'mom_10d', 'mom_20d', 'vol_20d', 'macd', 'vol_ratio']:
            if col in result.columns:
                result[f'{col}_rank'] = result[col].rank(pct=True)
                mean = result[col].mean()
                std = result[col].std()
                result[f'{col}_zscore'] = (result[col] - mean) / max(std, 1e-8)

        result['mkt_close'] = data['close'].mean()
        result['rel_ma5'] = result['ma_5_ratio'].fillna(0) * 0

        # 确保全部特征列存在且顺序正确
        for col in self._feature_cols:
            if col not in result.columns:
                result[col] = 0.0

        result = result[self._feature_cols]
        result = result.fillna(0).replace([np.inf, -np.inf], 0)

        return result.values

    def _get_prediction_series(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """重写：用 LightGBM 模型生成预测信号，替代从缓存读取。"""
        self._load_model()

        features = self._compute_features(data)
        if len(features) == 0:
            return None

        # 直接 predict（不用 proba，因为 booster predict 输出就是正类概率）
        probs = self._model.predict(features)

        # 转为连续信号：概率 → [-1, 1]
        # 0.5 是中性点
        signals = pd.Series((probs - 0.5) * 2, index=data.index)
        return signals

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """当前日期生成截面多空信号（Top K 选股）。"""
        indicators = self.calculate_indicators(data)
        signal_series = indicators.get("predicted_return", pd.Series(dtype=float))

        if signal_series.empty or current_date not in signal_series.index:
            return []

        current_idx = self._get_current_idx(data, current_date)
        if current_idx < 0:
            return []

        current_signal = float(signal_series.iloc[current_idx])
        current_price = float(indicators["price"].iloc[current_idx])

        # 分类信号类型
        signal_type = self._classify_signal_type(current_signal)
        if signal_type is None:
            return []

        stock_code = data.attrs.get("stock_code", "UNKNOWN")
        return [
            TradingSignal(
                timestamp=current_date,
                stock_code=stock_code,
                signal_type=signal_type,
                strength=min(1.0, abs(current_signal)),
                price=current_price,
                reason=f"LightGBM 信号 {current_signal:.3f}",
                metadata={
                    "model_path": self.lgb_model_path,
                    "raw_signal": current_signal,
                    "top_k": self.top_k,
                },
            )
        ]

    def _classify_signal_type(self, raw_signal: float) -> Optional[SignalType]:
        """信号分类：概率 > buy_threshold → BUY, < sell_threshold → SELL"""
        if pd.isna(raw_signal):
            return None
        if raw_signal >= self.buy_threshold * 2 - 1:
            return SignalType.BUY
        if raw_signal <= self.sell_threshold * 2 - 1:
            return SignalType.SELL
        return None


# 训练时使用的所有特征列（用于 fallback 时的对齐）
_DEFAULT_FEATURES = [
    "ma_5_ratio", "ma_10_ratio", "ma_20_ratio", "ma_60_ratio", "ma_120_ratio",
    "mom_1d", "mom_5d", "mom_10d", "mom_20d", "mom_60d",
    "vol_5d", "vol_10d", "vol_20d", "vol_60d",
    "vol_ratio",
    "rsi_6", "rsi_14",
    "macd", "macd_hist",
    "mom_5d_rank", "mom_5d_zscore",
    "mom_10d_rank", "mom_10d_zscore",
    "mom_20d_rank", "mom_20d_zscore",
    "vol_20d_rank", "vol_20d_zscore",
    "macd_rank", "macd_zscore",
    "vol_ratio_rank", "vol_ratio_zscore",
    "mkt_close", "rel_ma5",
]
