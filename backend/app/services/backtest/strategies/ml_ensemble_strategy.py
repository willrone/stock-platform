"""
ML 集成策略：LightGBM + XGBoost 双模型集成 + 风控

特点：
- 双模型回归/二分类集成（LightGBM + XGBoost）
- 支持统一训练引擎的多种特征集（alpha158 / technical_62）
- 支持回归和二分类两种标签类型
- 三重风控：止损 + 波动率缩放 + 市场过滤

命名：ml_ensemble_lgb_xgb_riskctl
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.error_handler import ErrorSeverity, TaskError

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal
from .ml_feature_adapter import build_feature_matrix, build_feature_matrix_batch
from .ml_model_loader import LoadedModelPair, load_model_pair

_logger = logging.getLogger(__name__)

# 回归模式阈值
BUY_RETURN_THRESHOLD = 0.0
SELL_RETURN_THRESHOLD = -0.002
# 二分类模式阈值
BINARY_BUY_THRESHOLD = 0.5
BINARY_SELL_THRESHOLD = 0.3
# 风控默认值
DEFAULT_STOP_LOSS = -0.02
DEFAULT_TARGET_VOL = 0.02
VOL_SCALE_RANGE = (0.5, 2.0)
MARKET_FILTER_WINDOW = 5
DAILY_RETURNS_WINDOW = 60
MIN_DATA_LENGTH = 60


class MLEnsembleLgbXgbRiskCtlStrategy(BaseStrategy):
    """ML 集成策略：LightGBM + XGBoost + 风控

    支持统一训练引擎的模型（alpha158/technical_62 特征集，
    regression/binary 标签类型），同时向后兼容 legacy 模型。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("ml_ensemble_lgb_xgb_riskctl", config)
        self._init_model_config(config)
        self._init_risk_config(config)
        self._init_runtime_state()
        self._load_models()

    def _init_model_config(self, config: Dict[str, Any]) -> None:
        """初始化模型相关配置"""
        self.lgb_weight = config.get("lgb_weight", 0.5)
        self.xgb_weight = config.get("xgb_weight", 0.5)
        self.top_n = config.get("top_n", 5)
        default_path = Path(__file__).parents[4] / "data" / "models"
        self.model_path = config.get("model_path", str(default_path))
        self._lgb_model_id = config.get("lgb_model_id")
        self._xgb_model_id = config.get("xgb_model_id")
        # 模型对象和元数据（加载后填充）
        self._model_pair: Optional[LoadedModelPair] = None

    def _init_risk_config(self, config: Dict[str, Any]) -> None:
        """初始化风控配置"""
        self.stop_loss = config.get("stop_loss", DEFAULT_STOP_LOSS)
        self.enable_vol_scaling = config.get("vol_scaling", True)
        self.enable_market_filter = config.get("market_filter", True)
        self.target_vol = config.get("target_vol", DEFAULT_TARGET_VOL)

    def _init_runtime_state(self) -> None:
        """初始化运行时状态"""
        self._daily_returns: List[float] = []
        self._market_returns: List[float] = []

    # ── 模型加载 ─────────────────────────────────────────

    def _load_models(self) -> None:
        """加载预训练模型"""
        model_dir = Path(self.model_path)
        self._model_pair = load_model_pair(
            model_dir, self._lgb_model_id, self._xgb_model_id,
        )

    @property
    def _has_models(self) -> bool:
        return self._model_pair is not None

    @property
    def _feature_set(self) -> str:
        if self._model_pair:
            return self._model_pair.feature_set
        return "technical_62"

    @property
    def _label_type(self) -> str:
        if self._model_pair:
            return self._model_pair.label_type
        return "regression"

    # ── 信号生成 ─────────────────────────────────────────

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime,
        stock_code: str = "",
    ) -> List[TradingSignal]:
        """生成交易信号"""
        if data is None or len(data) < MIN_DATA_LENGTH:
            return []

        idx = self._get_current_idx(data, current_date)
        if idx < MIN_DATA_LENGTH:
            return []

        if not stock_code:
            stock_code = data.attrs.get("stock_code", "UNKNOWN")

        indicators = self.get_cached_indicators(data)
        pred = self._predict_at_index(data, indicators, idx)
        if pred is None:
            return []

        self._update_market_state(indicators, idx)
        last_ret = self._get_indicator_value(indicators, "return_1d", idx)
        scale = self._calculate_position_scale(last_ret)
        self._record_daily_return(last_ret)

        return self._pred_to_signals(
            pred, scale, stock_code, data["close"].iloc[idx], current_date,
        )

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算技术指标（用于 fallback 和风控）"""
        if data is None or len(data) == 0:
            return {}
        from .ml_indicators import compute_strategy_indicators
        return compute_strategy_indicators(data)

    # ── 预计算（单股票） ─────────────────────────────────

    def precompute_all_signals(
        self, data: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """预计算所有信号"""
        if data is None or len(data) < MIN_DATA_LENGTH:
            return None

        if not self._has_models:
            _logger.warning("ML模型未加载，使用 fallback 规则")
            return self._precompute_fallback(data)

        try:
            X = build_feature_matrix(data, self._feature_set)
            if X is None:
                return self._precompute_fallback(data)

            pred = self._ensemble_predict(X)
            return self._pred_array_to_signals(pred, data.index)
        except Exception as e:
            _logger.error(f"ML策略预计算失败: {e}", exc_info=True)
            return self._precompute_fallback(data)

    # ── 预计算（批量多股票） ─────────────────────────────

    def precompute_all_signals_batch(
        self, combined_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """批量预计算所有股票的信号"""
        if combined_df is None or len(combined_df) == 0:
            return None
        if not self._has_models:
            _logger.warning("ML模型未加载，无法批量预计算")
            return None

        try:
            df = self._normalize_batch_index(combined_df)
            if "stock_code" not in df.columns:
                _logger.error("数据缺少 stock_code 列")
                return None

            X = build_feature_matrix_batch(df, self._feature_set)
            if X is None:
                return None

            pred = self._ensemble_predict(X)
            return self._build_batch_result(df, pred)
        except Exception as e:
            _logger.error(f"ML策略批量预计算失败: {e}", exc_info=True)
            return None

    # ── 预测核心 ─────────────────────────────────────────

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """双模型集成预测"""
        mp = self._model_pair
        lgb_pred = mp.lgb_model.predict(X.reshape(-1, X.shape[-1]) if X.ndim == 1 else X)
        xgb_pred = self._predict_xgb(X.reshape(-1, X.shape[-1]) if X.ndim == 1 else X)
        return self.lgb_weight * lgb_pred + self.xgb_weight * xgb_pred

    def _predict_xgb(self, X: np.ndarray) -> np.ndarray:
        """XGBoost 预测，兼容 sklearn 和 native API"""
        model = self._model_pair.xgb_model
        if hasattr(model, "get_booster"):
            return model.predict(X)
        import xgboost as xgb
        return model.predict(xgb.DMatrix(X))

    def _predict_at_index(
        self, data: pd.DataFrame,
        indicators: Dict[str, pd.Series], idx: int,
    ) -> Optional[float]:
        """在指定索引处预测"""
        if self._has_models:
            try:
                X = build_feature_matrix(
                    data.iloc[max(0, idx - MIN_DATA_LENGTH):idx + 1],
                    self._feature_set,
                )
                if X is not None and len(X) > 0:
                    pred = self._ensemble_predict(X[-1:])
                    return float(pred[0])
            except Exception:
                pass
        return self._fallback_prediction(indicators, idx)

    # ── 预测结果解读 ─────────────────────────────────────

    def _interpret_prediction(self, raw_pred: float) -> tuple:
        """根据标签类型解读预测值

        Returns:
            (signal_type_or_none, strength)
        """
        if self._label_type == "binary":
            return self._interpret_binary(raw_pred)
        return self._interpret_regression(raw_pred)

    @staticmethod
    def _interpret_regression(pred_return: float) -> tuple:
        """回归模式：预测值是收益率"""
        if pred_return > BUY_RETURN_THRESHOLD:
            strength = min(abs(pred_return) * 100, 1.0)
            return SignalType.BUY, strength
        if pred_return < SELL_RETURN_THRESHOLD:
            strength = min(abs(pred_return) * 100, 1.0)
            return SignalType.SELL, strength
        return None, 0.0

    @staticmethod
    def _interpret_binary(prob: float) -> tuple:
        """二分类模式：预测值是上涨概率"""
        if prob > BINARY_BUY_THRESHOLD:
            strength = min((prob - BINARY_BUY_THRESHOLD) * 2, 1.0)
            return SignalType.BUY, strength
        if prob < BINARY_SELL_THRESHOLD:
            strength = min((BINARY_SELL_THRESHOLD - prob) * 2, 1.0)
            return SignalType.SELL, strength
        return None, 0.0

    # ── 信号转换 ─────────────────────────────────────────

    def _pred_to_signals(
        self, pred: float, scale: float,
        stock_code: str, price: float, timestamp: datetime,
    ) -> List[TradingSignal]:
        """将单个预测值转换为交易信号"""
        # 风控触发清仓
        if scale == 0:
            return [TradingSignal(
                stock_code=stock_code, signal_type=SignalType.SELL,
                strength=0.8, price=price, timestamp=timestamp,
                reason=f"触发风控, pred={pred:.4f}",
            )]

        signal_type, strength = self._interpret_prediction(pred)
        if signal_type is None:
            return []

        strength = min(strength * scale, 1.0)
        label_desc = "概率" if self._label_type == "binary" else "收益"
        return [TradingSignal(
            stock_code=stock_code, signal_type=signal_type,
            strength=strength, price=price, timestamp=timestamp,
            reason=f"ML{label_desc}={pred:.4f}, 缩放={scale:.2f}",
        )]

    def _pred_array_to_signals(
        self, pred: np.ndarray, index,
    ) -> pd.Series:
        """将预测数组转换为 SignalType 序列"""
        signals = pd.Series([None] * len(index), index=index, dtype=object)

        for i, p in enumerate(pred):
            sig_type, _ = self._interpret_prediction(float(p))
            if sig_type is not None:
                signals.iloc[i] = sig_type

        buy_n = (signals == SignalType.BUY).sum()
        sell_n = (signals == SignalType.SELL).sum()
        _logger.info(
            f"ML预计算完成: BUY={buy_n}, SELL={sell_n}, "
            f"label_type={self._label_type}, 均值={pred.mean():.6f}"
        )
        return signals

    def _build_batch_result(
        self, df: pd.DataFrame, pred: np.ndarray,
    ) -> Optional[pd.DataFrame]:
        """构建批量预测结果 DataFrame"""
        signals = []
        for i, p in enumerate(pred):
            sig_type, strength = self._interpret_prediction(float(p))
            if sig_type is None:
                continue
            signals.append({
                "stock_code": df.iloc[i]["stock_code"],
                "date": df.iloc[i]["date"],
                "signal_type": sig_type,
                "strength": min(strength, 1.0),
                "price": df.iloc[i].get("close", 0),
            })

        if not signals:
            return None

        result = pd.DataFrame(signals)
        result.set_index(["stock_code", "date"], inplace=True)
        return result

    # ── 风控 ─────────────────────────────────────────────

    def _calculate_position_scale(self, daily_return: float) -> float:
        """计算仓位缩放因子"""
        if daily_return < self.stop_loss:
            return 0.0

        scale = 1.0
        if self.enable_vol_scaling and len(self._daily_returns) >= 20:
            recent_vol = np.std(self._daily_returns[-20:])
            if recent_vol > 0:
                vol_scale = self.target_vol / recent_vol
                scale *= np.clip(vol_scale, *VOL_SCALE_RANGE)

        if self.enable_market_filter:
            if len(self._market_returns) >= MARKET_FILTER_WINDOW:
                if np.mean(self._market_returns[-MARKET_FILTER_WINDOW:]) < 0:
                    scale *= 0.5

        return scale

    def _update_market_state(
        self, indicators: Dict[str, pd.Series], idx: int,
    ) -> None:
        """更新市场状态（用于市场过滤）"""
        ret = self._get_indicator_value(indicators, "return_1d", idx)
        if not pd.isna(ret):
            self._market_returns.append(ret)
            if len(self._market_returns) > 20:
                self._market_returns = self._market_returns[-20:]

    def _record_daily_return(self, ret: float) -> None:
        """记录日收益"""
        if not pd.isna(ret):
            self._daily_returns.append(ret)
            if len(self._daily_returns) > DAILY_RETURNS_WINDOW:
                self._daily_returns = self._daily_returns[-DAILY_RETURNS_WINDOW:]

    # ── Fallback ─────────────────────────────────────────

    def _fallback_prediction(
        self, indicators: Dict[str, pd.Series], idx: int,
    ) -> Optional[float]:
        """基于技术指标的 fallback 预测（回归模式）"""
        try:
            score = 0.0
            score += self._rsi_score(indicators, idx)
            score += self._macd_score(indicators, idx)
            score += self._momentum_score(indicators, idx)
            return float(score)
        except Exception:
            return None

    def _precompute_fallback(
        self, data: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """Fallback: 使用简化规则计算信号"""
        indicators = self.calculate_indicators(data)
        score = pd.Series(0.0, index=data.index)

        if "rsi_14" in indicators:
            rsi = indicators["rsi_14"]
            score += ((rsi < 30).astype(float) * 0.003)
            score -= ((rsi > 70).astype(float) * 0.003)

        if "macd_hist" in indicators and "macd_hist_slope" in indicators:
            mh = indicators["macd_hist"]
            ms = indicators["macd_hist_slope"]
            score += ((mh > 0) & (ms > 0)).astype(float) * 0.002
            score -= ((mh < 0) & (ms < 0)).astype(float) * 0.002

        signals = pd.Series([None] * len(data), index=data.index, dtype=object)
        signals[score > BUY_RETURN_THRESHOLD] = SignalType.BUY
        signals[score < SELL_RETURN_THRESHOLD] = SignalType.SELL
        return signals

    # ── 工具方法 ───────────��─────────────────────────────

    @staticmethod
    def _get_indicator_value(
        indicators: Dict[str, pd.Series], name: str, idx: int,
        default: float = 0.0,
    ) -> float:
        """安全获取指标值"""
        if name not in indicators:
            return default
        try:
            val = indicators[name].iloc[idx]
            return default if pd.isna(val) else float(val)
        except (IndexError, TypeError):
            return default

    def _rsi_score(self, ind: Dict[str, pd.Series], idx: int) -> float:
        rsi = self._get_indicator_value(ind, "rsi_14", idx, 50)
        if rsi < 30:
            return 0.003
        if rsi > 70:
            return -0.003
        return 0.0

    def _macd_score(self, ind: Dict[str, pd.Series], idx: int) -> float:
        mh = self._get_indicator_value(ind, "macd_hist", idx)
        ms = self._get_indicator_value(ind, "macd_hist_slope", idx)
        if mh > 0 and ms > 0:
            return 0.002
        if mh < 0 and ms < 0:
            return -0.002
        return 0.0

    def _momentum_score(self, ind: Dict[str, pd.Series], idx: int) -> float:
        mom = self._get_indicator_value(ind, "momentum_short", idx)
        return max(-0.002, min(mom, 0.002))

    @staticmethod
    def _normalize_batch_index(combined_df: pd.DataFrame) -> pd.DataFrame:
        """规范化批量数据的索引"""
        df = combined_df.reset_index()
        if "level_0" in df.columns:
            df = df.rename(columns={"level_0": "stock_code", "level_1": "date"})
        return df
