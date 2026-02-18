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

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal
from .ml_feature_adapter import build_feature_matrix, build_feature_matrix_batch
from .ml_model_loader import LoadedModelPair, load_model_pair

_logger = logging.getLogger(__name__)

# 回归模式阈值
# 注意：CSRankNorm 模型的预测值范围约 [-0.3, 0.3]，
# 需要更高的阈值来过滤噪声信号。
# 仅选择预测值在分布两端的股票（约 top/bottom 20%）
BUY_RETURN_THRESHOLD = 0.05
SELL_RETURN_THRESHOLD = -0.05
# 二分类模式阈值
BINARY_BUY_THRESHOLD = 0.5
BINARY_SELL_THRESHOLD = 0.3
# 风控默认值
DEFAULT_STOP_LOSS = -0.05
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
        self.top_n = config.get("top_n", 10)
        default_path = Path(__file__).parents[5] / "data" / "models"
        self.model_path = config.get("model_path", str(default_path))
        self._lgb_model_id = config.get("lgb_model_id")
        self._xgb_model_id = config.get("xgb_model_id")
        # 允许通过 config 覆盖买卖阈值（CSRankNorm 模型需要更高阈值）
        self._buy_threshold = config.get("buy_threshold", BUY_RETURN_THRESHOLD)
        self._sell_threshold = config.get("sell_threshold", SELL_RETURN_THRESHOLD)
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
            model_dir,
            self._lgb_model_id,
            self._xgb_model_id,
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
    def _feature_columns(self) -> list:
        if self._model_pair:
            return self._model_pair.feature_columns
        return []

    @property
    def _label_type(self) -> str:
        if self._model_pair:
            return self._model_pair.label_type
        return "regression"

    # ── 信号生成 ─────────────────────────────────────────

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_date: datetime,
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
            pred,
            scale,
            stock_code,
            data["close"].iloc[idx],
            current_date,
        )

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算技术指标（用于 fallback 和风控）"""
        if data is None or len(data) == 0:
            return {}
        from .ml_indicators import compute_strategy_indicators

        return compute_strategy_indicators(data)

    # ── 预计算（单股票） ─────────────────────────────────

    def precompute_all_signals(
        self,
        data: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """预计算所有信号"""
        if data is None or len(data) < MIN_DATA_LENGTH:
            return None

        if not self._has_models:
            _logger.warning("ML模型未加载，使用 fallback 规则")
            return self._precompute_fallback(data)

        try:
            X = build_feature_matrix(data, self._feature_set, self._feature_columns)
            if X is None:
                return self._precompute_fallback(data)

            pred = self._ensemble_predict(X)
            return self._pred_array_to_signals(pred, data.index)
        except Exception as e:
            _logger.error(f"ML策略预计算失败: {e}", exc_info=True)
            return self._precompute_fallback(data)

    # ── 预计算（批量多股票） ─────────────────────────────

    def precompute_all_signals_batch(
        self,
        combined_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """批量预计算所有股票的信号（含截面排名选股）

        关键优化：使用截面排名（cross-sectional ranking）替代固定阈值。
        每个交易日对所有股票的预测值排名，只选 Top-N 买入、Bottom-N 卖出。
        这与滚动训练回测（Top10 夏普 2.09）的选股逻辑一致。
        """
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

            X = build_feature_matrix_batch(df, self._feature_set, self._feature_columns)
            if X is None:
                return None

            pred = self._ensemble_predict(X)

            # 截面排名选股：每天只选 top_n 买入，bottom_n 卖出
            return self._build_batch_result_with_cross_sectional_rank(df, pred)
        except Exception as e:
            _logger.error(f"ML策略批量预计算失败: {e}", exc_info=True)
            return None

    # ── 预测核心 ─────────────────────────────────────────

    @staticmethod
    def _unwrap_model(model):
        """解包 Qlib 模型包装，获取底层原生模型"""
        # Qlib LGBModel / XGBModel 把原生模型存在 .model 属性
        if hasattr(model, "model") and model.model is not None:
            return model.model
        return model

    def _predict_lgb(self, X: np.ndarray) -> np.ndarray:
        """LightGBM 预测，兼容 Qlib 包装和原生 Booster"""
        raw = self._unwrap_model(self._model_pair.lgb_model)
        if raw is None:
            return np.zeros(X.shape[0])
        return raw.predict(X)

    def _predict_xgb(self, X: np.ndarray) -> np.ndarray:
        """XGBoost 预测，兼容 Qlib 包装、sklearn API、native Booster 和 LightGBM 模型

        当 lgb_model_id 和 xgb_model_id 指向同一个 LightGBM 模型时，
        自动检测模型类型并使用 LightGBM 的预测方式。
        """
        raw = self._unwrap_model(self._model_pair.xgb_model)
        if raw is None:
            return np.zeros(X.shape[0])

        # 检测是否实际上是 LightGBM 模型
        try:
            import lightgbm as lgb
            if isinstance(raw, lgb.Booster):
                _logger.debug("XGB 槽位检测到 LightGBM Booster，使用 lgb 预测")
                return raw.predict(X)
        except ImportError:
            pass

        # sklearn API (XGBRegressor/XGBClassifier)
        if hasattr(raw, "get_booster"):
            return raw.predict(X)
        # native xgb.Booster 需要 DMatrix
        import xgboost as xgb
        if isinstance(raw, xgb.Booster):
            return raw.predict(xgb.DMatrix(X))

        # 兜底：尝试直接调用 predict
        _logger.warning(f"XGB 槽位模型类型未知: {type(raw).__name__}，尝试直接 predict")
        return raw.predict(X)

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """双模型集成预测（支持单模型）

        不再静默吞掉异常——单个模型预测失败时记录错误并降级为另一个模型，
        两个都失败才返回零向量。
        """
        mp = self._model_pair
        X2d = X.reshape(-1, X.shape[-1]) if X.ndim == 1 else X

        has_lgb = mp.lgb_model is not None
        has_xgb = mp.xgb_model is not None

        lgb_pred = None
        xgb_pred = None

        if has_lgb:
            try:
                lgb_pred = self._predict_lgb(X2d)
            except Exception as e:
                _logger.error(f"LGB 预测失败: {e}", exc_info=True)

        if has_xgb:
            try:
                xgb_pred = self._predict_xgb(X2d)
            except Exception as e:
                _logger.error(f"XGB 预测失败: {e}", exc_info=True)

        if lgb_pred is not None and xgb_pred is not None:
            pred = self.lgb_weight * lgb_pred + self.xgb_weight * xgb_pred
            _logger.debug(
                f"集成预测: lgb=[{lgb_pred.min():.4f},{lgb_pred.max():.4f}], "
                f"xgb=[{xgb_pred.min():.4f},{xgb_pred.max():.4f}], "
                f"ensemble=[{pred.min():.4f},{pred.max():.4f}]"
            )
            return pred
        elif lgb_pred is not None:
            _logger.warning("仅 LGB 预测可用，XGB 失败或缺失")
            return lgb_pred
        elif xgb_pred is not None:
            _logger.warning("仅 XGB 预测可用，LGB 失败或缺失")
            return xgb_pred
        else:
            _logger.error("LGB 和 XGB 预测均失败，返回零向量")
            return np.zeros(X2d.shape[0])

    def _predict_at_index(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        idx: int,
    ) -> Optional[float]:
        """在指定索引处预测"""
        if self._has_models:
            try:
                X = build_feature_matrix(
                    data.iloc[max(0, idx - MIN_DATA_LENGTH) : idx + 1],
                    self._feature_set,
                    self._feature_columns,
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

    def _interpret_regression(self, pred_return: float) -> tuple:
        """回归模式：预测值是收益率（或 CSRankNorm 标准化值）

        CSRankNorm 模型输出范围约 [-0.3, 0.3]，
        strength 按预测值超出阈值的幅度线性映射到 [0, 1]。
        """
        buy_thr = self._buy_threshold
        sell_thr = self._sell_threshold
        # 预测值范围上限（用于 strength 归一化）
        _pred_range = 0.3
        if pred_return > buy_thr:
            strength = min((pred_return - buy_thr) / _pred_range, 1.0)
            return SignalType.BUY, strength
        if pred_return < sell_thr:
            strength = min((sell_thr - pred_return) / _pred_range, 1.0)
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
        self,
        pred: float,
        scale: float,
        stock_code: str,
        price: float,
        timestamp: datetime,
    ) -> List[TradingSignal]:
        """将单个预测值转换为交易信号"""
        # 风控触发清仓
        if scale == 0:
            return [
                TradingSignal(
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=0.8,
                    price=price,
                    timestamp=timestamp,
                    reason=f"触发风控, pred={pred:.4f}",
                )
            ]

        signal_type, strength = self._interpret_prediction(pred)
        if signal_type is None:
            return []

        strength = min(strength * scale, 1.0)
        label_desc = "概率" if self._label_type == "binary" else "收益"
        return [
            TradingSignal(
                stock_code=stock_code,
                signal_type=signal_type,
                strength=strength,
                price=price,
                timestamp=timestamp,
                reason=f"ML{label_desc}={pred:.4f}, 缩放={scale:.2f}",
            )
        ]

    def _pred_array_to_signals(
        self,
        pred: np.ndarray,
        index,
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
        self,
        df: pd.DataFrame,
        pred: np.ndarray,
    ) -> Optional[pd.DataFrame]:
        """构建批量预测结果 DataFrame"""
        signals = []
        for i, p in enumerate(pred):
            sig_type, strength = self._interpret_prediction(float(p))
            if sig_type is None:
                continue
            signals.append(
                {
                    "stock_code": df.iloc[i]["stock_code"],
                    "date": df.iloc[i]["date"],
                    "signal_type": sig_type,
                    "strength": min(strength, 1.0),
                    "price": df.iloc[i].get("close", 0),
                }
            )

        if not signals:
            return None

        result = pd.DataFrame(signals)
        result.set_index(["stock_code", "date"], inplace=True)
        return result

    def _build_batch_result_with_cross_sectional_rank(
        self,
        df: pd.DataFrame,
        pred: np.ndarray,
    ) -> Optional[pd.DataFrame]:
        """截面排名选股：每天对所有股票的预测值排名，Top-N 买入，Bottom-N 卖出。

        与滚动训练回测逻辑一致（Top10 夏普 2.09，年化 71.8%）。
        """
        if "date" not in df.columns or "stock_code" not in df.columns:
            return self._build_batch_result(df, pred)

        # 将预测值附加到 df
        df = df.copy()
        df["_pred"] = pred

        signals = []
        top_n = self.top_n  # 默认 5

        for date, group in df.groupby("date"):
            if len(group) < 2:
                continue

            # 按预测值排名
            sorted_group = group.sort_values("_pred", ascending=False)

            # Top-N 买入（预测值最高的 N 只）
            buy_candidates = sorted_group.head(top_n)
            for rank_i, (_, row) in enumerate(buy_candidates.iterrows()):
                pred_val = float(row["_pred"])
                # 仍需超过最低阈值，避免全市场下跌时强制买入
                if pred_val > self._buy_threshold * 0.5:
                    # strength 按排名位置线性衰减：rank 0 → 1.0, rank N-1 → 0.3
                    rank_strength = max(0.3, 1.0 - rank_i * 0.7 / max(1, top_n - 1))
                    signals.append({
                        "stock_code": row["stock_code"],
                        "date": date,
                        "signal_type": SignalType.BUY,
                        "strength": rank_strength,
                        "price": row.get("close", 0),
                    })

            # Bottom-N 卖出（预测值最低的 N 只）
            sell_candidates = sorted_group.tail(top_n)
            for _, row in sell_candidates.iterrows():
                pred_val = float(row["_pred"])
                if pred_val < self._sell_threshold * 0.5:
                    signals.append({
                        "stock_code": row["stock_code"],
                        "date": date,
                        "signal_type": SignalType.SELL,
                        "strength": min(1.0, abs(pred_val) / 0.1),
                        "price": row.get("close", 0),
                    })

        if not signals:
            _logger.warning("截面排名选股未产生任何信号")
            return None

        buy_n = sum(1 for s in signals if s["signal_type"] == SignalType.BUY)
        sell_n = sum(1 for s in signals if s["signal_type"] == SignalType.SELL)
        _logger.info(
            f"截面排名选股完成: BUY={buy_n}, SELL={sell_n}, "
            f"top_n={top_n}, 总交易日={df['date'].nunique()}"
        )

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
        self,
        indicators: Dict[str, pd.Series],
        idx: int,
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
        self,
        indicators: Dict[str, pd.Series],
        idx: int,
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
        self,
        data: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """Fallback: 使用简化规则计算信号"""
        indicators = self.calculate_indicators(data)
        score = pd.Series(0.0, index=data.index)

        if "rsi_14" in indicators:
            rsi = indicators["rsi_14"]
            score += (rsi < 30).astype(float) * 0.003
            score -= (rsi > 70).astype(float) * 0.003

        if "macd_hist" in indicators and "macd_hist_slope" in indicators:
            mh = indicators["macd_hist"]
            ms = indicators["macd_hist_slope"]
            score += ((mh > 0) & (ms > 0)).astype(float) * 0.002
            score -= ((mh < 0) & (ms < 0)).astype(float) * 0.002

        signals = pd.Series([None] * len(data), index=data.index, dtype=object)
        signals[score > self._buy_threshold] = SignalType.BUY
        signals[score < self._sell_threshold] = SignalType.SELL
        return signals

    # ── 工具方法 ───────────��─────────────────────────────

    @staticmethod
    def _get_indicator_value(
        indicators: Dict[str, pd.Series],
        name: str,
        idx: int,
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
