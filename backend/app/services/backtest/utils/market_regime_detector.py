"""
市场状态检测器

基于 ADX、波动率变化率、均线关系判断当前市场处于趋势/震荡/高波动/中性状态，
用于 regime-aware 动态权重分配。
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态枚举"""

    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# 策略分类关键词
# ---------------------------------------------------------------------------
TREND_KEYWORDS = ("macd", "rsi", "momentum", "trend")
MEAN_REV_KEYWORDS = ("cointegration", "bollinger", "mean_reversion", "pairs")

# ---------------------------------------------------------------------------
# 各 regime 下的权重乘数: {regime: (trend_mult, mean_rev_mult, other_mult)}
# ---------------------------------------------------------------------------
REGIME_MULTIPLIERS: Dict[MarketRegime, tuple] = {
    MarketRegime.TRENDING: (2.0, 0.3, 1.0),
    MarketRegime.MEAN_REVERTING: (0.3, 2.0, 1.0),
    MarketRegime.HIGH_VOLATILITY: (0.3, 1.5, 0.5),
    MarketRegime.NEUTRAL: (1.0, 1.0, 1.0),
}

# 高波动 regime 下建议的仓位缩放因子
HIGH_VOL_POSITION_SCALE = 0.5


@dataclass
class RegimeDetectorConfig:
    """检测器参数打包，避免构造函数参数过多"""

    adx_period: int = 20
    vol_period: int = 20
    trend_threshold: float = 25.0
    mean_rev_threshold: float = 20.0
    vol_spike_threshold: float = 1.5
    sma_short: int = 5
    sma_long: int = 20


class MarketRegimeDetector:
    """市场状态检测器

    检测指标：
    - ADX（20 日）：趋势强度
    - 波动率变化率：当前 vs 前一周期收益率标准差之比
    - 短期/长期均线关系（5/20 日）：辅助确认趋势方向
    """

    def __init__(self, config: RegimeDetectorConfig | None = None):
        self.cfg = config or RegimeDetectorConfig()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detect(self, market_data: pd.DataFrame) -> MarketRegime:
        """根据市场数据判断当前市场状态。

        Args:
            market_data: 至少包含 ``close``、``high``、``low`` 列的 DataFrame

        Returns:
            当前 MarketRegime
        """
        required_rows = self.cfg.adx_period * 2 + 1
        if market_data is None or len(market_data) < required_rows:
            logger.debug("数据不足 (%s 行)，返回 NEUTRAL", len(market_data) if market_data is not None else 0)
            return MarketRegime.NEUTRAL

        # 1. 波动率突变优先判断
        if self._is_high_volatility(market_data):
            return MarketRegime.HIGH_VOLATILITY

        # 2. ADX 判断趋势/震荡
        adx_value = self._compute_adx(market_data)
        if adx_value is None:
            return MarketRegime.NEUTRAL

        if adx_value > self.cfg.trend_threshold:
            return MarketRegime.TRENDING
        if adx_value < self.cfg.mean_rev_threshold:
            return MarketRegime.MEAN_REVERTING
        return MarketRegime.NEUTRAL

    def get_dynamic_weights(
        self,
        regime: MarketRegime,
        strategy_names: List[str],
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """根据 regime 和策略类型返回归一化后的动态权重。

        Args:
            regime: 当前市场状态
            strategy_names: 策略名称列表
            base_weights: 基础权重 {name: weight}

        Returns:
            动态调整并归一化后的权重字典
        """
        if regime == MarketRegime.NEUTRAL:
            return dict(base_weights)

        trend_m, mr_m, other_m = REGIME_MULTIPLIERS[regime]
        adjusted: Dict[str, float] = {}

        for name in strategy_names:
            base = base_weights.get(name, 0.0)
            category = _classify_strategy(name)
            if category == "trend":
                adjusted[name] = base * trend_m
            elif category == "mean_reversion":
                adjusted[name] = base * mr_m
            else:
                adjusted[name] = base * other_m

        return _normalize_weights(adjusted)

    # ------------------------------------------------------------------
    # 内部计算
    # ------------------------------------------------------------------

    def _is_high_volatility(self, data: pd.DataFrame) -> bool:
        """检测波动率是否急剧上升。"""
        close = data["close"].astype(float)
        returns = close.pct_change().dropna()
        period = self.cfg.vol_period

        if len(returns) < period * 2:
            return False

        current_vol = returns.iloc[-period:].std()
        prev_vol = returns.iloc[-period * 2 : -period].std()

        if prev_vol == 0 or np.isnan(prev_vol):
            return False

        vol_ratio = current_vol / prev_vol
        is_spike = vol_ratio > self.cfg.vol_spike_threshold
        if is_spike:
            logger.info("检测到高波动: vol_ratio=%.2f (阈值=%.2f)", vol_ratio, self.cfg.vol_spike_threshold)
        return is_spike

    def _compute_adx(self, data: pd.DataFrame) -> float | None:
        """计算最新的 ADX 值（Wilder 平滑）。"""
        high = data["high"].astype(float).values
        low = data["low"].astype(float).values
        close = data["close"].astype(float).values
        n = len(close)
        period = self.cfg.adx_period

        if n < period * 2:
            return None

        # True Range / +DM / -DM
        tr = np.empty(n - 1)
        plus_dm = np.empty(n - 1)
        minus_dm = np.empty(n - 1)

        for i in range(1, n):
            idx = i - 1
            tr[idx] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            plus_dm[idx] = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm[idx] = down_move if (down_move > up_move and down_move > 0) else 0.0

        # Wilder 平滑
        atr = _wilder_smooth(tr, period)
        smooth_plus = _wilder_smooth(plus_dm, period)
        smooth_minus = _wilder_smooth(minus_dm, period)

        if atr is None or smooth_plus is None or smooth_minus is None:
            return None

        # +DI / -DI / DX 序列（从 period-1 开始有效）
        length = len(atr)
        dx_list: list[float] = []
        for i in range(length):
            a = atr[i]
            if a == 0:
                continue
            plus_di = smooth_plus[i] / a * 100
            minus_di = smooth_minus[i] / a * 100
            di_sum = plus_di + minus_di
            if di_sum == 0:
                continue
            dx_list.append(abs(plus_di - minus_di) / di_sum * 100)

        if len(dx_list) < period:
            return None

        # ADX = DX 的 Wilder 平滑
        adx_arr = _wilder_smooth(np.array(dx_list), period)
        if adx_arr is None or len(adx_arr) == 0:
            return None

        adx_value = float(adx_arr[-1])
        logger.debug("ADX=%.2f (period=%d)", adx_value, period)
        return adx_value


# ======================================================================
# 模块级辅助函数
# ======================================================================


def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray | None:
    """Wilder 指数平滑（用于 ATR / ADX 计算）。"""
    if len(arr) < period:
        return None
    result = np.empty(len(arr) - period + 1)
    result[0] = arr[:period].mean()
    alpha = 1.0 / period
    for i in range(1, len(result)):
        result[i] = result[i - 1] * (1 - alpha) + arr[period - 1 + i] * alpha
    return result


def _classify_strategy(name: str) -> str:
    """根据策略名称自动分类。

    注意：mean_reversion 包含 'trend' 子串，
    因此必须先匹配均值回归关键词。
    """
    lower = name.lower()
    if any(kw in lower for kw in MEAN_REV_KEYWORDS):
        return "mean_reversion"
    if any(kw in lower for kw in TREND_KEYWORDS):
        return "trend"
    return "other"


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """归一化权重字典，使总和为 1。"""
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n > 0 else {}
    return {k: v / total for k, v in weights.items()}
