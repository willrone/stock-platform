"""
价值因子策略

2026-02：支持可选的估值阈值筛选（用于提升信号质量/稳定性）。
注意：当前回测数据源未包含真实财务报表 ROE 等字段；
因此这里的 PE/PB 为策略内部估计序列，仅用于"相对过滤/去极值"。
"""

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from ..base.factor_base import FactorStrategy
from ...models import SignalType, TradingSignal


class ValueFactorStrategy(FactorStrategy):
    """价值因子策略

    2026-02：支持可选的估值阈值筛选（用于提升信号质量/稳定性）。
    注意：当前回测数据源未包含真实财务报表 ROE 等字段；
    因此这里的 PE/PB 为策略内部估计序列，仅用于"相对过滤/去极值"。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("ValueFactor", config)
        self.pe_weight = config.get("pe_weight", 0.25)
        self.pb_weight = config.get("pb_weight", 0.25)
        self.ps_weight = config.get("ps_weight", 0.25)
        self.ev_ebitda_weight = config.get("ev_ebitda_weight", 0.25)

        # Optional valuation thresholds (filters)
        # If set, a BUY signal will only be emitted when estimated ratios satisfy the constraints.
        self.pe_max = config.get("pe_max", None)
        self.pb_max = config.get("pb_max", None)
        # placeholder for future fundamental integration
        self.roe_min = config.get("roe_min", None)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算价值因子指标"""
        close_prices = data["close"]

        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(window=21).std()

        pe_estimate = pd.Series(
            1 / (returns.rolling(window=252).mean() + 0.001), index=close_prices.index
        )
        pe_estimate = pe_estimate.clip(-100, 100).fillna(15)

        pb_estimate = pd.Series(
            1 / (volatility + 0.01) * 0.5 + 1, index=close_prices.index
        )
        pb_estimate = pb_estimate.clip(0.1, 10).fillna(2)

        ps_estimate = pd.Series(
            1 / (volatility + 0.01) * 0.3 + 1.5, index=close_prices.index
        )
        ps_estimate = ps_estimate.clip(0.5, 15).fillna(3)

        ev_ebitda_estimate = pd.Series(
            1 / (volatility + 0.01) * 0.4 + 5, index=close_prices.index
        )
        ev_ebitda_estimate = ev_ebitda_estimate.clip(2, 30).fillna(10)

        value_score = pd.Series(0.0, index=close_prices.index)

        pe_normalized = (pe_estimate - pe_estimate.rolling(window=252).mean()) / (
            pe_estimate.rolling(window=252).std() + 0.01
        )
        pb_normalized = (pb_estimate - pb_estimate.rolling(window=252).mean()) / (
            pb_estimate.rolling(window=252).std() + 0.01
        )
        ps_normalized = (ps_estimate - ps_estimate.rolling(window=252).mean()) / (
            ps_estimate.rolling(window=252).std() + 0.01
        )
        ev_normalized = (
            ev_ebitda_estimate - ev_ebitda_estimate.rolling(window=252).mean()
        ) / (ev_ebitda_estimate.rolling(window=252).std() + 0.01)

        value_score = (
            -pe_normalized * self.pe_weight
            + -pb_normalized * self.pb_weight
            + -ps_normalized * self.ps_weight
            + -ev_normalized * self.ev_ebitda_weight
        )

        return {
            "pe_ratio": pe_estimate,
            "pb_ratio": pb_estimate,
            "ps_ratio": ps_estimate,
            "ev_ebitda": ev_ebitda_estimate,
            "value_score": value_score,
            "price": close_prices,
        }

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        """价值因子信号生成"""
        signals = []

        try:
            indicators = self.get_cached_indicators(data)

            current_idx = self._get_current_idx(data, current_date)
            if current_idx < 260:
                return signals

            current_price = indicators["price"].iloc[current_idx]
            current_score = indicators["value_score"].iloc[current_idx]
            prev_score = indicators["value_score"].iloc[current_idx - 1]

            stock_code = data.attrs.get("stock_code", "UNKNOWN")

            if prev_score <= 0 and current_score > 0:
                # Optional valuation thresholds
                current_pe = float(indicators["pe_ratio"].iloc[current_idx])
                current_pb = float(indicators["pb_ratio"].iloc[current_idx])

                if self.pe_max is not None and current_pe > float(self.pe_max):
                    return signals
                if self.pb_max is not None and current_pb > float(self.pb_max):
                    return signals

                strength = min(1.0, current_score)
                reason = f"价值因子评分转正: {current_score:.3f}"
                if self.pe_max is not None or self.pb_max is not None:
                    reason += f" (PE≤{self.pe_max}, PB≤{self.pb_max})"

                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    price=current_price,
                    reason=reason,
                    metadata={
                        "value_score": current_score,
                        "pe_ratio": current_pe,
                        "pb_ratio": current_pb,
                        "pe_max": self.pe_max,
                        "pb_max": self.pb_max,
                    },
                )
                signals.append(signal)

            elif prev_score >= 0 and current_score < 0:
                strength = min(1.0, abs(current_score))
                signal = TradingSignal(
                    timestamp=current_date,
                    stock_code=stock_code,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    price=current_price,
                    reason=f"价值因子评分转负: {current_score:.3f}",
                    metadata={
                        "value_score": current_score,
                        "pe_ratio": indicators["pe_ratio"].iloc[current_idx],
                        "pb_ratio": indicators["pb_ratio"].iloc[current_idx],
                    },
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"价值因子策略信号生成失败: {e}")
            return []
