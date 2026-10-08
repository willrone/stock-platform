"""横截面因子打分 + TopK/Dropout 组合管理（因子排名策略）。

与阈值信号类因子策略（multi_factor 等）的本质区别：本策略不做逐股买卖择时，
而是每日对全市场股票输出一个可比的横截面排名分（ranking_score），组合构建
（持有 topk 只、每日换出排名掉出的最差 n_drop 只）由执行端
``TopkDropoutTradeModeExecutor`` 完成——即 qlib 官方 TopkDropout 范式。

复合因子 = 加权平均( z(动量), z(低波动), z(低振幅) )，按日期横截面 Z-Score：
  - 动量:   close[t-20] / close[t-146] - 1   （126 日收益，跳过最近 20 日，避开短期反转）
  - 低波动: -std(日收益率, 60)
  - 低振幅: -mean((high-low)/close, 10)

打分在 ``prepare_backtest_data`` 阶段一次性完成（全市场矩阵化），回测主循环
逐股查表，零额外计算开销。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal

_SCORES_ATTR = "_factor_ranking_scores"


class FactorRankingStrategy(BaseStrategy):
    """全市场复合因子横截面排名策略（TopK/Dropout 执行模式）。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("FactorRanking", config)
        self.topk = int(config.get("topk", 50))
        self.n_drop = int(config.get("n_drop", 5))
        self.hold_thresh = int(config.get("hold_thresh", 0))
        self.deal_price = str(config.get("deal_price", "close"))
        self.momentum_window = int(config.get("momentum_window", 126))
        self.momentum_skip = int(config.get("momentum_skip", 20))
        self.vol_window = int(config.get("vol_window", 60))
        self.amp_window = int(config.get("amp_window", 10))
        self.score_scale = float(config.get("score_scale", 10.0))
        weights = config.get("weights")
        self.weights: Dict[str, float] = dict(
            weights or {"momentum": 1.0, "low_vol": 1.0, "low_amp": 1.0}
        )
        if self.topk <= 0:
            raise ValueError("factor_ranking 策略要求 topk > 0")
        if self.n_drop <= 0:
            raise ValueError("factor_ranking 策略要求 n_drop > 0")

    # ── TopK/Dropout 执行契约 ──────────────────────────────────────────
    def get_trade_mode(self) -> str:
        return "topk_dropout"

    def get_trade_mode_config(self) -> Dict[str, object]:
        return {
            "topk": self.topk,
            "n_drop": self.n_drop,
            "hold_thresh": self.hold_thresh,
            "deal_price": self.deal_price,
        }

    # ── 全市场横截面打分（回测准备阶段一次性完成） ─────────────────────
    async def prepare_backtest_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        codes = [
            code
            for code, df in stock_data.items()
            if df is not None
            and not df.empty
            and "close" in df.columns
            and "high" in df.columns
            and "low" in df.columns
        ]
        if not codes:
            logger.warning("factor_ranking: 无可用股票数据，跳过打分")
            return

        indexes = [pd.DatetimeIndex(stock_data[c].index) for c in codes]
        master = indexes[0].append(indexes[1:]) if len(indexes) > 1 else indexes[0]
        master = master.unique().sort_values()
        n_dates, n_stocks = len(master), len(codes)

        def to_matrix(column: str) -> np.ndarray:
            out = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
            for j, code in enumerate(codes):
                df = stock_data[code]
                positions = master.get_indexer(pd.DatetimeIndex(df.index))
                ok = positions >= 0
                values = pd.to_numeric(df[column], errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
                out[positions[ok], j] = values[ok]
            return out

        frame_opts = {"index": master, "dtype": np.float64}
        close = pd.DataFrame(to_matrix("close"), **frame_opts)
        high = pd.DataFrame(to_matrix("high"), **frame_opts)
        low = pd.DataFrame(to_matrix("low"), **frame_opts)

        with np.errstate(divide="ignore", invalid="ignore"):
            momentum = (
                close.shift(self.momentum_skip)
                / close.shift(self.momentum_skip + self.momentum_window)
                - 1.0
            )
            low_vol = (
                -close.pct_change()
                .rolling(self.vol_window, min_periods=self.vol_window)
                .std()
            )
            amplitude = (high - low) / close.replace(0, np.nan)
            low_amp = -amplitude.rolling(
                self.amp_window, min_periods=self.amp_window
            ).mean()

        factors = {
            "momentum": momentum.to_numpy(),
            "low_vol": low_vol.to_numpy(),
            "low_amp": low_amp.to_numpy(),
        }
        min_cross = min(30, max(2, n_stocks // 10))
        weighted_sum = np.zeros((n_dates, n_stocks), dtype=np.float64)
        weight_count = np.zeros((n_dates, n_stocks), dtype=np.float64)

        for name, matrix in factors.items():
            weight = float(self.weights.get(name, 1.0))
            if weight == 0.0:
                continue
            valid = np.isfinite(matrix)
            counts = valid.sum(axis=1, keepdims=True)
            mean = np.where(valid, matrix, np.nan).mean(axis=1, keepdims=True)
            std = np.where(valid, matrix, np.nan).std(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                zscore = (matrix - mean) / np.where(std > 1e-12, std, np.nan)
            zscore = np.where(counts >= min_cross, zscore, np.nan)
            usable = np.isfinite(zscore)
            weighted_sum[usable] += weight * np.nan_to_num(zscore)[usable]
            weight_count[usable] += weight

        with np.errstate(invalid="ignore", divide="ignore"):
            composite = np.where(
                weight_count > 0,
                weighted_sum / np.where(weight_count > 0, weight_count, 1.0),
                np.nan,
            )

        stored = 0
        for j, code in enumerate(codes):
            column = composite[:, j]
            score_map: Dict[datetime, float] = {}
            for k in range(n_dates):
                value = column[k]
                if np.isfinite(value):
                    score_map[master[k].to_pydatetime()] = float(value)
            stock_data[code].attrs[_SCORES_ATTR] = score_map
            stored += len(score_map)
        logger.info(
            f"factor_ranking 横截面打分完成: 股票={n_stocks}, 日期={n_dates}, "
            f"有效分数={stored}, 最小截面要求={min_cross}"
        )

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        # 分数已在 prepare 阶段按横截面生成，逐股无需再算
        return None

    # ── 逐股读表输出 ranking_score 信号 ─────────────────────────────────
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        scores = data.attrs.get(_SCORES_ATTR) or {}
        index = pd.DatetimeIndex(data.index)
        score_series = pd.Series(
            [scores.get(ts.to_pydatetime()) for ts in index],
            index=index,
            dtype="float64",
        )
        return {"price": data["close"], "score": score_series}

    def generate_signals(
        self, data: pd.DataFrame, current_date: datetime
    ) -> List[TradingSignal]:
        scores = data.attrs.get(_SCORES_ATTR)
        if not scores:
            return []
        score = scores.get(current_date)
        if score is None or not np.isfinite(score):
            return []

        current_idx = self._get_current_idx(data, current_date)
        if current_idx < 0:
            return []
        price = float(data["close"].iloc[current_idx])
        if not np.isfinite(price) or price <= 0:
            return []
        stock_code = data.attrs.get("stock_code", "UNKNOWN")

        return [
            TradingSignal(
                timestamp=current_date,
                stock_code=stock_code,
                signal_type=SignalType.BUY,
                strength=min(1.0, abs(score) / self.score_scale),
                price=price,
                reason=(
                    f"复合因子排名分 {score:.4f} "
                    f"(TopK={self.topk}, n_drop={self.n_drop})"
                ),
                metadata={
                    "ranking_score": score,
                    "signal_role": "ranking_score",
                    "trade_mode": self.get_trade_mode(),
                    "topk": self.topk,
                    "n_drop": self.n_drop,
                    "hold_thresh": self.hold_thresh,
                    "deal_price": self.deal_price,
                },
            )
        ]
