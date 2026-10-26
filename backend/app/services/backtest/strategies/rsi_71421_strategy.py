"""RSI 均值回归 + 低波动 + 量价 → TopK-Dropout 组合管理（正式策略 rsi_71421）。

治本版 (2026-10-20 策略赛马验证): 原版 22f 的均值回归信号 MIN/MAX(距低点距离)
夏普只有 0.111, 本质是"赌大盘反弹"的 beta 策略。改用 RSI(超卖) 做均值回归信号后:
  - RSI 同时考虑跌的幅度和速度, 比"距低点距离"更准
  - 短周期 (7/14/21) 抓"刚跌完马上要弹"的快反转, 比长周期 (5/10/20/30/60) 夏普更高
  - 1 年回测: 夏普 0.111 → 0.537 (5 倍), 收益 +0.50% → +7.03%, 回撤 -17.9% → -15.7%

因子池 = RSI(7,14,21) + 低波动 STD(7,14,21) + 量价 CORR/CORD(7,14,21) = 12 因子:
  RSI{n}:   -1  (低 RSI=超卖=买, 信号为 -RSI)
  STD{n}:   -1  (低波动优先)
  CORR{n}:  -1  (量价相关)
  CORD{n}:  -1  (收益-量变相关)
复合 = 等权平均( sign(z(因子)) ), nan-aware 截面 Z-Score。
涨停掩码: 当日涨幅 >= limit_up 的票 composite = NaN (当日不可在收盘买入)。
lag: 0 = 用当日分数交易当日 (默认); 1 = 用前一日分数。

TopK-Dropout 执行端由 ``TopkDropoutTradeModeExecutor`` 完成,
config 提供 topk / ndrop / hold_thresh / max_position_size。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal
from app.services.trading.execution_rules import limit_up_mask_matrix

_SCORES_ATTR = "_rsi_71421_scores"

# 12 因子: RSI(超卖,信号为 -RSI) + 低波动 + 量价, 周期 7/14/21
# 全部负号: 值越小(超卖/低波/弱量价相关)越好
_RSI_PERIODS = (7, 14, 21)


def _rsi(series: pd.DataFrame, n: int) -> pd.DataFrame:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


class Rsi71421Strategy(BaseStrategy):
    """RSI 7/14/21 均值回归横截面排名策略（TopK/Dropout 执行模式）。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("Rsi71421", config)
        self.topk = int(config.get("topk", 30))
        self.n_drop = int(config.get("n_drop", config.get("ndrop", 3)))
        self.hold_thresh = int(config.get("hold_thresh", 0))
        self.lag = int(config.get("lag", 0))
        # 调仓日掩码：weekly → 只在周五出分（周频），daily → 每日出分。
        # 引擎不消费 BacktestConfig.rebalance_frequency，周频必须在策略侧
        # 把非调仓日分数置 NaN，执行器自然不换仓。
        self.rebalance_days = str(config.get("rebalance_days", "daily")).lower()
        self.rebalance_weekday = int(config.get("rebalance_weekday", 4))  # 4=周五
        self.limit_up = float(config.get("limit_up", 0.095))
        self.score_scale = float(config.get("score_scale", 10.0))

        cash_reserve = float(config.get("cash_reserve_ratio", 0.05))
        self.max_position_size = float(
            config.get("max_position_size", (1.0 - cash_reserve) / self.topk)
        )

        if self.topk <= 0:
            raise ValueError("rsi_71421 策略要求 topk > 0")
        if self.n_drop <= 0:
            raise ValueError("rsi_71421 策略要求 n_drop > 0")
        logger.info(
            f"rsi_71421 topk={self.topk} n_drop={self.n_drop} "
            f"lag={self.lag} limit_up={self.limit_up} "
            f"mps={self.max_position_size:.6f}"
        )

    # ── TopK/Dropout 执行契约 ──────────────────────────────────────────
    def get_trade_mode(self) -> str:
        return "topk_dropout"

    def get_trade_mode_config(self) -> Dict[str, object]:
        return {
            "topk": self.topk,
            "n_drop": self.n_drop,
            "hold_thresh": self.hold_thresh,
            "deal_price": "close",
        }

    def get_position_size_hint(self) -> Optional[float]:
        """等权上限 = (1-现金预留)/topk，与 TopK 持仓数匹配。"""
        return self.max_position_size

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
            if df is not None and not df.empty
            and "close" in df.columns and "volume" in df.columns
        ]
        if not codes:
            logger.warning("rsi_71421: 无可用股票数据，跳过打分")
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
        vol = pd.DataFrame(to_matrix("volume"), **frame_opts)

        with np.errstate(divide="ignore", invalid="ignore"):
            ret = close.pct_change(fill_method=None)
            vchg = vol.pct_change(fill_method=None)
            cands: Dict[str, pd.DataFrame] = {}
            for n in _RSI_PERIODS:
                cands[f"RSI{n}"] = -_rsi(close, n)              # 低RSI=超卖=买
                cands[f"STD{n}"] = -(close.rolling(n).std() / close)  # 低波动
                cands[f"CORR{n}"] = -close.rolling(n).corr(vol)  # 量价相关
                cands[f"CORD{n}"] = -ret.rolling(n).corr(vchg)   # 收益-量变相关

        day_ret = close.pct_change(fill_method=None).to_numpy()
        # 2026-10-25 修正：原来用统一的 self.limit_up（默认 0.095）全板通用。
        # 实测问题：182 只 20% 板（创业板/科创板）股票有 5380 个交易日涨幅落在
        # 9.5%~20%（并未涨停）却被误剔；ST 股触及 5% 涨停则完全没拦。
        # 改为逐票按板块判定（主板 10% / 创业板科创板 20% / 北交所 30%）。
        limit_up_mask = limit_up_mask_matrix(day_ret, close.columns)

        min_cross = max(2, min(30, n_stocks // 10))
        acc = np.zeros((n_dates, n_stocks), dtype=np.float64)
        wsum = np.zeros((n_dates, n_stocks), dtype=np.float64)

        def zscore_add(mat: np.ndarray, sign: float, min_cross: int) -> None:
            ok = np.isfinite(mat)
            cnt = ok.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
                sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan) * sign
            z = np.where(cnt >= min_cross, z, np.nan)
            good = np.isfinite(z)
            acc[good] += np.nan_to_num(z)[good]
            wsum[good] += 1

        for name in cands:
            zscore_add(cands[name].to_numpy(dtype=np.float64), 1.0, min_cross)

        composite = np.where(
            wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan
        )
        composite[limit_up_mask] = np.nan  # 涨停掩码: 当日不可买

        # 调仓日掩码: 非调仓日分数全部置 NaN → 执行器无分可依, 不换仓。
        if self.rebalance_days == "weekly":
            weekday = master.dayofweek  # 0=周一 … 4=周五
            composite[weekday != self.rebalance_weekday] = np.nan
        elif self.rebalance_days == "monthly":
            composite[master.day != 1] = np.nan

        if self.lag > 0:
            lagged = np.full_like(composite, np.nan)
            if self.lag < n_dates:
                lagged[self.lag:] = composite[: n_dates - self.lag]
            composite = lagged

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
            f"rsi_71421 横截面打分完成: 股票={n_stocks}, 日期={n_dates}, "
            f"有效分数={stored}, 最小截面要求={min_cross}, "
            f"涨停掩码={self.limit_up}, lag={self.lag}"
        )

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        return None

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
                    f"RSI 7/14/21 均值回归排名分 {score:.4f} "
                    f"(TopK={self.topk}, n_drop={self.n_drop}, lag={self.lag})"
                ),
                metadata={
                    "ranking_score": score,
                    "signal_role": "ranking_score",
                    "trade_mode": self.get_trade_mode(),
                    "topk": self.topk,
                    "n_drop": self.n_drop,
                    "hold_thresh": self.hold_thresh,
                    "deal_price": "close",
                },
            )
        ]
