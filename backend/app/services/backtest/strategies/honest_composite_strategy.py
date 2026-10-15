"""诚实版等权复合因子 → TopK-Dropout 组合管理（正式策略 honest_composite）。

诊断8 发现: v2 复合的头条 Sharpe 1.75 大部分是"当日涨停锁死"伪迹——
  内盘因子 (HIGH0/KLEN/KUP/KLOW/LOW0/RSQR5) 的 top50 全是当日涨停票，
  排除当日涨停后这些因子几乎失效。引擎未建模锁死 → 伪收益。

本版三条纪律（与 scripts/ic_composite_10y_honest.py 完全对齐）：
  1. 剔除全部内盘 OHLC 因子 (KLEN/KUP/KLOW/LOW0/HIGH0/RSQR5)
  2. 涨停掩码: 当日涨幅 >= 9.5% 的票分数置 NaN (当日不可在收盘买入)
  3. 保留 close/volume 族 (MIN/STD/CORR/CORD/MAX), MAX 亦被掩码约束

因子池 = 25 个诚实因子 (23 个唯一公式) + 18 个 Alpha158 面板因子 = 43：
  基础 23 公式: MIN/MAX/STD/CORR/CORD，n ∈ {5,10,20,30,60}
    MIN:  +1  (低于窗口最低价越深越好 → 正 IC)
    MAX/STD/CORR/CORD: -1
  Alpha158 面板因子 (读自 panel_dir/，MultiIndex 转 date)：
    9 族 × 2 窗口 (5,10): VSUMP/VSUMN/VSUMD/QTLD/IMAX/IMIN/WVMA/VMA/VSTD
    符号取自 5y IC study fwd1 IC 符号 (IMIN5 负、IMIN10 正)
  面板数据从 2021-05-11 起，此前 18 个因子为 NaN，复合自动退化为 23 公式版。

复合 = 等权平均( sign(z(因子)) )，nan-aware 截面 Z-Score（有效票数 ≥ min_cross）。
涨停掩码: 当日涨幅 ≥ limit_up 的票 composite = NaN。
lag: 0 = 用当日分数交易当日（默认）；1 = 用前一日分数。

TopK-Dropout 执行端由 ``TopkDropoutTradeModeExecutor`` 完成，
config 提供 topk / ndrop / hold_thresh / max_position_size。
max_position_size 默认 = (1-cash_reserve)/topk 等权，避免现金耗尽冻结。
config 可选 panel_dir 指定 Alpha158 面板因子路径（默认 ../data/qlib_data/features/day）。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal

_SCORES_ATTR = "_honest_composite_scores"

# 23 个唯一公式 (5 族 × 5 窗口)，MIN 为正号，其余为负号
_HONEST_FORMULAS: Dict[str, float] = {}
for _n in (5, 10, 20, 30, 60):
    _HONEST_FORMULAS[f"MIN{_n}"] = 1.0
    _HONEST_FORMULAS[f"MAX{_n}"] = -1.0
    _HONEST_FORMULAS[f"STD{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORR{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORD{_n}"] = -1.0

# 18 个 Alpha158 面板因子（来自 qlib_data/features/day/，MultiIndex 已转 date 列）
# 9 族 × 2 窗口 (5, 10)。符号取自 5y IC study fwd1 IC 符号
_ALPHAPANEL_FACTORS: Dict[str, float] = {
    "VSUMP5": -1.0, "VSUMP10": -1.0,
    "VSUMN5": 1.0, "VSUMN10": 1.0,
    "VSUMD5": -1.0, "VSUMD10": -1.0,
    "QTLD5": 1.0, "QTLD10": 1.0,
    "IMAX5": -1.0, "IMAX10": -1.0,
    "IMIN5": -1.0, "IMIN10": 1.0,
    "WVMA5": -1.0, "WVMA10": -1.0,
    "VMA5": 1.0, "VMA10": 1.0,
    "VSTD5": -1.0, "VSTD10": -1.0,
}


class HonestCompositeStrategy(BaseStrategy):
    """25 因子等权复合横截面排名策略（TopK/Dropout 执行模式）。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("HonestComposite", config)
        self.topk = int(config.get("topk", 60))
        self.n_drop = int(config.get("n_drop", config.get("ndrop", 5)))
        self.hold_thresh = int(config.get("hold_thresh", 0))
        self.lag = int(config.get("lag", 0))
        self.limit_up = float(config.get("limit_up", 0.095))
        self.score_scale = float(config.get("score_scale", 10.0))
        self.panel_dir = config.get(
            "panel_dir", "../data/qlib_data/features/day"
        )

        # max_position_size: 默认等权公式 (1-cash_reserve)/topk
        cash_reserve = float(config.get("cash_reserve_ratio", 0.05))
        self.max_position_size = float(
            config.get("max_position_size", (1.0 - cash_reserve) / self.topk)
        )

        if self.topk <= 0:
            raise ValueError("honest_composite 策略要求 topk > 0")
        if self.n_drop <= 0:
            raise ValueError("honest_composite 策略要求 n_drop > 0")
        logger.info(
            f"honest_composite topk={self.topk} n_drop={self.n_drop} "
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
            and "volume" in df.columns
        ]
        if not codes:
            logger.warning("honest_composite: 无可用股票数据，跳过打分")
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
        vol = pd.DataFrame(to_matrix("volume"), **frame_opts)

        # 计算 23 个诚实因子公式
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = close.pct_change(fill_method=None)
            vchg = vol.pct_change(fill_method=None)
            cands: Dict[str, pd.DataFrame] = {}
            for n in (5, 10, 20, 30, 60):
                cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
                cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
                cands[f"STD{n}"] = close.rolling(n).std() / close
                cands[f"CORR{n}"] = close.rolling(n).corr(vol)
                cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)

        # 涨停掩码: 当日涨幅 >= limit_up 的票不可在收盘买入
        day_ret = close.pct_change(fill_method=None).to_numpy()
        limit_up_mask = day_ret >= self.limit_up

        # nan-aware 等权复合 — 23 个诚实公式 + 18 个 Alpha158 面板因子
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

        # 23 个诚实公式
        for formula, sign in _HONEST_FORMULAS.items():
            zscore_add(cands[formula].to_numpy(dtype=np.float64), sign, min_cross)

        # 18 个 Alpha158 面板因子 — 每个因子构建一次全截面矩阵
        for factor_name, sign in _ALPHAPANEL_FACTORS.items():
            mat = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
            for j, code in enumerate(codes):
                try:
                    pf = pd.read_parquet(
                        os.path.join(self.panel_dir, code.replace(".", "_") + ".parquet")
                    ).reset_index()
                    pf = pf.set_index("date").sort_index()
                    pf = pf.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
                    pf = pf[~pf.index.duplicated(keep="first")]
                    pos = master.get_indexer(pd.DatetimeIndex(pf.index))
                    ok = pos >= 0
                    mat[pos[ok], j] = pd.to_numeric(pf[factor_name], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)[ok]
                except Exception:
                    continue
            zscore_add(mat, sign, min_cross)

        composite = np.where(
            wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan
        )
        composite[limit_up_mask] = np.nan  # 涨停掩码: 当日不可买

        # lag: 0 = 当日分数，1 = 前一日分数
        if self.lag > 0:
            lagged = np.full_like(composite, np.nan)
            if self.lag < n_dates:
                lagged[self.lag:] = composite[: n_dates - self.lag]
            composite = lagged

        # 存储每股分数
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
            f"honest_composite 横截面打分完成: 股票={n_stocks}, 日期={n_dates}, "
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
                    f"诚实复合因子排名分 {score:.4f} "
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
