"""暴露中性化复合策略（方法研究产出，新策略文件，不改动任何已有文件）。

## 与 honest_composite 的关系
本策略**完全复用** `honest_composite_strategy` 的因子定义、符号表、
面板读取、涨停掩码、调仓日掩码、lag 语义与 TopK-Dropout 执行契约。
唯一新增的是**组合构建层的一个可开关步骤**：把复合分对
「规模/流动性」风格暴露做逐日截面回归中性化（Barra APT 风格）。

## 抄的东西
1. **Barra APT 因子中性化**（Barra Global Equity Model Handbook,
   "Pure Factor Models" 一节）：先对风格暴露回归取残差再做组合。
   本实现用最小二乘残差（OLS），逐日横截面，暴露变量全部来自
   当日及之前的行情 → 无前视。
2. **Amihud(2002) 非流动性度量** ILLIQ = |r| / 成交额（20 日均值，
   取 log1p 抗肥尾）：作为流动性风格暴露。
3. **规模代理**：本平台**没有市值数据**（data/stocks/ 只有行情），
   因此规模代理只能用 20 日 log 成交额均值 —— 它同时含价格水平与
   成交量信息，是**流动性/规模混合**代理。报告里如实标注这一点。
4. **低换手率因子**（A 股特有）：负的 20 日 log 成交额均值 z-score。
   注意：本平台无流通股本数据，所以这是**成交额**代理而非严格的
   换手率（turnover = 成交量/流通股本）因子。报告中明确说明。

## 配置
    exposure_neutral : "none" | "size" | "illiq" | "size+illiq"
                      | "size+illiq+vol60"   要对哪些暴露中性化
    turnover_blend   : float  低换手率因子的混合权重（0 = 不混合）
    其余配置与 honest_composite 完全一致（topk / n_drop / lag /
    rebalance_days / limit_up / industry_neutral / panel_dir ...）

## 注册方式（**不要自己改注册表**）
本文件是**新策略文件**，需要两处注册才能被引擎按名字创建：

1. `backend/app/services/backtest/strategies/strategy_factory.py`
   的 `StrategyFactory._strategies` 字典里加一行：
       from .methods_composite_strategy import MethodsCompositeStrategy
       ...
       "methods_composite": MethodsCompositeStrategy,
2. `backend/scripts/verify_10y_pit.py` 的 `STRATEGIES` 字典里加一条：
       "methods_composite": {
           "cfg": {"topk": 60, "n_drop": 5, "rebalance_days": "weekly",
                   "rebalance_weekday": 4},
           "mps": 0.015833,
           "label": "42 因子 + 规模/流动性中性化",
       },
   之后即可：
       env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
         backend/scripts/verify_10y_pit.py --years 3 --lag 1 \
         --only methods_composite --out methods_composite.json

（研究脚本 `backend/scripts/verify_methods_engine.py` 用
`StrategyFactory.register_strategy()` 在运行时注册，**不碰注册表文件**。）
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.services.trading.execution_rules import (
    industry_ids,
    limit_up_mask_matrix,
    load_industry_map,
    load_st_flags,
    zscore_with_industry,
)

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal

# 复用 honest_composite 的因子表与常量（只读导入，不修改对方）
from .honest_composite_strategy import (  # noqa: F401
    _ALPHAPANEL_FACTORS,
    _HONEST_FORMULAS,
    _SCORES_ATTR,
    DEFAULT_PANEL_DIR,
)

_SCORES_ATTR_M = "_methods_composite_scores"

# 支持的中性化组合 → 暴露名
_NEUTRAL_SETS: Dict[str, tuple] = {
    "none": (),
    "size": ("size",),
    "illiq": ("illiq",),
    "size+illiq": ("size", "illiq"),
    "size+illiq+vol60": ("size", "illiq", "vol60"),
}


def _cs_zscore(mat: np.ndarray, min_cross: int = 30) -> np.ndarray:
    """逐日截面 z-score（nan-aware）。"""
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan)
    return np.where(cnt >= min_cross, z, np.nan)


def _amihud_illiq(
    close: np.ndarray,
    volume: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Amihud(2002) 非流动性: |r| / 成交额，窗口均值，取 log1p。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close[1:] / close[:-1] - 1.0
        ret = np.vstack([np.full((1, close.shape[1]), np.nan), ret])
        amt = close * volume
        daily = np.abs(ret) / np.where(amt > 0, amt, np.nan)
    m = pd.DataFrame(daily).rolling(window, min_periods=max(5, window // 2)).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log1p(m.to_numpy())


def _neutralize(
    score: np.ndarray,
    exposures: Dict[str, np.ndarray],
    min_cross: int = 50,
) -> np.ndarray:
    """逐日截面回归中性化：score 对各暴露做 OLS，取残差。

    Barra APT 风格。暴露全部来自当日及之前行情 → 无前视。
    """
    n_d, n_s = score.shape
    names = list(exposures)
    if not names:
        return score
    out = np.full_like(score, np.nan)
    for t in range(n_d):
        y = score[t]
        X = np.column_stack([exposures[k][t] for k in names])
        m = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if int(m.sum()) < min_cross:
            continue
        A = np.column_stack([X[m], np.ones(int(m.sum()))])
        try:
            beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[t, m] = y[m] - A @ beta
    return out


class MethodsCompositeStrategy(BaseStrategy):
    """42 因子复合 + 暴露中性化 + 可选低换手率因子混合。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("MethodsComposite", config)
        self.topk = int(config.get("topk", 60))
        self.n_drop = int(config.get("n_drop", config.get("ndrop", 5)))
        self.hold_thresh = int(config.get("hold_thresh", 0))
        self.lag = int(config.get("lag", 0))
        self.rebalance_days = str(config.get("rebalance_days", "daily")).lower()
        self.rebalance_weekday = int(config.get("rebalance_weekday", 4))
        self.limit_up = float(config.get("limit_up", 0.095))
        self.industry_neutral = bool(config.get("industry_neutral", False))
        self.score_scale = float(config.get("score_scale", 10.0))
        self.panel_dir = config.get("panel_dir", DEFAULT_PANEL_DIR)

        # ── 新增：暴露中性化 ────────────────────────────────────────
        self.exposure_neutral = str(config.get("exposure_neutral", "none")).lower()
        if self.exposure_neutral not in _NEUTRAL_SETS:
            raise ValueError(
                f"exposure_neutral 必须是 {list(_NEUTRAL_SETS)} 之一，"
                f"收到 {self.exposure_neutral!r}"
            )
        self.neutral_exposures = _NEUTRAL_SETS[self.exposure_neutral]
        # ── 新增：低换手率因子混合权重 ──────────────────────────────
        self.turnover_blend = float(config.get("turnover_blend", 0.0))

        cash_reserve = float(config.get("cash_reserve_ratio", 0.05))
        self.max_position_size = float(
            config.get("max_position_size", (1.0 - cash_reserve) / self.topk)
        )
        if self.topk <= 0 or self.n_drop <= 0:
            raise ValueError("methods_composite 要求 topk > 0 且 n_drop > 0")
        logger.info(
            f"methods_composite topk={self.topk} n_drop={self.n_drop} "
            f"lag={self.lag} neutral={self.exposure_neutral} "
            f"turnover_blend={self.turnover_blend} "
            f"mps={self.max_position_size:.6f}"
        )

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
        return self.max_position_size

    async def prepare_backtest_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date,
        end_date,
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
            logger.warning("methods_composite: 无可用股票数据，跳过打分")
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

        # ── 因子（与 honest_composite 完全一致）────────────────────
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

        day_ret = close.pct_change(fill_method=None).to_numpy()
        limit_up_mask = limit_up_mask_matrix(
            day_ret, close.columns, is_st=load_st_flags(list(close.columns))
        )

        min_cross = max(2, min(30, n_stocks // 10))
        acc = np.zeros((n_dates, n_stocks), dtype=np.float64)
        wsum = np.zeros((n_dates, n_stocks), dtype=np.float64)

        _ind_ids = None
        _n_ind = 0
        if self.industry_neutral:
            _imap = load_industry_map()
            _ind_ids = industry_ids(codes, _imap)
            _n_ind = int(_ind_ids.max()) + 1 if len(_ind_ids) else 0

        def zscore_add(mat: np.ndarray, sign: float) -> None:
            z = zscore_with_industry(
                mat, sign, min_cross, _ind_ids, _n_ind, self.industry_neutral
            )
            good = np.isfinite(z)
            acc[good] += np.nan_to_num(z)[good]
            wsum[good] += 1

        for formula, sign in _HONEST_FORMULAS.items():
            zscore_add(cands[formula].to_numpy(dtype=np.float64), sign)

        for factor_name, sign in _ALPHAPANEL_FACTORS.items():
            mat = np.full((n_dates, n_stocks), np.nan)
            for j, code in enumerate(codes):
                try:
                    pf = pd.read_parquet(
                        os.path.join(
                            self.panel_dir, code.replace(".", "_") + ".parquet"
                        )
                    ).reset_index()
                    pf = pf.set_index("date").sort_index()
                    pf = pf.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
                    pf = pf[~pf.index.duplicated(keep="first")]
                    pos = master.get_indexer(pd.DatetimeIndex(pf.index))
                    ok = pos >= 0
                    mat[pos[ok], j] = pd.to_numeric(
                        pf[factor_name], errors="coerce"
                    ).to_numpy(dtype=np.float64, na_value=np.nan)[ok]
                except Exception:
                    continue
            zscore_add(mat, sign)

        composite = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)

        # ── 新增：低换手率因子（负的 20 日 log 成交额均值 z）─────────
        # ⚠️ 顺序很重要：这个因子与下面的 `size` 暴露**完全共线**
        #    （z_to 是 size 的精确仿射变换，截面回归 R²=0.0）。
        #    若先混合再做 OLS 中性化，这一步会被中性化**完全消掉**（白做）。
        #    → 所以必须**先中性化、后混合**。
        if self.turnover_blend != 0.0:
            amount = close.to_numpy() * vol.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                log_amt = np.log(np.where(amount > 0, amount, np.nan))
            to_mean = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
            z_to = _cs_zscore(-to_mean, 30)
        else:
            z_to = None

        # ── 新增：暴露中性化（Barra APT 风格）──────────────────────
        if self.neutral_exposures:
            exposures: Dict[str, np.ndarray] = {}
            cn = close.to_numpy()
            vn = vol.to_numpy()
            if "size" in self.neutral_exposures:
                amount = cn * vn
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_amt = np.log(np.where(amount > 0, amount, np.nan))
                exposures["size"] = (
                    pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
                )
            if "illiq" in self.neutral_exposures:
                exposures["illiq"] = _amihud_illiq(cn, vn, 20)
            if "vol60" in self.neutral_exposures:
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = cn[1:] / cn[:-1] - 1.0
                    r = np.vstack([np.full((1, cn.shape[1]), np.nan), r])
                exposures["vol60"] = (
                    pd.DataFrame(r).rolling(60, min_periods=20).std().to_numpy()
                )
            composite = _neutralize(composite, exposures, min_cross=50)
            logger.info(f"methods_composite 暴露中性化: {self.neutral_exposures}")

        # ── 低换手率因子混合（必须在中性化**之后**）────────────────
        if z_to is not None:
            zb = _cs_zscore(composite, 30)
            w = float(self.turnover_blend)
            good = np.isfinite(zb) & np.isfinite(z_to)
            mixed = np.where(good, (zb + w * z_to) / (1.0 + w), np.nan)
            composite = np.where(np.isfinite(mixed), mixed, composite)

        composite[limit_up_mask] = np.nan

        if self.rebalance_days == "weekly":
            composite[master.dayofweek != self.rebalance_weekday] = np.nan
        elif self.rebalance_days == "monthly":
            composite[master.day != 1] = np.nan

        if self.lag > 0:
            lagged = np.full_like(composite, np.nan)
            if self.lag < n_dates:
                lagged[self.lag :] = composite[: n_dates - self.lag]
            composite = lagged

        stored = 0
        for j, code in enumerate(codes):
            column = composite[:, j]
            score_map: Dict[Any, float] = {}
            for k in range(n_dates):
                value = column[k]
                if np.isfinite(value):
                    score_map[master[k].to_pydatetime()] = float(value)
            stock_data[code].attrs[_SCORES_ATTR_M] = score_map
            stored += len(score_map)
        logger.info(
            f"methods_composite 横截面打分完成: 股票={n_stocks}, "
            f"日期={n_dates}, 有效分数={stored}, lag={self.lag}"
        )

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        return None

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        scores = data.attrs.get(_SCORES_ATTR_M) or {}
        index = pd.DatetimeIndex(data.index)
        score_series = pd.Series(
            [scores.get(ts.to_pydatetime()) for ts in index],
            index=index,
            dtype="float64",
        )
        return {"price": data["close"], "score": score_series}

    def generate_signals(self, data: pd.DataFrame, current_date) -> List[TradingSignal]:
        scores = data.attrs.get(_SCORES_ATTR_M)
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
                    f"暴露中性化复合分 {score:.4f} "
                    f"(neutral={self.exposure_neutral}, "
                    f"to_blend={self.turnover_blend}, "
                    f"TopK={self.topk}, n_drop={self.n_drop}, lag={self.lag})"
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
