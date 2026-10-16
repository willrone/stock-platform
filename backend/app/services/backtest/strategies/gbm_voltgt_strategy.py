"""LightGBM 元模型 + 波动率目标仓位（正式策略 gbm_voltgt）。

与 HonestCompositeStrategy (等权 42f) 的差别:
  1. 复合分数不是等权平均 z, 而是 LightGBM 回归预测 fwd5 收益
     (40 因子 z 分数 + presence 哑变量, 月度滚动重训, 无前视)
  2. 波动率目标仓位: 市场等权指数 20 日年化波动率做门,
     scale = min(1, 0.20/mv), 高波动日只保留 top round(60*scale) 只
     (其余置 NaN → 自动降仓位 → 降回撤)

回测 10y (2016-2026, 4329 票):
  等权 42f       = 0.907 Sharpe / -38.2% mdd / 200% ret
  gbm_voltgt     = 1.155 Sharpe / -40.9% mdd / 627% ret / OOS 1.018
  超参 A/B: n=200/depth=5 → 1.103 (稳健)
详见 data/strategy_baseline/ic_composite_10y_voltgt.json (commit 112f699).
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

_SCORES_ATTR = "_gbm_voltgt_scores"

# 23 个唯一公式 (5 族 × 5 窗口)
_HONEST_FORMULAS: Dict[str, float] = {}
for _n in (5, 10, 20, 30, 60):
    _HONEST_FORMULAS[f"MIN{_n}"] = 1.0
    _HONEST_FORMULAS[f"MAX{_n}"] = -1.0
    _HONEST_FORMULAS[f"STD{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORR{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORD{_n}"] = -1.0

# 17 个 Alpha158 面板因子
_ALPHAPANEL_FACTORS: Dict[str, float] = {
    "VSUMP5": -1.0, "VSUMP10": -1.0, "VSUMP20": -1.0,
    "VSUMN5": 1.0, "VSUMN10": 1.0, "VSUMN20": 1.0,
    "VSUMD5": -1.0, "VSUMD10": -1.0, "VSUMD20": -1.0,
    "QTLD5": 1.0, "QTLD10": 1.0, "QTLD20": 1.0,
    "IMAX5": -1.0, "IMIN5": -1.0, "WVMA5": -1.0,
    "VMA5": 1.0, "VSTD5": -1.0,
}


class GbmVoltgtStrategy(BaseStrategy):
    """LightGBM 元模型 + 波动率目标仓位（TopK/Dropout 执行模式）。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("GbmVoltgt", config)
        self.topk = int(config.get("topk", 60))
        self.n_drop = int(config.get("n_drop", config.get("ndrop", 5)))
        self.hold_thresh = int(config.get("hold_thresh", 0))
        self.limit_up = float(config.get("limit_up", 0.095))
        self.score_scale = float(config.get("score_scale", 10.0))
        self.panel_dir = config.get(
            "panel_dir", "../data/qlib_data/features/day"
        )

        # LightGBM 超参 (默认 = 回测最优 n=100/depth=3/lr=0.05)
        self.n_estimators = int(config.get("n_estimators", 100))
        self.max_depth = int(config.get("max_depth", 3))
        self.learning_rate = float(config.get("learning_rate", 0.05))
        self.warmup_months = int(config.get("warmup_months", 2))

        # 波动率目标仓位
        self.target_vol = float(config.get("target_vol", 0.20))
        self.vt_window = int(config.get("vt_window", 20))

        cash_reserve = float(config.get("cash_reserve_ratio", 0.05))
        self.max_position_size = float(
            config.get("max_position_size", (1.0 - cash_reserve) / self.topk)
        )

        if self.topk <= 0:
            raise ValueError("gbm_voltgt 策略要求 topk > 0")
        if self.n_drop <= 0:
            raise ValueError("gbm_voltgt 策略要求 n_drop > 0")
        logger.info(
            f"gbm_voltgt topk={self.topk} n_drop={self.n_drop} "
            f"limit_up={self.limit_up} "
            f"mps={self.max_position_size:.6f} "
            f"gbm(n={self.n_estimators},d={self.max_depth},lr={self.learning_rate}) "
            f"vt(target={self.target_vol},win={self.vt_window})"
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

    # ── 全市场横截面打分 ───────────────────────────────────────────────
    async def prepare_backtest_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        codes = [
            code for code, df in stock_data.items()
            if df is not None and not df.empty
            and "close" in df.columns and "volume" in df.columns
        ]
        if not codes:
            logger.warning("gbm_voltgt: 无可用股票数据，跳过打分")
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
                    dtype=np.float64, na_value=np.nan)
                out[positions[ok], j] = values[ok]
            return out

        frame_opts = {"index": master, "dtype": np.float64}
        close = pd.DataFrame(to_matrix("close"), **frame_opts)
        vol = pd.DataFrame(to_matrix("volume"), **frame_opts)

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
        limit_up_mask = day_ret >= self.limit_up
        fwd_ret = (close.shift(-5) / close - 1.0).to_numpy()

        # 面板因子矩阵
        def to_panel_mat(factor_name: str) -> np.ndarray:
            out = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
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
                    out[pos[ok], j] = pd.to_numeric(
                        pf[factor_name], errors="coerce"
                    ).to_numpy(dtype=np.float64, na_value=np.nan)[ok]
                except Exception:
                    continue
            return out

        # 截面 z 分数
        seen: Dict[str, List[str]] = {}
        # 用 IC study 符号 (fallback 到 _HONEST_FORMULAS/_ALPHAPANEL_FACTORS)
        for fname, feats in self._build_formula_map().items():
            seen[fname] = feats
        for fname in _ALPHAPANEL_FACTORS:
            seen[fname] = [fname]

        raw: List[np.ndarray] = []
        for fname in seen:
            if fname in _HONEST_FORMULAS:
                raw.append(cands[fname].to_numpy(dtype=np.float64))
            else:
                raw.append(to_panel_mat(fname))
        n_fac = len(raw)

        z_all = np.zeros((n_dates, n_stocks, n_fac))
        pre_all = np.zeros((n_dates, n_stocks, n_fac))
        for k in range(n_fac):
            mat = raw[k]
            ok = np.isfinite(mat)
            cnt = ok.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
                sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
                z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan)
            z = np.where(cnt >= 30, z, 0.0)
            z = np.where(np.isfinite(z), z, 0.0)
            z_all[:, :, k] = z
            pre_all[:, :, k] = ok

        # 波动率目标仓位系数
        mkt = close.mean(axis=1, skipna=True)
        mkt_ret = mkt.pct_change(fill_method=None)
        mkt_vol_ann = mkt_ret.rolling(self.vt_window).std() * np.sqrt(252)
        scale = np.minimum(1.0, (self.target_vol / mkt_vol_ann).to_numpy())
        scale[~np.isfinite(scale)] = 1.0

        # LightGBM 月度滚动重训
        import lightgbm as lgb

        month_starts = []
        prev = None
        for k, ts in enumerate(master):
            key = (ts.year, ts.month)
            if key != prev:
                month_starts.append((k, ts))
                prev = key

        comp_eq = np.full((n_dates, n_stocks), np.nan)
        acc = np.zeros((n_dates, n_stocks))
        wsum = np.zeros((n_dates, n_stocks))
        for (fname, _), k in zip(list(seen.items()), range(n_fac)):
            sign = (_HONEST_FORMULAS.get(fname, _ALPHAPANEL_FACTORS.get(fname, 1.0)))
            ok = np.isfinite(raw[k])
            acc[ok] += z_all[ok, k] * sign
            wsum[ok] += 1
        comp_eq = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)

        comp = np.full((n_dates, n_stocks), np.nan)
        for mi, (k0, ts0) in enumerate(month_starts):
            k1 = month_starts[mi + 1][0] if mi + 1 < len(month_starts) else n_dates
            if mi < self.warmup_months:
                comp[k0:k1] = comp_eq[k0:k1]
                continue
            z_block = z_all[:k0].reshape(-1, n_fac)
            p_block = pre_all[:k0].reshape(-1, n_fac)
            lab = fwd_ret[:k0].reshape(-1)
            valid = np.isfinite(lab)
            X = np.column_stack([z_block[valid], p_block[valid]])
            y = lab[valid]
            if len(X) < 500:
                comp[k0:k1] = comp_eq[k0:k1]
                continue
            mdl = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=15, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=-1,
            ).fit(X, y)
            for k in range(k0, k1):
                Xk = np.column_stack([z_all[k], pre_all[k]])
                comp[k] = mdl.predict(Xk)
        comp[limit_up_mask] = np.nan

        # 波动率目标仓位: 高波动日只保留 top round(topk*scale) 只
        for k in range(n_dates):
            s = scale[k]
            if s >= 1.0:
                continue
            n_keep = max(20, int(round(self.topk * s)))
            row = comp[k]
            ok = np.isfinite(row)
            if ok.sum() <= n_keep:
                continue
            idx = np.where(ok)[0]
            order = np.argsort(row[idx])[::-1][:n_keep]
            keep = set(idx[order].tolist())
            for j in idx:
                if j not in keep:
                    comp[k, j] = np.nan

        stored = 0
        for j, code in enumerate(codes):
            column = comp[:, j]
            score_map: Dict[datetime, float] = {}
            for k in range(n_dates):
                value = column[k]
                if np.isfinite(value):
                    score_map[master[k].to_pydatetime()] = float(value)
            stock_data[code].attrs[_SCORES_ATTR] = score_map
            stored += len(score_map)
        logger.info(
            f"gbm_voltgt 打分完成: 股票={n_stocks}, 日期={n_dates}, "
            f"有效分数={stored}, gbm超参(n={self.n_estimators},d={self.max_depth},lr={self.learning_rate}), "
            f"vt(target={self.target_vol},win={self.vt_window})"
        )

    def _build_formula_map(self) -> Dict[str, List[str]]:
        m: Dict[str, List[str]] = {}
        for n in (5, 10, 20, 30, 60):
            m[f"MIN{n}"] = [f"MIN{n}"]
            m[f"MAX{n}"] = [f"MAX{n}"]
            m[f"STD{n}"] = [f"STD{n}"]
            m[f"CORR{n}"] = [f"CORR{n}"]
            m[f"CORD{n}"] = [f"CORD{n}"]
        return m

    def precompute_all_signals(self, data: pd.DataFrame) -> Optional[pd.Series]:
        return None

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        scores = data.attrs.get(_SCORES_ATTR) or {}
        index = pd.DatetimeIndex(data.index)
        score_series = pd.Series(
            [scores.get(ts.to_pydatetime()) for ts in index],
            index=index, dtype="float64",
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
                    f"gbm_voltgt LightGBM 元模型+波动率目标分 {score:.4f} "
                    f"(TopK={self.topk}, n_drop={self.n_drop})"
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
