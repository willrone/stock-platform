"""模型线研究策略（GBM Lab）— 独立新文件，不改 gbm_voltgt / 回测引擎。

用途
----
`gbm_voltgt_strategy.py` 是只读的正式策略，本文件是它的**可配置研究版本**，
用于系统扫描"模型类策略"的各个维度：

1. 预测标签 horizon      : 1 / 3 / 5 / 10 / 20 / 60
2. 目标变量 target       : raw / rank / cs_z / ind_neutral / vol_std
3. 模型超参             : n_estimators / max_depth / learning_rate /
                          min_child_samples / subsample(+freq) / colsample /
                          reg_alpha / reg_lambda
4. 预测频率 retrain      : monthly / weekly / quarterly
5. 重训窗口 train_window : expand / rolling
6. 特征集 feature_set    : f25 / f42 / alpha158 / ic_topN
7. 模型族 model          : lgbm / xgb / ridge / elasticnet
8. 组合层 target_vol     : 0.10 / 0.15 / 0.20 / 0.25
9. 集成 blend            : 模型分 与 等权因子分 的混合

相对 gbm_voltgt 修掉的 6 个缺陷（全部实测确认，见 data/backtest/gbm_search_NOTES.md）
-----------------------------------------------------------------------------
- **D1 lag**     : gbm_voltgt 完全没有 self.lag → 旧数字其实是 lag 0（用当日收盘分
                   当日收盘成交，实盘做不到）。本文件实现真 lag（信号整体后移 lag 个
                   交易日），与 honest_composite 口径一致。
- **D2 purge**   : gbm_voltgt 用 fwd_ret[:k0] 训练，标签落在预测区间内（look-ahead）。
                   本文件按 qlib 标准 purge：训练行只到 `t + horizon + lag + 1 <= k0`。
- **D3 subsample**: gbm_voltgt 的 subsample=0.8 没设 subsample_freq → 行采样空操作。
                   本文件默认 subsample_freq=1（真正生效），并可关。
- **D4 面板源**  : features/day 的 parquet 是 block1+block2 拼接（日期重复），
                   策略侧读到的面板因子在引擎交易日格上覆盖率只有 ~49%。
                   本文件默认读 `data/qlib_data/features_full/`（无重复日期、
                   覆盖率 1.0、$close 逐位相同、带正确 PIT 截断）。
- **D5 涨停掩码**: gbm_voltgt / honest_composite 传 `close.columns`（整数列名）给
                   limit_up_mask_matrix → 涨跌停幅度全落 10%、ST 全漏。
                   本文件传真实代码 + ST 标记。
- **D6 标签锚点**: gbm_voltgt 的标签是 close[t+h]/close[t]，但 lag=1 时实际成交在
                   close[t+1] → 标签与持仓收益错位一天。本文件支持
                   label_anchor="trade"（默认，标签 = close[t+lag+h]/close[t+lag]）。

注册方式
--------
本文件**不改** strategy_factory.py（那是正式注册表）。搜索脚本
`backend/scripts/search_gbm_params.py` 在运行时调用
`StrategyFactory.register_strategy("gbm_lab", GbmLabStrategy)` 临时注册。
若要正式上线，由人工在 strategy_factory.py 的 `_strategies` 里加一行：
    "gbm_lab": GbmLabStrategy,
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.services.trading.execution_rules import (
    industry_ids,
    limit_up_mask_matrix,
    load_industry_map,
    load_st_flags,
)

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal

_SCORES_ATTR = "_gbm_lab_scores"

#: 25 个诚实公式（5 族 × 5 窗口），sign = +1 表示"值越高越好"
_HONEST_FORMULAS: Dict[str, float] = {}
for _n in (5, 10, 20, 30, 60):
    _HONEST_FORMULAS[f"MIN{_n}"] = 1.0
    _HONEST_FORMULAS[f"MAX{_n}"] = -1.0
    _HONEST_FORMULAS[f"STD{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORR{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORD{_n}"] = -1.0

#: 17 个 Alpha158 面板因子（与 gbm_voltgt / honest_composite 同口径）
_ALPHAPANEL_FACTORS: Dict[str, float] = {
    "VSUMP5": -1.0,
    "VSUMP10": -1.0,
    "VSUMP20": -1.0,
    "VSUMN5": 1.0,
    "VSUMN10": 1.0,
    "VSUMN20": 1.0,
    "VSUMD5": -1.0,
    "VSUMD10": -1.0,
    "VSUMD20": -1.0,
    "QTLD5": 1.0,
    "QTLD10": 1.0,
    "QTLD20": 1.0,
    "IMAX5": -1.0,
    "IMIN5": 1.0,
    "WVMA5": -1.0,
    "VMA5": 1.0,
    "VSTD5": -1.0,
}

#: features_full 里可用的 Alpha158 全集（162 列，不含 LABEL0 / ts_code）
_ALPHA158_FULL: List[str] = [
    "RET1",
    "RET5",
    "RET10",
    "RET20",
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
    "OPEN0",
    "HIGH0",
    "LOW0",
    "VWAP0",
    "ROC5",
    "ROC10",
    "ROC20",
    "ROC30",
    "ROC60",
    "MA5",
    "MA10",
    "MA20",
    "MA30",
    "MA60",
    "STD5",
    "STD10",
    "STD20",
    "STD30",
    "STD60",
    "BETA5",
    "BETA10",
    "BETA20",
    "BETA30",
    "BETA60",
    "RSQR5",
    "RSQR10",
    "RSQR20",
    "RSQR30",
    "RSQR60",
    "RESI5",
    "RESI10",
    "RESI20",
    "RESI30",
    "RESI60",
    "MAX5",
    "MAX10",
    "MAX20",
    "MAX30",
    "MAX60",
    "MIN5",
    "MIN10",
    "MIN20",
    "MIN30",
    "MIN60",
    "QTLU5",
    "QTLU10",
    "QTLU20",
    "QTLU30",
    "QTLU60",
    "QTLD5",
    "QTLD10",
    "QTLD20",
    "QTLD30",
    "QTLD60",
    "RANK5",
    "RANK10",
    "RANK20",
    "RANK30",
    "RANK60",
    "RSV5",
    "RSV10",
    "RSV20",
    "RSV30",
    "RSV60",
    "IMAX5",
    "IMAX10",
    "IMAX20",
    "IMAX30",
    "IMAX60",
    "IMIN5",
    "IMIN10",
    "IMIN20",
    "IMIN30",
    "IMIN60",
    "IMXD5",
    "IMXD10",
    "IMXD20",
    "IMXD30",
    "IMXD60",
    "CORR5",
    "CORR10",
    "CORR20",
    "CORR30",
    "CORR60",
    "CORD5",
    "CORD10",
    "CORD20",
    "CORD30",
    "CORD60",
    "CNTP5",
    "CNTP10",
    "CNTP20",
    "CNTP30",
    "CNTP60",
    "CNTN5",
    "CNTN10",
    "CNTN20",
    "CNTN30",
    "CNTN60",
    "CNTD5",
    "CNTD10",
    "CNTD20",
    "CNTD30",
    "CNTD60",
    "SUMP5",
    "SUMP10",
    "SUMP20",
    "SUMP30",
    "SUMP60",
    "SUMN5",
    "SUMN10",
    "SUMN20",
    "SUMN30",
    "SUMN60",
    "SUMD5",
    "SUMD10",
    "SUMD20",
    "SUMD30",
    "SUMD60",
    "VMA5",
    "VMA10",
    "VMA20",
    "VMA30",
    "VMA60",
    "VSTD5",
    "VSTD10",
    "VSTD20",
    "VSTD30",
    "VSTD60",
    "WVMA5",
    "WVMA10",
    "WVMA20",
    "WVMA30",
    "WVMA60",
    "VSUMP5",
    "VSUMP10",
    "VSUMP20",
    "VSUMP30",
    "VSUMP60",
    "VSUMN5",
    "VSUMN10",
    "VSUMN20",
    "VSUMN30",
    "VSUMN60",
    "VSUMD5",
    "VSUMD10",
    "VSUMD20",
    "VSUMD30",
    "VSUMD60",
    "ATR14",
    "BOLLINGER_LOWER",
    "BOLLINGER_MIDDLE",
    "BOLLINGER_UPPER",
    "KDJ_D",
    "KDJ_K",
    "MA50",
    "MACD",
    "MACD_HIST",
    "MACD_SIGNAL",
    "RSI14",
    "price_change",
    "price_change_5d",
    "price_change_20d",
    "volume_change",
    "volume_ma_ratio",
    "volatility_5d",
    "volatility_20d",
    "price_position",
]

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
)

#: 面板因子目录（绝对路径）。默认用 features_full（干净、无重复日期、PIT 正确）
DEFAULT_PANEL_DIR = os.path.join(_PROJECT_ROOT, "data", "qlib_data", "features_full")
#: 旧面板目录（block1+block2 拼接，日期重复）—— 仅用于 A/B 对照
LEGACY_PANEL_DIR = os.path.join(_PROJECT_ROOT, "data", "qlib_data", "features", "day")


class GbmLabStrategy(BaseStrategy):
    """可配置的模型类研究策略（GBM Lab）。

    所有维度都通过 strategy_config 传入，便于搜索脚本逐维扫描。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("GbmLab", config)

        self.topk = int(config.get("topk", 60))
        self.n_drop = int(config.get("n_drop", config.get("ndrop", 5)))
        self.hold_thresh = int(config.get("hold_thresh", 0))
        self.rebalance_days = str(config.get("rebalance_days", "weekly")).lower()
        self.rebalance_weekday = int(config.get("rebalance_weekday", 4))
        self.limit_up = float(config.get("limit_up", 0.095))
        self.score_scale = float(config.get("score_scale", 10.0))

        # ── D1 lag（真实盘口径的关键）────────────────────────────────
        self.lag = int(config.get("lag", 1))

        # ── 维度 1: 预测标签 ──────────────────────────────────────────
        self.horizon = int(config.get("horizon", 1))
        # ── D6 标签锚点: "signal" = close[t+h]/close[t]; "trade" = close[t+lag+h]/close[t+lag]
        self.label_anchor = str(config.get("label_anchor", "trade")).lower()
        # ── 维度 2: 目标变量 ──────────────────────────────────────────
        self.target = str(config.get("target", "cs_z")).lower()

        # ── 维度 3: 模型超参 ──────────────────────────────────────────
        self.model = str(config.get("model", "lgbm")).lower()
        self.n_estimators = int(config.get("n_estimators", 100))
        self.max_depth = int(config.get("max_depth", 3))
        self.learning_rate = float(config.get("learning_rate", 0.05))
        self.num_leaves = int(config.get("num_leaves", 15))
        self.min_child_samples = int(config.get("min_child_samples", 20))
        self.subsample = float(config.get("subsample", 0.8))
        self.subsample_freq = int(config.get("subsample_freq", 1))  # D3: 默认真正生效
        self.colsample_bytree = float(config.get("colsample_bytree", 0.8))
        self.reg_alpha = float(config.get("reg_alpha", 0.0))
        self.reg_lambda = float(config.get("reg_lambda", 0.0))
        self.n_jobs = int(config.get("n_jobs", 2))

        # ── 维度 4: 预测频率 + 重训窗口 ──────────────────────────────
        self.retrain = str(config.get("retrain", "monthly")).lower()
        self.train_window = str(config.get("train_window", "expand")).lower()
        self.train_window_days = int(config.get("train_window_days", 252))
        self.warmup_months = int(config.get("warmup_months", 2))
        self.train_stride = int(config.get("train_stride", 1))

        # ── 维度 6: 特征集 ────────────────────────────────────────────
        self.feature_set = str(config.get("feature_set", "f42")).lower()
        self.top_n_factors = int(config.get("top_n_factors", 0))
        self.panel_dir = str(config.get("panel_dir", DEFAULT_PANEL_DIR))
        self.use_presence = bool(config.get("use_presence", False))

        # ── 组合层波动率目标 ─────────────────────────────────────────
        self.target_vol = float(config.get("target_vol", 0.20))
        self.vt_window = int(config.get("vt_window", 20))

        # ── D5 A/B：涨停掩码口径。legacy = 复现 gbm_voltgt 的 bug ────
        # legacy: 传整数列名（close.columns 是 RangeIndex）→ limit_pct_array
        #   全部落 10%、ST 全漏。fixed: 传真实代码 + ST 标记。
        self.limit_mask_mode = str(config.get("limit_mask_mode", "fixed")).lower()

        # ── 维度 9: 与等权因子分集成（blend = 模型分权重）─────────────
        self.blend = float(config.get("blend", 1.0))

        # ── 其他 ─────────────────────────────────────────────────────
        self.purge = bool(config.get("purge", True))  # D2: 默认 purge
        self.seed = int(config.get("seed", 42))

        cash_reserve = float(config.get("cash_reserve_ratio", 0.05))
        self.max_position_size = float(
            config.get("max_position_size", (1.0 - cash_reserve) / self.topk)
        )
        if self.topk <= 0:
            raise ValueError("gbm_lab 策略要求 topk > 0")
        if self.n_drop <= 0:
            raise ValueError("gbm_lab 策略要求 n_drop > 0")

        logger.info(
            f"gbm_lab topk={self.topk} n_drop={self.n_drop} lag={self.lag} "
            f"horizon={self.horizon} anchor={self.label_anchor} "
            f"target={self.target} model={self.model} "
            f"retrain={self.retrain} window={self.train_window} "
            f"feat={self.feature_set} blend={self.blend} "
            f"vt={self.target_vol} panel={os.path.basename(self.panel_dir)}"
        )

    # ── TopK/Dropout 执行契约 ──────────────────────────────────────
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

    # ── 面板因子列 ──────────────────────────────────────────────────
    def _panel_columns(self) -> List[str]:
        fs = self.feature_set
        if fs in ("f25", "formulas"):
            return []
        if fs in ("f42", "42"):
            return list(_ALPHAPANEL_FACTORS.keys())
        if fs in ("alpha158", "a158", "f158"):
            return list(_ALPHA158_FULL)
        if fs in ("panel17", "p17"):
            return list(_ALPHAPANEL_FACTORS.keys())
        return list(_ALPHAPANEL_FACTORS.keys())

    # ── 特征构建 ────────────────────────────────────────────────────
    def _build_features(
        self,
        stock_data: Dict[str, pd.DataFrame],
        codes: List[str],
        master: pd.DatetimeIndex,
    ) -> Dict[str, Any]:
        """构造 (n_dates, n_stocks) 的原始因子矩阵 + close/vol 矩阵。"""
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
            for n in (5, 10, 20, 30, 60):
                cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
                cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
                cands[f"STD{n}"] = close.rolling(n).std() / close
                cands[f"CORR{n}"] = close.rolling(n).corr(vol)
                cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)

        # ── 面板因子（D4：只读需要的列，缓存每股 frame）──────────────
        panel_cols = self._panel_columns()
        raw: List[np.ndarray] = []
        names: List[str] = []
        signs: List[float] = []
        for fname in _HONEST_FORMULAS:
            raw.append(cands[fname].to_numpy(dtype=np.float64))
            names.append(fname)
            signs.append(_HONEST_FORMULAS[fname])

        if panel_cols:
            panel_stack = self._load_panel_stack(codes, master, panel_cols)
            for idx, fname in enumerate(panel_cols):
                raw.append(panel_stack[:, :, idx])
                names.append(fname)
                signs.append(_ALPHAPANEL_FACTORS.get(fname, 1.0))

        return {
            "raw": raw,
            "names": names,
            "signs": signs,
            "close": close,
            "vol": vol,
        }

    def _load_panel_stack(
        self,
        codes: List[str],
        master: pd.DatetimeIndex,
        panel_cols: List[str],
    ) -> np.ndarray:
        """一次性读取全部 parquet 的面板列 → (n_dates, n_stocks, n_panel) float32。

        只读需要的列（不是全表 188 列），且每只股票只读一次 parquet。
        parquet 的索引是 MultiIndex(stock_code, date) → 需要 reset_index 取 date。
        """
        n_dates, n_stocks = len(master), len(codes)
        out = np.full((n_dates, n_stocks, len(panel_cols)), np.nan, dtype=np.float32)
        start, end = self._start, self._end
        for j, code in enumerate(codes):
            path = os.path.join(self.panel_dir, code.replace(".", "_") + ".parquet")
            try:
                pf = pd.read_parquet(path, columns=["date"] + panel_cols)
            except Exception:
                continue
            try:
                pf = pf.reset_index()
                if "date" not in pf.columns:
                    continue
                pf["date"] = pd.to_datetime(pf["date"])
                pf = pf.set_index("date").sort_index()
                pf = pf.loc[pd.Timestamp(start) : pd.Timestamp(end)]
                pf = pf[~pf.index.duplicated(keep="first")]
                missing = [c for c in panel_cols if c not in pf.columns]
                if missing:
                    continue
                pos = master.get_indexer(pd.DatetimeIndex(pf.index))
                ok = pos >= 0
                if not ok.any():
                    continue
                block = pf[panel_cols].to_numpy(dtype=np.float64, na_value=np.nan)
                out[pos[ok], j, :] = block[ok].astype(np.float32)
            except Exception:
                continue
        return out

    # ── 目标变量 ────────────────────────────────────────────────────
    def _make_target(
        self,
        close: pd.DataFrame,
        fwd: np.ndarray,
        vol: pd.DataFrame,
    ) -> np.ndarray:
        """把前瞻收益转成训练目标。

        target:
          raw        : 原始前瞻收益
          rank       : 逐日横截面收益排名分位（0~1），消除极端值影响
          cs_z       : 逐日横截面 z 分数（去市场 beta，保留相对强弱幅度）
          ind_neutral: 收益 - 行业平均收益
          vol_std    : 收益 / 20 日波动率（波动率标准化）
        """
        if self.target == "raw":
            return fwd
        df = pd.DataFrame(fwd, index=close.index, columns=close.columns)
        if self.target == "rank":
            return df.rank(axis=1, pct=True, na_option="keep").to_numpy(
                dtype=np.float64
            )
        if self.target == "cs_z":
            mu = df.mean(axis=1)
            sd = df.std(axis=1)
            out = df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
            return out.to_numpy(dtype=np.float64)
        if self.target == "ind_neutral":
            imap = load_industry_map()
            ids = industry_ids(list(close.columns), imap)
            out = fwd.copy()
            if len(ids) and ids.max() >= 0:
                for k in np.unique(ids):
                    if k < 0:
                        continue
                    cols = np.where(ids == k)[0]
                    out[:, cols] -= np.nanmean(fwd[:, cols], axis=1, keepdims=True)
            return out
        if self.target == "vol_std":
            v = (
                vol.rolling(self.vt_window, min_periods=5)
                .std()
                .to_numpy(dtype=np.float64)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                out = fwd / np.where(v > 1e-12, v, np.nan)
            return out
        return fwd

    # ── 模型 ────────────────────────────────────────────────────────
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """训练模型并返回已 fit 的模型对象。"""
        if self.model in ("ridge", "elasticnet"):
            from sklearn.linear_model import ElasticNet, Ridge

            if self.model == "ridge":
                mdl = Ridge(alpha=float(self.reg_alpha or 1.0), random_state=self.seed)
            else:
                mdl = ElasticNet(
                    alpha=float(self.reg_alpha or 0.01),
                    l1_ratio=float(self.reg_lambda or 0.5),
                    max_iter=200,
                    random_state=self.seed,
                    n_jobs=self.n_jobs,
                )
            mdl.fit(X, y)
            return mdl
        if self.model == "xgb":
            import xgboost as xgb

            mdl = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.seed,
                n_jobs=self.n_jobs,
                tree_method="hist",
            )
            mdl.fit(X, y)
            return mdl
        import lightgbm as lgb

        mdl = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,  # D3
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            verbosity=-1,
            n_jobs=self.n_jobs,
        )
        mdl.fit(X, y)
        return mdl

    # ── 重训切点 ────────────────────────────────────────────────────
    def _retrain_points(self, master: pd.DatetimeIndex) -> List[int]:
        """返回每个重训切点的起始行号 k0。"""
        n = len(master)
        if self.retrain == "weekly":
            return list(range(0, n, 5))
        if self.retrain == "quarterly":
            pts: List[int] = []
            prev = None
            for k, ts in enumerate(master):
                key = (ts.year, (ts.month - 1) // 3)
                if key != prev:
                    pts.append(k)
                    prev = key
            return pts
        # monthly
        pts = []
        prev = None
        for k, ts in enumerate(master):
            key = (ts.year, ts.month)
            if key != prev:
                pts.append(k)
                prev = key
        return pts

    # ── 主流程 ──────────────────────────────────────────────────────
    async def prepare_backtest_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        single_model_from: Optional[datetime] = None,
    ) -> None:
        self._start = start_date
        self._end = end_date

        codes = [
            code
            for code, df in stock_data.items()
            if df is not None
            and not df.empty
            and "close" in df.columns
            and "volume" in df.columns
        ]
        if not codes:
            logger.warning("gbm_lab: 无可用股票数据，跳过打分")
            return

        indexes = [pd.DatetimeIndex(stock_data[c].index) for c in codes]
        master = indexes[0].append(indexes[1:]) if len(indexes) > 1 else indexes[0]
        master = master.unique().sort_values()
        n_dates, n_stocks = len(master), len(codes)

        feats = self._build_features(stock_data, codes, master)
        raw = feats["raw"]
        names = feats["names"]
        signs = feats["signs"]
        close = feats["close"]
        vol = feats["vol"]

        # ── 标签 ────────────────────────────────────────────────────
        anchor = self.lag if self.label_anchor == "trade" else 0
        with np.errstate(divide="ignore", invalid="ignore"):
            if anchor > 0:
                fwd = (
                    close.shift(-(anchor + self.horizon)) / close.shift(-anchor) - 1.0
                ).to_numpy(dtype=np.float64)
            else:
                fwd = (close.shift(-self.horizon) / close - 1.0).to_numpy(
                    dtype=np.float64
                )
        target = self._make_target(close, fwd, vol)

        # ── 特征集选择（ic_topN）────────────────────────────────────
        if self.top_n_factors > 0:
            keep_idx = self._select_by_ic(raw, names, target)
            raw = [raw[i] for i in keep_idx]
            names = [names[i] for i in keep_idx]
            signs = [signs[i] for i in keep_idx]
            logger.info(f"gbm_lab IC选因子: top{self.top_n_factors} = {names[:10]}...")

        n_fac = len(raw)

        # ── 横截面 z 分数 ────────────────────────────────────────────
        # pres_all 只在 use_presence=True 时才需要（省 640MB 内存/worker）
        z_all = np.zeros((n_dates, n_stocks, n_fac), dtype=np.float32)
        if self.use_presence:
            pres_all = np.zeros((n_dates, n_stocks, n_fac), dtype=np.float32)
        else:
            pres_all = None
        min_cross = max(2, min(30, n_stocks // 10))
        # 等权复合分的累加器（边算 z 边累加，省掉整个 pres_all）
        acc = np.zeros((n_dates, n_stocks), dtype=np.float64)
        wsum = np.zeros((n_dates, n_stocks), dtype=np.float64)
        for k in range(n_fac):
            mat = raw[k]
            ok = np.isfinite(mat)
            cnt = ok.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
                sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
                z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan)
            z = np.where(cnt >= min_cross, z, 0.0)
            z = np.where(np.isfinite(z), z, 0.0)
            z_all[:, :, k] = z
            if pres_all is not None:
                pres_all[:, :, k] = ok
            # 等权复合分（nan-aware，sign 由因子方向决定）
            sgn = signs[k]
            acc[ok] += z[ok] * sgn
            wsum[ok] += 1
        comp_eq = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1), np.nan)
        del acc, wsum, raw

        # ── 涨停掩码（D5：真实代码 + ST 标记）──────────────────────
        day_ret = close.pct_change(fill_method=None).to_numpy(dtype=np.float64)
        if self.limit_mask_mode == "legacy":
            # 复现 gbm_voltgt / honest_composite 的 bug：close.columns 是
            # RangeIndex(0..n-1) 的整数 → limit_pct_array 按 str(code)[:2]
            # 匹配板块前缀全部落 10%，ST 标记也全漏。
            limit_mask = limit_up_mask_matrix(day_ret, list(range(n_stocks)))
        else:
            limit_mask = limit_up_mask_matrix(
                day_ret, codes, is_st=load_st_flags(codes)
            )

        # ── 滚动重训 ───────────────────────────────────────────────
        pts = self._retrain_points(master)
        comp = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)

        if single_model_from is not None:
            target_ts = pd.Timestamp(single_model_from)
            k0 = int(np.searchsorted(master.values, target_ts.to_datetime64()))
            k0 = max(k0, 1)
            prev_starts = [k for k in pts if k <= k0]
            if prev_starts:
                k0 = prev_starts[-1]
            self._train_and_score(
                z_all, pres_all, target, k0, n_dates, comp, master, None
            )
        else:
            for mi, k0 in enumerate(pts):
                k1 = pts[mi + 1] if mi + 1 < len(pts) else n_dates
                if mi < self.warmup_months:
                    comp[k0:k1] = comp_eq[k0:k1]
                    continue
                self._train_and_score(z_all, pres_all, target, k0, k1, comp, master, mi)

        # ── 与等权因子分集成（维度 9）────────────────────────────────
        if self.blend < 1.0:
            z_model = _row_zscore(comp)
            z_eq = _row_zscore(comp_eq)
            both = np.isfinite(z_model) & np.isfinite(z_eq)
            mixed = np.full_like(comp, np.nan)
            mixed[both] = self.blend * z_model[both] + (1.0 - self.blend) * z_eq[both]
            only_eq = ~both & np.isfinite(z_eq)
            only_md = ~both & np.isfinite(z_model)
            mixed[only_eq] = z_eq[only_eq]
            mixed[only_md] = z_model[only_md]
            comp = mixed

        comp[limit_mask] = np.nan

        # ── 调仓日掩码 ──────────────────────────────────────────────
        if self.rebalance_days == "weekly":
            comp[master.dayofweek != self.rebalance_weekday] = np.nan
        elif self.rebalance_days == "monthly":
            comp[master.day != 1] = np.nan

        # ── D1 lag：信号整体后移 lag 个交易日 ─────────────────────────
        if self.lag > 0:
            lagged = np.full_like(comp, np.nan)
            if self.lag < n_dates:
                lagged[self.lag :] = comp[: n_dates - self.lag]
            comp = lagged

        # ── 波动率目标仓位（gbm_voltgt 口径：高分保留 topk*scale 只）──
        if self.target_vol > 0:
            mkt = close.mean(axis=1, skipna=True)
            mkt_ret = mkt.pct_change(fill_method=None)
            mkt_vol_ann = mkt_ret.rolling(self.vt_window).std() * np.sqrt(252)
            scale = np.minimum(1.0, (self.target_vol / mkt_vol_ann).to_numpy())
            scale[~np.isfinite(scale)] = 1.0
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

        # ── 存分数 ──────────────────────────────────────────────────
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
            f"gbm_lab 打分完成: 股票={n_stocks}, 日期={n_dates}, "
            f"有效分数={stored}, 特征={n_fac}, lag={self.lag}, "
            f"horizon={self.horizon}, anchor={self.label_anchor}, "
            f"target={self.target}, model={self.model}"
        )

    def _select_by_ic(
        self, raw: List[np.ndarray], names: List[str], target: np.ndarray
    ) -> List[int]:
        """按逐日 rank-IC 选 topN 因子（只用当日及之前的信息）。"""
        ics: List[tuple] = []
        for k, mat in enumerate(raw):
            df_a = pd.DataFrame(mat)
            df_b = pd.DataFrame(target)
            ra = df_a.rank(axis=1, na_option="keep")
            rb = df_b.rank(axis=1, na_option="keep")
            a = ra.to_numpy(dtype=np.float64)
            b = rb.to_numpy(dtype=np.float64)
            ok = np.isfinite(a) & np.isfinite(b)
            cnt = ok.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                ac = np.where(ok, a, 0.0)
                bc = np.where(ok, b, 0.0)
                ac = ac - ac.sum(axis=1, keepdims=True) / np.maximum(cnt, 1)[:, None]
                bc = bc - bc.sum(axis=1, keepdims=True) / np.maximum(cnt, 1)[:, None]
                num = (ac * bc).sum(axis=1)
                den = np.sqrt((ac**2).sum(axis=1) * (bc**2).sum(axis=1))
                ic = np.where(den > 1e-12, num / den, np.nan)
            good = np.isfinite(ic) & (cnt >= 50)
            if good.sum() == 0:
                continue
            ics.append((float(np.nanmean(ic[good])), k))
        ics.sort(key=lambda x: -abs(x[0]))
        keep = [k for _, k in ics[: self.top_n_factors]]
        return keep

    def _train_and_score(
        self,
        z_all: np.ndarray,
        pres_all: Optional[np.ndarray],
        target: np.ndarray,
        k0: int,
        k1: int,
        comp: np.ndarray,
        master: pd.DatetimeIndex,
        mi: Optional[int],
    ) -> None:
        """在切点 k0 训练模型，给 [k0, k1) 打分。

        purge（D2）：训练行 t 的标签在 close[t + anchor + horizon] 才实现
        （anchor = lag 用 trade 锚点时）。模型在切点 k0 拟合时只能看到
        已实现的标签 → 要求 t + anchor + horizon <= k0 - 1。
        """
        anchor = self.lag if self.label_anchor == "trade" else 0
        if self.purge:
            train_end = k0 - anchor - self.horizon
        else:
            train_end = k0
        if train_end <= 0:
            comp[k0:k1] = np.nan
            return
        if self.train_window == "rolling":
            lo = max(0, train_end - self.train_window_days)
        else:
            lo = 0
        if self.train_stride > 1:
            lo = lo - (lo % self.train_stride)

        z_blk = z_all[lo:train_end]
        t_blk = target[lo:train_end]
        if self.use_presence and pres_all is not None:
            p_blk = pres_all[lo:train_end]
            X = np.concatenate(
                [z_blk.reshape(-1, z_blk.shape[2]), p_blk.reshape(-1, p_blk.shape[2])],
                axis=1,
            ).astype(np.float32, copy=False)
        else:
            X = z_blk.reshape(-1, z_blk.shape[2]).astype(np.float32, copy=False)
        y = t_blk.reshape(-1)

        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() < 500:
            comp[k0:k1] = np.nan
            return
        Xv = X[valid]
        yv = y[valid]
        del X, valid

        try:
            mdl = self._fit_model(Xv, yv)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"gbm_lab 模型训练失败({exc})，该段置空")
            comp[k0:k1] = np.nan
            return
        del Xv, yv

        # 用训练好的模型给**预测窗口** [k0, k1) 打分
        z_pred = z_all[k0:k1]
        if self.use_presence and pres_all is not None:
            p_pred = pres_all[k0:k1]
            Xp = np.concatenate(
                [
                    z_pred.reshape(-1, z_pred.shape[2]),
                    p_pred.reshape(-1, p_pred.shape[2]),
                ],
                axis=1,
            ).astype(np.float32, copy=False)
        else:
            Xp = z_pred.reshape(-1, z_pred.shape[2]).astype(np.float32, copy=False)
        try:
            pred = np.asarray(mdl.predict(Xp), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"gbm_lab 模型预测失败({exc})，该段置空")
            comp[k0:k1] = np.nan
            return
        comp[k0:k1] = pred.reshape(k1 - k0, -1)

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
                    f"gbm_lab 模型分 {score:.6f} "
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


def _row_zscore(a: np.ndarray) -> np.ndarray:
    """逐行（逐日）横截面 z 分数，NaN 保留。"""
    out = np.full(a.shape, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a, axis=1, keepdims=True)
    good = np.isfinite(a) & np.isfinite(sd) & (sd > 1e-12)
    rows, cols = np.nonzero(good)
    if len(rows):
        out[rows, cols] = (
            a[rows, cols] - np.broadcast_to(mu, a.shape)[rows, cols]
        ) / np.broadcast_to(sd, a.shape)[rows, cols]
    return out
