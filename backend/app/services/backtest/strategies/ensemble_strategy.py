"""多策略信号合成策略（ensemble）—— 合成腿分数 → 单一横截面排名分 → TopK-Dropout。

## 目的
把多个已注册策略的**每日横截面分数**合成一个综合分，再做 TopK/Dropout 组合管理。
动机见 backend/scripts/verify_ensemble.py 的实测：
  - 42f 与 rsi 的分数 RankIC 相关 0.771（高相关 → 等权平均 ≈ 42f 本身，合成没意义）
  - gbm（LightGBM 元模型）与 42f 相关仅 0.112（近乎正交）→ 是真正的增量信息源
  - 但 gbm 自身 IC 很弱（ICIR 0.155）→ 必须靠合成 + 相关性惩罚把权重压准

## 合成方式（config 切换）
  weights          : {腿名: 权重}，缺省 = 等权
  normalize        : "z" | "rank" | "winsor" | "none"
  weight_mode      : "fixed" | "icir" | "corr_penalty"
                      icir          : 逐日滚动 RankICIR 加权（walk-forward，无前视）
                      corr_penalty  : w ∝ IR/(1+ρ)，ρ 为腿间平均横截面相关
  icir_window      : 滚动 IC 窗口（交易日），默认 120
  icir_smooth      : ICIR 序列的时序平滑窗口，默认 20
  legs             : 参与合成的腿，默认 ["42f", "rsi", "fr", "gbm"]

## 腿的分数从哪来
本策略**不重算**各腿的因子，而是复用各腿策略类的 ``prepare_backtest_data``：
调用它们的打分逻辑，把 ``stock_data[code].attrs[_SCORES_ATTR]`` 里的分数取回来，
统一做截面标准化后按权重合成。这样保证「合成用的分数」与「各腿单独跑时用的分数」
**逐股完全一致**（不引入第二套因子实现）。

## 执行契约
与其它横截面排名策略一致：``trade_mode="topk_dropout"``，
config 提供 topk / n_drop / hold_thresh / max_position_size。

## 注册（**需要改注册表，本次未改**）
在 ``backend/app/services/backtest/strategies/strategy_factory.py`` 里加两行：

    from .ensemble_strategy import EnsembleStrategy
    ...
    _strategies = {
        ...
        "ensemble": EnsembleStrategy,     # ← 加这一行
    }

用法：
    verify_10y_pit.py --years 3 --lag 1 --only ensemble --out ensemble_3y.json
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from app.services.trading.execution_rules import (
    limit_up_mask_matrix,
    load_st_flags,
)

from ..core.base_strategy import BaseStrategy
from ..models import SignalType, TradingSignal

_SCORES_ATTR = "_ensemble_scores"

# 腿名 → (策略模块路径, 类名, 分数 attr, 打分用 config)
_LEG_DEFS: Dict[str, Dict[str, Any]] = {
    # ⚠️ use_panel 必须**显式钉死**。
    # honest_composite 的因子集由 `use_panel` 决定：
    #   False → 25 个公式因子（2026-10-27 02:13 之后的主线默认）
    #   True  → 25 公式 + 17 个 Alpha158 面板因子 = 42 因子
    # 2026-10-27 实测：主线默认值从 42 因子切到 25 因子的那一次，
    # 同一个合成配置（42f+mf, winsor3, k30, d5）的引擎夏普从 1.2817 掉到 1.1081。
    # 不写死 → 主线改默认就会让本策略的行为静默漂移，历史回测不可复现。
    "42f": {
        "module": "app.services.backtest.strategies.honest_composite_strategy",
        "cls": "HonestCompositeStrategy",
        "attr": "_honest_composite_scores",
        "cfg": {
            "topk": 60,
            "n_drop": 5,
            "rebalance_days": "daily",
            "lag": 0,
            "use_panel": False,
        },
    },
    "42f42": {
        "module": "app.services.backtest.strategies.honest_composite_strategy",
        "cls": "HonestCompositeStrategy",
        "attr": "_honest_composite_scores",
        "cfg": {
            "topk": 60,
            "n_drop": 5,
            "rebalance_days": "daily",
            "lag": 0,
            "use_panel": True,
        },
    },
    "rsi": {
        "module": "app.services.backtest.strategies.rsi_71421_strategy",
        "cls": "Rsi71421Strategy",
        "attr": "_rsi_71421_scores",
        "cfg": {"topk": 30, "n_drop": 3, "rebalance_days": "daily", "lag": 0},
    },
    "fr": {
        "module": "app.services.backtest.strategies.factor_ranking_strategy",
        "cls": "FactorRankingStrategy",
        "attr": "_factor_ranking_scores",
        "cfg": {"topk": 50, "n_drop": 5, "rebalance_days": "daily", "lag": 0},
    },
    "gbm": {
        "module": "app.services.backtest.strategies.gbm_voltgt_strategy",
        "cls": "GbmVoltgtStrategy",
        "attr": "_gbm_voltgt_scores",
        "cfg": {
            "topk": 60,
            "n_drop": 5,
            "rebalance_days": "daily",
            "lag": 0,
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "warmup_months": 2,
            "target_vol": 0.20,
            "vt_window": 20,
        },
    },
    "mf": {
        "module": "app.services.backtest.strategies.strategies",
        "cls": "MultiFactorStrategy",
        "attr": "_multi_factor_scores",
        "cfg": {"topk": 30, "n_drop": 3, "rebalance_days": "daily", "lag": 0},
    },
}


def _import_leg_cls(module_path: str, cls_name: str):
    mod = __import__(module_path, fromlist=[cls_name])
    return getattr(mod, cls_name)


def cs_zscore(mat: np.ndarray, min_cross: int = 30) -> np.ndarray:
    """逐日截面 z-score（nan-aware）。"""
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan)
    return np.where(cnt >= min_cross, z, np.nan)


def cs_rank(mat: np.ndarray, min_cross: int = 30) -> np.ndarray:
    """逐日截面 rank → 均匀分布 [-0.5, 0.5]（对异常值稳健）。"""
    df = pd.DataFrame(mat)
    cnt = df.notna().sum(axis=1)
    r = df.rank(axis=1, na_option="keep")
    n = df.notna().sum(axis=1).replace(0, np.nan)
    out = ((r - 0.5) / n.to_numpy()[:, None]).to_numpy()
    out[(cnt < min_cross).to_numpy()] = np.nan
    return out


def cs_winsor_z(mat: np.ndarray, min_cross: int = 30, clip: float = 3.0) -> np.ndarray:
    z = cs_zscore(mat, min_cross)
    return np.where(np.isfinite(z), np.clip(z, -clip, clip), np.nan)


def _normalize(mat: np.ndarray, mode: str) -> np.ndarray:
    if mode == "z":
        return cs_zscore(mat)
    if mode == "rank":
        return cs_rank(mat)
    if mode == "winsor":
        return cs_winsor_z(mat)
    if mode == "none":
        return mat
    raise ValueError(f"ensemble: 未知 normalize={mode}")


class EnsembleStrategy(BaseStrategy):
    """多策略分数合成 → 单一横截面排名分 → TopK/Dropout。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("Ensemble", config)
        self.topk = int(config.get("topk", 60))
        self.n_drop = int(config.get("n_drop", config.get("ndrop", 5)))
        self.hold_thresh = int(config.get("hold_thresh", 0))
        self.lag = int(config.get("lag", 0))
        self.rebalance_days = str(config.get("rebalance_days", "daily")).lower()
        self.rebalance_weekday = int(config.get("rebalance_weekday", 4))
        self.limit_up = float(config.get("limit_up", 0.095))
        self.score_scale = float(config.get("score_scale", 10.0))
        self.normalize = str(config.get("normalize", "z")).lower()
        self.weight_mode = str(config.get("weight_mode", "fixed")).lower()
        self.icir_window = int(config.get("icir_window", 120))
        self.icir_min_periods = int(config.get("icir_min_periods", 40))
        self.icir_smooth = int(config.get("icir_smooth", 20))
        self.min_cross = int(config.get("min_cross", 30))
        self.residualize = bool(config.get("residualize", False))
        self.leg_weights: Dict[str, float] = {
            k: float(v) for k, v in (config.get("weights") or {}).items()
        }
        self.legs: List[str] = list(config.get("legs") or ["42f", "rsi", "fr", "gbm"])
        unknown = [x for x in self.legs if x not in _LEG_DEFS]
        if unknown:
            raise ValueError(f"ensemble: 未知腿 {unknown}，可用 {list(_LEG_DEFS)}")
        if self.topk <= 0 or self.n_drop <= 0:
            raise ValueError("ensemble 策略要求 topk > 0 且 n_drop > 0")

        cash_reserve = float(config.get("cash_reserve_ratio", 0.05))
        self.max_position_size = float(
            config.get("max_position_size", (1.0 - cash_reserve) / self.topk)
        )

        logger.info(
            f"ensemble legs={self.legs} weights={self.leg_weights or '等权'} "
            f"normalize={self.normalize} weight_mode={self.weight_mode} "
            f"topk={self.topk} n_drop={self.n_drop} lag={self.lag}"
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
        return self.max_position_size

    # ── 打分 ─────────────────────────────────────────────────────────
    async def prepare_backtest_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        codes = [
            c
            for c, df in stock_data.items()
            if df is not None and not df.empty and "close" in df.columns
        ]
        if not codes:
            logger.warning("ensemble: 无可用股票数据，跳过打分")
            return

        indexes = [pd.DatetimeIndex(stock_data[c].index) for c in codes]
        master = (
            indexes[0].append(indexes[1:]).unique().sort_values()
            if len(indexes) > 1
            else indexes[0]
        )
        master = pd.DatetimeIndex(master)
        n_dates, n_stocks = len(master), len(codes)
        pos_of = {d: i for i, d in enumerate(master)}

        # ---- 0. close 矩阵（IC 标签 + 涨停掩码都要用）----
        close_early = pd.DataFrame(
            np.full((n_dates, n_stocks), np.nan),
            index=master,
            columns=codes,
            dtype="float64",
        )
        for j, c in enumerate(codes):
            df = stock_data[c]
            pos = master.get_indexer(pd.DatetimeIndex(df.index))
            ok = pos >= 0
            vals = pd.to_numeric(df["close"], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            close_early.iloc[pos[ok], j] = vals[ok]

        # ---- 1. 逐条腿调用其策略类打分，取回分数矩阵 ----

        leg_scores: Dict[str, np.ndarray] = {}
        for leg in self.legs:
            spec = _LEG_DEFS[leg]
            cls = _import_leg_cls(spec["module"], spec["cls"])
            strat = cls(dict(spec["cfg"]))
            # 各腿用 daily + lag=0 打分 → 掩码与滞后统一由本策略处理
            await strat.prepare_backtest_data(stock_data, start_date, end_date)
            mat = np.full((n_dates, n_stocks), np.nan)
            got = 0
            for j, c in enumerate(codes):
                sm = stock_data[c].attrs.get(spec["attr"]) or {}
                for d, v in sm.items():
                    i = pos_of.get(pd.Timestamp(d))
                    if i is not None and np.isfinite(v):
                        mat[i, j] = v
                        got += 1
            leg_scores[leg] = mat
            logger.info(f"ensemble 腿 {leg}: 有效分数 {got}")

        if not leg_scores:
            logger.warning("ensemble: 没有可用腿分数")
            return

        # ---- 2. 权重 ----
        weights, daily_w = self._compute_weights(leg_scores, master, close_early)

        # ---- 2b. 残差正交化（可选）----
        # 把除第一腿外的每条腿对第一腿做截面回归取残差 → 得到"第一腿解释不了
        # 的那部分"，再等权合成 → 构造低相关 alpha。
        if self.residualize and len(leg_scores) >= 2:
            names = list(leg_scores)
            base = names[0]
            zbase = _normalize(leg_scores[base], "z")
            new_scores: Dict[str, np.ndarray] = {base: leg_scores[base]}
            for other in names[1:]:
                zo = _normalize(leg_scores[other], "z")
                new_scores[other] = self._cs_resid(zo, zbase)
            leg_scores = new_scores

        # ---- 3. 合成 ----
        total = None
        wsum = None
        for leg, w in weights.items():
            if w == 0 or leg not in leg_scores:
                continue
            m = _normalize(leg_scores[leg], self.normalize)
            good = np.isfinite(m)
            if daily_w is not None:
                # 逐日权重：形状 (n_dates, 1)，广播到整个截面
                # 注意取**绝对值**（权重用 |ICIR|，方向由腿自身的符号决定）
                wd = np.abs(daily_w[leg])[:, None]
                contrib = np.where(good & np.isfinite(wd), m * wd, 0.0)
                cnt = np.where(good & np.isfinite(wd), wd, 0.0)
            else:
                contrib = np.where(good, m * w, 0.0)
                cnt = np.where(good, abs(w), 0.0)
            total = contrib if total is None else total + contrib
            wsum = cnt if wsum is None else wsum + cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            composite = np.where(
                wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan
            )

        # ---- 4. 涨停掩码（合成后再做一次，保证合成分也遵守）----
        day_ret = close_early.pct_change(fill_method=None).to_numpy()
        composite[
            limit_up_mask_matrix(day_ret, codes, is_st=load_st_flags(list(codes)))
        ] = np.nan

        # ---- 5. 调仓日掩码 ----
        if self.rebalance_days == "weekly":
            composite[master.dayofweek != self.rebalance_weekday] = np.nan
        elif self.rebalance_days == "monthly":
            composite[master.day != 1] = np.nan

        # ---- 6. lag ----
        if self.lag > 0:
            lagged = np.full_like(composite, np.nan)
            if self.lag < n_dates:
                lagged[self.lag :] = composite[: n_dates - self.lag]
            composite = lagged

        # ---- 7. 存分数 ----
        stored = 0
        for j, c in enumerate(codes):
            column = composite[:, j]
            score_map: Dict[datetime, float] = {}
            for k in range(n_dates):
                v = column[k]
                if np.isfinite(v):
                    score_map[master[k].to_pydatetime()] = float(v)
                    stored += 1
            stock_data[c].attrs[_SCORES_ATTR] = score_map
        weight_repr = {k: round(float(v), 4) for k, v in weights.items()}
        logger.info(
            f"ensemble 合成完成: 股票={n_stocks}, 日期={n_dates}, "
            f"有效分数={stored}, 权重={weight_repr}"
        )

    # ── 权重计算 ─────────────────────────────────────────────────────
    def _compute_weights(
        self,
        leg_scores: Dict[str, np.ndarray],
        master: pd.DatetimeIndex,
        close: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
        """返回 (标量权重, 逐日权重或 None)。

        - fixed         : 标量权重，逐日权重 = None
        - corr_penalty  : 标量权重 w ∝ IR/(1+ρ)，逐日权重 = None
        - icir          : 标量权重仅用于日志，实际合成用逐日权重
        """
        if self.weight_mode == "fixed":
            if self.leg_weights:
                w = {k: v for k, v in self.leg_weights.items() if k in leg_scores}
            else:
                w = {k: 1.0 for k in leg_scores}
            return w, None

        if self.weight_mode == "corr_penalty":
            ir, rho = self._leg_ir_and_rho(leg_scores, close)
            out: Dict[str, float] = {}
            for k in leg_scores:
                others = [j for j in leg_scores if j != k]
                r = (
                    float(np.mean([rho.get((k, j), 0.0) for j in others]))
                    if others
                    else 0.0
                )
                out[k] = max(ir.get(k, 0.0), 0.0) / (1.0 + max(r, 0.0))
            s = sum(out.values())
            if s <= 0:
                return {k: 1.0 for k in leg_scores}, None
            return {k: v / s for k, v in out.items()}, None

        # icir：逐日权重
        daily = self._rolling_icir_weights(leg_scores, close)
        scalar = {k: float(np.mean(np.abs(daily[k]))) for k in leg_scores}
        return scalar, daily

    def _leg_ir_and_rho(
        self,
        leg_scores: Dict[str, np.ndarray],
        close: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """各腿的 ICIR（对 fwd1）与腿间平均横截面相关。

        注意：这里的 IC 用**全窗口**历史 → 属于「用了未来信息」的静态权重，
        只作探索/对照。真实可用的逐日无前视权重请用 weight_mode="icir"。
        """
        fwd = self._fwd1(close)
        ir: Dict[str, float] = {}
        for k, v in leg_scores.items():
            ics = self._rank_ic_rows(v, fwd)
            s = pd.Series(ics)
            mu, sd = float(s.mean()), float(s.std())
            ir[k] = mu / sd if sd > 0 else 0.0

        ranks = {k: self._rank_rows(v) for k, v in leg_scores.items()}
        rho: Dict[Tuple[str, str], float] = {}
        names = list(leg_scores)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                m = np.isfinite(ranks[a]) & np.isfinite(ranks[b])
                if m.sum() < 30:
                    rho[(a, b)] = rho[(b, a)] = 0.0
                    continue
                c = float(np.corrcoef(ranks[a][m], ranks[b][m])[0, 1])
                rho[(a, b)] = rho[(b, a)] = c
        return ir, rho

    @staticmethod
    def _fwd1(close: pd.DataFrame) -> np.ndarray:
        arr = close.to_numpy(dtype=np.float64)
        out = np.full_like(arr, np.nan)
        if arr.shape[0] > 1:
            out[:-1] = arr[1:] / arr[:-1] - 1.0
        return out

    @staticmethod
    def _cs_resid(y: np.ndarray, x: np.ndarray, min_cross: int = 30) -> np.ndarray:
        """逐日截面回归残差：y 对 x 做 OLS，取残差。"""
        out = np.full(y.shape, np.nan)
        for t in range(y.shape[0]):
            yy, xx = y[t], x[t]
            m = np.isfinite(yy) & np.isfinite(xx)
            if int(m.sum()) < min_cross:
                continue
            A = np.vstack([xx[m], np.ones(int(m.sum()))]).T
            beta, *_ = np.linalg.lstsq(A, yy[m], rcond=None)
            out[t, m] = yy[m] - A @ beta
        return out

    @staticmethod
    def _rank_rows(mat: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(mat)
        return df.rank(axis=1, na_option="keep").to_numpy(dtype=np.float64)

    def _rolling_icir_weights(
        self,
        leg_scores: Dict[str, np.ndarray],
        close: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """逐日滚动 RankICIR 权重（walk-forward）。

        IC 用**对 fwd1 收益**的横截面 RankIC；第 t 日的权重只用 [t-window, t-1]
        的历史 IC（.shift(1)）→ 无前视。ICIR 序列再做一次时序平滑。
        """
        next(iter(leg_scores.values())).shape[0]
        fwd = self._fwd1(close)
        out: Dict[str, np.ndarray] = {}
        for name, sc in leg_scores.items():
            ics = self._rank_ic_rows(sc, fwd)
            s = pd.Series(ics)
            m = s.rolling(self.icir_window, min_periods=self.icir_min_periods).mean()
            sd = s.rolling(self.icir_window, min_periods=self.icir_min_periods).std()
            icir = (m / sd.replace(0.0, np.nan)).shift(1)
            if self.icir_smooth > 1:
                icir = icir.rolling(
                    self.icir_smooth, min_periods=max(3, self.icir_smooth // 3)
                ).mean()
            out[name] = icir.fillna(0.0).to_numpy()
        return out

    @staticmethod
    def _rank_ic_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n_d = a.shape[0]
        out = np.full(n_d, np.nan)
        for t in range(n_d):
            x, y = a[t], b[t]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 30:
                continue
            rx = pd.Series(x[ok]).rank().to_numpy()
            ry = pd.Series(y[ok]).rank().to_numpy()
            if rx.std() > 0 and ry.std() > 0:
                out[t] = float(np.corrcoef(rx, ry)[0, 1])
        return out

    # ── 信号 ─────────────────────────────────────────────────────────
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
        self,
        data: pd.DataFrame,
        current_date: datetime,
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
                    f"多策略合成排名分 {score:.4f} "
                    f"(legs={self.legs}, TopK={self.topk}, n_drop={self.n_drop}, "
                    f"lag={self.lag})"
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
