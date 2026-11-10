"""factor_ranking 的 lag 契约测试（2026-11-06 补）。

背景：回测引擎没有 lag 机制，策略侧必须自己把分数下移。
lag=0（旧默认）= 当日分数当日收盘成交 —— **实盘做不到**，读数系统性偏乐观；
lag=1 = 前一调仓日的分数、当日收盘成交 = 纸面盘口径。
档案里 GBM 的 lag 缺失（CONCLUSION_2026-10-27）就是同一类漏洞，这里给 factor_ranking 补上并锁住。

要证明四件事：
1. lag=1 的出分日 = lag=0 的出分日各自往后挪一个交易日（不是挪 7 天、不是原地不动）；
2. lag=1 第一天没有分数（不可能拿到未来分数）；
3. 不传 lag 时行为与 lag=0 **完全一致**（既有各线/既有回测数字不变）；
4. lag 大于日期数 → 直接报错，不许静默跑成 lag=0。
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.strategies.factor_ranking_strategy import (
    _SCORES_ATTR,
    FactorRankingStrategy,
)

_DAYS = 220  # 够 momentum_window=126 + vol_window=60 出分
_IDX = pd.bdate_range("2024-01-01", periods=_DAYS)  # 起点是周一，无节假日
_CODES = ["600000.SH", "000001.SZ", "300750.SZ", "601398.SH", "002594.SZ"]


def _frames(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for code in _CODES:
        close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, _DAYS)))
        out[code] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.full(_DAYS, 1e6)},
            index=_IDX,
        )
    return out


def _score_dates(cfg: dict) -> dict[str, list[pd.Timestamp]]:
    strat = FactorRankingStrategy(cfg)
    frames = _frames()
    asyncio.run(
        strat.prepare_backtest_data(
            frames, _IDX[0].to_pydatetime(), _IDX[-1].to_pydatetime()
        )
    )
    # 策略里存的是 python datetime（master[k].to_pydatetime()）→ 统一成 Timestamp 再比
    return {c: sorted(pd.Timestamp(d) for d in (frames[c].attrs.get(_SCORES_ATTR) or {}))
            for c in _CODES}


_WEEKLY = {"topk": 5, "n_drop": 1, "rebalance_days": "weekly", "rebalance_weekday": 4,
           "weights": {"low_vol": 1.0}}


def test_lag1_dates_are_exactly_one_trading_day_later():
    lag0 = _score_dates(_WEEKLY | {"lag": 0})
    lag1 = _score_dates(_WEEKLY | {"lag": 1})
    assert all(v for v in lag0.values()), "fixture 没出分，测试无效"

    # lag=0 只在周五出分
    for dates in lag0.values():
        assert {d.dayofweek for d in dates} == {4}

    next_bd = {d: _IDX[_IDX.get_loc(d) + 1] for d in _IDX if _IDX.get_loc(d) + 1 < _DAYS}
    for code in _CODES:
        expected = [next_bd[d] for d in lag0[code] if d in next_bd]
        assert lag1[code] == expected, f"{code}: lag1 出分日不是 lag0 的后一个交易日"


def test_lag1_has_no_score_on_the_first_day():
    lag1 = _score_dates(_WEEKLY | {"lag": 1})
    assert _IDX[0] not in lag1[_CODES[0]]
    # 周频 + lag1 → 出分日全是周一（周五分数的下一个交易日），实盘这才做得到
    assert {d.dayofweek for d in lag1[_CODES[0]]} == {0}


def test_default_config_is_identical_to_lag_zero():
    """不传 lag 的既有各线行为不许变（生产口径回归闸门）。"""
    default = _score_dates(_WEEKLY)
    explicit = _score_dates(_WEEKLY | {"lag": 0})
    for code in _CODES:
        assert default[code] == explicit[code]


def test_lag_longer_than_history_raises_instead_of_running_silent():
    strat = FactorRankingStrategy(_WEEKLY | {"lag": 10_000})
    frames = _frames()
    with pytest.raises(ValueError):
        asyncio.run(
            strat.prepare_backtest_data(
                frames, _IDX[0].to_pydatetime(), _IDX[-1].to_pydatetime()
            )
        )
