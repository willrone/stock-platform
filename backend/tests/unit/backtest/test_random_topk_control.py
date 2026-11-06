"""random_topk_control 的契约测试（对照实验的地基，必须证明自己真的是噪声）。

要证明三件事：
1. 分数只由 (seed, 调仓日) 决定 —— 同一 seed 跑两遍分数完全一致，与行情字段无关；
2. 分数与「未来/过去收益」没有任何统计关系（把某只票的收益拉爆，它的分数不该变）；
3. 只有调仓日出分（周频掩码生效），非调仓日全部 NaN。
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.strategies.control_strategies import RandomTopKControlStrategy


def _frames(n_days: int = 120, seed: int = 7) -> dict[str, pd.DataFrame]:
    # 2024-01-01 起的前 120 个工作日
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for k, code in enumerate(["600000.SH", "000001.SZ", "300750.SZ", "601398.SH", "002594.SZ"]):
        close = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
        out[code] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
             "volume": np.full(n_days, 1e6)},
            index=idx,
        )
    return out


def _scores(strategy_cfg: dict) -> dict[str, dict]:
    strat = RandomTopKControlStrategy(strategy_cfg)
    frames = _frames()
    asyncio.run(
        strat.prepare_backtest_data(frames, frames["600000.SH"].index[0].to_pydatetime(),
                                    frames["600000.SH"].index[-1].to_pydatetime())
    )
    return {code: dict(df.attrs["_factor_ranking_scores"]) for code, df in frames.items()}


CFG = {"topk": 3, "n_drop": 1, "rebalance_days": "weekly", "rebalance_weekday": 4, "seed": 20261106}


def test_scores_are_deterministic_given_seed():
    a = _scores(CFG)
    b = _scores(CFG)
    assert a.keys() == b.keys()
    for code in a:
        assert a[code] == b[code], f"{code} 分数不稳定 → 随机源没有固定种子"


def test_scores_ignore_market_data():
    """把一只票的价格改成 10 倍暴涨，它的分数必须一模一样：分数不读行情。"""
    base = _scores(CFG)
    strat = RandomTopKControlStrategy(CFG)
    frames = _frames()
    frames["600000.SH"]["close"] = frames["600000.SH"]["close"] ** 4  # 极端收益
    asyncio.run(
        strat.prepare_backtest_data(frames, frames["600000.SH"].index[0].to_pydatetime(),
                                    frames["600000.SH"].index[-1].to_pydatetime())
    )
    after = {c: dict(df.attrs["_factor_ranking_scores"]) for c, df in frames.items()}
    for code in base:
        assert base[code] == after[code], f"{code} 分数被行情影响 → 对照不再是噪声"


def test_only_rebalance_days_get_scores():
    scores = _scores(CFG)["600000.SH"]
    dates = list(scores)
    assert dates, "没有任何调仓日出分"
    assert all(d.weekday() == 4 for d in dates), f"非周五出分了: {[str(d.date()) for d in dates if d.weekday() != 4]}"


def test_seed_changes_the_selection():
    other = _scores({**CFG, "seed": 999})
    same = sum(
        1
        for d, v in other["600000.SH"].items()
        if abs(v - _scores(CFG)["600000.SH"][d]) < 1e-12
    )
    # 4 个调仓日里允许偶然撞车，但全部撞车说明种子没起作用
    assert same < len(other["600000.SH"]), "换种子后分数完全相同 → 随机源退化"
