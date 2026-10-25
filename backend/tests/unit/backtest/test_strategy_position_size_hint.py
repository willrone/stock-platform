"""策略建议单仓上限 + 周频调仓掩码的回归测试。

背景（2026-10-25）:
  1. 策略内部按 (1-cash_reserve)/topk 自算的 max_position_size **从未传给引擎**，
     引擎只认 BacktestConfig.max_position_size（默认 0.2）。对 topk=30/50/60 的
     等权组合，0.2 × topk 远超可投现金 → 建仓到现金耗尽 → 组合冻结
     （实测 topk=60 只买到 9 只、559 笔交易塌成 9 笔）。
     修复：BaseStrategy.get_position_size_hint() + executor 在调用方未显式
     指定 max_position_size 时采用策略建议值。
  2. 引擎**不消费** BacktestConfig.rebalance_frequency（只有 strategies.py 里
     一个因子策略读了它做日志），周频必须在策略侧用分数掩码实现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.core.base_strategy import BaseStrategy
from app.services.backtest.strategies.factor_ranking_strategy import (
    FactorRankingStrategy,
)
from app.services.backtest.strategies.honest_composite_strategy import (
    HonestCompositeStrategy,
)
from app.services.backtest.strategies.rsi_71421_strategy import Rsi71421Strategy


class _PlainStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__("Plain", config)

    def calculate_indicators(self, data):
        return {"price": data["close"]}

    def generate_signals(self, data, current_date):  # pragma: no cover
        return []


TOPK_STRATEGIES = [
    (HonestCompositeStrategy, 60, 5),
    (Rsi71421Strategy, 30, 3),
    (FactorRankingStrategy, 50, 5),
]


@pytest.mark.parametrize("cls,topk,ndrop", TOPK_STRATEGIES)
def test_topk_strategy_hints_equal_weight_position_cap(cls, topk, ndrop) -> None:
    """TopK 类策略的建议单仓上限必须 = (1-cash_reserve)/topk。"""
    strategy = cls({"topk": topk, "n_drop": ndrop, "cash_reserve_ratio": 0.05})
    hint = strategy.get_position_size_hint()
    assert hint is not None
    assert hint == pytest.approx(0.95 / topk, rel=1e-9)
    # 关键不变量: topk × 单仓上限 ≤ 可投现金比例，否则组合会现金耗尽冻结
    assert topk * hint <= 0.95 + 1e-9


def test_plain_strategy_has_no_position_size_hint() -> None:
    """非 TopK 策略不该给出建议值（保持引擎默认 0.2 行为）。"""
    assert _PlainStrategy({}).get_position_size_hint() is None


def test_explicit_max_position_size_overrides_strategy_hint() -> None:
    """调用方显式传 max_position_size 时，策略建议值不得覆盖它。"""
    strategy = HonestCompositeStrategy(
        {"topk": 60, "n_drop": 5, "max_position_size": 0.05}
    )
    assert strategy.get_position_size_hint() == pytest.approx(0.05)


def _weekly_mask_matrix() -> tuple[pd.DatetimeIndex, np.ndarray]:
    """构造 2 周 × 7 天的 master 与分数矩阵（2024-01-01 是周一）。"""
    master = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=14, freq="D"))
    score = np.arange(14, dtype=np.float64).reshape(1, 14).repeat(2, axis=0)
    return master, score


def test_weekly_mask_keeps_only_rebalance_weekday() -> None:
    """周频掩码后，只有周五的行还留着分数。"""
    master, score = _weekly_mask_matrix()
    weekday = master.dayofweek
    score[:, weekday != 4] = np.nan
    kept = np.where(np.isfinite(score[0]))[0]
    assert len(kept) == 2
    assert all(master[i].dayofweek == 4 for i in kept)


def test_rsi_71421_weekly_reduces_signal_days() -> None:
    """rsi_71421 周频配置下，非调仓日不出分。"""
    strategy = Rsi71421Strategy({"topk": 30, "n_drop": 3, "rebalance_days": "weekly"})
    assert strategy.rebalance_days == "weekly"
    assert strategy.rebalance_weekday == 4
    assert strategy.get_position_size_hint() == pytest.approx(0.95 / 30, rel=1e-9)


def test_daily_is_default_rebalance_mode() -> None:
    """默认必须是 daily（不改变既有行为）。"""
    for cls, _topk, _ndrop in TOPK_STRATEGIES:
        assert cls({"topk": 10, "n_drop": 2}).rebalance_days == "daily"
