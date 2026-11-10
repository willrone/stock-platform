"""
LowVolatilityStrategy unit tests.

Created: 2026-07-25
Updated: 2026-10-25 — 契约变更（见下）

**2026-10-25 契约变更**：策略从「逐股时序信号」改为「全市场横截面排名 +
TopK/Dropout」。

变更原因（真实缺陷，不是风格调整）：
  旧版 `generate_signals` 只有 BUY 分支，触发条件是 `risk_adjusted_return > 0`
  —— 它**根本不看波动率高低**，也不是横截面比较。后果：
    - 3 年全市场只成交 **8 笔**（全是 BUY），7 只票各持有 660 天
    - 现金被 20% 单仓上限迅速吃光 → 组合永久冻结，后续再也买不进
    - 那个 0.579 的夏普不是策略表现，是「7 只股票满仓躺 3 年」
  现改为：低波动 + 低 RAR 波动 → 全市场横截面 z-score → TopK 等权持仓 →
  掉出 TopK 按 n_drop 卖出，与 honest_composite / factor_ranking 同一套
  执行语义。

因此旧的 `test_low_volatility_buy_when_rar_positive`（断言「RAR>0 就发 BUY」）
锁死的是缺陷本身，已按新契约重写为下面的横截面行为测试。
"""

import asyncio
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.models import SignalType
from app.services.backtest.strategies.strategies import LowVolatilityStrategy


def _make_data(
    close: Sequence[float], n_stocks: int = 3
) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    result = {}
    for i in range(1, n_stocks + 1):
        code = f"00000{i}.SZ"
        df = pd.DataFrame(
            {
                "open": [c * 0.99 for c in close],
                "high": [c * 1.02 for c in close],
                "low": [c * 0.98 for c in close],
                "close": close,
                "volume": [1_000_000] * len(close),
            },
            index=dates,
        )
        df.attrs["stock_code"] = code
        result[code] = df
    return result


def test_low_volatility_indicators_shape() -> None:
    """calculate_indicators returns volatility, risk_adjusted_return, price."""
    n = 100
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = LowVolatilityStrategy(
        {"volatility_period": 21, "volatility_window": 63}
    )
    ind = strategy.calculate_indicators(df)

    assert "volatility" in ind
    assert "risk_adjusted_return" in ind
    assert "price" in ind
    assert len(ind["volatility"]) == n


def test_low_volatility_generate_signals_returns_list() -> None:
    """generate_signals always returns a list."""
    n = 100
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = LowVolatilityStrategy(
        {"volatility_period": 21, "volatility_window": 63}
    )
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert isinstance(result, list)


def test_low_volatility_short_data_returns_empty() -> None:
    """Without cross-sectional scores, generate_signals returns []."""
    n = 30
    close = [50.0] * n
    data_dict = _make_data(close, n_stocks=1)
    df = data_dict["000001.SZ"]

    strategy = LowVolatilityStrategy(
        {"volatility_period": 21, "volatility_window": 63}
    )
    strategy.get_cached_indicators(df)
    result = strategy.generate_signals(df, df.index[-1])
    assert result == []


def test_low_volatility_never_emits_sell() -> None:
    """新契约：策略只输出排名分 BUY，卖出由 TopK-Dropout 执行器负责。

    旧版「只买不卖」是缺陷本身（持仓只增不减、现金耗尽后冻结）；
    现在必须确保**不会**退回逐股 SELL 逻辑。
    """
    n = 120
    close = list(range(50, 50 + n))
    data_dict = _make_data(close, n_stocks=2)
    df = data_dict["000001.SZ"]

    strategy = LowVolatilityStrategy({"topk": 2, "n_drop": 1})
    strategy.get_cached_indicators(df)
    for dt in df.index:
        for s in strategy.generate_signals(df, dt):
            assert s.signal_type == SignalType.BUY


def test_low_volatility_uses_topk_dropout_execution_mode() -> None:
    """策略必须声明 topk_dropout 执行模式并给出等权单仓上限。"""
    strategy = LowVolatilityStrategy(
        {"topk": 30, "n_drop": 3, "cash_reserve_ratio": 0.05}
    )
    assert strategy.get_trade_mode() == "topk_dropout"
    cfg = strategy.get_trade_mode_config()
    assert cfg["topk"] == 30
    assert cfg["n_drop"] == 3
    # 等权上限 = (1 - 现金预留) / topk
    hint = strategy.get_position_size_hint()
    assert hint is not None
    assert hint == pytest.approx(0.95 / 30, rel=1e-9)
    # 关键不变量：topk × 单仓上限 ≤ 可投现金比例，否则会现金耗尽冻结
    assert 30 * hint <= 0.95 + 1e-9


def test_low_volatility_scores_are_cross_sectional() -> None:
    """prepare_backtest_data 后应有横截面分数，且低波动的票分数更高。"""
    n = 120
    dates = pd.bdate_range("2024-01-01", periods=n)
    # 票1：稳定小涨（低波动）；票2：剧烈震荡（高波动）
    steady = [50.0 + i * 0.05 for i in range(n)]
    wild = [50.0 + (10.0 if i % 2 else -10.0) for i in range(n)]
    frames = {}
    for code, series in (("000001.SZ", steady), ("000002.SZ", wild)):
        df = pd.DataFrame(
            {
                "open": series,
                "high": [c * 1.01 for c in series],
                "low": [c * 0.99 for c in series],
                "close": series,
                "volume": [1_000_000] * n,
            },
            index=dates,
        )
        df.attrs["stock_code"] = code
        frames[code] = df

    strategy = LowVolatilityStrategy(
        {"topk": 2, "n_drop": 1, "volatility_window": 20, "volatility_period": 10}
    )
    asyncio.run(
        strategy.prepare_backtest_data(frames, dates[0].to_pydatetime(), dates[-1].to_pydatetime())
    )

    last = dates[-1].to_pydatetime()
    scores = {}
    for code, df in frames.items():
        s = df.attrs["_low_volatility_scores"].get(last)
        if s is not None:
            scores[code] = s
    assert scores, "prepare_backtest_data 应写入横截面分数"
    # 低波动票分数应高于高波动票
    if "000001.SZ" in scores and "000002.SZ" in scores:
        assert scores["000001.SZ"] > scores["000002.SZ"]
