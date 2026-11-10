"""etf_adj_caliber_audit 的纯函数契约测试（不打网络）。

锁三件事：
1. 复权口径判定必须由"与哪个参考口径更近"决定，不能靠脚本注释（注释曾经是错的）；
2. core_stats 的年度再平衡语义：年内不再衡 = 0.5×腿0涨幅 + 0.5×腿1涨幅；
3. 窗口对齐——长度不同的两个序列直接比夏普，会造出假漂移（上一版就是这么错的）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.etf_adj_caliber_audit import (
    align_window,
    core_stats,
    judge_caliber,
)


def _frame(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"515450": closes, "511220": [c * 7.0 for c in closes]},
        index=pd.to_datetime(dates),
    )


def _days(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start, periods=n)]


def test_judge_caliber_chooses_the_closer_reference():
    assert judge_caliber(abs_max_raw=26.6, abs_max_qfq=2.7) == "前复权"
    assert judge_caliber(abs_max_raw=0.0, abs_max_qfq=12.0) == "不复权"
    assert judge_caliber(abs_max_raw=0.0, abs_max_qfq=0.0) == "两口径同价（窗内无复权差异）"
    with pytest.raises(ValueError):
        judge_caliber(abs_max_raw=3.0, abs_max_qfq=3.0)


def test_core_stats_is_annual_rebalanced_not_daily():
    """一个自然年内不再衡 → 组合涨幅必须等于两条腿涨幅的算术平均。"""
    n = 150
    dates = _days("2023-01-02", n)
    frame = pd.DataFrame(
        {"515450": [1.0 * 1.01 ** i for i in range(n)],        # 每天 +1%
         "511220": [7.0] * n},                                  # 全程不动
        index=pd.to_datetime(dates),
    )
    st = core_stats(frame)
    expect = 0.5 * (1.01 ** (n - 1)) + 0.5 * 1.0 - 1
    # core_stats 对外报告统一保留 2 位小数 → 断言必须按报告精度，不能按机器精度
    assert st["total_pct"] == pytest.approx(expect * 100, abs=0.01)
    assert st["max_dd_pct"] == pytest.approx(0.0, abs=0.01)
    assert st["n_days"] == n


def test_core_stats_drawdown_is_peak_to_trough():
    n = 160
    closes = [1.0] * 50 + [1.0 - 0.5 * i / 50 for i in range(1, 51)] + [0.5 + 0.5 * i / 60 for i in range(1, 61)]
    st = core_stats(_frame(_days("2023-01-02", n), closes[:n]))
    assert st["max_dd_pct"] == pytest.approx(-50.0, abs=1.0)


def test_align_window_refuses_mismatched_series():
    a = _frame(_days("2023-05-12", 3), [1.0, 1.1, 1.2])
    b = _frame(_days("2023-05-12", 6), [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    ca, cb = align_window(a, b)
    assert list(ca.index) == list(cb.index)
    assert len(ca) == 3, "只保留共同交易日，不许一边 3 天一边 6 天算年化"
    with pytest.raises(ValueError):
        align_window(a, _frame(_days("2019-01-02", 3), [1.0, 1.1, 1.2]))


def test_core_stats_rejects_too_short_sample():
    with pytest.raises(AssertionError):
        core_stats(_frame(_days("2023-05-12", 20), [1.0] * 20))
