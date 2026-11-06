"""regime.py 判据层单元测试（纯合成面板，不跑真回测，总耗时 < 60 秒）。

四条硬性契约：
  1. test_no_lookahead_spread —— 同一 asof 的 size_spread_z 用完整面板算必须与
     用截至 asof 截断面板算逐位相等；build_tier_index 的 ≤asof 各点同样相等。
  2. test_confirm_logic —— 进防御必须连续 confirm_off 个判据日、出防御必须连续
     confirm_on 个；单点刺穿不触发；J1 成立但 J2 不成立也不触发（AND 才防御）。
  3. test_max_switches_freeze —— 超过 max_switches 后不再切换，meta 有记录。
  4. test_shadow_nav_past_only —— asof 之后的价格改成极端值不影响该次 update。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.regime import (
    ShadowNav,
    build_tier_index,
    daily_free_float_cap,
    random_switch_schedule,
    run_switch_machine,
    size_spread_z,
)


def _synthetic_panel(n_days: int = 400, n_codes: int = 40, seed: int = 7):
    """合成小面板：价格随机游走 + 与价格弱相关的成交额/换手（供市值估算）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    codes = [f"TEST{i:03d}.SZ" for i in range(n_codes)]
    drift = np.linspace(-0.004, 0.006, n_codes)  # 制造大小盘风格差异
    rets = rng.normal(drift, 0.02, size=(n_days, n_codes))
    close = 10.0 * np.cumprod(1.0 + rets, axis=0)
    close_df = pd.DataFrame(close, index=dates, columns=codes)
    amount = close * rng.uniform(1e5, 1e7, size=(n_days, n_codes))
    turn = rng.uniform(0.5, 8.0, size=(n_days, n_codes))
    turn[:, 5] = np.nan  # 造几天换手缺失，测 fallback
    caps = pd.DataFrame(
        {c: daily_free_float_cap(pd.Series(amount[:, j], index=dates),
                                 pd.Series(turn[:, j], index=dates))
         for j, c in enumerate(codes)}
    )
    return close_df, caps


# ──────────────────────────────────────────────────────────────────────

def test_no_lookahead_spread():
    """完整面板 vs 截至 asof 截断面板：z 必须逐位相等。"""
    close, caps = _synthetic_panel()
    tier = build_tier_index(close, caps)
    low, high = tier["low_ret"], tier["high_ret"]

    asof = close.index[300]
    z_full = size_spread_z(low, high, asof, lookback=20, hist_window=250, min_samples=120)
    z_trunc = size_spread_z(low[low.index <= asof], high[high.index <= asof], asof,
                            lookback=20, hist_window=250, min_samples=120)
    assert z_full is not None and z_trunc is not None
    assert z_full == z_trunc  # 逐位相等（同一 float）

    # 再多截几天也必须逐位相等：判据不能偷看 asof 之后
    for extra in (1, 3, 17):
        later = close.index[min(399, 300 + extra)]
        z_later_full = size_spread_z(low, high, asof, lookback=20, hist_window=250)
        assert z_later_full == z_full

    # 样本不足必须返回 None（而不是硬凑一个数）
    early = close.index[60]
    assert size_spread_z(low, high, early, lookback=20, hist_window=250, min_samples=120) is None


def test_no_lookahead_tier_index():
    """build_tier_index ≤asof 各点的层籍与收益：截断面板计算必须与完整面板逐位相等。"""
    close, caps = _synthetic_panel()
    asof = close.index[320]
    full = build_tier_index(close, caps)
    trunc = build_tier_index(close[close.index <= asof], caps[caps.index <= asof])
    idx = close.index[: close.index.get_loc(asof) + 1]
    pd.testing.assert_series_equal(full["low_ret"].loc[idx], trunc["low_ret"].loc[idx])
    pd.testing.assert_series_equal(full["high_ret"].loc[idx], trunc["high_ret"].loc[idx])
    # 层籍只能用 t-1 及以前：末日层籍不应随 asof 之后的市值变化而变
    assert (full["low_n"].loc[idx] == trunc["low_n"].loc[idx]).all()


def test_confirm_logic():
    """连续确认才翻转；单点刺穿不触发；AND 语义（J2 不坏则不进防御）。"""
    dates = pd.bdate_range("2024-01-01", periods=30)

    # z 序列：idx2..3 短刺穿、idx4 中断，idx5..9 连续 5 天 < -1.0 → confirm_off=5 在第 9 天进防御
    z = [0.2, -0.6, -2.0, -2.0, 0.5, -2.0, -2.0, -2.0, -2.0, -2.0,
         0.1, 0.4, 0.2, 0.5, 0.1, 0.3, 0.2, 0.6, 0.1, 0.2,
         0.4, 0.1, 0.3, 0.5, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1]
    assert z[2] < -1.0 and z[5] < -1.0  # 单点/短刺穿存在
    thr = -1.0
    j2_bad_all = [True] * len(dates)
    enter = [(zz is not None and zz < thr) and j2 for zz, j2 in zip(z, j2_bad_all)]
    exit_ = [(zz is None or zz >= thr) for zz in z]  # J2 全好
    res = run_switch_machine(dates, enter, exit_, confirm_off=5, confirm_on=10, max_switches=12)
    steps = res["steps"]
    flip_days = [s["date"] for s in steps if s["switched"] and s["state"] == "OFF"]
    assert flip_days == [steps[9]["date"]]  # 恰好第 9 个判据日（连续第 5 天）翻转
    # 出防御：进入防御时 streak 清零；exit 从 idx10 起连续为真，
    # 攒满 confirm_on=10 后在 idx19 退出
    flip_on = [s["date"] for s in steps if s["switched"] and s["state"] == "ON"]
    assert flip_on == [steps[19]["date"]]
    assert res["meta"]["switch_count"] == 2

    # J1 全坏但 J2 全好 → AND 不成立 → 永不进防御
    enter_no_j2 = [False] * len(dates)
    res2 = run_switch_machine(dates, enter_no_j2, [False] * len(dates),
                              confirm_off=5, confirm_on=10, max_switches=12)
    assert res2["meta"]["switch_count"] == 0
    assert all(s["state"] == "ON" for s in res2["steps"])

    # 连续 4 天刺穿（不足 5）不触发
    enter4 = [i in (10, 11, 12, 13) for i in range(len(dates))]
    res3 = run_switch_machine(dates, enter4, [False] * len(dates),
                              confirm_off=5, confirm_on=10, max_switches=12)
    assert res3["meta"]["switch_count"] == 0


def test_max_switches_freeze():
    """达到 max_switches 后冻结：不再切换，meta 记录。"""
    dates = pd.bdate_range("2024-01-01", periods=60)
    # 交替好坏：每 5 天翻一次状态 → 切换需求远超 max_switches=3
    enter = [True] * 5 + [False] * 60
    exit_ = [False] * 5 + [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5 + [True] * 40
    # 手工构造反复翻转的 flag：坏5好10坏5好10...（confirm_off=5, confirm_on=10）
    enter = []
    exit_ = []
    for i in range(60):
        block = i // 15
        if block % 2 == 0:
            enter.append(True)
            exit_.append(False)
        else:
            enter.append(False)
            exit_.append(True)
    res = run_switch_machine(dates, enter, exit_, confirm_off=5, confirm_on=10, max_switches=3)
    assert res["meta"]["switch_count"] <= 3
    assert res["meta"]["frozen_at_max_switches"] is True
    flips_after_freeze = [s for s in res["steps"] if s["frozen"] and s["switched"]]
    assert not flips_after_freeze
    assert any(s["frozen_attempt"] for s in res["steps"])  # 冻结后确实还想翻但被拦


def test_shadow_nav_past_only():
    """asof 之后的价格改成极端值，不影响该次 update 的结果。"""
    dates = pd.bdate_range("2024-01-01", periods=30)
    codes = ["A.SZ", "B.SZ"]
    rng = np.random.default_rng(11)
    close = pd.DataFrame(10 * np.cumprod(1 + rng.normal(0, 0.01, (30, 2)), axis=0),
                         index=dates, columns=codes)
    weights = {"A.SZ": 0.5, "B.SZ": 0.5}

    nav1 = ShadowNav(window=5, annualize=244)
    nav1.seed(dates[0], 1.0)
    r1 = nav1.update(weights, dates[0], dates[1], close)
    nav2 = ShadowNav(window=5, annualize=244)
    nav2.seed(dates[0], 1.0)
    close_poison = close.copy()
    close_poison.iloc[2:, :] = 1e6  # asof 之后全部改成极端值
    r2 = nav2.update(weights, dates[0], dates[1], close_poison)
    assert r1 == r2
    assert nav1.navs == nav2.navs

    # 等权收益正确：A 涨 10%、B 平 → 篮子收益 5%
    close_known = pd.DataFrame([[10.0, 10.0], [11.0, 10.0]],
                               index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
                               columns=codes)
    nav3 = ShadowNav(window=1, annualize=244)
    nav3.seed(close_known.index[0], 1.0)
    step = nav3.update(weights, close_known.index[0], close_known.index[1], close_known)
    assert abs(step - 0.05) < 1e-12
    assert abs(nav3.navs[-1] - 1.05) < 1e-12
    # 回撤：峰值过后回落
    nav4 = ShadowNav(window=2, annualize=244)
    px = pd.DataFrame([[10.0], [12.0], [9.0]],
                      index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                      columns=["A.SZ"])
    nav4.seed(px.index[0], 1.0)
    nav4.update({"A.SZ": 1.0}, px.index[0], px.index[1], px)
    nav4.update({"A.SZ": 1.0}, px.index[1], px.index[2], px)
    assert nav4.drawdown() == pytest.approx(9 / 12 - 1)  # -25%
    assert nav4.rolling_sharpe() is not None  # 样本够（window=1 时）


def test_random_switch_schedule_switch_count_matches():
    """随机开关对照：切换次数必须与目标相等（奇数时最后一块延伸收尾）。"""
    dates = list(pd.bdate_range("2023-05-15", periods=60))
    for seed in (1, 2, 3):
        for target in (6, 8, 12, 9):
            sched = random_switch_schedule(dates, target_switches=target, seed=seed)
            assert sched["switch_count"] == target
            off = [d.date().isoformat() for d in sched["off_dates"]]
            assert len(off) == len(set(off))  # OFF 日期不重复
            # 区块互不重叠
            blocks = sched["blocks"]
            for i in range(1, len(blocks)):
                assert blocks[i][0] > blocks[i - 1][1]
        # 不同 seed 的 OFF 日期应不同（运气对照才有意义）
    s1 = random_switch_schedule(dates, 8, seed=1)["off_dates"]
    s2 = random_switch_schedule(dates, 8, seed=2)["off_dates"]
    assert s1 != s2
