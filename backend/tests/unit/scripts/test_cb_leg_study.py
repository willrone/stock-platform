"""转债腿可行性研究 cb_leg_study.py 的契约测试（台账 §12.79）。

不联网、不读真数据，只锁口径：
- 真实 bar 闸门：被 ffill 平出来的价格不是可成交价，不许用它选券
- 手数闸门：1 手 = 10 张 = 1000 元面值，决定 9 万账户能持几只
- 剩余期限闸门：只用契约到期日
- 主仓口径：年度再平衡（§12.77），不是每天抹平
- 消失券打折 + 分段夏普
"""
import math

import numpy as np
import pandas as pd
import pytest

from scripts.cb_leg_study import (
    CAPITAL,
    MAX_YEARS_LEFT,
    START,
    END,
    abrupt_exit_events,
    annual_rebalanced_core,
    apply_recovery_haircut,
    build_leg,
    half_year_sharpes,
    perf,
    rebalance_days,
    select_leg,
    years_left,
)


def _cal(start="2024-01-01", periods=90):
    return pd.DatetimeIndex(pd.bdate_range(start, periods=periods))


def _panel(codes, cal, prices):
    return pd.DataFrame({c: [prices[i]] * len(cal) for i, c in enumerate(codes)}, index=cal)


def _known(codes, cal, flags=None):
    flags = flags or {c: [True] * len(cal) for c in codes}
    return pd.DataFrame(flags, index=cal)


class TestRealBarGate:
    """最重要的闸门：只有当日真有 bar 的券才能入选，停牌死券不能被平出来的价格骗进来。"""

    def test_ffilled_price_cannot_select_a_suspended_bond(self):
        cal = _cal()
        codes = ["LIVE", "DEAD"]
        px = _panel(codes, cal, [100.0, 95.0])          # DEAD 更便宜，会被低价腿优先选中
        known = _known(codes, cal, {"LIVE": [True] * len(cal),
                                    "DEAD": [True] * 10 + [False] * (len(cal) - 10)})
        expire = {c: pd.Timestamp("2026-06-01") for c in codes}
        d = cal[40]                                     # DEAD 早已没有 bar
        picked = select_leg(px.loc[d], expire, d, "low_price", 1.0, known.loc[d])
        assert picked == ["LIVE"]

    def test_all_leg_also_respects_real_bars(self):
        cal = _cal()
        codes = ["LIVE", "DEAD"]
        px = _panel(codes, cal, [100.0, 99.0])
        known = _known(codes, cal, {"LIVE": [True] * len(cal),
                                    "DEAD": [True] * 5 + [False] * (len(cal) - 5)})
        picked = select_leg(px.loc[cal[30]], {}, cal[30], "all", 1.0, known.loc[cal[30]])
        assert picked == ["LIVE"]

    def test_build_leg_passes_the_gate_through(self):
        cal = _cal()
        codes = ["LIVE", "DEAD"]
        px = _panel(codes, cal, [100.0, 99.0])
        known = _known(codes, cal, {"LIVE": [True] * len(cal),
                                    "DEAD": [True] * 5 + [False] * (len(cal) - 5)})
        expire = {c: pd.Timestamp("2026-06-01") for c in codes}
        wpanel, gross, counts, turnovers = build_leg(px, expire, cal, "low_price", 1.0, known)
        reb = rebalance_days(cal)
        assert wpanel.loc[reb[-1], "DEAD"] == 0.0
        assert wpanel.loc[reb[-1], "LIVE"] == pytest.approx(1.0)


class TestLotGate:
    """9 万账户下，转债腿能持几只由「1 手 = 10 张」硬性决定。"""

    def test_lot_gate_caps_holdings(self):
        cal = _cal()
        codes = [f"C{i}" for i in range(10)]
        px = _panel(codes, cal, [120.0] * 10)
        known = _known(codes, cal)
        expire = {c: pd.Timestamp("2026-06-01") for c in codes}
        # 腿资金 10% = 9000 元，单手 1200 元 → 7 只（9000/7=1286 ≥ 1200；9000/8=1125 < 1200）
        picked = select_leg(px.iloc[0], expire, cal[0], "low_price", 0.10, known.iloc[0])
        assert len(picked) == 7

    def test_cheaper_bonds_allow_more_names(self):
        cal = _cal()
        codes = [f"C{i}" for i in range(10)]
        px = _panel(codes, cal, [100.0] * 10)
        known = _known(codes, cal)
        expire = {c: pd.Timestamp("2026-06-01") for c in codes}
        picked = select_leg(px.iloc[0], expire, cal[0], "low_price", 0.10, known.iloc[0])
        assert len(picked) == 9          # 9000/9 = 1000 恰好等于单手 1000

    def test_picks_lowest_price_first(self):
        cal = _cal()
        codes = ["HI", "LO", "MID"]
        px = pd.DataFrame({"HI": [150.0] * len(cal), "LO": [101.0] * len(cal),
                           "MID": [120.0] * len(cal)}, index=cal)
        known = _known(codes, cal)
        expire = {c: pd.Timestamp("2026-06-01") for c in codes}
        picked = select_leg(px.iloc[0], expire, cal[0], "low_price", 1.0, known.iloc[0])
        assert picked[0] == "LO"

    def test_all_leg_is_not_lot_gated_it_is_an_index_reference(self):
        cal = _cal()
        codes = [f"C{i}" for i in range(10)]
        px = _panel(codes, cal, [120.0] * 10)
        known = _known(codes, cal)
        picked = select_leg(px.iloc[0], {}, cal[0], "all", 0.01, known.iloc[0])
        assert len(picked) == 10         # 宇宙不被偷偷裁剪，可行性另算


class TestMaturityGate:
    def test_far_maturity_excluded(self):
        cal = _cal()
        px = _panel(["NEAR", "FAR"], cal, [100.0, 99.0])
        known = _known(["NEAR", "FAR"], cal)
        expire = {"NEAR": pd.Timestamp("2026-06-01"), "FAR": pd.Timestamp("2030-06-01")}
        picked = select_leg(px.iloc[0], expire, cal[0], "low_price", 1.0, known.iloc[0])
        assert picked == ["NEAR"]
        assert MAX_YEARS_LEFT == 3.0

    def test_missing_expire_is_not_a_free_pass(self):
        cal = _cal()
        px = _panel(["X"], cal, [100.0])
        known = _known(["X"], cal)
        assert select_leg(px.iloc[0], {}, cal[0], "low_price", 1.0, known.iloc[0]) == []
        assert years_left(pd.NaT, cal[0]) == math.inf


class TestCoreLegConstruction:
    def test_core_is_annually_rebalanced_not_daily(self):
        cal = _cal(start="2024-01-01", periods=380)
        etf = pd.DataFrame({"A": np.linspace(100, 200, len(cal)),
                            "B": np.linspace(100, 110, len(cal))}, index=cal)
        r = annual_rebalanced_core(etf, cal)
        daily_blend = 0.5 * etf["A"].pct_change().reindex(cal).fillna(0) + \
            0.5 * etf["B"].pct_change().reindex(cal).fillna(0)
        # 年内权重会漂移（A 涨得快），所以日收益必须和日频等权不一样
        assert not np.allclose(r.values, daily_blend.values)
        # 每年第一个交易日必须回到 50/50 → 该日收益恰为两腿均值
        jan1 = [d for i, d in enumerate(cal) if i == 0 or d.year != cal[i - 1].year]
        d0 = jan1[1]
        i = cal.get_loc(d0)
        want = 0.5 * etf["A"].pct_change().iloc[i] + 0.5 * etf["B"].pct_change().iloc[i]
        assert r.loc[d0] == pytest.approx(want, rel=1e-9)

    def test_core_window_matches_ledger_baseline(self):
        assert (START, END) == ("2023-05-12", "2026-05-08")


class TestTurnoverConvention:
    def test_from_empty_to_full_is_one_hundred_percent_two_way(self):
        cal = _cal()
        codes = ["A", "B"]
        px = _panel(codes, cal, [100.0, 100.0])
        known = _known(codes, cal)
        expire = {c: pd.Timestamp("2026-06-01") for c in codes}
        _, _, counts, turnovers = build_leg(px, expire, cal, "low_price", 1.0, known)
        assert turnovers[0] == pytest.approx(1.0)

    def test_unchanged_book_has_zero_turnover(self):
        cal = _cal()
        codes = ["A", "B"]
        px = _panel(codes, cal, [100.0, 100.0])
        known = _known(codes, cal)
        expire = {c: pd.Timestamp("2026-06-01") for c in codes}
        _, _, counts, turnovers = build_leg(px, expire, cal, "low_price", 1.0, known)
        assert turnovers[1] == pytest.approx(0.0)


class TestRecoveryHaircut:
    def test_haircut_hits_only_exit_day(self):
        cal = _cal()
        gross = pd.Series(0.001, index=cal)
        events = [{"exit_date": str(cal[5].date()), "code": "A", "weight": 0.1}]
        out = apply_recovery_haircut(gross, events, 1.0, 0.5)
        assert out.iloc[5] == pytest.approx(0.001 - 0.1 * 0.5)
        assert out.iloc[4] == pytest.approx(0.001)
        assert out.iloc[6] == pytest.approx(0.001)

    def test_full_recovery_is_no_op(self):
        cal = _cal()
        gross = pd.Series(0.001, index=cal)
        events = [{"exit_date": str(cal[3].date()), "code": "A", "weight": 0.2}]
        out = apply_recovery_haircut(gross, events, 1.0, 1.0)
        assert np.allclose(out.values, gross.values)

    def test_abrupt_exit_detected_when_no_real_bar_in_holding_span(self):
        cal = _cal()
        codes = ["A", "B"]
        px = _panel(codes, cal, [100.0, 100.0])
        known = _known(codes, cal, {"A": [True] * len(cal),
                                    "B": [True] * 15 + [False] * (len(cal) - 15)})
        expire = {c: pd.Timestamp("2026-06-01") for c in codes}
        wpanel, gross, counts, turnovers = build_leg(px, expire, cal, "low_price", 1.0, known)
        reb = rebalance_days(cal)
        events = abrupt_exit_events(known, wpanel, reb, cal)
        # B 在真实 bar 闸门下不会再被选中，所以事件必须为空 —— 这条锁死"死券不进组合"
        assert events == []


class TestMetrics:
    def test_sharpe_formula_matches_platform(self):
        rng = np.random.default_rng(7)
        cal = _cal(periods=252)
        r = pd.Series(rng.normal(0.0005, 0.01, len(cal)), index=cal)
        assert perf(r)["sharpe"] == round(float(r.mean() / r.std(ddof=1) * math.sqrt(252)), 4)

    def test_max_drawdown(self):
        cal = _cal(periods=10)
        r = pd.Series([0.1, -0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], index=cal)
        # 峰值 1.10 → 谷底 0.55：回撤相对峰值，不是相对本金
        assert perf(r)["max_drawdown_pct"] == pytest.approx(-50.0, abs=0.01)

    def test_half_year_split_labels(self):
        cal = _cal(start="2024-01-01", periods=380)
        r = pd.Series(0.0004, index=cal)
        segs = half_year_sharpes(r)
        assert "24H1" in segs and "24H2" in segs


class TestConstants:
    def test_capital_is_the_real_account(self):
        assert CAPITAL == 90_000.0
