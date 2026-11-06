"""etf_virtual_account（¥90,000 ETF 核心虚拟账户）的契约测试。

用合成价格面板，不碰真实账本文件（STATE_PATH/NAV_CSV 指到 tmp），不联网。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import pandas as pd
import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPT = os.path.join(BACKEND, "scripts", "etf_virtual_account.py")


def load_mod(tmp_path, prices: dict[str, dict[str, float]]):
    """加载被测模块，并把它的价源与落盘路径换成测试替身。"""
    spec = importlib.util.spec_from_file_location("eva_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["eva_test"] = m
    spec.loader.exec_module(m)
    m.STATE_PATH = os.path.join(str(tmp_path), "etf_core_90k.json")
    m.NAV_CSV = os.path.join(str(tmp_path), "etf_core_90k_nav.csv")
    m.PAPER_DIR = str(tmp_path)

    def fake_load_close(code: str, upto=None):
        raw = prices.get(code)
        if raw is None:
            raise SystemExit(f"missing {code}")
        s = pd.Series({k: v for k, v in raw.items() if upto is None or k <= upto}, dtype=float)
        if s.empty:
            raise SystemExit(f"no price for {code} before {upto}")
        return s.sort_index()

    m.load_close = fake_load_close
    return m


P0 = {"515450": {"2026-11-06": 1.484, "2026-11-13": 1.520, "2026-12-31": 1.500, "2027-01-05": 1.470, "2027-01-12": 1.510},
      "511220": {"2026-11-06": 10.450, "2026-11-13": 10.460, "2026-12-31": 10.440, "2027-01-05": 10.480, "2027-01-12": 10.470}}


def test_target_lots_are_integer_and_within_budget():
    m = load_mod(tempfile.mkdtemp(), {})   # 纯函数，不需要价格替身
    lots = m.target_lots(90_000.0, {"515450": 1.484, "511220": 10.45})
    assert lots["515450"] == 303 and lots["511220"] == 43
    spent = lots["515450"] * 100 * 1.484 + lots["511220"] * 100 * 10.45
    assert spent <= 90_000.0
    assert all(isinstance(v, int) for v in lots.values())


def test_commission_rate_and_floor():
    m = load_mod(tempfile.mkdtemp(), {})
    assert m.commission(1_000.0) == 5.0                 # 万3 = 0.3 元 → 抬到最低 5 元
    assert abs(m.commission(44_965.2) - 13.4896) < 1e-3  # 万3
    assert abs(m.commission(100_000.0) - 30.0) < 1e-9


def test_open_then_marks_nav_tracks_price_and_skips_duplicate(tmp_path):
    m = load_mod(tmp_path, P0)
    st = m.blank_state()
    st = m.cmd_open(st, "2026-11-06")
    assert st["status"] == "opened" and st["opened_date"] == "2026-11-06"
    assert st["holdings"]["515450"]["shares"] == 30_300 and st["holdings"]["511220"]["shares"] == 4_300
    nav0 = st["nav_daily"][-1]["nav"]
    assert nav0 == pytest.approx(90_000.0 - 13.49 - 13.48, abs=0.02)   # 净值 = 本金 − 佣金
    st = m.cmd_mark(st, "2026-11-13")
    nav1 = st["nav_daily"][-1]["nav"]
    expect = nav0 + 30_300 * (1.520 - 1.484) + 4_300 * (10.460 - 10.450)
    assert nav1 == pytest.approx(expect, abs=0.02)
    assert st["last_rebalance_date"] == "2026-11-06"                   # 同年内不得再平衡
    n_before = len(st["nav_daily"])
    st = m.cmd_mark(st, "2026-11-13")                                  # 同一天重复盯市必须被跳过
    assert len(st["nav_daily"]) == n_before


def test_rebalance_only_on_first_trading_day_of_new_year(tmp_path):
    m = load_mod(tmp_path, P0)
    st = m.blank_state()
    st = m.cmd_open(st, "2026-11-06")
    st = m.cmd_mark(st, "2026-12-31")
    trades_before = len(st["trades"])
    st = m.cmd_mark(st, "2027-01-05")                                  # 跨年首个交易日 → 再平衡
    assert st["last_rebalance_date"] == "2027-01-05"
    assert len(st["trades"]) > trades_before
    rebal = [t for t in st["trades"] if "年度再平衡" in t.get("reason", "")]
    assert rebal, "没有记录再平衡交易"
    w = st["nav_daily"][-1]["weights"]
    assert abs(w["515450"] - 0.5) < 0.02 and abs(w["511220"] - 0.5) < 0.02  # 换回目标权重
    trades_after = len(st["trades"])
    st = m.cmd_mark(st, "2027-01-12")                                  # 同年内不再动
    assert len(st["trades"]) == trades_after and st["last_rebalance_date"] == "2027-01-05"


def test_mark_refuses_earlier_date_than_last(tmp_path):
    m = load_mod(tmp_path, P0)
    st = m.cmd_open(m.blank_state(), "2026-11-13")
    with pytest.raises(SystemExit):
        m.cmd_mark(st, "2026-11-06")                                    # 不许回填历史造成乱序


def test_cash_shortage_is_logged_not_fabricated(tmp_path):
    m = load_mod(tmp_path, P0)
    st = m.blank_state()
    st["capital"] = 1_200.0
    st["cash"] = 1_200.0
    st["status"], st["opened_date"] = "opened", "2026-11-06"
    m.apply_targets(st, "2026-11-06", {"515450": 1.484, "511220": 10.45}, "建仓")
    sides = [t["side"] for t in st["trades"]]
    assert "unfilled" in sides or "511220" not in st["holdings"], "现金不足时必须如实记录未成交，不能凭空成交"
    held = st["holdings"]
    assert held["515450"]["shares"] % 100 == 0
    assert st["cash"] >= 0.0
