"""转债数据入口脚本的契约测试（2026-11-10 新增）。

为什么必须有：这条线的全部风险都在「宇宙判定」和「终点语义」两个函数上——
1) 登记报告混进分离交易债/老三板退债，闸门分错就会把纯债当转债做截面；
2) 已退市转债的最后一根 bar 早于摘牌日 1~47 天，若按 DELIST_DATE 填充
   就会凭空造出没有成交的 bar。这两件事都必须能变红，不能靠人眼看。
测试不联网、不依赖 data/cb 产物，全部用注入的日期与桩数据。
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load(name: str, filename: str):
    path = os.path.join(ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


uni = _load("cb_universe_probe", "fetch_convertible_bond_universe.py")
daily = _load("cb_daily_probe", "fetch_convertible_bond_daily.py")


# ── 券种类闸门 ────────────────────────────────────────────────────────
def test_real_convertible_bonds_pass_the_gate():
    for name in ["中鼎转债", "东财转3", "中鼎转03", "巨轮转2", "振26转债"]:
        assert uni.classify_instrument(name, "CNSESZ") == "可转债", name


def test_non_convertible_rows_are_excluded():
    assert uni.classify_instrument("08宝钢债", "CNSESH") == "分离交易债/公司债"
    assert uni.classify_instrument("中兴债1", "CNSESZ") == "分离交易债/公司债"
    assert uni.classify_instrument("普利退债", "STAS00") == "退债(老三板)"
    assert uni.classify_instrument("普利转退", "CNSESZ") == "转债退市过渡码"


def test_third_board_rows_classify_by_market_not_name():
    # 名字里没有「转」也不影响：老三板一律归退债，避免被当成普通债
    assert uni.classify_instrument("搜特退债", "STAS00") == "退债(老三板)"


# ── 交易所映射：显式市场优先于代码段猜测 ─────────────────────────────
def test_exchange_of_prefers_declared_market():
    assert uni.exchange_of("CNSESH", "113001") == "sh"
    assert uni.exchange_of("CNSESZ", "127025") == "sz"
    # 115003 中兴债1：11 开头却挂深市，按代码段猜会取错市场
    assert uni.exchange_of("CNSESZ", "115003") == "sz"
    assert uni.exchange_of("STAS00", "404002") == ""
    assert uni.exchange_of("STAS00", "404002") != "sh"


def test_exchange_of_falls_back_to_code_segment():
    assert uni.exchange_of("", "110076") == "sh"
    assert uni.exchange_of("", "128136") == "sz"
    assert uni.exchange_of("", "404005") == ""


# ── 状态分层：边界必须确定 ───────────────────────────────────────────
def test_status_boundaries():
    today = "2026-11-10"
    pre = {"LISTING_DATE": "2026-11-13 00:00:00", "DELIST_DATE": None}
    alive = {"LISTING_DATE": "2026-11-10 00:00:00", "DELIST_DATE": None}
    dead = {"LISTING_DATE": "2019-01-01 00:00:00", "DELIST_DATE": "2026-11-10 00:00:00"}
    assert uni.classify(pre, today) == uni.STATUS_PRELIST
    assert uni.classify(alive, today) == uni.STATUS_ALIVE
    assert uni.classify(dead, today) == uni.STATUS_DEAD


def test_as_date_normalises_timestamp_strings():
    assert uni._as_date("2026-11-13 00:00:00") == "2026-11-13"
    assert uni._as_date(None) is None
    assert uni._as_date("") is None


# ── 取价符号与闸门语义 ───────────────────────────────────────────────
def test_sina_symbol_uses_exchange_when_given():
    assert daily.sina_symbol("115003", "sz") == "sz115003"
    assert daily.sina_symbol("113001", "sh") == "sh113001"
    assert daily.sina_symbol("127025") == "sz127025"


def test_bars_to_rows_coerces_placeholder_values():
    bars = [{"day": "2026-11-09", "open": "100.0", "high": "-", "low": None,
             "close": "101.5", "volume": "12345"}]
    row = daily.bars_to_rows("113001", bars)[0]
    assert row["open"] == 100.0
    assert row["high"] is None          # "-" 这类占位值不能变成数字
    assert row["low"] is None
    assert row["close"] == 101.5
    assert row["volume"] == 12345.0


def test_dead_bond_last_bar_may_precede_delist_date():
    """已退市：终点日与摘牌日之间允许有间隔，闸门不得把它当缺失数据。"""
    univ = {"code": "110076", "name": "华海转债", "status": uni.STATUS_DEAD,
            "listing_date": "2019-06-01", "delist_date": "2026-11-02",
            "instrument_class": "可转债", "exchange": "sh"}
    bars = [{"day": "2019-06-01", "close": "100"}, {"day": "2026-10-19", "close": "130"}]
    audit = daily.coverage_row(univ, bars, "2026-11-10")
    assert audit["is_covered"] is True
    assert audit["gap_start_days"] == 0
    assert audit["gap_end_days"] == 14   # 停止交易早于摘牌，属正常语义


def test_start_misaligned_bond_is_flagged_not_covered():
    univ = {"code": "123456", "name": "测试转债", "status": uni.STATUS_ALIVE,
            "listing_date": "2021-01-04", "delist_date": None,
            "instrument_class": "可转债", "exchange": "sz"}
    bars = [{"day": "2021-06-01", "close": "100"}]
    audit = daily.coverage_row(univ, bars, "2026-11-10")
    assert audit["is_covered"] is False
    assert audit["gap_start_days"] > 100


def test_missing_bars_carry_a_skip_reason():
    univ = {"code": "404002", "name": "搜特退债", "status": uni.STATUS_ALIVE,
            "listing_date": "2023-10-20", "delist_date": None,
            "instrument_class": "退债(老三板)", "exchange": ""}
    rows, audit = daily.ingest_one(univ, "2026-11-10")
    assert rows == []
    assert audit["bars"] == 0
    assert audit["skip_reason"] == "老三板无日线源"
    assert audit["is_covered"] is False


def test_coverage_summary_splits_by_instrument_class():
    audits = [
        {"instrument_class": "可转债", "is_covered": True, "status": uni.STATUS_DEAD,
         "gap_end_days": 10},
        {"instrument_class": "可转债", "is_covered": False, "status": uni.STATUS_DEAD,
         "gap_end_days": 5},
        {"instrument_class": "退债(老三板)", "is_covered": False, "status": uni.STATUS_ALIVE,
         "gap_end_days": 0},
    ]
    out = daily._coverage_by_class(audits)
    assert out["可转债"] == {"requested": 2, "covered": 1, "coverage_rate": 0.5}
    assert out["退债(老三板)"]["requested"] == 1
