"""宇宙可筛选性闸门的契约测试（台账 §12.66-D）。

要点：闸门判的是「能不能被风控看见」，不是「有没有价格」。
合法中途上市的票必须留下；价格全有但风控库查不到的票必须剔除。
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.backtest.universe_gate import (
    MIN_SCREEN_RATIO,
    GateReport,
    build_gate_report,
    coverage_ratio,
    load_or_build_report,
)


def _days(start: str, n: int) -> list[int]:
    """造 n 个连续自然日（只要两库比的是同一批日期，是不是交易日不影响闸门逻辑）。"""
    return [int(d.normalize().value) for d in pd.to_datetime(pd.date_range(start, periods=n))]


# ---------- 纯函数核心 ----------

def test_full_coverage_keeps():
    p = _days("2024-01-01", 100)
    assert coverage_ratio(p, p) == 1.0


def test_screen_store_blind_drops():
    """价格 723 天、风控库只有 122 天（北交所真实情形）→ 必须剔除。"""
    price = _days("2024-01-01", 723)      # 全窗都有价格
    screen = _days("2025-01-01", 122)     # 风控库只覆盖窗口末尾一小段
    assert coverage_ratio(price, screen) < 0.2


def test_legit_ipo_not_killed():
    """中途上市：两个库同时从 2025-01 开始 → 覆盖率 1.0，不能被误杀。"""
    price = _days("2025-01-01", 100)
    assert coverage_ratio(price, price) == 1.0


def test_partial_holes_above_threshold_kept():
    price = _days("2024-01-01", 100)
    screen = price[:96]  # 缺 4 天
    assert coverage_ratio(price, screen) == 0.96
    assert coverage_ratio(price, screen) >= MIN_SCREEN_RATIO


def test_no_price_data_is_zero():
    assert coverage_ratio([], _days("2024-01-01", 10)) == 0.0


# ---------- 端到端（造两个库） ----------

def _write_qlib(path: str, code: str, dates: list[str], close=10.0) -> None:
    idx = pd.MultiIndex.from_product([[code], pd.to_datetime(dates)], names=["stock_code", "date"])
    df = pd.DataFrame({"$close": [close] * len(dates), "$open": [close] * len(dates)}, index=idx)
    df.to_parquet(path)


def _write_baostock(path: str, dates: list[str]) -> None:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "trade_status": [1] * len(dates)})
    df.to_parquet(path)


def test_gate_end_to_end(tmp_path):
    full = [f"2024-01-{d:02d}" for d in range(1, 29)]           # 28 天，两库都有
    blind = [f"2024-02-{d:02d}" for d in range(1, 21)]          # 20 天有价格，风控库只覆盖最后 2 天
    ipo = [f"2024-01-{d:02d}" for d in range(15, 29)]           # 中途上市，两库同步

    qlib = tmp_path / "qlib_data/features/day"
    bs = tmp_path / "parquet/baostock_daily"
    qlib.mkdir(parents=True); bs.mkdir(parents=True)

    _write_qlib(str(qlib / "000001_SZ.parquet"), "000001.SZ", full)
    _write_baostock(str(bs / "000001_SZ.parquet"), full)

    _write_qlib(str(qlib / "920819_BJ.parquet"), "920819.BJ", blind)
    _write_baostock(str(bs / "920819_BJ.parquet"), blind[-2:])   # 只有 02-19 / 02-20 两天可见

    _write_qlib(str(qlib / "301999_SZ.parquet"), "301999.SZ", ipo)
    _write_baostock(str(bs / "301999_SZ.parquet"), ipo)

    rep = build_gate_report(["000001.SZ", "920819.BJ", "301999.SZ"], "2024-01-01", "2024-02-28", str(tmp_path))
    assert isinstance(rep, GateReport)
    assert rep.kept == ["000001.SZ", "301999.SZ"], rep.summary()
    assert [d["code"] for d in rep.dropped] == ["920819.BJ"]
    drop = rep.dropped[0]
    assert drop["price_days"] == 20 and drop["screen_days"] == 2
    assert drop["screen_ratio"] == pytest.approx(0.1)
    assert drop["blind_span"] == ["2024-02-01", "2024-02-18"]
    # 窗口内「风控真看得见」的那两天
    assert drop["screened_span"] == ["2024-02-19", "2024-02-20"]


def test_holiday_nan_rows_do_not_count_as_price_days(tmp_path):
    """qlib features/day 里有 A股休市日的行（值为 NaN）。这些天不得算进分母，
    否则正常票会因为「价格 781 天 / 风控 723 天」被闸门误杀（2026-11-06 第一版真实踩过）。"""
    qlib = tmp_path / "qlib_data/features/day"; bs = tmp_path / "parquet/baostock_daily"
    qlib.mkdir(parents=True); bs.mkdir(parents=True)
    real = [f"2024-01-{d:02d}" for d in range(1, 11)]           # 10 个真实交易日
    hol = ["2024-01-11"]                                        # 休市日：价格库有行但值为 NaN
    idx = pd.MultiIndex.from_product([["600000.SH"], pd.to_datetime(real + hol)],
                                     names=["stock_code", "date"])
    closes = [10.0] * len(real) + [float("nan")]
    pd.DataFrame({"$close": closes}, index=idx).to_parquet(str(qlib / "600000_SH.parquet"))
    _write_baostock(str(bs / "600000_SH.parquet"), real)

    rep = build_gate_report(["600000.SH"], "2024-01-01", "2024-01-31", str(tmp_path))
    assert rep.kept == ["600000.SH"], rep.summary()
    assert rep.dropped == []


def test_report_cache_round_trip(tmp_path):
    qlib = tmp_path / "qlib_data/features/day"; bs = tmp_path / "parquet/baostock_daily"
    qlib.mkdir(parents=True); bs.mkdir(parents=True)
    d = [f"2024-01-{i:02d}" for i in range(1, 11)]
    _write_qlib(str(qlib / "000001_SZ.parquet"), "000001.SZ", d)
    _write_baostock(str(bs / "000001_SZ.parquet"), d)
    out = str(tmp_path / "gate.json")
    codes = ["000001.SZ"]
    r1 = load_or_build_report(codes, "2024-01-01", "2024-01-31", str(tmp_path), out)
    r2 = load_or_build_report(codes, "2024-01-01", "2024-01-31", str(tmp_path), out)  # 走缓存
    assert r1.kept == r2.kept == ["000001.SZ"]
    # 窗口变了不能复用旧缓存
    r3 = load_or_build_report(codes, "2024-01-01", "2024-02-28", str(tmp_path), out)
    assert r3.window == ("2024-01-01", "2024-02-28")
