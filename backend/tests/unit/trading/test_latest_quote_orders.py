"""最新行情 + 可成交性 + 可下单清单补字段（建设项 C）——含「必须变红」的场景。

关键红线（本文件第 2 条测试守的就是它）：
行情层 `pct_chg` 是**百分数**，`check_tradability` 收**小数**。
换算写错时两种错法都静默：传 9.95 会被判「涨跌幅异常、本日回避」，
传 0.0995 则涨停永远拦不住 —— 后者最坏，因为它报「可成交」。
"""

import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "app"))  # noqa: E402

from app.services.trading import latest_quote as lq  # noqa: E402
from app.services.trading.latest_quote import (  # noqa: E402
    enrich_order_row,
    enrich_orders,
    load_latest_bar,
    tradability_for_order,
)


TODAY = date.today().isoformat()


def _daily(dir_, code, rows):
    """写一个和 data/parquet/baostock_daily 同构的文件。"""
    os.makedirs(dir_, exist_ok=True)
    df = pd.DataFrame(rows, columns=[
        "date", "open", "high", "low", "close",
        "volume", "amount", "pct_chg", "is_st", "trade_status",
    ])
    df.to_parquet(os.path.join(dir_, f"{code.replace('.', '_')}.parquet"), index=False)
    return dir_


def _etf(dir_, code, rows):
    os.makedirs(dir_, exist_ok=True)
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df.to_parquet(os.path.join(dir_, f"{code}.parquet"), index=False)
    return dir_


# 正常交易：主板 +1.00%，有量有额
OK_A = ["2026-11-03", 10.0, 10.2, 9.9, 10.1, 1_000_000.0, 1.0e8, 1.00, 0.0, 1.0]


# ── 1. 代码形状决定读哪一层（ETF 腿不在 A 股层里）────────────────
def test_etf_leg_reads_etf_layer_a_share_reads_daily_layer(tmp_path):
    daily = _daily(str(tmp_path / "daily"), "600000.SH", [OK_A])
    etf = _etf(str(tmp_path / "etf"), "518880", [["2026-11-03", 8.0, 8.1, 7.9, 8.05, 5.0e6]])

    bar_a = load_latest_bar("600000.SH", daily, etf_dir=etf, use_cache=False)
    bar_etf = load_latest_bar("518880", daily, etf_dir=etf, use_cache=False)

    assert bar_a is not None and bar_a["bar_source"] == "baostock_daily"
    assert bar_etf is not None and bar_etf["bar_source"] == "etf"
    assert bar_etf["close"] == pytest.approx(8.05)
    # ETF 层没有 amount / pct_chg → 必须是 None，不能是 0（0 会被当成「无成交」）
    assert bar_etf["amount"] is None
    assert bar_etf["pct_chg"] is None


# ── 2. 涨跌幅单位换算（红线）────────────────────────────────────
def test_percent_pct_triggers_limit_up_block(tmp_path):
    """+9.95% 写成分数（正确口径）→ 买不进（涨停）。"""
    daily = _daily(str(tmp_path / "d"), "600001.SH",
                   [["2026-11-03", 10.0, 10.99, 10.0, 10.99, 1e6, 1e8, 9.95, 0.0, 1.0]])
    bar = load_latest_bar("600001.SH", daily, etf_dir=str(tmp_path / "none"), use_cache=False)
    t = tradability_for_order("600001.SH", "buy", bar)
    assert t["tradable"] is False
    assert "涨停" in t["tradable_reason"]


def test_double_division_silently_misses_limit_up(tmp_path):
    """换算只能做一次。多做一次（0.0995 → 0.000995）涨停就永远拦不住。

    这是本模块最危险的静默错法：它不会报错，只会把一根封死的涨停板
    标成「可成交」。所以把它钉成测试，而不是写在注释里。
    """
    daily = _daily(str(tmp_path / "d"), "600002.SH",
                   [["2026-11-03", 10.0, 10.99, 10.0, 10.99, 1e6, 1e8, 9.95, 0.0, 1.0]])
    bar = load_latest_bar("600002.SH", daily, etf_dir=str(tmp_path / "none"), use_cache=False)

    from app.services.trading.execution_rules import check_tradability

    # 错法 A：除两次 → 0.0995/100 = 0.000995，涨停被放过（静默）
    too_small = check_tradability("600002.SH", close=10.99, pct_chg=9.95 / 100.0 / 100.0,
                                 volume=1e6, amount=1e8, pct_reliable=True)
    assert too_small.can_buy is True  # 单位错就是这么静默
    # 错法 B：不换算直接喂百分数 → 9.95 远超阈值，同样被判「涨停买不进」，
    # 但阈值失效：任何 +10% 以上的正常波动都会被一律拦掉（误杀面极大）
    too_big = check_tradability("600002.SH", close=10.99, pct_chg=9.95,
                               volume=1e6, amount=1e8, pct_reliable=True)
    assert too_big.can_buy is False
    flat = check_tradability("600002.SH", close=10.99, pct_chg=1.0,   # 平盘被写成 1.0（百分数）
                             volume=1e6, amount=1e8, pct_reliable=True)
    assert flat.can_buy is False  # 平盘也被拦 → 这就是不换算的误杀面
    # 本模块的入口：做一次换算 → 判成涨停（买不进），且没把整只票打成回避
    t = tradability_for_order("600002.SH", "buy", bar)
    assert t["tradable"] is False and "涨停" in t["tradable_reason"]
    assert "异常" not in t["tradable_reason"]
    tsell = tradability_for_order("600002.SH", "sell", bar)
    assert tsell["tradable"] is True  # 涨停不影响卖


# ── 3. 停牌 / 无成交 必须标出来 ────────────────────────────────
def test_suspended_stock_is_not_tradable(tmp_path):
    daily = _daily(str(tmp_path / "d"), "600003.SH",
                   [["2026-11-03", 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    bar = load_latest_bar("600003.SH", daily, etf_dir=str(tmp_path / "none"), use_cache=False)
    tbuy = tradability_for_order("600003.SH", "buy", bar)
    tsell = tradability_for_order("600003.SH", "sell", bar)
    assert tbuy["tradable"] is False and tsell["tradable"] is False
    assert tbuy["tradable_reason"] in ("停牌", "当日无成交（停牌）")


# ── 4. 缺行情 ≠ 可成交（不能猜价）──────────────────────────────
def test_missing_quote_is_none_not_true(tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty, exist_ok=True)
    bar = load_latest_bar("600999.SH", empty, etf_dir=empty, use_cache=False)
    assert bar is None
    t = tradability_for_order("600999.SH", "buy", bar)
    assert t["tradable"] is None
    assert "无行情" in t["tradable_reason"]
    assert t["latest_price"] is None


# ── 5. 盘中 bar 不判涨跌停 ─────────────────────────────────────
def test_intraday_bar_is_flagged_and_not_judged(tmp_path, monkeypatch):
    """把「收盘时刻」推到 23:59，当日 bar 就一定是盘中 bar。"""
    monkeypatch.setattr(lq, "_MARKET_CLOSE", (23, 59))
    daily = _daily(str(tmp_path / "d"), "600004.SH",
                   [[TODAY, 10.0, 11.0, 10.0, 10.99, 1e6, 1e8, 9.9, 0.0, 1.0]])
    bar = load_latest_bar("600004.SH", daily, etf_dir=str(tmp_path / "none"), use_cache=False)
    assert bar["intraday"] is True
    t = tradability_for_order("600004.SH", "buy", bar)
    assert t["tradable"] is None
    assert t["price_basis"] == "盘中未收盘"
    assert "未收盘" in t["tradable_reason"]


def test_closed_bar_same_day_is_judged(tmp_path, monkeypatch):
    """收盘后（15:10 之前用 0:00 模拟）同一根 bar 要正常判涨停。"""
    monkeypatch.setattr(lq, "_MARKET_CLOSE", (0, 0))
    daily = _daily(str(tmp_path / "d"), "600005.SH",
                   [[TODAY, 10.0, 11.0, 10.0, 10.99, 1e6, 1e8, 9.9, 0.0, 1.0]])
    bar = load_latest_bar("600005.SH", daily, etf_dir=str(tmp_path / "none"), use_cache=False)
    assert bar["intraday"] is False
    t = tradability_for_order("600005.SH", "buy", bar)
    assert t["tradable"] is False and t["price_basis"] == "收盘价"


# ── 6. 撮合口径不动，只补参考金额与权重 ──────────────────────
def test_enrich_adds_latest_amount_without_touching_fill_fields(tmp_path):
    daily = _daily(str(tmp_path / "d"), "600006.SH",
                   [["2026-11-04", 10.0, 10.5, 10.0, 10.5, 1e6, 1e8, 5.0, 0.0, 1.0]])
    row = {"code": "600006.SH", "name": "样本", "price": 10.0,
           "shares": 1000, "amount": 10000.0, "fee": 8.0}
    out = enrich_order_row(row, "buy", portfolio_value=100000.0,
                           daily_dir=daily, etf_dir=str(tmp_path / "none"))
    # 撮合口径字段原样（纸面账本对账靠它们）
    assert out["price"] == 10.0 and out["shares"] == 1000 and out["amount"] == 10000.0
    # 最新收盘口径
    assert out["latest_price"] == pytest.approx(10.5)
    assert out["ref_amount_latest"] == pytest.approx(10500.0)
    assert out["target_weight_pct"] == pytest.approx(10.0)
    assert out["lots"] == 10 and out["lot_ok"] is True


def test_odd_lot_is_flagged_not_silently_rounded(tmp_path):
    daily = _daily(str(tmp_path / "d"), "600007.SH",
                   [["2026-11-04", 10.0, 10.5, 10.0, 10.5, 1e6, 1e8, 5.0, 0.0, 1.0]])
    row = {"code": "600007.SH", "name": "碎股", "price": 10.0,
           "shares": 1050, "amount": 10500.0, "fee": 8.0}
    out = enrich_order_row(row, "buy", portfolio_value=100000.0,
                           daily_dir=daily, etf_dir=str(tmp_path / "none"))
    assert out["lots"] == 10
    assert out["lot_ok"] is False  # 1050 股不是整手，必须显式标出来


# ── 7. 汇总计数：不可成交笔数不能被吞掉 ──────────────────────
def test_enrich_orders_counts_blocked_and_missing(tmp_path):
    daily = str(tmp_path / "d")
    # 600008 的行情只到 11-03，600009 到 11-04 → 汇总必须报**最旧**那根
    _daily(daily, "600008.SH", [
        ["2026-11-03", 10.0, 10.2, 10.0, 10.1, 1e6, 1e8, 1.0, 0.0, 1.0]])
    _daily(daily, "600009.SH", [
        ["2026-11-03", 10.0, 10.2, 10.0, 10.1, 1e6, 1e8, 1.0, 0.0, 1.0],
        ["2026-11-04", 10.1, 10.99, 10.0, 10.99, 1e6, 1e8, 9.9, 0.0, 1.0]])
    _daily(daily, "600010.SH", [["2026-11-04", 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    buys = [
        {"code": "600008.SH", "name": "正常", "price": 10.0, "shares": 100, "amount": 1000.0},
        {"code": "600009.SH", "name": "涨停", "price": 10.0, "shares": 100, "amount": 1000.0},
        {"code": "600999.SH", "name": "缺行情", "price": 10.0, "shares": 100, "amount": 1000.0},
    ]
    sells = [{"code": "600010.SH", "name": "停牌", "price": 10.0, "shares": 100, "amount": 1000.0}]

    res = enrich_orders(buys, sells, portfolio_value=100000.0,
                        daily_dir=daily, etf_dir=str(tmp_path / "none"))
    assert res["n_orders"] == 4
    # 涨停买不进 + 停牌卖不出 = 2 笔不可成交；缺行情 1 笔（单独计数，不混进来）
    assert res["n_not_tradable"] == 2
    assert res["n_no_quote"] == 1
    # 不可成交的标的必须还在清单里（显式列出，不是静默剔除）
    assert len(res["buys"]) == 3 and len(res["sells"]) == 1
    codes = {r["code"] for r in res["buys"]}
    assert {"600009.SH", "600999.SH"} <= codes
    # 新鲜度必须报**最旧**那根，不能报最新的糊弄过去
    assert res["price_as_of"] == "2026-11-03"
    assert res["price_as_of_newest"] == "2026-11-04"
    assert res["n_price_dates"] == 2
