"""ETF 袖套腿 / 母线（定稿线）日频盯市的门禁测试（2026-11-05，自用化 A 项）。

用**合成数据**（手写 ledger + 手写 ETF bars + 小交易日日历），
不依赖真实 `data/paper/` 与全市场行情 → 跑得快、可离线复现。

## 守什么

| # | 门禁 | 破坏后会怎样 |
|---|---|---|
| a | 袖套估值 = **股数 × 不晚于当日的 ETF 收盘价**（前复权，**不做后复权比值**） | 用 A 股腿的 `cost×比值` 会在 ETF 分红重锚后系统性错算 |
| b | 袖套持仓语义是 `moving_average`（**部分卖出**、每股成本不变） | 用 A 股腿的 `full_liquidation` 会把「削减超配」卖掉的仓位当成整仓清空 |
| c | 行情缺口允许前向填充，但**陈旧度必须记账**（`stale_price` + `bar_lag_days`） | 静默前向填充 = 造出 0 收益点 → 压低波动、稀释夏普 |
| d | 新鲜度门禁：最新点滞后 > `SLEEVE_MAX_STALE_DAYS` → `bar_freshness_ok=False` | ETF 管道没到货却把旧价当今天净值，没人报警 |
| e | 现金残差门禁：重放现金 ≠ holdings 现金 → `cash_residual_ok=False` | 记账口径漂移被曲线掩盖 |
| f | 门禁 a（日频 vs 周频末点）对袖套腿同样生效 | 两条腿各自"看起来一致" |
| g | 母线只合并**两条腿都有点**的日期，且每个点 `legs` 两项都 > 0 | 缺一条腿还相加 = 造出 −50% 的假暴跌点 |
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[3]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SCRIPT = BACKEND / "scripts" / "paper_daily_nav.py"

DATES = pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04",
                        "2026-11-05", "2026-11-06"])

CAP0 = 500_000.0
ETF_COMMISSION = 0.0003  # 与 app 层同一数值（真正的同源检查在 test_ledger_replay_etf_rate.py）

# 合成成交：11-02 建仓两笔；11-04 **部分卖出**（袖套调仓削减超配）
LEDGER_ROWS_SPEC = [
    dict(date="2026-11-02", action="BUY", code="518880", name="华安黄金ETF",
         price=7.941, shares=7000.0, amount=55592.56),
    dict(date="2026-11-02", action="BUY", code="159980", name="大成有色金属期货ETF",
         price=2.094, shares=26000.0, amount=54444.00),
    dict(date="2026-11-04", action="SELL", code="518880", name="华安黄金ETF",
         price=8.100, shares=2000.0, amount=15882.00),
]

CLOSES: Dict[str, List[float]] = {
    "518880": [7.941, 8.000, 8.100, 8.200, 8.300],
    "159980": [2.094, 2.100, 2.120, 2.150, 2.160],
}


def _bars(up_to: int = 5):
    """造 `EtfBars`；`up_to` 控制行情只给到第几个交易日（模拟管道缺口）。"""
    frames = {}
    for code, closes in CLOSES.items():
        sl = closes[:up_to]
        frames[code] = pd.DataFrame({
            "date": DATES[:up_to],
            "open": sl, "high": sl, "low": sl, "close": sl,
            "volume": [1_000_000.0] * len(sl),
        })
    from app.services.trading import multi_asset_service as mas
    return mas.EtfBars(frames)


def _cash_after(idx: int) -> float:
    """按 ETF 费率手算到第 idx 笔之后的现金（测试的独立参照实现）。"""
    cash = CAP0
    for r in LEDGER_ROWS_SPEC[:idx]:
        if r["action"] == "BUY":
            cash -= r["amount"] * (1.0 + ETF_COMMISSION)
        else:
            cash += r["amount"] * (1.0 - ETF_COMMISSION)
    return cash


CASH_AFTER_BUILD = _cash_after(2)     # 建仓后
CASH_AFTER_SELL = _cash_after(3)      # 部分卖出后
# holdings 里写的是**舍入到 2 位**的真值 → 重放残差应在 0.05 内
HOLDINGS_CASH = round(CASH_AFTER_SELL, 2)


def _holdings(cash: float = HOLDINGS_CASH,
              nav_history: Any = None) -> Dict[str, Any]:
    return {
        "initial_capital": CAP0,
        "cash": cash,
        "positions": {
            "518880": {"shares": 5000.0, "cost": 7.941794,
                       "buy_date": "2026-11-02", "is_etf": True},
            "159980": {"shares": 26000.0, "cost": 2.094209,
                       "buy_date": "2026-11-02", "is_etf": True},
        },
        "nav_history": nav_history if nav_history is not None else [],
    }


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dn():
    return _load(SCRIPT, "sleeve_daily_nav_test")


@pytest.fixture
def wired(dn, monkeypatch):
    """把袖套腿的账本/持仓/行情源接到合成数据上（不碰真实 data/paper）。"""
    hold = _holdings()

    def fake_load_holdings(strategy: str, cap: float) -> Dict[str, Any]:
        h = dict(hold)
        h["_requested_cap"] = cap
        return h

    def fake_parse_ledger(path: str):
        rows = [dn.lr.LedgerRow(
            date=r["date"], action=r["action"], code=r["code"], name=r["name"],
            price=r["price"], shares=r["shares"], amount=r["amount"],
            commission=round(r["amount"] * ETF_COMMISSION, 2),
            stamp_tax=0.0, slippage=0.0, score=None,
            cost_amount=None, realized_pnl=None,
            cash_after=None, portfolio_value=None)
            for r in LEDGER_ROWS_SPEC]
        return rows, {"n_rows": len(rows), "contract": "15"}

    monkeypatch.setattr(dn.pp, "load_holdings", fake_load_holdings)
    monkeypatch.setattr(dn.lr, "parse_ledger", fake_parse_ledger)
    monkeypatch.setattr(dn.mas, "load_etf_bars",
                        lambda codes, base_dir=None: _bars())
    monkeypatch.setattr(dn.mas, "missing_codes", lambda codes, base_dir=None: [])
    return dn


def _run(dn, **kw):
    return dn.build_sleeve_points(DATES, **kw)


def _by_date(points):
    return {p["date"]: p for p in points}


# ── a：估值口径 = 股数 × 当日收盘价（无前复权比值）────────────────────
def test_sleeve_value_is_shares_times_close_on_that_day(wired):
    points, diag = _run(wired)
    assert [p["date"] for p in points] == [str(d.date()) for d in DATES]
    assert diag["cash_residual_ok"] is True, diag.get("cash_residual")

    by = _by_date(points)
    d0 = str(DATES[0].date())
    expect = (CASH_AFTER_BUILD + 7000 * 7.941 + 26000 * 2.094)
    assert by[d0]["value"] == pytest.approx(expect, abs=0.01)
    # 建仓当天没有滞后 → 不许出现 stale
    assert by[d0]["stale_price"] is None
    assert by[d0]["bar_lag_days"] == 0


# ── b：部分卖出后仍持有，每股成本不变（moving_average）─────────────────
def test_partial_sell_keeps_remaining_position_at_same_cost(wired):
    points, _ = _run(wired)
    by = _by_date(points)
    d_sell = "2026-11-04"
    # 7000 − 2000 = 5000 股，另一只不变
    expect = (CASH_AFTER_SELL + 5000 * 8.100 + 26000 * 2.120)
    assert by[d_sell]["value"] == pytest.approx(expect, abs=0.01)
    assert by[d_sell]["n_positions"] == 2
    # 卖出**当天之前**的日子必须还是 7000 股
    d_before = "2026-11-03"
    expect_before = (CASH_AFTER_BUILD + 7000 * 8.000 + 26000 * 2.100)
    assert by[d_before]["value"] == pytest.approx(expect_before, abs=0.01)


# ── c：行情缺口前向填充，但陈旧度必须记账 ─────────────────────────────
def test_forward_fill_is_flagged_not_silent(wired, monkeypatch):
    # 行情只给到 11-02（管道没到货）
    monkeypatch.setattr(wired.mas, "load_etf_bars",
                        lambda codes, base_dir=None: _bars(up_to=1))
    points, diag = _run(wired)
    by = _by_date(points)
    assert by["2026-11-02"]["stale_price"] is None
    for d in ("2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"):
        assert by[d]["stale_price"], f"{d} 用了旧价却没标记"
        assert by[d]["bar_lag_days"] >= 1
    assert by["2026-11-06"]["bar_lag_days"] == 4
    assert diag["n_stale_days"] == 4


# ── d：新鲜度门禁（> SLEEVE_MAX_STALE_DAYS 必须失败）──────────────────
def test_freshness_gate_fails_when_bars_too_old(wired, monkeypatch):
    monkeypatch.setattr(wired.mas, "load_etf_bars",
                        lambda codes, base_dir=None: _bars(up_to=1))
    _points, diag = _run(wired)
    assert diag["bar_freshness_ok"] is False
    assert "etf_data_pipeline" in diag.get("bar_freshness_note", "")


def test_freshness_gate_tolerates_within_limit(wired, monkeypatch):
    limit = wired.SLEEVE_MAX_STALE_DAYS
    monkeypatch.setattr(wired.mas, "load_etf_bars",
                        lambda codes, base_dir=None: _bars(up_to=len(DATES) - limit))
    _points, diag = _run(wired)
    assert diag["bar_lag_days_last"] == limit
    assert diag["bar_freshness_ok"] is True


# ── e：现金残差门禁 ───────────────────────────────────────────────────
def test_cash_residual_gate_fires_when_holdings_cash_is_off(wired, monkeypatch):
    """holdings 的 cash 被人为抬高 5 元 → 门禁必须红（不许"差不多就行"）。"""
    bad = _holdings(cash=HOLDINGS_CASH + 5.0)
    monkeypatch.setattr(wired.pp, "load_holdings", lambda s, cap: dict(bad))
    _points, diag = _run(wired)
    assert diag["cash_residual_ok"] is False
    # 残差 = 重放现金 − holdings 现金 ≈ −5.00
    assert diag["cash_residual"] == pytest.approx(
        CASH_AFTER_SELL - HOLDINGS_CASH - 5.0, abs=1e-3)


def test_cash_residual_gate_passes_on_consistent_holdings(wired):
    _points, diag = _run(wired)
    assert diag["cash_residual_ok"] is True
    assert abs(diag["cash_residual"]) <= 0.05


# ── f：门禁 a 对袖套腿同样生效 ────────────────────────────────────────
def test_nav_history_consistency_gate_on_sleeve(wired, monkeypatch):
    def value_at(points, date):
        return _by_date(points)[date]["value"]

    points, _ = _run(wired)
    nav_ok = [{"date": "2026-11-02", "value": value_at(points, "2026-11-02")}]
    monkeypatch.setattr(wired.pp, "load_holdings",
                        lambda s, cap: _holdings(nav_history=nav_ok))
    _p, diag = _run(wired)
    assert diag["nav_consistency_ok"] is True

    nav_bad = [{"date": "2026-11-02",
                "value": value_at(points, "2026-11-02") + 1.0}]
    monkeypatch.setattr(wired.pp, "load_holdings",
                        lambda s, cap: _holdings(nav_history=nav_bad))
    _p, diag = _run(wired)
    assert diag["nav_consistency_ok"] is False
    assert diag["nav_consistency_diff"] == pytest.approx(1.0, abs=0.01)


# ── g：母线只在两条腿齐的日期合并，且腿值必须都在 ─────────────────────
A_POINTS = [
    {"date": "2026-11-02", "value": 499225.89, "n_positions": 60},
    {"date": "2026-11-03", "value": 499500.00, "n_positions": 60},
    {"date": "2026-11-04", "value": 499225.89, "n_positions": 60},
]
S_POINTS = [
    {"date": "2026-11-02", "value": 499800.07, "n_positions": 9, "bar_lag_days": 0},
    {"date": "2026-11-04", "value": 501589.10, "n_positions": 9, "bar_lag_days": 0},
]


def test_parent_merges_only_dates_where_both_legs_exist(dn):
    points, diag = dn.build_parent_points(A_POINTS, S_POINTS, 1_000_000.0)
    assert [p["date"] for p in points] == ["2026-11-02", "2026-11-04"]
    assert diag["dates_only_a"] == ["2026-11-03"]      # 缺袖套 → 不许合并
    assert diag["dates_only_sleeve"] == []


def test_parent_point_is_sum_of_legs_with_both_legs_present(dn):
    points, _diag = dn.build_parent_points(A_POINTS, S_POINTS, 1_000_000.0)
    for p in points:
        assert p["legs"]["a"] > 0 and p["legs"]["sleeve"] > 0, "缺一条腿还相加"
        assert p["value"] == pytest.approx(p["legs"]["a"] + p["legs"]["sleeve"],
                                          abs=0.01)
    assert points[0]["value"] == pytest.approx(999025.96, abs=0.01)
    assert points[1]["value"] == pytest.approx(1000814.99, abs=0.01)
    # return_pct 必须相对**母线** 100 万，不是相对某一条腿的 50 万
    assert points[0]["return_pct"] == pytest.approx(-0.0974, abs=0.001)


def test_parent_never_produces_half_portfolio_crash(dn):
    """只有一条腿时，母线必须是空序列（而不是 −50% 的假暴跌）。"""
    points, diag = dn.build_parent_points(A_POINTS, [], 1_000_000.0)
    assert points == []
    assert diag["n_a_points"] == 3 and diag["n_sleeve_points"] == 0
    assert diag["dates_only_a"] == ["2026-11-02", "2026-11-03", "2026-11-04"]


def test_parent_stale_marks_union_of_legs(dn):
    a = [{"date": "2026-11-02", "value": 100.0, "n_positions": 1,
          "stale_price": ["600000.SH"]}]
    s = [{"date": "2026-11-02", "value": 200.0, "n_positions": 1,
          "stale_price": ["518880"], "bar_lag_days": 2}]
    points, _ = dn.build_parent_points(a, s, 1_000_000.0)
    assert points[0]["stale_price"] == ["518880", "600000.SH"]
    assert points[0]["bar_lag_days"] == 2


# ── h：缺 ETF 行情文件必须硬失败（不静默降级）─────────────────────────
def test_missing_etf_files_hard_fail(wired, monkeypatch):
    monkeypatch.setattr(wired.mas, "missing_codes",
                        lambda codes, base_dir=None: ["518880"])
    points, diag = _run(wired)
    assert points == []
    assert "缺 ETF 行情文件" in diag.get("status", "")
