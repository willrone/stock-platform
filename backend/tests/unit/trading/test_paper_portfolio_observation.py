"""纸面盘观测口径层（方案书附录 A，2026-11-02）测试。

为什么要有这层测试 —— 2026-11-02 实测到两个误读：

  ① 从未交易的线（0 持仓 / 100% 现金）裸 PnL 显示 **0.00%** → 在裸 PnL 榜上
     排第 2，**高于 4 条真正亏了钱的线**；它其实只是"什么都没做"。
  ② 付费档 gbm_voltgt 裸 PnL **-0.05%**（看着像打平），但它 82.5% 是现金 →
     真正投进去的那 17.5% 实际亏 **0.30%**。

根因：只报**总盈亏**，不报**暴露度** → 仓位量差 5 倍的线放一起比就是误导。
修法：每条线同时给出「投入段收益」+「暴露度」+「as-of」+「成交笔数」+「状态」，
      并且从未成交的线**不显示 0.00%**（显示 — 且 status=未建仓）。

口径只实现一次（paper_portfolio.line_observation_metrics）→ 重建路径与新跑路径
共用，不会出现"两套口径互相共错"。
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "paper_portfolio.py"


def load_pp_module() -> Any:
    spec = importlib.util.spec_from_file_location("paper_portfolio_obs", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def pp(tmp_path, monkeypatch):
    """把纸面盘的 PAPER_DIR 指到临时目录 → 测试不碰真实账本。"""
    module = load_pp_module()
    monkeypatch.setattr(module, "PAPER_DIR", str(tmp_path))
    return module


def _write_holdings(pp, name: str, data: dict) -> None:
    pp.save_holdings(name, data)


def _write_ledger(pp, name: str, rows: list[dict]) -> None:
    p = pp.ledger_path(name)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "action", "code", "price",
                                           "shares", "amount"])
        w.writeheader()
        w.writerows(rows)


# ── 门禁 b：从未成交的线不能显示 0.00% ────────────────────────────────

def test_never_traded_line_has_no_pnl_not_zero(pp):
    """从未成交的线：pnl_pct 必须是 None（不是 0.00），status=未建仓。"""
    _write_holdings(pp, "phantom", {
        "cash": 1_000_000.0, "positions": {}, "initial_capital": 1_000_000.0,
        "last_value": 1_000_000.0,
    })
    line = {"cash": 1_000_000.0, "value": 1_000_000.0, "n_positions": 0}

    m = pp.line_observation_metrics("phantom", line, 1_000_000.0)

    assert m["pnl_pct"] is None, "从未成交的线不得显示 0.00%（会被读成打平）"
    assert m["pnl_pct"] != 0.0
    assert m["status"] == "未建仓"
    assert m["invested_return_pct"] is None, "没投钱就没有投入段收益"
    assert m["exposure"] == 0.0
    assert m["n_trades"] == 0
    assert m["n_trade_days"] == 0


def test_empty_shell_line_is_flagged_not_ranked_as_flat(pp):
    """空壳线（multi_asset_final 现状：有 value 无持仓无账本）→ 未建仓。

    这条线在旧表里显示 0.00% 并排在亏损线之前 → 必须被标出来。
    """
    m = pp.line_observation_metrics("multi_asset_final",
                                   {"cash": 1_000_000.0, "value": 1_000_000.0,
                                    "n_positions": 0}, 1_000_000.0)
    assert m["status"] == "未建仓"
    assert m["pnl_pct"] is None


# ── 门禁 c：投入段收益消除暴露度差异 ─────────────────────────────────

def test_invested_return_excludes_undinvested_cash(pp):
    """82.5% 现金的线：总盈亏 -0.05%，但投入段亏 0.30%（gbm_voltgt 实测值）。"""
    initial = 1_000_000.0
    cash = 824_667.4
    value = 999_477.4
    _write_holdings(pp, "gbm_voltgt", {
        "cash": cash, "positions": {}, "initial_capital": initial,
        "last_value": value, "last_trade": "2026-10-23",
    })
    _write_ledger(pp, "gbm_voltgt", [
        {"date": "2026-10-23", "action": "BUY", "code": "600000.SH",
         "price": "10.0", "shares": "100", "amount": "1000"},
    ])
    line = {"cash": cash, "value": value, "n_positions": 12}

    m = pp.line_observation_metrics("gbm_voltgt", line, initial)

    assert m["status"] == "持仓中"
    # 暴露度 = 1 − cash/value = 174810/999477.4
    assert m["exposure"] == pytest.approx(0.174901, abs=1e-5)
    # 总盈亏：-0.052%
    assert m["pnl_pct"] == pytest.approx(-0.05226, abs=1e-4)
    # 投入段收益：-0.30%（这才是"你投进去的钱赚了/亏了多少"）
    assert m["invested_return_pct"] == pytest.approx(-0.29874, abs=1e-3)


def test_invested_return_normalizes_exposure_and_keeps_sign(pp):
    """投入段收益 = 总盈亏 ÷ 在场资金：同号，但把仓位量差异放大出来。

    同一个总盈亏（+0.05%），暴露度 10% 的线 → 投入段 +0.55%；
    暴露度 90% 的线 → 投入段 +0.06%。这就是"裸 PnL 不可比"的解法。
    """
    initial = 1_000_000.0

    def _mk(name: str, cash: float, value: float) -> dict:
        _write_holdings(pp, name, {"cash": cash, "positions": {},
                                   "initial_capital": initial,
                                   "last_value": value,
                                   "last_trade": "2026-10-23"})
        _write_ledger(pp, name, [{"date": "2026-10-23", "action": "BUY",
                                  "code": "600000.SH", "price": "10.0",
                                  "shares": "100", "amount": "1000"}])
        return pp.line_observation_metrics(name, {"cash": cash, "value": value,
                                                  "n_positions": 1}, initial)

    low_exp = _mk("low", 900_000.0, 1_000_500.0)    # 暴露度 10%
    high_exp = _mk("high", 100_000.0, 1_000_500.0)   # 暴露度 90%

    assert low_exp["pnl_pct"] == pytest.approx(high_exp["pnl_pct"], abs=1e-6)
    # 同号（分子相同），但幅度不同
    assert low_exp["invested_return_pct"] > 0
    assert high_exp["invested_return_pct"] > 0
    assert low_exp["invested_return_pct"] > high_exp["invested_return_pct"]
    assert low_exp["exposure"] < high_exp["exposure"]


def test_zero_value_line_has_no_exposure_and_no_crash(pp):
    """value=0：不能除零崩，也不能编出暴露度。"""
    m = pp.line_observation_metrics("z", {"cash": 0.0, "value": 0.0,
                                          "n_positions": 0}, 1_000_000.0)
    assert m["exposure"] is None
    assert m["pnl_pct"] is None
    assert m["invested_return_pct"] is None


# ── 账本统计 / as-of 诚实 ────────────────────────────────────────────

def test_ledger_stats_counts_rows_and_unique_days(pp):
    _write_ledger(pp, "s", [
        {"date": "2026-10-16", "action": "BUY", "code": "A", "price": "1",
         "shares": "1", "amount": "1"},
        {"date": "2026-10-16", "action": "BUY", "code": "B", "price": "1",
         "shares": "1", "amount": "1"},
        {"date": "2026-10-23", "action": "SELL", "code": "A", "price": "2",
         "shares": "1", "amount": "2"},
    ])
    st = pp.ledger_stats("s")
    assert st["n_trades"] == 3
    assert st["n_trade_days"] == 2
    assert st["last_ledger_date"] == "2026-10-23"


def test_as_of_prefers_recorded_value_date(pp):
    """as_of 优先级：value_as_of > last_trade > 账本最后成交日。"""
    _write_ledger(pp, "s", [{"date": "2026-10-23", "action": "BUY", "code": "A",
                             "price": "1", "shares": "1", "amount": "1"}])
    _write_holdings(pp, "s", {"cash": 1.0, "positions": {},
                              "initial_capital": 1.0, "last_value": 1.0,
                              "last_trade": "2026-10-26",
                              "value_as_of": "2026-10-30"})
    m = pp.line_observation_metrics("s", {"cash": 1.0, "value": 1.0,
                                          "n_positions": 1}, 1.0)
    assert m["as_of"] == "2026-10-30"

    # 没有 value_as_of 时退到 last_trade
    _write_holdings(pp, "s2", {"cash": 1.0, "positions": {},
                               "initial_capital": 1.0, "last_value": 1.0,
                               "last_trade": "2026-10-26"})
    _write_ledger(pp, "s2", [{"date": "2026-10-23", "action": "BUY", "code": "A",
                              "price": "1", "shares": "1", "amount": "1"}])
    m2 = pp.line_observation_metrics("s2", {"cash": 1.0, "value": 1.0,
                                            "n_positions": 1}, 1.0)
    assert m2["as_of"] == "2026-10-26"


def test_missing_ledger_is_not_an_error(pp):
    """没有账本文件 → 0 笔，不是异常（重建路径常见）。"""
    st = pp.ledger_stats("never_existed")
    assert st["n_trades"] == 0
    assert st["last_ledger_date"] is None


# ── 端到端门禁：真实 portfolio.json 的不变量 ─────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3].parent


def test_real_portfolio_json_never_traded_lines_have_no_pnl() -> None:
    """MCP 读的就是这个文件 → 这里破了，MCP 就会把 0.00% 送出去。

    不变量：从未成交的线（n_trades == 0）pnl_pct 必须是 None。
    文件不存在时跳过（CI 上没有数据盘）。
    """
    p = _repo_root() / "data" / "paper" / "portfolio.json"
    if not p.exists():
        pytest.skip("paper portfolio.json 不存在（无数据盘）")
    with open(p) as fh:
        data = json.load(fh)
    lines = data.get("lines") or {}
    assert lines, "portfolio.json 里没有线"
    for name, ln in lines.items():
        if ln.get("n_trades") == 0:
            assert ln.get("pnl_pct") is None, (
                f"{name} 从未成交却带了 pnl_pct={ln.get('pnl_pct')!r} —— "
                f"0.00% 会被读成打平")
            assert ln.get("status") == "未建仓", f"{name} 状态应为未建仓"
        # 已成交的线必须有完整观测字段
        if ln.get("n_trades", 0) > 0:
            for k in ("exposure", "invested_return_pct", "as_of", "status"):
                assert k in ln, f"{name} 缺少观测字段 {k}"
