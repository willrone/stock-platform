"""重建路径守卫：ledger 有成交、但 holdings 文件缺失 → 拒绝重建（2026-11-02）。

背景（实测出来的坑）：
  `load_holdings()` 在 holdings 文件**缺失**时返回**空壳**
  （cash=capital、positions={}）。对 multi_asset_final 这种「从没建过仓」的线，
  空壳是对的；但对「ledger 里明明有成交、只是 holdings 文件丢了」的线，
  照空壳重建会让 portfolio.json 静默变成「未建仓」——
  展示层读起来像「这条线没做过」，而它其实做过，ledger 还在。

  这正是本项目反复吃过的亏：**静默降级比报错更贵**（台账 §12.28 审计器/
  回填器共错、§12.31 三处「读反」同源）。

守卫语义：
  · ledger 有成交 + holdings 缺失 → 抛异常 → 上层 main 循环打印告警并
    **保留 portfolio.json 里的旧值**，其它文件一概不动（可从 ledger 恢复）；
  · ledger 无成交（或不存在、只有表头）+ holdings 缺失 → 合法空壳，正常重建；
  · holdings 存在 → 一律正常重建（守卫不拦）。
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "paper_portfolio.py"

LEDGER_HEADER = ["date", "code", "action", "price", "shares", "amount"]


def load_pp_module() -> Any:
    spec = importlib.util.spec_from_file_location("paper_portfolio_guard", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def pp(tmp_path, monkeypatch):
    """PAPER_DIR → 临时目录：测试绝不碰真实账本。"""
    module = load_pp_module()
    monkeypatch.setattr(module, "PAPER_DIR", str(tmp_path))
    return module


def _write_ledger(pp, strategy: str, rows: list[dict]) -> Path:
    p = Path(pp.ledger_path(strategy))
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _write_holdings(pp, strategy: str, data: dict) -> Path:
    p = Path(pp.holdings_path(strategy))
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ── 1) 核心：ledger 有成交 + holdings 缺失 → 抛异常 ────────────────────
def test_guard_raises_when_ledger_has_trades_but_holdings_missing(pp):
    _write_ledger(pp, "22f", [
        {"date": "2026-11-02", "code": "600000", "action": "BUY",
         "price": 10.5, "shares": 1000, "amount": 10500.0},
    ])
    assert not Path(pp.holdings_path("22f")).exists()

    with pytest.raises(RuntimeError) as ei:
        pp.rebuild_line_from_files("22f", 1_000_000)

    msg = str(ei.value)
    assert "拒绝重建" in msg
    assert "1 笔成交" in msg
    assert "22f_holdings.json" in msg


# ── 2) 合法空壳：ledger 不存在 → 不拦（multi_asset_final 的真实状态）───
def test_shell_rebuild_allowed_when_no_ledger_and_no_holdings(pp):
    assert not Path(pp.ledger_path("multi_asset_final")).exists()
    out = pp.rebuild_line_from_files("multi_asset_final", 1_000_000)
    assert isinstance(out, dict)
    # 空壳的价值 = 初始资金（不是 0）：未建仓 ≠ 爆仓
    assert float(out.get("value") or out.get("last_value") or 0) == pytest.approx(1_000_000)


# ── 3) ledger 只有表头（0 笔成交）→ 不拦 ───────────────────────────────
def test_shell_rebuild_allowed_when_ledger_header_only(pp):
    _write_ledger(pp, "22f", [])
    assert pp.ledger_stats("22f")["n_trades"] == 0
    out = pp.rebuild_line_from_files("22f", 1_000_000)
    assert isinstance(out, dict)


# ── 4) holdings 存在 → 一律正常重建（守卫不误伤）──────────────────────
def test_rebuild_ok_when_holdings_present(pp):
    _write_ledger(pp, "22f", [
        {"date": "2026-11-02", "code": "600000", "action": "BUY",
         "price": 10.5, "shares": 1000, "amount": 10500.0},
    ])
    _write_holdings(pp, "22f", {
        "cash": 900_000.0,
        "positions": {"600000": {"quantity": 1000, "last_price": 12.0,
                                 "market_value": 12_000.0}},
        "last_value": 912_000.0,
        "last_trade": "2026-11-02",
        "initial_capital": 1_000_000.0,
    })
    out = pp.rebuild_line_from_files("22f", 1_000_000)
    assert isinstance(out, dict)
    assert float(out.get("value") or 0) == pytest.approx(912_000.0)


# ── 5) 守卫触发时**不写任何文件**（ledger 不动、holdings 不被凭空造出）──
def test_guard_writes_nothing(pp):
    lp = _write_ledger(pp, "22f", [
        {"date": "2026-11-02", "code": "600000", "action": "BUY",
         "price": 10.5, "shares": 1000, "amount": 10500.0},
    ])
    before = lp.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError):
        pp.rebuild_line_from_files("22f", 1_000_000)

    assert lp.read_text(encoding="utf-8") == before, "守卫不得改写 ledger"
    assert not Path(pp.holdings_path("22f")).exists(), "守卫不得凭空造出 holdings"
