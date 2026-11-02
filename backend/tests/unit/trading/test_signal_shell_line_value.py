"""空壳线（从未成交、无 holdings 文件）的市值回落测试 —— 2026-11-02。

为什么要有这组测试
------------------
纸面盘里有一条"空壳线" `multi_asset_final`：注册表 portfolio.json 里有
value=1,000,000 / cash=1,000,000 / 0 持仓 / status=未建仓，但它**从来没有
holdings 文件**（没建过仓，自然没有持仓文件）。

原实现：

    cash  = float(holdings.get("cash", 0) or 0)        # → 0
    value = float(holdings.get("last_value", 0) or 0)  # → 0

→ 接口报 "市值 0" → 前端卡片大字显示 "¥0" → 读者读成"这条线亏光了/爆仓了"，
实际是"**100 万全在现金里，一股没买**"。这是典型的读反。

修法：以 portfolio.json 注册表为准回落（注册表才是纸面盘的真相来源）。
同时守住一个坑：**cash=0 / value=0 是合法值**（满仓线的现金就是 0），
判空必须用 `is not None` 而不是 `or`，否则满仓线会被注册表覆盖回 100 万。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.services.trading import signal_service as ss


SHELL = "multi_asset_final"
FULL = "22f"


@pytest.fixture()
def paper_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 PAPER_DIR 指到临时目录 → 测试不碰真实账本。"""
    monkeypatch.setattr(ss, "PAPER_DIR", str(tmp_path))
    return tmp_path


def _write_portfolio(paper_dir: Path, lines: dict[str, dict]) -> None:
    (paper_dir / "portfolio.json").write_text(
        json.dumps({"meta": {}, "lines": lines}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_holdings(paper_dir: Path, key: str, data: dict) -> None:
    (paper_dir / f"{key}_holdings.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def _shell_line() -> dict[str, Any]:
    """注册表里的空壳线：100 万现金、0 持仓、从未成交。"""
    return {
        "strategy": SHELL,
        "cash": 1_000_000.0,
        "value": 1_000_000.0,
        "n_positions": 0,
        "buys": [],
        "sells": [],
        "holds": [],
        "pnl_pct": None,
        "pnl_display": "—",
        "exposure": 0.0,
        "invested_return_pct": None,
        "as_of": None,
        "status": "未建仓",
        "n_trades": 0,
        "initial_capital": 1_000_000.0,
    }


# ── ① 空壳线：市值回落到注册表，不能是 0 ────────────────────────────

def test_shell_line_value_is_not_zero(paper_dir: Path) -> None:
    """无 holdings 文件的线：value/cash = 100 万（不是 0）。"""
    _write_portfolio(paper_dir, {SHELL: _shell_line()})

    out = ss.TradingSignalService().daily_signal(SHELL)

    assert out["value"] == 1_000_000.0, (
        f"空壳线市值回落成 {out['value']} —— 会被前端读成『市值归零/爆仓』，"
        "实际是 100 万全在现金里"
    )
    assert out["cash"] == 1_000_000.0
    assert out["capital"] == 1_000_000.0
    assert out["pnl_display"] == "—", "从未成交的线不得显示 0.00%（会被读成打平）"
    assert out["status"] == "ok"


# ── ② 满仓线：cash=0 / value 合法，不能被注册表覆盖 ─────────────────

def test_zero_cash_line_is_not_overridden_by_registry(paper_dir: Path) -> None:
    """持仓文件里 cash=0 是**合法值**（满仓）→ 不能回落到注册表的 100 万现金。"""
    _write_portfolio(paper_dir, {
        FULL: {
            "strategy": FULL,
            "cash": 1_000_000.0,          # 注册表是旧的/错的（空壳态）
            "value": 1_000_000.0,
            "n_positions": 3,
            "pnl_pct": -1.2,
            "pnl_display": "-1.20%",
            "status": "持仓中",
            "n_trades": 12,
            "initial_capital": 1_000_000.0,
        }
    })
    _write_holdings(paper_dir, FULL, {
        "cash": 0.0,                       # 满仓：现金真的是 0
        "last_value": 988_000.0,
        "initial_capital": 1_000_000.0,
        "positions": {
            "600000": {"shares": 1000, "cost": 10.0},
        },
    })

    out = ss.TradingSignalService().daily_signal(FULL)

    assert out["cash"] == 0.0, "满仓线的现金是 0，不能被注册表回填成 100 万"
    assert out["value"] == 988_000.0, "市值必须以持仓文件为准"


# ── ③ 策略列表：last_value 同样回落，别给 null ───────────────────────

def test_list_strategies_last_value_falls_back_to_registry(paper_dir: Path) -> None:
    """列表接口的 last_value 对空壳线也不能是 null（=100 万）。"""
    _write_portfolio(paper_dir, {SHELL: _shell_line()})

    items = {s["key"]: s for s in ss.TradingSignalService().list_strategies()}

    assert SHELL in items, "multi_asset_final 应在策略列表里"
    assert items[SHELL]["last_value"] == 1_000_000.0, (
        "空壳线的 last_value 不能是 null —— 会被前端读成『这条线没数据』，"
        "实际是『100 万现金待部署』"
    )
    assert items[SHELL]["status"] == "未建仓"
