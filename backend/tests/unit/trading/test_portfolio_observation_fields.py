"""防回归：portfolio.json 的 line 必须带齐观测口径层字段。

背景（2026-11-03，回归）：`fix_adj_cost_backfill.py` 的写盘路径曾只调
`rebuild_line_from_files`，漏掉 `line_observation_metrics` → portfolio.json 的
`as_of / exposure / invested_return_pct / pnl_display / status / n_trades` **全丢**：
- 产物级门禁 `verify_paper_observation_gates.py` 的 a/b/c 全部读到 None → 全挂
- 前端 `app/services/trading/signal_service.py` 读的 exposure / invested_return_pct /
  pnl_display 也变 None → 用户页面上的"暴露度 / 投入段收益"消失

现在口径已收口到 `paper_portfolio.rebuild_portfolio_lines()`，
本测试盯住**产物**（落盘的 portfolio.json），而不是函数——
函数对、但写盘路径绕过它的情况，正是这次回归的形态。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[3]
ROOT = BACKEND_ROOT.parent
PORTFOLIO = ROOT / "data" / "paper" / "portfolio.json"

REQUIRED_KEYS = (
    "as_of", "exposure", "invested_return_pct",
    "pnl_display", "status", "n_trades", "n_trade_days",
)


@pytest.fixture(scope="module")
def portfolio() -> dict:
    if not PORTFOLIO.exists():
        pytest.skip(f"没有 {PORTFOLIO}（未跑过纸面盘）")
    with open(PORTFOLIO) as fh:
        return json.load(fh)


def test_every_line_has_observation_fields(portfolio):
    lines = portfolio.get("lines") or {}
    assert lines, "portfolio.json 没有 lines"
    missing = {}
    for name, ln in lines.items():
        gap = [k for k in REQUIRED_KEYS if k not in ln]
        if gap:
            missing[name] = gap
    assert not missing, f"这些线缺观测口径字段（重建路径绕过 line_observation_metrics?）: {missing}"


def test_observation_values_are_self_consistent(portfolio):
    """exposure 必须等于 1 − cash/value（门禁 a 的口径）。"""
    lines = portfolio.get("lines") or {}
    for name, ln in lines.items():
        value = ln.get("value")
        cash = ln.get("cash")
        exposure = ln.get("exposure")
        if value is None or cash is None or exposure is None:
            continue
        expected = 1.0 - float(cash) / float(value)
        assert math.isclose(float(exposure), expected, abs_tol=1e-4), (
            f"{name}: exposure={exposure} ≠ 1−cash/value={expected}")


def test_rebuild_portfolio_lines_exists_and_returns_all_lines():
    """共享收口函数必须在，且覆盖全部策略。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pp_mod", str(BACKEND_ROOT / "scripts" / "paper_portfolio.py"))
    assert spec and spec.loader
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    assert hasattr(pp, "rebuild_portfolio_lines"), "缺少共享重建函数 rebuild_portfolio_lines"
    lines = pp.rebuild_portfolio_lines(1_000_000.0, strict=True)
    assert set(lines) == set(pp.STRATEGIES), (
        f"重建的线与 STRATEGIES 不一致: {sorted(set(lines) ^ set(pp.STRATEGIES))}")
    for name, ln in lines.items():
        gap = [k for k in REQUIRED_KEYS if k not in ln]
        assert not gap, f"{name} 缺字段 {gap}"


def test_observation_gates_script_passes_on_current_artifact():
    """产物级门禁脚本对**当前落盘产物**必须通过（2026-11-03 §12.40 建议：挂进常规回归）。

    之前门禁是手动跑的 —— 它坏了（§12.39 那次 21 条全挂）没人知道，
    是靠一条过期通知才被牵出来的。这里把它接进 `tests/unit`：
    每次跑单测都会验一遍门禁脚本对产物的判定。
    """
    import subprocess
    import sys

    script = BACKEND_ROOT / "scripts" / "verify_paper_observation_gates.py"
    if not script.exists():
        pytest.skip(f"门禁脚本不存在: {script}")
    if not PORTFOLIO.exists():
        pytest.skip(f"没有 {PORTFOLIO}（未跑过纸面盘）")

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=120,
        cwd=str(BACKEND_ROOT),
    )
    assert proc.returncode == 0, (
        "产物级门禁没过（verify_paper_observation_gates.py 退出码 "
        f"{proc.returncode}）：\n{proc.stdout}\n{proc.stderr}")

