"""fix_adj_cost_backfill 的守卫测试（故障注入）。

背景（2026-11-02）：袖套估值抛异常时，旧代码把 s_value 置 None，
`total = a_value + (s_value or 0.0)` → 市值只剩 A 腿（约 499,226），
**却照样打印成正常结果并写进 portfolio.json** —— 袖套腿凭空消失而不报错。
现在改成：袖套估值失败 → 整条合并线标记 failed → main 中止写盘并 exit 1。

这里不靠"读代码觉得对"，而是**真的注入故障**验证三条断言：
1. fix_merged_line 返回 failed=True 且 value=None（不是半个市值）
2. 合并线 holdings 完全没被写（save_holdings 未调用）
3. main 遇到 failed 会 exit(1) 且不重建 portfolio.json
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = BACKEND_ROOT / "scripts" / "fix_adj_cost_backfill.py"
_spec = importlib.util.spec_from_file_location("fix_adj_cost_backfill_mod", str(_SCRIPT))
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class _Boom(RuntimeError):
    pass


def _fake_paper(n_pos: int = 3):
    return {
        "positions": {f"60000{i}.SH": {"shares": 100, "adj_cost": 10.0, "buy_date": "2026-10-16"}
                      for i in range(n_pos)},
        "cash": 500_000.0,
        "last_value": 999_025.96,
    }


@pytest.fixture()
def sleeve_broken(monkeypatch):
    """让 load_etf_bars 抛异常 = 袖套估值失败。"""
    def _boom(*_a, **_k):
        raise _Boom("模拟 ETF 行情缺失")
    monkeypatch.setattr(mod, "load_etf_bars", _boom)


def test_merged_line_marks_failed_and_returns_no_value(sleeve_broken, monkeypatch):
    monkeypatch.setattr(mod.pp, "load_holdings", lambda *_a, **_k: _fake_paper())
    monkeypatch.setattr(mod.pp, "total_return_portfolio_value",
                        lambda *_a, **_k: (499_225.89, None))
    monkeypatch.setattr(mod.pp, "save_holdings", lambda *_a, **_k: pytest.fail("不应写盘"))

    r = mod.fix_merged_line(pd.DataFrame(), pd.Timestamp("2026-11-02"), dry=True)
    assert r["failed"] is True, f"应标记 failed，实际 {r}"
    assert r["value"] is None, f"不应返回半残市值，实际 {r['value']}"
    assert "模拟 ETF 行情缺失" in (r.get("error") or "")


def test_merged_line_ok_path_unaffected(monkeypatch):
    """对照组：袖套估值正常 → 不标记 failed，市值 = A 腿 + 袖套。"""
    monkeypatch.setattr(mod.pp, "load_holdings", lambda *_a, **_k: _fake_paper())
    monkeypatch.setattr(mod.pp, "total_return_portfolio_value",
                        lambda *_a, **_k: (499_225.89, None))
    monkeypatch.setattr(mod, "load_etf_bars", lambda *_a, **_k: object())
    monkeypatch.setattr(mod, "sleeve_positions_value", lambda *_a, **_k: 499_800.07)
    saved = {}
    monkeypatch.setattr(mod.pp, "save_holdings",
                        lambda strat, paper: saved.setdefault(strat, paper))

    r = mod.fix_merged_line(pd.DataFrame(), pd.Timestamp("2026-11-02"), dry=False)
    assert r["failed"] is False
    # s_value = 袖套现金(500,000) + 袖套持仓市值(499,800.07) = 999,800.07
    # total  = A 腿 499,225.89 + 袖套 999,800.07 = 1,499,025.96
    assert r["value"] == 1_499_025.96, r["value"]
    assert mod.MERGED in saved, "正常路径必须写合并线"


def test_main_exits_nonzero_and_does_not_write_portfolio(sleeve_broken, monkeypatch):
    """main 遇到合并线失败 → exit(1)，且 portfolio.json 不被重建/写入。"""
    monkeypatch.setattr(mod.pp.ds, "list_universe", lambda: [])
    monkeypatch.setattr(mod.pp.ds, "load_data", lambda _c: ([], None, [pd.Timestamp("2026-11-02")]))
    monkeypatch.setattr(mod.pp, "load_adjusted_close", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(mod, "fix_line", lambda *_a, **_k: {
        "strategy": "x", "n_pos": 0, "fixed": 0, "date_fixed": 0,
        "value": 1.0, "value_before": 1.0})
    monkeypatch.setattr(mod.pp, "load_holdings", lambda *_a, **_k: _fake_paper())
    monkeypatch.setattr(mod.pp, "total_return_portfolio_value",
                        lambda *_a, **_k: (499_225.89, None))
    monkeypatch.setattr(mod.pp, "save_holdings", lambda *_a, **_k: pytest.fail("不应写盘"))

    # argparse 会读 sys.argv（pytest 的参数）→ 必须先清掉
    monkeypatch.setattr(sys, "argv", ["fix_adj_cost_backfill.py"])
    # 指向临时目录：失败时不得在这里生成 portfolio.json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(mod.pp, "PAPER_DIR", td)
        with pytest.raises(SystemExit) as e:
            mod.main()
        assert e.value.code == 1, f"退出码应为 1，实际 {e.value.code}"
        assert not (Path(td) / "portfolio.json").exists(), "失败时不得写 portfolio.json"
