"""`sync_pct_chg_from_tdx.py` 的 `--since` 日期格式回归测试（2026-11-05）。

真实事故（不是假想）：排期脚本传 `--since "$(date -v-7d +%Y%m%d)"` → `20261029`，
而 TDX 分区目录名是 `date=2026-11-04`。字符串比较里
`"2026-11-04" < "20261029"` 恒为 True（第 5 位 `'-'`(45) < `'1'`(49)）
→ **所有分区被判"太旧"全部跳过** → 脚本打印"没有可用的 TDX raw 分区"却**退出 0**
→ shell 报 "✓ sync pct_chg done" → 后复权涨跌幅层从 2026-11-02 起掉队
→ `adj_close` 11-03/11-04 全 NaN → 纸面盘所有 A 股腿净值被冻在 11-02 的价。

这里锁四件事：格式归一、格式错必须报错、分区真的被选中、"一个分区都没有"必须非 0 退出。
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT = os.path.join(ROOT, "scripts", "sync_pct_chg_from_tdx.py")
SHELL = os.path.join(ROOT, "..", "scripts", "daily_tdx_update.sh")


def _load_module():
    spec = importlib.util.spec_from_file_location("spc_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spc_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


spc = _load_module()


# ── 1. 两种格式必须等价 ─────────────────────────────────────────────────────
def test_compact_and_iso_since_are_equivalent():
    assert spc._normalize_since("20261029") == "2026-10-29"
    assert spc._normalize_since("2026-10-29") == "2026-10-29"
    assert spc._normalize_since("") == ""


def test_bad_since_raises_instead_of_silently_matching_nothing():
    for bad in ("上周", "2026-13-40", "20261"):
        with pytest.raises(SystemExit):
            spc._normalize_since(bad)


# ── 2. 回归：紧凑格式曾经选不出任何分区 ────────────────────────────────────
def _make_raw_dir(tmp_path, days):
    for d in days:
        p = tmp_path / f"date={d}"
        p.mkdir()
        (p / "part.parquet").write_bytes(b"x")
    return str(tmp_path)


def test_partitions_selected_with_compact_since(tmp_path, monkeypatch):
    monkeypatch.setattr(spc, "RAW_DIR", _make_raw_dir(
        tmp_path, ["2026-10-01", "2026-11-03", "2026-11-04"]))
    got = spc.raw_partitions("20261029")          # 修之前：[] —— 这就是事故
    assert [d for d, _ in got] == ["2026-11-03", "2026-11-04"]


def test_old_since_really_returned_nothing_under_string_compare():
    """把根因写死成断言：`2026-11-04 < 20261029` 在字符串比较下成立。"""
    assert "2026-11-04" < "20261029"


# ── 3. "一个分区都没有"必须非 0 退出，不能报成功 ───────────────────────────
def test_no_partitions_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(spc, "RAW_DIR", str(tmp_path))     # 空目录
    monkeypatch.setattr(sys, "argv", ["sync_pct_chg_from_tdx.py",
                                      "--since", "2026-10-29"])
    with pytest.raises(SystemExit) as e:
        spc.main()
    assert e.value.code == 1


# ── 4. 排期脚本不能再传回紧凑格式 ───────────────────────────────────────────
def test_scheduler_passes_iso_since():
    text = open(SHELL, encoding="utf-8").read()
    assert 'sync_pct_chg_from_tdx.py --since "$(date -v-7d +%Y-%m-%d)"' in text
    assert 'sync_pct_chg_from_tdx.py --since "$(date -v-7d +%Y%m%d)"' not in text
