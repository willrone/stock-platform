"""`check_data_freshness.py` 的门禁测试（2026-11-05 新增，此前该脚本**零测试**）。

为什么必须有：这道闸的职责就是"数据陈旧就报警"，而它刚刚漏掉了一次真实事故——
TDX 原始包到 2026-11-04、后复权涨跌幅层停在 11-02，它照样打印
"✅ 数据已是最新（2026-11-04）"。原因是它只查 `data/tdx_daily/raw` 分区，
不查真正喂估值的 `data/parquet/baostock_daily`（pct_chg）层。

测试口径：直接调 `main()` 拿退出码（不启子进程），`is_trading_day` 打桩，
日期一律用 `--latest-override` / `--pct-latest-override` / `--now` 注入，
所以不依赖仓库数据、也不依赖今天星期几。
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCRIPT = os.path.join(ROOT, "scripts", "check_data_freshness.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("cdf_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cdf_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


cdf = _load_module()


def _run(monkeypatch, argv, trading_day=True):
    monkeypatch.setattr(cdf, "is_trading_day", lambda d: trading_day)
    monkeypatch.setattr(sys, "argv", ["check_data_freshness.py", *argv])
    return cdf.main()


# ── 原有语义：原始包陈旧必须变红（此前无测试证明它能变红） ──────────────────
def test_trading_day_past_publish_with_stale_data_fails(monkeypatch):
    rc = _run(monkeypatch, [
        "--fail-if-stale",
        "--now", "2026-11-04 18:00",
        "--latest-override", "2026-11-02",
    ])
    assert rc == 1, "交易日已过发布时刻但数据停在 11-02 → 必须退出 1"


def test_trading_day_before_publish_is_tolerated(monkeypatch):
    rc = _run(monkeypatch, [
        "--fail-if-stale",
        "--now", "2026-11-04 16:00",
        "--latest-override", "2026-11-03",
    ])
    assert rc == 0, "还没到发布时刻，数据落后属正常"


def test_non_trading_day_with_stale_data_is_ok(monkeypatch):
    rc = _run(monkeypatch, [
        "--fail-if-stale",
        "--now", "2026-11-08 18:00",
        "--latest-override", "2026-11-06",
    ], trading_day=False)
    assert rc == 0


# ── 新增：估值层（涨跌幅）掉队必须变红 —— 这是原门禁的盲区 ──────────────────
def test_pct_layer_lagging_raw_turns_gate_red(monkeypatch):
    """正是 2026-11-03/11-04 的真实状态：原始包到 11-04，涨跌幅层停在 11-02。"""
    rc = _run(monkeypatch, [
        "--fail-if-stale",
        "--check-pct-layer",
        "--now", "2026-11-04 18:00",
        "--latest-override", "2026-11-04",
        "--pct-latest-override", "2026-11-02",
    ])
    assert rc == 1, "估值层落后原始包 → 门禁必须变红（否则纸面盘净值被冻在旧价）"


def test_pct_layer_fresh_passes(monkeypatch):
    rc = _run(monkeypatch, [
        "--fail-if-stale",
        "--check-pct-layer",
        "--now", "2026-11-04 18:00",
        "--latest-override", "2026-11-04",
        "--pct-latest-override", "2026-11-04",
    ])
    assert rc == 0


def test_pct_layer_missing_turns_gate_red(monkeypatch):
    """涨跌幅层整个找不到 = 后复权估值无从算起，不能当正常。"""
    monkeypatch.setattr(cdf, "latest_pct_chg_date", lambda sample=24: None)
    rc = _run(monkeypatch, [
        "--fail-if-stale",
        "--check-pct-layer",
        "--now", "2026-11-04 18:00",
        "--latest-override", "2026-11-04",
    ])
    assert rc == 1


def test_pct_gate_off_by_default_does_not_change_old_behaviour(monkeypatch):
    """不开 --check-pct-layer 时行为不变（不破坏既有排期/测试）。"""
    monkeypatch.setattr(cdf, "latest_pct_chg_date",
                        lambda sample=24: pytest.fail("不该被调用"))
    rc = _run(monkeypatch, [
        "--fail-if-stale",
        "--now", "2026-11-04 18:00",
        "--latest-override", "2026-11-04",
    ])
    assert rc == 0
