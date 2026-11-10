"""纸面盘**进程互斥锁**测试（2026-11-02）。

为什么要这个锁 —— 2026-11-02 实测：
  纸面盘有**三个排期**都会调 `scripts/paper_portfolio.py`：
    ① launchd com.rone.tdx-update  16:05（daily_tdx_update.sh 第 5 步，周一）
    ② Hermes cron tdx-daily-update 17:30（同一个 daily_tdx_update.sh）
    ③ launchd com.rone.paper-portfolio 17:40（+ 周二 09:20 兜底）
  实测 ② 跑到 17:44:55，而 ③ 在 17:40:04 就启动了 → **两个进程并发写**
  同一批 holdings/ledger/portfolio.json → 后写者赢 → 成交明细可能被覆盖。
  今晚没出事是运气（② 正好漏了 honest_42f_neu，③ 补上了）。

锁的语义（关键）：**等待，不跳过**。
  若 ③ 直接跳过，② 漏掉的线就永远补不回来 —— 今晚 honest_42f_neu 正是
  靠 ③ 才补成交成功的（买 3 卖 5）。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "paper_portfolio.py"


def load_pp_module() -> Any:
    spec = importlib.util.spec_from_file_location("paper_portfolio_lock", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def pp(tmp_path, monkeypatch):
    module = load_pp_module()
    monkeypatch.setattr(module, "PAPER_DIR", str(tmp_path))
    return module


def _holder(tmp_path, hold_s: float) -> subprocess.Popen:
    """起一个子进程持锁 hold_s 秒（模拟另一个排期正在跑纸面盘）。"""
    lock_file = tmp_path / ".paper_portfolio.lock"
    code = (
        "import fcntl, os, sys, time\n"
        f"fd = os.open({str(lock_file)!r}, os.O_CREAT | os.O_RDWR, 0o644)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "print('locked', flush=True)\n"
        f"time.sleep({hold_s})\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "locked"
    return proc


# ── 1) 无竞争时正常拿到锁 ────────────────────────────────────────────
def test_acquires_lock_when_free(pp):
    fd = pp.acquire_run_lock(timeout_s=5)
    assert fd is not None
    assert os.path.exists(pp.lock_path())


# ── 2) 别人持锁 + 超时 → exit 75（不是"跳过"，是明确失败）────────────
def test_timeout_exits_75_when_other_instance_holds_lock(pp, tmp_path):
    proc = _holder(tmp_path, 30.0)
    try:
        with pytest.raises(SystemExit) as ei:
            pp.acquire_run_lock(timeout_s=1, poll_s=0.2)
        assert ei.value.code == 75, "超时必须以 EX_TEMPFAIL(75) 退出"
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ── 3) 别人持锁但很快释放 → **等到**再拿锁（核心语义）────────────────
def test_waits_for_lock_then_acquires(pp, tmp_path):
    proc = _holder(tmp_path, 2.0)
    try:
        t0 = time.time()
        fd = pp.acquire_run_lock(timeout_s=30, poll_s=0.2)
        waited = time.time() - t0
        assert fd is not None
        assert waited >= 1.5, f"必须等到对方释放（只等了 {waited:.1f}s）"
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ── 4) 锁文件路径随 PAPER_DIR 走（测试可隔离，不碰真实账本）─────────
def test_lock_path_follows_paper_dir(pp, tmp_path):
    assert pp.lock_path() == str(tmp_path / ".paper_portfolio.lock")
