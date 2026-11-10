"""`fix_adj_cost_backfill.py` 的**互斥锁**门禁（§12.48 R2）。

## 背景

这个脚本会重写 `<strategy>_holdings.json` 与 `data/paper/portfolio.json`，
但它**不走 `paper_portfolio.main()`** → 拿不到那把进程互斥锁
（`acquire_run_lock`，见 `test_paper_run_lock.py`）。

排期层面本来已两重堵死（单一责任源 `com.rone.paper-portfolio` + fcntl 互斥锁），
但这是**唯一还能真并发写 holdings/portfolio.json 的路径**：
手工跑它、恰好撞上周一 17:40 那档 → 两个进程同时写同一批文件 → 后写者赢。

## 本文件守什么

1. 非 dry-run 时**必须取锁**；dry-run（不写盘）**不取锁**
2. **取锁发生在加载行情之前** —— 那 2~4 分钟里排期实例完全可能已经开跑
3. 别的实例持锁时 → 本脚本**等待**（而不是并发写）
4. 锁路径与纸面盘主流程**同一个**（PAPER_DIR 下）
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import fix_adj_cost_backfill as fab
from scripts import paper_portfolio as pp


# ══════════════════════════════════════════════════════════════
# 1. 取锁与否
# ══════════════════════════════════════════════════════════════

class TestLockAcquisition:
    """注意：脚本用 importlib 加载 paper_portfolio（模块名 paper_portfolio_mod），
    **和 `from scripts import paper_portfolio` 不是同一个模块对象**（各自一份模块级状态）。
    所以 monkeypatch 必须打在 `fab.pp` 上 —— 打错对象会让测试"通过但没测到东西"。"""

    def test_dry_run_does_not_take_lock(self, monkeypatch):
        called = []
        monkeypatch.setattr(fab.pp, "acquire_run_lock", lambda *a, **k: called.append(1))
        assert fab.acquire_lock_if_writing(True) is None
        assert not called, "dry-run 不写盘，不该取锁"

    def test_write_mode_takes_lock(self, monkeypatch):
        called = []

        def fake_lock(*a, **k):
            called.append(1)
            return 42

        monkeypatch.setattr(fab.pp, "acquire_run_lock", fake_lock)
        fd = fab.acquire_lock_if_writing(False)
        assert fd == 42
        assert called, "非 dry-run 必须取锁"

    def test_uses_same_lock_path_as_main_flow(self, tmp_path, monkeypatch):
        """锁文件必须和 paper_portfolio 主流程是同一个（否则等于没锁）。"""
        monkeypatch.setattr(fab.pp, "PAPER_DIR", str(tmp_path))
        monkeypatch.setattr(pp, "PAPER_DIR", str(tmp_path))
        assert fab.pp.lock_path() == pp.lock_path()


# ══════════════════════════════════════════════════════════════
# 2. 取锁发生在加载行情之前
# ══════════════════════════════════════════════════════════════

class TestLockOrdering:

    def test_lock_taken_before_market_data_load(self):
        """源码顺序门禁：取锁调用必须出现在 load_data 之前。

        跑 main() 要加载行情（2~4 分钟），不适合做行为测试；
        这里直接断言源码里的调用顺序 —— 顺序错了就是并发窗口。
        """
        src = Path(fab.__file__).read_text(encoding="utf-8")
        i_lock = src.find("acquire_lock_if_writing(args.dry_run)")
        i_load = src.find("pp.ds.load_data(codes)")
        assert i_lock != -1, "main() 里没有调用 acquire_lock_if_writing"
        assert i_load != -1, "没找到 load_data 调用（源码结构变了？）"
        assert i_lock < i_load, (
            f"取锁位置({i_lock}) 必须早于加载行情({i_load}) —— "
            "否则加载行情那几分钟里排期实例可能已经开跑"
        )

    def test_dry_run_skips_lock_entirely(self):
        """dry-run 路径不写盘 → 不该取锁（否则会白等 30 分钟）。"""
        src = Path(fab.__file__).read_text(encoding="utf-8")
        assert "if dry_run:" in src
        assert "return None" in src


# ══════════════════════════════════════════════════════════════
# 3. 真的会被互斥（跨进程）
# ══════════════════════════════════════════════════════════════

class TestCrossProcessMutualExclusion:

    def test_blocks_when_other_process_holds_lock(self, tmp_path, monkeypatch):
        """另一个进程持锁时，本脚本的取锁必须失败/等待，而不是拿到锁。"""
        monkeypatch.setattr(fab.pp, "PAPER_DIR", str(tmp_path))
        monkeypatch.setattr(pp, "PAPER_DIR", str(tmp_path))
        os.makedirs(tmp_path, exist_ok=True)
        fd = os.open(pp.lock_path(), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # 本测试进程独占
            # 子进程尝试取锁 → 必须拿不到
            # 子进程必须锁**同一个文件** → 显式把 PAPER_DIR 指到 tmp_path
            # （`from scripts import paper_portfolio` 拿到的是真实 PAPER_DIR）
            code = (
                "import sys; sys.path.insert(0, %r);"
                "from scripts import paper_portfolio as pp;"
                "import fcntl, os;"
                "pp.PAPER_DIR = %r;"
                "fd = os.open(pp.lock_path(), os.O_CREAT | os.O_RDWR, 0o644);"
                "\ntry:\n fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); print('GOT_LOCK')\n"
                "except OSError:\n print('BLOCKED')\n"
            ) % (str(Path(pp.__file__).resolve().parents[1]), str(tmp_path))
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=60)
            assert "GOT_LOCK" not in r.stdout, (
                "并发写没被挡住！子进程拿到了锁 —— 互斥失效"
            )
            assert "BLOCKED" in r.stdout, f"子进程输出异常: {r.stdout!r} {r.stderr[-300:]!r}"
        finally:
            os.close(fd)
