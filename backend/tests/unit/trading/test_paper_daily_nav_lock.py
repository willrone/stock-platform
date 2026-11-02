"""`paper_daily_nav.py` 的**互斥锁**门禁（§12.52）。

## 背景

`scripts/daily_nav.sh`（Hermes cron `0294eb16149f`，工作日 18:20）的执行链是：

```
shell 探测锁空闲  →  python scripts/paper_daily_nav.py
```

探测与执行之间存在 **TOCTOU 窗口**：探测刚通过，排期实例
（`com.rone.paper-portfolio` 周一 17:40）恰好抢到锁并开始改写
`holdings` / `ledger` → 本脚本在**半写状态**上算净值。

## 为什么"只读不写"也要取锁

本脚本**只写** `<strategy>_nav_daily.json`，**不碰账本** → 不会损坏账本。
但它**要读** `holdings` 与 `ledger`（`pp.load_holdings` / `read_ledger`）：

- 读到半成品 → 算出**错的净值点**
- 净值点一旦写进 `nav_daily.json`，就是"**看起来正常的错数据**"
  （有日期、有单调性、门禁还可能过）→ 比直接失败更难发现

所以必须取锁。**dry-run 也要取**（它照样读盘）——
这与 `fix_adj_cost_backfill` 不同：那个脚本 dry-run 时既不读也不写。

## 本文件守什么

1. `acquire_lock()` 必须真的调 `pp.acquire_run_lock()`
2. 超时是 **120 秒**（不是默认 1800）—— 外层 shell 已等过 40 分钟，双层等待冗余
3. 取锁发生在**加载行情之前**（那 2~4 分钟里排期实例完全可能已经开跑）
4. 锁路径与纸面盘主流程**同一个**（否则等于没锁）
5. 跨进程真互斥
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

from scripts import paper_daily_nav as pdn
from scripts import paper_portfolio as pp


# ══════════════════════════════════════════════════════════════
# 1. 取锁行为
# ══════════════════════════════════════════════════════════════

class TestLockAcquisition:
    """注意：`paper_daily_nav.py` 用 importlib 加载 paper_portfolio
    （模块名 `paper_portfolio_mod`），**和 `from scripts import paper_portfolio`
    不是同一个模块对象**（各自一份模块级状态）。
    所以 monkeypatch 必须打在 `pdn.pp` 上 —— 打错对象会让测试"通过但没测到东西"。"""

    def test_takes_lock(self, monkeypatch):
        called = {}

        def fake_lock(*a, **k):
            called["args"] = (a, k)
            return 42

        monkeypatch.setattr(pdn.pp, "acquire_run_lock", fake_lock)
        fd = pdn.acquire_lock()
        assert fd == 42
        assert called, "必须取锁"

    def test_uses_short_timeout(self, monkeypatch):
        """超时必须是 120 秒。

        默认 1800 秒 + 外层 shell 已经等 40 分钟 → 最坏 70 分钟挂起一个 cron。
        """
        seen = {}

        def fake_lock(*a, **k):
            seen.update(k)
            seen["positional"] = a
            return 1

        monkeypatch.setattr(pdn.pp, "acquire_run_lock", fake_lock)
        pdn.acquire_lock()
        assert seen.get("timeout_s") == 120, (
            f"超时应为 120s（与外层 shell 的等待不重复），实际 {seen!r}"
        )

    def test_dry_run_also_takes_lock(self, monkeypatch):
        """dry-run 照样读 holdings/ledger → 同样会读到半成品 → 必须取锁。

        这条与 `fix_adj_cost_backfill` 的"dry-run 不取锁"**故意不同**，
        改动时别照抄那一条。
        """
        called = []
        monkeypatch.setattr(pdn.pp, "acquire_run_lock",
                            lambda *a, **k: called.append(1) or 7)
        assert pdn.acquire_lock() == 7
        assert called, "dry-run 也读盘，必须取锁"

    def test_uses_same_lock_path_as_main_flow(self, tmp_path, monkeypatch):
        """锁文件必须和 paper_portfolio 主流程是同一个（否则等于没锁）。"""
        monkeypatch.setattr(pdn.pp, "PAPER_DIR", str(tmp_path))
        monkeypatch.setattr(pp, "PAPER_DIR", str(tmp_path))
        assert pdn.pp.lock_path() == pp.lock_path()


# ══════════════════════════════════════════════════════════════
# 2. 取锁发生在加载行情之前
# ══════════════════════════════════════════════════════════════

class TestLockOrdering:

    def test_lock_taken_before_market_data_load(self):
        """源码顺序门禁：取锁调用必须出现在 load_data 之前。

        跑 main() 要加载行情（2~4 分钟），不适合做行为测试；
        这里直接断言源码里的调用顺序 —— 顺序错了就是并发窗口。
        """
        src = Path(pdn.__file__).read_text(encoding="utf-8")
        i_lock = src.find("lock_fd = acquire_lock()")
        i_load = src.find("pp.ds.load_data(codes)")
        assert i_lock != -1, "main() 里没有调用 acquire_lock()"
        assert i_load != -1, "没找到 load_data 调用（源码结构变了？）"
        assert i_lock < i_load, (
            f"取锁位置({i_lock}) 必须早于加载行情({i_load}) —— "
            "否则加载行情那几分钟里排期实例可能已经开跑"
        )

    def test_lock_fd_kept_alive_until_exit(self):
        """fd 必须保持引用直到进程结束。

        fcntl.flock 随 fd 关闭/进程退出释放；若 fd 被 GC 掉，锁会**提前释放**，
        互斥等于没加。
        """
        src = Path(pdn.__file__).read_text(encoding="utf-8")
        assert "lock_fd = acquire_lock()" in src
        # fd 必须赋给一个函数作用域里的变量（不是匿名临时值）
        assert "lock_fd = acquire_lock()" in src


# ══════════════════════════════════════════════════════════════
# 3. 真的会被互斥（跨进程）
# ══════════════════════════════════════════════════════════════

class TestCrossProcessMutualExclusion:

    def test_blocks_when_other_process_holds_lock(self, tmp_path, monkeypatch):
        """另一个进程持锁时，本脚本的取锁必须失败，而不是拿到锁。"""
        monkeypatch.setattr(pdn.pp, "PAPER_DIR", str(tmp_path))
        monkeypatch.setattr(pp, "PAPER_DIR", str(tmp_path))
        os.makedirs(tmp_path, exist_ok=True)
        fd = os.open(pp.lock_path(), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # 本测试进程独占
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
