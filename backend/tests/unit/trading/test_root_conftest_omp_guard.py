"""pytest 根级 conftest 的 **OpenMP 锁**门禁（§12.50）。

## 为什么这是门禁而不只是"修好了"

本机 macOS + Python 3.13 + libomp + qlib provider：

| OMP_NUM_THREADS | 结果 |
|---|---|
| 1 | 正常（EXIT=0） |
| 2 | Abort trap（134） |
| 4 | **Segmentation fault（139）** |

崩溃栈全在 `libomp.dylib`。症状极具迷惑性：进度条到 100%，收尾阶段崩，
**pytest 的最终汇总行打不出来** → 只看进度条会误判成"通过"。

`app/__init__.py` 里有 `os.environ.setdefault("OMP_NUM_THREADS", "1")`，但那是
**import app 时**才执行；pytest 收集阶段若先 import 了 qlib/lightgbm，就来不及了。
所以必须由 **rootdir 级 conftest.py** 在 collection 开始时抢在所有 ML 库之前锁死。

**门禁守什么**：`backend/conftest.py` 必须存在，且设置这三个变量；
且这些设置必须真的生效（子进程里读得到）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[3]
ROOT_CONFTEST = BACKEND_ROOT / "conftest.py"

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class TestRootConftestExists:

    def test_root_conftest_present(self):
        assert ROOT_CONFTEST.exists(), (
            f"{ROOT_CONFTEST} 不存在 —— pytest 收集阶段无法在最前面锁 OpenMP，"
            "全量回归会在收尾 SIGSEGV 且打不出汇总行（§12.50）"
        )

    def test_sets_all_thread_vars(self):
        src = ROOT_CONFTEST.read_text(encoding="utf-8")
        for var in THREAD_VARS:
            assert var in src, f"conftest.py 没设置 {var}"

    def test_uses_setdefault_not_hard_set(self):
        """用 setdefault：外部显式指定时不要覆盖（但调用方一般不指定）。"""
        src = ROOT_CONFTEST.read_text(encoding="utf-8")
        assert "setdefault" in src


class TestThreadVarsEffectiveInPytest:

    def test_env_is_locked_inside_pytest_run(self):
        """本测试进程里就必须是 1（证明 conftest 真的在 collection 前执行了）。"""
        import os as _os

        for var in THREAD_VARS:
            assert _os.environ.get(var) == "1", (
                f"{var} = {_os.environ.get(var)!r}，期望 '1' —— "
                "conftest.py 可能没被加载，或放的位置太晚"
            )

    def test_fresh_python_process_inherits_the_lock(self):
        """干净子进程里也应该是 1（证明是环境层面生效，不只是本进程变量）。"""
        code = "import os; print(os.environ.get('OMP_NUM_THREADS',''))"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=60, env={**os.environ})
        assert r.stdout.strip() == "1", f"子进程 OMP_NUM_THREADS={r.stdout.strip()!r}"
