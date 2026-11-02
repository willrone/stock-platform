"""pytest **根级** conftest —— 必须在任何测试模块被 import 之前执行。

## 为什么必须在这里锁 OpenMP

§12.36 定死的事实（本机 macOS + Python 3.13 + libomp + qlib provider）：

| OMP_NUM_THREADS | 结果 |
|---|---|
| 1 | 正常（EXIT=0） |
| 2 | Abort trap（134） |
| 4 | **Segmentation fault（139）** |

崩溃栈在 `libomp.dylib`（LLVM OpenMP 运行时）里，是**进程内 OpenMP 运行时状态**问题
（模型与数据都没问题：dump 到文件 → 干净进程 load + predict 完全正常）。

## 原来的洞

`app/__init__.py:18` 和 `app/services/qlib/official_lgbm_model.py` 顶层都有
`os.environ.setdefault("OMP_NUM_THREADS", "1")` —— 但那是**import app 时**才执行。
pytest 收集阶段如果**先** import 了 qlib / lightgbm / numpy（某些测试模块顶层就 import），
`app` 还没被 import → setdefault 来得太晚 → OpenMP 多线程已初始化 → 段错误。

**真实的崩溃形态**：进程 SIGSEGV / SIGABRT → **exit code 139 / 134**，
并在 `~/Library/Logs/DiagnosticReports/Python-*.ips` 里留下 `libomp.dylib` 崩溃报告。

2026-11-03 实测：修复前多次全量回归在收尾阶段崩溃（`.ips` 报告时间戳全部早于 05:13 的修复，
05:13 之后再无新崩溃报告）。加了 `OMP_NUM_THREADS=1` 后，同一份代码 exit code 0。

**判据（本项目只认这三条）**：exit code + `-v` 的最终计数行 + `.ips` 崩溃报告。

> 注意：`pytest -q` 在 **pytest 9.0.3** 下**本来就不打印最终计数行**（新 reporter 行为），
> 这跟崩不崩无关。排查时务必用 `-v` 跑一个小集合交叉验证，别拿 `-q` 的输出当通过/失败判据。

## 这里的做法

根级 conftest 在 **collection 开始时**加载，早于任何 test module 的 import
→ 抢在所有 ML 库之前把线程数锁成 1。**调用方不再需要记得带环境变量。**

生产参数本来就是 `num_threads=1`（§12.36），所以**无性能损失**。
将来若要提速，须先解决 libomp 与 fork 的兼容问题，不能直接调大线程数。
"""

from __future__ import annotations

import os

# 必须在任何 ML 库（numpy / lightgbm / qlib）被 import 之前生效
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
