"""
Stock Prediction Platform Backend

AI-powered stock prediction platform with task management,
built with FastAPI, Qlib, and modern ML frameworks.
"""

import os

# ⚠️⚠️ 必须在任何 ML 库（lightgbm / xgboost / sklearn 的 OpenMP 后端）被加载
# 之前锁定 OpenMP 线程数。多线程 OpenMP 在本机（macOS + Python 3.13 + libomp
# + qlib provider 的 fork/线程）会让 lightgbm 的 C 层**原生崩溃**：
#   OMP_NUM_THREADS=1 → 正常；=2 → Abort trap(134)；=4 → SIGSEGV(139)
# 症状：训练照常完成，但评估阶段 model.predict() 里进程直接死，
# Python 层 catch 不到 → API 服务进程被杀、pytest 整轮回归结果全丢。
# 这里放在 app 包初始化处，保证任何 `import app.*` 的入口都先设好。
# §12.44
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
__author__ = "Stock Prediction Team"
