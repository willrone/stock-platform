"""3 年对照: 完全相同的 10 年管线(4329 宇宙+stock_data+时段化清洗), 窗口缩到 3 年.
目的: 隔离 10 年 R1 1.24→0.57 的下降是 regime 效应还是数据源/宇宙切换假象.
  若 3年/4329/stock_data 的 R1 ≈ 3年/5513/qlib 的 1.24 → 管线一致, 10 年下降=真实 regime.
  若 3年/4329/stock_data 的 R1 << 1.24 → 数据源/宇宙切换本身解释了下降."""
from __future__ import annotations
import asyncio
import importlib
import importlib.util
import os
import sys
import time
from datetime import datetime

import numpy as np

BACKEND_ROOT = os.path.abspath('/Users/rone/projects/stock-platform/backend')
sys.path.insert(0, BACKEND_ROOT)
_spec = importlib.util.spec_from_file_location(
    "rp10", os.path.join(BACKEND_ROOT, "scripts", "risk_parity_10y_experiment.py"))
rp10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp10)

# 复用 10 年管线, 仅改窗口到 3 年
rp10.START = datetime(2023, 5, 12)
rp10.END = datetime(2026, 5, 8)
# 3 年对照的宇宙: 用 3 年窗口中点(2024-05)的 point-in-time, 保持同方法
rp10.MIDPOINT = datetime(2024, 5, 9)
rp10.OUT_JSON = "/Users/rone/projects/stock-platform/data/strategy_baseline/risk_parity_3y_control.json"

if __name__ == "__main__":
    rp10.main()
