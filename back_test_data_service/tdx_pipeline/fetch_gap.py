#!/usr/bin/env python3
"""补齐 2026-05-09 → 2026-10-16 的全市场日线（105 个交易日）。

用法:
    cd ~/Projects/stock-platform
    env -u PYTHONPATH -u PYTHONHOME backend/.venv/bin/python3 \
        back_test_data_service/tdx_pipeline/fetch_gap.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from back_test_data_service.tdx_pipeline import run_incremental, loaded_dates

if __name__ == "__main__":
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"拉取前已有分区: {len(loaded_dates())} 天")
    print("=" * 60)

    # 2026-05-09(周六) 起，run_incremental 会自动跳过非交易日
    stats = run_incremental(
        start="2026-05-09",
        end="2026-10-16",
        sleep_between=0.4,
        verbose=True,
    )

    print("=" * 60)
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最终统计: {stats}")
    dates = loaded_dates()
    print(f"总分区数: {len(dates)}")
    if dates:
        print(f"日期范围: {dates[0]} → {dates[-1]}")
