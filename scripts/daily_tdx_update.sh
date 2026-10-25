#!/bin/bash
# 每个交易日 16:05 由 launchd 调用：拉通达信盘后包 → 合并进 stock_data；
# 周五额外跑 22f 周频纸面信号，更新 data/signals/ 与 frontend/public/paper-signal.json。
set -euo pipefail

cd /Users/rone/Projects/stock-platform
mkdir -p runtime/logs
LOG="runtime/logs/tdx_update_$(date +%Y%m%d).log"

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') start ==="

  # 1) 拉当日增量（已拉过的日期会自动跳过）
  env -u PYTHONPATH backend/.venv-py313/bin/python3 -c "
import sys
sys.path.insert(0, '.')
from back_test_data_service.tdx_pipeline import run_incremental
print(run_incremental())
"

  # 2) 合并到 data/parquet/stock_data（--end 默认今天，防止硬编码漏日期）
  env -u PYTHONPATH backend/.venv-py313/bin/python3 \
    back_test_data_service/tdx_pipeline/merge_to_stock_data.py --end "$(date +%F)"

  echo "=== $(date '+%Y-%m-%d %H:%M:%S') data done ==="

  # 3) 数据新鲜度校验（2026-10-25 新增）
  #    交易日过了 17:30 还没拿到当天数据 = 真失败 → 非零退出码，cron 报警到微信
  #    （原来 16:05 拉不到当天包被静默跳过，导致周频调仓两周没跑）
  env -u PYTHONPATH backend/.venv-py313/bin/python3 \
    backend/scripts/check_data_freshness.py --fail-if-stale

  # 4) 周五收盘后自动跑多策略纸面盘（4 条线各自 100 万独立记账）
  #    22f / honest_42f / rsi_71421 / factor_ranking
  #    判「周五」用真实日期，不用数据最新日
  if [ "$(date +%u)" = "5" ]; then
    echo "=== Friday: multi-strategy paper portfolio ==="
    env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/paper_portfolio.py
    echo "=== paper portfolio done ==="
  fi

  echo "=== $(date '+%Y-%m-%d %H:%M:%S') all done ==="
} >> "$LOG" 2>&1
