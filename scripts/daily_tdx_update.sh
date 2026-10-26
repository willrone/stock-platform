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

  # 3) 同步精确涨跌幅（2026-10-26 新增）
  #    TDX raw 的 prev_close 是交易所口径前收盘（已除权除息调整），
  #    close/prev_close-1 = 精确涨跌幅。
  #    用途：
  #      - 补上 baostock 拉不到的北交所（354 只）
  #      - 每日给已有股票补当日行（比重拉 5240 只快两个数量级）
  echo "=== sync pct_chg from TDX ==="
  env -u PYTHONPATH backend/.venv-py313/bin/python3 \
    backend/scripts/sync_pct_chg_from_tdx.py --since "$(date -v-7d +%Y%m%d)" || \
    echo "sync_pct_chg 失败（不影响主流程）"
  echo "=== sync pct_chg done ==="

  # 4) 数据新鲜度校验（2026-10-25 新增）
  #    交易日过了 17:30 还没拿到当天数据 = 真失败 → 非零退出码，cron 报警到微信
  #    （原来 16:05 拉不到当天包被静默跳过，导致周频调仓两周没跑）
  env -u PYTHONPATH backend/.venv-py313/bin/python3 \
    backend/scripts/check_data_freshness.py --fail-if-stale

  # 5) 周五收盘后自动跑多策略纸面盘（各线独立 100 万记账）
  #    22f / honest_42f / rsi_71421 / factor_ranking
  #    判「周五」用真实日期，不用数据最新日
  if [ "$(date +%u)" = "5" ]; then
    echo "=== Friday: multi-strategy paper portfolio ==="
    env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/paper_portfolio.py
    echo "=== paper portfolio done ==="

    # 6) 信号质量统计（2026-10-25 新增）
    #    回答「这个信号到底准不准」：每个历史买入信号发出后
    #    T+1/T+5/T+10/T+20 的超额收益与胜率。
    #    同时把后复权价格与基准算好，供信号服务使用。
    echo "=== signal quality ==="
    env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/signal_quality.py || \
      echo "signal_quality 失败（不影响主流程）"
    echo "=== signal quality done ==="
  fi

  echo "=== $(date '+%Y-%m-%d %H:%M:%S') all done ==="
} >> "$LOG" 2>&1
