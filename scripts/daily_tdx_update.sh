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

  # 5) 纸面盘 + 信号质量统计 —— **已移出本脚本**（2026-11-02）
  #    原来这里在周一跑 paper_portfolio.py + signal_quality.py，但本脚本被
  #    **两个排期**调用：
  #      · launchd com.rone.tdx-update  16:05（周一~周五）
  #      · Hermes cron tdx-daily-update 17:30（同一脚本）
  #    两个问题：
  #      ① 16:05 那次**注定失败**：TDX 当日日线包实测 17:30 才落地
  #         → 16:05 时数据还停在上一交易日 → lag1 口径下 7 条线全部拒绝成交
  #         → 脚本非零退出（launchctl 里长期显示 exit 3，看着像故障）。
  #      ② **并发写账本**：2026-11-02 实测 17:30 那次跑到 17:44:55，
  #         而 17:40 的专用排期 17:40:04 就启动了 → 两个进程同时写
  #         holdings/ledger/portfolio.json → 后写者赢 → 成交明细可能被覆盖。
  #    → 现在：纸面盘**只由专用排期** com.rone.paper-portfolio 负责
  #      （周一 17:40 + 周二 09:20 兜底），signal_quality 跟在它后面跑；
  #      本脚本**只管拉数据**。
  #    → 并发问题另由 paper_portfolio.py 里的进程互斥锁兜底。

  echo "=== $(date '+%Y-%m-%d %H:%M:%S') all done ==="
} >> "$LOG" 2>&1
