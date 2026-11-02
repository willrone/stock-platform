#!/bin/bash
# 每个交易日由 launchd(16:05) + Hermes cron(17:30) 调用：拉通达信盘后包 → 合并进 stock_data；
# 同步精确涨跌幅；校验数据新鲜度。
#
# 2026-11-02 修订（三个实测缺陷）：
#  ① **16:05 那次注定失败**：TDX 当日日线包实测 ~17:30 才落地 → 16:05 时数据还停在
#     上一交易日 → 旧版脚本直接失败，launchctl 长期显示 exit 3，看着像故障。
#     → 现在：<17:00 只跑一次、不重试（快速失败不算故障）；>=17:00 最多重试 3 次
#     （17:30 / 18:00 / 18:30，间隔 30 分钟）直到数据落地。
#  ② **sync_pct_chg 失败被吞**：旧版 `|| echo "不影响主流程"` → 后复权价矩阵缺当天
#     → 纸面盘 adj_cost 写成 NaN → **市值被冻结在买入金额**（台账 §12.35）。
#     → 现在：sync 失败 = 本次运行失败（退出非零），并触发重试。
#     （TDX 的包到了以后 sync 一定能成功；它失败只可能是包还没到。）
#  ③ **并发写**：launchd(16:05) 与 cron(17:30) 调同一个脚本 → 两份实例可能同时跑。
#     → 现在：mkdir 原子锁（mkdir 在 POSIX 下是原子的），拿不到锁就直接退出。
set -uo pipefail

cd /Users/rone/Projects/stock-platform
mkdir -p runtime/logs
LOG="runtime/logs/tdx_update_$(date +%Y%m%d).log"
LOCKDIR="runtime/.tdx_update.lock"

# ── 并发锁 ────────────────────────────────────────────────────────────
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  # 已有实例：若是 3 小时前的僵尸锁则清理，否则跳过
  if [ -f "$LOCKDIR/pid" ]; then
    old_pid=$(cat "$LOCKDIR/pid" 2>/dev/null || echo 0)
    if ! kill -0 "$old_pid" 2>/dev/null; then
      echo "[$(date '+%F %T')] 清理僵尸锁(pid=$old_pid 已退出)" >> "$LOG"
      rm -rf "$LOCKDIR"
      mkdir "$LOCKDIR" 2>/dev/null || { echo "锁争用，跳过" >> "$LOG"; exit 0; }
    else
      echo "[$(date '+%F %T')] 已有实例在跑(pid=$old_pid)，跳过" >> "$LOG"
      exit 0
    fi
  else
    echo "[$(date '+%F %T')] 锁存在但无 pid，跳过" >> "$LOG"
    exit 0
  fi
fi
echo $$ > "$LOCKDIR/pid"
cleanup() { rm -rf "$LOCKDIR"; }
trap cleanup EXIT INT TERM

# ── 重试次数：>=17:00 才重试（16:05 那次注定拿不到当天包）──────────────
HOUR=$(date +%H)
MAX_ATTEMPT=1
if [ "$HOUR" -ge 17 ]; then MAX_ATTEMPT=3; fi

run_once() {
  echo "[$(date '+%F %T')] === attempt $1/$MAX_ATTEMPT ==="

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

  # 3) 同步精确涨跌幅（2026-10-26 新增）
  #    TDX raw 的 prev_close 是交易所口径前收盘（已除权除息调整），
  #    close/prev_close-1 = 精确涨跌幅。
  #    用途：
  #      - 补上 baostock 拉不到的北交所（354 只）
  #      - 每日给已有股票补当日行（比重拉 5240 只快两个数量级）
  #      - **把 TDX 有、baostock(T+1) 还没有的当天整行补进来**（2026-11-02 修，
  #        原来 how="left" 会丢掉当天 → 后复权矩阵缺当天 → 纸面盘净值冻结）
  #    ⚠️ 失败 = 本次运行失败（不是"不影响主流程"）：后复权矩阵会缺当天。
  if ! env -u PYTHONPATH backend/.venv-py313/bin/python3 \
        backend/scripts/sync_pct_chg_from_tdx.py --since "$(date -v-7d +%Y%m%d)"; then
    echo "[$(date '+%F %T')] ✗ sync_pct_chg 失败（本次运行判失败）"
    return 1
  fi
  echo "[$(date '+%F %T')] ✓ sync pct_chg done"

  # 4) 数据新鲜度校验（2026-10-25 新增）
  #    交易日过了 17:30 还没拿到当天数据 = 真失败 → 非零退出码，cron 报警到微信
  #    （原来 16:05 拉不到当天包被静默跳过，导致周频调仓两周没跑）
  if ! env -u PYTHONPATH backend/.venv-py313/bin/python3 \
        backend/scripts/check_data_freshness.py --fail-if-stale; then
    echo "[$(date '+%F %T')] ✗ 数据新鲜度校验失败（本次运行判失败）"
    return 1
  fi
  echo "[$(date '+%F %T')] ✓ 数据新鲜度 OK"
  return 0
}

{
  echo "=== $(date '+%F %T') start (max_attempt=$MAX_ATTEMPT) ==="

  ok=0
  for i in $(seq 1 "$MAX_ATTEMPT"); do
    if run_once "$i"; then ok=1; break; fi
    if [ "$i" -lt "$MAX_ATTEMPT" ]; then
      echo "[$(date '+%F %T')] 30 分钟后重试（TDX 包通常 17:30 前后落地）"
      sleep 1800
    fi
  done

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

  if [ "$ok" -eq 1 ]; then
    echo "=== $(date '+%F %T') all done ==="
    exit 0
  else
    echo "=== $(date '+%F %T') FAILED after $MAX_ATTEMPT attempt(s) ==="
    exit 1
  fi
} >> "$LOG" 2>&1
