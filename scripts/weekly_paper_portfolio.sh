#!/bin/bash
# 纸面盘信号线：每周五收盘后跑一次（tdx 数据更新 16:05 之后）。
#
# ⚠️ 退出码纪律：`python ... | grep -v INFO` 会让 $? 变成 grep 的退出码，
# 脚本崩了也显示 0。必须先重定向取真实 RC，再 grep 生成 .log。
set -uo pipefail

ROOT="/Users/rone/Projects/stock-platform"
BACKEND="$ROOT/backend"
LOG_DIR="$ROOT/runtime/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RAW="$LOG_DIR/paper_portfolio_$STAMP.raw"
LOG="$LOG_DIR/paper_portfolio_$STAMP.log"

cd "$BACKEND" || exit 90

PY=".venv-py313/bin/python3"
if [ ! -x "$PY" ]; then
  echo "❌ 找不到 $BACKEND/$PY" | tee "$LOG"
  exit 91
fi

echo "=== paper_portfolio 开始 $(date '+%F %T') ===" | tee "$LOG"

# ── 第 1 步：刷新 ETF 行情（袖套腿的数据底座）────────────────────────
# ⚠️ 必须**全量重拉**，不能加 --incremental：
#   ETF 行情是**前复权**序列（前值随分红整体重锚）。增量模式只重拉最近 10 天
#   并覆盖合并 → 10 天窗口之前的历史段留着**旧锚点**，两段之间出现人为跳空
#   （债券 ETF 每月付息，锚点偏移 ≈ 0.17%/次）。
#   实测本地 9 只 ETF 在 2023-11-28 ~ 2023-12-04 有 6 天成段缺失，
#   说明这个管道确实会漏批 —— 全量重拉能一并修掉。
ETF_CODES="518880,159934,159937,511260,511010,511020,159980,159985,159981"
ETFRC=0
env -u PYTHONPATH -u PYTHONHOME "$PY" scripts/etf_data_pipeline.py \
  --codes "$ETF_CODES" --start 2019-01-01 --end "$(date +%F)" \
  >>"$RAW" 2>&1 || ETFRC=$?
if [ "$ETFRC" -ne 0 ]; then
  echo "⚠️ ETF 数据刷新失败（RC=$ETFRC）—— 袖套腿将用旧数据运行" | tee -a "$LOG"
fi

# ── 第 2 步：跑纸面盘（含多资产组合线）───────────────────────────────
env -u PYTHONPATH -u PYTHONHOME "$PY" scripts/paper_portfolio.py >"$RAW" 2>&1
RC=$?

grep -vE "INFO |WARNING " "$RAW" >"$LOG.tmp"
{
  echo "=== paper_portfolio 结束 $(date '+%F %T')  RC=$RC ==="
  echo "--- 非 INFO 输出 ---"
  cat "$LOG.tmp"
} >>"$LOG"
rm -f "$LOG.tmp"

# 保留最近 8 次日志，避免 runtime/logs 无限增长
ls -1t "$LOG_DIR"/paper_portfolio_*.log 2>/dev/null | tail -n +9 | while read -r f; do
  rm -f "$f" "$f.raw"
done

if [ "$RC" -ne 0 ]; then
  echo "❌ paper_portfolio 失败 RC=$RC，日志：$LOG" | tee -a "$LOG"
  exit "$RC"
fi

echo "✅ paper_portfolio 完成，日志：$LOG"
exit 0
