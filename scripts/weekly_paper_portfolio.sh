#!/bin/bash
# 纸面盘信号线：**周一 17:40** 跑一次（launchd com.rone.paper-portfolio），
# 周二 09:20 兜底补跑。
#
# ⚠️ 时点是**数据落地时间**决定的，不是拍的：
#   TDX 当日日线包实测 17:30 左右落地 → 17:40 跑才有当天的收盘价；
#   纸面盘是严格 lag1（周五信号 → 下一交易日成交），没有成交日数据就拒绝成交。
#   周一没跑成时（TDX 包偶尔晚到）周二早上补一次；按成交日判重，不会重复下单。
#
# ⚠️ 退出码纪律：`python ... | grep -v INFO` 会让 $? 变成 grep 的退出码，
# 脚本崩了也显示 0。必须先重定向取真实 RC，再 grep 生成 .log。
#
# ⚠️ locale 陷阱（2026-11-01 修复）：bash 在 **UTF-8 locale** 下会把紧跟变量名的
#    **非 ASCII 字符算进变量名** —— `"RC=$RC，日志"` 里的变量名其实是 "RC，"，
#    于是报 `RC: unbound variable` 并**当场退出**（launchd 走 C locale 所以排期跑
#    一直没事，手工在终端跑必炸；ETF 刷新失败那行更致命：它跑在纸面盘之前，
#    一炸就整条线跳过）。规矩：**变量后面紧跟中文/全角标点，一律写 ${VAR}**。
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
  echo "⚠️ ETF 数据刷新失败（RC=${ETFRC}）—— 袖套腿将用旧数据运行" | tee -a "$LOG"
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

# ── 第 3 步：信号质量统计（2026-11-02 从 daily_tdx_update.sh 搬过来）──
# 为什么搬：它统计「每个历史买入信号发出后 T+1/T+5/T+10/T+20 的超额收益与
# 胜率」，读的是**纸面盘 ledger 里的成交记录** → 必须**排在纸面盘之后**跑，
# 才统计得到当次成交。原来它跟纸面盘写在同一个 if 块里（周一），
# 现在纸面盘由本脚本负责，它也归本脚本 —— 单一责任源。
echo "=== signal quality ==="
env -u PYTHONPATH -u PYTHONHOME "$PY" scripts/signal_quality.py >>"$RAW" 2>&1 || \
  echo "⚠️ signal_quality 失败（不影响主流程）"
echo "=== signal quality done ==="

# 保留最近 8 次日志，避免 runtime/logs 无限增长
ls -1t "$LOG_DIR"/paper_portfolio_*.log 2>/dev/null | tail -n +9 | while read -r f; do
  rm -f "$f" "$f.raw"
done

if [ "$RC" -ne 0 ]; then
  echo "❌ paper_portfolio 失败 RC=${RC}，日志：$LOG" | tee -a "$LOG"
  exit "$RC"
fi

echo "✅ paper_portfolio 完成，日志：$LOG"
exit 0
