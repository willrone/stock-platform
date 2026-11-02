#!/bin/bash
# 纸面盘**日频盯市**排期入口（每个交易日 18:20，由 Hermes cron 调）。
#
# 为什么需要等锁：
#   纸面盘周频调仓（launchd com.rone.paper-portfolio，周一 17:40）
#   与数据更新（17:30 起，最多重试到 18:30）都可能还在跑
#   → 本脚本必须等它们跑完，否则会在半更新的数据上算净值。
#
# ⚠️ 判锁方式：必须用 **flock 探测**，不能看锁文件是否存在。
#   paper_portfolio.py 用的是内核级 fcntl.flock（进程退出自动释放），
#   而锁文件里的 "pid=... started=..." 只是**信息**（进程被 kill -9 后文件仍在）。
#   → 早先版本按"文件存在"判断 → 遇到僵尸锁文件会白等 40 分钟。
set -uo pipefail

cd /Users/rone/Projects/stock-platform
mkdir -p runtime/logs
LOG="runtime/logs/daily_nav_$(date +%Y%m%d).log"

WAIT_MIN=40
LOCK="data/paper/.paper_portfolio.lock"
PY="backend/.venv-py313/bin/python3"

{
  echo "=== $(date '+%F %T') daily_nav start ==="

  waited=0
  while [ "$waited" -lt 80 ]; do    # 最多等 40 分钟（每次 30 秒）
    if env -u PYTHONPATH -u PYTHONHOME "$PY" - "$LOCK" <<'PYEOF'
import fcntl, os, sys
p = sys.argv[1]
os.makedirs(os.path.dirname(p), exist_ok=True)
fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)          # 拿不到 → 有人在跑
else:
    fcntl.flock(fd, fcntl.LOCK_UN)
    sys.exit(0)          # 拿得到 → 无人跑
PYEOF
    then
      echo "[$(date '+%F %T')] 纸面盘锁空闲（已等 ${waited}0 秒）"
      break
    else
      echo "[$(date '+%F %T')] 纸面盘仍在跑，等待中（${waited}0 秒）"
      sleep 30
      waited=$((waited + 1))
    fi
  done

  if [ "$waited" -ge 80 ]; then
    echo "[$(date '+%F %T')] ✗ 等待超时（40 分钟），本次跳过（次日补齐序列）"
    exit 0
  fi

  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/paper_daily_nav.py
  rc=$?
  echo "[$(date '+%F %T')] paper_daily_nav exit=$rc"
  exit $rc
} >> "$LOG" 2>&1
