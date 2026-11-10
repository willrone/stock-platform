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

  # ── 刷新袖套 ETF 行情（2026-11-05 新增，盯市的前置条件）─────────────
  # 为什么必须在这里刷：ETF 管道**没有自己的排期**，只有周一 17:40 调仓时
  # 被 `paper_portfolio` 顺带刷一次 → 周二~周五袖套腿只能拿旧收盘价估值，
  # 实测 2026-11-04：ETF 停在 11-02 → 袖套 11-03/11-04 两个点被估成**持平**
  # （假的 0 收益点 → 压低波动、稀释夏普，正是方案书 §0 要避免的那种错）。
  # 放在「锁已空闲」之后：此时不会有 paper_portfolio 在写 ETF parquet。
  # 刷新失败**不阻断盯市**：袖套的新鲜度门禁（c）会把滞后的点标红并让退出码变非 0。
  ETF_CODES=$(env -u PYTHONPATH -u PYTHONHOME "$PY" - <<'PYEOF'
import os, sys
sys.path.insert(0, os.path.join("backend"))
from app.services.trading import multi_asset_service as mas
print(",".join(c for lst in mas.SLEEVE_HELD.values() for c in lst))
PYEOF
)
  if [ -n "$ETF_CODES" ]; then
    echo "[$(date '+%F %T')] 刷新袖套 ETF 行情：$ETF_CODES"
    if env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
         backend/scripts/etf_data_pipeline.py --codes "$ETF_CODES" --incremental; then
      echo "[$(date '+%F %T')] ETF 刷新完成"
    else
      echo "[$(date '+%F %T')] ⚠ ETF 刷新失败 → 继续盯市，但袖套新鲜度门禁会报滞后"
    fi
  else
    echo "[$(date '+%F %T')] ⚠ 取不到袖套 ETF 代码清单（SLEEVE_HELD 为空？）"
  fi

  # ── 零技巧基准电平（2026-11-05 建设项 B）──────────────────────────────
  # 必须在盯市之前跑：`paper_daily_nav.py` 要在净值点里写
  # benchmark_return_pct / excess_pp，没有基准产物它的门禁 d 会直接判失败。
  # 口径 = 全池等权买入持有（A）/ 9 只定稿 ETF 等权买入持有（袖套）/ 两者各 50%（母线）。
  echo "[$(date '+%F %T')] 构建零技巧基准电平（全池等权买入持有 + 9 ETF 等权）"
  if env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
       backend/scripts/build_benchmark_index.py --start 2026-01-01 --incremental; then
    echo "[$(date '+%F %T')] 基准电平构建完成"
  else
    echo "[$(date '+%F %T')] ✗ 基准构建失败 → 盯市门禁 d 会判失败（不许没有基准就谈超额）"
  fi

  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/paper_daily_nav.py
  rc=$?
  echo "[$(date '+%F %T')] paper_daily_nav exit=$rc"
  exit $rc
} >> "$LOG" 2>&1
