#!/bin/bash
# 研究产物备份 + 指纹校验（2026-11-02）
#
# 为什么需要这个脚本（不是一次性拷贝）：
#   1. data/ 被 .gitignore（.gitignore:296）→ 指纹清单 110 个产物里 108 个**不在 git**
#      （只有 data/industry/industry_map.json、data/stocks/st_names.json 被 ! 反选跟踪）
#   2. 本机 Time Machine 失效：tmutil latestbackup → "Failed to mount destination"
#      → 系统级备份当前不提供任何保护
#   3. data/backtest = 309 文件 / 2.0 GB，是 P1~P18 全部结论的**唯一原始证据**
#   → 结论：一块内置盘 = 单点。本脚本是第二副本 + 可重复执行 + 自带逐字节校验。
#
# 目标卷：外接 SSD /Volumes/DATA（未挂载则**明确报错退出**，不做半吊子备份）
#
# 用法:
#   bash scripts/backup_research_artifacts.sh            # 备份 + 校验
#   bash scripts/backup_research_artifacts.sh --verify    # 只校验现有副本
#
# 注意（bash locale 陷阱，2026-11-01 踩过）：变量后紧跟中文会被 UTF-8 locale
# 算进变量名 → 一律写 ${VAR}。
set -uo pipefail

ROOT="/Users/rone/Projects/stock-platform"
VOL="/Volumes/DATA"
DEST="${VOL}/stock-platform-backup"
PY="${ROOT}/backend/.venv-py313/bin/python3"

if [ ! -d "${VOL}" ]; then
  echo "❌ 外接盘 ${VOL} 未挂载 → 拒绝备份（研究产物目前无副本，请插盘后重试）"
  exit 2
fi

# ── 只校验模式（不拿锁：备份正在跑时校验仍应如实报告）────────────────
if [ "${1:-}" = "--verify" ]; then
  "${PY}" "${ROOT}/backend/scripts/verify_artifact_backup.py" --backup-root "${DEST}"
  exit $?
fi

# ── 并发锁 ────────────────────────────────────────────────────────────
# 两个排期器可能同时触发（launchd + Hermes cron）→ 两个 rsync 写同一副本
# 会打架。用 mkdir 做原子锁：拿不到锁就**跳过**（不是排队，避免积压的备份风暴）。
# 残留锁（进程被强杀）超过 6 小时视为失效 → 清理后重试一次。
# 注意：started_at 读不出来时**不**判定为残留（宁可多跳过一次，也不能误删活锁）。
LOCK_DIR="${ROOT}/runtime/backup.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  stale=1
  if [ -f "${LOCK_DIR}/started_at" ]; then
    started="$(cat "${LOCK_DIR}/started_at" 2>/dev/null || true)"
    if [ -n "${started}" ]; then
      age=$(( $(date +%s) - started ))
      if [ "${age}" -gt 21600 ]; then stale=0; fi
    fi
  fi
  if [ "${stale}" -eq 0 ]; then
    rm -rf "${LOCK_DIR}"
    mkdir "${LOCK_DIR}" 2>/dev/null || { echo "❌ 清理残留锁后仍拿不到锁 → 退出"; exit 5; }
    echo "⚠️ 清理了残留锁（>6h）→ 继续"
  else
    echo "⚠️ 已有备份实例在运行（锁 ${LOCK_DIR}）→ 本次跳过"
    exit 0
  fi
fi
date +%s > "${LOCK_DIR}/started_at"
trap 'rm -rf "${LOCK_DIR}"' EXIT

echo "=== 备份研究产物 → ${DEST} ==="
mkdir -p "${DEST}/data"
# 整棵 data/ 树，排除两类：
#   (a) 活状态目录 data/paper、data/signals、data/tdx_daily —— 运行时状态，
#       清单本来就不给它们指纹（LIVE_STATE_PREFIXES），备份只会每次都不一致；
#   (b) data/logs —— 367 MB 运行日志，无证据价值。
# 保留 qlib_data(28G)/parquet(1.5G)：它们可由 build_full_history_features.py 重建，
# 但重建依赖上游下载源（qlib_official_data + parquet），盘坏后重拉很慢 → 一起备份。
# --delete 保证副本与源一致（不残留旧版本产物）。
# --max-delete=50 是**护栏**：openrsync 的 --max-delete 不会报错退出（只打印
# "Deletions stopped due to --max-delete limit" 并 rc=0）→ 光加这个参数等于
# "静默截断删除还报成功"的假成功。所以：捕获 rsync 输出里的这个信号，
# 命中即视为**备份失败**（即便指纹校验通过也失败），让"源树被误清空"这种
# 灾难不会悄悄把唯一副本一起清空。
rsync_log="${ROOT}/runtime/logs/backup_research_artifacts.rsync.log"
mkdir -p "$(dirname "${rsync_log}")"
rsync -a --delete --max-delete=50 \
  --exclude 'paper/' --exclude 'signals/' --exclude 'tdx_daily/' --exclude 'logs/' \
  "${ROOT}/data/" "${DEST}/data/" > "${rsync_log}" 2>&1
rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "❌ rsync 失败（rc=${rc}）→ 不进入校验"
  tail -5 "${rsync_log}"
  exit "${rc}"
fi
delete_capped=0
if grep -q "max-delete limit" "${rsync_log}"; then
  delete_capped=1
  echo "❌ rsync 触到 --max-delete=50 上限（源→副本的删除量过大，详见 ${rsync_log}）"
  grep "max-delete limit" "${rsync_log}"
fi

# 证据链自描述文件（清单 + 台账），虽在 git 里，但副本要自包含
rsync -a "${ROOT}/docs/backtest/ARTIFACT_MANIFEST.json" "${DEST}/" 2>/dev/null
rsync -a "${ROOT}/docs/backtest/MULTI_ASSET_LEDGER_2026-10-31.md" "${DEST}/" 2>/dev/null

echo "--- 逐字节校验副本 ---"
"${PY}" "${ROOT}/backend/scripts/verify_artifact_backup.py" --backup-root "${DEST}"
rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "❌ 副本校验未通过 → 备份视为失败"
  exit "${rc}"
fi

du -sh "${DEST}" | sed 's/^/  副本体积: /'
if [ "${delete_capped}" -eq 1 ]; then
  echo "❌ 删除量超过护栏阈值 → 备份视为失败（副本可能只同步了一部分，下次成功跑会修复）"
  exit 4
fi
echo "✅ 备份完成并通过指纹校验"
exit 0
