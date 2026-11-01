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

if [ "${1:-}" = "--verify" ]; then
  "${PY}" "${ROOT}/backend/scripts/verify_artifact_backup.py" --backup-root "${DEST}"
  exit $?
fi

echo "=== 备份研究产物 → ${DEST} ==="
mkdir -p "${DEST}/data"
# 整棵 data/ 树，排除两类：
#   (a) 活状态目录 data/paper、data/signals、data/tdx_daily —— 运行时状态，
#       清单本来就不给它们指纹（LIVE_STATE_PREFIXES），备份只会每次都不一致；
#   (b) data/logs —— 367 MB 运行日志，无证据价值。
# 保留 qlib_data(28G)/parquet(1.5G)：它们可由 build_full_history_features.py 重建，
# 但重建依赖上游下载源（qlib_official_data + parquet），盘坏后重拉很慢 → 一起备份。
# --delete 保证副本与源一致（不残留旧版本产物）。
rsync -a --delete \
  --exclude 'paper/' --exclude 'signals/' --exclude 'tdx_daily/' --exclude 'logs/' \
  "${ROOT}/data/" "${DEST}/data/"
rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "❌ rsync 失败（rc=${rc}）→ 不进入校验"
  exit "${rc}"
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
echo "✅ 备份完成并通过指纹校验"
exit 0
