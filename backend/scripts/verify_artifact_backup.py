"""校验研究产物备份副本与指纹清单一致（2026-11-02）。

用法:
    python verify_artifact_backup.py --backup-root /Volumes/DATA/stock-platform-backup

为什么不用原始字节哈希：清单的指纹算法是 **normalized-json-v2**（剔除易变字段
wall_s/total_wall_s/... → 规范化 JSON → sha256），见 artifact_manifest.SHA256_MODE。
原始字节哈希与清单**必然不等**（首次实测：110 个里 106 个"不符"，是本脚本的
算法错，不是副本错）。因此这里**直接 import 生成器的 sha256**，让两端共用同一
份实现 → 算法永不漂移。

背景: data/ 被 .gitignore（2.0 GB / 309 文件），指纹清单里 110 个产物有 108 个
**不在 git**；本机 Time Machine 失效（Failed to mount destination）→ 研究证据一度
是单点（只有一块内置盘）。本脚本证明备份副本与清单登记的指纹一致。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import artifact_manifest as am  # noqa: E402  单一权威哈希实现


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-root", required=True)
    args = parser.parse_args()

    with open(am.MANIFEST, encoding="utf-8") as fh:
        manifest = json.load(fh)
    files: Dict[str, Dict] = manifest.get("files", {})
    if not files:
        print("❌ 清单为空")
        return 1
    mode = manifest.get("sha256_mode", "?")
    if mode != am.SHA256_MODE:
        print(f"❌ 清单哈希模式 {mode} 与生成器 {am.SHA256_MODE} 不一致 → 拒绝校验")
        return 1

    ok: List[str] = []
    missing: List[str] = []
    mismatch: List[Tuple[str, str, str]] = []
    for key, meta in files.items():
        # 键空间与生成器一致: TRACKED 用 basename(DATA_DIR 下), TRACKED_EXTRA 用相对 ROOT 路径
        rel = key if "/" in key else os.path.join("data", "backtest", key)
        path = os.path.join(args.backup_root, rel)
        if not os.path.exists(path):
            missing.append(key)
            continue
        digest = am.sha256(path)
        if digest != meta.get("sha256"):
            mismatch.append((key, meta.get("sha256", ""), digest))
        else:
            ok.append(key)

    total = len(files)
    print(f"=== 备份副本指纹校验（{args.backup_root}）===")
    print(f"  哈希模式: {mode}（与生成器共用实现）")
    print(f"  清单产物: {total} | 一致: {len(ok)} | 缺失: {len(missing)} | 不符: {len(mismatch)}")
    for key in missing[:10]:
        print(f"  ❌ 缺失: {key}")
    for key, expect, actual in mismatch[:10]:
        print(f"  ❌ 不符: {key} 期望 {expect[:16]} 实得 {actual[:16]}")

    if missing or mismatch:
        print("❌ 备份副本与清单不一致")
        return 1
    print(f"✅ 备份副本与清单一致（{total} 个产物，哈希模式 {mode}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
