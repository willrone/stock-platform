#!/usr/bin/env python3
"""回测产物指纹清单 —— 防"静默覆盖 / 产物丢失"。

## 为什么需要

- `data/` 被 .gitignore，**所有回测产物零版本控制**。
- 文档里的每个数字都靠 `audit_doc_numbers.py` 回读 JSON 产物核对。
- 一旦某个历史脚本（或迟到的后台进程）覆盖了 JSON，文档数字就不可验证，
  而且**不会有任何报错** —— 审计会"成功"核对出一份已经被改过的产物。

## 做法

`build` 模式：对关键产物算 SHA256，连同生成时间、窗口、脚本 git commit 写进
`docs/backtest/ARTIFACT_MANIFEST.json`（**这个文件纳入 git**，是唯一的版本化锚）。

`verify` 模式：重算 SHA256 与清单比对，任何一个产物被改动/丢失都报出来。

## 用法

    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/artifact_manifest.py build
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/artifact_manifest.py verify
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA_DIR = os.path.join(ROOT, "data", "backtest")
MANIFEST = os.path.join(ROOT, "docs", "backtest", "ARTIFACT_MANIFEST.json")

# 关键产物：文档数字全部依赖这些文件
TRACKED = [
    "ALL_real_lag1_v2.json",              # A 股腿权威基线
    "multi_asset_l3.json",                # L3 类内动量
    "multi_asset_l3_oos.json",
    "multi_asset_p2_main.json",           # P2 商品类内 + 类间配权
    "multi_asset_p2_oos.json",
    "multi_asset_p2_long.json",
    "multi_asset_p2_attrib.json",
    "multi_asset_p3_main.json",           # P3 bond 类纠错
    "multi_asset_p3_oos.json",
    "multi_asset_p3_long.json",
    "p3_recon.json",
    "p3_bond_recon.json",
    "multi_asset_p4_main.json",           # P4 配权（后续追加）
    "multi_asset_p4_oos.json",
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build() -> int:
    entries: Dict[str, Any] = {}
    missing: List[str] = []
    for name in TRACKED:
        p = os.path.join(DATA_DIR, name)
        if not os.path.exists(p):
            missing.append(name)
            continue
        st = os.stat(p)
        entries[name] = {
            "sha256": sha256(p),
            "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        }
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data_dir": os.path.relpath(DATA_DIR, ROOT),
        "files": entries,
        "missing": missing,
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"✓ 清单已写入 {os.path.relpath(MANIFEST, ROOT)}")
    print(f"  指纹 {len(entries)} 个产物"
          + (f"；缺失 {len(missing)} 个: {missing}" if missing else ""))
    return 0


def verify() -> int:
    if not os.path.exists(MANIFEST):
        print("❌ 清单不存在，请先 build")
        return 1
    with open(MANIFEST, encoding="utf-8") as fh:
        manifest = json.load(fh)
    bad: List[str] = []
    for name, meta in manifest.get("files", {}).items():
        p = os.path.join(DATA_DIR, name)
        if not os.path.exists(p):
            bad.append(f"{name}: 文件丢失")
            continue
        cur = sha256(p)
        if cur != meta["sha256"]:
            bad.append(f"{name}: sha256 不符\n      清单 {meta['sha256'][:16]}…"
                       f"\n      当前 {cur[:16]}…")
    if bad:
        print(f"❌ 产物指纹校验失败（{len(bad)} 项）：")
        for b in bad:
            print(f"   - {b}")
        print("\n   → 产物被覆盖或丢失，文档数字不可信。先查清是谁改的再重跑。")
        return 1
    print(f"✅ 产物指纹校验通过：{len(manifest.get('files', {}))} 个产物与清单一致"
          f"（清单生成于 {manifest.get('generated_at')}，commit {manifest.get('git_commit')}）")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if mode == "build":
        sys.exit(build())
    elif mode == "verify":
        sys.exit(verify())
    else:
        print(__doc__)
        sys.exit(2)
