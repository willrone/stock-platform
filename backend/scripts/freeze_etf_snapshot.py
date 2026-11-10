"""生成 ETF 行情的**冻结快照**（研究侧专用）。

为什么需要：
  ETF 行情是**前复权**序列 —— 每次分红都会把历史价格整体重锚。
  实盘周任务会全量重拉 ETF 数据（`scripts/weekly_paper_portfolio.sh`），
  重锚之后研究端的历史价格全变了 → 研究产物（袖套夏普 1.6244、
  组合 1.4114）**无法复现**，审计链断掉。

  所以：研究侧读 `data/etf/parquet_frozen/`（一次性冻结），
  实盘/产品侧读 `data/etf/parquet/`（每周更新）。

用法：
  python3 scripts/freeze_etf_snapshot.py            # 生成快照 + 校验和清单
  python3 scripts/freeze_etf_snapshot.py --verify   # 只校验现有快照
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import pandas as pd

#: 项目根（backend/scripts/ → 上两级）。研究脚本里的 ROOT 也是这一层，
#: 必须一致，否则快照会落到 backend/data 下面、研究侧根本找不到。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIVE = os.path.join(ROOT, "data", "etf", "parquet")
FROZEN = os.path.join(ROOT, "data", "etf", "parquet_frozen")
MANIFEST = os.path.join(FROZEN, "FROZEN_MANIFEST.json")


def build_manifest() -> dict:
    files = {}
    for f in sorted(os.listdir(FROZEN)):
        if not f.endswith(".parquet"):
            continue
        p = os.path.join(FROZEN, f)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        df = pd.read_parquet(p)
        files[f] = {
            "sha256": h,
            "rows": int(len(df)),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        }
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="只校验，不重新生成")
    args = ap.parse_args()

    if args.verify:
        if not os.path.exists(MANIFEST):
            print(f"❌ 没有快照清单 {MANIFEST}")
            return 1
        old = json.load(open(MANIFEST))["files"]
        new = build_manifest()
        bad = []
        for f, meta in old.items():
            if f not in new:
                bad.append(f"{f}: 缺失")
            elif new[f]["sha256"] != meta["sha256"]:
                bad.append(f"{f}: 校验和不符")
        for f in new:
            if f not in old:
                bad.append(f"{f}: 快照外新增（未登记）")
        if bad:
            print(f"❌ 快照校验失败 {len(bad)} 项：")
            for b in bad[:20]:
                print("   ", b)
            return 1
        print(f"✅ 快照校验通过：{len(old)} 只 ETF 全部一致")
        return 0

    if not os.path.isdir(LIVE):
        print(f"❌ 实时目录不存在 {LIVE}")
        return 1
    if os.path.exists(FROZEN):
        print(f"⚠️ 快照已存在，先删除再重建：{FROZEN}")
        return 1

    shutil.copytree(LIVE, FROZEN)
    files = build_manifest()
    json.dump({
        "frozen_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "why": (
            "ETF 行情是**前复权**序列（前值随分红整体重锚）。实盘周任务会全量重拉，"
            "重锚会让历史价格整体偏移 → 研究产物（袖套夏普 1.6244 / 组合 1.4114）"
            "无法复现。因此冻结一份快照供研究侧脚本使用，实盘继续用 data/etf/parquet。"
        ),
        "files": files,
    }, open(MANIFEST, "w"), ensure_ascii=False, indent=1)
    print(f"✅ 快照已生成：{FROZEN}（{len(files)} 只 ETF）")
    print(f"   清单 → {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
