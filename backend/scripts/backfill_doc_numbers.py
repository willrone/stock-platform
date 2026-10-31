#!/usr/bin/env python3
"""文档数字回填器 —— 与 audit_doc_numbers.py 成对使用。

## 为什么需要

`audit_doc_numbers.py` 用「文档名 + 表序号 + 行标签 + 列名 + 产物路径」定位单元格，
但**只报告不修改**。引擎修 bug 后（比如 L3.simulate 的现金约束顺序敏感性），
所有产物重算，历史文档里手抄的数字就过时了 —— 而叙述性文字里的数字
审计器根本抓不到，只能人肉同步。

本脚本复用审计器的**同一套锚点与解析逻辑**，把表格单元格的数值
**从产物自动回填进文档**，做到「文档数字由产物生成」。

## 安全约束

- 只改**锚点定位到的那个单元格**，用 `line.split('|')` 局部替换 ——
  单元格里�� `**`、`` ` ``、`+`、百分号等装饰全部原样保留；
- 只改数字字符，保留原有小数位数（从旧值推断），所以表格排版不跳；
- **不碰行标签、列名、正文叙述**；
- 默认 dry-run，只报告；`--apply` 才写盘；
- 写盘前先备份到 `.bak`。

## 用法
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/backfill_doc_numbers.py            # 报告
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/backfill_doc_numbers.py --apply    # 写盘
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DOCS = os.path.join(ROOT, "docs", "backtest")
AUDIT = os.path.join(BACKEND_DIR, "scripts", "audit_doc_numbers.py")


def load_audit():
    spec = importlib.util.spec_from_file_location("aud_mod", AUDIT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[backfill] ❌ 无法加载审计器 {AUDIT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aud_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


AUD = load_audit()


def table_blocks(lines: List[str]) -> List[List[int]]:
    """返回表格块：每块是行号列表（与 AUD.extract_tables 的块顺序一致）。"""
    blocks: List[List[int]] = []
    cur: List[int] = []
    for i, line in enumerate(lines):
        if line.strip().startswith("|"):
            cur.append(i)
        else:
            if len(cur) >= 2:
                blocks.append(cur)
            cur = []
    if len(cur) >= 2:
        blocks.append(cur)
    # 与 extract_tables 一致：去掉全分隔行的块会得到空 rows → 被丢弃
    keep: List[List[int]] = []
    for b in blocks:
        rows = []
        for li in b:
            cells = lines[li].split("|")[1:-1]
            if cells and all(re.fullmatch(r"\s*:?-{2,}:?\s*", c or " ") for c in cells):
                continue
            rows.append(cells)
        if rows:
            keep.append(b)
    return keep


def fmt_like(old: str, want: float) -> str:
    """按旧单元格的格式重写数值（保留装饰、符号、小数位数）。

    **定位规则必须与审计器 `target_number` 完全一致**：
      - 单元格只有一个数字且含 `%` → 百分比格，写 want×100；
      - 否则（纯数值格 / 复合格如 `1.3767 (+59.0%)`）→ 写 want，且只替换**第一个**数字。

    早期版本用「格子里出现 % 就整格 ×100」的粗暴规则，
    会把 `**1.3767 (+59.0%)` 的夏普写成 `137.8600`，
    而审计器当时用同一套错误逻辑 → 审计与回填**同时失灵**，互为盲区。
    """
    nums = AUD.cell_num_positions(old)
    if not nums:
        return old
    _n, s, e = nums[0]
    if len(nums) == 1 and "%" in old:
        val = want * 100.0
    else:
        val = want
    raw = old[s:e]
    dec = len(raw.split(".")[1]) if "." in raw else 0
    out = f"{val:.{dec}f}"
    # 原值带显式正号且新值非负 → 保留正号
    if raw.startswith("+") and not out.startswith("-"):
        out = "+" + out
    return old[:s] + out + old[e:]


def backfill_doc(doc_path: str, anchors: List[Any]) -> Tuple[str, int, int]:
    """返回 (新文本, 修改单元格数, 未能定位数)"""
    with open(doc_path, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.split("\n")
    blocks = table_blocks(lines)

    # 锚点按 (tidx, row_label, col_hdr) 分组
    changed = 0
    missed = 0
    for a in anchors:
        doc_pref, tidx, row_label, col_hdr, spec = a[0], a[1], a[2], a[3], a[4]
        if isinstance(tidx, str):
            cands = []
            for bi, b in enumerate(blocks):
                hdr = [AUD.norm(c) for c in lines[b[0]].split("|")[1:-1]]
                if any(AUD.norm(col_hdr) == AUD.norm(c) for c in hdr):
                    cands.append(bi)
            if not cands:
                missed += 1
                continue
            tidx = cands[0]
        if tidx >= len(blocks):
            missed += 1
            continue
        b = blocks[tidx]
        # 表头行 + 数据行（跳过分隔行），保持与 extract_tables 一致
        hdr_cells: List[str] = []
        data: List[Tuple[int, List[str]]] = []
        for li in b:
            cells = lines[li].split("|")[1:-1]
            if cells and all(re.fullmatch(r"\s*:?-{2,}:?\s*", c or " ") for c in cells):
                continue
            if not hdr_cells:
                hdr_cells = cells
            else:
                data.append((li, cells))
        hdr = [AUD.norm(c) for c in hdr_cells]
        want_col = AUD.norm(col_hdr)
        if want_col not in hdr:
            missed += 1
            continue
        ci = hdr.index(want_col)
        # 行键可以是字符串（匹配第 0 列），也可以是元组（依次匹配前 N 列）——
        # 与审计器 find_cell 同构。P12 §九 E2 那类表首列有重复值，单列定位会歧义。
        is_key = isinstance(row_label, (tuple, list))
        if is_key:
            want_cols = [AUD.norm(x) for x in row_label]
            n_key = len(want_cols)
        else:
            want_cols = [AUD.norm(row_label)]
            n_key = 1
        hit = None
        for li, cells in data:
            if not cells or len(cells) < n_key:
                continue
            if is_key:
                if not AUD.norm(cells[0]).startswith(want_cols[0]):
                    continue
                if not all(AUD.norm(cells[i]) == want_cols[i] for i in range(1, n_key)):
                    continue
            else:
                if AUD.norm(cells[0]) != want_cols[0]:
                    continue
            if hit is not None:
                hit = None
                break
            hit = (li, cells)
        if hit is None or ci >= len(hit[1]):
            missed += 1
            continue
        li, cells = hit
        old = cells[ci]
        want = AUD.resolve(spec)
        new_cell = fmt_like(old, want)
        if new_cell != old:
            parts = lines[li].split("|")
            # split 后 parts[0] 与 parts[-1] 是行首行尾的空串，数据格从 1 开始
            parts[ci + 1] = new_cell
            lines[li] = "|".join(parts)
            changed += 1
    return "\n".join(lines), changed, missed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写盘（默认只报告）")
    args = ap.parse_args()

    anchors = AUD.build_anchors()
    by_doc: Dict[str, List[Any]] = {}
    for a in anchors:
        by_doc.setdefault(a[0], []).append(a)

    total_changed = 0
    total_missed = 0
    for doc_pref, doc_anchors in sorted(by_doc.items()):
        path = None
        for name in sorted(os.listdir(DOCS)):
            if name.startswith(doc_pref) and name.endswith(".md"):
                path = os.path.join(DOCS, name)
                break
        if path is None:
            print(f"❌ {doc_pref}: 找不到文档")
            continue
        new_text, changed, missed = backfill_doc(path, doc_anchors)
        total_changed += changed
        total_missed += missed
        flag = "改" if changed else "—"
        print(f"{flag} {os.path.basename(path):40s} 改动 {changed:4d} 格，"
              f"未定位 {missed:3d} 个锚点")
        if args.apply and changed:
            shutil.copy2(path, path + ".bak")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new_text)
    print(f"\n合计：改动 {total_changed} 格，未定位 {total_missed} 个锚点"
          + ("（已写盘，备份为 *.bak）" if args.apply else "（dry-run，未写盘）"))
    if total_missed:
        print("⚠️ 有锚点没能定位到单元格 —— 回填不完整，审计会失败")


if __name__ == "__main__":
    main()
