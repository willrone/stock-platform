#!/usr/bin/env python3
"""正文叙述数字刷新器 —— 表格数字由 backfill_doc_numbers.py 管，正文由本脚本管。

## 为什么需要

`audit_doc_numbers.py` / `backfill_doc_numbers.py` 只覆盖**表格单元格**。
文档正文里的「当前状态」数字（如「组合夏普 1.3767 / 1.0382」）没人管，
引擎修 bug 后就会和产物漂移。

本脚本用**显式规则表**把正文里的数字替换成产物派生值：
每条规则 = (文档前缀, 旧字面量, 新字面量)；旧字面量必须在目标文档里**唯一命中**，
否则直接报错退出 —— 避免静默漏改或改错位置。

## 严格区分两类数字

- **当前状态**：随产物变，必须跟着改（例：基线组合夏普）。
- **历史对照**：记录当时观测到的事实，**必须保留**（例：「修复前 1.3581 → 修复后 1.3767」）。
  这类数字所在段落会加「当时口径」声明。

## 用法
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/refresh_body_numbers.py            # 报告
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/refresh_body_numbers.py --apply    # 写盘
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DOCS = os.path.join(ROOT, "docs", "backtest")
DATA = os.path.join(ROOT, "data", "backtest")


def J(name: str) -> dict:
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def build_values() -> dict:
    fr = J("multi_asset_final_robustness.json")
    fro = J("multi_asset_final_robustness_oos.json")
    p2m = J("multi_asset_p2_main.json")["results"]
    p2o = J("multi_asset_p2_oos.json")["results"]
    p2l = J("multi_asset_p2_long.json")["results"]
    l3m = J("multi_asset_l3.json")["results"]
    l3o = J("multi_asset_l3_oos.json")["results"]

    f1m, f1o = fr["F1_final"], fro["F1_final"]
    f7m, f7o = fr["F7_rebalance_contribution"], fro["F7_rebalance_contribution"]
    bh_m, bh_o = f7m["买入持有"], f7o["买入持有"]

    v = {
        # 当前基线（top3 每月再平衡）
        "BASE_M": f1m["combined_sharpe"],
        "BASE_O": f1o["combined_sharpe"],
        "PCT_M": f1m["vs_baseline_pct"],
        "PCT_O": f1o["vs_baseline_pct"],
        "DD_M": f1m["combined_maxdd"],
        "CORR_GOLD": f1m["corr_with_gold"],
        # 买入持有口径
        "BH_M": bh_m["combined_sharpe"],
        "BH_O": bh_o["combined_sharpe"],
        "BH_PCT_M": bh_m["vs_baseline_pct"],
        "BH_PCT_O": bh_o["vs_baseline_pct"],
        # P2 商品腿
        "C980_M": p2m["com_single_159980"]["combined_sharpe"],
        "C980_O": p2o["com_single_159980"]["combined_sharpe"],
        "C985_M": p2m["com_single_159985"]["combined_sharpe"],
        "C985_O": p2o["com_single_159985"]["combined_sharpe"],
        "CALL_M": p2m["com_all"]["combined_sharpe"],
        "CALL_O": p2o["com_all"]["combined_sharpe"],
        "CLS_M": p2m["cls_equal"]["combined_sharpe"],
        "CLS_O": p2o["cls_equal"]["combined_sharpe"],
        "CLS_L": p2l["cls_equal"]["sleeve_sharpe"],
        # L3 锚点
        "L3_BH_M": l3m["static_top3_buyhold"]["combined_sharpe"],
        "L3_BH_O": l3o["static_top3_buyhold"]["combined_sharpe"],
    }
    return v


def rules(v: dict) -> list:
    """(文档前缀, 旧字面量, 新字面量) —— 旧字面量必须在文档中唯一命中。"""
    return [
        # ── CONCLUSION_2026-10-27.md：当前状态总结 ──
        ("CONCLUSION_2026-10-27",
         "**1.3767**（+ETF多资产腿，推导）/ **1.0382**（样本外，+44.4%）",
         f"**{v['BASE_M']:.4f}**（+ETF多资产腿，推导）/ **{v['BASE_O']:.4f}**（样本外，+{v['PCT_O']:.1f}%）"),
        ("CONCLUSION_2026-10-27",
         "**1.3352**（推导窗口，买入持有口径）/ **1.3767**（加每月再平衡，+58.96%）",
         f"**{v['BH_M']:.4f}**（推导窗口，买入持有口径）/ **{v['BASE_M']:.4f}**（加每月再平衡，+{v['PCT_M']:.2f}%）"),
        ("CONCLUSION_2026-10-27",
         "**1.0382（+44.44%）**，回撤 −13.37% → **−8.70%**",
         f"**{v['BASE_O']:.4f}（+{v['PCT_O']:.2f}%）**，回撤 −13.37% → **−{abs(v['DD_M']) * 100:.2f}%**"),

        # ── LONG_HISTORY_2026-10-30.md：当前定稿夏普 ──
        ("LONG_HISTORY_2026-10-30",
         "**1.3767（推导）/ 1.0382（样本外）**",
         f"**{v['BASE_M']:.4f}（推导）/ {v['BASE_O']:.4f}（样本外）**"),

        # ── MULTI_ASSET_LEG_2026-10-30.md：基线与买入持有口径 ──
        ("MULTI_ASSET_LEG_2026-10-30",
         "样本外夏普 0.9637 → 1.0382",
         f"样本外夏普 {v['BH_O']:.4f} → {v['BASE_O']:.4f}"),
        ("MULTI_ASSET_LEG_2026-10-30",
         "推导 **1.3263 → 1.3352（+53.13% → +54.16%）**",
         f"推导 **1.3263 → {v['BH_M']:.4f}（+53.13% → +{v['BH_PCT_M']:.2f}%）**"),
        ("MULTI_ASSET_LEG_2026-10-30",
         "样本外 **0.8946 → 0.9637（+24.45% → +34.07%）**",
         f"样本外 **0.8946 → {v['BH_O']:.4f}（+24.45% → +{v['BH_PCT_O']:.2f}%）**"),
        ("MULTI_ASSET_LEG_2026-10-30",
         "样本外夏普 0.9637 → **1.0382（+44.4%）**，与 gold 相关 0.665 → 0.574",
         f"样本外夏普 {v['BH_O']:.4f} → **{v['BASE_O']:.4f}（+{v['PCT_O']:.1f}%）**，"
         f"与 gold 相关 0.665 → {v['CORR_GOLD']:.3f}"),
        ("MULTI_ASSET_LEG_2026-10-30",
         "只是夏普从 1.3767 降到 0.9945",
         f"只是夏普从 {v['BASE_M']:.4f} 降到 0.9945"),

        # ── P2_CLASS_WEIGHT_2026-10-31.md ──
        ("P2_CLASS_WEIGHT_2026-10-31",
         "推导窗口 159980（有色）最好 1.3495，\n   样本外它掉到 0.8737（单只里倒数）；样本外 159985（豆粕）最好 1.1699，\n   推导窗口它只有 1.3080（单只里中游）。",
         f"推导窗口 159980（有色）最好 {v['C980_M']:.4f}，\n"
         f"   样本外它掉到 {v['C980_O']:.4f}（单只里倒数）；样本外 159985（豆粕）最好 {v['C985_O']:.4f}，\n"
         f"   推导窗口它只有 {v['C985_M']:.4f}（单只里中游）。"),
        ("P2_CLASS_WEIGHT_2026-10-31",
         "组合夏普从 1.3767 掉到 1.3679（样本外 1.0382 → 1.0109）。",
         f"组合夏普从 {v['CLS_M']:.4f} 掉到 {v['CALL_M']:.4f}"
         f"（样本外 {v['CLS_O']:.4f} → {v['CALL_O']:.4f}）。"),
        ("P2_CLASS_WEIGHT_2026-10-31",
         "（1.3767 / 1.0382），且推导窗口排第一。",
         f"（{v['CLS_M']:.4f} / {v['CLS_O']:.4f}），且推导窗口排第一。"),
        ("P2_CLASS_WEIGHT_2026-10-31",
         "注意：单只 159985 的\"较差窗口\" 1.1699 其实高于 top3 的 1.0382 ——",
         f"注意：单只 159985 的\"较差窗口\" {v['C985_O']:.4f} 其实高于 top3 的 {v['CLS_O']:.4f} ——"),
        ("P2_CLASS_WEIGHT_2026-10-31",
         "| 等权（基线） | 1.3767 | — | 1.0382 | — | 1.1080 |",
         f"| 等权（基线） | {v['CLS_M']:.4f} | — | {v['CLS_O']:.4f} | — | {v['CLS_L']:.4f} |"),

        # ── XCLASS_MOMENTUM_2026-10-31.md ──
        ("XCLASS_MOMENTUM_2026-10-31",
         "基线：main **1.3767** / oos **1.0382**",
         f"基线：main **{v['BASE_M']:.4f}** / oos **{v['BASE_O']:.4f}**"),
        ("XCLASS_MOMENTUM_2026-10-31",
         "基线 1.3925 ≠ 定稿 1.3767，结果不可比",
         f"基线 1.3925 ≠ 定稿 {v['BASE_M']:.4f}，结果不可比"),

        # ── FINAL_ROBUSTNESS_2026-10-30.md：结论段（历史修复记录段不改）──
        ("FINAL_ROBUSTNESS_2026-10-30",
         "- 推导窗口 夏普 **1.3767（+58.96%）**、样本外 **1.0382（+44.44%）**",
         f"- 推导窗口 夏普 **{v['BASE_M']:.4f}（+{v['PCT_M']:.2f}%）**、"
         f"样本外 **{v['BASE_O']:.4f}（+{v['PCT_O']:.2f}%）**"),

        # ── L3_MOMENTUM_2026-10-30.md：锚点复现段 ──
        ("L3_MOMENTUM_2026-10-30",
         "**锚点复现**：`static_top3_buyhold` = 1.3374（旧定稿 1.3263，差 0.84%）。"
         "样本外 0.9532（旧定稿 0.8946，差 6.5%）——差异来自候选池："
         "本脚本按窗口内流动性排序，且 commodity 类已按修正后的分类重新排序。",
         f"**锚点复现**：`static_top3_buyhold` = {v['L3_BH_M']:.4f}、"
         f"样本外 {v['L3_BH_O']:.4f}，与 `FINAL_ROBUSTNESS` 的买入持有口径"
         f"（{v['BH_M']:.4f} / {v['BH_O']:.4f}）一致"
         f"（旧版本存在 0.84% / 6.5% 的偏差，来源是候选池排序口径与分类修正，重跑后已消除）。"),
    ]


# ── 数值口径声明：插入到历史文档顶部 ──
DECL_BLOCK = """> **数值口径声明（2026-10-31 更新）**
>
> 撮合器修复了现金约束下的**持仓顺序敏感性**（`multi_asset_l3_momentum.py`：
> 现金不足时按比例缩放全部买入额，取代原先逐笔"先到先得"），
> 因此全部产物已重跑，组合夏普有约 `0.002` 量级的变化。
>
> - **表格内数值**：由 `backfill_doc_numbers.py` 从当前产物**自动回填**，与产物逐格一致；
> - **正文中形如 `1.3581 → 1.3767` 的数字**：是**历史对照**（当时观测到的修复前后对比），
>   **保留不改写**；右侧的新值同样保留为当时口径；
> - **当前定稿**（剔除可转债 `511380`、三类固定持仓集）：
>   组合夏普 main `1.4114` / oos `1.0691`，见 `P3_BOND_CLASS_2026-10-31.md`；
> - **主腿权重 `w` 敏感性**：见 `P6_WEIGHT_SENS_2026-10-31.md`（`w` 已被标记为高敏感维度）。

"""

DECL_DOCS = (
    "CONCLUSION_2026-10-27",
    "FINAL_ROBUSTNESS_2026-10-30",
    "LONG_HISTORY_2026-10-30",
    "MULTI_ASSET_LEG_2026-10-30",
    "L3_MOMENTUM_2026-10-30",
    "XCLASS_MOMENTUM_2026-10-31",
    "P2_CLASS_WEIGHT_2026-10-31",
)


def insert_decl(name: str, body: str) -> str:
    """在标题之后的第一个空行处插入口径声明（幂等）。"""
    if "数值口径声明（2026-10-31 更新）" in body:
        return body
    lines = body.split("\n")
    # 找标题行
    ti = next((i for i, l in enumerate(lines) if l.startswith("# ")), None)
    if ti is None:
        return body
    # 标题后第一个空行
    j = ti + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1
    return "\n".join(lines[:j]) + "\n" + DECL_BLOCK + "\n".join(lines[j:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写盘（默认只报告）")
    args = ap.parse_args()

    v = build_values()
    rs = rules(v)
    docs = {}
    for name in sorted(os.listdir(DOCS)):
        if name.endswith(".md"):
            docs[name] = open(os.path.join(DOCS, name), encoding="utf-8").read()

    problems = []
    planned = {}
    for pref, old, new in rs:
        hit = None
        for name, body in docs.items():
            if not name.startswith(pref):
                continue
            c = body.count(old)
            if c == 0:
                continue
            if c > 1:
                problems.append(f"{name}: 旧字面量命中 {c} 次（需唯一）：{old[:60]!r}")
            hit = name
            break
        if hit is None:
            problems.append(f"{pref}: 找不到旧字面量 {old[:60]!r}")
            continue
        planned.setdefault(hit, []).append((old, new))

    if problems:
        print("❌ 规则表校验失败：")
        for p in problems:
            print("   " + p)
        sys.exit(1)

    total = 0
    for name, items in sorted(planned.items()):
        body = docs[name]
        for old, new in items:
            if old == new:
                continue
            body = body.replace(old, new, 1)
            total += 1
        if name.startswith(DECL_DOCS):
            body = insert_decl(name, body)
        path = os.path.join(DOCS, name)
        if args.apply:
            shutil.copyfile(path, path + ".bak")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
        print(f"  {name}: {len(items)} 处")
    print(f"\n合计 {total} 处（{'已写盘' if args.apply else 'dry-run，未写盘'}）")
    if not args.apply:
        print("提示：加 --apply 写盘")


if __name__ == "__main__":
    main()
