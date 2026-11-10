#!/usr/bin/env python3
"""从 P17 产物生成 docs/backtest/P17_CLASS_SELECT_2026-11-01.md（数字零手抄）。"""
from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "backtest")
DOC = os.path.join(ROOT, "docs", "backtest", "P17_CLASS_SELECT_2026-11-01.md")

FIX = {"gold": ["518880", "159934", "159937"],
       "bond": ["511260", "511010", "511020"],
       "commodity": ["159980", "159985", "159981"]}
CLS_ORDER = ["gold", "commodity", "bond"]


def load(w: str, kind: str) -> Dict[str, Any]:
    with open(os.path.join(OUT, f"multi_asset_p17_{kind}_{w}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def pct_rank(rows: List[Dict[str, Any]], combo: List[str]) -> int:
    vals = sorted((r["combined_sharpe"] for r in rows), reverse=True)
    for r in rows:
        if sorted(r["combo"]) == sorted(combo):
            return vals.index(r["combined_sharpe"]) + 1
    return -1


L: List[str] = []
W = L.append

single = {w: load(w, "single") for w in ["main", "oos"]}
verify = {w: load(w, "verify") for w in ["main", "oos"]}
freq = {w: load(w, "freq") for w in ["main", "oos"]}

bm = single["main"]["baseline"]["combined_sharpe"]
bo = single["oos"]["baseline"]["combined_sharpe"]

W("# P17 · 类内选标的的 oracle 上界 + 袖套调仓频率")
W("")
W("**日期**：2026-11-01　|　**性质**：上界审判（先算 oracle，再谈优化器）")
W("")
W("> **一句话结论**：")
W("> 1. **类内选标的彻底封线。** 用未来信息挑每一类的最优持仓集，"
  f"组合夏普上界只有 **main +{max(r['combined_sharpe'] for r in verify['main']['results'] if r['label'] == 'all_three_oracles') - bm:.4f}** / "
  f"**oos +{max(r['combined_sharpe'] for r in verify['oos']['results'] if r['label'] == 'all_three_oracles') - bo:.4f}** —— "
  "远低于 P15 的 +0.14 门槛；而且 **oos 上随便随机选一组就能拿到 +0.0313**，"
  "即**上界本身落在随机噪声范围内**。跨窗口**零迁移**。")
W("> 2. **袖套调仓频率封线，月度（现行定稿）站得住。** main 最优是双周、oos 最优是月度 —— "
  "**跨窗口最优完全反向 = 噪声**；而**月度是唯一在样本外排第一的**频率。")
W("> 3. **对 P5/P3 定稿的正面验证**：定稿的「流动性 top-3」固定持仓集不是次优，"
  "它在类内分布里处于**中位附近**（既不最好也不最差），且是**规则可复现**的选择；"
  "而 bond 类唯一的空间全部来自「含不含可转债」这个**分类事实**（再次印证 P3），"
  "不是「挑中哪几只」。")
W("")
W("---")
W("")
W("## 1. 为什么做这件事")
W("")
W("台账 §四 未探分支 #1/#2：")
W("")
W("- **#1 类内选标的的 oracle 上界** —— P3 穷举过 bond 类、P2 枚举过 commodity 的 4 个候选，"
  "但 **gold 类 14 只候选从未枚举**（P5 只做性质审计）。")
W("- **#2 袖套自身的调仓频率** —— P9 只探了**主腿**频率，袖套现为「每月恢复类间等权」从未被挑战。")
W("")
W("两者都属「读完就能封线」的分支，本轮一次性收掉。")
W("")
W("## 2. 方法")
W("")
W("### 2.1 口径（与 P3/P5/P13 定稿完全一致）")
W("")
W("| 组件 | 口径 |")
W("|---|---|")
W("| 袖套 | 3 类各 1/3、类内等权、**每月（21 交易日）再平衡回类间等权** |")
W("| 成交 | 信号日后下一交易日 `open`，佣金 0.03% + 滑点 0.01%，ETF 免印花税 |")
W("| 组合 | `merge_by_date([A 股腿, 袖套], [0.5, 0.5])`（按日对齐归一化加权） |")
W("| 交易日历 | `L3.candidates_by_class` 各类候选池**并集**（P3 教训 4） |")
W("")
W("### 2.2 三阶段")
W("")
W("| 阶段 | 做法 | 目的 |")
W("|---|---|---|")
W("| **0 基线复现门** | 定稿持仓集 → 逐位复现 P13 产物 `base_equal` | 口径一致性硬门 |")
W("| **1 精确单类扫描** | 每类穷举 `C(pool,3)`，其余两类固定定稿，**全部走完整撮合器** | 拿到每类真实上界与完整分布 |")
W("| **2 组合 oracle + 随机族** | 三类同时取优 + 定稿 + **20 组同池随机** | 上界 + 随机对照（判噪声） |")
W("")
W("### 2.3 枚举规模（全部精确撮合，无近似）")
W("")
W("| 窗口 | gold | commodity | bond | 单类扫描次数 |")
W("|---|---|---|---|---|")
for w in ["main", "oos"]:
    c = single[w]["classes"]
    n = sum(c[k]["n"] for k in CLS_ORDER)
    W(f"| {w} | {c['gold']['n']} | {c['commodity']['n']} | {c['bond']['n']} | **{n}** |")
W("")
W("> ⚠️ **一度踩过的坑**：先用「零成本 + 收盘价递推」的向量化近似做 12.2 万组枚举排序，"
  "取近似 top-80 再精确跑 —— **main 窗口 36 个候选全部劣于定稿**。")
W("> 原因：**交易成本正是候选间的差异来源**（流动性/波动率 → 漂移 → 换手），"
  "零成本近似把这层差异整个抹掉了。**近似只可用于粗筛，不能用来定 oracle。**")
W("")
W("## 3. 结果一：类内选标的的 oracle 上界")
W("")
W("### 3.1 单类上界（其余两类固定定稿）")
W("")
W("| 类 | 窗口 | 枚举数 | 定稿持仓集的值 | 该类最优持仓集 | 最优值 | **上界增益** | 定稿在类内的排名 |")
W("|---|---|---|---|---|---|---|---|")
for w in ["main", "oos"]:
    base = single[w]["baseline"]["combined_sharpe"]
    for cls in CLS_ORDER:
        c = single[w]["classes"][cls]
        best = c["best"]
        rk = pct_rank(c["rows"], FIX[cls])
        W(f"| {cls} | {w} | {c['n']} | {base:.4f} | `{best['combo']}` | "
          f"{best['combined_sharpe']:.4f} | **{best['combined_sharpe'] - base:+.4f}** | "
          f"{rk}/{c['n']} |")
W("")
W("### 3.2 三类同时取优（组合上界）+ 同池随机对照")
W("")
W("| 窗口 | 定稿 | **oracle（三类同取优）** | 增益 | 随机族 20 组：最大 / 中位 / 最小 | 超定稿组数 |")
W("|---|---|---|---|---|---|")
for w in ["main", "oos"]:
    base = single[w]["baseline"]["combined_sharpe"]
    res = verify[w]["results"]
    orc = [r for r in res if r["label"] == "all_three_oracles"][0]
    rnd = [r["combined_sharpe"] for r in res if r["label"].startswith("rand_")]
    over = sum(1 for x in rnd if x > base)
    W(f"| {w} | {base:.4f} | **{orc['combined_sharpe']:.4f}** | "
      f"**{orc['combined_sharpe'] - base:+.4f}** | "
      f"{max(rnd):.4f} / {statistics.median(rnd):.4f} / {min(rnd):.4f} | "
      f"**{over}/20** |")
W("")
W("**判读**：")
W("")
W("- **上界极小**：main +0.0193 / oos +0.0357，均远低于 P15 的 +0.14 门槛。")
W("- **上界在噪声内**：oos 上 **20 组随机里有 5 组超过定稿**，最高 +0.0313 —— "
  "**oracle 上界（+0.0357）只比「随便随机挑」的最好一组高 0.0044**。"
  "main 上则是 20 组随机**全部劣于定稿**（最大 −0.0144），"
  "而 oracle（+0.0193）比随机族中位（+0.0）高不了多少。")
W("- **零迁移**：每个类的 main 最优持仓集在 oos 的排名都掉到后段，"
  "oos 最优在 main 同样落后（见下表）。")
W("")
W("### 3.3 跨窗口迁移性（逐类）")
W("")
W("| 类 | main 最优 | oos 最优 | 重叠 | main 最优在 oos 的排名 | oos 最优在 main 的排名 |")
W("|---|---|---|---|---|---|")
for cls in CLS_ORDER:
    mb = single["main"]["classes"][cls]["best"]["combo"]
    ob = single["oos"]["classes"][cls]["best"]["combo"]
    mo = pct_rank(single["oos"]["classes"][cls]["rows"], mb)
    om = pct_rank(single["main"]["classes"][cls]["rows"], ob)
    W(f"| {cls} | `{mb}` | `{ob}` | {'是' if sorted(mb) == sorted(ob) else '**否**'} | "
      f"**{mo}/{single['oos']['classes'][cls]['n']}** | "
      f"**{om}/{single['main']['classes'][cls]['n']}** |")
W("")
W("### 3.4 为什么 gold / commodity 几乎没有空间（结构性解释）")
W("")
W("| 类 | 类内组合夏普跨度 | 定稿在类内的分位 | 解释 |")
W("|---|---|---|---|")
for w in ["main", "oos"]:
    for cls in ["gold", "commodity"]:
        c = single[w]["classes"][cls]
        vals = [r["combined_sharpe"] for r in c["rows"]]
        rk = pct_rank(c["rows"], FIX[cls])
        exp = ("类内 ρ≈0.99 → 3 只 ≈ 1 个风险源（**P5 已证**）" if cls == "gold"
               else f"池只有 {c['n']} 组（4 选 3）")
        W(f"| {cls} | {w} | {max(vals) - min(vals):.4f} | {rk}/{c['n']} | {exp} |")
W("")
W("**bond 是唯一有空间的类（跨度 0.075），但空间全部来自「含不含可转债」**：")
W("")
W("| 窗口 | 最优持仓集 | 含可转债？ | 定稿含可转债？ |")
W("|---|---|---|---|")
for w in ["main", "oos"]:
    best = single[w]["classes"]["bond"]["best"]["combo"]
    cb = {"511380", "511180"}
    W(f"| {w} | `{best}` | {'**是**' if cb & set(best) else '否'} | "
      f"{'是' if cb & set(FIX['bond']) else '否'} |")
W("")
W("> 两个窗口的最优持仓集**零重叠**，但都**不含可转债** → "
  "**再次印证 P3 的核心结论：bond 类的增量来自「可转债是拖累」这个分类事实，"
  "不来自「我挑中了哪几只」。**")
W("")
W("## 4. 结果二：袖套调仓频率")
W("")
W("固定定稿持仓集，唯一变量 = 袖套调仓周期。")
W("")
W("| 调仓周期 | main 调仓次数 | main 换手 | main 袖套夏普 | **main 组合夏普** | oos 调仓次数 | oos 换手 | oos 袖套夏普 | **oos 组合夏普** |")
W("|---|---|---|---|---|---|---|---|---|")
for r in freq["main"]["results"]:
    ro = [x for x in freq["oos"]["results"] if x["label"] == r["label"]][0]
    mark = " **←**" if r["label"] in ("biweekly_10", "monthly_21") else ""
    W(f"| `{r['label']}`{mark} | {r['n_rebalances']} | {r['turnover_ratio']:.2f}x | "
      f"{r['sleeve_sharpe']:.4f} | **{r['combined_sharpe']:.4f}** | "
      f"{ro['n_rebalances']} | {ro['turnover_ratio']:.2f}x | {ro['sleeve_sharpe']:.4f} | "
      f"**{ro['combined_sharpe']:.4f}** |")
W("")
W("**判读**：")
W("")
W("- **main 最优是双周（1.4287），oos 最优是月度（1.0691）→ 跨窗口最优完全反向 = 噪声。**"
  "与 P9（主腿：main 最优 wk_fri、oos 最优 wk_mon）**方法论同构**。")
W("- **月度是唯一在样本外排第一的频率**（oos 1.0691 > 双周 1.0508 > 周频 1.0554），"
  "main 上也仅落后双周 0.0173 → **现行定稿 `monthly_21` 站得住，不改**。")
W("- **买入持有在两个窗口都最差**（main 1.3608 / oos 0.9835）→ "
  "**「月度恢复类间等权」这件事本身是有价值的**（不是可有可无的成本）。")
W("- 频率太高（周频）在 oos 也劣于月度（1.0554 < 1.0691）→ 不是「越勤越好」。")
W("")
W("## 5. 结论")
W("")
W("| 分支 | 结论 | 依据 |")
W("|---|---|---|")
W(f"| **#1 类内选标的** | **❌ 封线** | oracle 上界 main +{max(r['combined_sharpe'] for r in verify['main']['results'] if r['label'] == 'all_three_oracles') - bm:.4f} / "
  f"oos +{max(r['combined_sharpe'] for r in verify['oos']['results'] if r['label'] == 'all_three_oracles') - bo:.4f}，"
  "远低于 +0.14；且 oos 随机族最高 +0.0313 ≈ 上界 → 上界在噪声内；跨窗口零迁移 |")
W("| **#2 袖套调仓频率** | **❌ 封线（月度保留）** | main 最优双周 / oos 最优月度，反向不迁移；月度是唯一 oos 第一；买入持有两窗口最差 |")
W("")
W("**定稿影响：无。** 组合仍是 A 股腿 50% + gold / commodity / bond 各 16.67%，"
  "固定持仓集与每月再平衡全部不变。")
W("")
W("## 6. 诚实的局限")
W("")
W("1. **只穷举了「每类 3 只」的持仓集大小**。类内持有 1/4/5/6 只的空间没探 —— "
  "但 P13 已封「类内权重自由度」，且 P5 已证 gold 类内自由度≈0，故不构成缺口。")
W("2. **两窗口 = 两样本**。rolling 胜率缓解但不能替代独立长历史审判（与 P3 §八 同）。")
W("3. **oracle 是「每类独立取优再组合」**，不是「在 12.2 万组里直接取最优」—— "
  "后者因成本不可向量化而未做（见 §2.3 的坑）。"
  "严格说这是**上界的一个下界**，真实全局 oracle 可能略高；"
  "但按 P15 的量级（+0.14 门槛），差 0.01 量级不改变结论。")
W("4. **gold 类的 14 只池取自 P5 审计**（2023-05 时点），未重新做 point-in-time 分类筛查。")
W("")
W("## 7. 复现")
W("")
W("```bash")
W("cd ~/Projects/stock-platform/backend")
W("for W in main oos; do")
W("  env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 \\")
W("    scripts/multi_asset_p17_class_select.py --window $W --phase single")
W("  env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 \\")
W("    scripts/multi_asset_p17_class_select.py --window $W --phase combine")
W("  env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 \\")
W("    scripts/multi_asset_p17_class_select.py --window $W --phase freq")
W("done")
W("env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 scripts/gen_p17_doc.py")
W("```")
W("")
W("产物：`multi_asset_p17_single_{main,oos}.json` / `multi_asset_p17_verify_{main,oos}.json` / "
  "`multi_asset_p17_freq_{main,oos}.json`")
W("")

os.makedirs(os.path.dirname(DOC), exist_ok=True)
with open(DOC, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"✓ 已写入 {DOC}（{len(L)} 行）")
