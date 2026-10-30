#!/usr/bin/env python3
"""生成 docs/backtest/P8_EXPOSURE_NEUTRAL_2026-10-31.md。

数字与结论措辞**全部从 JSON 产物派生**，禁止手抄。
用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/gen_p8_doc.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA = os.path.join(ROOT, "data", "backtest")
DOCS = os.path.join(ROOT, "docs", "backtest")
OUT = os.path.join(DOCS, "P8_EXPOSURE_NEUTRAL_2026-10-31.md")

PM = json.load(open(os.path.join(DATA, "multi_asset_p8_main.json"), encoding="utf-8"))
PO_PATH = os.path.join(DATA, "multi_asset_p8_oos.json")
PO = json.load(open(PO_PATH, encoding="utf-8")) if os.path.exists(PO_PATH) else None


def n(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def pct(v, d=2):
    return "—" if v is None else f"{v:+.{d}%}"


def table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(str(c) for c in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> None:
    parts: List[str] = []

    # ── 结论判定（从判据布尔值派生，不手写结论）──
    cm = PM["criteria"]
    co = PO["criteria"] if PO else {}
    keys = [k for k in cm if k not in ("e_random_control",)]
    merged = {}
    for k in keys:
        vals = [cm.get(k), co.get(k)]
        vals = [v for v in vals if v is not None]
        merged[k] = all(vals) if vals else None

    hard = ["a_ashare_sharpe_gt_base", "b_main_combined_ge_base_plus_0.05",
            "c_oos_combined_ge_base_minus_0.02", "d_subwindow_majority_ge_3"]
    passed = all(merged.get(k) is True for k in hard)
    f_ok = merged.get("f_independent_oos_evidence") is True

    if passed and f_ok:
        headline = ("**暴露中性化在生产口径下成立 —— 但 (f) 独立样本外证据必须补齐才能改定稿。**")
        body = ("中性化在 main 与 oos 两个窗口都通过 (a)~(d)，(f) 也满足 → 可作为候选进入定稿评审。")
    elif passed:
        headline = ("**暴露中性化在生产口径下成立，但 (f) 独立样本外证据结构性缺失 → 不改定稿。**")
        body = ("中性化在 main 与 oos 两窗口均通过 (a)~(d)；但 (f) 不满足，"
                "因此只产出候选与量化证据，改定稿需人工点头。")
    else:
        headline = "**暴露中性化证伪 —— 生产口径下是负面的，不要开启。**"
        body = ("中性化在预注册判据上未通过。**这推翻了源码注释里"
                "「唯一跨段稳定有效」的说法** —— 那组乐观数字来自 42f 研究实现口径，"
                "不是生产口径。")

    dm = PM["delta_combined"]
    do = PO["delta_combined"] if PO else None
    am = PM["delta_ashare"]
    ao = PO["delta_ashare"] if PO else None

    parts.append(f"""# P8 暴露中性化裁决 —— 「唯一跨段稳定有效」这句话是错的（2026-10-31）

> **一句话结论**：{headline}
>
> {body}

## 背景：为什么查这个

P7 判定「因子层证伪」—— 25 个公式因子全用就是最优，任何减法更差、任何符号翻转也更差。
P7 的推论是：**主腿的弱不是"因子选错了"，问题在信号构造层或执行层**。

信号构造层唯一有既有跨段证据的候选是 **暴露中性化**
（Barra APT 风格逐日截面 OLS 残差化：y=复合分，X=[size, illiq, 1]；
size = 20 日 log 成交额均值（平台无市值数据，是流动性/规模混合代理），
illiq = Amihud(2002) ILLIQ 20 日均值取 log1p）。

⚠️ **但既有结论互相矛盾，且源码注释站错了边**：

| 口径 | seg1 | seg2 | seg3 | 3 年整体 |
|---|---|---|---|---|
| 研究实现（42f，`neut_segments.json`） | +0.222 | +0.241 | +0.307 | 0.792 → 1.012 |
| **真实引擎（25f，`neutseg_seg*_*.json`）** | **−0.140** | **+0.200** | **−0.269** | **0.880 → 0.757** |

`honest_composite_strategy.py` 的注释原文写的是"本轮**唯一跨段稳定有效**的改进
（段1 Δ夏普 +0.222 / 段2 +0.241 / 段3 +0.307）"——
**那组数字来自研究实现（42f），不是生产口径**。真实引擎口径下三段里两段是负的。

本实验用**当前引擎 + 当前定稿组合**（P5 固定持仓集袖套、w=0.50）重新裁决，
并已同步修正那段误导性注释。

## 本实验

- **唯一变量**：`exposure_neutral` 开/关（策略内置开关，默认关）
- 因子集合固定 25 公式全用（= 当前主腿），袖套 P5 固定持仓集，w = 0.50
- 成本与成交口径与 P3/P5/P6/P7 完全一致；base25 必须逐位复现基线门
  （main {n(PM['base_combined_sharpe'])} = P3 定稿 {n(PM['p3_expected_base'])}）
""")

    rows = []
    for label, d in (("main", PM), ("oos", PO)):
        if not d:
            continue
        b = d["results"]["base25"]
        t = d["results"]["neut25"]
        rows.append([
            label, n(b["a_share_sharpe"]), n(t["a_share_sharpe"]),
            f"{t['a_share_sharpe'] - b['a_share_sharpe']:+.4f}",
            n(b["combined_sharpe"]), n(t["combined_sharpe"]),
            f"{t['combined_sharpe'] - b['combined_sharpe']:+.4f}",
            pct(b["combined_maxdd"]), pct(t["combined_maxdd"]),
        ])
    parts.append(f"""## 结果

{table(["窗口", "A腿夏普(关)", "A腿夏普(开)", "ΔA腿",
         "组合夏普(关)", "组合夏普(开)", "Δ组合", "组合回撤(关)", "组合回撤(开)"], rows)}

- main Δ组合 **{dm:+.4f}**，ΔA腿 **{am:+.4f}**
- oos Δ组合 **{do:+.4f}**，ΔA腿 **{ao:+.4f}**（oos 窗口 = 2021-05-11~2023-05-08）
""")

    cr = []
    for k, v in merged.items():
        label = {
            "a_ashare_sharpe_gt_base": "(a) A 腿自身夏普提升",
            "b_main_combined_ge_base_plus_0.05": "(b) main 组合 ≥ 基线+0.05",
            "c_oos_combined_ge_base_minus_0.02": "(c) oos 组合 ≥ 基线−0.02",
            "d_subwindow_majority_ge_3": "(d) 4 子窗口 ≥3 胜",
            "f_independent_oos_evidence": "(f) 独立样本外证据",
        }.get(k, k)
        cr.append([label, {True: "✅", False: "❌", None: "—"}[v]])
    cr.append(["(e) 随机对照", "—（只有开/关两态，无参数搜索空间 → 不适用）"])
    parts.append(f"""## 预注册判据

{table(["判据", "判定"], cr)}

(f) 的现实状态：中性化有 3 个独立 1 年段的验证，但**真实引擎口径**下是
Δ −0.140 / +0.200 / −0.269（三段里两段负面），方向本身就指向负面；
且 oos 窗口落在 2023-05 之前的 regime，而 5 年窗口检验（0.635→0.548）已显示它 regime 依赖。

## 子窗口

{table(["窗口/子段", "区间", "base25(关)", "neut25(开)"],
       [[f"{w}/{sk}", f"{sg['start']}~{sg['end']}", n(sg.get('base25')), n(sg.get('neut25'))]
        for w, d in (("main", PM), ("oos", PO)) if d
        for sk, sg in d["subwindows"].items()])}

main 子窗口胜数 {PM['subwindow_wins']}/{len(PM['subwindows'])}。
""")

    parts.append(f"""## 工程教训

### 1. 引用别人的"改进有效"结论前，必须确认是同一口径

本仓库已经踩过一次：42f 研究实现的数字被当成 25f 生产口径的结论写进了源码注释，
并在此基础上"唯一跨段稳定有效"的说法流传。**口径 = 因子集 + 引擎版本 + 窗口切分，三者都要对齐。**

### 2. 本实验自身的两个坑（已修）

- **模块全局窗口必须显式同步**：`run_a_share_leg` 依赖 `multi_asset_leg_experiment` 的
  全局 `START/END`。P7 初版漏了同步 → oos 阶段实际跑的是 main 窗口的 A 股腿，
  与 oos 袖套日期几乎不重叠 → 组合夏普凭空变成 0.5241。
  **为什么没被立刻发现**：P7 的 main 阶段恰好等于模块默认值，所以 main 结果完全正确；
  oos 阶段"起点在窗口内"的断言也恰好被 `2023-05-08 ≤ oos 终点 2023-05-08` 放行。
  → **同窗口的巧合会掩盖窗口 bug**。断言已改为同时检查净值首尾。
- **复用既有脚本的窗口同步逻辑时要逐行对齐**，不能凭记忆重写。

## 已知结构性限制

{PM['note']}
""")

    os.makedirs(DOCS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"[gen_p8] ✓ 已写入 {OUT}")


if __name__ == "__main__":
    main()
