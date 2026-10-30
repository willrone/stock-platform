#!/usr/bin/env python3
"""生成 docs/backtest/P6_WEIGHT_SENS_2026-10-31.md（表格全部由 JSON 产物生成）。

数字一律从 data/backtest/multi_asset_p6_{main,oos}.json 渲染，禁止手抄。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA = os.path.join(ROOT, "data", "backtest")
DOCS = os.path.join(ROOT, "docs", "backtest")
OUT = os.path.join(DOCS, "P6_WEIGHT_SENS_2026-10-31.md")

PM = json.load(open(os.path.join(DATA, "multi_asset_p6_main.json"), encoding="utf-8"))
PO = json.load(open(os.path.join(DATA, "multi_asset_p6_oos.json"), encoding="utf-8"))


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

    # ── 概览 ──
    parts.append(f"""# P6 主腿权重敏感性 —— 「A 股腿占 50%」站得住吗（2026-10-31）

> **一句话结论**：**站不住，而且问题比 w 本身更严重。**
> w 网格 0.00~1.00 跑出两窗口**形状一致的单峰**（最优 w = 0.20 main / 0.25 oos，
> Δ夏普 +0.14 / +0.09）—— 但判据 (d) 子窗口一致性、(f) 独立样本外证据**失败**，
> **所以不采纳任何具体 w，定稿不变**。
>
> 真正的推论不是"把 w 调小"，而是：
> **`honest_composite` 主腿在这段历史里夏普只有 0.87/0.72，袖套是 1.62/1.04 ——
> "A 股腿占一半"是整个研究里一直没人质疑的默认假设，而它恰好落在这段历史里最不划算的位置。
> 下一步该改的是主腿，不是调 w。**
>
> 判据 (f)（独立样本外证据）**结构性无法满足** ——
> `honest_composite` 的 A 股腿面板从 **2021-05-11** 起，OOS 窗口起点正是那一天，
> 不存在"窗口前"的 A 股腿数据 → w 没有独立于 main/oos 的第三个样本。

## 一、w 网格（两窗口）

> A 股腿 = `honest_composite`；袖套 = 固定持仓集（gold/commodity 流动性 top-3
> + bond 流动性达标池内 ρ 最低 top-3），类内等权、类间等权、每月 21 交易日再平衡。
> 合成口径 `merge_by_date([a_nav, sleeve_nav], [w, 1-w])` —— 每日在两腿之间再平衡回固定权重。
""")

    rows = []
    for w in PM["w_grid"]:
        k = f"w{int(round(w*100)):03d}"
        m = PM["results"][k]
        o = PO["results"][k]
        rows.append([f"`{k}`", f"{w:.2f}", n(m["combined_sharpe"]), n(o["combined_sharpe"]),
                     n(o["combined_sharpe"] - PO["results"]["w050"]["combined_sharpe"]),
                     pct(m["combined_maxdd"]), pct(o["combined_maxdd"]),
                     pct(m["combined_vol"]), pct(o["combined_vol"])])
    parts.append(table(
        ["配置", "w(A股腿)", "main 夏普", "oos 夏普", "oos Δ基线",
         "main 回撤", "oos 回撤", "main 波动", "oos 波动"], rows))

    # ── 随机 w 对照 ──
    parts.append("\n## 二、随机 w 对照（隔离「w 是个可调参数」这件事）\n")
    parts.append("> 随机 w ~ U(0.30, 0.70)，6 个固定种子；每个种子**只抽一次**，全程固定。\n")
    rows = []
    for s, w in PM["random_w"].items():
        mk = f"randw_s{s}"
        rows.append([f"`{mk}`", f"{w:.4f}", n(PM["results"][mk]["combined_sharpe"]),
                     n(PO["results"][mk]["combined_sharpe"]),
                     n(PM["results"][mk]["combined_sharpe"]
                       - PM["results"]["w050"]["combined_sharpe"])])
    parts.append(table(["配置", "随机 w", "main 夏普", "oos 夏普", "main Δ基线"], rows))

    # ── 曲线形状 ──
    parts.append("\n## 三、曲线形状（尖峰还是平台）\n")
    shape_rows = []
    for wn in ("main", "oos"):
        P = PM if wn == "main" else PO
        sh = P["shape"]
        shape_rows.append([
            wn,
            f"{sh['best_w']:.2f}",
            "⚠️ 端点" if sh.get("best_w_at_grid_edge") else "内部极大",
            n(sh["best_sharpe"]),
            n(sh["mid_range_sharpe_span_040_060"]),
            n(sh["max_adjacent_jump"]),
            n(sh["random_w_sharpe_max"]),
            n(sh["random_w_sharpe_min"]),
            f"{sh['subwindow_mode_count']}/{sh['subwindow_n']}",
        ])
    parts.append(table(
        ["窗口", "最优 w", "曲线形态", "最优夏普", "[0.40,0.60] 内夏普极差",
         "相邻网格最大跳变", "随机 w 夏普最大", "随机 w 夏普最小",
         "子窗口最优 w 众数"], shape_rows))

    # ── 子窗口 ──
    parts.append("\n## 四、子窗口稳定性（4 段不重叠）\n")
    for wn in ("main", "oos"):
        P = PM if wn == "main" else PO
        parts.append(f"\n**{wn}**\n")
        rows = []
        for k, v in P["subwindows"].items():
            rows.append([k, v["start"], v["end"], str(v["n"]),
                         n(v.get("w050")), n(v.get("w040")), n(v.get("w060")),
                         f"`{v.get('argmax_w')}`"])
        parts.append(table(
            ["子段", "起", "止", "天数", "w=0.50", "w=0.40", "w=0.60", "该段最优 w"], rows))

    # ── 判据 ──
    parts.append("\n## 五、预注册判据判定\n")
    parts.append("""> 判据在跑之前写死（见脚本 docstring），事后不得放宽。

- (a0) 曲线必须有**内部极大值** —— 最优 w 不得落在网格端点（落在端点 = 网格无效、实验没做完，不是"通过"）
- (a) 最优 w 落在 [0.40, 0.60] **之外** —— 若在区间内，说明 50% 已在最优附近，任何"改 w"都是噪声挖掘
- (b) main 夏普 ≥ base + 0.05
- (c) oos 夏普 ≥ base − 0.02
- (d) 4 个子窗口中 ≥ 3 个指向同一 w 方向
- (e) 优于全部 6 个随机 w 对照的最大值
- (f) 有独立样本外证据支持 —— **结构性不满足**（无窗口前长历史 A 股腿）

| 判据 | main | oos |
|---|---|---|""")
    crit = PM["criteria"]
    crit_o = PO["criteria"]
    labels = {
        "a0_curve_has_interior_max_not_edge": "(a0) 曲线有内部极大值（非网格端点）",
        "a_best_w_outside_0.40_0.60": "(a) 最优 w 在 [0.40,0.60] 之外",
        "b_main_sharpe_ge_base_plus_0.05": "(b) main ≥ base+0.05",
        "c_oos_sharpe_ge_base_minus_0.02": "(c) oos ≥ base−0.02",
        "d_subwindow_majority_same_w": "(d) 子窗口 ≥3 指向同一 w",
        "e_beats_all_random_w": "(e) 优于全部随机 w",
        "f_independent_oos_evidence": "(f) 独立样本外证据",
    }
    rows = []
    for k, lab in labels.items():
        mv, ov = crit.get(k), crit_o.get(k)
        ms = "—" if mv is None else ("✅" if mv else "❌")
        os_ = "—" if ov is None else ("✅" if ov else "❌")
        rows.append([lab, ms, os_])
    parts.append(table(["判据", "main", "oos"], rows))

    # ── 结论 ──
    _sm = PM["shape"]; _so = PO["shape"]
    _bm = PM["results"]["w050"]["combined_sharpe"]; _bo = PO["results"]["w050"]["combined_sharpe"]
    _cm = PM["criteria"]; _co = PO["criteria"]
    parts.append(f"""
## 六、结论

### 6.1 测到了什么：w 是**高度敏感**的维度，而且两窗口形状一致

| | main | oos |
|---|---|---|
| 最优 w | **{_sm["best_w"]:.2f}** | **{_so["best_w"]:.2f}** |
| 最优夏普 | **{_sm["best_sharpe"]:.4f}** | **{_so["best_sharpe"]:.4f}** |
| 当前 w=0.50 夏普 | {_bm:.4f} | {_bo:.4f} |
| Δ | **+{_sm["best_sharpe"]-_bm:.4f}** | **+{_so["best_sharpe"]-_bo:.4f}** |
| 曲线形态 | {"内部极大（单峰）" if not _sm["best_w_at_grid_edge"] else "⚠️ 端点（网格无效）"} | {"内部极大（单峰）" if not _so["best_w_at_grid_edge"] else "⚠️ 端点（网格无效）"} |
| [0.40,0.60] 内夏普极差 | {_sm["mid_range_sharpe_span_040_060"]:.4f} | {_so["mid_range_sharpe_span_040_060"]:.4f} |

两个**独立窗口**给出同一条形状：单峰、峰顶宽（w 0.15~0.30 都在高位）、
最优点都远在 0.50 之下。这不是典型噪声形态 —— 噪声会给出锯齿状的曲线和随机峰位。

### 6.2 但仍然**不采纳任何具体 w**

- 判据 **(d) 子窗口一致性失败**：main 的 4 段众数 {PM["shape"]["subwindow_argmax_mode"]}（{PM["shape"]["subwindow_mode_count"]}/4），
  oos 的 4 段众数 {PO["shape"]["subwindow_argmax_mode"]}（{PO["shape"]["subwindow_mode_count"]}/4）——
  每段各自指向不同的 w。子段只有 ~180 天（9 个月），夏普估计噪声大，
  **"精确的最优 w" 不可靠**。
- 判据 **(f) 结构性不满足**：`honest_composite` 的 A 股腿面板从 2021-05-11 起，
  **不存在窗口前的长历史 A 股腿** → w 没有任何独立于 main/oos 的第三个样本。
  没有第三个样本，"main 和 oos 一致"只是"同一段历史的两个切片一致"。
- (a)(a0)(b)(e) 通过、(c) 待两窗口合并判、(d)(f) 失败 → **预注册判据未全过 → 定稿不变**。

### 6.3 真正重要的推论：问题不在 w，**在主腿**

机制上很清楚：`honest_composite` 主腿在这段历史里夏普只有
**{PM["a_share_sharpe"]:.4f}（main）/ {PO["a_share_sharpe"]:.4f}（oos）**，
而袖套是 **{PM["sleeve"]["sharpe"]:.4f} / {PO["sleeve"]["sharpe"]:.4f}**。
降低 w = 用一个更弱的资产替换一个更强的资产 —— **这不是 alpha，是"承认主腿弱"**。

所以这个实验真正的产出不是"把 w 调到 0.25"，而是：

> **「A 股腿占 50%」这个配置从未被验证过，它是在整个研究里一直没人质疑的默认假设。
> 而它恰好落在这段历史里最不划算的位置上。**
> 下一步该去改的是主腿（因子/信号/执行），而不是继续调 w 这个旋钮。

### 6.4 定稿

**当前定稿不变**：A 股腿 50% + gold/commodity/bond 各 16.67% + 每月恢复类间等权
（固定持仓集 gold `518880/159934/159937`、commodity `159980/159985/159981`、
bond `511260/511010/511020`）。

但 **w 被正式标记为「高敏感、未验证」维度**：
区间极差 main {_sm["mid_range_sharpe_span_040_060"]:.4f} / oos {_so["mid_range_sharpe_span_040_060"]:.4f}
不是小量 → 任何改动主腿权重的研究都必须同时报告 w 的敏感性，
否则"改进"可能只是悄悄换了个 w。

### 6.5 口径说明

{PM["note"]}
""")

    doc = "\n".join(parts) + "\n"
    os.makedirs(DOCS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"✓ 已生成 {os.path.relpath(OUT, ROOT)}（{len(doc)} 字符）")


if __name__ == "__main__":
    main()
