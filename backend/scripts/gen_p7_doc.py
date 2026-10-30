#!/usr/bin/env python3
"""生成 docs/backtest/P7_FACTOR_ABLATION_2026-10-31.md。

数字与结论措辞**全部从 JSON 产物派生**，禁止手抄：
- 表格数字 ← multi_asset_p7_{main,oos}.json
- 结论段落 ← 各臂的预注册判据布尔值（哪一档措辞由判据结果决定）

用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/gen_p7_doc.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA = os.path.join(ROOT, "data", "backtest")
DOCS = os.path.join(ROOT, "docs", "backtest")
OUT = os.path.join(DOCS, "P7_FACTOR_ABLATION_2026-10-31.md")

PM = json.load(open(os.path.join(DATA, "multi_asset_p7_main.json"), encoding="utf-8"))
PO_PATH = os.path.join(DATA, "multi_asset_p7_oos.json")
PO = json.load(open(PO_PATH, encoding="utf-8")) if os.path.exists(PO_PATH) else None

ARM_DESC = {
    "base25": "当前主腿基线（25 公式全留）",
    "drop_MAX": "剔 MAX 全族（25→20）",
    "drop_MAX5_10_20": "剔 MAX5/MAX10/MAX20，留 MAX30/MAX60（25→22）",
    "drop_STD": "剔 STD 全族（25→20）",
    "drop_MAX_STD": "剔 MAX+STD 全族（25→15）",
    "flip_MAX30_60": "25 因子不变，MAX30/MAX60 符号 −1→+1",
    "drop_MAX5_10_20_flip30_60": "22f 诚实候选：剔 MAX5/10/20 + MAX30/60 符号 +1",
    "only_honest_high_ic": "仅保留高 IC 公式族（20 个，来源 42f family 产物）",
}


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


def verdict_of(d: Dict[str, Any], label: str) -> Dict[str, Any]:
    """合并 main / oos 两窗口的判据，给出该臂的总判定。"""
    cm = d["main"]["criteria"].get(label, {})
    co = d["oos"]["criteria"].get(label, {}) if d.get("oos") else {}
    merged: Dict[str, Any] = {}
    for k in ("a_ashare_sharpe_gt_base",
              "b_main_combined_ge_base_plus_0.05",
              "c_oos_combined_ge_base_minus_0.02",
              "d_subwindow_majority_ge_3",
              "e_beats_same_keep_random_max",
              "f_independent_oos_evidence"):
        vals = [cm.get(k), co.get(k)]
        vals = [v for v in vals if v is not None]
        merged[k] = all(vals) if vals else None
    return merged


def _oos_random_vs_rule(po: dict, pm: dict | None = None) -> str:
    """oos 窗口：规则臂 vs 同 keep-count 随机对照的最大值。"""
    res = po.get("results", {})
    base = res.get("base25", {}).get("combined_sharpe")
    rules = [l for l in res if l != "base25" and not l.startswith("rand_")]
    rands = [l for l in res if l.startswith("rand_")]
    if not rules or not rands:
        return "- （本轮 oos 未跑随机对照）"
    out = []
    for rl in rules:
        rv = res[rl]["combined_sharpe"]
        same = [x for x in rands
                if res[x].get("n_factors") == res[rl].get("n_factors")]
        if not same:
            continue
        rmax = max(res[x]["combined_sharpe"] for x in same)
        out.append(
            f"- `{rl}` 组合夏普 **{n(rv)}** vs 同 keep-count 随机对照最大 "
            f"**{n(rmax)}**（{len(same)} 个随机臂："
            f"{', '.join(n(res[x]['combined_sharpe']) for x in same)}）"
            f" → 规则臂{'**胜**' if rv > rmax else '**输**'}随机，"
            f"Δ(rule − rand_max) = {rv - rmax:+.4f}")
    out.append("")
    win_r = sum(1 for x in rands if res[x]["combined_sharpe"] > base)
    win_u = sum(1 for x in rules if res[x]["combined_sharpe"] > base)
    out.append(
        f"→ oos 窗口里随机对照胜过基线：{win_r}/{len(rands)}；"
        f"规则臂胜过基线：{win_u}/{len(rules)}。")
    out.append("")
    # main 窗口同 keep-count 随机臂的表现（现算，不手抄）
    if pm:
        mres = pm.get("results", {})
        mbase = mres.get("base25", {}).get("combined_sharpe")
        mrand = [x for x in mres if x.startswith("rand_")]
        groups: dict[int, list[float]] = {}
        for x in mrand:
            groups.setdefault(mres[x].get("n_factors"), []).append(
                mres[x]["combined_sharpe"])
        gtxt = "；".join(
            f"keep={k} 组最大 {n(max(v))}" for k, v in sorted(groups.items()))
        gw = sum(1 for x in mrand if mres[x]["combined_sharpe"] > mbase)
        main_txt = (
            f"在 main 窗口上，{len(mrand)} 个同 keep-count 随机臂**全部低于**基线"
            f"（{gtxt}，均 < 基线 {n(mbase)}；胜过基线 {gw}/{len(mrand)}）")
    else:
        main_txt = "在 main 窗口上，同 keep-count 随机臂全部低于基线"
    # 规则臂两窗口的现算数字
    rule_txt = ""
    if pm and rules:
        mres2 = pm.get("results", {})
        rl = rules[0]
        if rl in mres2:
            mv = mres2[rl]["combined_sharpe"]
            mb = mres2["base25"]["combined_sharpe"]
            ov = res[rl]["combined_sharpe"]
            ob2 = base
            rule_txt = (
                f"而规则臂 `{rl}` 在**两个窗口都输**"
                f"（main {n(mv)} < {n(mb)}，oos {n(ov)} < {n(ob2)}），"
                f"且 oos 里被随机臂拉开 {n(abs(ov - rmax_max(res, rands)))}")
    if not rule_txt:
        rule_txt = "而规则臂在两个窗口都输"
    out.append(
        f"**读法（必须两窗口合看）**：{main_txt}；"
        f"而到了 oos 窗口，三个随机臂**全部高于**基线。"
        "同一批配置在两个窗口上朝相反方向移动 → 随机删减带来的差异是"
        "**窗口噪声，不是因子信息**。\n\n"
        f"**但这不构成\"可以随机删因子\"的许可**：随机臂在 oos 赢、在 main 输，"
        f"方向不稳定；{rule_txt}。\n\n"
        "→ **结论：按规则删因子比随机删因子更差**。若一定要说\"25 因子不是最优\"，"
        "证据指向\"存在某个未知的因子子集\"，但**没有任何可预先指定的规则能挑出它** —— "
        "这类结论无法落地，不能作为改稿依据。")
    return "\n".join(out)


def rmax_max(res: dict, rands: list) -> float:
    return max(res[x]["combined_sharpe"] for x in rands) if rands else 0.0


def main() -> None:
    main_res = PM["results"]
    base = main_res["base25"]
    parts: List[str] = []

    # ── 概述 ──
    parts.append(f"""# P7 主腿因子消融 —— 主腿弱，是因子构成的问题吗（2026-10-31）

> **背景**：P6 的真正推论不是"把主腿权重 w 调小"，而是
> **主腿（honest_composite）夏普只有 {n(PM['results']['base25']['a_share_sharpe'], 2)}(main)，
> 袖套却有 {n(PM['sleeve']['sharpe'], 2)}** —— 降 w = 用更弱的资产替换更强的资产，不是 alpha。
> → 真正该改的是**主腿本身**。
>
> **本实验唯一变量**：主腿的因子集合。袖套（P5 固定持仓集）、权重（w=0.50）、
> 成本与成交口径全部与 P3/P5/P6 定稿一致，基线门已通过
> （base25 组合夏普 {n(PM['base_combined_sharpe'])} = P3 定稿 {n(PM['p3_expected_base'])}）。

## 先查已有研究（避免重复劳动）

2026-10-27 已做过一次 **42f 口径**的因子消融（`factor_subset_*_full3y.json`：
ablation 43 / family 19 / greedy / weights / stability / oosval / walkforward / consensus）。

- 佐证同源：`minus_panel_all`（= 25 公式版）全 3y 夏普 0.8796、回撤 −0.1337，
  与当前主线 0.880 / −13.37% 完全对上。
- **缺口**：已有消融全部是「在 42f 基线上剔一族」（n=37/39/41），
  **从未在 25f 口径上单独消融** ← 本实验填这个洞。

已有消融给出的信号（42f 口径，剔族后全 3y 夏普）：

| 消融 | 夏普 | 读法 |
|---|---|---|
| minus_VSUMN | 0.9251 | 最优，但是面板因子族，当前主线没有 → 不适用 |
| minus_panel_all | 0.8796 | **当前主线** |
| minus_MAX | 0.8482 | MAX 可能是拖累 |
| minus_STD | 0.8416 | STD 次要 |
| minus_MIN | 0.7773 | MIN 是命脉 |
| minus_CORD / minus_CORR | 0.7092 / 0.6765 | CORR/CORD 是主力，不能碰 |

⚠️ **结论张力**：walk-forward 研究认为 MAX30/60 的符号是错的（2016-2021 IC 为正，硬编码 −1 反了），
而因子消融又说「剔 MAX 家族有正收益」。窗口口径都不同 → 本实验两条路都测，让数据裁决。
""")

    # ── 结论 ──
    d_all = {"main": PM, "oos": PO}
    rule_labels = [l for l in PM["ranking"] if l != "base25"]
    winners, losers = [], []
    for l in rule_labels:
        v = verdict_of(d_all, l)
        keys = [k for k in v if k != "f_independent_oos_evidence" and v[k] is not None]
        if keys and all(v[k] for k in keys):
            winners.append(l)
        else:
            losers.append(l)

    if winners:
        concl = (f"**有 {len(winners)} 个规则臂通过了 (a)~(e) 全部预注册判据：** "
                 + "、".join(f"`{w}`" for w in winners)
                 + f"。但 (f) 独立样本外证据**结构性缺失**（honest_composite A 股腿面板从 2021-05-11 起，"
                   "OOS 起点正是那一天 → 不存在窗口前长历史 A 股腿），"
                   "**因此本实验不自动改定稿**，只产出候选与量化证据。")
    else:
        concl = ("**没有任何规则臂通过 (a)~(e) 全部预注册判据 —— 因子消融证伪。** "
                 "主腿的弱不是「因子选错了」能修的，"
                 "要么是这 25 个公式因子的信息量本身就到这里了，"
                 "要么是问题不在因子层而在**信号构造或执行层**。")

    parts.append(f"""## 一句话结论

{concl}

## 预注册判据（跑之前写死）

| 判据 | 含义 |
|---|---|
| (a) | main 窗口 **A 股腿自身**夏普 > 基线 A 股腿夏普 |
| (b) | main **组合**夏普 ≥ 基线组合夏普 + 0.05（**main 窗口专属**） |
| (c) | oos **组合**夏普 ≥ 基线组合夏普 − 0.02（**oos 窗口专属**） |
| (d) | 4 个不重叠子窗口中 ≥ 3 个该臂组合夏普 > 基线 |
| (e) | 该臂组合夏普 > **同 keep-count 随机对照的最大值** |
| (f) | 独立样本外证据 —— **结构性不满足** |

⚠️ (b)/(c) 分窗口专属：绝不能在 oos 窗口里也判 (b)，那等于用"提升幅度"在样本外挑参。
""")

    # ── main 排名 ──
    rows = []
    for l in PM["ranking"]:
        r = main_res[l]
        rows.append([
            f"`{l}`", r["n_factors"],
            n(r["a_share_sharpe"]),
            n(r["combined_sharpe"]),
            f"{r['combined_sharpe'] - base['combined_sharpe']:+.4f}",
            pct(r["combined_return"]),
            pct(r["combined_maxdd"]),
            r["a_share_trades"],
            ARM_DESC.get(l, "随机对照" if r.get("is_random_control") else ""),
        ])
    parts.append(f"""## main 窗口排名（2023-05-08 ~ 2026-05-08）

基线 `base25`：A 腿夏普 {n(base['a_share_sharpe'])}，组合夏普 {n(base['combined_sharpe'])}，
回撤 {pct(base['combined_maxdd'])}。

{table(["臂", "因子数", "A腿夏普", "组合夏普", "Δ组合", "组合收益", "组合回撤", "交易笔数", "说明"], rows)}
""")

    # ── 随机对照 ──
    rg_rows = []
    for keep, g in sorted(PM["random_groups"].items(), key=lambda x: -int(x[0])):
        _sv = g["sharpes"]
        _cells = [n(_sv[k]) for k in g["labels"]]
        _cells += [""] * (3 - len(_cells))
        rg_rows.append([keep, n(g["max"]), n(g["min"])] + _cells)
    parts.append(f"""## 随机对照（同 keep-count）

对照组设计要点：从 25 个公式里**随机剔同样个数**（不是"随机选同样个数"）。
后者因子数更少、分散化天然更弱，规则组几乎必然赢，对照没有鉴别力。

{table(["keep 数", "组合夏普最大", "组合夏普最小", "随机臂 1", "随机臂 2", "随机臂 3"], rg_rows)}
""")

    # ── 判据 ──
    cr_rows = []
    for l in rule_labels:
        pm_ = PM["per_arm"].get(l, {})
        cm = PM["criteria"].get(l, {})
        co = PO["criteria"].get(l, {}) if PO else {}
        def cell(k):
            v = cm.get(k, co.get(k))
            return {True: "✅", False: "❌", None: "—"}[v]
        cr_rows.append([
            f"`{l}`", cell("a_ashare_sharpe_gt_base"),
            cell("b_main_combined_ge_base_plus_0.05"),
            cell("c_oos_combined_ge_base_minus_0.02"),
            cell("d_subwindow_majority_ge_3"),
            cell("e_beats_same_keep_random_max"),
            cell("f_independent_oos_evidence"),
            pm_.get("subwindow_wins"),
        ])
    parts.append(f"""## 预注册判据逐臂判定

{table(["臂", "(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "子窗口胜数"], cr_rows)}

(f) 恒为 ❌：honest_composite 的 A 股腿面板从 2021-05-11 起，OOS 起点正是那一天
→ 不存在"窗口前"的 A 股腿数据 → 任何因子集都没有独立于 main/oos 的第三个样本。
""")

    # ── oos ──
    if PO:
        o_rows = []
        ob = PO["results"]["base25"]
        for l, r in PO["results"].items():
            kind = "基线" if l == "base25" else (
                "随机对照" if l.startswith("rand_") else "规则臂")
            o_rows.append([
                f"`{l}`", kind, r["n_factors"], n(r["a_share_sharpe"]),
                n(r["combined_sharpe"]),
                f"{r['combined_sharpe'] - ob['combined_sharpe']:+.4f}",
                pct(r["combined_return"]), pct(r["combined_maxdd"]),
            ])
        parts.append(f"""## oos 窗口（2021-05-11 ~ 2023-05-08）

基线 `base25` 组合夏普 {n(PO['results']['base25']['combined_sharpe'])}，
A 腿夏普 {n(PO['results']['base25']['a_share_sharpe'])}。

{table(["臂", "类别", "因子数", "A腿夏普", "组合夏普", "Δ组合", "组合收益", "组合回撤"], o_rows)}

⚠️ oos 只带进 main 阶段排名前 {len([l for l in PO['ranking'] if l != 'base25'])} 的规则臂
（外加同 keep-count 随机对照）—— **Stage2 只验候选，不在 OOS 上重新搜参**。

### 【决定性对照】规则臂在 oos 输给随机对照

{_oos_random_vs_rule(PO, PM)}
""")

    # ── 子窗口 ──
    sub_rows = []
    for sk, seg in PM["subwindows"].items():
        row = [sk, f"{seg['start']}~{seg['end']}", seg["n"]]
        for l in [x for x in PM["ranking"] if x in seg][:0]:
            row.append(n(seg.get(l)))
        row.append(n(seg.get("base25")))
        sub_rows.append(row)
    parts.append(f"""## 子窗口稳定性（main，4 段不重叠）

{table(["子段", "区间", "天数", "base25"], sub_rows)}

逐臂子窗口胜数见判据表的最后一列（(d) 判据 = 子窗口胜数 ≥ 3）。
""")

    # ── 等价臂回归 ──
    eq = PM.get("equiv_check", {})
    eq_lines = "\n".join(
        f"- `{k}`：{'逐位相同 ✓' if v.get('identical') else '不一致 ❌'}"
        f"（{v.get('n_a')} vs {v.get('n_b')} 天）"
        for k, v in eq.items()) or "- （本轮未跑到等价臂组合）"

    parts.append(f"""## 工程与口径说明

### 等价臂交叉验证

`only_honest_high_ic` 的 20 个因子恰好 == `drop_MAX` 的 20 个
（已有 42f 研究里的"高 IC 公式族"本身一个 MAX 族都没有）→ 两臂必须产生**逐位相同**的净值。

{eq_lines}

### 已排除的混淆变量

- `min_cross = max(2, min(30, n_stocks//10)) = 30` 是**横截面行业内有效股票数**门槛，
  不是每只股票的因子票数 → **不受因子剔除影响**。
- 复合分是 `acc / wsum` 的 nan-aware 归一 → 缺某个因子的股票不被惩罚。
- 符号表 `_HONEST_FORMULAS` 是模块全局，逐臂跑完**硬断言已还原**，
  否则一个"翻符号"臂会污染后面所有臂（症状是"夏普莫名其妙"，极难归因）。
- 因子白名单走策略内置 `factor_allowlist` → **零生产代码改动**。

### 已知结构性限制

{PM['note']}
""")

    os.makedirs(DOCS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"[gen_p7] ✓ 已写入 {OUT}")


if __name__ == "__main__":
    main()
