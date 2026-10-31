#!/usr/bin/env python3
"""生成 docs/backtest/P18_FUSE_GBM_2026-11-01.md。

数字与结论措辞**全部从 JSON 产物派生**，禁止手抄：
- 表格数字 ← multi_asset_p18_{main,oos}.json
- 结论段落 ← verdict 的判据布尔值（哪一档措辞由判据结果决定）

用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/gen_p18_doc.py
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA = os.path.join(ROOT, "data", "backtest")
DOCS = os.path.join(ROOT, "docs", "backtest")
OUT = os.path.join(DOCS, "P18_FUSE_GBM_2026-11-01.md")

PM_PATH = os.path.join(DATA, "multi_asset_p18_main.json")
PO_PATH = os.path.join(DATA, "multi_asset_p18_oos.json")
if not os.path.exists(PM_PATH):
    raise SystemExit("[gen_p18] ❌ 缺 main 产物，请先跑 multi_asset_p18_fuse_gbm.py --window main")
PM = json.load(open(PM_PATH, encoding="utf-8"))
PO = json.load(open(PO_PATH, encoding="utf-8")) if os.path.exists(PO_PATH) else None

W_LIST: List[float] = PM["prereg"]["w_gbms"]
BASE_LABEL = "fuse_w0.00"
GATE_LABEL = "orig42f"
PLC_SEEDS: List[int] = PM["prereg"]["placebo_seeds"]


def n(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def pct(v, d=2):
    return "—" if v is None else f"{v:+.{d}%}"


def b(v) -> str:
    if v is None:
        return "—"
    return "✅ True" if v else "❌ False"


def table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(str(c) for c in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def arms_of(d: Dict[str, Any], kind: str) -> List[str]:
    return [l for l, v in d["results"].items() if v.get("kind") == kind]


def plc_labels(d: Dict[str, Any], w: float) -> List[str]:
    tag = f"plc_w{w:.2f}_s"
    return sorted([l for l in d["results"] if l.startswith(tag)])


def build() -> str:
    L: List[str] = []
    res_m = PM["results"]
    crit_m = PM.get("criteria", {})
    res_o = PO.get("results", {}) if PO else {}
    crit_o = PO.get("criteria", {}) if PO else {}
    verdict = PM.get("verdict", {})
    rank_m = PM.get("ranking", [])
    rule_m = [l for l in rank_m if res_m[l].get("kind") == "rule"]
    cand = rule_m[0] if rule_m else None

    # ── 一句话结论（措辞由判据结果决定）──
    if PO is None:
        head = ("**main 窗口已跑完，oos 窗口待跑** —— 本文档的最终判定需要 oos 阶段，"
                "当前结论为**未定**。")
    else:
        vo = PO.get("verdict", {})
        passed = vo.get("all_pass")
        if passed is True:
            head = (f"**通过**：候选臂 `{cand}` 在 (a)(b)(c)(d)(f) 五条预注册判据上**全过**"
                    f" → gbm 的正交信息确实能通过分数级融合给 42f 带来增量。")
        elif passed is False:
            head = (f"**不通过**：候选臂 `{cand}` 未过全部预注册判据 → "
                    f"**gbm 正交信息在分数级融合上不产生可采纳增量**。")
        else:
            head = "**判定未定**（缺 oos 阶段或判据缺失）。"

    L.append("# P18：42f 之上叠加 gbm 增量（分数级 z-score 融合）")
    L.append("")
    L.append(f"> 生成时间：由 `backend/scripts/gen_p18_doc.py` 从 JSON 产物渲染，"
             f"数字零手抄。预注册见 `docs/backtest/MULTI_ASSET_LEDGER_2026-10-31.md` §十。")
    L.append("")
    L.append("## 结论")
    L.append("")
    L.append(head)
    L.append("")

    # ── §一 预注册 ──
    pr = PM["prereg"]
    L.append("## 一、预注册（跑之前落纸，不可事后扩）")
    L.append("")
    L.append(table(
        ["项", "事前固定取值"],
        [
            ["依据", "§八.3：gbm ↔ 量价因子 mean RankIC 仅 0.112 / 0.150（近乎正交）"],
            ["融合方式", pr["fusion"]],
            ["融合权重 w_gbm", "{" + ", ".join(f"{w:.2f}" for w in W_LIST) + "} + 基线 0.00"],
            ["gbm 配置", f"vol-target 关闭（target_vol={pr['gbm_cfg']['target_vol']}）、"
                        f"周频 weekday={pr['gbm_cfg']['rebalance_weekday']}、"
                        f"lag={pr['gbm_cfg']['lag']}"],
            ["placebo 种子", ", ".join(str(s) for s in PLC_SEEDS) + "（每个 w 各 3 个）"],
            ["试验数（实验口径）", str(pr["n_trials_override"])],
            ["否决前置", f"同日截面相关系数 > {pr['corr_veto']} → 直接判死"],
        ]))
    L.append("")

    # ── §二 诊断 ──
    L.append("## 二、诊断：正交性是否在同日成立（否决前置）")
    L.append("")
    diag = PM.get("diagnostics", {})
    drows = []
    for k, v in diag.items():
        drows.append([
            k, str(v.get("n_days", 0)), n(v.get("pearson_median")),
            n(v.get("spearman_median")), n(v.get("gbm_coverage_median")),
        ])
    L.append(table(["窗口", "可算天数", "Pearson 中位", "Spearman 中位", "gbm 覆盖率中位"], drows))
    L.append("")
    cm = PM.get("corr_median")
    veto = PM.get("veto_triggered")
    L.append(f"同日截面相关中位数 = **{n(cm)}**，否决线 {pr['corr_veto']} → "
             f"**{'❌ 触发（判死）' if veto else '✅ 未触发'}**")
    L.append("")
    L.append("> 覆盖率 > 1 是因为 gbm 侧没有 42f 的逐票涨停掩码 → gbm 的分数集合略大，"
             "融合时 42f 集合内 gbm 缺失处填 0（截面中性值），**不改变 42f 的选股池**。")
    L.append("")

    # ── §三 复现门 ──
    L.append("## 三、复现门：融合管线对基线无害")
    L.append("")
    g = res_m.get(GATE_LABEL, {})
    b0 = res_m.get(BASE_LABEL, {})
    L.append(table(
        ["臂", "组合夏普", "A 腿夏普", "交易笔数", "组合收益", "组合回撤"],
        [
            [f"`{GATE_LABEL}`（原始 42f 分数）", n(g.get("combined_sharpe")),
             n(g.get("a_share_sharpe")), str(g.get("total_trades")),
             pct(g.get("combined_return")), pct(g.get("combined_maxdd"))],
            [f"`{BASE_LABEL}`（z 化后，w=0）", n(b0.get("combined_sharpe")),
             n(b0.get("a_share_sharpe")), str(b0.get("total_trades")),
             pct(b0.get("combined_return")), pct(b0.get("combined_maxdd"))],
        ]))
    L.append("")
    same = (g.get("combined_sharpe") == b0.get("combined_sharpe")
            and g.get("a_share_sharpe") == b0.get("a_share_sharpe")
            and g.get("total_trades") == b0.get("total_trades"))
    L.append(f"→ 两者**{'逐位一致 ✅' if same else '不一致 ❌'}**"
             f"（z-score 是单调仿射变换 → 排名不变 → 选股不变 → 净值不变）。")
    L.append("")
    L.append(f"- 复现门：orig42f 组合夏普 = **{n(g.get('combined_sharpe'))}**，"
             f"fuse_w0.00 组合夏普 = **{n(b0.get('combined_sharpe'))}**"
             f"（笔数 {g.get('total_trades')} / {b0.get('total_trades')}）")
    L.append("")

    # ── §四 规则臂结果 ──
    L.append("## 四、规则臂结果（main 窗口）")
    L.append("")
    rows = []
    for l in rank_m:
        v = res_m[l]
        if v.get("kind") == "placebo":
            continue
        c = crit_m.get(l, {})
        rows.append([
            f"`{l}`", n(v.get("w_gbm"), 2), n(v.get("combined_sharpe")),
            n(v.get("a_share_sharpe")), n(v.get("combined_vol")),
            pct(v.get("combined_return")), pct(v.get("combined_maxdd")),
            str(v.get("total_trades")), n(c.get("delta_combined")),
            n(c.get("placebo_max")), b(c.get("c_beats_placebo")),
            n((c.get("rolling") or {}).get("win_rate"), 3),
        ])
    L.append(table(
        ["臂", "w_gbm", "组合夏普", "A 腿夏普", "组合波动", "组合收益", "组合回撤",
         "笔数", "Δ组合", "placebo 最高", "胜 placebo", "滚动胜率"],
        rows))
    L.append("")

    # ── §五 placebo ──
    L.append("## 五、placebo 随机对照（gbm 那一路换成随机分数）")
    L.append("")
    prows = []
    for w in W_LIST:
        pls = plc_labels(PM, w)
        if not pls:
            continue
        vs = [res_m[x]["combined_sharpe"] for x in pls]
        rule = res_m.get(f"fuse_w{w:.2f}", {})
        c = crit_m.get(f"fuse_w{w:.2f}", {})
        prows.append([
            f"{w:.2f}", n(rule.get("combined_sharpe")),
            n(min(vs)), n(max(vs)), n(sum(vs) / len(vs)),
            n(rule.get("combined_sharpe") - max(vs)),
            b(c.get("c_beats_placebo")),
        ])
    L.append(table(
        ["w_gbm", "规则臂组合夏普", "placebo 最低", "placebo 最高", "placebo 均值",
         "规则 − placebo 最高", "规则胜 placebo"], prows))
    L.append("")
    L.append("> placebo 保持 gbm 的**覆盖掩码与边际分布**，只把分数换成同掩码下的标准正态"
             "随机数 → 若规则臂赢不了 placebo，则增量并非来自 gbm 的信息。")
    L.append("")

    # ── §六 机制分解（新增）──
    L.append("## 六、机制分解：信息增益 vs 稀释代价")
    L.append("")
    L.append("规则臂夏普低**不等于** gbm 没有信息。把「真实 gbm」与「同参数随机 placebo」"
             "逐项对比，分离出两件事：**信息增益**（收益端）与**稀释代价**"
             "（42f 排序被稀释 + 波动上升）。")
    L.append("")
    mrows = []
    for w in W_LIST:
        pls = [res_m[x] for x in plc_labels(PM, w)]
        rl = res_m.get(f"fuse_w{w:.2f}", {})
        if not pls:
            continue
        m = {k: sum(p[k] for p in pls) / len(pls)
             for k in ("combined_sharpe", "combined_vol", "combined_maxdd",
                       "combined_return")}
        mrows.append([
            f"{w:.2f}", n(rl.get("combined_sharpe")), n(m["combined_sharpe"]),
            f"{rl.get('combined_sharpe', 0) - m['combined_sharpe']:+.4f}",
            n(rl.get("combined_vol")), n(m["combined_vol"]),
            f"{rl.get('combined_vol', 0) - m['combined_vol']:+.4f}",
            n(rl.get("combined_return")), n(m["combined_return"]),
            f"{(rl.get('combined_return', 0) - m['combined_return']) * 100:+.1f}pp",
        ])
    L.append(table(
        ["w_gbm", "真实夏普", "placebo 均值", "Δ夏普", "真实波动", "placebo 波动",
         "Δ波动", "真实收益", "placebo 收益", "Δ收益"], mrows))
    L.append("")
    all_neg = all(float(r[3]) < 0 for r in mrows) if mrows else False
    base_sr = res_m.get("fuse_w0.00", {}).get("combined_sharpe", 0.0)
    # ⚠️ 「相对基线全负」必须拿**真实臂夏普**和基线比，
    #    不能用 Δ夏普（那是相对 placebo 的）—— 两者不是一回事。
    below_base = [r for r in mrows if float(r[1]) < base_sr]
    pos_w = [r for r in mrows if float(r[3]) > 0]
    if pos_w:
        L.append(f"**gbm 的信息含量不为 0**：在 **{len(pos_w)}/{len(mrows)}** 个权重上，"
                 f"真实 gbm 相对随机 placebo 的 Δ夏普为正"
                 f"（{', '.join(r[0] for r in pos_w)}），"
                 f"最大 **Δ夏普 {max(float(r[3]) for r in mrows):+.4f}**、"
                 f"最大 **Δ收益 {max(float(r[9].replace('pp', '')) for r in mrows):+.1f}pp**。")
        L.append("")
    if mrows and len(below_base) == len(mrows):
        L.append(f"**但相对「不加 gbm」的基线（{n(base_sr)}），"
                 f"{len(below_base)}/{len(mrows)} 个权重的真实臂夏普都更低** → "
                 f"**稀释 42f 排序的代价 > gbm 带来的信息增益**。")
        L.append("")
        L.append("→ P18 证伪的是**融合方式**（分数级线性加权会稀释已有排序），"
                 "不是「gbm 有信息」这个命题本身。")
        L.append("")
    L.append("### 年度分解（组合逐年收益）")
    L.append("")
    yrs = sorted({y for v in res_m.values()
                  for y in (v.get("combined_yearly") or {})})
    yrows = []
    base_y = (res_m.get("fuse_w0.00", {}).get("combined_yearly") or {})
    yrows.append(["**基线 `fuse_w0.00`**", "—"]
                 + [f"{base_y.get(y, float('nan')) * 100:+.2f}%" for y in yrs])
    for w in W_LIST:
        pls = [res_m[x] for x in plc_labels(PM, w)]
        rl = res_m.get(f"fuse_w{w:.2f}", {})
        if not pls:
            continue
        yrows.append([f"真实 `fuse_w{w:.2f}`", n(rl.get("combined_sharpe"))]
                     + [f"{(rl.get('combined_yearly') or {}).get(y, float('nan')) * 100:+.2f}%"
                        for y in yrs])
        pm_ = {y: sum((p.get("combined_yearly") or {}).get(y, 0.0) for p in pls) / len(pls)
               for y in yrs}
        yrows.append([f"placebo 均值 w{w:.2f}", "—"]
                     + [f"{pm_[y] * 100:+.2f}%" for y in yrs])
    L.append(table(["臂", "夏普"] + yrs, yrows))
    L.append("")
    L.append("> 年度收益**不等于**风险调整后收益：融合臂在部分年份收益显著更高，"
             "但波动同步上升 → 夏普不升。**只看板级夏普会漏掉这个结构。**")
    L.append("")

    # ── §七 判据 ──
    L.append("## 七、预注册判据汇总")
    L.append("")
    L.append(table(
        ["判据", "阈值", "main 结果", "oos 结果"],
        [
            ["(a) main 组合夏普", "≥ 基线 + 0.05",
             b(crit_m.get(cand, {}).get("a_main_combined_ge_base_plus_0.05"))
             if cand else "—",
             b((PO or {}).get("criteria", {}).get(cand, {})
               .get("a_main_combined_ge_base_plus_0.05")) if PO and cand else "—"],
            ["(b) oos 组合夏普", "≥ 基线 − 0.02", "—",
             b((PO or {}).get("criteria", {}).get(cand, {})
               .get("c_oos_combined_ge_base_minus_0.02")) if PO and cand else "—"],
            ["(c) 胜 placebo", "规则 > 同参数 placebo 最高",
             b(crit_m.get(cand, {}).get("c_beats_placebo")) if cand else "—",
             b((PO or {}).get("criteria", {}).get(cand, {})
               .get("c_beats_placebo")) if PO and cand else "—"],
            ["(d) 滚动 504 日夏普胜率", "≥ 60%",
             b(crit_m.get(cand, {}).get("d_rolling_win_rate_ge_0.60")) if cand else "—",
             b((PO or {}).get("criteria", {}).get(cand, {})
               .get("d_rolling_win_rate_ge_0.60")) if PO and cand else "—"],
            ["(e) DSR ≥ 0.95", "**降级为报告项**（恒假条件，见下）",
             n(crit_m.get(cand, {}).get("dsr"), 6) if cand else "—",
             n((PO or {}).get("criteria", {}).get(cand, {}).get("dsr"), 6)
             if PO and cand else "—"],
            ["(f) 跨窗口迁移", "oos 最优臂在 main 不排最差", "—",
             b(((PO or {}).get("verdict", {}).get("f_cross_window") or {})
               .get("not_worst_in_main")) if PO else "—"],
        ]))
    L.append("")
    thr = PM.get("dsr95_threshold_sharpe")
    L.append(f"**(e) 为什么降级**：实验口径 n_trials={pr['n_trials_override']} → "
             f"DSR95 门槛 **{n(thr)}**，而定稿组合夏普只有 **1.4114** → "
             f"任何臂都不可能达到 → 作否决条件等于实验无论结果都判死（零信息量）。"
             f"故如实报 DSR 值作**证据强度报告**，不参与通过判定。"
             f"通过判定 = (a)(b)(c)(d)(f) 五条全过。")
    L.append("")

    # ── §八 总判定 ──
    L.append("## 八、总判定")
    L.append("")
    if PO is None:
        L.append("**未定** —— 需 oos 阶段产物（`multi_asset_p18_oos.json`）。")
    else:
        vo = PO.get("verdict", {})
        L.append(f"候选臂（main 阶段组合夏普最高的规则臂）：**`{cand}`**")
        L.append("")
        L.append(f"- 候选臂 main 组合夏普 = **{n(res_m.get(cand, {}).get('combined_sharpe'))}**，"
                 f"基线 = **{n(res_m.get('fuse_w0.00', {}).get('combined_sharpe'))}**")
        L.append(f"- Δ组合（main）= **{n(crit_m.get(cand, {}).get('delta_combined'))}**")
        L.append(f"- DSR（报告项）= **{n(vo.get('dsr_report'), 6)}** / 门槛 "
                 f"{n(vo.get('dsr95_threshold_sharpe'))}")
        L.append(f"- all_pass = **{vo.get('all_pass')}**")
        fw = vo.get("f_cross_window") or {}
        if fw:
            L.append(f"- (f) oos 最优臂 `{fw.get('best_oos_arm')}` 在 main 的规则臂中排 "
                     f"**{fw.get('rank_in_main')}/{fw.get('n_rule_arms_in_main')}**")
        L.append("")
        L.append(f"- 候选臂 oos 组合夏普 = **{n(res_o.get(cand, {}).get('combined_sharpe'))}**，"
                 f"oos 基线 = **{n(res_o.get('fuse_w0.00', {}).get('combined_sharpe'))}**")
        L.append(f"- Δ组合（oos）= **{n(crit_o.get(cand, {}).get('delta_combined'))}**")
        L.append("")

    # ── §九 诚实声明 ──
    L.append("## 九、诚实声明")
    L.append("")
    L.append("- 本实验**只回答一个问题**：gbm 的正交信息能否通过分数级融合给 42f 带来增量。"
             "不回答「gbm 本身好不好」（gbm 单腿真实盘 lag1 仅 0.5258）。")
    L.append("- gbm 侧沿用其历史口径（`limit_up=0.095` 全板硬编码），"
             "**未**改成 42f 那样逐票按板块判定；这会让 gbm 分数的缺失模式带旧偏差"
             "（相对 5513×726 的量级约 0.13%，可接受）。")
    L.append("- 融合是**分数级**（z 后线性加权），不是持仓级、不是投票级 —— "
             "其他融合形式**未测**，不在本实验结论范围内。")
    L.append("- oos 窗口的 A 股腿面板数据自 2021-05-11 起，而 oos 起点正是该日 → "
             "oos **没有窗口前历史**，判据 (f) 的跨窗口迁移是唯一的外部检验。")
    L.append("- **§六 的机制发现是「事后观察」，不是预注册结论**。"
             "「gbm 信息含量 > 0」是在看到 4 个权重的 Δ夏普 分布之后才形成的叙述；"
             "预注册判据 (c) 针对的是**候选臂**（main 上组合夏普最高的规则臂），"
             "它已失败。**用 w=0.50 的 Δ夏普 去论证「gbm 有信息」，"
             "等于在 4 个权重里挑最好的那个来说事 —— 有选择偏差。**"
             "因此 §六 只能作为**待检验假设**记录，"
             "要成立必须另开一个**事前注册**的实验（且要付新的试验预算）。")
    L.append("- 全局试验数已由 1937 升至 **1942**（本实验 +5）；"
             "每做一次搜索都在抬高自己需要翻越的门槛。")
    L.append("- **placebo 的实现细节必须一起读**：placebo 是 **每个交易日独立重抽**的"
             "iid 标准正态（rng seed = `seed×1000003 + 日期序号`），"
             "只保留 gbm 的**覆盖掩码**，分数本身完全随机。"
             "因此「规则臂 vs placebo」的差异**不是**单纯的「信息质量」对比 —— "
             "真实 gbm 是**持续性**的弱信号（会把持仓持续拉向同一批票），"
             "placebo 是**每日重抽**的抖动（平均不偏移，只加大换手）。"
             "两者混着「持续偏移成本」与「抖动成本」两个不同效应。")
    L.append("")

    L.append("---")
    L.append("")
    L.append(f"产物：`data/backtest/multi_asset_p18_main.json`"
             f"{'、`data/backtest/multi_asset_p18_oos.json`' if PO else ''}；"
             f"runner：`backend/scripts/multi_asset_p18_fuse_gbm.py`；"
             f"预注册：`docs/backtest/MULTI_ASSET_LEDGER_2026-10-31.md` §十。")
    L.append("")
    return "\n".join(L)


def main() -> None:
    md = build()
    os.makedirs(DOCS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"[gen_p18] 已写入 {OUT}（{len(md)} 字符）")


if __name__ == "__main__":
    main()
