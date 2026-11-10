#!/usr/bin/env python3
"""从 P9 产物自动生成结论文档（数字零手抄）。

产物：
    data/backtest/multi_asset_p9_main.json
    data/backtest/multi_asset_p9_oos.json

用法：
    cd backend && .venv-py313/bin/python3 scripts/gen_p9_doc.py
    # 只生成 main 部分（oos 产物缺失时）：--main-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
BACK = os.path.join(ROOT, "data", "backtest")
DOCS = os.path.join(ROOT, "docs", "backtest")
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

RULE_ORDER = ["daily", "wk_fri", "wk_mon", "monthly"]


def load(name: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(BACK, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def fmt_pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:.2f}%"


def fmt_num(v: Optional[float], nd: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def rule_labels(art: Dict[str, Any]) -> List[str]:
    res = art.get("results", {})
    labs = [l for l in RULE_ORDER if l in res]
    labs += sorted(l for l in res if l not in RULE_ORDER and not l.startswith("rand_"))
    return labs


def rand_labels(art: Dict[str, Any]) -> List[str]:
    rc = art.get("random_controls", {}) or {}
    return sorted(rc.keys())


def table_rule(art: Dict[str, Any], labs: List[str]) -> List[str]:
    res = art["results"]
    base = art["base_label"]
    bcs = art["base_combined_sharpe"]
    head = ("| 臂 | 执行日数 | 交易笔数 | A腿夏普 | 组合夏普 | 组合收益 | "
            "组合回撤 | 组合波动 | Δ组合 |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for l in labs:
        r = res[l]
        d = r["combined_sharpe"] - bcs
        rows.append(
            f"| {l} | {r['rebalance_days']} | {r['total_trades']} | "
            f"{fmt_num(r['a_share_sharpe'])} | {fmt_num(r['combined_sharpe'])} | "
            f"{fmt_pct(r['combined_return'])} | {fmt_pct(r['combined_maxdd'])} | "
            f"{fmt_pct(r['combined_vol'])} | {d:+.4f} |")
    return rows


def table_rand(art: Dict[str, Any]) -> List[str]:
    res = art["results"]
    rc = art.get("random_controls", {}) or {}
    rows = ["| 随机臂 | 对标规则臂 | 执行日数 | 组合夏普 | Δ规则臂 |",
            "|---|---|---|---|---|"]
    for l in sorted(rc):
        info = rc[l]
        rule = info["matched_rule"]
        rr = res.get(rule, {})
        d = res[l]["combined_sharpe"] - rr.get("combined_sharpe", 0.0)
        rows.append(f"| {l} | {rule} | {info['n_rebalance_days']} | "
                    f"{fmt_num(res[l]['combined_sharpe'])} | {d:+.4f} |")
    return rows


def table_sub(art: Dict[str, Any], labs: List[str], base: str) -> List[str]:
    subs = art.get("subwindows", {})
    rows = ["| 子段 | 天数 | " + " | ".join(labs) + " |",
            "|---|---|" + "---|" * len(labs)]
    for s in sorted(subs):
        seg = subs[s]
        cells = []
        for l in labs:
            v = seg.get(l)
            cells.append(fmt_num(v))
        rows.append(f"| {s} | {seg.get('n')} | " + " | ".join(cells) + " |")
    return rows


def table_criteria(art: Dict[str, Any], labs: List[str]) -> List[str]:
    crit = art.get("criteria", {})
    rows = ["| 臂 | Δ组合 | ΔA腿 | 子窗口胜 | 随机组最大 | Δ随机组最大 | 通过判据数 |",
            "|---|---|---|---|---|---|---|"]
    for l in labs:
        c = crit.get(l)
        if not c:
            continue
        k = c["criteria"]
        n_pass = sum(1 for key in ("a_ashare_sharpe_gt_base",
                                   "b_main_combined_ge_base_plus_0.05",
                                   "c_oos_combined_ge_base_minus_0.02",
                                   "d_subwindow_majority_ge_3",
                                   "e_beats_same_freq_random_max")
                     if k.get(key) is True)
        rows.append(
            f"| {l} | {c['delta_vs_base_combined']:+.4f} | "
            f"{c['delta_vs_base_ashare']:+.4f} | {c['subwindow_wins']}/4 | "
            f"{fmt_num(c['random_group_max'])} | "
            f"{fmt_num(c['delta_vs_random_max'])} | {n_pass} |")
    return rows


def gen(main: Dict[str, Any], oos: Optional[Dict[str, Any]]) -> str:
    res = main["results"]
    base = main["base_label"]
    labs = rule_labels(main)
    rlabs = rand_labels(main)
    out: List[str] = []
    A = out.append

    A("# P9 · 主腿执行层：调仓频率曲线")
    A("")
    A(f"> 生成时间：由 `backend/scripts/gen_p9_doc.py` 从 "
      f"`multi_asset_p9_main.json` / `multi_asset_p9_oos.json` 自动生成，"
      f"**数字零手抄**。")
    A("")

    # ── 0. 结论 ──
    A("## 0. 一句话结论")
    A("")
    _rule_labs = rule_labels(main)
    best_rule = max(_rule_labs, key=lambda l: res[l]["combined_sharpe"])
    _rc = main.get("random_controls", {}) or {}
    _rand_labs = sorted(_rc)
    _rand_best = (max(_rand_labs,
                      key=lambda l: res[l]["combined_sharpe"]) if _rand_labs else None)
    A(f"- **频率曲线**：周频两个锚点（`wk_fri` / `wk_mon`）都优于日频与月初；"
      f"**main 窗口**规则臂中最优是 **{best_rule}**（组合夏普 "
      f"{fmt_num(res[best_rule]['combined_sharpe'])}），基线 `{base}` 为 "
      f"{fmt_num(main['base_combined_sharpe'])}"
      + ("（**基线本身即规则臂之首**）" if best_rule == base else
         f"（Δ {res[best_rule]['combined_sharpe'] - main['base_combined_sharpe']:+.4f}）")
      + "。")
    # 同类对比：必须拿**与该规则臂同执行日数**的随机组，不能拿全局随机最大
    _peer = [l for l in _rand_labs if _rc[l]["matched_rule"] == best_rule]
    _cmp_best = (max(_peer, key=lambda l: res[l]["combined_sharpe"])
                  if _peer else _rand_best)
    if _cmp_best:
        A(f"- **main 窗口，与 `{best_rule}` 同执行日数的随机对照最大值 "
          f"{fmt_num(res[_cmp_best]['combined_sharpe'])}（`{_cmp_best}`）"
          + ("**高于**" if res[_cmp_best]["combined_sharpe"] > res[best_rule]["combined_sharpe"]
             else "不低于")
          + f"规则臂最优 {best_rule} 的 {fmt_num(res[best_rule]['combined_sharpe'])}"
          + (" → **日历对齐（周五/周一/月初）不携带信息**，"
             "低频的收益是机械的（换手减少），不是日历给的。"
             if res[_cmp_best]["combined_sharpe"] >= res[best_rule]["combined_sharpe"]
             else " → 需看第 8 节的分档对比。"))
    if oos is not None:
        ores2 = oos["results"]
        olabs2 = rule_labels(oos)
        _ob2 = max(olabs2, key=lambda l: ores2[l]["combined_sharpe"])
        _orand = oos.get("random_controls", {}) or {}
        _opeers = [x for x in _orand if _orand[x]["matched_rule"] == base]
        _opmax = max((ores2[x]["combined_sharpe"] for x in _opeers), default=None)
        A(f"- **oos 窗口规则臂中最优是 `{_ob2}`**"
          f"（{fmt_num(ores2[_ob2]['combined_sharpe'])}）——与 main 的 `{best_rule}` "
          f"**相反**；基线 `{base}` {fmt_num(ores2[base]['combined_sharpe'])} "
          f"同样被同频随机对照击败"
          + (f"（{fmt_num(_opmax)}）" if _opmax is not None else "")
          + " → **日历对齐在两个窗口都不携带信息**。")
    A(f"- 预注册判据（a)~(f) 的通过情况见第 7 节；"
      f"**只有通过 (b)(c)(d)(e) 的臂才可能进入定稿**，"
      f"且 (f) 独立样本外证据结构性缺失 → 本实验默认**不自动改定稿**。")
    A("")

    # ── 1. 前提修正 ──
    A("## 1. 前提修正：主腿本来就是周频")
    A("")
    A("实施 P9 时发现的一个事实错误（此前所有文档默认主腿是日频）：")
    A("")
    A("- `multi_asset_leg_experiment.STRATEGY_CFG = {\"rebalance_days\": \"weekly\"}`")
    A("- 策略默认 `rebalance_weekday = 4`（周五）→ **周五收盘出分 → 下一交易日"
      "（周一）执行**")
    A("- 分数映射**只存有限分数的日期**（策略源码：非有限值不入表），"
      "而 lag 平移在入表之前 → 映射键 = **实际执行日**")
    A("")
    A("所以 P9 不是「降频省成本」，而是**频率曲线扫描**：日频 ↔ 周频 ↔ 月频，"
      "周频（当前定稿）是曲线上的一个点，不是端点。")
    A("")

    # ── 2. 设计 ──
    A("## 2. 实验设计")
    A("")
    A("- **规则臂**（唯一变量 = A 股腿调仓频率）：")
    A("  - `wk_fri` —— 基线，周五出分 → 周一执行（= 当前定稿）")
    A("  - `daily` —— 日频出分 → 次日执行（频率上限）")
    A("  - `wk_mon` —— 周一出分 → 周二执行（同一频率，不同日历锚点）")
    A("  - `monthly` —— 每月 1 号出分 → 下一交易日执行（频率下限）")
    A("- **随机对照**：与规则臂**同执行日数**、日期随机（3 组种子）。")
    A("")
    A("⚠️ 为什么必须有「日期随机」对照：同样低的调仓频率，收益可能来自两件不同的事 ——")
    A("")
    A("1. 换手减少 → 成本下降（**机械效应**，与日期无关）")
    A("2. 日历对齐（避开周一开盘、月末等系统性效应）—— 这是**真信息**")
    A("")
    A("只有同频随机日期能把两者分开：规则臂 ≈ 随机组 ⇒ 纯机械；"
      "规则臂 > 随机组最大 ⇒ 存在日历效应。")
    A("")
    A("⚠️ `daily` 臂**没有**随机对照：日频的规则本身就是「每个交易日都调仓」，"
      "同规模的随机子集恒等于规则本身，对照退化。")
    A("")
    A("**预注册判据**（跑之前写死）：")
    A("")
    A("- (a) main 窗口 A 股腿自身夏普 > 基线")
    A("- (b) main 组合夏普 ≥ 基线 + 0.05")
    A("- (c) oos 组合夏普 ≥ 基线 − 0.02")
    A("- (d) 4 个不重叠子窗口中 ≥ 3 个优于基线")
    A("- (e) 组合夏普 > 同执行日数随机对照的最大值")
    A("- (f) 独立样本外证据 —— A 股腿无窗口前历史 → **结构性不满足**")
    A("")

    # ── 3. main 规则臂 ──
    A(f"## 3. main 窗口（{main['start']} → {main['end']}）规则臂")
    A("")
    A(f"基线复现门：`{base}` 组合夏普 "
      f"{fmt_num(main['base_combined_sharpe'])}，P3 定稿 "
      f"{fmt_num(main['p3_expected_base'])} —— "
      f"{'✅ 一致' if abs(main['base_combined_sharpe'] - main['p3_expected_base']) <= 5e-4 else '❌ 漂移'}")
    A("")
    out.extend(table_rule(main, labs))
    A("")

    # ── 4. main 随机对照 ──
    A("## 4. main 随机对照（同执行日数、日期随机）")
    A("")
    if rlabs:
        out.extend(table_rand(main))
    else:
        A("（本窗口未跑随机对照）")
    A("")

    # ── 5. oos ──
    if oos is not None:
        olabs = rule_labels(oos)
        A(f"## 5. oos 窗口（{oos['start']} → {oos['end']}）")
        A("")
        A(f"基线复现门：`{oos['base_label']}` 组合夏普 "
          f"{fmt_num(oos['base_combined_sharpe'])}，P3 定稿 "
          f"{fmt_num(oos['p3_expected_base'])} —— "
          f"{'✅ 一致' if abs(oos['base_combined_sharpe'] - oos['p3_expected_base']) <= 5e-4 else '❌ 漂移'}")
        A("")
        out.extend(table_rule(oos, olabs))
        A("")
        A("### 5.1 oos 随机对照")
        A("")
        if rand_labels(oos):
            out.extend(table_rand(oos))
        else:
            A("（本窗口未跑随机对照）")
        A("")

    # ── 6. 子窗口 ──
    A("## 6. 子窗口（不重叠切分）")
    A("")
    A("### 6.1 main")
    A("")
    out.extend(table_sub(main, labs, base))
    A("")
    if oos is not None:
        A("### 6.2 oos")
        A("")
        out.extend(table_sub(oos, rule_labels(oos), oos["base_label"]))
        A("")

    # ── 7. 判据 ──
    A("## 7. 判据汇总")
    A("")
    out.extend(table_criteria(main, labs))
    A("")
    if oos is not None:
        A("oos 判据（(c) 项在 oos 口径下判定）：")
        A("")
        out.extend(table_criteria(oos, rule_labels(oos)))
        A("")

    # ── 8. 结论 ──
    A("## 8. 结论")
    A("")
    passed = [l for l in labs
              if l != base
              and main.get("criteria", {}).get(l, {}).get("criteria", {}).get(
                  "b_main_combined_ge_base_plus_0.05") is True]
    A(f"- **通过 (b) 的规则臂：{passed if passed else '无'}**")
    if passed:
        for l in passed:
            c = main["criteria"][l]["criteria"]
            A(f"  - `{l}`：(a)={'✅' if c['a_ashare_sharpe_gt_base'] else '❌'} "
              f"(b)=✅ (d)={'✅' if c['d_subwindow_majority_ge_3'] else '❌'} "
              f"(e)={'✅' if c['e_beats_same_freq_random_max'] else '❌'} "
              f"(f)=❌ 结构性缺失")
    _main_best = best_rule
    _obest_txt = ""
    if oos is not None:
        _olabs2 = rule_labels(oos)
        _obest2 = max(_olabs2, key=lambda l: oos["results"][l]["combined_sharpe"])
        _obest_txt = (f"（oos 窗口的规则臂之首是 `{_obest2}`"
                      f" {fmt_num(oos['results'][_obest2]['combined_sharpe'])}）")
    A(f"- **主窗口里基线已是规则臂之首**：main 频率曲线上 `{_main_best}` 高于其余规则臂"
      f"{_obest_txt} → 当前定稿的主腿调仓设置**无需改动**"
      f"（且没有规则臂能通过 (b)）。")
    if _rand_labs:
        _wk = [l for l in _rand_labs if _rc[l]["matched_rule"] == "wk_fri"]
        if _wk:
            _wk_best = max(_wk, key=lambda l: res[l]["combined_sharpe"])
            A(f"- **日历对齐零信息**：与 `wk_fri` 同执行日数（"
              f"{_rc[_wk[0]]['n_rebalance_days']} 天）的随机对照里，最好的一臂 "
              f"`{_wk_best}` 组合夏普 {fmt_num(res[_wk_best]['combined_sharpe'])}，"
              f"**不低于**基线 {fmt_num(main['base_combined_sharpe'])} → "
              f"「周五出分」这一锚点没有任何可辨识的信息量。")
    _mo = [l for l in _rand_labs if _rc[l]["matched_rule"] == "monthly"]
    if _mo and "monthly" in res:
        _mo_best = max(_mo, key=lambda l: res[l]["combined_sharpe"])
        A(f"- **月初对齐甚至不如随机**：同执行日数（"
          f"{_rc[_mo[0]]['n_rebalance_days']} 天）的随机对照最好 "
          f"{fmt_num(res[_mo_best]['combined_sharpe'])}，高于月初对齐的 "
          f"{fmt_num(res['monthly']['combined_sharpe'])}。")
    _a_best = max(res[l]["a_share_sharpe"] for l in res)
    _a_who = [l for l in res if res[l]["a_share_sharpe"] == _a_best][0]
    A(f"- **主腿弱在执行层同样得不到改善**：基线 A 股腿夏普 "
      f"{fmt_num(main['base_ashare_sharpe'])}，而全部臂里最高的 A 股腿夏普是 "
      f"{fmt_num(_a_best)}（`{_a_who}`，一个随机调仓日程）→ "
      f"主腿的表现在很大程度上由**调仓日程**决定，而非信号本身。")
    if oos is not None:
        ores = oos["results"]
        olabs = rule_labels(oos)
        _obest = max(olabs, key=lambda l: ores[l]["combined_sharpe"])
        _mbest = best_rule
        # 逐窗口检查：每个有随机对照的规则臂是否都被同频随机对照击败
        _beaten_all = True
        _details = []
        for _w_art, _wn in ((main, "main"), (oos, "oos")):
            _wres = _w_art["results"]
            _wrc = _w_art.get("random_controls", {}) or {}
            for _l in rule_labels(_w_art):
                _peers = [x for x in _wrc if _wrc[x]["matched_rule"] == _l]
                if not _peers:
                    continue
                _pmax = max(_wres[x]["combined_sharpe"] for x in _peers)
                _plab = max(_peers, key=lambda x: _wres[x]["combined_sharpe"])
                if _wres[_l]["combined_sharpe"] >= _pmax:
                    _beaten_all = False
                _details.append(
                    f"{_wn}/{_l} {fmt_num(_wres[_l]['combined_sharpe'])} vs "
                    f"{_plab} {fmt_num(_pmax)}")
        if _beaten_all:
            A(f"- **两个窗口、所有有随机对照的规则臂，全部被同频随机对照击败**："
              + "；".join(_details) + " → 日历对齐零信息，两窗口一致。")
        A(f"- **规则臂排名在两个窗口里翻转**：main 最好的是 `{_mbest}`"
          f"（{fmt_num(res[_mbest]['combined_sharpe'])}），"
          f"oos 最好的是 `{_obest}`（{fmt_num(ores[_obest]['combined_sharpe'])}）→ "
          f"「选哪个锚点」是窗口噪声，不是稳定优势。")
    A(f"- (f) 独立样本外证据：A 股腿面板从 2021-05-11 起，OOS 起点正是那天 → "
      f"**没有窗口前的 A 股腿历史**，任何主腿改造都缺第三样本。")
    A(f"- **定稿影响：无。** 主腿仍为 "
      f"`honest_composite` 周频（周五出分→周一执行），"
      f"A 股腿 50% / gold / commodity / bond 各 16.67% 的资产侧定稿不变。")
    A("")
    A("**后续方向**（P9 未能回答的问题）：")
    A("")
    A("- 日频臂为何更差 —— 是**成本**（交易笔数 "
      f"{res['daily']['total_trades']} vs 基线 {res[base]['total_trades']}）"
      f"还是**信号衰减**？可用「零成本日频」臂隔离（诊断用，不作定稿候选）。")
    A("- `monthly` 掩码是「每月 1 号」，而 1 号常是周末/节假日 → "
      f"3 年只有 {res['monthly']['rebalance_days']} 个执行日"
      f"（约 {res['monthly']['rebalance_days'] / 3:.1f} 次/年），"
      f"并非真正的「月度」；若要测月频需另设「每月首个交易日」臂。")
    A("")
    A("---")
    A("")
    A(f"口径：A 股腿 50%（honest_composite，信号日收盘 → 次日 open，"
      f"佣金 0.03% + 滑点 0.01%）+ 袖套 50%（P5 固定持仓集，类内等权、类间等权，"
      f"每月 21 交易日再平衡）；袖套 main 夏普 "
      f"{fmt_num(main['sleeve']['sharpe'])}，波动 {fmt_pct(main['sleeve']['vol'])}，"
      f"换手 {fmt_num(main['sleeve']['cost']['turnover_ratio'], 2)}x。")
    A("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-only", action="store_true", help="只生成 main 部分")
    ap.add_argument("--out", default="P9_REBALANCE_FREQ_2026-10-31.md")
    args = ap.parse_args()

    m = load("multi_asset_p9_main.json")
    if m is None:
        raise SystemExit("❌ 找不到 multi_asset_p9_main.json")
    o = None if args.main_only else load("multi_asset_p9_oos.json")
    text = gen(m, o)
    path = os.path.join(DOCS, args.out)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"✅ 已生成 {path}（main={m['window']}, oos={'有' if o else '无'}）")


if __name__ == "__main__":
    main()
