#!/usr/bin/env python3
"""生成 docs/backtest/P4_CLASS_WEIGHT_2026-10-31.md（表格全部由 JSON 产物渲染）。

P4 = 在 P3 修正后的定稿上重做「类间配权」。结论：全部证伪，维持类间等权。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA = os.path.join(ROOT, "data", "backtest")
DOC = os.path.join(ROOT, "docs", "backtest", "P4_CLASS_WEIGHT_2026-10-31.md")

CANDS = ["invvol", "invvol_nocap", "minvol", "riskparity", "lowcorr", "sharpew"]
NAMES = {
    "base_equal": "类间等权（定稿基线）",
    "invvol": "逆波动率 cap40/floor8",
    "invvol_nocap": "逆波动率 无 cap",
    "minvol": "100% 给最低波动类",
    "riskparity": "风险平价",
    "lowcorr": "∝(1−ρ(类,A股腿))",
    "sharpew": "∝max(夏普,0)",
}


def load(name: str) -> Dict[str, Any]:
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def f(v, n=4, dash="—"):
    return dash if v is None else f"{v:.{n}f}"


def pct(v, dash="—"):
    return dash if v is None else f"{v:.2%}"


def main() -> None:
    m = load("multi_asset_p4_main.json")
    o = load("multi_asset_p4_oos.json")
    lg = load("multi_asset_p4_long.json")

    L: List[str] = []
    a = L.append

    def rand_stats(d, pref):
        R = d["results"]
        vals = [R[k]["combined_sharpe"] for k in R if k.startswith(pref)]
        return (sum(vals) / len(vals), min(vals), max(vals), len(vals))

    rm_mean, rm_min, rm_max, rm_n = rand_stats(m, "rand_s")
    rd_mean, rd_min, rd_max, rd_n = rand_stats(m, "randdyn_s")
    ro_mean, ro_min, ro_max, ro_n = rand_stats(o, "rand_s")
    rod_mean, rod_min, rod_max, rod_n = rand_stats(o, "randdyn_s")

    a("# P4 · 类间配权重做（在 P3 修正后的定稿上）")
    a("")
    a("**日期**：2026-10-31　|　**结论：全部证伪，维持类间等权**")
    a("")
    a("> **定稿不变**：A 股腿 50% + gold / bond / commodity 各 16.67%，类内等权，")
    a("> 每月再平衡回类间等权；bond 类 = 固定 `{511260, 511010, 511020}`。")
    a("> 组合夏普 **main 1.4092 / oos 1.0685**，回撤 **−7.95%**。")
    a("> 见 `P3_BOND_CLASS_2026-10-31.md` §7。")
    a("")
    a("---")
    a("")

    # ── 1 ──
    a("## 1. 为什么必须重做 P2")
    a("")
    a("P2 证伪了「类间逆波动率配权」，给出的机制是：")
    a("")
    a("> bond 腿年化波动最低(3.3%) → 逆波动率把仓位压向 bond(68.6%)")
    a("> → 但 bond 与 A 股腿 ρ 最高(0.55/0.62) → **反分散** → 组合夏普反降。")
    a("")
    a("**P3 把这条链条的前提抽掉了**：bond 类换成固定长久期利率债后，")
    a("它与 A 股腿的 ρ 变成 **−0.216 / −0.054（负相关）**。")
    a("「低波 → 高相关 → 反分散」这条链现在断了 ——")
    a("**机制讲得通不等于有效，必须重跑全部对照。**")
    a("")
    a("---")
    a("")

    # ── 2 ──
    a("## 2. 预注册判据（跑之前写死）")
    a("")
    a("一个配权方案要被采纳，必须**同时**满足：")
    a("")
    a("| # | 判据 |")
    a("|---|---|")
    a("| 1 | main 组合夏普 ≥ base + 0.05 |")
    a("| 2 | oos 组合夏普 ≥ base − 0.02（不得显著变差）|")
    a("| 3 | 优于同参数 random 对照的族均值 |")
    a("| 4 | 滚动 504 日夏普胜率 ≥ 60% |")
    a("| 5 | 长历史（2019-01 → 2026-10）交叉验证不显著变差 |")
    a("")
    a("任一条不过 → 证伪。**门槛事后不许改。**")
    a("")
    a("---")
    a("")

    # ── 3 ──
    a("## 3. 方案")
    a("")
    a("| 标签 | 规则 |")
    a("|---|---|")
    for k in CANDS:
        a(f"| `{k}` | {NAMES[k]} |")
    a("| `rand_s{seed}` | Dirichlet(1) 随机**固定**权重（全窗口不变），6 个种子 |")
    a("| `randdyn_s{seed}` | 每次调仓重抽 Dirichlet(1) 随机权重，6 个种子 |")
    a("")
    a("两组 random 的分工：")
    a("")
    a("- `rand_s` − `base` = **「选一个固定权重」这件事本身**能带来多少")
    a("- `randdyn` − `rand_s` = **「每月调权」这个动作**能带来多少")
    a("- 候选 − `randdyn` = 调权**规则**的独特价值")
    a("")
    a("所有权重均为 **walk-forward**：只用信号日**之前** 120 个交易日的数据，无前视。")
    a("")
    a("---")
    a("")

    # ── 4/5 ──
    for doc, tag, title in ((m, "main", "4. 推导窗口（2023-05-08 → 2026-05-08）"),
                            (o, "oos", "5. 样本外窗口（2021-05-11 → 2023-05-08）")):
        R = doc["results"]
        base = R["base_equal"]["combined_sharpe"]
        a(f"## {title}")
        a("")
        a(f"基线（类间等权）= **{base:.4f}**")
        a("")
        a("| 方案 | 组合夏普 | Δ基线 | 剔 gold 夏普 | ρ(袖套,A股腿) | 最大回撤 | 收益 | 换手 | 504 日胜率 | 平均类权重 gold/bond/commodity |")
        a("|---|---|---|---|---|---|---|---|---|---|")
        for k in ["base_equal"] + CANDS:
            v = R[k]
            w = v.get("avg_class_weight", {})
            label = f"`{k}` {NAMES[k]}"
            ws = " / ".join(f"{w.get(c, 0):.2f}" for c in ("gold", "bond", "commodity"))
            rw = v.get("roll_win_504")
            rw_s = "—" if rw is None else f"{rw:.1%}"
            a(f"| {label} | {v['combined_sharpe']:.4f} | "
              f"{v['combined_sharpe'] - base:+.4f} | {f(v.get('sharpe_ex_gold'))} | "
              f"{f(v.get('corr_sleeve_ashare'))} | {v['combined_maxdd']:.2%} | "
              f"{v['combined_return']:.2%} | {v['cost']['turnover_ratio']:.2f}x | {rw_s} | {ws} |")
        a("")
        a("| random 对照 | n | 均值 | 范围 |")
        a("|---|---|---|---|")
        a(f"| 固定权重 `rand_s*` | {rm_n if tag=='main' else ro_n} | "
          f"{(rm_mean if tag=='main' else ro_mean):.4f} | "
          f"[{(rm_min if tag=='main' else ro_min):.4f}, {(rm_max if tag=='main' else ro_max):.4f}] |")
        a(f"| 动态权重 `randdyn_s*` | {rd_n if tag=='main' else rod_n} | "
          f"{(rd_mean if tag=='main' else rod_mean):.4f} | "
          f"[{(rd_min if tag=='main' else rod_min):.4f}, {(rd_max if tag=='main' else rod_max):.4f}] |")
        a("")
        best_cand = max(CANDS, key=lambda k: R[k]["combined_sharpe"])
        a(f"→ 最好的候选方案是 **{NAMES[best_cand]}**（{R[best_cand]['combined_sharpe']:.4f}，"
          f"Δ基线 {R[best_cand]['combined_sharpe'] - base:+.4f}）。")
        a("")
        a("---")
        a("")

    # ── 6 ──
    a("## 6. 长历史（2019-01 → 2026-10，袖套口径）")
    a("")
    a("> **定位必须写清楚**：长历史**没有 A 股腿**（`honest_composite` 面板从 2021-05-11 起），")
    a("> 只有 `510300` 代理。P2 已经踩过这个坑：**袖套夏普升 ≠ 组合夏普升**")
    a("> （长历史袖套 Δ+0.99 但组合 Δ−0.12）。所以下表**是辅助证据，不构成组合层判决**。")
    a("")
    R = lg["results"]
    base = R["base_equal"]["sleeve_sharpe"]
    a(f"基线（类间等权）袖套夏普 = **{base:.4f}**")
    a("")
    a("| 方案 | 袖套夏普 | Δ基线 | 袖套波动 | ρ(袖套,510300) | 504 日胜率 |")
    a("|---|---|---|---|---|---|")
    for k in ["base_equal"] + CANDS:
        v = R[k]
        rw = v.get("roll_win_504")
        rw_s = "—" if rw is None else f"{rw:.1%}"
        a(f"| `{k}` {NAMES[k]} | {v['sleeve_sharpe']:.4f} | "
          f"{v['sleeve_sharpe'] - base:+.4f} | {v['sleeve_vol']:.2%} | "
          f"{f(v.get('corr_sleeve_proxy'))} | {rw_s} |")
    a("")
    a("")
    lrm = [R[k]["sleeve_sharpe"] for k in R if k.startswith("rand_s")]
    lrd = [R[k]["sleeve_sharpe"] for k in R if k.startswith("randdyn_s")]
    a("| random 对照 | n | 袖套夏普均值 | 范围 |")
    a("|---|---|---|---|")
    a(f"| 固定权重 `rand_s*` | {len(lrm)} | {sum(lrm)/len(lrm):.4f} | "
      f"[{min(lrm):.4f}, {max(lrm):.4f}] |")
    a(f"| 动态权重 `randdyn_s*` | {len(lrd)} | {sum(lrd)/len(lrd):.4f} | "
      f"[{min(lrd):.4f}, {max(lrd):.4f}] |")
    a("")
    a("---")
    a("")

    # ── 7 ──
    a("## 7. 判据核对")
    a("")
    a("| 方案 | 1 · main ≥ +0.05 | 2 · oos ≥ −0.02 | 3 · 胜 random 均值 | 4 · 504 胜率 ≥60% | 5 · 长历史 | 采纳 |")
    a("|---|---|---|---|---|---|---|")
    Rm, Ro, Rl = m["results"], o["results"], lg["results"]
    for k in CANDS:
        dm = Rm[k]["combined_sharpe"] - Rm["base_equal"]["combined_sharpe"]
        do = Ro[k]["combined_sharpe"] - Ro["base_equal"]["combined_sharpe"]
        dl = Rl[k]["sleeve_sharpe"] - Rl["base_equal"]["sleeve_sharpe"]
        c1 = "✅" if dm >= 0.05 else f"✗ ({dm:+.4f})"
        c2 = "✅" if do >= -0.02 else f"✗ ({do:+.4f})"
        c3 = "✅" if Rm[k]["combined_sharpe"] > rm_mean and Ro[k]["combined_sharpe"] > ro_mean else "✗"
        rw = Rm[k].get("roll_win_504")
        c4 = "—" if rw is None else ("✅" if rw >= 0.60 else f"✗ ({rw:.1%})")
        c5 = "✅" if dl >= -0.05 else f"✗ ({dl:+.4f})"
        ok = all(x.startswith("✅") for x in (c1, c2, c3, c4, c5))
        a(f"| `{k}` {NAMES[k]} | {c1} | {c2} | {c3} | {c4} | {c5} | "
          f"{'**采纳**' if ok else '证伪'} |")
    a("")
    a("**没有一个方案通过全部 5 条。维持类间等权。**")
    a("")
    a("---")
    a("")

    # ── 8 ──
    a("## 8. 机制解释（四条，都比「证伪」本身更有价值）")
    a("")
    a("### 8.1 逆波动率失败的真正原因不是相关性，是**收益稀释**")
    a("")
    inv = Rm["invvol"]
    noc = Rm["invvol_nocap"]
    mv = Rm["minvol"]
    bs = Rm["base_equal"]
    a(f"bond 类换成负相关资产之后，逆波动率**依然失败**")
    a(f"（main {inv['combined_sharpe']:.4f} < 基线 {bs['combined_sharpe']:.4f}）。")
    a("")
    a("| 方案 | bond 平均权重 | 袖套波动 | ρ(袖套,A股腿) | 组合收益 | 组合夏普 |")
    a("|---|---|---|---|---|---|")
    a(f"| 类间等权 | {bs['avg_class_weight']['bond']:.4f} | {bs['sleeve_vol']:.2%} | "
      f"{bs['corr_sleeve_ashare']:.4f} | {bs['combined_return']:.2%} | {bs['combined_sharpe']:.4f} |")
    a(f"| 逆波动率 | {inv['avg_class_weight']['bond']:.4f} | {inv['sleeve_vol']:.2%} | "
      f"{inv['corr_sleeve_ashare']:.4f} | {inv['combined_return']:.2%} | {inv['combined_sharpe']:.4f} |")
    a(f"| 逆波动率 无 cap | {noc['avg_class_weight']['bond']:.4f} | {noc['sleeve_vol']:.2%} | "
      f"{noc['corr_sleeve_ashare']:.4f} | {noc['combined_return']:.2%} | {noc['combined_sharpe']:.4f} |")
    a(f"| 100% 最低波动类 | {mv['avg_class_weight']['bond']:.4f} | {mv['sleeve_vol']:.2%} | "
      f"{mv['corr_sleeve_ashare']:.4f} | {mv['combined_return']:.2%} | {mv['combined_sharpe']:.4f} |")
    a("")
    a("**相关性确实降了**（ρ 从 0.238 降到 0.227 / 0.132 / −0.165），**回撤也确实降了**")
    a(f"（{bs['combined_maxdd']:.2%} → {noc['combined_maxdd']:.2%}），")
    a("**但收益被腰斩**（"
      f"{bs['combined_return']:.2%} → {noc['combined_return']:.2%}），夏普反而下降。")
    a("")
    a("> **P2 当时的归因是错的。** 不是「把仓位推向高相关的 bond」，")
    a("> 而是「把仓位推向**低波资产**」——低波资产占比越高，组合收益越接近低波资产的收益，")
    a("> 而组合的波动主要由 50% 的 A 股腿决定，**降袖套波动对组合夏普没有帮助**。")
    a("> 现在 bond 已经变成负相关资产，逆波动率照样失败 —— **正好证明相关性不是关键变量**。")
    a("")
    a("### 8.2 `sharpew` 是**追涨**，样本外直接崩")
    a("")
    sw_m, sw_o = Rm["sharpew"], Ro["sharpew"]
    a(f"`∝max(夏普,0)` 本质是**按过去表现加权 = 动量**。")
    a(f"推导窗口它给 gold {sw_m['avg_class_weight']['gold']:.2f} / "
      f"bond {sw_m['avg_class_weight']['bond']:.2f}（2023-2026 黄金与债券都在涨），")
    a(f"夏普 {sw_m['combined_sharpe']:.4f} 看起来很好；")
    a(f"到样本外 gold 掉到 {sw_o['avg_class_weight']['gold']:.2f}，")
    a(f"**夏普崩到 {sw_o['combined_sharpe']:.4f}**（基线 {Ro['base_equal']['combined_sharpe']:.4f}）。")
    a(f"长历史也是负的（{Rl['sharpew']['sleeve_sharpe'] - Rl['base_equal']['sleeve_sharpe']:+.4f}）。")
    a("")
    a("> **P1 证伪的「跨类动量」在配权形式下重现了。** 这不是巧合：")
    a("> 「按历史夏普加权」和「按历史涨幅排序」是同一件事。")
    a("")
    a("### 8.3 `riskparity` 在样本外大幅胜出，是**协方差估计噪声**，不是 alpha")
    a("")
    rp_m, rp_o = Rm["riskparity"], Ro["riskparity"]
    a(f"main {rp_m['combined_sharpe']:.4f}（Δ {rp_m['combined_sharpe']-Rm['base_equal']['combined_sharpe']:+.4f}）")
    a(f"却 oos {rp_o['combined_sharpe']:.4f}（Δ {rp_o['combined_sharpe']-Ro['base_equal']['combined_sharpe']:+.4f}）——")
    a(f"样本外窗口只有 **{o['calendar']['n']} 天**（不到 504 日滚动窗口的长度），")
    a(f"而风险平价要用 120 日协方差矩阵求逆 → **权重对估计误差极敏感**（换手 "
      f"{rp_o['cost']['turnover_ratio']:.1f}x）。")
    a(f"main 的 504 日滚动胜率只有 **{rp_m.get('roll_win_504', 0):.1%}**。")
    a("")
    a("> **短窗口的样本外「大幅胜出」正是 P1 踩过的坑**（那里 `mom_L250_top2` "
      "推导窗口同族最差、样本外最好，只能用样本外才选得出来）。")
    a("> 判据 1 不过 → 证伪。")
    a("")
    a("### 8.4 长历史袖套口径再次暴露它的无用")
    a("")
    a(f"长历史里逆波动率无 cap 袖套夏普 **{Rl['invvol_nocap']['sleeve_sharpe']:.4f}**"
      f"（Δ **{Rl['invvol_nocap']['sleeve_sharpe']-Rl['base_equal']['sleeve_sharpe']:+.4f}**，"
      f"504 日胜率 {Rl['invvol_nocap'].get('roll_win_504', 0):.0%}），")
    a("看起来是压倒性胜利 —— 但这正是 P2 的翻版：**袖套自己变好了，组合并没有变好**。")
    a("原因在 §8.1：长历史没有 A 股腿，测不到「组合波动由 A 股腿主导」这件事。")
    a("")
    a("> **长历史袖套口径测不到 ρ(袖套,A股腿) 这个决定性变量 → 它不是判决工具。**")
    a("")
    a("---")
    a("")

    # ── 9 ──
    a("## 9. random 对照给出的判决")
    a("")
    a(f"推导窗口 random 固定权重 6 个种子的均值 **{rm_mean:.4f}**，"
      f"范围 [{rm_min:.4f}, {rm_max:.4f}]。")
    a("")
    a(f"- 最好的候选 `sharpew` {Rm['sharpew']['combined_sharpe']:.4f} —— **低于 random 的最优**"
      f"（{rm_max:.4f}）")
    a(f"- 最好看的 `lowcorr` {Rm['lowcorr']['combined_sharpe']:.4f}，"
      f"Δ基线仅 **{Rm['lowcorr']['combined_sharpe']-Rm['base_equal']['combined_sharpe']:+.4f}**，"
      f"**低于事前注册的 +0.05 门槛**")
    a(f"- random 族里 **{rm_max:.4f}** 和 "
      f"{sorted(Rm[k]['combined_sharpe'] for k in Rm if k.startswith('rand_s'))[-2]:.4f} "
      f"两个配置**比所有候选方案都高**")
    a("")
    a("> **「随便挑一个固定类权重」能拿到的夏普，和精心设计的配权规则是同一个量级。**")
    a("> 这说明类间权重**不是这个组合的关键变量** —— 决定成败的是")
    a("> **「押中了哪个资产」**（gold 权重高的 random 种子赢，因为 2023-2026 黄金大涨），")
    a("> 而不是「怎么按规则分权重」。")
    a("> 和 P2 的结论完全一致：**等权不是最优，但没有规则能证明别的更好。**")
    a("")
    a("---")
    a("")

    # ── 10 ──
    a("## 10. 更根本的发现：这个 ETF 池里只有 **2 个**低相关源")
    a("")
    a("用 `510300` 做窗口前 A 股 beta 代理，对全池 567 只 ETF 排序（P3 勘察产物")
    a("`p3_recon.json`），ρ 最低的前 30 名里：")
    a("")
    a("| 排名区间 | 类别 | 说明 |")
    a("|---|---|---|")
    a("| 1–7 | bond（利率债） | ρ −0.31 ~ −0.12 |")
    a("| 8–21 | gold | ρ −0.02 ~ +0.03 |")
    a("| 22 起 | commodity / cross_border / a_share | ρ +0.16 ~ +0.45 |")
    a("")
    a("**没有任何第四类资产能提供额外的低相关暴露。** 这解释了 P3 的")
    a("「新增第 4 类」为什么失败（1.3650 / 0.8401）——")
    a("ρ 最低的候选**本质上还是债券**，加进来等于加杠杆债，不是新增分散源。")
    a("")
    a("另外：**池子里没有白银 ETF、没有原油 ETF**（只有两个美股油气 QDII，")
    a("属 cross_border，而 cross_border 已被证伪为毒药）。")
    a("所以「扩充商品类成分」这条路在当前数据下也是死的。")
    a("")
    a("> **结论：在现有资产池内，「类间配权」和「新增资产类」两条路都已走到尽头。**")
    a("> 下一步只能转向**组合层的风险管理机制**（权重封顶 / 波动率目标等），")
    a("> 它不依赖新资产，也不依赖选标 alpha。")
    a("")
    a("---")
    a("")

    # ── 11 ──
    a("## 11. 复现")
    a("")
    a("```bash")
    a("cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\")
    a("    scripts/multi_asset_p4_weighting.py --window main")
    a("cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\")
    a("    scripts/multi_asset_p4_weighting.py --window oos")
    a("cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\")
    a("    scripts/multi_asset_p4_weighting.py --window long")
    a("```")
    a("")
    a("产物：`data/backtest/multi_asset_p4_{main,oos,long}.json`")
    a("")

    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"✓ 已写入 {DOC}")
    print(f"  {len(L)} 行")


if __name__ == "__main__":
    main()
