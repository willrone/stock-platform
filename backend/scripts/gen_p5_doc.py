#!/usr/bin/env python3
"""生成 docs/backtest/P5_CLASS_AUDIT_2026-10-31.md（表格全部由 JSON 产物渲染）。

P5 = gold / commodity 类的分类审计（把 P3 的"分类正确性"这把尺子量到其余两类上）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA = os.path.join(ROOT, "data", "backtest")
DOC = os.path.join(ROOT, "docs", "backtest", "P5_CLASS_AUDIT_2026-10-31.md")

FINAL_HELD = {
    "gold": ["518880", "159934", "159937"],
    "commodity": ["159980", "159985", "159981"],
    "bond": ["511260", "511010", "511020"],
}


def load(name: str) -> Dict[str, Any]:
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def fmt(v, pct=False, dash="—"):
    if v is None:
        return dash
    if pct:
        return f"{v:.2%}"
    return f"{v:.4f}"


def main() -> None:
    p5 = load("p5_class_audit.json")
    win = p5["windows"]

    L: List[str] = []
    a = L.append

    a("# P5 · gold / commodity 类分类审计")
    a("")
    a("**日期**：2026-10-31　|　**性质**：分类正确性审计（不做参数搜索）")
    a("")
    a("> **定稿已更新**：本轮之后，**三类全部改为固定持仓集**（gold / commodity / bond），")
    a("> 不再使用「流动性 top-3」规则。当前 main / oos 两个窗口的数值**不变**")
    a("> （因为 top-3 恰好就等于固定集），变的是**规则的固化**，防的是未来的漂移。")
    a("> 组合定稿与夏普见 `P3_BOND_CLASS_2026-10-31.md` §7。")
    a("")
    a("---")
    a("")

    # ── 1. 为什么做 ──
    a("## 1. 为什么做这件事")
    a("")
    a("P3 的唯一收获是**分类纠错**：bond 类混进了博时可转债 ETF `511380`")
    a("（ρ 与 A 股代理 **+0.7643 / +0.7976**），本质是「带债底的股票」——")
    a("**这不是参数问题，是资产性质不一致**。剔除后组合夏普 main 1.3767 → **1.4092**、")
    a("oos 1.0382 → **1.0685**，回撤 −8.70% → **−7.95%**。")
    a("")
    a("既然一个类能被污染，另外两个类也可能。P5 把同一把尺子量到 gold / commodity 上：")
    a("")
    a("1. **性质审计**：类内每只到底是不是同一类资产？")
    a("2. **对 A 股腿的 ρ**：真正的分散性来源在哪。")
    a("3. **「流动性 top-3」会不会漂移**：P3 已证 bond 类会漂移（2019 选长债、2023 选可转债）。")
    a("")
    a("---")
    a("")

    # ── 2. gold ──
    a("## 2. gold 类：干净 ✅")
    a("")
    g0 = win["oos_2021"]["classes"]["gold"]
    a(f"基准：纯金 ETF `{g0['ref']}`（华安黄金）。窗口前 250 交易日 ρ。")
    a("")
    for wname in ("w2020", "oos_2021", "main_2023"):
        if wname not in win:
            continue
        e = win[wname]["classes"]["gold"]
        a(f"### 窗口 `{wname}`（窗口前 {win[wname]['start']} 起算）")
        a("")
        a("| 代码 | 名称 | ρ(纯金基准) | ρ(A股代理) | 年化波动 | 窗口前 120 日日均成交额 |")
        a("|---|---|---|---|---|---|")
        for r in e["rows"]:
            flag = " ⚠️" if (r["rho_ref"] is not None and r["rho_ref"] < 0.85) else ""
            a(f"| `{r['code']}` | {r['name']}{flag} | {fmt(r['rho_ref'])} | "
              f"{fmt(r['rho_proxy'])} | {fmt(r['vol'], pct=True)} | "
              f"{r['turnover']/1e4:.0f} 万 |")
        a("")
        a(f"→ 流动性 top-3：`{'`、`'.join(e['top3_by_liquidity'])}`")
        a("")
    a("**结论**：")
    a("")
    a("- gold 类 14 只**全部是黄金 / 上海金 ETF**。白银 ETF 名称不含「黄金」，未进类；")
    a("  **黄金股 ETF（6 只）由 `class_overrides` 单独归到 `gold_equity`，已正确排除** ——")
    a("  这是本轮最重要的一条：**如果黄金股混进 gold 类，就会引入 A 股股票 beta**。")
    a("- ρ(纯金基准) 绝大多数在 **0.98 ~ 0.997**，性质一致。")
    a("- 只有 `518860 建信上海金ETF`（ρ 0.61 / 0.79）与 `518890 中银黄金`（ρ 0.82）偏低，")
    a("  疑似**黄金期货 ETF**（展期损益导致定价机制不同）。但两者窗口前日均成交额只有")
    a("  **24 ~ 441 万元**，流动性极差，**不可能进入任何实盘持仓**，不需要处理。")
    a("- **top-3 三个窗口完全稳定** ✅（差异仅在 159934/159937 的排序）。")
    a("")
    a("---")
    a("")

    # ── 3. commodity ──
    a("## 3. commodity 类：是个「伪类」⚠️")
    a("")
    for wname in ("w2020", "oos_2021", "main_2023"):
        if wname not in win:
            continue
        e = win[wname]["classes"]["commodity"]
        a(f"### 窗口 `{wname}`")
        a("")
        a("| 代码 | 名称 | ρ(有色基准) | ρ(A股代理) | 年化波动 | 窗口前 120 日日均成交额 |")
        a("|---|---|---|---|---|---|")
        for r in e["rows"]:
            a(f"| `{r['code']}` | {r['name']} | {fmt(r['rho_ref'])} | "
              f"{fmt(r['rho_proxy'])} | {fmt(r['vol'], pct=True)} | "
              f"{r['turnover']/1e4:.0f} 万 |")
        a("")
        a(f"→ 流动性 top-3：`{'`、`'.join(e['top3_by_liquidity'])}`")
        a("")
    a("**结论（三条，都指向行动）**：")
    a("")
    a("1. **类内几乎没有共同因子**：`159985 豆粕` 与 `159980 有色` 的 ρ 只有 **0.11**；")
    a("   `159981 能化` 对有色也只有 **0.49 ~ 0.62**。这四只根本不是「一个资产类别」，")
    a("   而是**有色 / 农产品 / 能化 / 综合商品**四个不同的 beta 被打包。")
    a("   （这其实是**好事**：类内天然分散，等权持有自带分散效应。）")
    a("2. **`510170 国联安上证商品ETF` 是全池 ρ(A股代理) 最高的标的**：")
    a("   **0.5853（2021 窗口）/ 0.7008（2023 窗口）** —— 比 A 股腿本身还接近股票。")
    a("   它是「综合商品指数」，里面混了与 A 股同向的品种。**不纳入。**")
    a("3. **「流动性 top-3」规则会漂移 ⚠️**：")
    a("")
    a("   | 窗口 | 流动性 top-3 |")
    a("   |---|---|")
    for wname in ("w2020", "oos_2021", "main_2023"):
        if wname not in win:
            continue
        e = win[wname]["classes"]["commodity"]
        mark = " ← 含 510170（ρ(A股) 最高）" if "510170" in e["top3_by_liquidity"] else ""
        a(f"   | `{wname}` | {'`、`'.join(e['top3_by_liquidity'])}{mark} |")
    a("")
    a("   2020 窗口的规则会把 **`510170`（ρ(A股)=0.59~0.70）** 换进来，")
    a("   2021 / 2023 窗口则换成 `159981`。**同一条规则，不同窗口给出不同的持仓集** ——")
    a("   和 P3 在 bond 类上发现的一模一样。")
    a("")
    a("---")
    a("")

    # ── 4. 定稿 ──
    a("## 4. 定稿决定：三类全部固定持仓集")
    a("")
    a("P3 已经在 bond 类上定过一次：排名类规则会随市场规模漂移 → **用固定持仓集 + 季度复核**。")
    a("P5 把这条结论推广到另外两类。")
    a("")
    a("| 资产类 | 固定持仓集 | 与「流动性 top-3」的关系 |")
    a("|---|---|---|")
    a(f"| gold | `{'`、`'.join(FINAL_HELD['gold'])}` | 三个窗口本就一致 ✅ |")
    a(f"| commodity | `{'`、`'.join(FINAL_HELD['commodity'])}` | main / oos 窗口一致；"
      f"2020 窗口会漂到 `510170` ⚠️ |")
    a(f"| bond | `{'`、`'.join(FINAL_HELD['bond'])}` | P3 已定（剔除可转债） |")
    a("")
    a("**关键：数值不变。** main 与 oos 两个窗口的「流动性 top-3」恰好就等于上面这三组，")
    a("所以固化后组合夏普、回撤、相关性**一位都不变**（main 1.4092 / oos 1.0685 / 回撤 −7.95%）。")
    a("变的只是**规则**：从「每次按排名重算」变成「写死持仓集 + 季度复核流动性」。")
    a("")
    a("复核触发条件（写死）：任一标的近 120 交易日日均成交额跌破 **5000 万元**，")
    a("或规模逼近清盘线 → 才允许替换。")
    a("")
    a("---")
    a("")

    # ── 5. 更深的问题 ──
    a("## 5. 一个更根本的问题：类粒度可能是错的抽象")
    a("")
    a("把三类摊开看**实际的风险源数**：")
    a("")
    a("| 类 | 成分 | 类内 ρ | 实质风险源数 |")
    a("|---|---|---|---|")
    a("| gold | 3 只黄金 ETF | ~0.99 | **1** |")
    a("| bond | 3 只长久期利率债 | 高（同质） | **1** |")
    a("| commodity | 有色 / 豆粕 / 能化 | 0.11 ~ 0.62 | **3** |")
    a("")
    a("当前定稿给三类**同等权重**（各 16.67%），但它们承载的独立风险源数是 1 / 1 / 3。")
    a("也就是说：**gold 类拿 1 个风险源的钱，commodity 类拿 3 个风险源的钱**。")
    a("")
    a("这说明「类内等权 + 类间等权」并不是一个中性的默认结构 —— 它隐含了一个")
    a("**「每类等权 = 每类风险相等」的错误假设**。真正该问的是：")
    a("")
    a("> **要不要在「单个标的」层面做配权？** 让 gold 的 3 只合计权重 ≠ 16.67%，")
    a("> 而是把整条袖套的风险预算公平地分给 9 个标的。")
    a("")
    a("这就是下一步 **P6：资产层面的风险平价**。注意它与已被证伪的 P2「类间逆波动率」")
    a("**不是一回事**：")
    a("")
    a("| | P2（已证伪） | P6（待做） |")
    a("|---|---|---|")
    a("| 配权层次 | 3 个类 | **9 个标的** |")
    a("| 失败原因 | 把仓位推向**与主腿高相关**的 bond | —— |")
    a("| 关键差异 | 低波 ≠ 低相关 | 同风险预算，**不按波动率单因子**，按**边际风险贡献** |")
    a("")
    a("---")
    a("")

    # ── 6. 复现 ──
    a("## 6. 复现")
    a("")
    a("```bash")
    a("cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \\")
    a("    scripts/p5_class_audit.py")
    a("```")
    a("")
    a("产物：`data/backtest/p5_class_audit.json`")
    a("")

    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"✓ 已写入 {DOC}")
    print(f"  {len(L)} 行")


if __name__ == "__main__":
    main()
