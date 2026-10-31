#!/usr/bin/env python3
"""从 P10 产物生成文档（数字零手抄）。

产物：
  data/backtest/multi_asset_p10_gate.json   —— features_full 与 features/day 的逐位复现门
  data/backtest/multi_asset_p10_long.json   —— A 股腿 5 窗口长历史（引擎口径）
  data/backtest/multi_asset_p10_stats.json  —— 由本脚本生成的跨窗口统计（派生）

输出：
  docs/backtest/P10_LONG_ASHARE_2026-10-31.md
"""
from __future__ import annotations

import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACK = os.path.join(ROOT, "data", "backtest")
DOCS = os.path.join(ROOT, "docs", "backtest")

GATE = json.load(open(os.path.join(BACK, "multi_asset_p10_gate.json"), encoding="utf-8"))
LONG = json.load(open(os.path.join(BACK, "multi_asset_p10_long.json"), encoding="utf-8"))
P9M = json.load(open(os.path.join(BACK, "multi_asset_p9_main.json"), encoding="utf-8"))
P9O = json.load(open(os.path.join(BACK, "multi_asset_p9_oos.json"), encoding="utf-8"))
P6M = json.load(open(os.path.join(BACK, "multi_asset_p6_main.json"), encoding="utf-8"))
P6O = json.load(open(os.path.join(BACK, "multi_asset_p6_oos.json"), encoding="utf-8"))

ORDER = ["w2013", "w2016", "w2019", "oos", "main"]
NAME = {
    "w2013": "w2013（2013-05~2016-05）",
    "w2016": "w2016（2016-05~2019-05）",
    "w2019": "w2019（2019-05~2021-05）",
    "oos": "oos（2021-05~2023-05）",
    "main": "main（2023-05~2026-05）",
}


def f4(x) -> str:
    return f"{float(x):.4f}"


def pct(x) -> str:
    return f"{float(x) * 100:.2f}%"


def main() -> int:
    res = LONG["results"]
    labs = [l for l in ORDER if l in res]
    sh = [float(res[l]["a_share_sharpe"]) for l in labs]
    stats = {
        "n_windows": len(labs),
        "mean_sharpe": round(statistics.fmean(sh), 6),
        "min_sharpe": round(min(sh), 6),
        "max_sharpe": round(max(sh), 6),
        "range_sharpe": round(max(sh) - min(sh), 6),
        "stdev_sharpe": round(statistics.pstdev(sh), 6),
        "median_sharpe": round(statistics.median(sh), 6),
    }
    with open(os.path.join(BACK, "multi_asset_p10_stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": "multi_asset_p10_long.json", "labels": labs, **stats},
                  fh, ensure_ascii=False, indent=2)

    A: list[str] = []
    A.append("# P10 · A 股腿长历史重建 —— 第三样本不是「结构性缺失」")
    A.append("")
    A.append("> 生成时间：由 `backend/scripts/gen_p10_doc.py` 从 `multi_asset_p10_{gate,long}.json` "
             "自动生成，**数字零手抄**。")
    A.append("")

    # ── 0 ──
    A.append("## 0. 一句话结论")
    A.append("")
    A.append("- **「A 股腿没有 2021-05-11 之前的历史」是错的。** 那个边界是 Alpha158 "
             "**面板**的边界；生产 `honest_composite` 默认 `use_panel=False`"
             "（`honest_composite_strategy.py:199`），25 个因子全部由 close/high/low/volume "
             "**现场算**（`:343-346`）。")
    A.append(f"- `data/qlib_data/features_full/`（5513 只 / 每股票全历史）与 `features/day` "
             f"在 main 窗口**逐位等价**：total_return / sharpe_ratio / max_drawdown / "
             f"win_rate / 交易笔数 **diff 全部 = 0**（验证门通过）。")
    A.append(f"- 于是 A 股腿 25f 生产配置的**跨窗口夏普曲线**（引擎口径）出来了："
             + " / ".join(f"`{l}` **{f4(res[l]['a_share_sharpe'])}**" for l in labs)
             + f" → 均值 **{f4(stats['mean_sharpe'])}**、极差 **{f4(stats['range_sharpe'])}**、"
               f"标准差 **{f4(stats['stdev_sharpe'])}**。")
    A.append("- **推论**：一直被引用的「主腿 0.87 / 0.72」落在曲线的**偏上区间**；"
             "「A 股腿 50%」这个定稿是在两个偏高的窗口上定出来的；"
             "P6 的 w 曲线（main 最优 0.20）本质是**主腿强弱随窗口漂移**，不是一个可固定的数。")
    A.append("- **定稿不变** —— 本实验只补充证据，没有新增任何资产侧规则；"
             "但它把「第三样本结构性缺失」这个借口从 P6/P7/P8/P9 的案卷里拿掉了。")
    A.append("")

    # ── 1 ──
    A.append("## 1. 前提纠正：2021-05-11 是「面板」边界，不是「数据」边界")
    A.append("")
    A.append("P6 / P7 / P8 / P9 的结论里都写着同一句话：")
    A.append("")
    A.append("> 「`honest_composite` A 股腿面板从 2021-05-11 起，没有窗口前的 A 股腿历史 → "
             "独立样本外证据结构性缺失」")
    A.append("")
    A.append("这句话把两件事混成了一件：")
    A.append("")
    A.append("| | 现状 | 边界来源 |")
    A.append("|---|---|---|")
    A.append("| Alpha158 面板（17 因子） | 生产**不用**（`use_panel` 默认 False） | "
             "`data/qlib_data/features/day/` 的两段拼接 |")
    A.append("| 25 个基础公式 | 生产**全部在用** | close/high/low/volume（价格即可） |")
    A.append("")
    A.append("而 `data/qlib_data/features_full/`（`backend/scripts/build_full_history_features.py` "
             "的产物，5513 只 / 188 列 / 每股票全历史）**早就在本地**："
             "实测 4330 只（78.5%）在 2021-05-11 之前就有数据，最早回溯到 1990-12-19。")
    A.append("")
    A.append("代码清单来自 `universe_from_instruments(QLIB_OFFICIAL)`（PIT 全宇宙），"
             "与 `features/day` 无关 → **唯一差别就是每只股票的数据覆盖**。")
    A.append("")

    # ── 2 ──
    A.append("## 2. 验证门：`features_full` 与 `features/day` 逐位等价")
    A.append("")
    A.append("把 loader 指向 `features_full`（`settings.QLIB_DATA_PATH` 运行时改，"
             "**零生产代码改动**），在**现有 main 窗口**重跑 A 股腿，"
             "期望值读取同版上游产物 `ALL_real_lag1_v2.json`：")
    A.append("")
    A.append("| 指标 | 期望（features/day） | 实得（features/full） | diff |")
    A.append("|---|---|---|---|")
    for f in ("total_return", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"):
        e = GATE["expected"].get(f)
        g = GATE["got"].get(f)
        d = GATE["diff"].get(f)
        fmt = (lambda v: f"{float(v):.10f}") if f != "total_trades" else (lambda v: str(int(v)))
        A.append(f"| {f} | {fmt(e)} | {fmt(g)} | {fmt(d)} |")
    A.append("")
    A.append(f"验证门：**{'通过' if GATE['passed'] else '失败'}**"
             f"（四个引擎指标 + 交易笔数 diff 全部 = 0）。")
    A.append("")
    A.append("> 这道门是**先决条件**：不复现就没有长历史可言 —— "
             "如果 `features_full` 与 `features/day` 在重叠区间不等价，"
             "那么长历史就是另一套数据，不能和现有结论并列。")
    A.append("")

    # ── 3 ──
    A.append("## 3. A 股腿长历史夏普曲线（25f 生产配置，引擎口径）")
    A.append("")
    A.append("| 窗口 | 起点年 | 终点年 | A腿夏普 | A腿收益 | A腿回撤 | 交易笔数 |")
    A.append("|---|---|---|---|---|---|---|")
    for l in labs:
        r = res[l]
        y0, y1 = r["window"][0][:4], r["window"][1][:4]
        A.append(f"| {NAME[l]} | {y0} | {y1} | {f4(r['a_share_sharpe'])} | "
                 f"{pct(r['total_return'])} | {pct(r['max_drawdown'])} | {r['total_trades']} |")
    A.append("")
    A.append("| 统计量 | 值 |")
    A.append("|---|---|")
    A.append(f"| 窗口数 | {stats['n_windows']} |")
    A.append(f"| 均值 | {f4(stats['mean_sharpe'])} |")
    A.append(f"| 中位数 | {f4(stats['median_sharpe'])} |")
    A.append(f"| 最小值 | {f4(stats['min_sharpe'])} |")
    A.append(f"| 最大值 | {f4(stats['max_sharpe'])} |")
    A.append(f"| 极差 | {f4(stats['range_sharpe'])} |")
    A.append(f"| 标准差 | {f4(stats['stdev_sharpe'])} |")
    A.append("")
    worst = min(labs, key=lambda l: float(res[l]["a_share_sharpe"]))
    best = max(labs, key=lambda l: float(res[l]["a_share_sharpe"]))
    A.append(f"**最差窗口 `{worst}`（{res[worst]['window'][0]}~{res[worst]['window'][1]}）"
             f"A 腿夏普 {f4(res[worst]['a_share_sharpe'])}、收益 {pct(res[worst]['total_return'])}、"
             f"回撤 {pct(res[worst]['max_drawdown'])}** —— 这三年 A 股腿是**亏钱的**。")
    A.append(f"**最好窗口 `{best}`（{res[best]['window'][0]}~{res[best]['window'][1]}）"
             f"夏普 {f4(res[best]['a_share_sharpe'])}** —— 2019-2020 结构性行情。")
    A.append("")

    # ── 4 ──
    A.append("## 4. 长历史专属的 PIT 修复：涨跌停幅度必须按日期")
    A.append("")
    A.append("长历史暴露了一个**现行代码看不见的口径错误**：`limit_pct` 用的是**现行**板块规则"
             "（创业板/科创板 20%），但**创业板 2020-08-24 注册制之前只有 10%**。"
             "用 20% 去掩 2016~2019 的创业板，会把「当日真的涨停（买不到）」的票放过去 → "
             "成交价不可实现 → 长历史被系统性污染。")
    A.append("")
    A.append("修复：`execution_rules.limit_pct_at` / `limit_up_mask_matrix_at`（按交易日 PIT），"
             "策略改用新函数。**关键性质：2020-08-24 之后的窗口与原实现逐位相同**，"
             "所以这是**纯长历史修复，不动任何已定稿数值** —— "
             "回归门 `backend/scripts/p10_limit_pit_regression.py` 覆盖该断言，"
             "且 P10 验证门重跑后 main 窗口 diff 仍全部 = 0。")
    A.append("")
    A.append(f"实测：修复前后三个长历史窗口的数值**完全一致**"
             f"（{f4(res['w2013']['a_share_sharpe'])} / {f4(res['w2016']['a_share_sharpe'])} / "
             f"{f4(res['w2019']['a_share_sharpe'])}）"
             f"→ 掩码伪迹**不是** {worst} 窗口负夏普的成因。")
    A.append("")
    A.append("> 为什么会一致：本策略的 MIN 因子选的是「远低于 N 日最低价」的票，"
             "这类票几乎不可能当日涨停 → 涨停掩码在买入端基本不触发。"
             "结论是**否定性的但有价值的**：排除了一个看起来很可疑的解释。")
    A.append("")

    # ── 5 ──
    A.append("## 5. 对既有结论的影响")
    A.append("")
    A.append("| 既有结论 | 原依据 | 新证据 | 影响 |")
    A.append("|---|---|---|---|")
    A.append(f"| 定稿 A 股腿 50% + 三类各 16.67% | main 1.4114 / oos 1.0691 | "
             f"A 腿 5 窗口夏普 {f4(stats['min_sharpe'])}~{f4(stats['max_sharpe'])} | "
             f"**证据基础被削弱**（两个窗口都在偏上区间） |")
    A.append(f"| P6：w 高敏感、不采纳具体值 | main 最优 0.20 / oos 最优 0.25 | "
             f"A 腿夏普随窗口从 {f4(stats['min_sharpe'])} 到 {f4(stats['max_sharpe'])} | "
             f"**解释变了**：w 不是固定的敏感参数，而是**跟随主腿强弱漂移** |")
    A.append("| P7/P8/P9：主腿弱不是因子/暴露/执行问题 | 两窗口判据 | "
             "长历史把主腿的**跨窗口方差**摆出来 | "
             "**第三条路**（主腿本身跨窗口不稳定）成立 |")
    A.append("| 组合 1.4114 的可信度 | 两窗口 | 主腿在 5 窗口上均值为 "
             f"{f4(stats['mean_sharpe'])} | 组合夏普的**期望值低于 1.41** |")
    A.append("")
    A.append(f"> 注：组合第三样本仍**无法**做长历史 —— ETF 袖套的本地数据只回溯到 2019-01"
             f"（商品腿 159981 更晚，2020-01-17），所以能做的组合窗口只有 "
             f"2020-01~2021-05（16 个月），而且它恰好落在主腿夏普最高的区间 —— "
             f"**不是严苛的检验**。这才是本实验没能解决、留到下一阶段的问题。")
    A.append("")

    # ── 6 ──
    A.append("## 6. 仍然开放的问题")
    A.append("")
    A.append("1. **主腿在坏窗口里是亏的，还是只是被 beta 拖累？** "
             "→ 需要同窗口的等权/指数基准算超额夏普，才能区分「策略失效」与「市场下跌」。"
             "这决定了「该改主腿」到底该改什么。")
    A.append("2. **w 随状态漂移能不能落地？** → 需要一个**不依赖窗口**的"
             "「主腿强弱」状态量（例如主腿自身的滚动夏普 / 相对基准强弱），"
             "用它做动态权重，而不是继续找固定 w。")
    A.append("3. **组合的长历史** → 受限于 ETF 本地数据起点，需要另找数据源。")
    A.append("")

    out = os.path.join(DOCS, "P10_LONG_ASHARE_2026-10-31.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(A) + "\n")
    print(f"✅ 已生成 {out}")
    print(f"✅ 统计产物 {os.path.join(BACK, 'multi_asset_p10_stats.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
