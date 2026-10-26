#!/usr/bin/env python3
"""汇总所有实验结果 → 生成结论简报（2026-10-27 凌晨）

用途：明早自动推送给用户。扫描 `data/backtest/` 下所有实验结果 JSON，
按「真实盘口径基线 / 各研究方向 / 与目标差距」组织成一份可读的结论。

只读数据，不改任何东西。
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

ROOT = "/Users/rone/Projects/stock-platform"
BT = os.path.join(ROOT, "data", "backtest")
OUT = os.path.join(BT, "CONCLUSION_2026-10-27.md")

#: 用户目标
GOALS = {"sharpe": 1.5, "win_rate": 0.75, "max_dd": 0.20}

#: 已知基线（真实盘口径：lag1 + 可成交性 + 42 因子真生效 + 5238 只 / 3 年）
BASELINE_FILE = "ALL_real_lag1.json"
#: 25 因子版（panel 失效时）曾达到的数字，作为对照
BASELINE_25F = {"sharpe_ratio": 0.880, "total_return": 0.3551,
                 "max_drawdown": -0.1337, "total_trades": 1190, "win_rate": 0.671}


def load(fn: str) -> Dict[str, Any]:
    p = os.path.join(BT, fn)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return {}


def fmt_res(name: str, v: Dict[str, Any]) -> str:
    if not v or "error" in v:
        return f"| {name} | — | — | — | — | (失败) |"
    sr = v.get("sharpe_ratio")
    tr = v.get("total_return")
    md = v.get("max_drawdown")
    tn = v.get("total_trades")
    wr = v.get("win_rate")
    f = lambda x, p=3: ("—" if x is None else f"{x:.{p}f}")
    return (f"| {name} | {f(sr)} | {f(tr*100 if tr is not None else None,2)}% "
            f"| {f(md*100 if md is not None else None,2)}% | {tn or '—'} "
            f"| {f(wr*100 if wr is not None else None,1)}% |")


def collect_research() -> List[tuple]:
    """收集各研究方向的结果文件。"""
    found = []
    patterns = {
        "因子子集搜索": "factor_subset_*.json",
        "多策略合成": "ensemble_*.json",
        "模型线重扫(gbm)": "gbm_search_*.json",
        "开源方法调研实测": "methods_*.json",
    }
    for label, pat in patterns.items():
        for p in sorted(glob.glob(os.path.join(BT, pat))):
            found.append((label, os.path.basename(p)))
    return found


def is_realistic(d: Dict[str, Any]) -> bool:
    """是否真实盘口径。

    `verify_10y_pit.py` 从 2026-10-27 起会在结果里写 `methodology.realistic`。
    **没有这个标记的旧文件一律视为旧口径（lag0 + 无可成交性）→ 不参与排名。**
    旧口径的数字虚高（能买涨停卖跌停），混进排名会得出完全错误的结论。
    """
    m = d.get("methodology")
    if not isinstance(m, dict):
        return False
    return bool(m.get("realistic"))


def scan_best() -> List[tuple]:
    """扫描所有结果 JSON，挑出**真实盘口径**里夏普最高的若干条。"""
    rows = []
    for p in sorted(glob.glob(os.path.join(BT, "*.json"))):
        fn = os.path.basename(p)
        if fn in (BASELINE_FILE,):
            continue
        d = load(fn)
        if not d:
            continue
        res = d.get("results")
        if not isinstance(res, dict):
            continue
        if not is_realistic(d):
            continue
        for k, v in res.items():
            if not isinstance(v, dict) or "error" in v:
                continue
            sr = v.get("sharpe_ratio")
            if sr is None:
                continue
            rows.append((sr, fn, k, v))
    rows.sort(key=lambda x: -x[0])
    return rows


def main() -> None:
    L: List[str] = []
    A = L.append

    A("# 股票平台策略结论简报")
    A("")
    A("生成时间：自动汇总。口径：**3 年真实盘口径**（lag1 + 可成交性检查开 "
      "+ 42 因子真生效 + 5238 只股票池）。")
    A("")

    # ── 1. 基线 ────────────────────────────────────────────
    A("## 一、基线（全策略，真实盘口径）")
    A("")
    base = load(BASELINE_FILE)
    base.setdefault("methodology", {"realistic": True})
    A("| 策略 | 夏普 | 3年收益 | 最大回撤 | 笔数 | 逐笔胜率 |")
    A("|---|---|---|---|---|---|")
    if base.get("results"):
        for k, v in base["results"].items():
            A(fmt_res(k, v))
    else:
        A("| (基线文件缺失) | — | — | — | — | — |")
    A("")

    # ── 2. 各方向结果 ──────────────────────────────────────
    A("## 二、各研究方向结果")
    A("")
    files = collect_research()
    if not files:
        A("（暂无子方向结果文件）")
        A("")
    else:
        by_label = {}
        for label, fn in files:
            by_label.setdefault(label, []).append(fn)
        for label, fns in by_label.items():
            A(f"### {label}")
            A("")
            A("| 变体 | 夏普 | 收益 | 回撤 | 笔数 | 胜率 |")
            A("|---|---|---|---|---|---|")
            for fn in fns:
                d = load(fn)
                res = d.get("results") if isinstance(d, dict) else None
                if isinstance(res, dict):
                    if not is_realistic(d):
                        A(f"| {fn} | — | — | — | — | ⚠️ 旧口径，已作废 |")
                        continue
                    for k, v in res.items():
                        A(fmt_res(f"{k}", v))
                else:
                    A(f"| {fn} | — | — | — | — | (结构不同) |")
            A("")

    # ── 3. 全部结果里夏普最高的 ────────────────────────────
    A("## 三、所有实验里夏普最高的 15 条（含 OOS 标注）")
    A("")
    A("| 夏普 | 结果文件 | 变体 | 收益 | 回撤 | 笔数 |")
    A("|---|---|---|---|---|---|")
    rows = scan_best()
    if not rows:
        A("| — | (无) | — | — | — | — |")
    for sr, fn, k, v in rows[:15]:
        tr = v.get("total_return")
        md = v.get("max_drawdown")
        tn = v.get("total_trades")
        oos = v.get("oos_sharpe") or v.get("oos")
        osfx = f" (OOS {oos:.3f})" if isinstance(oos, (int, float)) else ""
        A(f"| **{sr:.3f}**{osfx} | {fn} | {k} | "
          f"{('%.2f%%' % (tr*100)) if tr is not None else '—'} | "
          f"{('%.2f%%' % (md*100)) if md is not None else '—'} | {tn or '—'} |")
    A("")

    # ── 4. 与目标差距 ──────────────────────────────────────
    A("## 四、与目标的差距")
    A("")
    best = rows[0][0] if rows else None
    if best is None and base.get("results"):
        best = max(v.get("sharpe_ratio") or 0
                   for v in base["results"].values()
                   if isinstance(v, dict) and "error" not in v)
    best_wr = None
    best_dd = None
    for sr, fn, k, v in rows:
        wr = v.get("win_rate")
        md = v.get("max_drawdown")
        if wr is not None and (best_wr is None or wr > best_wr):
            best_wr = wr
        if md is not None and (best_dd is None or md > best_dd):
            best_dd = md
    A("| 目标 | 要求 | **主线基线** | 状态 |")
    A("|---|---|---|---|")
    # 主线基线（不是扫描出来的最大值 —— 扫描值含子集/单段数字，会误导）
    _main = {}
    # baseline_final.json = 25 公式因子主线（2026-10-27 切换后）
    for _fn in ("baseline_final.json", "ALL_real_lag1.json"):
        _d = load(_fn)
        if _d.get("results"):
            for _k, _v in _d["results"].items():
                if isinstance(_v, dict) and "error" not in _v:
                    if _k == "honest_composite":
                        _main = _v
                        break
        if _main:
            break
    if _main:
        _sr = _main.get("sharpe_ratio") or 0
        _wr = _main.get("win_rate") or 0
        _md = _main.get("max_drawdown") or 0
        _ok = "✅" if _sr >= GOALS["sharpe"] else "❌"
        A(f"| 夏普 | ≥ {GOALS['sharpe']} | **{_sr:.3f}** | {_ok} 差 "
          f"{max(0, GOALS['sharpe']-_sr):.3f} |")
        _ok = "✅" if _wr >= GOALS["win_rate"] else "❌"
        A(f"| 逐笔胜率 | ≥ {GOALS['win_rate']*100:.0f}% | **{_wr*100:.1f}%** | {_ok} 差 "
          f"{max(0, (GOALS['win_rate']-_wr)*100):.1f}pp |")
        _ok = "✅" if abs(_md) <= GOALS["max_dd"] else "❌"
        A(f"| 最大回撤 | ≤ {GOALS['max_dd']*100:.0f}% | **{abs(_md)*100:.2f}%** | {_ok} |")
        A("")
        A("（以上是**主线** `honest_composite` 25 公式因子版的真实数字。"
          "下面第三节里更高的夏普来自**因子子集 / 单段窗口 / 候选策略**，"
          "不是主线，不能对外承诺。）")
    A("")
    if best is not None:
        A(f"> 本次全部实验里扫到的最高夏普是 **{best:.3f}**（来自因子子集 / "
          f"候选策略 / 单段窗口），**不是主线水平**，不能对外承诺。")
        A("")

    A("---")
    A("")
    A("说明：表中所有数字都来自**真实盘口径**（lag1 + 可成交性 + 面板因子真生效）。")
    A("旧口径（lag0 + 无可成交性检查）的数字**已全部作废**，不要混用。")
    A("")

    text = "\n".join(L)
    os.makedirs(BT, exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
