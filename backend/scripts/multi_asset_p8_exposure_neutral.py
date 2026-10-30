#!/usr/bin/env python3
"""P8：暴露中性化（size+illiq）在**当前定稿组合**下是负还是正？

## 为什么做这个

P7 判定「因子层证伪」—— 25 个公式因子全用就是最优，任何减法都更差，任何符号翻转也更差。
P7 的推论是：**主腿的弱不是"因子选错了"，问题在信号构造层或执行层**。

信号构造层唯一有既有跨段证据的候选是 **暴露中性化**
（Barra APT 风格逐日截面 OLS 残差化，y=复合分，X=[size, illiq, 1]；
size = 20 日 log 成交额均值（平台无市值数据），illiq = Amihud 20 日均值取 log1p）。

⚠️ **但既有结论互相矛盾，必须用生产口径重新裁决**：

| 口径 | seg1 | seg2 | seg3 | 3 年整体 |
|---|---|---|---|---|
| 研究实现（42f，methods_lab，`neut_segments.json`） | +0.222 | +0.241 | +0.307 | 0.792→1.012 |
| **真实引擎（25f，`neutseg_seg*_*.json`）** | **−0.140** | **+0.200** | **−0.269** | **0.880→0.757** |

策略源码里的注释引用的是**前者（42f 研究实现）**，写着"唯一跨段稳定有效"——
**在生产口径下这是错的**（三段里两段负面，3 年整体夏普 −0.123、回撤恶化 7.1pp）。
本实验用**当前引擎 + 当前定稿组合**重新裁决，并修正那段误导性注释。

## 本实验

- **唯一变量**：`exposure_neutral` 开/关（策略内置开关，默认关）
- 因子集合固定 25 公式全用（= 当前主腿），袖套 P5 固定持仓集，w = 0.50
- 两个窗口各跑 2 臂：base25 / neut25
- 成本与成交口径与 P3/P5/P6/P7 完全一致；base25 必须逐位复现基线门

## 预注册判据（跑之前写死）

  (a) A 股腿自身夏普 > 基线
  (b) main 组合夏普 ≥ 基线 + 0.05（main 窗口专属）
  (c) oos 组合夏普 ≥ 基线 − 0.02（oos 窗口专属）
  (d) 4 个不重叠子窗口中 ≥ 3 个组合夏普 > 基线
  (e) ~~随机对照~~ —— 本实验只有开/关两态，**不存在参数搜索空间**，
      无需随机对照（与 P7 的因子搜索性质不同）。代之以：
  (e') **独立分段证据**：中性化在 3 个独立 1 年段上的方向一致性（既有证据 + 本次重跑）
  (f) 独立样本外证据 —— 见 note

## 用法

    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p8_exposure_neutral.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p8_exposure_neutral.py --window oos
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p2 as P2  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
import multi_asset_leg_experiment as M  # noqa: E402
import multi_asset_p7_factor_ablation as P7  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    load_etf_universe,
    maxdd,
    merge_by_date,
    sharpe,
    total_ret,
    yearly,
)

CLASSES = ["gold", "bond", "commodity"]
W_A_SHARE = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    args = ap.parse_args()

    start, end = (M.START, M.END) if args.window == "main" else (M.OOS_START, M.OOS_END)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    out_path = os.path.join(ROOT, "data", "backtest", f"multi_asset_p8_{args.window}.json")

    print(f"[p8] 窗口 {s_str} → {e_str}", flush=True)

    # ── 袖套（与 P3/P5/P6/P7 同口径）──
    pool = P3.build_pool(start)
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    held = {c: list(P7.P5_FIXED[c]) for c in P7.P5_FIXED}
    _known = set(base_codes) | {r["code"] for r in pool}
    missing = [c for lst in held.values() for c in lst if c not in _known]
    if missing:
        raise SystemExit(f"[p8] ❌ 固定持仓集缺代码 {missing}")

    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    tg, tmeta = P3.build_targets(calendar, held, P3.fixed_weights_for(CLASSES))
    c_of = {c: cls for cls, lst in held.items() for c in lst}
    sleeve_nav, srec = L3.simulate(
        calendar, prices, tg, c_of, {c: 1.0 / len(CLASSES) for c in CLASSES})
    print(f"[p8] 袖套 {len(sleeve_nav)} 天，夏普 {sharpe(sleeve_nav):.4f}", flush=True)

    # ── A 股腿：base25 / neut25 ──
    from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
    codes = universe_from_instruments(QLIB_OFFICIAL)
    # ⚠️ 显式同步模块全局窗口 —— A 股腿用的是 multi_asset_leg_experiment 的全局
    M.START, M.END = start, end

    arms = [
        {"label": "base25", "extra": {}, "desc": "暴露中性化关闭（当前主腿）"},
        {"label": "neut25", "extra": {"exposure_neutral": True},
         "desc": "暴露中性化开启（size+illiq 逐日截面 OLS 残差化）"},
    ]
    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}
    for ai, arm in enumerate(arms, 1):
        t0 = time.perf_counter()
        rec, a_nav = P7.run_leg(codes, list(P7.ALL_FACTORS), {}, arm["extra"],
                                 start_date=start, end_date=end)
        wall = time.perf_counter() - t0
        if not a_nav:
            raise SystemExit(f"[p8] ❌ 臂 {arm['label']} 净值为空")
        if not (s_str <= a_nav[0][0] <= e_str) or not (s_str <= a_nav[-1][0] <= e_str):
            raise SystemExit(f"[p8] ❌ 臂 {arm['label']} A 股腿区间 {a_nav[0][0]}~{a_nav[-1][0]} "
                             f"不在目标窗口 {s_str}~{e_str} 内")
        comb = merge_by_date([a_nav, sleeve_nav], [W_A_SHARE, 1 - W_A_SHARE])
        results[arm["label"]] = {
            "desc": arm["desc"],
            "exposure_neutral": bool(arm["extra"].get("exposure_neutral", False)),
            "a_share_sharpe": round(sharpe(a_nav), 4),
            "a_share_return": round(total_ret(a_nav), 4),
            "a_share_maxdd": round(maxdd(a_nav), 4),
            "a_share_trades": rec["total_trades"],
            "combined_sharpe": round(sharpe(comb), 4),
            "combined_return": round(total_ret(comb), 4),
            "combined_maxdd": round(maxdd(comb), 4),
            "combined_vol": round(P2.nav_vol(comb), 4),
            "combined_yearly": yearly(comb),
            "wall_s": round(wall, 1),
        }
        navs[arm["label"]] = comb
        print(f"  [{ai}/{len(arms)}] {arm['label']:10s} A腿夏普 {sharpe(a_nav):.4f}  "
              f"组合夏普 {sharpe(comb):.4f}  回撤 {maxdd(comb):+.2%}  ({wall:.0f}s)", flush=True)

    # ── 基线复现门 ──
    base_cs = results["base25"]["combined_sharpe"]
    _p3p = os.path.join(ROOT, "data", "backtest", f"multi_asset_p3_{args.window}.json")
    with open(_p3p, encoding="utf-8") as fh:
        _p3res = json.load(fh).get("results", {})
    _p3key = ("bond_lowrho3_noliq" if "bond_lowrho3_noliq" in _p3res else "bond_lowrho3")
    exp = float(_p3res[_p3key]["combined_sharpe"])
    print(f"[p8] 基线复现门: base25 → {base_cs:.4f}（P3 定稿 {exp}）", flush=True)
    if abs(base_cs - exp) > 5e-4:
        raise SystemExit(f"[p8] ❌ 基线漂移 {base_cs} != {exp}，退出")

    # ── 子窗口 ──
    subs = P7.subwindow_compare(navs, "base25", list(navs))
    wins = sum(1 for s in subs.values()
               if s.get("neut25") is not None and s.get("base25") is not None
               and s["neut25"] > s["base25"])

    nb = results["neut25"]
    criteria = {
        "a_ashare_sharpe_gt_base": nb["a_share_sharpe"] > results["base25"]["a_share_sharpe"],
        "b_main_combined_ge_base_plus_0.05":
            ((nb["combined_sharpe"] - base_cs) >= 0.05) if args.window == "main" else None,
        "c_oos_combined_ge_base_minus_0.02":
            ((nb["combined_sharpe"] - base_cs) >= -0.02) if args.window == "oos" else None,
        "d_subwindow_majority_ge_3": wins >= 3,
        "e_random_control": None,       # 只有开/关两态，无参数搜索空间 → 不适用
        "f_independent_oos_evidence": False,
    }

    print(f"\n[p8] === 判据 ===", flush=True)
    for k, v in criteria.items():
        print(f"  {'✅' if v is True else ('—' if v is None else '❌')} {k} = {v}", flush=True)
    print(f"[p8] 子窗口胜数 {wins}/{len(subs)}", flush=True)

    store = {
        "window": args.window,
        "start": s_str,
        "end": e_str,
        "held": held,
        "sleeve": {"sharpe": round(sharpe(sleeve_nav), 4),
                   "vol": round(P2.nav_vol(sleeve_nav), 4),
                   "cost": srec,
                   "n_rebalances": tmeta["n_rebalances"]},
        "w_a_share": W_A_SHARE,
        "base_combined_sharpe": base_cs,
        "p3_expected_base": exp,
        "results": results,
        "subwindows": subs,
        "subwindow_wins": wins,
        "criteria": criteria,
        "delta_combined": round(nb["combined_sharpe"] - base_cs, 4),
        "delta_ashare": round(nb["a_share_sharpe"] - results["base25"]["a_share_sharpe"], 4),
        "note": ("暴露中性化只有开/关两态 → 无参数搜索空间，故无随机对照。"
                 "(f) 独立样本外证据：中性化的既有 3 段独立验证在**真实引擎口径**下"
                 "是 Δ −0.140/+0.200/−0.269（三段里两段负面），指向负面；"
                 "且 oos 窗口（2021-05~2023-05）属 2023-05 之前的 regime。"),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)
    print(f"[p8] ✓ 已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
