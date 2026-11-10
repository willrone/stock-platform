#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P15 · past_j 在长历史 3 窗口的独立复现 + 上界结论的跨窗口复制

P14 结论（main/oos 两窗口）：
  1. 袖套资产级权重优化在**可实现口径**下证伪（past_s 双窗口败）
  2. 决定性证据是**上界本身低**：oos 的 oracle_s 只有 1.3262，仅比随机族最优高 0.135
  3. `past_j`（放开 A 股腿权重）看起来两窗口都赢，但赢的方式是**低 β / 低波**
     （main 年化 6.4% vs 定稿 12.4%），与 P12 的 E1 同源，不推翻 P12

P15 做两件事：

**任务 A —— past_j 的独立复现（P14 的待办 1）**
在 P12 已建好的长历史窗口 w2013 / w2016 / w2019（降级 gold+bond 两类袖套）上，
用**完全相同**的优化器 + 估计窗口（信号日之前 252 交易日）跑 past_s / past_j。

⚠️ 长历史窗口的**降级配置**：commodity 三只 ETF 物理上无 2019 前数据 →
   袖套只有 gold + bond 两类、每类 1~3 只。**不能外推为定稿配置的业绩预期**，
   只能验证「可实现口径的权重优化在长历史上是否仍成立」。

**任务 B —— 上界结论的跨窗口复制**
在同样 3 个窗口跑 `oracle_s`（未来 252 天的 μ/Σ），
检验 P14 的核心断言「上界本身低」是否只出现在 main/oos，还是跨窗口成立。

**预注册判据**（事前写死，不看结果改）：
  (a) 三窗口**一致**优于各自定稿（全窗口口径或 trim 口径都算）
  (b) **绝对收益不显著低于定稿**：年化收益 ≥ 定稿 × 0.9
      ← 只赢夏普不算赢（低 β 假象）
  (c) 不是靠压低风险换来的：组合年化 σ ≥ 定稿 × 0.7
  任一不满足 → past_j 正式封死，连同 A 腿权重自由度一起。

用法：
    env -u PYTHONPATH .venv-py313/bin/python3 scripts/multi_asset_p15_long_pastj.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_p3_bond as P3            # noqa: E402
import multi_asset_p12_alpha_weight as P12   # noqa: E402
import multi_asset_p13_asset_weight as P13   # noqa: E402
import multi_asset_p14_oracle_bound as P14   # noqa: E402
from multi_asset_leg_experiment import (      # noqa: E402
    maxdd, merge_by_date, sharpe,
)

OUT_DIR = os.path.join(ROOT, "data", "backtest")

LONG_WINDOWS = [
    # 归档值 = P12 multi_asset_p12_long.json（撮合器已对齐 L3 后重跑，2026-10-31）
    ("w2013", datetime(2013, 5, 8), datetime(2016, 5, 8), 0.7780),
    ("w2016", datetime(2016, 5, 8), datetime(2019, 5, 8), -0.0936),
    ("w2019", datetime(2019, 5, 8), datetime(2021, 5, 11), 1.0448),
]

EST_HORIZON = P14.ORACLE_HORIZON   # 252
MIN_OBS = P14.MIN_ORACLE_OBS        # 60
SINGLE_CAP = P14.SINGLE_CAP         # 0.40
TRIM_PERIODS = (6, 12)              # trim 口径（与 P14 一致）


# ───────────────────────── 正确撮合器 ─────────────────────────
# ⚠️ P12 的 `simulate_long` **缺少 L3.simulate 的第 4 步**（"卖出超配的部分"），
#    买入段又是 `shares[code] = shares.get(code,0)+sh`（只加仓、从不减仓）
#    → **目标权重永远达不到**，「每月恢复类等权」实际没发生，等价于买入持有 + 加仓。
#    L3.simulate 是完整 4 步（卖非目标 → pro-rata 买不足 → 卖超配 → 收盘估值）。
#    这里按 L3.simulate 逐行复制到长历史价格上。


# ── 撮合器：单一事实来源 ─────────────────────────────────────────────
# ⚠️ 2026-10-31：原先这里另抄了一份 L3.simulate 的 4 步实现
#    (simulate_long_correct)，与 P12.simulate_long 分叉
#    （同一天同目标权重，w2013 组合夏普 0.7780 vs 0.7727）。
#    两套撮合器 = 未来必然再次对不上账。现已统一：
#    P12.simulate_long 已按 L3.simulate 原样对齐（含 pro-rata 买不足），
#    P15 直接复用它，本文件不再保留任何副本。
simulate_long_correct = P12.simulate_long

def load_window(label: str, start: datetime, end: datetime) -> Dict[str, Any]:
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    long_codes = sorted({c for lst in P12.LONG_FIXED.values() for c in lst})
    prices = P12.load_long_prices(long_codes)
    missing = [c for c in long_codes if c not in prices]
    if missing:
        raise SystemExit(f"[p15] ❌ 长历史 ETF 缺数据: {missing}")

    calendar = P12.build_calendar_from(prices, long_codes, s_str, e_str)
    if len(calendar) < 300:
        raise SystemExit(f"[p15] ❌ 日历过短 {len(calendar)}")

    # 类内可用性（某些标的窗口内才上市）
    avail: Dict[str, List[str]] = {}
    for c, lst in P12.LONG_FIXED.items():
        ok = [x for x in lst if any(d in prices[x] for d in calendar)]
        if not ok:
            raise SystemExit(f"[p15] ❌ 类 {c} 在 {label} 内无可用标的")
        avail[c] = ok
    codes = sorted({c for lst in avail.values() for c in lst})
    class_of = {c: cl for cl, lst in avail.items() for c in lst}

    # A 股腿（复用 P11 长历史产物）
    p11p = os.path.join(OUT_DIR, "multi_asset_p11_alpha.json")
    with open(p11p, encoding="utf-8") as fh:
        p11 = json.load(fh)
    v = (p11.get("results") or {}).get(label)
    if not v or not v.get("leg_nav_pairs"):
        raise SystemExit(f"[p15] ❌ P11 产物缺 {label} 的 A 腿净值")
    a_nav = [(d, float(x)) for d, x in v["leg_nav_pairs"]]

    # 估计矩阵：窗口前 500 自然日 + 窗口内
    est_start = (start - timedelta(days=500)).strftime("%Y-%m-%d")
    est_cal = sorted({d for c in codes for d in prices.get(c, {})
                      if est_start <= d <= e_str} | set(calendar))
    R_est, _ = P13.build_returns(est_cal, prices, codes)

    # 所有标的都上市后的起点（前值填充后仍 NaN 的行 = 上市前）
    first_valid = []
    for j in range(len(codes)):
        col = R_est[:, j]
        idx = np.where(np.isfinite(col))[0]
        first_valid.append(int(idx[0]) if len(idx) else 10 ** 9)
    est_valid_from = max(first_valid) if first_valid else 0

    # A 腿日收益对齐 est_cal（A 腿 NAV 只覆盖窗口内 → 之前为 NaN）
    amap = dict(a_nav)
    a_ret_est = np.full(len(est_cal), np.nan)
    for k in range(1, len(est_cal)):
        d0, d1 = est_cal[k - 1], est_cal[k]
        if d0 in amap and d1 in amap and amap[d0] > 0:
            a_ret_est[k] = amap[d1] / amap[d0] - 1.0

    return {
        "label": label, "start": s_str, "end": e_str,
        "prices": prices, "calendar": calendar, "avail": avail,
        "codes": codes, "class_of": class_of, "a_nav": a_nav,
        "est_cal": est_cal, "R_est": R_est, "a_ret_est": a_ret_est,
        "est_pos": {d: i for i, d in enumerate(est_cal)},
        "est_valid_from": est_valid_from,
    }


# ───────────────────────── 权重调度 ─────────────────────────

def _equal_class_weights(w: Dict[str, Any]) -> Dict[str, float]:
    """类间等权 × 类内等权（与 P12.build_sleeve 完全同口径）。

    ⚠️ 不能用「每只 1/len(codes)」：长历史 held 每类标的数不等
    （w2013 bond 只有 1 只），必须按类分配。
    """
    class_of = w["class_of"]
    n_cls = len({class_of[c] for c in w["codes"]})
    out: Dict[str, float] = {}
    for c in w["codes"]:
        cls = class_of[c]
        n_in = len(w["avail"][cls])
        out[c] = (1.0 / n_cls) / n_in
    return out


def make_schedule(w: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """mode: 'past_s' | 'past_j' | 'oracle_s'"""
    calendar = w["calendar"]
    R = w["R_est"]
    a_ret = w["a_ret_est"]
    codes = w["codes"]
    free_a = (mode == "past_j")
    use_past = mode.startswith("past")
    est_valid_from = w["est_valid_from"]

    sched: Dict[str, Dict[str, float]] = {}
    a_sched: Dict[str, float] = {}
    exec_of: Dict[str, str] = {}
    fallback = 0
    solved = 0
    est_sharpes: List[float] = []

    eq_w = _equal_class_weights(w)

    for i in range(1, len(calendar)):
        if (i - 1) % P12.REB_EVERY != 0:
            continue
        signal_date = calendar[i - 1]
        exec_date = calendar[i]
        p = w["est_pos"].get(signal_date)
        if p is None:
            continue

        if use_past:
            hi = p
            lo = max(est_valid_from, p - EST_HORIZON)
        else:
            lo = p - 1                       # 未来收益从信号日收盘起
            hi = min(len(R), lo + EST_HORIZON)
        if hi <= lo:
            sched[signal_date] = dict(eq_w)
            a_sched[signal_date] = P14.A_LEG_W
            exec_of[signal_date] = exec_date
            fallback += 1
            continue

        seg_s = R[lo:hi]
        seg_a = a_ret[lo:hi]
        ok = np.isfinite(seg_a) & np.all(np.isfinite(seg_s), axis=1)
        seg_s = seg_s[ok]
        seg_a = seg_a[ok]
        if seg_s.shape[0] < MIN_OBS:
            sched[signal_date] = dict(eq_w)
            a_sched[signal_date] = P14.A_LEG_W
            exec_of[signal_date] = exec_date
            fallback += 1
            continue

        mu = np.mean(seg_s, axis=0)
        Sigma = np.atleast_2d(np.cov(seg_s, rowvar=False))
        Sigma = np.nan_to_num(Sigma, nan=0.0)
        cvec = np.array([np.cov(seg_s[:, j], seg_a)[0, 1] for j in range(len(codes))])
        mu_a = float(np.mean(seg_a))
        var_a = float(np.var(seg_a, ddof=1))

        x = P14._solve_oracle(mu, Sigma, cvec, mu_a, var_a, free_a_leg=free_a)
        if x is None:
            sched[signal_date] = dict(eq_w)
            a_sched[signal_date] = P14.A_LEG_W
            exec_of[signal_date] = exec_date
            fallback += 1
            continue
        wv = x[:len(codes)]
        wa = float(x[len(codes)]) if free_a else P14.A_LEG_W

        sched[signal_date] = {c: float(wv[j]) for j, c in enumerate(codes)}
        a_sched[signal_date] = wa
        exec_of[signal_date] = exec_date
        solved += 1
        try:
            est_sharpes.append(float(
                P14._sharpe_obj(wv, wa, mu, Sigma, cvec, mu_a, var_a) * math.sqrt(252)))
        except Exception:
            pass

    return {"sched": sched, "a_sched": a_sched, "exec_of": exec_of,
            "n_solved": solved, "n_fallback": fallback,
            "est_sharpe_mean": (round(float(np.mean(est_sharpes)), 4)
                                if est_sharpes else None)}


def eval_schedule(w: Dict[str, Any], sc: Dict[str, Any]) -> Dict[str, Any]:
    calendar = w["calendar"]
    prices = w["prices"]
    codes = w["codes"]
    sched = sc["sched"]

    targets: Dict[str, Dict[str, float]] = {}
    for signal_date, cw in sched.items():
        s = sum(cw.values())
        if s <= 0:
            continue
        targets[sc["exec_of"][signal_date]] = {c: v / s for c, v in cw.items() if v > 0}
    sleeve_nav, srec = simulate_long_correct(calendar, prices, targets)
    if not sleeve_nav:
        raise SystemExit("[p15] ❌ 袖套模拟为空")

    if all(abs(sc["a_sched"].get(k, P14.A_LEG_W) - P14.A_LEG_W) < 1e-9
           for k in sc["a_sched"]):
        comb = merge_by_date([w["a_nav"], sleeve_nav], [P14.A_LEG_W, 1.0 - P14.A_LEG_W])
    else:
        comb = P14.merge_variable(w["a_nav"], sleeve_nav, sc["a_sched"], sc["exec_of"])

    rets = _daily_rets(comb)
    sd = float(rets.std(ddof=1)) if len(rets) > 2 else 0.0
    return {
        "sleeve_sharpe": round(sharpe(sleeve_nav), 4),
        "combined_sharpe": round(sharpe(comb), 4),
        "ann_return": round(float(rets.mean() * 252), 6),
        "ann_vol": round(sd * math.sqrt(252), 6),
        "total_return": round(comb[-1][1] / comb[0][1] - 1.0, 6),
        "combined_maxdd": round(maxdd(comb), 6),
        "turnover_ratio": srec["turnover_ratio"],
        "n_rebalances": srec["n_rebalances"],
        "a_leg_weight_avg": round(
            float(np.mean([sc["a_sched"].get(r, P14.A_LEG_W)
                           for r in sched])), 4) if sched else P14.A_LEG_W,
        "_rets": rets,
    }


def _daily_rets(nav: List[Tuple[str, float]]) -> np.ndarray:
    v = np.array([x for _, x in nav], dtype=float)
    if len(v) < 3:
        return np.zeros(1)
    return v[1:] / v[:-1] - 1.0


def _trim_stats(rets: np.ndarray, cut: int) -> Optional[Dict[str, float]]:
    if cut >= len(rets) - 120:
        return None
    seg = rets[cut:]
    sd = seg.std(ddof=1)
    sh = (seg.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0
    return {"sharpe": round(float(sh), 4),
            "n_days": int(len(seg)),
            "ann_return": round(float(seg.mean() * 252), 6),
            "ann_vol": round(float(sd * math.sqrt(252)), 6)}


# ───────────────────────── 主流程 ─────────────────────────

def main() -> None:
    out: Dict[str, Any] = {
        "phase": "long", "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": "DEGRADED gold+bond only（commodity 无 2019 前数据）"
                  "—— 不能外推为定稿配置的业绩预期",
        "note": "past_j / past_s 的估计窗口 = 信号日之前 252 交易日（完全可实现）；"
                "oracle_s 用未来 252 天（含未来信息，仅用于上界定标）。"
                "长历史窗口 A 股腿 NAV 只覆盖窗口内 → 早期调仓期有效观测不足，"
                "按规则退化为等权，故必须同时看 trim 口径。",
        "criteria": {
            "a_three_windows_consistent_better": "三窗口一致优于各自定稿",
            "b_ann_return_ge_90pct_of_final": "年化收益 ≥ 定稿 × 0.9（只赢夏普不算赢）",
            "c_ann_vol_ge_70pct_of_final": "组合年化 σ ≥ 定稿 × 0.7（排除纯低 β）",
        },
        "results": {},
    }

    verdict_rows: List[Dict[str, Any]] = []

    for label, start, end, p12_combo in LONG_WINDOWS:
        print(f"\n===== {label} {start:%Y-%m-%d}~{end:%Y-%m-%d} =====", flush=True)
        w = load_window(label, start, end)
        print(f"  held={json.dumps(w['avail'], ensure_ascii=False)} "
              f"日历 {len(w['calendar'])} 天 | est {len(w['est_cal'])} 天 "
              f"(valid_from={w['est_valid_from']})", flush=True)

        # ── 基线：直接用 P12.build_sleeve（其内部走 P12.simulate_long，
        #    已于 2026-10-31 与 L3.simulate 的 4 步逐行对齐）。
        #    必须逐位复现 P12 归档的 combo_sharpe，否则本窗口结果不可比。
        #    ⚠️ 2026-10-31：原先这里有「基线A(P12 旧撮合器)」与
        #    「基线B(P15 自带的 L3 副本)」两套，两套数值不同
        #    （w2013 0.8077 vs 0.7780）→ 已统一为一套。
        sleeve_base, rec_base = P12.build_sleeve(w["calendar"], w["prices"],
                                                  w["avail"], P12.LONG_CLASSES)
        base = merge_by_date([w["a_nav"], sleeve_base], [0.5, 0.5])
        base_sh = round(sharpe(base), 4)
        gate_ok = abs(base_sh - p12_combo) <= 5e-4
        print(f"  基线(P12 归档值复现门): 组合夏普 {base_sh} "
              f"(P12 {p12_combo}) → {'✅' if gate_ok else '❌'}"
              f" | 换手 {rec_base['turnover_ratio']:.4f}x", flush=True)
        if not gate_ok:
            raise SystemExit("[p15] ❌ 基线未复现 P12 归档值，全部结果不可比，退出")

        base_rets = _daily_rets(base)
        base_ann_ret = float(base_rets.mean() * 252)
        base_ann_vol = float(base_rets.std(ddof=1) * math.sqrt(252))
        base_trim = {c: _trim_stats(base_rets, c * P12.REB_EVERY)
                     for c in TRIM_PERIODS}

        entry: Dict[str, Any] = {
            "window": [w["start"], w["end"]],
            "held": w["avail"],
            "n_calendar": len(w["calendar"]),
            "baseline": {
                "combo_sharpe": base_sh,
                "expected_from_p12": p12_combo,
                "gate_ok": gate_ok,
                "matcher": "P12.simulate_long (与 L3.simulate 4 步逐行对齐)",
                "turnover_ratio": rec_base["turnover_ratio"],
                "ann_return": round(base_ann_ret, 6),
                "ann_vol": round(base_ann_vol, 6),
                "total_return": round(base[-1][1] / base[0][1] - 1.0, 6),
                "trim": base_trim,
            },
            "modes": {},
        }

        for mode in ("past_s", "past_j", "oracle_s"):
            sc = make_schedule(w, mode)
            m = eval_schedule(w, sc)
            trim = {c: _trim_stats(m["_rets"], c * P12.REB_EVERY) for c in TRIM_PERIODS}
            m.pop("_rets", None)
            m["n_solved"] = sc["n_solved"]
            m["n_fallback"] = sc["n_fallback"]
            m["est_sharpe_mean"] = sc["est_sharpe_mean"]
            m["trim"] = trim
            n_total = sc["n_solved"] + sc["n_fallback"]
            m["n_periods_total"] = n_total
            m["optimized_share"] = round(sc["n_solved"] / max(1, n_total), 4)
            entry["modes"][mode] = m
            print(f"  {mode:10s} 组合夏普 {m['combined_sharpe']:8.4f} "
                  f"(定稿 {base_sh:.4f}) | 年化收益 {m['ann_return']:+.4f} "
                  f"(定稿 {base_ann_ret:+.4f}) | 年化σ {m['ann_vol']:.4f} "
                  f"(定稿 {base_ann_vol:.4f}) | A腿 {m['a_leg_weight_avg']:.3f} "
                  f"| 换手 {m['turnover_ratio']:.2f}x "
                  f"| fallback {m['n_fallback']}/{sc['n_solved']+sc['n_fallback']}",
                  flush=True)

        # 窗口可用性：真正跑过优化的调仓期占比不足一半 → 该窗口不具备检验力
        share = max(entry["modes"][m]["optimized_share"]
                    for m in ("past_s", "past_j"))
        entry["usable"] = bool(share >= 0.5)
        entry["optimized_share_max"] = share
        if not entry["usable"]:
            entry["unusable_reason"] = (
                f"仅 {share:.0%} 的调仓期有足够历史做估计（其余退化为等权）"
                f" → 该窗口无法检验权重优化")

        out["results"][label] = entry
        verdict_rows.append({
            "label": label,
            "usable": entry["usable"],
            "optimized_share": share,
            "base_sharpe": base_sh,
            "base_ann_return": round(base_ann_ret, 6),
            "base_ann_vol": round(base_ann_vol, 6),
            "past_s": entry["modes"]["past_s"]["combined_sharpe"],
            "past_j": entry["modes"]["past_j"]["combined_sharpe"],
            "oracle_s": entry["modes"]["oracle_s"]["combined_sharpe"],
            "past_j_ann_return": entry["modes"]["past_j"]["ann_return"],
            "past_j_ann_vol": entry["modes"]["past_j"]["ann_vol"],
            "past_j_a_leg_w": entry["modes"]["past_j"]["a_leg_weight_avg"],
        })

    # ── 跨窗口汇总与判决 ──
    usable = [r for r in verdict_rows if r["usable"]]
    a_legs = [r["past_j_a_leg_w"] for r in verdict_rows]
    out["summary"] = {
        "rows": verdict_rows,
        "n_usable_windows": len(usable),
        "usable_labels": [r["label"] for r in usable],
        "past_j_a_leg_weights_across_windows": a_legs,
        "past_j_a_leg_spread": round(max(a_legs) - min(a_legs), 4),
    }

    # 判据 (a)：可用窗口上 past_j 一致优于定稿？
    wins = [r for r in usable if r["past_j"] > r["base_sharpe"]]
    crit_a = bool(usable) and len(wins) == len(usable)
    # 判据 (b)：绝对收益不显著低于定稿（年化 ≥ 定稿 × 0.9）
    crit_b = all(r["past_j_ann_return"] >= 0.9 * r["base_ann_return"]
                 for r in usable) if usable else False
    # 判据 (c)：不是靠压低风险换来的（σ ≥ 定稿 × 0.7）
    crit_c = all(r["past_j_ann_vol"] >= 0.7 * r["base_ann_vol"]
                 for r in usable) if usable else False
    out["criteria_result"] = {
        "a_three_windows_consistent_better": crit_a,
        "b_ann_return_ge_90pct_of_final": crit_b,
        "c_ann_vol_ge_70pct_of_final": crit_c,
        "all_pass": bool(crit_a and crit_b and crit_c),
        "verdict": ("past_j 通过全部预注册判据 → 值得进下一轮独立验证"
                    if (crit_a and crit_b and crit_c)
                    else "past_j 未通过预注册判据 → 正式封死"),
    }
    print("\n===== 跨窗口汇总 =====")
    for r in verdict_rows:
        print(f"  {r['label']} usable={r['usable']} "
              f"(优化期 {r['optimized_share']:.0%}) | 定稿 {r['base_sharpe']:.4f} "
              f"past_s {r['past_s']:.4f} past_j {r['past_j']:.4f} "
              f"oracle_s {r['oracle_s']:.4f} | A腿w {r['past_j_a_leg_w']:.3f}")
    print(f"  past_j 的 A 腿权重跨窗口: {a_legs} 跨度 {out['summary']['past_j_a_leg_spread']}")
    print(f"  判据 (a) 一致更优: {crit_a} | (b) 绝对收益: {crit_b} | (c) 波动下限: {crit_c}")
    print(f"  → {out['criteria_result']['verdict']}")

    with open(os.path.join(OUT_DIR, "multi_asset_p15_long_pastj.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n[p15] ✓ 已写入 {OUT_DIR}/multi_asset_p15_long_pastj.json", flush=True)


if __name__ == "__main__":
    main()
