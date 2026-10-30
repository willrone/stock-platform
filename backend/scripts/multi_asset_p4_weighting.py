#!/usr/bin/env python3
"""P4 · 类间配权（在 P3 修正后的定稿上重做）。

## 为什么 P2 的结论不能直接继承

P2 证伪了「类间逆波动率配权」，机制是：
  bond 腿年化波动最低(3.3%) → 逆波动率把仓位压向 bond(68.6%)
  → 但 bond 与 A 股腿 ρ 最高(0.55/0.62) → **反分散** → 组合夏普反降。

**P3 把这个前提抽掉了**：bond 类已换成固定长久期利率债
`{511260, 511010, 511020}`，与 A 股腿 ρ = **−0.216 / −0.054**（负相关）。
"低波 → 高相关 → 反分散"这条链条现在断了。**机制讲得通不等于有效，必须重跑。**

## 预注册判据（跑之前写死，不许事后改）

一个配权方案要**被采纳**，必须同时满足：
1. main 组合夏普 ≥ base + 0.05
2. oos 组合夏普 ≥ base − 0.02（不得显著变差）
3. 同参数配对：优于 random 族均值（random 同为 3 类、同样每月再平衡）
4. 滚动 504 日夏普胜率 ≥ 60%
5. 长历史（2019-01 → 2026-10）交叉验证不显著变差

任一条不过 → 证伪，维持类间等权。

## 方案

| label | 规则 |
|---|---|
| `equal` | 类间等权（= P3 定稿基线） |
| `invvol` | 逆波动率，权重 ∝ 1/σ，单类 cap 40% / floor 8% |
| `invvol_nocap` | 逆波动率，不设 cap |
| `minvol` | 100% 给最低波动的类（极端对照） |
| `riskparity` | 风险平价：各类对组合的边际风险贡献相等 |
| `lowcorr` | 权重 ∝ (1 − ρ(类, A股腿))，直接惩罚同向性 |
| `sharpew` | 权重 ∝ max(夏普, 0) |
| `rand_s{seed}` | Dirichlet(1) 随机**固定**权重（全窗口不变），6 个种子 |

## 防错要点

- **walk-forward**：权重只用 signal_date **之前** L=120 交易日的数据，无前视。
- **权重和 = 1** 必须断言。否则总仓位 < 1、余下现金 → 波动假降、夏普虚高。
- random 对照用 **全窗口固定权重**，换手与等权完全一致 → 隔离"多调权多换手"的混杂。
- 类内持仓代码不得重复（重复会被 dict 覆盖 → 有效持仓数减少、权重和 < 1）。
- 价格索引覆盖所有候选，**交易日历只按基线候选并集**（否则基线漂移）。
- 基线复现门：base 必须逐位复现 P3 定稿（main 1.4092 / oos 1.0685）。

## 用法

    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p4_weighting.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p4_weighting.py --window oos
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p4_weighting.py --window long
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p2 as P2  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    END,
    OOS_END,
    OOS_START,
    START,
    load_etf_universe,
    maxdd,
    merge_by_date,
    sharpe,
    total_ret,
    yearly,
)

CLASSES = ["gold", "bond", "commodity"]
REB_EVERY = 21
LOOKBACK = 120          # 权重估计回看交易日
VOL_CAP = 0.40          # 单类权重上限
VOL_FLOOR = 0.08        # 单类权重下限
ANN = 244               # 年化因子
SEEDS = (20261030, 7, 99, 1234, 555, 8888)
BOND_FINAL = ["511260", "511010", "511020"]
EXPECT_BASE = {"main": 1.4092, "oos": 1.0685}
LONG_START = datetime(2019, 1, 1)
LONG_END = datetime(2026, 10, 30)


# ───────────────────────── 收益序列 ─────────────────────────


def class_returns(calendar: List[str], prices, held: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """每个类 = 类内持仓等权的 close-to-close 日收益。"""
    import numpy as np

    out: Dict[str, Dict[str, float]] = {}
    for cls, codes in held.items():
        if not codes:
            continue
        series: Dict[str, float] = {}
        prev: Dict[str, Optional[float]] = {c: None for c in codes}
        for d in calendar:
            cur: Dict[str, Optional[float]] = {}
            for c in codes:
                p = prices.get(c, {}).get(d, {}).get("close")
                cur[c] = p
            rs = []
            for c in codes:
                p0 = prev[c]
                p1 = cur[c]
                if p0 is not None and p1 is not None and p0 > 0:
                    rs.append(p1 / p0 - 1.0)
                prev[c] = cur[c]
            if len(rs) == len(codes) and rs:
                series[d] = float(np.mean(rs))
        out[cls] = series
    return out


def nav_returns(nav: List[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i in range(1, len(nav)):
        d0, v0 = nav[i - 1]
        d1, v1 = nav[i]
        if v0:
            out[d1] = v1 / v0 - 1.0
    return out


# ───────────────────────── 权重引擎 ─────────────────────────


class WeightEngine:
    """walk-forward 类间权重：只用 signal_date 之前的数据。"""

    def __init__(self, calendar, prices, held, a_nav, classes, method,
                 lookback=LOOKBACK, cap=VOL_CAP, floor=VOL_FLOOR, seed=None):
        self.calendar = calendar
        self.classes = list(classes)
        self.method = method
        self.seed = seed
        self.lookback = lookback
        self.cap = cap
        self.floor = floor
        self.crets = class_returns(calendar, prices, held)
        self.arets = nav_returns(a_nav)
        self.idx = {d: i for i, d in enumerate(calendar)}
        self.weight_log: List[Dict[str, Any]] = []
        self._fixed: Optional[Dict[str, float]] = None
        if method == "rand":
            rng = random.Random(seed)
            w = [rng.gammavariate(1.0, 1.0) for _ in self.classes]   # Dirichlet(1)
            s = sum(w)
            self._fixed = {c: x / s for c, x in zip(self.classes, w)}

    # ── 统计量 ──
    def _stats(self, signal_date: str):
        import numpy as np

        i = self.idx.get(signal_date)
        if i is None or i < 30:
            return None
        win = self.calendar[max(0, i - self.lookback):i]
        dates = [d for d in win if d in self.arets]
        if len(dates) < 30:
            return None
        a = np.array([self.arets[d] for d in dates], dtype=float)
        # ⚠️ 所有类必须用**同一个共同日期掩码**：否则各类收益序列长度不一致，
        #    风险平价里 np.vstack 直接崩（"axis 1 size mismatch"）。
        raw: Dict[str, np.ndarray] = {}
        mask = np.ones(len(dates), dtype=bool)
        for c in self.classes:
            v = np.array([self.crets.get(c, {}).get(d, np.nan) for d in dates],
                         dtype=float)
            raw[c] = v
            mask &= ~np.isnan(v)
        if mask.sum() < 30:
            return None
        a2 = a[mask]
        sig: Dict[str, float] = {}
        mu: Dict[str, float] = {}
        rho: Dict[str, float] = {}
        rets: Dict[str, np.ndarray] = {}
        for c in self.classes:
            r = raw[c][mask]
            rets[c] = r
            sig[c] = float(r.std(ddof=1) * np.sqrt(ANN))
            mu[c] = float(r.mean() * ANN)
            if r.std(ddof=1) <= 0 or a2.std(ddof=1) <= 0:
                rho[c] = 0.0
            else:
                rho[c] = float(np.corrcoef(r, a2)[0, 1])
        return {"sig": sig, "mu": mu, "rho": rho, "rets": rets, "a": a2}

    def _cap_floor(self, w: Dict[str, float]) -> Dict[str, float]:
        """把单类权重压到 [floor, cap]。迭代处理（保持和为 1）。"""
        if self.cap >= 1.0 - 1e-9 and self.floor <= 0.0:
            return w
        out = dict(w)
        for _ in range(60):
            over = [c for c in out if out[c] > self.cap + 1e-12]
            under = [c for c in out if out[c] < self.floor - 1e-12]
            if not over and not under:
                break
            if over:
                excess = sum(out[c] - self.cap for c in over)
                for c in over:
                    out[c] = self.cap
                rest = [c for c in out if c not in over]
                if rest:
                    tot = sum(out[c] for c in rest)
                    if tot > 0:
                        for c in rest:
                            out[c] += excess * out[c] / tot
            if under:
                deficit = sum(self.floor - out[c] for c in under)
                for c in under:
                    out[c] = self.floor
                rest = [c for c in out if c not in under]
                if rest:
                    tot = sum(out[c] for c in rest)
                    if tot > 0:
                        for c in rest:
                            out[c] -= deficit * out[c] / tot
                        if min(out[c] for c in rest) < 0:
                            out = {c: max(out[c], 0.0) for c in out}
            s = sum(out.values())
            if s <= 0:
                return {c: 1.0 / len(self.classes) for c in self.classes}
            out = {c: v / s for c, v in out.items()}
        return out

    def _risk_parity(self, rets: Dict[str, Any]) -> Dict[str, float]:
        import numpy as np

        n = len(self.classes)
        mat = np.vstack([rets[c] for c in self.classes])
        cov = np.cov(mat) * ANN
        w = np.ones(n) / n
        for _ in range(500):
            mrc = cov @ w
            contrib = w * mrc
            if np.any(contrib <= 0) or contrib.sum() <= 0:
                break
            target = contrib.sum() / n
            w = w * (target / contrib)
            s = w.sum()
            if s <= 0:
                break
            w = w / s
        return {c: float(x) for c, x in zip(self.classes, w)}

    def __call__(self, signal_date: str) -> Dict[str, float]:
        if self.method == "rand":
            assert self._fixed is not None
            w = dict(self._fixed)
        elif self.method == "randdyn":
            # 动态随机：每次调仓重抽一次 Dirichlet(1) 权重。
            # 种子由 (seed, signal_date) 派生 → 可复现，且与"固定随机"区分开：
            # randdyn − rand = "每月调权"这个动作本身的价值；
            # 候选 − randdyn   = 调权**规则**的独特价值。
            rng = random.Random(f"{self.seed}:{signal_date}")
            raw = [rng.gammavariate(1.0, 1.0) for _ in self.classes]
            s0 = sum(raw)
            w = {c: x / s0 for c, x in zip(self.classes, raw)}
        else:
            st = self._stats(signal_date)
            if st is None:
                w = {c: 1.0 / len(self.classes) for c in self.classes}
            elif self.method == "equal":
                w = {c: 1.0 / len(self.classes) for c in self.classes}
            elif self.method in ("invvol", "invvol_nocap"):
                inv = {c: (1.0 / st["sig"][c] if st["sig"][c] > 0 else 0.0)
                       for c in self.classes}
                s = sum(inv.values())
                if s <= 0:
                    w = {c: 1.0 / len(self.classes) for c in self.classes}
                else:
                    w = {c: inv[c] / s for c in self.classes}
                    if self.method == "invvol":
                        w = self._cap_floor(w)
            elif self.method == "minvol":
                lo = min(self.classes, key=lambda c: st["sig"][c])
                w = {c: (1.0 if c == lo else 0.0) for c in self.classes}
            elif self.method == "riskparity":
                w = self._risk_parity(st["rets"])
            elif self.method == "lowcorr":
                raw = {c: max(0.0, 1.0 - st["rho"][c]) for c in self.classes}
                s = sum(raw.values())
                w = ({c: raw[c] / s for c in self.classes} if s > 0
                     else {c: 1.0 / len(self.classes) for c in self.classes})
            elif self.method == "sharpew":
                raw = {c: max(0.0, st["mu"][c] / st["sig"][c]) if st["sig"][c] > 0 else 0.0
                       for c in self.classes}
                s = sum(raw.values())
                w = ({c: raw[c] / s for c in self.classes} if s > 0
                     else {c: 1.0 / len(self.classes) for c in self.classes})
            else:
                raise SystemExit(f"[p4] ❌ 未知方法 {self.method}")
        # ── 权重和必须为 1：否则余下是现金 → 波动假降、夏普虚高 ──
        s = sum(w.values())
        if abs(s - 1.0) > 1e-9:
            raise SystemExit(f"[p4] ❌ 权重和 {s} != 1（{self.method} @ {signal_date}）")
        if min(w.values()) < -1e-12:
            raise SystemExit(f"[p4] ❌ 出现负权重（{self.method} @ {signal_date}）")
        self.weight_log.append({"date": signal_date,
                                "w": {c: round(v, 6) for c, v in w.items()}})
        return w


# ───────────────────────── 主流程 ─────────────────────────


def _load_a_leg(window: str, start, end, quick: bool):
    import multi_asset_leg_experiment as M

    M.START, M.END = start, end          # ⚠️ 模块全局窗口，必须显式同步
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    cache = os.path.join(ROOT, "data", "backtest", f"_a_share_nav_{window}.json")
    if quick and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            c = json.load(fh)
        print(f"[p4] 读缓存 A 股腿 {cache}", flush=True)
        return [(d, float(v)) for d, v in c["a_share_nav"]], c.get("a_share_leg", {})
    from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
    codes = universe_from_instruments(QLIB_OFFICIAL)
    print(f"[p4] 跑 A 股腿（{len(codes)} 只）...", flush=True)
    rec, nav = M.run_a_share_leg(codes)
    if window == "main":
        ver = M.verify_baseline(rec)
        print(f"[p4] 基线复现门: {json.dumps(ver, ensure_ascii=False)}", flush=True)
        if ver.get("checked") and not ver.get("ok"):
            raise SystemExit("[p4] ❌ A 股腿基线未复现，退出")
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"a_share_nav": nav, "a_share_leg": rec}, fh)
    return nav, rec


def run_window(args) -> None:
    import multi_asset_leg_experiment as M

    window = args.window
    if window == "main":
        start, end = START, END
    else:
        start, end = OOS_START, OOS_END
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    out_path = os.path.join(ROOT, "data", "backtest", f"multi_asset_p4_{window}.json")

    a_nav, a_rec = _load_a_leg(window, start, end, args.quick)
    if not a_nav:
        raise SystemExit("[p4] ❌ A 股腿净值为空")
    if not (s_str <= a_nav[0][0] <= e_str):
        raise SystemExit(f"[p4] ❌ A 股腿起点 {a_nav[0][0]} 不在窗口 {s_str}~{e_str} 内")
    base_sharpe = sharpe(a_nav)

    # ── 候选池 + 持仓集 ──
    pool = P3.build_pool(start)
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    held = {c: lst[:P3.K_IN_CLASS] for c, lst in by_class_raw.items()}
    # P3 定稿：bond 类固定长久期利率债
    held["bond"] = [c for c in BOND_FINAL if c in {r["code"] for r in pool}] or list(BOND_FINAL)
    print(f"[p4] 定稿持仓: {json.dumps(held, ensure_ascii=False)}", flush=True)

    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    all_codes = sorted(set(base_codes) | {r["code"] for r in pool} | set(BOND_FINAL))
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    print(f"[p4] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)
    if len(calendar) < 300:
        raise SystemExit("[p4] ❌ 交易日历过短，退出")

    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}

    def run_cfg(label: str, method: str, meta: Optional[Dict[str, Any]] = None,
                seed=None, cap=VOL_CAP, floor=VOL_FLOOR):
        for _c, _lst in held.items():
            if len(set(_lst)) != len(_lst):
                raise SystemExit(f"[p4] ❌ {label}: 类 {_c} 持仓重复 {_lst}，退出")
        eng = WeightEngine(calendar, prices, held, a_nav, CLASSES, method,
                           cap=cap, floor=floor, seed=seed)
        tg, tmeta = P3.build_targets(calendar, held, eng, reb_every=REB_EVERY)
        c_of = {c: cls for cls, lst in held.items() for c in lst}
        sleeve_nav, srec = L3.simulate(calendar, prices, tg, c_of,
                                       {c: 1.0 / len(CLASSES) for c in CLASSES})
        if not sleeve_nav:
            print(f"  {label}: ❌ 无净值", flush=True)
            return
        combined = merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])
        cs = sharpe(combined)
        navs[label] = combined

        # 各类与 A 股腿的 ρ（配权方案是否真的降低了同向性）
        cls_rho: Dict[str, Any] = {}
        for c in CLASSES:
            codes = held.get(c) or []
            if not codes:
                continue
            sub = {cc: cc for cc in codes}
            t2 = {d: {k: v for k, v in t.items() if k in codes} for d, t in tg.items()}
            t2 = {d: t for d, t in t2.items() if t}
            n2, _ = L3.simulate(calendar, prices, t2, sub, {c: 1.0})
            cls_rho[c] = {"rho_a": P2.nav_corr(n2, a_nav) if n2 else None,
                          "sleeve_sharpe": round(sharpe(n2), 4) if n2 else None}

        # 剔 gold 口径：gold 仓位重新分配给剩余两类并补满（scale = len/len(ng)）。
        # 与 P3 修正版一致；并断言剔 gold 后每个调仓日权重和 == 1。
        ng = [c for c in CLASSES if c != "gold"]
        sharpe_ex_gold: Optional[float] = None
        if ng and len(ng) < len(CLASSES):
            ng_t: Dict[str, Dict[str, float]] = {}
            for d, t in tg.items():
                # gold 的实际权重合计（**不是** 1/len(classes) —— 动态权重下会变）
                gold_w = sum(w for code, w in t.items() if c_of.get(code) == "gold")
                rest_w = sum(w for code, w in t.items() if c_of.get(code) != "gold")
                if rest_w <= 1e-12:
                    continue
                # 把 gold 的份额按原比例分给剩余类 → 总仓位补满 1（可比口径）
                scale = 1.0 / rest_w
                nt = {code: w * scale for code, w in t.items()
                      if c_of.get(code) != "gold"}
                if nt:
                    ng_t[d] = nt
            for _d, _t in ng_t.items():
                if abs(sum(_t.values()) - 1.0) > 1e-6:
                    raise SystemExit(
                        f"[p4] ❌ 剔 gold 后权重和 {sum(_t.values())} != 1（{_d}）")
            ng_sleeve, _ = L3.simulate(calendar, prices, ng_t, c_of,
                                       {c: 1.0 / len(ng) for c in ng})
            ng_comb = merge_by_date([a_nav, ng_sleeve], [0.5, 0.5]) if ng_sleeve else []
            sharpe_ex_gold = round(sharpe(ng_comb), 4) if ng_comb else None

        # 实际类权重时间序列的均值（机制归因）
        wsum: Dict[str, float] = {c: 0.0 for c in CLASSES}
        for row in eng.weight_log:
            for c, v in row["w"].items():
                wsum[c] += v
        n_log = max(1, len(eng.weight_log))
        avg_w = {c: round(wsum[c] / n_log, 4) for c in CLASSES}

        rec = {
            "method": method,
            "seed": seed,
            "held": {c: held.get(c) for c in CLASSES},
            "classes": CLASSES,
            "combined_sharpe": round(cs, 4),
            "vs_baseline": round(cs / base_sharpe - 1, 4),
            "combined_return": round(total_ret(combined), 4),
            "combined_maxdd": round(maxdd(combined), 4),
            "combined_yearly": yearly(combined),
            "sleeve_sharpe": round(sharpe(sleeve_nav), 4),
            "sleeve_vol": round(P2.nav_vol(sleeve_nav), 4),
            "sleeve_return": round(total_ret(sleeve_nav), 4),
            "corr_sleeve_ashare": P2.nav_corr(sleeve_nav, a_nav),
            "class_rho": cls_rho,
            "sharpe_ex_gold": sharpe_ex_gold,
            "avg_class_weight": avg_w,
            "weight_log": eng.weight_log,
            "cost": srec,
            **tmeta,
            **(meta or {}),
        }
        results[label] = rec
        print(f"  {label:22s} 组合夏普 {cs:.4f} ({rec['vs_baseline']:+.2%})  "
              f"收益 {rec['combined_return']:+.2%}  回撤 {rec['combined_maxdd']:+.2%}  | "
              f"袖套 {rec['sleeve_sharpe']:.4f} 波 {rec['sleeve_vol']:.2%}  "
              f"ρ(袖套,A) {rec['corr_sleeve_ashare']}  换手 {srec['turnover_ratio']:.2f}x  "
              f"均权 {avg_w}", flush=True)

    print(f"\n[p4] === 类间配权（{window}）===", flush=True)
    run_cfg("base_equal", "equal", {"note": "类间等权 = P3 定稿基线"})

    # ── 基线复现门 ──
    got = results["base_equal"]["combined_sharpe"]
    if abs(got - EXPECT_BASE[window]) > 5e-4:
        raise SystemExit(f"[p4] ❌ 基线未复现：{got} ≠ P3 定稿 {EXPECT_BASE[window]}，退出")
    print(f"[p4] ✅ 基线复现门通过：{got}", flush=True)

    for m, note in [("invvol", "逆波动率 cap40/floor8"),
                    ("invvol_nocap", "逆波动率 无 cap"),
                    ("minvol", "100% 给最低波动类"),
                    ("riskparity", "风险平价"),
                    ("lowcorr", "∝(1-ρ(类,A股腿))"),
                    ("sharpew", "∝max(夏普,0)")]:
        run_cfg(m, m, {"note": note})

    # random 对照 A：全窗口固定权重 → 换手与等权一致，隔离"多调权多换手"的混杂
    for s in SEEDS:
        run_cfg(f"rand_s{s}", "rand", {"note": f"Dirichlet(1) 随机固定权重 seed={s}"}, seed=s)
    # random 对照 B：每次调权重抽 → 与候选方案同"每月调权"参数
    #   randdyn − rand    = "每月调权"这个动作本身的价值
    #   候选   − randdyn  = 调权**规则**的独特价值
    for s in SEEDS:
        run_cfg(f"randdyn_s{s}", "randdyn",
                {"note": f"每月重抽 Dirichlet(1) 随机权重 seed={s}"}, seed=s)

    # ── 滚动夏普胜率 vs base ──
    for label in list(results):
        if label == "base_equal":
            continue
        w252 = P3.rolling_sharpe_win(navs["base_equal"], navs[label], 252)
        w504 = P3.rolling_sharpe_win(navs["base_equal"], navs[label], 504)
        results[label]["roll_win_252"] = w252
        results[label]["roll_win_504"] = w504

    rand_keys = [k for k in results if k.startswith("rand_s")]
    randdyn_keys = [k for k in results if k.startswith("randdyn_s")]
    cand_keys = [k for k in results if k not in rand_keys and k != "base_equal"]
    rand_main = [results[k]["combined_sharpe"] for k in rand_keys]
    randdyn_main = [results[k]["combined_sharpe"] for k in randdyn_keys]

    def _agg(v):
        if not v:
            return None
        return {"mean": round(sum(v) / len(v), 4), "max": round(max(v), 4),
                "min": round(min(v), 4), "n": len(v)}

    rm = _agg(rand_main)
    rd = _agg(randdyn_main)
    if rm is None or rd is None:
        raise SystemExit("[p4] ❌ random 对照组为空，退出")
    stats = {
        "random_static": rm,
        "random_dynamic": rd,
        "random_mean": rm["mean"] if rm else None,
        "n_random": len(rand_keys),
        "candidates_beating_random_static_mean": sorted(
            k for k in cand_keys
            if rm and results[k]["combined_sharpe"] > rm["mean"]),
        "candidates_beating_random_dynamic_mean": sorted(
            k for k in cand_keys
            if rd and results[k]["combined_sharpe"] > rd["mean"]),
        "candidates_beating_random_max": sorted(
            k for k in cand_keys
            if rm and results[k]["combined_sharpe"] > rm["max"]),
    }
    print(f"\n[p4] random 固定权重族: 均值 {rm['mean']} 范围 [{rm['min']}, {rm['max']}] n={rm['n']}", flush=True)
    print(f"[p4] random 动态权重族: 均值 {rd['mean']} 范围 [{rd['min']}, {rd['max']}] n={rd['n']}", flush=True)
    print(f"[p4] 优于 random 固定均值的候选: {stats['candidates_beating_random_static_mean']}", flush=True)
    print(f"[p4] 优于 random 动态均值的候选: {stats['candidates_beating_random_dynamic_mean']}", flush=True)
    print(f"[p4] 超过 random 固定最优({rm['max']})的候选: {stats['candidates_beating_random_max']}", flush=True)

    out = {
        "window": window,
        "window_range": [s_str, e_str],
        "baseline_held": held,
        "a_share_leg_sharpe": round(base_sharpe, 6),
        "calendar": {"n": len(calendar), "start": calendar[0], "end": calendar[-1]},
        "lookback": LOOKBACK,
        "vol_cap": VOL_CAP,
        "vol_floor": VOL_FLOOR,
        "results": results,
        "stats": stats,
        "navs": {k: v for k, v in navs.items()},
    }
    if a_rec:
        out["a_share_leg"] = a_rec
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n[p4] ✓ 已写入 {out_path}", flush=True)


def run_long(args) -> None:
    """长历史（2019-01 → 2026-10）：袖套口径交叉验证。

    **定位必须写清楚**：长历史**没有 A 股腿**（面板从 2021-05-11 起），
    测不到 ρ(袖套, A股腿) 这个决定组合夏普的变量。
    P2 已证：袖套夏普升 ≠ 组合夏普升（长历史袖套 Δ+0.99 但组合 Δ−0.12）。
    → 长历史只能作为**辅助证据**，不构成组合层判决。
    """
    import multi_asset_long_history as LH

    s_str, e_str = LONG_START.strftime("%Y-%m-%d"), LONG_END.strftime("%Y-%m-%d")
    out_path = os.path.join(ROOT, "data", "backtest", "multi_asset_p4_long.json")
    universe = load_etf_universe()
    by_class_raw = LH.pick_point_in_time(universe, CLASSES, s_str, e_str)
    held = {c: lst[:P3.K_IN_CLASS] for c, lst in by_class_raw.items()}
    held["bond"] = list(BOND_FINAL)
    print(f"[p4] 长历史持有: {json.dumps(held, ensure_ascii=False)}", flush=True)

    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    prices = L3.load_prices(sorted(set(base_codes) | set(BOND_FINAL) | {"511380"}))
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    print(f"[p4] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)
    if len(calendar) < 500:
        raise SystemExit("[p4] ❌ 长历史日历过短，退出")

    # 长历史没有 A 股腿 → 用 510300 代理做权重估计（明确标注为代理）
    proxy = P3._load_code(P3.PROXY)
    if proxy is None:
        raise SystemExit("[p4] ❌ 代理 510300 无数据")
    pnav = [(str(d.date()), float(v))
            for d, v in zip(proxy["date"], proxy["close"])]
    pnav = [(d, v) for d, v in pnav if s_str <= d <= e_str]
    print(f"[p4] 代理 A 股腿 {len(pnav)} 天（长历史无 honest_composite，仅代理）", flush=True)

    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}

    def run_cfg(label: str, method: str, meta=None, seed=None):
        eng = WeightEngine(calendar, prices, held, pnav, CLASSES, method, seed=seed)
        tg, tmeta = P3.build_targets(calendar, held, eng, reb_every=REB_EVERY)
        c_of = {c: cls for cls, lst in held.items() for c in lst}
        sleeve_nav, srec = L3.simulate(calendar, prices, tg, c_of,
                                       {c: 1.0 / len(CLASSES) for c in CLASSES})
        if not sleeve_nav:
            print(f"  {label}: ❌ 无净值", flush=True)
            return
        cs = sharpe(sleeve_nav)
        navs[label] = sleeve_nav
        wsum = {c: 0.0 for c in CLASSES}
        for row in eng.weight_log:
            for c, v in row["w"].items():
                wsum[c] += v
        n_log = max(1, len(eng.weight_log))
        rec = {
            "method": method, "seed": seed,
            "held": {c: held.get(c) for c in CLASSES},
            "sleeve_sharpe": round(cs, 4),
            "sleeve_return": round(total_ret(sleeve_nav), 4),
            "sleeve_maxdd": round(maxdd(sleeve_nav), 4),
            "sleeve_vol": round(P2.nav_vol(sleeve_nav), 4),
            "corr_sleeve_proxy": P2.nav_corr(sleeve_nav, pnav),
            "avg_class_weight": {c: round(wsum[c] / n_log, 4) for c in CLASSES},
            "weight_log": eng.weight_log,
            "cost": srec, **tmeta, **(meta or {}),
        }
        results[label] = rec
        print(f"  {label:22s} 袖套夏普 {cs:.4f} 收益 {rec['sleeve_return']:+.2%} "
              f"回撤 {rec['sleeve_maxdd']:+.2%} 波 {rec['sleeve_vol']:.2%} "
              f"ρ(袖套,510300) {rec['corr_sleeve_proxy']} 均权 {rec['avg_class_weight']}",
              flush=True)

    print(f"\n[p4] === 长历史类间配权（袖套口径）===", flush=True)
    run_cfg("base_equal", "equal", {"note": "类间等权"})
    for m in ("invvol", "invvol_nocap", "minvol", "riskparity", "lowcorr", "sharpew"):
        run_cfg(m, m)
    for s in SEEDS:
        run_cfg(f"rand_s{s}", "rand", {"note": f"随机固定权重 seed={s}"}, seed=s)
    for s in SEEDS:
        run_cfg(f"randdyn_s{s}", "randdyn",
                {"note": f"每月重抽随机权重 seed={s}"}, seed=s)

    for label in list(results):
        if label == "base_equal":
            continue
        results[label]["roll_win_252"] = P3.rolling_sharpe_win(
            navs["base_equal"], navs[label], 252)
        results[label]["roll_win_504"] = P3.rolling_sharpe_win(
            navs["base_equal"], navs[label], 504)

    out = {
        "window": "long",
        "window_range": [s_str, e_str],
        "caveat": ("长历史无 honest_composite A 股腿，仅袖套口径 + 510300 代理；"
                   "P2 已证袖套夏普升≠组合夏普升 → 辅助证据，不构成组合层判决"),
        "baseline_held": held,
        "calendar": {"n": len(calendar), "start": calendar[0], "end": calendar[-1]},
        "results": results,
        "navs": {k: v for k, v in navs.items()},
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n[p4] ✓ 已写入 {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos", "long"])
    ap.add_argument("--quick", action="store_true", help="读缓存的 A 股腿")
    args = ap.parse_args()
    if args.window == "long":
        run_long(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
