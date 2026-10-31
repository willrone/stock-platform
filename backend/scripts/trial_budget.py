#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""试验预算账本 + 统计闸门（DSR / PBO）。

**为什么需要这个脚本**
项目累计试验数已到 1937 次（1070 历史 + P17 的 867）→ 95% 显著性所需年化夏普阈值
≈ 2.35。而定稿 A 股腿 honest_composite 只有 0.8796 → DSR ≈ 0.0113，**不过**。

关键性质：**n_trials 上升 ⇒ DSR 阈值上升 ⇒ 结论只会更差**。
也就是说，「再试一个源」的期望收益为负 —— 每做一次试验都在抬高自己需要翻越的门槛。
把它做成**机制**（账本只增不减 + 闸门可 exit 1），而不是口头纪律。

**用法**
    # 全局状态：当前试验数 / 阈值 / 各候选 DSR / 还能不能做实验
    python3 scripts/trial_budget.py --report

    # 登记一次新试验（累加试验数）
    python3 scripts/trial_budget.py --register "P18 42f+gbm" --trials 8 --note "先验限定"

    # 对某个 NAV 做 DSR 判定；--gate 时不通过则 exit 1（可挂 CI / 审计器）
    python3 scripts/trial_budget.py --check data/backtest/xxx.json --nav-key navs.honest
    python3 scripts/trial_budget.py --check ... --gate

    # 多候选 PBO（前半选优 → 后半排名）
    python3 scripts/trial_budget.py --pbo data/backtest/_blend_nav_25f.json

    # 降低阈值（仅用于「本实验的搜索空间被先验压缩到 N 次」这种正当理由）
    python3 scripts/trial_budget.py --register "..." --trials 8 --override --note "先验压缩到 8 次"
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from verify_deflated_sharpe import (  # noqa: E402
    deflated_sharpe,
    norm_cdf,
    pbo_cscv,
)

LEDGER_PATH = os.path.join(ROOT, "data", "backtest", "trial_budget.json")
BLEND_NAV = os.path.join(ROOT, "data", "backtest", "_blend_nav_25f.json")

#: 账本初始状态。1070 = 截至 2026-10-31 项目累计试验数（来源：data/backtest/
#: deflated_sharpe.json 的 methodology.n_trials）；867 = P17 新增（类内选标的候选
#: 452 main + 403 oos + 袖套频率 6×2）。
LEDGER_SEED = {
    "schema": "trial-budget-v1",
    "created": "2026-11-01",
    "n_trials_total": 1937,
    "entries": [
        {
            "date": "2026-10-31",
            "label": "历史累计（截至 P16）",
            "trials": 1070,
            "note": "来源 deflated_sharpe.json methodology.n_trials；未逐条登记",
            "source": "data/backtest/deflated_sharpe.json",
        },
        {
            "date": "2026-11-01",
            "label": "P17-A 类内选标的穷举",
            "trials": 855,
            "note": "gold 364 + commodity 4 + bond 84(main) / 35(oos) = 452 + 403 次精确撮合",
            "source": "data/backtest/multi_asset_p17_single_{main,oos}.json",
        },
        {
            "date": "2026-11-01",
            "label": "P17-B 袖套调仓频率",
            "trials": 12,
            "note": "6 个频率 × 2 窗口",
            "source": "data/backtest/multi_asset_p17_freq_{main,oos}.json",
        },
    ],
    "best_known": {
        "label": "honest_composite 42f（A 腿单策略，main 真实盘 lag1）",
        "sharpe": 0.8796,
        "source": "data/backtest/ALL_real_lag1_v2.json",
    },
    "note": (
        "n_trials 只能增不能减。降低阈值等于放宽统计标准，必须显式 --override 并留痕，"
        "且仅在「本实验的搜索空间被先验压缩到 N 次」这种正当理由下使用。"
    ),
}


# ----------------------------------------------------------------------------- 账本


def load_ledger(path: str = LEDGER_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return json.loads(json.dumps(LEDGER_SEED))
    with open(path) as fh:
        return json.load(fh)


def save_ledger(ledger: Dict[str, Any], path: str = LEDGER_PATH) -> None:
    ledger["updated"] = datetime.date.today().isoformat()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


# ----------------------------------------------------------------- NAV 读取 / 收益率


def _nav_series(payload: Any) -> Dict[str, Dict[str, float]]:
    """从多种产物形态里抽出 {name: {date: nav}}。"""
    if isinstance(payload, dict) and "navs" in payload:
        return {k: {str(d): float(v) for d, v in v0.items()} for k, v0 in payload["navs"].items()}
    if isinstance(payload, dict) and payload and all(
        isinstance(v, (int, float)) for v in payload.values()
    ):
        return {"_single": {str(d): float(v) for d, v in payload.items()}}
    raise SystemExit("[trial_budget] 无法识别的 NAV 格式（需要 {'navs': {...}} 或 {date: value}）")


def nav_to_returns(nav: Dict[str, float]) -> List[float]:
    """{date: nav} → 日收益率序列（首元素 0 对齐起点，与 deflated_sharpe 约定一致）。"""
    items = sorted(nav.items(), key=lambda kv: kv[0])
    vals = [v for _, v in items]
    out = [0.0]
    for prev, cur in zip(vals, vals[1:]):
        out.append((cur / prev - 1.0) if prev else 0.0)
    return out


def load_navs(path: str) -> Dict[str, Dict[str, float]]:
    with open(path) as fh:
        return _nav_series(json.load(fh))


# ------------------------------------------------------------------------ 阈值/报告


def sharpe_needed_for_dsr95(n_trials: int, skew: float = -0.6956,
                            kurt: float = 17.4961, n_obs: int = 726) -> float:
    """给定试验数与分布形状，求 **满足 DSR ≥ 0.95 所需的年化夏普**。

    ⚠️ 口径必须分清（极易混淆）：
      - `sr0`（`deflated_sharpe` 返回的 `sr0_threshold_annualized`）= 零假设下的期望最大夏普，
        即「n_trials 次试验里纯靠运气能拿到的最好夏普」→ 它是 **SR0，不是显著性门槛**；
      - DSR = Φ((sr_d − sr0)/sd_sr) ≥ 0.95 需要 sr_d ≥ sr0 + 1.6449×sd_sr
        → 本函数返回的就是这个**实际门槛**（比 SR0 高一截）。

    复刻 deflated_sharpe 的正态极值近似：
        sr0 = sd_sr·√(2 ln N),  sd_sr = √((1 − skew·sr + (kurt−1)/4·sr²)/T)
    其中 sr 依赖门槛本身 → 不动点迭代。

    ⚠️ **不要**用「迭代后的 sr」去反推 SR0（那会循环依赖、把 SR0 高估）。
    SR0 必须用**策略实际的日夏普**来算 —— 见 `deflated_sharpe` 的实现。
    """
    import math

    n_obs = max(60, int(n_obs))
    sr = 0.05
    for _ in range(200):
        var = max(1e-12, (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / n_obs)
        sd_sr = math.sqrt(var)
        sr0 = sd_sr * math.sqrt(2.0 * math.log(max(2, n_trials)))
        new = sr0 + norm_cdf_inverse(0.95) * sd_sr
        if abs(new - sr) < 1e-13:
            sr = new
            break
        sr = new
    return sr * math.sqrt(252.0)


def norm_cdf_inverse(p: float) -> float:
    """标准正态分位数（Acklam 近似，足够工程精度）。"""
    import math

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def report(ledger: Dict[str, Any]) -> int:
    n = int(ledger["n_trials_total"])
    thr = sharpe_needed_for_dsr95(n)
    best = ledger.get("best_known", {})
    best_sh = float(best.get("sharpe", 0.0))
    print(f"[trial_budget] 账本: {LEDGER_PATH}")
    print(f"[trial_budget] 累计试验数 n_trials = {n}")
    print(f"[trial_budget] 要让 DSR ≥ 0.95，实际夏普需 ≥            = {thr:.4f}")
    print(f"[trial_budget] 已知最优: {best.get('label')} = {best_sh}")
    gap = best_sh - thr
    print(f"[trial_budget] 差距 = {gap:+.4f}  →  "
          f"{'✅ 当前最优已能过 DSR 95%' if gap > 0 else '❌ 当前最优仍不过 DSR 95%'}")
    print()
    print("[trial_budget] 各登记条目：")
    for e in ledger.get("entries", []):
        ov = " [override]" if e.get("override") else ""
        print(f"  {e['date']}  {e.get('label','')}: +{e.get('trials',0)}{ov}")
    print()

    if os.path.exists(BLEND_NAV):
        navs = load_navs(BLEND_NAV)
        rets = {k: nav_to_returns(v) for k, v in navs.items()}
        print(f"{'候选':<24}{'夏普':>9}{'SR0':>9}{'DSR':>10}{'过95%':>8}")
        print("-" * 62)
        for k, r in rets.items():
            ds = deflated_sharpe(r, n)
            if "error" in ds:
                continue
            print(f"{k:<24}{ds['sharpe_annualized']:>9.4f}"
                  f"{ds['sr0_threshold_annualized']:>9.4f}{ds['deflated_sharpe']:>10.5f}"
                  f"{'  PASS' if ds['pass_95'] else '  FAIL':>8}")
        pbo = pbo_cscv(rets)
        print(f"\n[trial_budget] PBO = {pbo.get('pbo')}（{pbo.get('n_variants')} 候选 × "
              f"{pbo.get('n_splits')} 切分；>0.5 = 前半选出的最优在后半系统性跑输）")
    return 0


# --------------------------------------------------------------------------- 子命令


def cmd_init(_args: argparse.Namespace) -> int:
    """把种子账本落盘（首次建立基线）。已存在则拒绝覆盖。"""
    if os.path.exists(LEDGER_PATH):
        raise SystemExit(f"[trial_budget] 账本已存在，拒绝覆盖：{LEDGER_PATH}")
    save_ledger(load_ledger())
    print(f"[trial_budget] 已初始化账本 {LEDGER_PATH}"
          f"（n_trials = {load_ledger()['n_trials_total']}）")
    return 0


def _rebuild_manifest() -> None:
    """账本变更后重建指纹清单 —— 否则 verify 会把「正常登记」误报成「产物被覆盖」。

    账本历史由 git 记录（每次 register 伴随一次 commit）→ 事后篡改可追；
    指纹清单保证「当前状态」与清单一致。
    """
    try:
        import artifact_manifest
        artifact_manifest.build()
    except Exception as exc:  # noqa: BLE001
        print(f"[trial_budget] ⚠️ 指纹清单重建失败：{exc}")
        print("[trial_budget]    → 需手动执行：python3 scripts/artifact_manifest.py build")


def cmd_register(args: argparse.Namespace) -> int:
    ledger = load_ledger()
    before = int(ledger["n_trials_total"])
    trials = int(args.trials)
    if trials < 0:
        raise SystemExit("[trial_budget] --trials 不能为负")
    after = before + trials
    if after < before:
        raise SystemExit("[trial_budget] n_trials 不可减少")
    if args.override and not args.note:
        raise SystemExit("[trial_budget] --override 必须给 --note 说明理由")
    # ⚠️ 全局 n_trials_total **一律累加**（门槛只升不降）。
    #    --override 只额外记录「本实验自身的 DSR 口径」——
    #    绝不能改全局，否则一次 override 就能把整个项目的显著性门槛永久拉低。
    ledger["n_trials_total"] = before + trials
    entry = {
        "date": datetime.date.today().isoformat(),
        "label": args.register,
        "trials": trials,
        "note": args.note or "",
        "override": bool(args.override),
        "n_trials_total_after": after,
    }
    if args.override:
        # 本实验自己的口径（用于 DSR 判定）；全局口径仍然累加
        entry["experiment_trials"] = trials
        entry["experiment_dsr95_threshold"] = round(
            sharpe_needed_for_dsr95(trials), 4)
    ledger.setdefault("entries", []).append(entry)
    save_ledger(ledger)
    _rebuild_manifest()
    print(f"[trial_budget] 已登记「{args.register}」+{trials} 次试验"
          f"{'（override）' if args.override else ''}；"
          f"全局 n_trials {before} → {after}")
    print(f"[trial_budget] 全局口径：要 DSR ≥ 0.95，实际夏普需 ≥ "
          f"{sharpe_needed_for_dsr95(after):.4f}")
    if args.override:
        print(f"[trial_budget] 本实验口径（n_trials={trials}）：要 DSR ≥ 0.95，"
              f"实际夏普需 ≥ {sharpe_needed_for_dsr95(trials):.4f}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    ledger = load_ledger()
    n = int(ledger["n_trials_total"])
    navs = load_navs(args.check)
    if args.nav_key:
        if args.nav_key not in navs:
            raise SystemExit(f"[trial_budget] NAV key '{args.nav_key}' 不存在，可选：{list(navs)}")
        navs = {args.nav_key: navs[args.nav_key]}
    print(f"[trial_budget] 口径 n_trials = {n}（账本 {LEDGER_PATH}）")
    print(f"[trial_budget] 要让 DSR ≥ 0.95，实际夏普需 ≥ {sharpe_needed_for_dsr95(n):.4f}")
    print(f"{'候选':<24}{'夏普':>9}{'SR0':>9}{'DSR':>10}{'过95%':>8}")
    print("-" * 62)
    all_pass = True
    for k, v in navs.items():
        ds = deflated_sharpe(nav_to_returns(v), n)
        if "error" in ds:
            print(f"{k:<24} 样本过短，跳过")
            continue
        all_pass = all_pass and bool(ds["pass_95"])
        print(f"{k:<24}{ds['sharpe_annualized']:>9.4f}{ds['sr0_threshold_annualized']:>9.4f}"
              f"{ds['deflated_sharpe']:>10.5f}{'  PASS' if ds['pass_95'] else '  FAIL':>8}")
    if args.gate and not all_pass:
        print("\n[trial_budget] ❌ 闸门未通过（DSR < 0.95）→ exit 1")
        print("[trial_budget]    含义：即便按当前口径，这个结果的夏普仍与 0 不可区分；"
              "要么增加样本长度，要么承认它不是统计显著的改进。")
        return 1
    return 0


def cmd_pbo(args: argparse.Namespace) -> int:
    rets: Dict[str, List[float]] = {}
    for path in args.pbo:
        for k, v in load_navs(path).items():
            rets[k] = nav_to_returns(v)
    out = pbo_cscv(rets, n_splits=args.splits)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    p = out.get("pbo")
    if p is not None:
        print(f"\n[trial_budget] PBO = {p} → "
              f"{'❌ 候选间排序不可信（≈抛硬币或更差）' if p > 0.5 else '✅ 候选间排序有一定稳定性'}")
    return 0


def selfcheck() -> int:
    """自检：门槛函数与 `deflated_sharpe` 的 DSR 公式自洽。

    做法：取门槛对应的日夏普，代回 DSR 公式 Φ((sr_d − sr0)/sd_sr)，应恰好 = 0.95。
    这验证不动点迭代确实收敛到 DSR = 0.95 的解（而不是别处）。
    """
    import math

    ok = True
    print("[selfcheck] 门槛反算 DSR（应 = 0.95000000）")
    for n_trials in (82, 1070, 1937, 5000):
        for skew, kurt, n_obs in ((-0.6956, 17.4961, 726), (0.0, 3.0, 726),
                                  (0.2, 6.0, 1000), (-1.2, 25.0, 500)):
            thr_daily = sharpe_needed_for_dsr95(n_trials, skew, kurt, n_obs) / math.sqrt(252.0)
            var = max(1e-12, (1 - skew * thr_daily + (kurt - 1) / 4 * thr_daily ** 2) / n_obs)
            sd_sr = math.sqrt(var)
            sr0 = sd_sr * math.sqrt(2.0 * math.log(max(2, n_trials)))
            dsr = norm_cdf((thr_daily - sr0) / sd_sr)
            good = abs(dsr - 0.95) < 1e-6
            ok = ok and good
            print(f"  N={n_trials:>5} skew={skew:>7} kurt={kurt:>6} T={n_obs:>5} → "
                  f"门槛年化夏普={thr_daily * math.sqrt(252.0):>7.4f}  "
                  f"反算DSR={dsr:.8f} {'✅' if good else '❌'}")
    print()
    print(f"[selfcheck] norm_cdf_inverse(0.95) = {norm_cdf_inverse(0.95):.6f}（应 ≈ 1.644854）")
    print(f"[selfcheck] 结果：{'✅ 全部通过' if ok else '❌ 有失败'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="试验预算账本 + DSR/PBO 闸门")
    ap.add_argument("--init", action="store_true", help="初始化账本（种子状态）")
    ap.add_argument("--report", action="store_true", help="打印全局状态")
    ap.add_argument("--register", metavar="LABEL", help="登记一次试验")
    ap.add_argument("--trials", type=int, default=0, help="本次试验的变体尝试次数")
    ap.add_argument("--note", help="备注 / override 理由")
    ap.add_argument("--override", action="store_true",
                    help="用本实验自身试验数覆盖全局口径（必须给 --note）")
    ap.add_argument("--check", metavar="JSON", help="对某个 NAV 产物做 DSR 判定")
    ap.add_argument("--nav-key", help="NAV 文件中的具体 key")
    ap.add_argument("--gate", action="store_true", help="DSR 不过则 exit 1")
    ap.add_argument("--pbo", nargs="+", help="多候选 NAV 文件，算 PBO")
    ap.add_argument("--splits", type=int, default=200, help="PBO 切分次数")
    ap.add_argument("--selfcheck", action="store_true", help="自检门槛函数与 DSR 公式自洽")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.init:
        return cmd_init(args)
    if args.register:
        return cmd_register(args)
    if args.check:
        return cmd_check(args)
    if args.pbo:
        return cmd_pbo(args)
    return report(load_ledger())


if __name__ == "__main__":
    raise SystemExit(main())
