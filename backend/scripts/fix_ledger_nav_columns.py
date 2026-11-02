#!/usr/bin/env python3
"""一次性修复：ledger.csv 的 `portfolio_value` / `cash_after` 两列 + holdings 的 `nav_history`。

## 触发（2026-11-03，§12.43）

11-02 17:35 那次周频跑**发生在进程互斥锁落地之前**（`00ada47d` 18:51 才提交，
台账 §12.34 记的就是"三档排期并发写账本"）→ `data/paper/<策略>_ledger.csv`
里 **2026-11-02 的 6 行字段语义坏掉**：

```
SELL 600690.SH  cash_after=30495.0   portfolio_value=-54.89   ← 实际是【单笔盈亏】
SELL 601021.SH  cash_after=31256.0   portfolio_value=-56.26
SELL 000958.SZ  cash_after=31360.0   portfolio_value=-56.45
BUY  603357.SH  cash_after=(空)      portfolio_value=(空)
...
```

后果（用户可见）：
- 后端 `signal_service` 的净值曲线直接读 ledger 的 `portfolio_value`
  → 22f 的末日点变成 **−56.45**（真值 978,982.93）
  → 图表上从 990,864 直接掉到 −56（像清零）
- 现在 `nav_points` 只有 4 个 < 20 → 夏普/回撤返回 None → **尚未爆**；
  但 5 个月后攒够 20 个调仓点，这条脏点就会把回撤变成 −100% 的垃圾值。

同时：`holdings.nav_history` 也只剩 **1 个点** —— 某次跑时 holdings 文件缺失，
`load_holdings()` 返回空壳（`nav_history=[]`）→ 只追加当前点 → 历史被抹掉
（§12.36 的缺失守卫 `d5a37982` 就是为这个加的，**已经发生过了**）。

## 修法：以 `nav_daily.json` 为权威源

`data/paper/<策略>_nav_daily.json` 是**每个交易日一个点**的盯市值（§12.37 守卫生效），
比 ledger 的冗余列可靠 —— ledger 的两列是"**每批写盘当时**"的状态，
同一交易日可能有多批（10-23 就有 4 个不同值：980115 / 991150 / 991689 / 991972）。

- `portfolio_value`：该成交日的所有行 → 统一改成 nav_daily 该日的盯市值
- `cash_after`：该成交日**最后一批**的现金（11-02 用 holdings.cash）
- `nav_history`：按**成交日**重建（保持"调仓日净值点"语义，不混日频）
  → 末日值必须与 portfolio.json 的 value 一致（残差 ≤0.05）

## 用法

```bash
python3 scripts/fix_ledger_nav_columns.py --dry-run   # 只报告，不写
python3 scripts/fix_ledger_nav_columns.py --apply     # 写盘（先自动备份）
```

**验收**（脚本自己会查，任一条不过就 exit 1，不写盘）：
1. 每个成交日的 portfolio_value 全部非空且 > 0
2. holdings.nav_history 点数 == ledger 成交日数
3. nav_history 末日值 vs portfolio.json value 残差 ≤ 0.05
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(BACKEND_ROOT)
PAPER_DIR = os.path.join(ROOT, "data", "paper")

# 这 6 条是单条腿的线；multi_asset_final 的合并账本口径不同（两腿净值相加），
# 且它的 portfolio_value 已经是合并口径 → 不动。
TARGET_LINES = ["22f", "honest_42f", "rsi_71421", "factor_ranking",
                "gbm_voltgt", "honest_42f_neu"]

VALUE_TOL = 0.05  # 与 portfolio.json 的残差阈值（元）


def _f(v: Any) -> float | None:
    try:
        if v is None or str(v).strip() == "":
            return None
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def load_nav_daily(strategy: str) -> Dict[str, float]:
    """日频盯市值 {日期: 组合市值}。"""
    p = os.path.join(PAPER_DIR, f"{strategy}_nav_daily.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, float] = {}
    for row in d.get("points") or []:
        ds = str(row.get("date") or "")[:10]
        v = _f(row.get("value"))
        if ds and v is not None:
            out[ds] = v
    return out


def load_holdings(strategy: str) -> Dict[str, Any]:
    p = os.path.join(PAPER_DIR, f"{strategy}_holdings.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def read_ledger(strategy: str) -> Tuple[List[str], List[Dict[str, str]]]:
    p = os.path.join(PAPER_DIR, f"{strategy}_ledger.csv")
    with open(p, newline="") as fh:
        r = csv.DictReader(fh)
        return list(r.fieldnames or []), list(r)


def write_ledger(strategy: str, fieldnames: List[str],
                 rows: List[Dict[str, str]]) -> None:
    p = os.path.join(PAPER_DIR, f"{strategy}_ledger.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_holdings(strategy: str, h: Dict[str, Any]) -> None:
    p = os.path.join(PAPER_DIR, f"{strategy}_holdings.json")
    with open(p, "w") as fh:
        json.dump(h, fh, ensure_ascii=False, indent=1)


def backup(strategy: str, tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = []
    for name in (f"{strategy}_ledger.csv", f"{strategy}_holdings.json"):
        src = os.path.join(PAPER_DIR, name)
        if os.path.exists(src):
            dst = f"{src}.bak-{tag}-{ts}"
            shutil.copy2(src, dst)
            out.append(dst)
    return ", ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写盘")
    ap.add_argument("--apply", action="store_true", help="写盘（先自动备份）")
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.print_help()
        return 2

    pf_path = os.path.join(PAPER_DIR, "portfolio.json")
    with open(pf_path) as fh:
        pf = json.load(fh)
    pf_lines = pf.get("lines") or {}

    plan: List[Dict[str, Any]] = []
    problems: List[str] = []

    for strat in TARGET_LINES:
        lp = os.path.join(PAPER_DIR, f"{strat}_ledger.csv")
        if not os.path.exists(lp):
            problems.append(f"{strat}: 没有 ledger 文件")
            continue
        fieldnames, rows = read_ledger(strat)
        daily = load_nav_daily(strat)
        holdings = load_holdings(strat)
        h_cash = _f(holdings.get("cash"))
        cap0 = _f(holdings.get("initial_capital")) or 1_000_000.0

        # 成交日 → 该日的行
        by_day: "OrderedDict[str, List[int]]" = OrderedDict()
        for i, r in enumerate(rows):
            d = str(r.get("date") or "").strip()
            if d:
                by_day.setdefault(d, []).append(i)

        # ── 1. 该日权威组合值：优先 nav_daily，其次该日最后一批的 portfolio_value
        day_value: Dict[str, float] = {}
        for d, idxs in by_day.items():
            v = daily.get(d)
            if v is None:
                for i in reversed(idxs):          # 同日最后一批
                    v = _f(rows[i].get("portfolio_value"))
                    if v is not None and v > 0:
                        break
            if v is not None and v > 0:
                day_value[d] = v

        # ── 2. 该日权威现金：同日最后一批的 cash_after（坏值回退到 holdings.cash）
        day_cash: Dict[str, float] = {}
        for d, idxs in by_day.items():
            v = None
            for i in reversed(idxs):
                v = _f(rows[i].get("cash_after"))
                if v is not None:
                    break
            if v is None or v < 0:
                v = h_cash
            if v is not None:
                day_cash[d] = v

        # ── 3. 改写：portfolio_value 统一为该日权威值；cash_after 统一为该日权威现金
        changed = 0
        for d, idxs in by_day.items():
            pv, ca = day_value.get(d), day_cash.get(d)
            for i in idxs:
                old_pv = (rows[i].get("portfolio_value") or "").strip()
                old_ca = (rows[i].get("cash_after") or "").strip()
                new_pv = f"{pv:.2f}" if pv is not None else old_pv
                new_ca = f"{ca:.2f}" if ca is not None else old_ca
                if new_pv != old_pv or new_ca != old_ca:
                    rows[i]["portfolio_value"] = new_pv
                    rows[i]["cash_after"] = new_ca
                    changed += 1

        # ── 4. 重建 nav_history（按成交日，保持"调仓日净值点"语义）
        nav_hist: List[Dict[str, Any]] = []
        for d in by_day:
            v = day_value.get(d)
            if v is None:
                continue
            nav_hist.append({
                "date": d,
                "value": round(v, 2),
                "return_pct": round((v / cap0 - 1.0) * 100, 4),
            })
        old_nav = holdings.get("nav_history") or []

        # ── 5. 验收
        pf_val = _f((pf_lines.get(strat) or {}).get("value"))
        last_nav = nav_hist[-1]["value"] if nav_hist else None
        resid = None
        if pf_val is not None and last_nav is not None:
            resid = last_nav - pf_val
        bad_pv = [d for d in day_value if day_value[d] <= 0]
        if bad_pv:
            problems.append(f"{strat}: 这些成交日仍无有效组合值 {bad_pv}")
        if resid is None or abs(resid) > VALUE_TOL:
            problems.append(
                f"{strat}: nav_history 末日 {last_nav} vs portfolio.json value "
                f"{pf_val} 残差 {resid}（阈值 {VALUE_TOL}）→ 不写盘")

        plan.append({
            "strategy": strat,
            "rows_total": len(rows),
            "rows_changed": changed,
            "trade_days": len(by_day),
            "nav_daily_points": len(daily),
            "nav_hist_old": len(old_nav),
            "nav_hist_new": len(nav_hist),
            "last_nav": last_nav,
            "pf_value": pf_val,
            "resid": resid,
            "day_value": day_value,
            "rows": rows,
            "fieldnames": fieldnames,
            "nav_hist": nav_hist,
            "holdings": holdings,
        })

    # ── 报告
    print("=" * 78)
    print("ledger nav 列修复 —— DRY RUN" if args.dry_run else "ledger nav 列修复 —— APPLY")
    print("=" * 78)
    for p in plan:
        print(f"\n[{p['strategy']}] 成交日 {p['trade_days']} | ledger 行 {p['rows_total']}"
              f" | 将改 {p['rows_changed']} 行")
        print(f"    nav_daily 点 {p['nav_daily_points']} | nav_history {p['nav_hist_old']}"
              f" → {p['nav_hist_new']} 点")
        print(f"    末日 nav {p['last_nav']} vs portfolio.json {p['pf_value']}"
              f" → 残差 {p['resid']}")
        for d, v in p["day_value"].items():
            print(f"      {d}: {v:,.2f}")

    if problems:
        print("\n❌ 验收不过，不写盘：")
        for x in problems:
            print("   -", x)
        return 1

    if args.dry_run:
        print("\n✅ 验收通过（dry-run，未写盘）。加 --apply 执行。")
        return 0

    # ── 写盘
    for p in plan:
        strat = p["strategy"]
        b = backup(strat, "navcols")
        print(f"\n[{strat}] 备份: {b}")
        write_ledger(strat, p["fieldnames"], p["rows"])
        h = dict(p["holdings"])
        h["nav_history"] = p["nav_hist"]
        save_holdings(strat, h)
        print(f"[{strat}] 已写：ledger {p['rows_changed']} 行，nav_history {p['nav_hist_new']} 点")

    print("\n✅ 已写盘。下一步必须跑门禁 a/b/c + 后端 nav 末日核对。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
