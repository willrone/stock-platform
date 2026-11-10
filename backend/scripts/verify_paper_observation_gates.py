#!/usr/bin/env python3
"""纸面盘观测口径层的**产物级门禁**（方案书附录 A.4 的 a/b/c 三条）。

为什么是独立脚本而不是只靠单测：
  单测验的是**函数**，这里验的是**产物**（data/paper/portfolio.json）——
  MCP 工具、API、前端读的都是这个文件，产物不合规=所有出口都不合规。

用法：
    python3 scripts/verify_paper_observation_gates.py [--json]

退出码：0 = 三条门禁全过；1 = 有门禁不过（逐条打印实测值）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# 本脚本在 backend/scripts/ → 上两级是仓库根（与 paper_portfolio.py 的
# BACKEND_ROOT/ROOT 口径一致，别自己发明第二套路径推导）
# backend/scripts/x.py → 上三级才是仓库根（data/paper 在那儿）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(BACKEND_ROOT)
PORTFOLIO = os.path.join(ROOT, "data", "paper", "portfolio.json")


def _f(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args()

    if not os.path.exists(PORTFOLIO):
        print(f"❌ 找不到 {PORTFOLIO}")
        return 1
    with open(PORTFOLIO) as fh:
        data = json.load(fh)
    lines: Dict[str, Any] = data.get("lines") or {}
    meta = data.get("meta") or {}
    capital = _f(meta.get("capital_per_line")) or 1_000_000.0

    failures: List[str] = []
    report: Dict[str, Any] = {}

    # ── 门禁 a：口径自洽 ────────────────────────────────────────────
    #   exposure == 1 − cash/value；invested_return == (value−initial)/(value−cash)
    a_rows = []
    for name, ln in lines.items():
        value = _f(ln.get("value"))
        cash = _f(ln.get("cash"))
        exposure = _f(ln.get("exposure"))
        inv = _f(ln.get("invested_return_pct"))
        if value is None or cash is None:
            failures.append(f"[a] {name}: 缺 value/cash")
            continue
        exp_expect = (1.0 - cash / value) if value > 0 else None
        invested = value - cash
        inv_expect = ((value - capital) / invested * 100.0) if invested > 1e-9 else None
        ok_exp = (exposure is None and exp_expect is None) or (
            exposure is not None and exp_expect is not None
            and abs(exposure - exp_expect) < 1e-4)
        ok_inv = (inv is None and inv_expect is None) or (
            inv is not None and inv_expect is not None
            and abs(inv - inv_expect) < 1e-3)
        if not ok_exp:
            failures.append(
                f"[a] {name}: exposure={exposure} ≠ 1−cash/value={exp_expect}")
        if not ok_inv:
            failures.append(
                f"[a] {name}: invested_return_pct={inv} ≠ "
                f"(value−initial)/(value−cash)={inv_expect}")
        a_rows.append({"strategy": name, "exposure": exposure,
                       "invested_return_pct": inv, "ok": ok_exp and ok_inv})
    report["a"] = {"rows": a_rows, "failures": len([f for f in failures if f.startswith("[a]")])}

    # ── 门禁 b：绝不冒充打平 ────────────────────────────────────────
    #   n_trades == 0 → pnl_display 必须是 "—"（不是 "0.00%"）、status=未建仓、
    #   invested_return_pct is None
    b_rows = []
    for name, ln in lines.items():
        n_trades = int(ln.get("n_trades") or 0)
        if n_trades != 0:
            continue
        disp = ln.get("pnl_display")
        status = ln.get("status")
        inv = ln.get("invested_return_pct")
        pnl = ln.get("pnl_pct")
        ok = (disp == "—" and status == "未建仓" and inv is None
              and (pnl is None or pnl == 0.0))
        if not ok:
            failures.append(
                f"[b] {name}: 从未成交却 pnl_display={disp!r} status={status!r} "
                f"pnl_pct={pnl!r} invested_return_pct={inv!r}")
        b_rows.append({"strategy": name, "pnl_display": disp, "status": status,
                       "invested_return_pct": inv, "ok": ok})
    report["b"] = {"rows": b_rows,
                   "failures": len([f for f in failures if f.startswith("[b]")])}

    # ── 门禁 c：as-of 诚实 ──────────────────────────────────────────
    #   每条线的 as_of 必须是它自己估值用的数据日；不一致时汇总层必须标注
    asofs = sorted({str(ln.get("as_of")) for ln in lines.values() if ln.get("as_of")})
    no_asof = [n for n, ln in lines.items()
               if int(ln.get("n_trades") or 0) > 0 and not ln.get("as_of")]
    if no_asof:
        failures.append(f"[c] 已成交但缺 as_of 的线: {no_asof}")
    c_ok = not no_asof
    report["c"] = {"as_of_dates": asofs, "consistent": len(asofs) <= 1,
                   "missing_as_of": no_asof, "ok": c_ok}

    if args.json:
        print(json.dumps({"ok": not failures, "failures": failures,
                          "report": report}, ensure_ascii=False, indent=2))
        return 0 if not failures else 1

    print("=" * 72)
    print("纸面盘观测口径层 · 产物级门禁（附录 A.4 a/b/c）")
    print("=" * 72)
    print(f"\n[门禁 a] 口径自洽（exposure / invested_return_pct 与 portfolio.json 现值逐项相符）")
    for r in a_rows:
        mark = "✅" if r["ok"] else "❌"
        print(f"  {mark} {r['strategy']:<20} 暴露={r['exposure']}  投入段={r['invested_return_pct']}")
    print(f"\n[门禁 b] 绝不冒充打平（从未成交的线：pnl_display='—' 且 status=未建仓）")
    if not b_rows:
        print("  （本次所有线都有成交记录，无从未成交的线）")
    for r in b_rows:
        mark = "✅" if r["ok"] else "❌"
        print(f"  {mark} {r['strategy']:<20} pnl_display={r['pnl_display']!r}  status={r['status']!r}")
    print(f"\n[门禁 c] as-of 诚实")
    print(f"  各线 as_of: {asofs}")
    print(f"  {'✅ 一致' if len(asofs) <= 1 else '⚠️ 不一致 → 汇总层已标注「排名仅供参考，不是同一时点的比较」'}")
    if no_asof:
        print(f"  ❌ 缺 as_of 的线: {no_asof}")
    print()
    if failures:
        print(f"❌ {len(failures)} 条门禁不过：")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("✅ 门禁 a/b/c 全过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
