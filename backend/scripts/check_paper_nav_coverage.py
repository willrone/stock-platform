#!/usr/bin/env python3
"""纸面盘**覆盖 + 合并**门禁（2026-11-05，自用化 A 项）。

## 为什么需要它

自用版的第一条 KPI 是「收盘后 10 分钟出可下单清单 + 定稿线可跟」。
如果某条线**根本没有日频产物**，那它的曲线、夏普、超额全都是空气 ——
而空气不会自己报错。实测就是这个局面：6 条单线有 `<line>_nav_daily.json`，
唯独定稿线 `multi_asset_final` 和它的两条子腿没有。

## 本脚本判什么（任一不满足 → 退出码非 0）

| # | 判据 | 红的时候说明什么 |
|---|---|---|
| 1 | 每条**有线上成交**的线都有 `<line>_nav_daily.json` 且 ≥1 个点 | 日频盯市漏了这条线（覆盖缺口） |
| 2 | 带 `legs` 的母线，**每条子腿**都要有日频产物 | 母线只跟了一半 |
| 3 | 母线最新点的两条腿日期必须**相同** | 两条腿不同步（拿不同日拼出来的净值是假的） |
| 4 | 母线最新点的 `value` 必须等于两腿 `value` 之和（容差 0.05） | 账户层合并口径漂移 |
| 5 | 每条线的日频末点日期必须 ≥ 它自己的 `last_trade` | 成交之后没有盯市（曲线停在成交日） |

用法：

```bash
cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/check_paper_nav_coverage.py
```

⚠️ 只读，但**照样取纸面盘互斥锁**（§12.52：读 holdings 读到半成品
会产出"看起来正常的错结论"，门禁自己就会误报绿）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

_SCRIPT = os.path.join(BACKEND_ROOT, "scripts", "paper_portfolio.py")
_spec = importlib.util.spec_from_file_location("pp_for_coverage", _SCRIPT)
assert _spec and _spec.loader
pp = importlib.util.module_from_spec(_spec)
sys.modules["pp_for_coverage"] = pp
_spec.loader.exec_module(pp)

PAPER_DIR = pp.PAPER_DIR
#: 母线末点与两腿之和的容差（元）—— 与门禁 a 同一量级（逐票 2dp 舍入）
MERGE_TOLERANCE = 0.05


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


def list_lines() -> Tuple[List[str], List[str]]:
    """`data/paper/*_holdings.json` → (母线, 子腿线)。

    子腿用 `paper_portfolio` 的常量判定，不用字符串后缀猜测
    （后缀猜测会把将来出现的 `*_a2` 之类误判成母线）。
    """
    subleg_keys = {pp.MULTI_ASSET_A_KEY, pp.MULTI_ASSET_SLEEVE_KEY}
    mains: List[str] = []
    sublegs: List[str] = []
    for name in sorted(os.listdir(PAPER_DIR)):
        if not name.endswith("_holdings.json"):
            continue
        line = name[: -len("_holdings.json")]
        (sublegs if line in subleg_keys else mains).append(line)
    return mains, sublegs


def main() -> int:
    if not os.path.isdir(PAPER_DIR):
        print(f"❌ 找不到纸面盘目录：{PAPER_DIR}")
        return 1

    try:
        fd = pp.acquire_run_lock(timeout_s=120)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 取纸面盘互斥锁失败：{exc}")
        return 1
    print(f"✅ 已取得纸面盘互斥锁（fd={fd}）", flush=True)

    mains, sublegs = list_lines()
    problems: List[str] = []
    rows: List[Dict[str, Any]] = []

    for line in mains:
        hold = _load_json(os.path.join(PAPER_DIR, f"{line}_holdings.json")) or {}
        nav = _load_json(os.path.join(PAPER_DIR, f"{line}_nav_daily.json"))
        last_trade = hold.get("last_trade")
        points = list((nav or {}).get("points") or [])
        nav_last = points[-1] if points else None

        row: Dict[str, Any] = {
            "line": line,
            "last_trade": last_trade,
            "nav_points": len(points),
            "nav_last_date": (nav_last or {}).get("date"),
            "basis": (nav or {}).get("basis"),
        }

        # 门禁 e（2026-11-05 建设项 B）：净值点必须带零技巧基准两列。
        # 理由：自用 KPI 是"相对零技巧基准的超额"，一个没有基准的净值点
        # 只能说明"赚了多少"，不能说明"做对了没有"——不许出表。
        bench_missing = sum(
            1 for p in points
            if p.get("benchmark_return_pct") is None or p.get("excess_pp") is None)
        row["bench_missing_points"] = bench_missing
        row["excess_pp_last"] = (nav_last or {}).get("excess_pp")
        if points and bench_missing:
            problems.append(
                f"{line}：{bench_missing}/{len(points)} 个净值点缺基准列"
                "（跑 build_benchmark_index.py 后重跑 paper_daily_nav.py）")

        if last_trade and not points:
            problems.append(f"{line}：有成交（{last_trade}）却没有日频产物")
        if nav and nav.get("_error"):
            problems.append(f"{line}：nav_daily 读不出来（{nav['_error']}）")
        if nav_last and last_trade and str(nav_last.get("date")) < str(last_trade):
            problems.append(
                f"{line}：日频末点 {nav_last.get('date')} 早于最后成交日 {last_trade}"
                " → 成交之后没有盯市")

        # ── 母线：子腿覆盖 + 逐位相加 ────────────────────────────────
        legs = hold.get("legs") or {}
        if legs:
            leg_last: Dict[str, Dict[str, Any]] = {}
            for leg_name in legs:
                leg_key = f"{line}_{leg_name}"
                leg_nav = _load_json(os.path.join(PAPER_DIR,
                                                  f"{leg_key}_nav_daily.json"))
                leg_pts = list((leg_nav or {}).get("points") or [])
                row[f"leg_{leg_name}_points"] = len(leg_pts)
                if not leg_pts:
                    problems.append(
                        f"{line}：子腿 {leg_key} 没有日频产物（母线只跟了一半）")
                    continue
                leg_last[leg_name] = leg_pts[-1]
            if len(leg_last) == len(legs) and legs and nav_last:
                dates = {str(p.get("date")) for p in leg_last.values()}
                if len(dates) != 1:
                    problems.append(
                        f"{line}：两条腿末点日期不齐 {sorted(dates)}"
                        " → 不同日的点不能相加")
                else:
                    total = sum(float(p.get("value") or 0.0)
                                for p in leg_last.values())
                    diff = abs(total - float(nav_last.get("value") or 0.0))
                    row["merge_diff"] = round(diff, 4)
                    if diff > MERGE_TOLERANCE:
                        problems.append(
                            f"{line}：母线末点 {nav_last.get('value')} ≠ "
                            f"两腿之和 {total:.2f}（差 {diff:.2f} > "
                            f"{MERGE_TOLERANCE}）")
        rows.append(row)

    print(f"\n母线 {len(mains)} 条 / 子腿 {len(sublegs)} 条"
          f"（{', '.join(sublegs)}）")
    print(f"{'线':<22} {'成交日':<12} {'点数':>5}  {'末点日期':<12} {'合并差':>8}")
    for r in rows:
        print(f"{r['line']:<22} {str(r['last_trade'] or '-'):<12} "
              f"{r['nav_points']:>5}  {str(r['nav_last_date'] or '-'):<12} "
              f"{('%.4f' % r['merge_diff']) if r.get('merge_diff') is not None else '-':>8}")

    if problems:
        print("\n❌ 覆盖/合并门禁失败：")
        for p in problems:
            print(f"   · {p}")
        return 1

    print("\n✅ 覆盖/合并门禁全部通过"
          f"（{len(mains)} 条母线都有日频产物；母线=两腿之和）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
