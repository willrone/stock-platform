#!/usr/bin/env python3
"""多资产腿的**对照组** —— 回答「这 +38% 到底是不是真的？」。

## 为什么必须有对照组

`multi_asset_oos_validation.py` 证明了「A股腿 + ETF 多资产腿」在推导窗口和
样本外窗口都 beat 纯 A 股腿。但**这还没排除三个平凡解释**：

| 对照组 | 排除的解释 | 预期结果 |
|---|---|---|
| **现金**（50% A股 + 50% 现金） | 「夏普提升只是因为降杠杆」 | 夏普**不变**（缩放不改变夏普），只有回撤减半 |
| **沪深300 ETF**（50% A股 + 50% 沪深300） | 「只要换个证券类型就有用」 | 相关 ≈0.9 → 几乎无提升 |
| **中证500 ETF** | 同上，但风格更远 | 相关略低 → 提升有限 |
| **等权 4 类 ETF**（推荐配置） | 「真正起作用的是低相关资产」 | **明显提升** |

只有当「真配置」显著优于三个对照组时，结论才成立。

## 用法

    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/multi_asset_controls.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

from multi_asset_oos_validation import (  # noqa: E402
    ann_vol,
    corr,
    daily_rets,
    maxdd,
    merge_by_date,
    sharpe,
    total_ret,
)

MAIN_NAVS = os.path.join(ROOT, "data", "backtest", "multi_asset_navs.json")
OOS_NAVS = os.path.join(ROOT, "data", "backtest", "multi_asset_navs_oos.json")
ETF_PARQUET = os.path.join(ROOT, "data", "etf", "parquet")
OUT_PATH = os.path.join(ROOT, "data", "backtest", "multi_asset_controls.json")

Nav = List[Tuple[str, float]]


def load(path: str) -> Tuple[Nav, Dict[str, Nav]]:
    with open(path, encoding="utf-8") as fh:
        c = json.load(fh)
    return (
        [(d, v) for d, v in c["a_share"]],
        {k: [(d, v) for d, v in n] for k, n in c["classes"].items()},
    )


def etf_leg(code: str, start: str, end: str, capital: float = 1_000_000.0) -> Nav:
    """单只 ETF 的静态 buy & hold 腿（口径同 multi_asset_leg_experiment）。"""
    import pandas as pd

    path = os.path.join(ETF_PARQUET, f"{code}.parquet")
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path).sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(df) < 60:
        return []
    dates = sorted(df["date"])
    buy_date = dates[1]
    row = df[df["date"] == buy_date]
    if row.empty:
        return []
    px = float(row.iloc[0]["open"]) * (1 + 0.0001)
    shares = capital / px
    comm = shares * px * 0.0003
    nav = []
    for d in dates:
        r = df[df["date"] == d]
        p = float(r.iloc[0]["close"]) if not r.empty else px
        nav.append((d, shares * p))
    nav.insert(0, (dates[0], capital - comm))
    return nav


def main() -> None:
    out: Dict[str, Any] = {}

    for tag, path in (("main(推导)", MAIN_NAVS), ("oos(样本外)", OOS_NAVS)):
        a_nav, class_navs = load(path)
        start, end = a_nav[0][0], a_nav[-1][0]
        base_s = sharpe(a_nav)
        print("=" * 84)
        print(f"{tag}  {start} → {end}")
        print("=" * 84)
        print(f"  纯 A 股腿: 夏普 {base_s:.4f}  收益 {total_ret(a_nav):+.2%}  "
              f"回撤 {maxdd(a_nav):.2%}  波动 {ann_vol(a_nav):.4f}")
        print()
        res: Dict[str, Any] = {
            "window": f"{start}→{end}",
            "baseline_sharpe": round(base_s, 6),
            "baseline": {
                "sharpe": round(base_s, 6),
                "total_return": round(total_ret(a_nav), 6),
                "max_drawdown": round(maxdd(a_nav), 6),
                "annual_vol": round(ann_vol(a_nav), 6),
            },
            "controls": {},
        }

        # ── 对照组定义 ──
        # ★推荐配置（由「按与 A 股腿相关性降序剔除」的预先设定规则导出，
        #   两个窗口独立收敛到同一集合，见 multi_asset_oos_validation.py）
        RECOMMENDED = ["gold", "bond", "commodity"]
        if not all(c in class_navs for c in RECOMMENDED):
            RECOMMENDED = [c for c in sorted(class_navs) if c != "cross_border"]
        keep = RECOMMENDED

        rows: List[Tuple[str, str, Nav, Optional[List[float]], Dict[str, Any]]] = []

        # 对照 1：现金（零收益、零波动）
        cash_nav: Nav = [(d, 1.0) for d, _ in a_nav]
        rows.append((
            "对照1_现金50%", "50% A股 + 50% 现金（零波动）",
            merge_by_date([a_nav, cash_nav], [0.5, 0.5]),
            None, {"scheme": "capital_equal", "legs": ["a_share", "cash"]},
        ))

        # 对照 2/3：宽基 ETF（高 beta、高相关）
        for code, label in (("510300", "对照2_沪深300ETF50%"), ("510500", "对照3_中证500ETF50%")):
            leg = etf_leg(code, start, end)
            if not leg:
                print(f"  {label}: 无数据，跳过")
                continue
            cr = corr(daily_rets(a_nav), daily_rets(leg))
            rows.append((
                label, f"50% A股 + 50% {code}（宽基beta）",
                merge_by_date([a_nav, leg], [0.5, 0.5]),
                None, {"scheme": "capital_equal", "legs": ["a_share", code],
                       "corr_with_a_share": round(cr, 4)},
            ))

        # 对照 4：推荐的 3 类低相关组合
        rows.append((
            "★真配置_3类低相关", "50% A股 + 黄金/债券/商品 各16.67%",
            merge_by_date(
                [a_nav] + [class_navs[c] for c in keep],
                [0.5] + [0.5 / len(keep)] * len(keep),
            ),
            None, {"scheme": "capital_equal", "legs": ["a_share"] + keep},
        ))

        # 对照 4b：4 类版本（去 cross_border 但保留 dividend）
        keep4 = [c for c in sorted(class_navs) if c != "cross_border"]
        if len(keep4) > len(keep):
            rows.append((
                "参考_4类含dividend", "50% A股 + 黄金/债券/商品/红利 各12.5%",
                merge_by_date(
                    [a_nav] + [class_navs[c] for c in keep4],
                    [0.5] + [0.5 / len(keep4)] * len(keep4),
                ),
                None, {"scheme": "capital_equal", "legs": ["a_share"] + keep4},
            ))

        # 对照 5：gold + bond
        if "gold" in class_navs and "bond" in class_navs:
            rows.append((
                "真配置_gold+bond", "50% A股 + 黄金/债券 各25%",
                merge_by_date([a_nav, class_navs["gold"], class_navs["bond"]], [0.5, 0.25, 0.25]),
                None, {"scheme": "capital_equal", "legs": ["a_share", "gold", "bond"]},
            ))

        print(f"  {'配置':22s} {'夏普':>8s} {'vs基线':>9s} {'收益':>9s} {'回撤':>9s} {'波动':>8s} {'与A股相关':>10s}")
        print("  " + "-" * 80)
        for key, desc, combo, _w, extra in rows:
            s = sharpe(combo)
            cr = extra.get("corr_with_a_share")
            print(f"  {key:22s} {s:8.4f} {(s/base_s-1)*100:+8.2f}% {total_ret(combo):+8.2%} "
                  f"{maxdd(combo):8.2%} {ann_vol(combo):8.4f} "
                  f"{(f'{cr:+.4f}' if cr is not None else '     -')}")
            res["controls"][key] = {
                "desc": desc,
                **extra,
                "sharpe": round(s, 6),
                "vs_base_pct": round((s / base_s - 1) * 100, 3),
                "total_return": round(total_ret(combo), 6),
                "max_drawdown": round(maxdd(combo), 6),
                "annual_vol": round(ann_vol(combo), 6),
                "beats": s > base_s,
            }
        out[tag] = res
        print()

    # ── 结论判定 ──
    print("=" * 84)
    print("判定：真配置是否显著优于对照组？")
    print("=" * 84)
    verdicts: Dict[str, Any] = {}
    for tag in ("main(推导)", "oos(样本外)"):
        c = out[tag]["controls"]
        base_s = out[tag]["baseline_sharpe"]
        true_key = "★真配置_3类低相关"
        true_s = c[true_key]["sharpe"]
        cash_s = c["对照1_现金50%"]["sharpe"]
        # 注意：现金对照**不会精确等于**基线夏普。
        # 「50% 现金 + 50% 股票，买入持有不再平衡」的收益是
        # (0.5V_t + 0.5C)/(0.5V_{t-1} + 0.5C) - 1，不是 0.5·r_t，
        # 下跌时现金垫放大波动 → 轻微拖累，夏普略低于基线。
        # 结论只看方向：**现金不提升夏普**。
        cash_ok = cash_s <= base_s + 1e-9
        verdicts[tag] = {
            "cash_does_not_improve_sharpe": cash_ok,
            "cash_sharpe": round(cash_s, 6),
            "baseline_sharpe": round(base_s, 6),
            "true_config_sharpe": true_s,
            "true_vs_cash_sharpe_gap": round(true_s - cash_s, 6),
        }
        print(f"  {tag}:")
        print(f"    现金对照夏普 {cash_s:.4f} vs 基线 {base_s:.4f} "
              f"({(cash_s/base_s-1)*100:+.2f}%) → "
              f"{'✅ 降杠杆不提升夏普（平凡解释被排除）' if cash_ok else '❌ 异常：现金竟然提升了夏普'}")
        for k in ("对照2_沪深300ETF50%", "对照3_中证500ETF50%"):
            if k in c:
                v = c[k]
                ok = v["vs_base_pct"] < 10
                print(f"    {k}: 夏普 {v['sharpe']:.4f} ({v['vs_base_pct']:+.2f}%) "
                      f"相关 {v.get('corr_with_a_share'):+.4f} → "
                      f"{'✅ 高相关宽基无提升（相关性机制成立）' if ok else '⚠️ 提升超预期，需复核'}")
        print(f"    ★真配置: 夏普 {true_s:.4f} ({c[true_key]['vs_base_pct']:+.2f}%) "
              f"→ 比现金对照高 {true_s - cash_s:+.4f}")
        print()
    out["verdicts"] = verdicts

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"[ctl] ✓ 已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
