#!/usr/bin/env python3
"""P5 · gold / commodity 类的分类审计。

## 为什么做这个

P3 的最大收获是：**bond 类混进了可转债 ETF（资产性质不一致）→ 分类纠错带来唯一被采纳的改进。**
既然一个类能被污染，其他类也可能。P5 就是把同一把尺子量到 gold / commodity 上。

## 三个检查

1. **性质审计**：类内每只到底是不是"同一类资产"？
   - gold 类：与纯金基准 518880 的 ρ（白银/黄金股会明显偏低）
   - commodity 类：与商品基准 159980 的 ρ（`510170` 已知与商品其余近乎零相关）
2. **对 A 股腿的 ρ**：真正的"分散性"来源。
3. **"流动性 top-3"规则的漂移**：P3 已证 bond 类的 top-3 会随市场规模漂移
   （2019 选长债、2023 选可转债）。gold / commodity 类是否也有同样问题？
   → 三个窗口各算一次 top-3，看是否收敛到同一组。

## 用法

    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/p5_class_audit.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_p3_bond as P3  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    END,
    OOS_END,
    OOS_START,
    START,
    load_etf_universe,
)

PROXY = "510300"
GOLD_REF = "518880"       # 华安黄金 ETF —— 纯金基准
COMM_REF = "159980"       # 大成有色金属期货 ETF —— 商品基准
ANN = 244


def _classify(universe: Dict[str, Any], meta: Dict[str, Any]):
    name = meta.get("name", "") or ""
    tags = [str(t) for t in (meta.get("tags") or [])]
    hay = name + " " + " ".join(tags)
    for o in universe.get("class_overrides", []):
        if any(k in hay for k in o["keywords"]):
            return o["target"]
    for cls, kws in universe.get("asset_classes", {}).items():
        if any(k in hay for k in kws):
            return cls
    return None


def members(universe, cls: str, have_data) -> List[str]:
    out = []
    for code, meta in universe.get("etfs", {}).items():
        if code not in have_data:
            continue
        if _classify(universe, meta) == cls:
            out.append(code)
    return sorted(out)


def _rets(code: str):
    df = P3._load_code(code)
    if df is None or df.empty:
        return None
    return P3._rets(df)


def _corr(a, b, lo, hi):
    import numpy as np
    import pandas as pd
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    j = j[(j.index >= lo) & (j.index <= hi)]
    if len(j) < 60:
        return None, 0
    x, y = j.iloc[:, 0], j.iloc[:, 1]
    if x.std() == 0 or y.std() == 0:
        return None, len(j)
    return float(np.corrcoef(x, y)[0, 1]), len(j)


def _stats(code: str, lo, hi):
    r = _rets(code)
    if r is None:
        return None
    s = r[(r.index >= lo) & (r.index <= hi)]
    if len(s) < 60:
        return None
    import numpy as np
    return {"vol": float(s.std() * np.sqrt(ANN)), "n": int(len(s))}


def _turnover(code: str, before):
    df = P3._load_code(code)
    if df is None or df.empty:
        return 0.0
    df = df[df["date"] < before].tail(P3.LIQ_LOOKBACK)
    if not len(df):
        return 0.0
    return float((df["volume"].fillna(0) * df["close"] * 100).mean())


def main() -> None:
    import pandas as pd

    universe = load_etf_universe()
    have_data = {os.path.basename(p)[:-8]
                 for p in glob.glob(os.path.join(P3.PARQUET_DIR, "*.parquet"))}
    out_path = os.path.join(ROOT, "data", "backtest", "p5_class_audit.json")

    proxy_r = _rets(PROXY)
    gold_r = _rets(GOLD_REF)
    comm_r = _rets(COMM_REF)

    # 三个窗口：长历史起点 / OOS 起点 / 推导窗口起点
    # 三个起点：2020-01（代理 510300 有 2019 全年做窗口前）/ OOS 起点 / 推导窗口起点
    # 注意：2019-01 起点做不了 —— 代理数据从 2019-01-02 起，窗口前为空。
    windows = [
        ("w2020", datetime(2020, 1, 1)),
        ("oos_2021", OOS_START),
        ("main_2023", START),
    ]

    report: Dict[str, Any] = {"windows": {}, "reference": {
        "proxy": PROXY, "gold_ref": GOLD_REF, "commodity_ref": COMM_REF}}

    for wname, wstart in windows:
        ws = pd.Timestamp(wstart.strftime("%Y-%m-%d"))
        pre = proxy_r[proxy_r.index < ws].iloc[-P3.RHO_LOOKBACK:]
        if len(pre) < 60:
            print(f"[p5] ⚠️ {wname}: 代理窗口前数据不足（{len(pre)}），跳过", flush=True)
            continue
        lo, hi = pre.index[0], pre.index[-1]

        entry: Dict[str, Any] = {"start": str(ws.date()), "classes": {}}
        print(f"\n{'='*78}\n[p5] 窗口 {wname}  窗口前 {lo.date()} ~ {hi.date()}\n{'='*78}",
              flush=True)

        for cls in ("gold", "commodity"):
            codes = members(universe, cls, have_data)
            ref_r = gold_r if cls == "gold" else comm_r
            ref_code = GOLD_REF if cls == "gold" else COMM_REF
            rows = []
            for c in codes:
                r = _rets(c)
                if r is None:
                    continue
                rho_a, n_a = _corr(r, proxy_r, lo, hi)
                rho_ref, _ = _corr(r, ref_r, lo, hi)
                st = _stats(c, lo, hi)
                tq = _turnover(c, ws)
                rows.append({
                    "code": c,
                    "name": universe["etfs"].get(c, {}).get("name", ""),
                    "rho_proxy": round(rho_a, 4) if rho_a is not None else None,
                    "rho_ref": round(rho_ref, 4) if rho_ref is not None else None,
                    "vol": round(st["vol"], 4) if st else None,
                    "turnover": round(tq, 0),
                    "n": n_a,
                    "liquid": tq >= P3.MIN_TURNOVER,
                })
            rows.sort(key=lambda x: (x["turnover"] is None, -(x["turnover"] or 0)))
            entry["classes"][cls] = {
                "ref": ref_code,
                "rows": rows,
                "by_code": {x["code"]: {k2: x[k2] for k2 in
                                        ("rho_ref", "rho_proxy", "vol", "turnover")}
                            for x in rows},
                "top3_by_liquidity": [x["code"] for x in rows[:3]],
                "lowest_rho_ref": [x["code"] for x in
                                   sorted(rows, key=lambda y: (y["rho_ref"] is None,
                                                              y["rho_ref"] if y["rho_ref"] is not None else 9))[:3]],
                "suspicious": [x["code"] for x in rows
                               if x["rho_ref"] is not None and x["rho_ref"] < 0.85],
            }
            print(f"\n[p5] --- {cls} 类（基准 {ref_code}）---", flush=True)
            for x in rows:
                flag = "  ⚠️ 性质存疑" if (x["rho_ref"] is not None and x["rho_ref"] < 0.85) else ""
                vol_s = f"{x['vol']:.2%}" if x["vol"] is not None else "n/a"
                tq_s = f"{x['turnover']/1e4:,.0f}万" if x["turnover"] else "n/a"
                print(f"  {x['code']}  ρ(基准) {str(x['rho_ref']):>7}  ρ(A股代理) {str(x['rho_proxy']):>7}  "
                      f"波 {vol_s:>7}  成交额 {tq_s:>11}  {x['name']}{flag}",
                      flush=True)
            print(f"  → 流动性 top-3: {entry['classes'][cls]['top3_by_liquidity']}", flush=True)
            if entry["classes"][cls]["suspicious"]:
                print(f"  → ⚠️ 与基准 ρ < 0.85（性质可能不一致）: "
                      f"{entry['classes'][cls]['suspicious']}", flush=True)

        report["windows"][wname] = entry

    # ── top-3 漂移检查 ──
    drift: Dict[str, Any] = {}
    for cls in ("gold", "commodity"):
        tops = {w: report["windows"].get(w, {}).get("classes", {}).get(cls, {})
                .get("top3_by_liquidity") for w, _ in windows
                if w in report["windows"]}
        tops = {k: v for k, v in tops.items() if v}
        same = len({tuple(sorted(v)) for v in tops.values()}) == 1 if tops else None
        drift[cls] = {"top3_by_window": tops, "stable": same}
        print(f"\n[p5] {cls} 类 top-3 漂移检查: "
              f"{'稳定 ✅' if same else '漂移 ⚠️'}  {tops}", flush=True)
    report["top3_drift"] = drift

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\n[p5] ✓ 已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
