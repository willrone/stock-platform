#!/usr/bin/env python3
"""验证：产品用的袖套实现（app/services/trading/multi_asset_service.py）
是否与研究侧口径（P3.build_targets + L3.simulate）一致。

五步：
  ① 研究口径复现   —— open 成交 + 小数股 + 研究侧撮合器 → 夏普应 = 1.6244
  ② 产品实现 @open —— 同一日历、open 成交、**小数股** → 应与 ① **逐点一致**
                      （这一步证明「实现等价」，不是「数字巧合」）
  ③ 产品实现 @close + 整手 —— 产品真实口径（成交日收盘价 + 100 股整手）
  ④ 组合层重算：把研究袖套换成产品袖套，重算组合夏普
  ⑤ 产物落盘

**关键区分（别搞混）**：
  - 成交价（trade price）：调仓日实际成交的价格 → 研究=open，产品=close
  - 标记价（mark price）  ：每日净值估值的价格 → **永远是 close**
  研究侧 simulate 是「open 成交、open 算目标权重、close 记净值」；
  产品是「close 成交、close 算目标权重、close 记净值」。

用法:
    cd backend && env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 \
        scripts/verify_sleeve_impl.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p2 as P2  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
from multi_asset_leg_experiment import load_etf_universe  # noqa: E402

from app.services.trading.multi_asset_service import (  # noqa: E402
    ETF_PARQUET_FROZEN,
    is_rebalance_day,
    load_etf_bars,
    rebalance_sleeve,
    sleeve_positions_value,
    sleeve_target_weights,
)

P5_FIXED = {
    "gold": ["518880", "159934", "159937"],
    "bond": ["511260", "511010", "511020"],
    "commodity": ["159980", "159985", "159981"],
}
CLASSES = ["gold", "bond", "commodity"]
WINDOW = ("2023-05-08", "2026-05-08")


def sharpe(nav: List[Tuple[str, float]]) -> float:
    """**必须与研究侧完全同口径**：pstdev（总体标准差）+ 年化 √244。

    研究侧 `multi_asset_leg_experiment.sharpe` 用 statistics.pstdev + sqrt(244)。
    换成 pandas 的 ddof=1 + sqrt(252) 会得到 1.6497 而不是 1.6244
    （差 1.7%），数字对不上就会被误判成"实现不等价"。
    """
    import statistics
    rets = []
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        if p0 and p0 > 0:
            rets.append(p1 / p0 - 1.0)
    if len(rets) < 3:
        return 0.0
    mu = statistics.fmean(rets)
    sd = statistics.pstdev(rets)
    return mu / sd * np.sqrt(244.0) if sd > 0 else 0.0


def maxdd(nav: List[Tuple[str, float]]) -> float:
    s = pd.Series([v for _, v in nav])
    return float((s / s.cummax() - 1.0).min())


def product_sleeve_nav(
    calendar: List[str],
    bars: Dict[str, pd.DataFrame],
    trade_price_field: str,
    board_lot: int,
    capital: float = 1_000_000.0,
    anchor: int = 1,
    charge_sell_commission: bool = True,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """用产品实现跑袖套。

    - 调仓网格：每 21 个交易日一次，锚点 calendar[1]（与研究侧 P3.build_targets 同口径）
    - 成交价 = trade_price_field
    - **净值标记价 = close（永远）**
    """
    target = sleeve_target_weights()
    positions: Dict[str, Dict[str, Any]] = {}
    cash = capital
    nav: List[Tuple[str, float]] = []
    turnover = 0.0
    n_rebal = 0

    for i, d in enumerate(calendar):
        ts = pd.Timestamp(d)
        if is_rebalance_day(i, anchor):
            n_rebal += 1
            positions, cash, buys, sells = rebalance_sleeve(
                positions, cash, bars, ts, target,
                price_field=trade_price_field, board_lot=board_lot,
                charge_sell_commission=charge_sell_commission,
            )
            turnover += sum(b["amount"] for b in buys) + sum(s["amount"] for s in sells)
        val = cash + sleeve_positions_value(positions, bars, ts, price_field="close")
        nav.append((d, val / capital))

    rec = {
        "n_rebal": n_rebal,
        "turnover_ratio": turnover / capital,
        "final_value": nav[-1][1] * capital,
    }
    return nav, rec


def main() -> None:
    s_str, e_str = WINDOW
    held = {c: list(P5_FIXED[c]) for c in P5_FIXED}

    # ── 与 P18 完全同口径地重建日历 ──────────────────────────────────
    pool = P3.build_pool(datetime.strptime(s_str, "%Y-%m-%d"))
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(
        universe, datetime.strptime(s_str, "%Y-%m-%d"),
        datetime.strptime(e_str, "%Y-%m-%d"),
    )
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    print(f"[verify] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}"
          f"（P18 用的也是这个日历）")

    # ── ① 研究口径复现 ───────────────────────────────────────────────
    tg, _meta = P3.build_targets(calendar, held, P3.fixed_weights_for(CLASSES))
    c_of = {c: cls for cls, lst in held.items() for c in lst}
    ref_abs, ref_rec = L3.simulate(
        calendar, prices, tg, c_of, {c: 1.0 / len(CLASSES) for c in CLASSES}
    )
    ref_nav = [(d, v / ref_abs[0][1]) for d, v in ref_abs]
    ref_sh = sharpe(ref_nav)
    print(f"[verify] ① 研究口径（open 成交 + 小数股）: 夏普 {ref_sh:.4f} | "
          f"波动 {P2.nav_vol(ref_nav):.4f} | 换手 {ref_rec['turnover_ratio']:.4f} | "
          f"末值 {ref_nav[-1][1]:.6f}  （P18 记录值 1.6244 / 0.0848 / 1.8524）")

    # ── ② 产品实现 @open（等价性验证）────────────────────────────────
    # 等价性对比必须用**冻结快照**：研究侧读的是冻结目录，
    # 实盘目录会被周任务重拉（前复权锚点重锚）→ 两边数据源必须一致
    bars = load_etf_bars(
        sorted({c for lst in held.values() for c in lst}),
        base_dir=ETF_PARQUET_FROZEN,
    )
    nav_open_frac, rec_open_frac = product_sleeve_nav(
        calendar, bars, trade_price_field="open", board_lot=0,
        charge_sell_commission=False,   # 对齐研究侧口径（研究侧卖出不扣佣金）
    )
    sh_open_frac = sharpe(nav_open_frac)
    maxdiff = max(abs(a[1] - b[1]) for a, b in zip(ref_nav, nav_open_frac))
    print(f"[verify] ② 产品实现 @open（无整手）: 夏普 {sh_open_frac:.4f} | "
          f"波动 {P2.nav_vol(nav_open_frac):.4f} | "
          f"换手 {rec_open_frac['turnover_ratio']:.4f} | "
          f"末值 {nav_open_frac[-1][1]:.6f}")
    print(f"[verify]    与研究口径净值最大逐点差 = {maxdiff:.3e}  "
          f"({'✅ 逐点一致（实现等价）' if maxdiff < 1e-9 else '⚠️ 有差异'})")
    if maxdiff >= 1e-9:
        for (d1, v1), (d2, v2) in zip(ref_nav, nav_open_frac):
            if abs(v1 - v2) > 1e-9:
                print(f"      首个差异 {d1}: 研究 {v1:.8f} vs 产品 {v2:.8f}")
                break

    # ── ③ 产品实现 @close + 整手（产品真实口径）──────────────────────
    nav_close, rec_close = product_sleeve_nav(
        calendar, bars, trade_price_field="close", board_lot=100
    )
    sh_close = sharpe(nav_close)
    print(f"[verify] ③ 产品实现 @close + 整手: 夏普 {sh_close:.4f} | "
          f"波动 {P2.nav_vol(nav_close):.4f} | "
          f"换手 {rec_close['turnover_ratio']:.4f} | "
          f"末值 {nav_close[-1][1]:.6f} | 回撤 {maxdd(nav_close):.4f}")

    # ── ④ 组合层重算 ────────────────────────────────────────────────
    p18_path = os.path.join(ROOT, "data", "backtest", "multi_asset_p18_main.json")
    with open(p18_path) as fh:
        p18 = json.load(fh)
    comb = [(d, float(v)) for d, v in p18["navs"]["orig42f"]]
    # ⚠️ 反推 A 腿净值时必须用**研究侧袖套**（ref_nav），
    #    用产品袖套会把 sleeve 项抵消掉（组合 = 0.5a + 0.5s，s 一进一出就没了）
    s_open_map = dict(ref_nav)
    s_close_map = dict(nav_close)

    def _ff(m: Dict[str, float], d: str):
        """取 d 当天或之前最近的一个袖套净值（袖套日历比引擎日历少 1 天）。"""
        if d in m:
            return m[d]
        prev = [k for k in m if k <= d]
        return m[max(prev)] if prev else None

    def build_comb(new_map: Dict[str, float]) -> List[Tuple[str, float]]:
        """combined = 0.5*a_norm + 0.5*s_norm；由研究组合反推 a_norm，再换上新袖套。

        袖套日历（ETF 交易日并集，726 天）与引擎日历（727 天）差一天，
        缺的那天用袖套的前一个值（前向填充）——否则组合点数对不上，
        算出来的夏普会差 0.04（实测 1.4545 vs 1.4114）而看不出原因。
        """
        out: List[Tuple[str, float]] = []
        for d, c in comb:
            s_old = _ff(s_open_map, d)
            s_new = _ff(new_map, d)
            if s_old is None or s_new is None:
                continue
            a_norm = 2.0 * c - s_old             # 研究袖套下的 A 腿归一净值
            out.append((d, 0.5 * a_norm + 0.5 * s_new))
        return out

    comb_ref = build_comb(s_open_map)
    comb_prod = build_comb(s_close_map)
    if len(comb_ref) != len(comb):
        print(f"[verify]    ⚠️ 组合点数 {len(comb_ref)}/{len(comb)}（日历需对齐）")
    print(f"[verify] ④ 组合 @研究袖套(open)   夏普 {sharpe(comb_ref):.4f}  "
          f"（P18 记录的 base = {p18['base_combined_sharpe']:.4f}，应一致）")
    print(f"[verify]    组合 @产品袖套(close+整手) 夏普 {sharpe(comb_prod):.4f}  "
          f"回撤 {maxdd(comb_prod):.4f}  末值 {comb_prod[-1][1]:.4f}")

    out = {
        "research_open_fractional": {
            "sharpe": round(ref_sh, 4), "vol": round(P2.nav_vol(ref_nav), 4),
            "turnover": round(ref_rec["turnover_ratio"], 4),
            "p18_reference": {"sharpe": 1.6244, "vol": 0.0848, "turnover": 1.8524},
        },
        "product_open_fractional": {
            "sharpe": round(sh_open_frac, 4),
            "max_pointwise_diff_vs_research": maxdiff,
            "equivalent": bool(maxdiff < 1e-9),
        },
        "product_close_boardlot": {
            "sharpe": round(sh_close, 4), "vol": round(P2.nav_vol(nav_close), 4),
            "turnover": round(rec_close["turnover_ratio"], 4),
            "maxdd": round(maxdd(nav_close), 4),
        },
        "combined_research_sleeve": round(sharpe(comb_ref), 4),
        "combined_product_sleeve": round(sharpe(comb_prod), 4),
        "combined_product_sleeve_maxdd": round(maxdd(comb_prod), 4),
    }
    # 2026-11-01：结论产物归位到 data/backtest/（与其它产物同目录）
    # —— 原来写在 backend/scripts/ 下 → 不在任何指纹清单的覆盖范围内
    #    （指纹门只管 data/backtest/），等于一份没人守护的结论。
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "backtest", "sleeve_verify.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n[verify] 结论产物 → data/backtest/sleeve_verify.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
