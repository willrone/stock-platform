#!/usr/bin/env python3
"""一次性修复：adj_cost 无效（NaN/None）的持仓 → 用 ledger 的**成交日**回填，
并用**修正后的口径**重算组合市值 + 重算 nav_history 的最后一个点。

为什么需要（2026-11-02 发现的真 bug，污染了当晚的账本）
--------------------------------------------------------
`paper_portfolio.py` 里买入时写 `adj_cost` 取的是 **信号日**（`latest`）的后复权价，
而买入发生在 **成交日**（lag1 口径）→ 分子分母落在不同日期；
更糟的是信号日该票没有后复权价时得到 **NaN**，而 `if a_cost else None` 里
**NaN 是 truthy** → NaN 被写进持仓 → `_adj_ratio` 因
`not np.isfinite(a_cost)` 恒返回 1.0 → **该票市值被冻结在买入金额，
永不反映涨跌**。

实测 2026-11-02 首跑：multi_asset_final **69/69 只**、其余各线 3~12 只中招
→ 那条线当时显示的 "-0.10%" 其实是**手续费损耗**，不含任何市场波动。

本脚本做三件事（**只改估值字段，不成交、不改 ledger**）
------------------------------------------------------
1. 用 ledger 最近一次 BUY 的**成交日**修正每只持仓的 `buy_date`
2. 给 `adj_cost` 无效的持仓回填「成交日的后复权价」
3. 用修正后的口径（as_of = 数据末日）**重算组合市值** → 写回
   `last_value` / `value_as_of`，并**重算 nav_history 的最后一个点**
   （历史点需要历史持仓才能重算 → 留给日频盯市的回补，见方案书正文）

用法（必须在 backend 目录下、用项目 venv）：
    env -u PYTHONPATH .venv-py313/bin/python3 scripts/fix_adj_cost_backfill.py [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

_SCRIPT = os.path.join(BACKEND_ROOT, "scripts", "paper_portfolio.py")
_spec = importlib.util.spec_from_file_location("paper_portfolio_mod", _SCRIPT)
assert _spec and _spec.loader
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

from app.services.trading.multi_asset_service import (  # noqa: E402
    load_etf_bars,
    sleeve_positions_value,
)

# 需要修的线：6 条单条线 + 多资产线的 A 腿（袖套腿用前复权价，不依赖 adj_cost）
SINGLE_LINES = ["22f", "honest_42f", "rsi_71421", "factor_ranking",
                 "gbm_voltgt", "honest_42f_neu"]
A_LEG = "multi_asset_final_a"
MERGED = "multi_asset_final"
SLEEVE = "multi_asset_final_sleeve"


def _valid_adj(v: Any) -> bool:
    if v is None:
        return False
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(f) and f > 0)


def fix_line(strategy: str, adj_close: pd.DataFrame, as_of: pd.Timestamp,
             dry: bool) -> Dict[str, Any]:
    """修一条线的 buy_date / adj_cost，并返回重算后的市值。"""
    paper = pp.load_holdings(strategy, 1_000_000.0)
    positions = paper.get("positions") or {}
    if not positions:
        return {"strategy": strategy, "n_pos": 0, "fixed": 0, "value": None,
                "value_before": None}
    value_before = paper.get("last_value")
    buy_dates = pp.ledger_buy_dates(strategy)

    fixed = 0
    date_fixed = 0
    for code, pos in positions.items():
        bd = buy_dates.get(code)
        if bd and str(pos.get("buy_date") or "") != str(bd):
            pos["buy_date"] = str(bd)
            date_fixed += 1
        # ⚠️ 2026-11-02 修正（第三个同款口径错位）：
        #   原来只在 adj_cost **为 NaN** 时回填 → 建仓当时按**信号日**写下的
        #   adj_cost（信号日 ≠ 成交日）被原样保留 → 那部分票的比值分子分母
        #   落在不同日期 → 收益失真（日频/周频对账实测差 300~850 元）。
        #   现在：**按成交日无条件重算**，并记 adj_cost_date 便于校验。
        try:
            ts = pd.Timestamp(pos.get("buy_date"))
        except (TypeError, ValueError):
            continue
        v, vd = pp.last_valid_adj(adj_close, code, ts)
        if v is not None:
            prev_v = pos.get("adj_cost")
            pos["adj_cost"] = round(v, 6)
            if vd:
                pos["adj_cost_date"] = vd
            if not _valid_adj(prev_v) or abs(float(prev_v) - v) > 1e-6:
                fixed += 1

    value, _details = pp.total_return_portfolio_value(
        positions, float(paper.get("cash") or 0.0), adj_close, as_of)

    if not dry:
        paper["positions"] = positions
        paper["last_value"] = round(value, 2)
        paper["value_as_of"] = str(as_of.date())
        paper["nav_value_basis"] = "total_return@trade_day"
        # nav_history 的**最后一个点**用新口径重算（该点对应"当前持仓"）
        nav = list(paper.get("nav_history") or [])
        cap0 = float(paper.get("initial_capital") or 1_000_000)
        if nav:
            nav[-1] = {
                "date": nav[-1].get("date"),
                "value": round(value, 2),
                "return_pct": round((value / cap0 - 1.0) * 100, 4) if cap0 else 0.0,
            }
            paper["nav_history"] = nav
        pp.save_holdings(strategy, paper)

    return {"strategy": strategy, "n_pos": len(positions),
            "fixed": fixed, "date_fixed": date_fixed,
            "value": round(value, 2), "value_before": value_before}


def fix_merged_line(adj_close: pd.DataFrame, as_of: pd.Timestamp,
                    dry: bool) -> Dict[str, Any]:
    """多资产合并线：A 腿（总回报口径）+ 袖套腿（前复权价口径）。"""
    merged = pp.load_holdings(MERGED, 1_000_000.0)
    a_paper = pp.load_holdings(A_LEG, 1_000_000.0)
    s_paper = pp.load_holdings(SLEEVE, 1_000_000.0)
    a_positions = a_paper.get("positions") or {}
    s_positions = s_paper.get("positions") or {}

    a_value, _ = pp.total_return_portfolio_value(
        a_positions, float(a_paper.get("cash") or 0.0), adj_close, as_of)

    s_value = None
    # ETF 代码直接取袖套持仓文件（权威：实际持有哪些就取哪些）
    etf_codes = sorted(s_positions.keys())
    try:
        bars = load_etf_bars(list(etf_codes))
        s_value = (float(s_paper.get("cash") or 0.0)
                   + sleeve_positions_value(s_positions, bars, as_of))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 袖套估值失败（不影响 A 腿修复）：{e}", flush=True)

    total = a_value + (s_value or 0.0)
    value_before = merged.get("last_value")
    if not dry and s_value is not None:
        merged["last_value"] = round(total, 2)
        merged["value_as_of"] = str(as_of.date())
        merged["nav_value_basis"] = "total_return@trade_day (A腿) + etf_fq(trade_day) (袖套)"
        nav = list(merged.get("nav_history") or [])
        cap0 = float(merged.get("initial_capital") or 1_000_000)
        if nav:
            nav[-1] = {
                "date": nav[-1].get("date"),
                "value": round(total, 2),
                "return_pct": round((total / cap0 - 1.0) * 100, 4) if cap0 else 0.0,
            }
            merged["nav_history"] = nav
        pp.save_holdings(MERGED, merged)

    return {"strategy": MERGED, "n_pos": len(merged.get("positions") or {}),
            "fixed": 0, "date_fixed": 0,
            "value": round(total, 2), "value_before": value_before}


def main() -> None:
    ap = argparse.ArgumentParser(description="回填 adj_cost 并重算市值")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写盘")
    args = ap.parse_args()

    print("加载行情与后复权价格（较慢，约 2~4 分钟）……", flush=True)
    # 复用 paper_portfolio 已加载好的模块（ds = scripts/daily_signal.py），
    # 不自己再 import 一份 —— 避免两份模块对象导致口径分叉。
    codes = pp.ds.list_universe()
    frames, panel, master = pp.ds.load_data(codes)
    adj_close = pp.load_adjusted_close(codes, master)
    as_of = master[-1]
    print(f"数据末日 = {as_of.date()} | 后复权矩阵 {adj_close.shape}", flush=True)

    rows: List[Dict[str, Any]] = []
    for s in SINGLE_LINES + [A_LEG]:
        r = fix_line(s, adj_close, as_of, args.dry_run)
        rows.append(r)
        if r["n_pos"]:
            print(f"  [{s}] {r['n_pos']} 只 | 回填 adj_cost {r['fixed']} 只 | "
                  f"修正 buy_date {r['date_fixed']} 只 | "
                  f"市值 {r['value_before']} → {r['value']}", flush=True)
    rows.append(fix_merged_line(adj_close, as_of, args.dry_run))
    m = rows[-1]
    print(f"  [{MERGED}] {m['n_pos']} 只 | 市值 {m['value_before']} → {m['value']}",
          flush=True)

    if args.dry_run:
        print("\n[dry-run] 未写盘", flush=True)
        return

    # 重建 portfolio.json（读修正后的 holdings.last_value）
    import json
    pp_path = os.path.join(pp.PAPER_DIR, "portfolio.json")
    existing = {}
    if os.path.exists(pp_path):
        with open(pp_path) as fh:
            existing = json.load(fh) or {}
    merged_lines = dict(existing.get("lines") or {})
    for name in list(pp.STRATEGIES.keys()):
        try:
            merged_lines[name] = pp.rebuild_line_from_files(name, 1_000_000.0)
        except Exception as e:  # noqa: BLE001
            print(f"  重建 {name} 失败（保留旧值）: {e}", flush=True)
    existing["lines"] = merged_lines
    with open(pp_path, "w") as fh:
        json.dump(existing, fh, ensure_ascii=False, indent=1)
    print(f"\n✅ 已写回 portfolio.json（{pp_path}）", flush=True)


if __name__ == "__main__":
    main()
