#!/usr/bin/env python3
"""纸面盘**日频盯市**（Daily Mark-to-Market）—— 方案书正文 §1 方案 A。

为什么做（方案书 §0）
--------------------
纸面盘 7 条线的净值 `nav_history` **只在周更任务跑的时候追加一个点**
→ 一年 52 个点、一个季度 12 个点。**12 个点算不出可信的夏普**
（年化波动要靠 252 个点估）。这不是"曲线不好看"，是**指标没有统计效力**。
→ 目标：每个交易日一个点（约 240 点/年）。

口径纪律（方案书 §2.2，本项目最贵的错误类型是口径漂移）
------------------------------------------------------
- **估值只实现一次**：调 `paper_portfolio.total_return_portfolio_value()`
  （周频调仓与日频盯市**同一份代码**），不复制粘贴、不另立公式。
- **复权同一源**：同一个 `adj_close`（后复权价矩阵）。
- **持仓从 ledger 重放**：周频每 7 天才调一次仓，日频要的是**每天的持仓**
  → 持仓快照由 ledger 逐笔重放得出（ledger 的 date = 成交日，权威）。
- **日期用数据日，不用墙钟日**：TDX 当日日线包 17:30 后才落地
  → 数据没更新到今天就不写今天的点，**绝不用旧价冒充**（那会造出假的 0 收益点
  → 压低波动、稀释夏普）。

落盘：独立文件 `data/paper/<strategy>_nav_daily.json`（**不混进 nav_history**）
  理由见方案书 §2.1：语义不同（周频=调仓快照 / 日频=每日盯市）、
  写者冲突（同一成交日两边各写一次 → 谁后写谁赢）、周更重跑会污染。

用法（backend 目录下、项目 venv）：
    env -u PYTHONPATH .venv-py313/bin/python3 scripts/paper_daily_nav.py
    env -u PYTHONPATH .venv-py313/bin/python3 scripts/paper_daily_nav.py --dry-run
    env -u PYTHONPATH .venv-py313/bin/python3 scripts/paper_daily_nav.py --backfill 60
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

_SCRIPT = os.path.join(BACKEND_ROOT, "scripts", "paper_portfolio.py")
_spec = importlib.util.spec_from_file_location("paper_portfolio_mod", _SCRIPT)
assert _spec and _spec.loader
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

# 日频盯市覆盖的线：6 条单条线。
# multi_asset_final 暂**不纳入**：它的估值是"A腿(总回报口径) + 袖套(前复权口径)"
# 两套口径拼接，且袖套的 ETF 历史价存在数据源差异（台账 §12.22 C1）→
# 混进同一条日频曲线会引入不可解释的差异。它从建仓日起单独跟（见 --sleeve）。
DAILY_LINES = ["22f", "honest_42f", "rsi_71421", "factor_ranking",
               "gbm_voltgt", "honest_42f_neu"]

NAV_BASIS = "total_return@trade_day (paper_portfolio.total_return_portfolio_value)"


def acquire_lock() -> int:
    """取纸面盘互斥锁（与 `paper_portfolio.main()` **同一把**）。

    ## 为什么本脚本也要取锁

    它**只写** `<strategy>_nav_daily.json`（不碰账本）→ 不会损坏账本；
    但它**要读** `holdings` 与 `ledger`（`pp.load_holdings` / `read_ledger`）
    → **读到被并发写的半成品就会产出错误的净值点**，
    而净值点一旦写进 `nav_daily.json` 就是"看起来正常的错数据"。

    ## 为什么 dry-run 也取锁（与 `fix_adj_cost_backfill` 不同）

    那个脚本 dry-run 时**既不读也不写**盘 → 不取锁合理；
    本脚本 dry-run **照样读** holdings/ledger → 同样会读到半成品
    → **两种模式都必须取锁**。

    取锁失败（超时）→ `pp.acquire_run_lock()` 走 `sys.exit(75)`（EX_TEMPFAIL），
    由排期脚本记日志、次日补齐序列（`scripts/daily_nav.sh`）。

    ## 为什么超时只给 120 秒（不是默认的 1800 秒）

    `scripts/daily_nav.sh` **在脚本外面已经等过同一把锁**（最多 40 分钟）。
    脚本内部再等 30 分钟是**双层等待冗余** —— 最坏情况 cron 任务会挂 70 分钟。
    正常路径下 shell 刚探测过锁空闲，本脚本立刻就能拿到；
    真正会等到的只有 TOCTOU 窗口（shell 探测通过后、排期实例恰好抢先），
    那时**宁可今天不算、次日补齐，也不要产出错净值**。
    """
    return pp.acquire_run_lock(timeout_s=120)


# ── 账本读取 ──────────────────────────────────────────────────────────
def read_ledger(strategy: str) -> List[Dict[str, str]]:
    p = pp.ledger_path(strategy)
    if not os.path.exists(p):
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


COMMISSION = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.0005
SLIPPAGE = 0.001


def _num(v: Any) -> Optional[float]:
    """把 ledger 的字符串字段解析成 float；空/非法 → None。"""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def replay_holdings(rows: List[Dict[str, str]],
                    adj_close: Optional[pd.DataFrame],
                    initial_capital: float = 0.0,
                    ) -> Dict[str, Dict[str, Any]]:
    """按 ledger 逐笔重放 → **每个有成交的日期**的持仓快照。

    返回：{成交日: {"positions": {code: {...}}, "cash": float}}

    重放规则（与 paper_portfolio 的记账口径一致）：
      BUY : positions[code] = {shares, cost(每股), buy_date, adj_cost}
            cash ← 该行 `cash_after`（ledger 直接给了，不用自己算）
      SELL: shares 扣减；**每股成本不变**（总成本按股数比例扣减 = 移动平均成本法）；
            shares 归零 → 移除该 code；cash ← 该行 `cash_after`

    ⚠️ 已知假设：卖出的 `cost` 按**股数比例**扣减（每股成本不变）。
    现有账本里卖出案例都是"全清仓后重建"（清仓时该比例假设不影响结果），
    所以本假设**未被现有数据证伪**；若将来出现"部分卖出仍持有"的案例，
    需要用 §3 门禁 a 重新验证（回放重建的持仓必须与真实 holdings 逐位相等）。
    """
    positions: Dict[str, Dict[str, Any]] = {}
    cash = float(initial_capital or 0.0)
    snaps: Dict[str, Dict[str, Any]] = {}

    def _adj_at(date: str, code: str) -> Optional[float]:
        """≤ date 的最后有效后复权价（与 paper_portfolio.last_valid_adj 同口径）。"""
        if adj_close is None:
            return None
        try:
            v, _d = pp.last_valid_adj(adj_close, code, pd.Timestamp(date))
        except (KeyError, TypeError, ValueError):
            return None
        return v

    for r in rows:
        d = str(r.get("date") or "").strip()
        code = str(r.get("code") or "").strip()
        action = str(r.get("action") or "").strip().upper()
        if not d or not code:
            continue
        try:
            sh = float(r.get("shares") or 0)
            px = float(r.get("price") or 0)
        except (TypeError, ValueError):
            continue

        if action == "BUY":
            p = positions.setdefault(code, {
                "shares": 0.0, "cost": 0.0, "buy_date": d, "adj_cost": None})
            p["shares"] += sh
            # 每股成本：多笔买入取**加权平均**（cost 是每股成本，不是总成本）
            new_shares = p["shares"]
            p["cost"] = round((p["cost"] * (new_shares - sh) + px * sh)
                              / new_shares, 3) if new_shares else 0.0
            p["buy_date"] = d
            a = _adj_at(d, code)
            if a is not None:
                p["adj_cost"] = round(a, 6)
            # 现金：与 paper_portfolio._record_buy **完全同口径**
            #   cost = gross + max(5, gross×万3) + gross×千1
            #   gross 优先用**账本 amount 字段**（ledger 记的就是成交额，权威），
            #   没有才用 shares×price 兜底。
            gross = _num(r.get("amount"))
            if gross is None or gross <= 0:
                gross = sh * px
            cash -= gross + max(MIN_COMMISSION, gross * COMMISSION) + gross * SLIPPAGE
        elif action == "SELL":
            p = positions.get(code)
            if p is not None:
                # 卖出所得用**总回报口径**（与 paper_portfolio.run_line 的 sells 一致）：
                #   gross = 每股成本 × 股数 × 后复权比值
                #   → 这就是"总回报口径账本"的定义：账户现金与持仓估值同源
                ratio = 1.0
                a_cost = p.get("adj_cost")
                a_now = _adj_at(d, code)
                if a_cost and a_now:
                    try:
                        if np.isfinite(float(a_cost)) and float(a_cost) > 0:
                            ratio = float(a_now) / float(a_cost)
                    except (TypeError, ValueError):
                        ratio = 1.0
                # 卖出的 gross 用**账本 amount 字段**（= 总回报口径成交额，
                #   ledger 记的就是它，权威）；没有才按 cost×股数×比值自算。
                gross = _num(r.get("amount"))
                if gross is None or gross <= 0:
                    gross = float(p.get("cost") or 0.0) * sh * ratio
                cash += (gross - max(MIN_COMMISSION, gross * COMMISSION)
                         - gross * STAMP_TAX - gross * SLIPPAGE)
                p["shares"] = max(0.0, float(p["shares"]) - sh)
                # 每股成本不变（总成本随股数同比例减少）
                if p["shares"] <= 0:
                    positions.pop(code, None)
        else:
            continue

        snaps[d] = {"positions": {k: dict(v) for k, v in positions.items()},
                    "cash": cash}

    return snaps


def trading_days(master: pd.DatetimeIndex, start: pd.Timestamp,
                 end: pd.Timestamp) -> List[pd.Timestamp]:
    days = [d for d in master if start <= d <= end]
    return sorted(days)


def nav_daily_path(strategy: str) -> str:
    return os.path.join(pp.PAPER_DIR, f"{strategy}_nav_daily.json")


def load_nav_daily(strategy: str) -> Dict[str, Any]:
    p = nav_daily_path(strategy)
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh) or {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def write_nav_daily(strategy: str, points: List[Dict[str, Any]]) -> None:
    os.makedirs(pp.PAPER_DIR, exist_ok=True)
    payload = {
        "strategy": strategy,
        "basis": NAV_BASIS,
        "initial_capital": None,   # 由调用方填
        "updated_at": None,        # 由调用方填
        "points": points,
    }
    with open(nav_daily_path(strategy), "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)


def build_line_points(strategy: str, adj_close: Optional[pd.DataFrame],
                       master: pd.DatetimeIndex,
                       backfill_days: Optional[int] = None,
                       ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """算出一条线的日频序列 + 诊断信息。"""
    paper = pp.load_holdings(strategy, 1_000_000.0)
    cap0 = float(paper.get("initial_capital") or 1_000_000.0)
    rows = read_ledger(strategy)
    snaps = replay_holdings(rows, adj_close, initial_capital=cap0)

    diag: Dict[str, Any] = {"strategy": strategy, "n_ledger_rows": len(rows),
                            "n_snapshots": len(snaps)}

    if not snaps:
        diag["status"] = "no_ledger（从未成交 → 无日频点）"
        return [], diag

    # 从**首次建仓日**开始（之前是"未建仓"，不该出现 0 收益点）
    first_trade = min(snaps.keys())
    start = pd.Timestamp(first_trade)
    end = master[-1]
    days = trading_days(master, start, end)
    if backfill_days:
        days = days[-int(backfill_days):]

    points: List[Dict[str, Any]] = []
    cur_pos: Dict[str, Any] = {}
    cur_cash = 0.0
    snap_dates = sorted(snaps.keys())
    si = 0
    for d in days:
        ds_str = str(d.date())
        while si < len(snap_dates) and snap_dates[si] <= ds_str:
            snap = snaps[snap_dates[si]]
            cur_pos = {k: dict(v) for k, v in (snap["positions"] or {}).items()}
            cur_cash = float(snap["cash"] or 0.0)
            si += 1
        if not cur_pos and cur_cash <= 0:
            continue
        value, details = pp.total_return_portfolio_value(
            cur_pos, cur_cash, adj_close, d)
        stale = [x["code"] for x in details if x.get("stale_price")]
        points.append({
            "date": ds_str,
            "value": round(value, 2),
            "return_pct": round((value / cap0 - 1.0) * 100, 4) if cap0 else 0.0,
            "n_positions": len(cur_pos),
            "stale_price": stale or None,
        })

    # ── 门禁 a：日频末点必须与周频 nav_history 末点**逐位一致** ──────────
    # 两者本该是同一口径（同一估值函数、同一持仓、同一 as_of）。
    # 不一致 = 两套口径漂移 → 直接报出来，不许悄悄通过。
    nav = list(paper.get("nav_history") or [])
    if nav and points:
        last_nav = nav[-1]
        if str(last_nav.get("date")) == points[-1]["date"]:
            diff = abs(points[-1]["value"] - float(last_nav.get("value") or 0))
            diag["nav_consistency_date"] = points[-1]["date"]
            diag["nav_consistency_diff"] = round(diff, 2)
            # 阈值 0.05 元：逐票 round(value, 2) 的舍入累积（实测最大 0.02 元）
            diag["nav_consistency_ok"] = bool(diff < 0.05)
        else:
            diag["nav_consistency_ok"] = None
            diag["nav_consistency_note"] = (
                f"周频末点日期 {last_nav.get('date')} ≠ 日频末点 {points[-1]['date']}")

    diag.update({"first_trade": first_trade, "last_day": str(end.date()),
                 "n_points": len(points),
                 "n_stale_days": sum(1 for p in points if p.get("stale_price"))})
    return points, diag


def merge_points(existing: List[Dict[str, Any]],
                 new_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 date upsert（同日覆盖，不追加）+ 按日期排序 —— 幂等的基础。"""
    by_date: Dict[str, Dict[str, Any]] = {}
    for p in existing:
        by_date[str(p.get("date"))] = p
    for p in new_points:
        by_date[str(p.get("date"))] = p
    return [by_date[k] for k in sorted(by_date.keys())]


def main() -> None:
    ap = argparse.ArgumentParser(description="纸面盘日频盯市")
    ap.add_argument("--dry-run", action="store_true", help="只算不写盘")
    ap.add_argument("--backfill", type=int, default=None,
                    help="只保留最近 N 个交易日（默认全量）")
    ap.add_argument("--strategies", nargs="*", default=DAILY_LINES)
    args = ap.parse_args()

    # ── 互斥锁（§12.52）───────────────────────────────────────────
    # 必须在**加载行情之前**取锁 —— 那 2~4 分钟里排期实例完全可能已经开跑，
    # 而本脚本读 holdings/ledger 算净值，读到半成品就是错的净值点。
    lock_fd = acquire_lock()
    print(f"✅ 已取得纸面盘互斥锁（fd={lock_fd}）", flush=True)
    # lock_fd 必须保持引用直到进程结束（fcntl.flock 随进程退出自动释放）

    print("加载行情与后复权价格（较慢，约 2~4 分钟）……", flush=True)
    codes = pp.ds.list_universe()
    frames, panel, master = pp.ds.load_data(codes)
    adj_close = pp.load_adjusted_close(codes, master)
    print(f"数据末日 = {master[-1].date()}", flush=True)

    for s in args.strategies:
        points, diag = build_line_points(s, adj_close, master, args.backfill)
        if not points:
            print(f"  [{s}] {diag.get('status', '无点')}", flush=True)
            continue
        old = load_nav_daily(s)
        merged = merge_points(list(old.get("points") or []), points)
        if not args.dry_run:
            cap0 = float((pp.load_holdings(s, 1_000_000.0) or {}).get(
                "initial_capital") or 1_000_000.0)
            payload = {
                "strategy": s,
                "basis": NAV_BASIS,
                "initial_capital": cap0,
                "updated_at": pd.Timestamp.now().isoformat(),
                "points": merged,
            }
            with open(nav_daily_path(s), "w") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
        cons = diag.get("nav_consistency_ok")
        cons_txt = ("一致 ✓" if cons is True else
                    ("**不一致 ✗**" if cons is False else "无法比对"))
        if cons is False:
            cons_txt += f"（差 {diag.get('nav_consistency_diff')}）"
        print(f"  [{s}] 点数 {len(merged)}（新增 {len(points)}，"
              f"区间 {merged[0]['date']} ~ {merged[-1]['date']}）| "
              f"停牌/缺价日 {diag.get('n_stale_days', 0)} | "
              f"末日市值 ¥{merged[-1]['value']:,.0f} "
              f"（{(merged[-1]['return_pct'] or 0):+.2f}%）| 与周频末点{cons_txt}",
              flush=True)

    if args.dry_run:
        print("\n[dry-run] 未写盘", flush=True)


if __name__ == "__main__":
    main()
