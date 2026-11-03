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

历史区间回溯（方案 D，2026-11-03 新增）：
    env -u PYTHONPATH .venv-py313/bin/python3 scripts/paper_daily_nav.py \\
        --as-of 2026-10-26 --dry-run
    env -u PYTHONPATH .venv-py313/bin/python3 scripts/paper_daily_nav.py \\
        --range 2026-10-16:2026-10-30 --dry-run

⚠️ **默认行为完全不变**（不带 --as-of/--range 时，逐位等同改造前）。
`--as-of` / `--range` 只**收窄**要产出的净值点区间，
持仓重放与现金重算统一走 `scripts/ledger_replay.py`（**单一实现**，
口径纪律：同一件事只写一遍，见方案书 §2.2）。
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

# 方案 D：持仓重放 / 现金重算的**单一实现**放在 ledger_replay.py。
# 为什么不让本脚本自己再写一遍：项目最贵的错误类型是「口径漂移」
# （同一件事两份实现，迟早分叉）。§12.53 记录过一次真实教训。
_LR_SCRIPT = os.path.join(BACKEND_ROOT, "scripts", "ledger_replay.py")
_lr_spec = importlib.util.spec_from_file_location("ledger_replay_mod", _LR_SCRIPT)
assert _lr_spec and _lr_spec.loader
lr = importlib.util.module_from_spec(_lr_spec)
# ⚠️ 必须**先注册进 sys.modules 再 exec_module**：
# `ledger_replay.LedgerRow` 是 `@dataclass`，且模块里有 `from __future__ import
# annotations` → 字段注解是**字符串** → dataclasses 装饰器要回 `sys.modules`
# 里找类所在的模块（`sys.modules.get(cls.__module__).__dict__`）来解析注解。
# 不注册 → `sys.modules.get(...)` 返回 None → `AttributeError: 'NoneType'
# object has no attribute '__dict__'`（pytest 收集期直接报错）。
sys.modules["ledger_replay_mod"] = lr
_lr_spec.loader.exec_module(lr)

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

    ## ⚠️ 本函数是 `scripts/ledger_replay.py` 的**薄封装**（方案 D，2026-11-03）

    重放逻辑（持仓 + 现金）**只有一份实现** → `ledger_replay.replay_rows`。
    为什么不给本脚本另写一份：项目最贵的错误类型是**口径漂移**
    （同一件事两份实现，迟早分叉；§12.53 记录过真实教训）。

    本封装保留 **移动平均成本**语义（`semantics="moving_average"`），
    与本脚本改造前的行为**逐位一致**（既有门禁测试 `test_paper_daily_nav.py`
    依赖它）；方案 D 的历史回溯路径用**全清仓**语义（与生产 `run_line` 一致），
    两者在当前账本上结果相同（实测 6 条线 0 部分卖出、0 重复买入）。

    重放规则（与 paper_portfolio 的记账口径一致）：
      BUY : positions[code] = {shares, cost(每股), buy_date, adj_cost}
            cash ← 按生产费率从 `amount` 重算（**不读 cash_after**，§12.53）
      SELL: shares 扣减；**每股成本不变**（总成本按股数比例扣减 = 移动平均成本法）；
            shares 归零 → 移除该 code；cash ← 按生产费率从 `amount` 重算

    ⚠️ 已知假设：卖出的 `cost` 按**股数比例**扣减（每股成本不变）。
    现有账本里卖出案例都是"全清仓后重建"（清仓时该比例假设不影响结果），
    所以本假设**未被现有数据证伪**；若将来出现"部分卖出仍持有"的案例，
    需要用 §3 门禁 a 重新验证（回放重建的持仓必须与真实 holdings 逐位相等）。
    """
    parsed: List[Any] = []
    for r in rows:
        d = str(r.get("date") or "").strip()
        code = str(r.get("code") or "").strip()
        action = str(r.get("action") or "").strip().upper()
        if not d or not code:
            continue
        try:
            float(r.get("shares") or 0)
            float(r.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if action not in ("BUY", "SELL"):
            continue
        parsed.append(lr.LedgerRow(
            date=d, action=action, code=code,
            name=str(r.get("name") or "").strip(),
            price=_num(r.get("price")), shares=_num(r.get("shares")),
            amount=_num(r.get("amount")),
            commission=_num(r.get("commission")),
            stamp_tax=_num(r.get("stamp_tax")),
            slippage=_num(r.get("slippage")),
            score=_num(r.get("score")),
            cost_amount=_num(r.get("cost_amount")),
            realized_pnl=_num(r.get("realized_pnl")),
            cash_after=_num(r.get("cash_after")),
            portfolio_value=_num(r.get("portfolio_value")),
        ))

    snaps: Dict[str, Dict[str, Any]] = {}
    if parsed:
        snapshots, _diag = lr.replay_rows(
            parsed, initial_capital=initial_capital,
            adj_close=adj_close,
            last_valid_fn=pp.last_valid_adj,
            semantics="moving_average",
            cash_model="rate",
        )
        for snap in snapshots:
            snaps[str(snap["date"])[:10]] = {
                "positions": {k: dict(v) for k, v in (snap["positions"] or {}).items()},
                "cash": float(snap["cash"]),
            }
    return snaps


def trading_days(master: pd.DatetimeIndex, start: pd.Timestamp,
                 end: pd.Timestamp) -> List[pd.Timestamp]:
    days = [d for d in master if start <= d <= end]
    return sorted(days)


def _nearest_trading_day(master: pd.DatetimeIndex,
                          ts: pd.Timestamp) -> Optional[str]:
    """离 `ts` 最近的**有交易日**（用于报错提示，不用于静默取整）。"""
    if master is None or len(master) == 0:
        return None
    before = [d for d in master if d <= ts]
    after = [d for d in master if d > ts]
    if before:
        return str(before[-1].date())
    if after:
        return str(after[0].date())
    return None


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
                       as_of: Optional[str] = None,
                       range_: Optional[str] = None,
                       ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """算出一条线的日频序列 + 诊断信息。

    ## `as_of` / `range_`（方案 D：历史区间回溯）

    - 两者都为空 → **默认行为**（首次建仓日 ~ 数据末日，逐位等同改造前）
    - `as_of="YYYY-MM-DD"` → 只产出**该交易日**那一个点
    - `range_="START:END"` → 只产出 **[START, END] 区间**内的点
      （两个端点都要是交易日；非交易日 → 报错，不静默取整）

    ⚠️ 区间收窄只影响**产出哪些点**，不影响**重放口径**：
    持仓与现金仍然从**账本开头**开始逐笔重放
    （区间起点之前的持仓必须算对，否则区间内的市值就是错的）。
    """
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

    # ── 区间收窄（方案 D）───────────────────────────────────────────
    win_start: Optional[pd.Timestamp] = None
    win_end: Optional[pd.Timestamp] = None
    if range_:
        parts = str(range_).split(":")
        if len(parts) != 2 or not all(p.strip() for p in parts):
            raise ValueError(f"--range 格式应为 START:END，收到 {range_!r}")
        win_start = pd.Timestamp(parts[0].strip())
        win_end = pd.Timestamp(parts[1].strip())
        if win_start > win_end:
            raise ValueError(f"--range 起点晚于终点：{range_}")
        for label, ts in (("START", win_start), ("END", win_end)):
            if ts not in master:
                raise ValueError(
                    f"--range {label}={ts.date()} 不是交易日"
                    f"（最近的有交易日：{_nearest_trading_day(master, ts)}）")
    if as_of:
        ts = pd.Timestamp(str(as_of).strip()[:10])
        if ts not in master:
            raise ValueError(
                f"--as-of {ts.date()} 不是交易日"
                f"（最近的有交易日：{_nearest_trading_day(master, ts)}）")
        # `--as-of` 单独给 → 窗口 = [as_of, as_of]（**只产出那一个点**）
        # `--as-of` 与 `--range` 同给 → 保留 range 的起点，终点取**更早**的那个
        if win_start is None:
            win_start = ts
        win_end = ts if win_end is None else min(win_end, ts)
    diag["window_start"] = str(win_start.date()) if win_start is not None else None
    diag["window_end"] = str(win_end.date()) if win_end is not None else None

    days = trading_days(master, start, end)
    if backfill_days:
        days = days[-int(backfill_days):]
    if win_start is not None:
        days = [d for d in days if d >= win_start]
    if win_end is not None:
        days = [d for d in days if d <= win_end]
    if win_start is not None and win_end is not None and win_start > win_end:
        # 交集为空（如 --as-of 早于 --range 起点）→ 不产出点，但**明确记录**
        diag["status"] = (f"区间为空（{win_start.date()} > {win_end.date()}）"
                          f" —— --as-of 与 --range 没有交集")
        return [], diag

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

    # ── 门禁 a：日频必须与周频 nav_history 末点**逐位一致** ──────────────
    # 两者本该是同一口径（同一估值函数、同一持仓、同一 as_of）。
    # 不一致 = 两套口径漂移 → 直接报出来，不许悄悄通过。
    #
    # ⚠️ 比对对象的选取（2026-11-03 修正）：
    #   原来只在「日频末点日期 == 周频末点日期」时才比对。但数据末日常常**比周频末日新**
    #   （当天盘中数据未出 → 末点整屏 stale_price），于是门禁永远落到「无法比对」分支
    #   → 门禁形同虚设（实测 2026-11-03：日频末点 11-03、周频末点 11-02 → 无法比对）。
    #   现在改成：**按周频末点日期去日频点里取对应点**来比对；
    #   该点带滞后价 → 仍判「无法比对」（值不可信）；日频里没有该日期 → 也判「无法比对」。
    nav = list(paper.get("nav_history") or [])
    if nav and points:
        last_nav = nav[-1]
        nav_date = str(last_nav.get("date"))
        pt = next((p for p in reversed(points)
                   if str(p.get("date")) == nav_date), None)
        if pt is None:
            diag["nav_consistency_ok"] = None
            diag["nav_consistency_note"] = (
                f"日频点里没有 {nav_date}（周频末点日期）→ 无法比对")
        elif pt.get("stale_price"):
            diag["nav_consistency_ok"] = None
            diag["nav_consistency_note"] = (
                f"{nav_date} 的日频点带滞后价（{len(pt['stale_price'])} 只）"
                f"→ 值不可信，不作比对")
        else:
            diff = abs(float(pt["value"]) - float(last_nav.get("value") or 0))
            diag["nav_consistency_date"] = nav_date
            diag["nav_consistency_diff"] = round(diff, 2)
            # 阈值 0.05 元：逐票 round(value, 2) 的舍入累积（实测最大 0.02 元）
            diag["nav_consistency_ok"] = bool(diff < 0.05)

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
    ap.add_argument("--as-of", default=None,
                    help="方案 D：只产出该交易日那一个点（如 2026-10-26）")
    ap.add_argument("--range", dest="range_", default=None,
                    help="方案 D：只产出该区间的点，格式 START:END（两端都要是交易日）")
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

    gate_failures: List[str] = []
    for s in args.strategies:
        points, diag = build_line_points(s, adj_close, master, args.backfill,
                                         as_of=args.as_of,
                                         range_=args.range_)
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
        elif cons is None and diag.get("nav_consistency_note"):
            cons_txt += f"（{diag['nav_consistency_note']}）"
        # ⚠️ 打印口径：窗口模式（--as-of / --range）下要报**本次产出点**的区间与末值，
        #    不能用 merged（合并后的全量）—— 否则 --range 2026-10-20:2026-10-30
        #    会打印出「区间 ~ 2026-11-02、末日市值 ¥978,983」，而它实际只产到 10-30，
        #    与下面的「与周频末点」比对口径也对不上（比的是 points 末日）。
        _last = points[-1]
        print(f"  [{s}] 点数 {len(merged)}（新增 {len(points)}，"
              f"本次区间 {points[0]['date']} ~ {_last['date']}）| "
              f"停牌/缺价日 {diag.get('n_stale_days', 0)} | "
              f"末日市值 ¥{_last['value']:,.0f} "
              f"（{(_last['return_pct'] or 0):+.2f}%）| 与周频末点{cons_txt}",
              flush=True)
        # 门禁 a 失败（日频末点与周频 nav_history 末点不一致）必须让**退出码**变非 0，
        #    否则 cron 每天跑完都是 0，曲线与周频不一致没人会发现（静默失效）。
        if cons is False:
            gate_failures.append(s)

    if gate_failures:
        print(f"\n❌ 门禁 a 失败（日频末点与周频 nav_history 末点不一致）："
              f"{', '.join(gate_failures)}", flush=True)
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] 未写盘", flush=True)


if __name__ == "__main__":
    main()
