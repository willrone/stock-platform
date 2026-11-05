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

覆盖范围（2026-11-05 起，自用化 A 项）
------------------------------------------------
9 条产物：6 条 A 股单线 + 定稿线的三条（A 腿 / ETF 袖套腿 / 母线）。

    22f  honest_42f  rsi_71421  factor_ranking  gbm_voltgt  honest_42f_neu
    multi_asset_final_a        ← 与上面 6 条**同一条代码路径**（后复权总回报）
    multi_asset_final_sleeve   ← ETF 前复权价×股数 + ETF 费率重放现金
    multi_asset_final          ← 当日两腿相加（静态 50/50，腿间不再平衡）

三道门禁（任一失败 → 退出码非 0，cron 日志立刻可见）：
  a  日频末点 == 周频 `nav_history` 末点（逐票 2dp 舍入容差 0.05）
  b  ledger 重放出的现金 == holdings 的 cash（袖套腿专用，容差 0.05）
  c  行情新鲜度：最新点用的价格滞后 > `SLEEVE_MAX_STALE_DAYS` 就报红
     （ETF 管道没有自己的排期 → 由 `scripts/daily_nav.sh` 在盯市前刷新）

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

# 袖套腿的估值/行情实现就在 app 层（生产同一份），直接复用，不另立公式。
from app.services.trading import multi_asset_service as mas  # noqa: E402
# 零技巧基准（自用 KPI 对照组）的读取与锚定，口径定义见 benchmark_service 文档串。
from app.services.trading import benchmark_service as bsvc  # noqa: E402

# 日频盯市覆盖的线。
#
# 2026-11-05 变更（自用化 A 项）：**定稿线补上日频盯市**。
# 原来的注释写的是「multi_asset_final 暂不纳入」，理由（台账 §12.22 C1）是成立的：
# 两条腿的估值口径不同 —— A 股腿 = 后复权**总回报**（`cost×比值`），
# ETF 袖套 = **前复权价 × 股数**（前值随分红重锚，不能再用比值）——
# 硬拼成一条曲线确实会带来不可解释的差异。
# 但"口径不同"的正确处理是**分开跟、再在账户层相加**，不是不跟：
#   · `multi_asset_final_a`      → 与 6 条单线**完全同一条代码路径**（cap 50 万）
#   · `multi_asset_final_sleeve` → ETF 价 + ETF 费率重放现金（cap 50 万）
#   · `multi_asset_final`        → 当日两腿市值**相加**（cap 100 万）
# 这与周频写 `nav_history` 的口径一致（实测母线末点
# 999,025.96 = 499,225.89 + 499,800.07）→ 门禁 a 依然可比对、依然能抓漂移。
A_SHARE_LINES = ["22f", "honest_42f", "rsi_71421", "factor_ranking",
                 "gbm_voltgt", "honest_42f_neu"]
MULTI_ASSET_A = pp.MULTI_ASSET_A_KEY
MULTI_ASSET_SLEEVE = pp.MULTI_ASSET_SLEEVE_KEY
MULTI_ASSET_PARENT = "multi_asset_final"
#: 走 A 股重放路径的线（A 腿与单线共用 `build_line_points`）
DAILY_LINES = A_SHARE_LINES + [MULTI_ASSET_A]
#: 袖套/母线不是 A 股重放路径，在 `main()` 里单独处理
LEG_LINES = (MULTI_ASSET_SLEEVE, MULTI_ASSET_PARENT)

NAV_BASIS = "total_return@trade_day (paper_portfolio.total_return_portfolio_value)"
#: 袖套口径（与研究侧一致：前复权价×股数，**不做二次复权**）
SLEEVE_NAV_BASIS = ("shares×etf_close@trade_day "
                    "(multi_asset_service.sleeve_positions_value) "
                    "+ cash (ledger_replay cash_model=etf_rate)")
#: 母线口径：账户层相加，腿间**不再平衡**（与 paper_portfolio 的静态 50/50 一致）
PARENT_NAV_BASIS = (f"a[{SLEEVE_NAV_BASIS.split('(')[0]}] + sleeve[前复权价×股数] "
                    "@trade_day (静态 50/50，腿间不再平衡)")
#: 价格陈旧容忍度 —— 与生产成交口径同源（`MAX_STALE_TRADE_DAYS`）
#: 超过它就是用旧价冒充当日价（铁律：绝不用旧价冒充）
SLEEVE_MAX_STALE_DAYS = mas.MAX_STALE_TRADE_DAYS


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


def write_nav_daily(strategy: str, points: List[Dict[str, Any]],
                    initial_capital: Optional[float] = None,
                    basis: Optional[str] = None) -> None:
    """写 `<strategy>_nav_daily.json`（**唯一的落盘口**）。

    `basis` 必须与产出这些点的口径一致（A 股腿 / 袖套腿 / 母线各不相同），
    口径写进文件本身 —— 读的人不用猜这条曲线是怎么算出来的。
    """
    os.makedirs(pp.PAPER_DIR, exist_ok=True)
    payload = {
        "strategy": strategy,
        "basis": basis or NAV_BASIS,
        "initial_capital": initial_capital,   # 由调用方填
        "updated_at": pd.Timestamp.now().isoformat(),
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
    #    窗口解析抽到 `_resolve_window`（2026-11-05）：A 股腿 / 袖套腿 / 母线
    #    共用同一份，否则「同一次运行里三条线各自理解 --as-of」就是口径漂移。
    win_start, win_end = _resolve_window(master, range_, as_of)
    diag["window_start"] = str(win_start.date()) if win_start is not None else None
    diag["window_end"] = str(win_end.date()) if win_end is not None else None

    days = _filter_days(trading_days(master, start, end),
                        backfill_days, win_start, win_end)
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
    #    实现抽到 `_nav_consistency_check`（2026-11-05）：A 股腿 / 袖套腿 / 母线
    #    共用同一份比对逻辑（同口径、同阈值、同「无法比对」判据），
    #    否则「三条线各自解释一致」本身就是新的口径漂移。
    diag.update(_nav_consistency_check(points, paper.get("nav_history")))

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


# ── 共用口径助手（2026-11-05 抽出，避免袖套/母线另立一份实现）──────────
def _resolve_window(master: pd.DatetimeIndex,
                    range_: Optional[str] = None,
                    as_of: Optional[str] = None,
                    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """把 `--range` / `--as-of` 解析成 `(win_start, win_end)`（方案 D）。

    A 股腿、袖套腿、母线**共用这一份**窗口解析：
    否则「同一次运行里三条线各自理解 --as-of」就是口径漂移。
    """
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
    return win_start, win_end


def _filter_days(days: List[pd.Timestamp],
                 backfill_days: Optional[int],
                 win_start: Optional[pd.Timestamp],
                 win_end: Optional[pd.Timestamp]) -> List[pd.Timestamp]:
    """交易日序列 → 本次要产出的交易日（`--backfill` / 窗口收窄）。"""
    out = days
    if backfill_days:
        out = out[-int(backfill_days):]
    if win_start is not None:
        out = [d for d in out if d >= win_start]
    if win_end is not None:
        out = [d for d in out if d <= win_end]
    return out


def _nav_consistency_check(points: List[Dict[str, Any]],
                           nav_history: Any) -> Dict[str, Any]:
    """门禁 a：日频点必须与周频 `nav_history` 末点**逐位一致**。

    两者本该是同一口径（同一估值函数、同一持仓、同一 as_of）。
    不一致 = 两套口径漂移 → 直接报出来，不许悄悄通过。

    ⚠️ 比对对象的选取（2026-11-03 修正）：
      原来只在「日频末点日期 == 周频末点日期」时才比对。但数据末日常常**比周频末日新**
      （当天盘中数据未出 → 末点整屏 stale_price），于是门禁永远落到「无法比对」分支
      → 门禁形同虚设（实测 2026-11-03：日频末点 11-03、周频末点 11-02 → 无法比对）。
      现在改成：**按周频末点日期去日频点里取对应点**来比对；
      该点带滞后价 → 仍判「无法比对」（值不可信）；日频里没有该日期 → 也判「无法比对」。
    """
    nav = list(nav_history or [])
    if not nav or not points:
        return {"nav_consistency_ok": None,
                "nav_consistency_note": "无周频 nav_history 或无日频点 → 无法比对"}
    last_nav = nav[-1]
    nav_date = str(last_nav.get("date"))
    pt = next((p for p in reversed(points)
               if str(p.get("date")) == nav_date), None)
    if pt is None:
        return {"nav_consistency_ok": None,
                "nav_consistency_note":
                    f"日频点里没有 {nav_date}（周频末点日期）→ 无法比对"}
    if pt.get("stale_price"):
        return {"nav_consistency_ok": None,
                "nav_consistency_note":
                    f"{nav_date} 的日频点带滞后价（{len(pt['stale_price'])} 只）"
                    f"→ 值不可信，不作比对"}
    diff = abs(float(pt["value"]) - float(last_nav.get("value") or 0))
    # 阈值 0.05 元：逐票 round(value, 2) 的舍入累积（实测最大 0.02 元）
    return {"nav_consistency_date": nav_date,
            "nav_consistency_diff": round(diff, 2),
            "nav_consistency_ok": bool(diff < 0.05)}


def _consistency_text(diag: Dict[str, Any]) -> str:
    """把门禁 a 的结果渲染成一行可读文本（三条线共用同一措辞）。"""
    cons = diag.get("nav_consistency_ok")
    txt = ("一致 ✓" if cons is True else
           ("**不一致 ✗**" if cons is False else "无法比对"))
    if cons is False:
        txt += f"（差 {diag.get('nav_consistency_diff')}）"
    elif cons is None and diag.get("nav_consistency_note"):
        txt += f"（{diag['nav_consistency_note']}）"
    return txt


# ── ETF 袖套腿 ──────────────────────────────────────────────────────────
#: 母线 100 万 × 静态 50%（`paper_portfolio` 的静态划分，腿间不再平衡）
LEG_CAPITAL = 500_000.0


def build_sleeve_points(master: pd.DatetimeIndex,
                        backfill_days: Optional[int] = None,
                        as_of: Optional[str] = None,
                        range_: Optional[str] = None,
                        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """ETF 袖套腿的日频盯市点（2026-11-05 自用化 A 项）。

    口径（与研究侧/生产侧同源，不另立公式）：

      市值 = 现金 + Σ(股数 × **不晚于当日的最后一条** ETF 收盘价)

      - 估值用 `multi_asset_service.sleeve_positions_value`（app 层生产实现）：
        ETF 行情是**前复权**序列（前值随分红重锚）→ 必须「股数×当前价」，
        **不能**用 A 股腿的「买入成本×后复权比值」。
      - 现金：`ledger_replay.replay_rows(cash_model="etf_rate")`
        （袖套无印花税、滑点已在成交价内、无 5 元最低佣金；用默认 `rate`
        会把袖套现金算成 **负数**，实测残差 −499.84）。
      - 持仓语义 `moving_average`：**与生产 `rebalance_sleeve` 一致** ——
        袖套调仓是「削减超配」的**部分卖出**（`shares − sell_shares`，每股成本不变），
        A 股腿才是「全清仓」。两条腿的语义**不同**，不能共用 `full_liquidation`。
      - 价格缺口：允许前向填充（ETF 行情文件存在成段缺口，精确匹配会把持仓
        当天估成 0 → 净值假暴跌 ~98%，`EtfBars` 的注释记录了这次实踩），
        但陈旧度**必须显式记账**：每个点写 `stale_price` + `bar_lag_days`，
        最新点的陈旧交易日数 > `SLEEVE_MAX_STALE_DAYS` → **门禁失败**（不静默放行）。
    """
    strategy = MULTI_ASSET_SLEEVE
    paper = pp.load_holdings(strategy, LEG_CAPITAL)
    cap0 = float(paper.get("initial_capital") or LEG_CAPITAL)
    rows, parse_diag = lr.parse_ledger(pp.ledger_path(strategy))
    diag: Dict[str, Any] = {
        "strategy": strategy,
        "initial_capital": cap0,
        "n_ledger_rows": len(rows),
        "ledger_contracts": sorted({r.contract for r in rows}),
        "ledger_parse": {k: v for k, v in parse_diag.items() if k != "rows"},
    }
    if not rows:
        diag["status"] = "no_ledger（袖套从未成交 → 无日频点）"
        return [], diag

    # 持仓/现金重放（单一实现：ledger_replay）
    snaps_list, rdiag = lr.replay_rows(
        rows, initial_capital=cap0,
        semantics="moving_average",      # ← 袖套是部分卖出，与 A 股腿不同
        cash_model="etf_rate",
    )
    snaps: Dict[str, Dict[str, Any]] = {
        str(s["date"])[:10]: {
            "positions": {k: dict(v) for k, v in (s["positions"] or {}).items()},
            "cash": float(s["cash"]),
        } for s in snaps_list}
    diag["n_snapshots"] = len(snaps)
    diag["replay_diag"] = {k: rdiag.get(k) for k in ("n_buy", "n_sell",
                                                     "cash_unavailable_rows")}

    # ── 门禁 b：重放现金必须与 holdings 的 cash 对得上（≤0.05）────────
    residual = (float(snaps_list[-1]["cash"]) - float(paper.get("cash") or 0.0)
                ) if snaps_list else None
    diag["cash_replay"] = (round(float(snaps_list[-1]["cash"]), 4)
                           if snaps_list else None)
    diag["cash_holdings"] = float(paper.get("cash") or 0.0)
    diag["cash_residual"] = round(residual, 4) if residual is not None else None
    diag["cash_residual_ok"] = (bool(abs(residual) <= 0.05)
                                if residual is not None else None)

    # ── ETF 行情：缺文件直接硬失败，不静默降级（load_etf_bars 的要求）──
    codes = sorted({c for s in snaps.values() for c in s["positions"]})
    missing_files = mas.missing_codes(codes)
    if missing_files:
        diag["status"] = (f"❌ 缺 ETF 行情文件：{missing_files} "
                          "→ 先跑 scripts/etf_data_pipeline.py --codes <缺的> 再盯市")
        return [], diag
    bars = mas.load_etf_bars(codes)
    diag["etf_codes"] = codes
    # 每只 ETF 最新的行情日期（盯市前先看清管道到哪一天）
    bar_last: Dict[str, Optional[str]] = {}
    for c in codes:
        bar = bars.get(c, master[-1])
        bar_last[c] = str(bar["last_date"]) if bar else None
    diag["etf_bar_last"] = bar_last

    first_trade = min(snaps.keys())
    start = pd.Timestamp(first_trade)
    end = master[-1]
    win_start, win_end = _resolve_window(master, range_, as_of)
    diag["window_start"] = str(win_start.date()) if win_start is not None else None
    diag["window_end"] = str(win_end.date()) if win_end is not None else None

    days = _filter_days(trading_days(master, start, end),
                        backfill_days, win_start, win_end)
    if win_start is not None and win_end is not None and win_start > win_end:
        diag["status"] = (f"区间为空（{win_start.date()} > {win_end.date()}）"
                          " —— --as-of 与 --range 没有交集")
        return [], diag

    day_index = {str(d.date()): i for i, d in enumerate(master)}
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
        # 估值：袖套生产实现（股数×前复权收盘价）
        value = cur_cash + mas.sleeve_positions_value(cur_pos, bars, d)
        stale_codes: List[str] = []
        max_lag = 0
        for code in cur_pos:
            bar = bars.get(code, d)
            if bar is None:
                # 该日**早于**这只 ETF 的第一条行情 → 无价可估（必须记账，
                # 不能让 sleeve_positions_value 静默把它跳过）
                stale_codes.append(code)
                max_lag = max(max_lag, 999)
                continue
            if bar.get("stale"):
                stale_codes.append(code)
                lag = (day_index.get(ds_str, 0)
                       - day_index.get(str(bar.get("last_date")), 0))
                max_lag = max(max_lag, max(lag, 1))
        points.append({
            "date": ds_str,
            "value": round(value, 2),
            "return_pct": round((value / cap0 - 1.0) * 100, 4) if cap0 else 0.0,
            "n_positions": len(cur_pos),
            "stale_price": stale_codes or None,
            "bar_lag_days": max_lag,
        })

    diag["first_trade"] = first_trade
    diag["last_day"] = str(end.date())
    diag["n_points"] = len(points)
    diag["n_stale_days"] = sum(1 for p in points if p.get("stale_price"))

    # ── 门禁 c：价格新鲜度（ETF 管道没到货就不许当成今天的净值）───────
    last_pt = points[-1] if points else None
    diag["bar_lag_days_last"] = int((last_pt or {}).get("bar_lag_days") or 0)
    diag["bar_freshness_ok"] = (
        bool(diag["bar_lag_days_last"] <= SLEEVE_MAX_STALE_DAYS)
        if last_pt is not None else None)
    if diag["bar_freshness_ok"] is False:
        diag["bar_freshness_note"] = (
            f"最新点 {last_pt['date']} 用的是 {diag['bar_lag_days_last']} 个交易日前的"
            f" ETF 收盘价（容忍上限 {SLEEVE_MAX_STALE_DAYS}）"
            " → ETF 管道未更新到数据日，先跑 scripts/etf_data_pipeline.py --incremental")

    diag.update(_nav_consistency_check(points, paper.get("nav_history")))
    return points, diag


def build_parent_points(a_points: List[Dict[str, Any]],
                        sleeve_points: List[Dict[str, Any]],
                        cap0: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """母线（定稿线）= 当日两条腿市值**相加**（静态 50/50，腿间不再平衡）。

    ⚠️ 只合并**两条腿都有点**的日期。缺一条腿还照样相加，
    等于把组合当成只剩一半 → 会造出 −50% 的假暴跌点。
    缺哪条腿、缺哪些日期，写进 diag（不许静默）。
    """
    a_by = {str(p.get("date")): p for p in a_points}
    s_by = {str(p.get("date")): p for p in sleeve_points}
    both = sorted(set(a_by) & set(s_by))
    points: List[Dict[str, Any]] = []
    for d in both:
        pa, ps = a_by[d], s_by[d]
        va, vs = float(pa.get("value") or 0.0), float(ps.get("value") or 0.0)
        v = va + vs
        stale = sorted(set((pa.get("stale_price") or []))
                       | set((ps.get("stale_price") or [])))
        points.append({
            "date": d,
            "value": round(v, 2),
            "return_pct": round((v / cap0 - 1.0) * 100, 4) if cap0 else 0.0,
            "n_positions": int(pa.get("n_positions") or 0)
                           + int(ps.get("n_positions") or 0),
            "legs": {"a": round(va, 2), "sleeve": round(vs, 2)},
            "stale_price": stale or None,
            "bar_lag_days": int(ps.get("bar_lag_days") or 0),
        })
    diag = {"strategy": MULTI_ASSET_PARENT,
            "n_points": len(points),
            "n_a_points": len(a_by), "n_sleeve_points": len(s_by),
            "dates_only_a": sorted(set(a_by) - set(s_by)),
            "dates_only_sleeve": sorted(set(s_by) - set(a_by)),
            "n_stale_days": sum(1 for p in points if p.get("stale_price"))}
    return points, diag


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

    produced: Dict[str, List[Dict[str, Any]]] = {}
    gate_failures: List[str] = []          # 门禁 a：日频 vs 周频
    residual_failures: List[str] = []      # 门禁 b：重放现金 vs holdings 现金
    freshness_failures: List[str] = []     # 门禁 c：行情新鲜度（不许旧价冒充）
    benchmark_failures: List[str] = []     # 门禁 d：基准列（缺基准 = 不许出表）

    # ── 门禁 d：零技巧基准必须存在（自用 KPI 的对照组，缺了就没有"跑赢"可言）─
    try:
        b_levels, b_sleeve, b_meta = bsvc.load_benchmark()
        print(f"基准电平：{b_meta['pool_size']} 只，{b_meta['first_date']} → "
              f"{b_meta['last_date']}（覆盖率最低 {b_meta['coverage_min']:.2%}"
              f" @ {b_meta['coverage_min_date']}）", flush=True)
    except FileNotFoundError as exc:
        b_levels, b_sleeve, b_meta = None, {}, {}
        benchmark_failures.append(f"基准产物缺失：{exc}")
        print(f"❌ {exc}", flush=True)

    def attach_bench(line: str, merged: List[Dict[str, Any]]) -> Dict[str, Any]:
        """给这条线的全部净值点加 benchmark_return_pct / excess_pp（锚在最早点）。

        ⚠️ 必须在 `write_nav_daily` **之前**调用，否则 json 里没有这两列。
        """
        if b_levels is None:
            return {}
        kind = bsvc.LINE_KIND.get(line)
        if kind is None:
            # 新线没在 LINE_KIND 里表态 → 不许默认当 A 股基准
            benchmark_failures.append(f"{line} 未声明基准类型（benchmark_service.LINE_KIND）")
            return {}
        diag = bsvc.attach_benchmark_columns(merged, kind, b_levels, b_sleeve, b_meta)
        if diag.get("missing_benchmark_days"):
            benchmark_failures.append(
                f"{line}（{diag['missing_benchmark_days']} 天缺基准）")
        if diag.get("coverage_low_days"):
            benchmark_failures.append(
                f"{line}（{diag['coverage_low_days']} 天基准覆盖率<"
                f"{bsvc.MIN_COVERAGE:.0%}）")
        return diag

    def bench_txt(last: Dict[str, Any], diag: Dict[str, Any]) -> str:
        if not diag:
            return "基准 ✗（未加载）"
        if last.get("benchmark_return_pct") is None or last.get("excess_pp") is None:
            return "基准 ✗（末日缺基准）"
        return (f"基准 {last['benchmark_return_pct']:+.2f}% → "
                f"超额 {last['excess_pp']:+.2f}pp（锚 {diag.get('anchor')}）")

    for s in args.strategies:
        points, diag = build_line_points(s, adj_close, master, args.backfill,
                                         as_of=args.as_of,
                                         range_=args.range_)
        produced[s] = points
        if not points:
            print(f"  [{s}] {diag.get('status', '无点')}", flush=True)
            continue
        old = load_nav_daily(s)
        merged = merge_points(list(old.get("points") or []), points)
        b_diag = attach_bench(s, merged)          # 必须在写盘前，基准列要进 json
        if not args.dry_run:
            cap0 = float((pp.load_holdings(s, 1_000_000.0) or {}).get(
                "initial_capital") or 1_000_000.0)
            write_nav_daily(s, merged, cap0, NAV_BASIS)
        # ⚠️ 打印口径：窗口模式（--as-of / --range）下要报**本次产出点**的区间与末值，
        #    不能用 merged（合并后的全量）—— 否则 --range 2026-10-20:2026-10-30
        #    会打印出「区间 ~ 2026-11-02、末日市值 ¥978,983」，而它实际只产到 10-30，
        #    与下面的「与周频末点」比对口径也对不上（比的是 points 末日）。
        _last = points[-1]
        print(f"  [{s}] 点数 {len(merged)}（新增 {len(points)}，"
              f"本次区间 {points[0]['date']} ~ {_last['date']}）| "
              f"停牌/缺价日 {diag.get('n_stale_days', 0)} | "
              f"末日市值 ¥{_last['value']:,.0f} "
              f"（{(_last['return_pct'] or 0):+.2f}%）| {bench_txt(_last, b_diag)} | "
              f"与周频末点"
              f"{_consistency_text(diag)}", flush=True)
        # 门禁 a 失败（日频末点与周频 nav_history 末点不一致）必须让**退出码**变非 0，
        #    否则 cron 每天跑完都是 0，曲线与周频不一致没人会发现（静默失效）。
        if diag.get("nav_consistency_ok") is False:
            gate_failures.append(s)

    # ── 多资产定稿线：ETF 袖套腿 + 母线（2026-11-05 自用化 A 项）────────
    # A 腿已在上面的循环里产出（与 6 条单线同一条路径），这里补另两条：
    #   multi_asset_final_sleeve  → ETF 前复权价×股数 + ETF 费率重放现金
    #   multi_asset_final         → 当日两腿相加（与周频写 nav_history 同口径）
    do_multi = any(x in args.strategies for x in (MULTI_ASSET_A,
                                                  MULTI_ASSET_SLEEVE,
                                                  MULTI_ASSET_PARENT))
    if do_multi and MULTI_ASSET_A not in produced:
        print(f"  [{MULTI_ASSET_PARENT}] 跳过：本次没有 A 腿日频点"
              f"（--strategies 未含 {MULTI_ASSET_A}）", flush=True)
        do_multi = False

    if do_multi:
        s_points, s_diag = build_sleeve_points(master, args.backfill,
                                               as_of=args.as_of,
                                               range_=args.range_)
        if not s_points:
            print(f"  [{MULTI_ASSET_SLEEVE}] {s_diag.get('status', '无点')}",
                  flush=True)
        else:
            old = load_nav_daily(MULTI_ASSET_SLEEVE)
            merged = merge_points(list(old.get("points") or []), s_points)
            sb_diag = attach_bench(MULTI_ASSET_SLEEVE, merged)
            if not args.dry_run:
                write_nav_daily(MULTI_ASSET_SLEEVE, merged,
                                s_diag.get("initial_capital"), SLEEVE_NAV_BASIS)
            _last = s_points[-1]
            fresh = s_diag.get("bar_freshness_ok")
            fresh_txt = (f"新鲜度 ✓（滞后 {s_diag.get('bar_lag_days_last', 0)} 日）"
                         if fresh is True else
                         f"**新鲜度 ✗** {s_diag.get('bar_freshness_note', '')}")
            resid = s_diag.get("cash_residual")
            resid_txt = (f"现金残差 {resid:+.4f} ✓"
                         if s_diag.get("cash_residual_ok") is True else
                         f"**现金残差 {resid} ✗**")
            print(f"  [{MULTI_ASSET_SLEEVE}] 点数 {len(merged)}"
                  f"（新增 {len(s_points)}，本次区间 {s_points[0]['date']} ~ "
                  f"{_last['date']}）| ETF 滞后日 {s_diag.get('n_stale_days', 0)} | "
                  f"末日市值 ¥{_last['value']:,.0f} "
                  f"（{(_last['return_pct'] or 0):+.2f}%）| {bench_txt(_last, sb_diag)} | "
                  f"{resid_txt} | "
                  f"{fresh_txt} | 与周频末点{_consistency_text(s_diag)}",
                  flush=True)
            if s_diag.get("nav_consistency_ok") is False:
                gate_failures.append(MULTI_ASSET_SLEEVE)
            if s_diag.get("cash_residual_ok") is False:
                residual_failures.append(
                    f"{MULTI_ASSET_SLEEVE}（残差 {s_diag.get('cash_residual')}）")
            if s_diag.get("bar_freshness_ok") is False:
                freshness_failures.append(MULTI_ASSET_SLEEVE)

        parent_cap = float((pp.load_holdings(MULTI_ASSET_PARENT, 1_000_000.0)
                            or {}).get("initial_capital") or 1_000_000.0)
        p_points, p_diag = build_parent_points(produced[MULTI_ASSET_A],
                                               s_points, parent_cap)
        p_diag["initial_capital"] = parent_cap
        if not p_points:
            print(f"  [{MULTI_ASSET_PARENT}] 无合并点"
                  f"（A 腿 {p_diag.get('n_a_points')} 点 / "
                  f"袖套 {p_diag.get('n_sleeve_points')} 点）", flush=True)
        else:
            old = load_nav_daily(MULTI_ASSET_PARENT)
            merged = merge_points(list(old.get("points") or []), p_points)
            pb_diag = attach_bench(MULTI_ASSET_PARENT, merged)
            if not args.dry_run:
                write_nav_daily(MULTI_ASSET_PARENT, merged, parent_cap,
                                PARENT_NAV_BASIS)
            _last = p_points[-1]
            _cons_diag = dict(p_diag)
            _cons_diag.update(_nav_consistency_check(
                p_points,
                (pp.load_holdings(MULTI_ASSET_PARENT, parent_cap) or {}).get(
                    "nav_history")))
            print(f"  [{MULTI_ASSET_PARENT}] 点数 {len(merged)}"
                  f"（新增 {len(p_points)}，本次区间 {p_points[0]['date']} ~ "
                  f"{_last['date']}）| 腿：A ¥{_last['legs']['a']:,.0f} + "
                  f"袖套 ¥{_last['legs']['sleeve']:,.0f} | "
                  f"末日市值 ¥{_last['value']:,.0f} "
                  f"（{(_last['return_pct'] or 0):+.2f}%）| {bench_txt(_last, pb_diag)} | "
                  f"缺一条腿的日期 A_only {len(p_diag.get('dates_only_a') or [])} / "
                  f"S_only {len(p_diag.get('dates_only_sleeve') or [])} | "
                  f"与周频末点{_consistency_text(_cons_diag)}", flush=True)
            if _cons_diag.get("nav_consistency_ok") is False:
                gate_failures.append(
                    f"{MULTI_ASSET_PARENT}"
                    f"（差 {_cons_diag.get('nav_consistency_diff')}）")
            if p_diag.get("dates_only_a") or p_diag.get("dates_only_sleeve"):
                freshness_failures.append(
                    f"{MULTI_ASSET_PARENT}（两条腿日期不齐："
                    f"只有 A 腿 {p_diag.get('dates_only_a')}，"
                    f"只有袖套 {p_diag.get('dates_only_sleeve')}）")

    if gate_failures:
        print(f"\n❌ 门禁 a 失败（日频末点与周频 nav_history 末点不一致）："
              f"{', '.join(gate_failures)}", flush=True)
    if residual_failures:
        print(f"❌ 门禁 b 失败（ledger 重放现金 ≠ holdings 现金，容差 0.05）："
              f"{', '.join(residual_failures)}", flush=True)
    if freshness_failures:
        print(f"❌ 门禁 c 失败（行情新鲜度 / 两条腿日期不齐）："
              f"{', '.join(freshness_failures)}", flush=True)
    if benchmark_failures:
        print(f"❌ 门禁 d 失败（零技巧基准列缺失/覆盖率不足 —— 没有基准就不许谈超额）："
              f"{', '.join(benchmark_failures)}", flush=True)
    if gate_failures or residual_failures or freshness_failures or benchmark_failures:
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] 未写盘", flush=True)


if __name__ == "__main__":
    main()
