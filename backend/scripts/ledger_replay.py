#!/usr/bin/env python3
"""纸面盘 **ledger 重放**（方案 D：历史区间回溯）—— 可复用模块。

## 这个模块解决什么

方案 B′（`paper_daily_nav.py` 的日频盯市）画的是 **cron 从 2026-11-02 起每天记的点**。
但 ledger 里有 **2026-10-16 起的全部成交流水** → 可以把更早的日频曲线**回溯**出来。

本模块是方案 D 的**账本侧**实现（方案书 §2.3 步骤 1）：

1. **解析 ledger**（`parse_ledger`）
   - 固定 **15 列**契约（`paper_portfolio.LEDGER_FIELDS`）
   - 同时兼容 **13 列**旧契约（无 `cost_amount` / `realized_pnl`）
   - **按行长度判定契约**，不靠表头 → 表头与数据行错位时仍能读出真值
     （证据见台账 §12.53：`*.bak-navcols-20261103-031718` 里 13 列表头 + 15 列数据行）
2. **重建任意 as_of 的持仓与现金**（`replay_rows` / `state_at`）
   - 现金**从 `amount` + 费率重算**，**绝不读 `cash_after`**（三重语义混用，不可信）
   - 持仓语义：**全清仓 + 从 positions 删除**（与生产 `run_line` 一致）
3. **算日频净值点并标注缺价**（`build_nav_points`）
   - 每个交易日一个点；缺行情的日子**显式标注**（missing / stale / frozen + 计数），
     **不静默用前值填充**

## 现金为什么必须自己重算（台账 §12.53）

`cash_after` 混着三种语义，**任何一种都不能当权威值**：

| 批次 | 语义 | 证据（实测） |
|---|---|---|
| 2026-11-02（22f） | **错位值**（迁移把 `cost_amount` 落到 `cash_after`） | 真值 78942.93（备份原行 index13），账本里 31360.00 |
| 2026-10-23（22f） | **逐批快照**（同一天 4 批 → 4 个不同值） | 76669.81 / 73061.74 / 72616.55 / 77720.51 |
| 当前代码 | **整批同值**（一次 `run_line` 的所有行共享同一个 `round(cash,2)`） | 只有「该批是当日最后一批」时才等于日末现金 |

→ 重算口径（与 `paper_portfolio` 记账**同一套费率**）：

```
BUY : cash -= gross + max(5, gross×0.0003) + gross×0.001        （无印花税）
SELL: cash += gross - max(5, gross×0.0003) - gross×0.0005 - gross×0.001
gross = ledger 的 amount（买入时 = shares×price；卖出时 = **总回报口径** gross，
         不是 price×shares —— 卖出金额含后复权比值）
```

⚠️ **为什么不用 ledger 里现成的 `commission` / `stamp_tax` / `slippage` 三列**
（`cash_model="ledger_columns"`）：

那三列是**四舍五入到 2 位小数**的（`paper_portfolio` 写盘时 `round(x, 2)`），
而生产算现金用的是**未舍入**值 → **逐行舍入误差会累积**。
实测（§12.53 表，6 条线真实重放）：

| 现金模型 | 6 条线残差（vs `holdings.json` 的 cash） | 门禁 ≤0.05 |
|---|---|---|
| `ledger_columns`（用账本三列） | 0.02 / 0.06 / 0.05 / **0.11** / 0.02 / 0.02 | **2 条线不过** |
| `rate`（默认：按生产费率重算） | 0.0105 / 0.0028 / 0.0161 / 0.0009 / 0.0035 / 0.0035 | **6 条线全过** |

误差量级解释：每行 2dp 舍入误差 ≤0.005 元/费用列，
`ledger_columns` 每行累加 2~3 个费用列 → 148 行的线漂移可达 ~0.1 元；
`rate` 模型只剩「卖出 gross 的 2dp 舍入」一项 → 漂移 ≤0.02 元。
**默认用 `rate`**；`ledger_columns` 保留仅作对照/审计。

⚠️ **与方案书 §2.2 字面公式的关系（有意偏离）**：§2.2 写的「从
`amount/commission/stamp_tax/slippage` 重算」字面上等于 `ledger_columns`，
但按字面实现**过不了 §2.6 的「残差 ≤0.05」门禁**（实测 2 条线不过）。
默认 `rate` 与 §2.2 的**意图**一致（用 `amount` + 生产同一套费率重算，
而不是读 `cash_after`），只是费率由代码重算而非读账本那三列。

## 持仓语义：全清仓（与生产一致）

`paper_portfolio.run_line` 卖出时是**卖掉整个持仓并从 positions 删除**
（`del positions[c]`），买入时 `positions[c] = {...}` **整体覆盖**。
→ 本模块采用同样语义（`semantics="full_liquidation"`）。
现有账本里**没有任何部分卖出、也没有对同一只票的重复买入**
（6 条线实测：partial sell 0 次、BUY 已在持仓 0 次），
所以"全清仓"与"移动平均成本"两种语义在现有数据上**结果完全相同** ——
这个选择**无法被现有数据证伪**，因此选**与生产一致**的那个。

## 用法

```python
from scripts import ledger_replay as lr

rows, diag = lr.parse_ledger("data/paper/22f_ledger.csv")
snaps = lr.replay_rows(rows, initial_capital=1_000_000.0, adj_close=adj)
pos, cash = lr.state_at(snaps, "2026-10-26")
points = lr.build_nav_points(snaps, adj_close=adj, trading_days=days, ...)
```

命令行（**只读盘、不写 `data/paper/`**；会取纸面盘同一把互斥锁）：

```bash
env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/ledger_replay.py --strategy 22f --json /tmp/replay_22f.json
```
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

# ── 记账费率（与 paper_portfolio 完全一致，**单一来源**）──────────────────
COMMISSION_RATE = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX_RATE = 0.0005
SLIPPAGE_RATE = 0.001

# ── ETF 袖套费率（2026-11-05 新增）──────────────────────────────────────
# ⚠️ **从 app 层 import，不在这里抄数字**：袖套的生产实现在
# `app/services/trading/multi_asset_service.py`（`rebalance_sleeve`）。
# 抄一遍就是「口径漂移」的起点（本项目最贵的错误类型）。
from app.services.trading.multi_asset_service import (  # noqa: E402
    ETF_COMMISSION as ETF_COMMISSION_RATE,
    ETF_SLIPPAGE as ETF_SLIPPAGE_RATE,
)

#: 15 列契约（当前 `paper_portfolio.LEDGER_FIELDS`）
LEDGER_FIELDS_15 = [
    "date", "action", "code", "name", "price", "shares", "amount",
    "commission", "stamp_tax", "slippage", "score",
    "cost_amount", "realized_pnl", "cash_after", "portfolio_value",
]

#: 13 列旧契约（早于 `cost_amount` / `realized_pnl` 的写入）
LEDGER_FIELDS_13 = [
    "date", "action", "code", "name", "price", "shares", "amount",
    "commission", "stamp_tax", "slippage", "score",
    "cash_after", "portfolio_value",
]

#: 合并账本 `multi_asset_final_ledger.csv` 是 14 列（多一个 leg 列）→ 不支持
_UNSUPPORTED_CONTRACTS = {14}

_BOOL_FIELDS = ("price", "shares", "amount", "commission", "stamp_tax",
                "slippage", "score", "cost_amount", "realized_pnl",
                "cash_after", "portfolio_value")


# ══════════════════════════════════════════════════════════════════════
# 1. 解析
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LedgerRow:
    """一行成交（字段名固定，缺失值为 None）。"""

    date: str
    action: str
    code: str
    name: str = ""
    price: Optional[float] = None
    shares: Optional[float] = None
    amount: Optional[float] = None
    commission: Optional[float] = None
    stamp_tax: Optional[float] = None
    slippage: Optional[float] = None
    score: Optional[float] = None
    cost_amount: Optional[float] = None
    realized_pnl: Optional[float] = None
    cash_after: Optional[float] = None
    portfolio_value: Optional[float] = None
    contract: str = "15"
    index: int = -1

    @property
    def is_buy(self) -> bool:
        return self.action == "BUY"

    @property
    def is_sell(self) -> bool:
        return self.action == "SELL"


def _to_float(v: Any) -> Optional[float]:
    """字符串 → float；空/非法 → None（**不是 0**：0 是合法值）。"""
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


def _contract_for(n_fields: int) -> Optional[List[str]]:
    """按**行长度**判定列契约（不靠表头）。"""
    if n_fields == 15:
        return list(LEDGER_FIELDS_15)
    if n_fields == 13:
        return list(LEDGER_FIELDS_13)
    return None


def parse_ledger(path: str) -> Tuple[List[LedgerRow], Dict[str, Any]]:
    """解析 ledger.csv → (行列表, 诊断信息)。

    ## 为什么要"按行长度"而不是"按表头"

    台账 §12.48 记录的真实事故：表头是 13 列、数据行是 15 列 →
    `csv.DictReader` 把 `cash_after` 读成了 `cost_amount`。
    事后按**旧表头位置**做的迁移又把错位值**固化**进了新文件（§12.53）。

    → 本模块**不看表头**做字段映射，而是按**每行的字段数**判定契约：
      - 15 个字段 → 15 列契约（`cash_after` 在第 14 位）
      - 13 个字段 → 13 列契约（`cash_after` 在第 12 位）
    这样即使表头与数据行错位，也能读出**真值**，
    并在诊断里报告"表头契约 vs 数据行契约"的不一致。

    诊断字段：
      - `header_contract`：表头声明的契约（"15" / "13" / "unknown"）
      - `row_contracts`：{"15": n, "13": n}
      - `contract_mismatch_rows`：数据行契约与表头不一致的行号
      - `unparsed_rows`：无法判定的行号（长度既非 13 也非 15）
    """
    diag: Dict[str, Any] = {
        "path": path,
        "header_contract": "unknown",
        "row_contracts": {},
        "contract_mismatch_rows": [],
        "unparsed_rows": [],
    }
    rows: List[LedgerRow] = []
    if not os.path.exists(path):
        diag["error"] = "ledger 不存在"
        return rows, diag

    with open(path, newline="", encoding="utf-8") as fh:
        raw = list(csv.reader(fh))
    if not raw:
        diag["error"] = "空文件"
        return rows, diag

    header = [h.strip() for h in raw[0]]
    h_fields = _contract_for(len(header))
    diag["header_contract"] = "15" if h_fields == LEDGER_FIELDS_15 else (
        "13" if h_fields == LEDGER_FIELDS_13 else "unknown")

    for i, raw_row in enumerate(raw[1:]):
        if not raw_row or all(not str(c).strip() for c in raw_row):
            continue
        fields = _contract_for(len(raw_row))
        if fields is None:
            diag["unparsed_rows"].append(i + 2)   # +2：含表头的 1-based 行号
            continue
        contract = "15" if fields == LEDGER_FIELDS_15 else "13"
        diag["row_contracts"][contract] = diag["row_contracts"].get(contract, 0) + 1
        if contract != diag["header_contract"]:
            diag["contract_mismatch_rows"].append(i + 2)

        vals = dict(zip(fields, raw_row))
        row = LedgerRow(
            date=str(vals.get("date") or "").strip(),
            action=str(vals.get("action") or "").strip().upper(),
            code=str(vals.get("code") or "").strip(),
            name=str(vals.get("name") or "").strip(),
            contract=contract,
            index=len(rows),
        )
        for f in _BOOL_FIELDS:
            setattr(row, f, _to_float(vals.get(f)))
        if not row.date or not row.code or row.action not in ("BUY", "SELL"):
            diag["unparsed_rows"].append(i + 2)
            continue
        rows.append(row)

    diag["n_rows"] = len(rows)
    return rows, diag


# ══════════════════════════════════════════════════════════════════════
# 2. 重放：持仓 + 现金
# ══════════════════════════════════════════════════════════════════════

def buy_cost_from_amount(amount: Optional[float], shares: Optional[float],
                         price: Optional[float]) -> Optional[float]:
    """买入 gross：优先用账本 `amount`，缺失时回退 `shares × price`。"""
    if amount is not None and amount > 0:
        return float(amount)
    if shares is not None and price is not None:
        return float(shares) * float(price)
    return None


def cash_delta(row: LedgerRow,
               cash_model: str = "rate",
               ) -> Optional[float]:
    """这一行对现金的**影响**（元）。`None` 表示无法计算。

    ## `cash_model`

    - `"rate"`（**默认**）：按 `paper_portfolio` 的**生产费率**从 `amount` 重算
      ```
      BUY : -(gross + max(5, gross×0.0003) + gross×0.001)        # 无印花税
      SELL: +(gross - max(5, gross×0.0003) - gross×0.0005 - gross×0.001)
      ```
      这与生产算现金的表达式**逐项相同**（生产用的是未舍入值，
      这里只有 `gross` 本身经过账本的 2dp 舍入）。

    - `"ledger_columns"`：用账本里现成的 `commission` / `stamp_tax` / `slippage`
      三列：
      ```
      BUY : -(amount + commission + slippage)                      # 无印花税
      SELL: +(amount - commission - stamp_tax - slippage)
      ```
      ⚠️ 这三列是**写盘时 `round(x, 2)` 后的值**，生产算现金用的是未舍入值
      → 逐行舍入误差累积（实测 `factor_ranking` 残差 +0.11 元，超 0.05 门禁）。
      **仅保留作对照/审计，默认不用。**

    - `"etf_rate"`（**ETF 袖套腿专用**，2026-11-05 新增）：
      ```
      BUY : -(amount × (1 + ETF_COMMISSION))
      SELL: +(amount × (1 − ETF_COMMISSION))
      ```
      为什么**不能**用默认的 `rate` 重放袖套账本（三处实质差异）：
      ① ETF **无印花税**（`rate` 会给每笔卖出扣 0.0005）；
      ② ETF 的滑点**不在现金里** —— `rebalance_sleeve` 把滑点做进成交价
      （`buy_px = p×(1+ETF_SLIPPAGE)` / `sell_px = p×(1−ETF_SLIPPAGE)`），
      账本 `slippage` 列只是展示值，`rate` 会**再扣一遍** 0.001；
      ③ ETF **没有 5 元最低佣金**（`max(5, ...)` 会把小额腿的佣金放大）。
      实测量级（2026-11-05，袖套账本 9 笔首建、成交额 ¥499,980）：
      `rate` → 重放出现金 **−479.57**（真值 +20.27，残差 −499.84，现金被算成负数）；
      `ledger_columns` → 残差 **−50.00**（`slippage` 列被重复扣）；
      `etf_rate` → 残差 **−0.0089** ✓ 过 ≤0.05 门禁。
      费率 **import 自 `multi_asset_service`**（生产实现），不在本文件抄数字。
    """
    if cash_model not in ("rate", "ledger_columns", "etf_rate"):
        raise ValueError(f"未知 cash_model: {cash_model}")

    if row.is_buy:
        # 买入 gross：账本 amount 优先；缺失/非正 → shares×price 兜底
        gross = buy_cost_from_amount(row.amount, row.shares, row.price)
    else:
        gross = row.amount
    if gross is None or gross <= 0:
        # SELL 的 gross 兜底（cost×股数×后复权比值）需要持仓上下文 →
        # 由 `replay_rows` 处理（它知道当时的持仓）。
        return None

    if cash_model == "etf_rate":
        commission = gross * ETF_COMMISSION_RATE
        if row.is_buy:
            return -(gross + commission)
        if row.is_sell:
            return gross - commission
        return None

    if cash_model == "ledger_columns":
        commission = float(row.commission or 0.0)
        slippage = float(row.slippage or 0.0)
        if row.is_buy:
            # 买入无印花税
            return -(gross + commission + slippage)
        if row.is_sell:
            stamp_tax = float(row.stamp_tax or 0.0)
            return (gross - commission - stamp_tax - slippage)
        return None

    commission = max(MIN_COMMISSION, gross * COMMISSION_RATE)
    slippage = gross * SLIPPAGE_RATE
    if row.is_buy:
        # 买入无印花税
        return -(gross + commission + slippage)
    if row.is_sell:
        # 卖出：gross 是**总回报口径**（含后复权比值），不是 price×shares
        return (gross - commission - gross * STAMP_TAX_RATE - slippage)
    return None


def replay_rows(rows: Sequence[LedgerRow],
                initial_capital: float = 1_000_000.0,
                *,
                adj_close: Optional[pd.DataFrame] = None,
                last_valid_fn: Optional[Callable[..., Any]] = None,
                semantics: str = "full_liquidation",
                cash_model: str = "rate",
                ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """逐笔重放 ledger → 每个成交日一个快照。

    返回 `(snapshots, diag)`：
      - `snapshots`：[{"date", "index", "positions", "cash"}]（按行序）
      - `diag`：{"n_buy", "n_sell", "n_unpriced", "partial_sells", "re_buys",
                "cash_unavailable_rows", "final_cash"}

    ## `semantics`

    - `"full_liquidation"`（**默认**，与生产 `run_line` 一致）：
      卖出 = 卖掉整个持仓并**从 positions 删除**；买入 = 整体覆盖持仓。
    - `"moving_average"`（旧 `paper_daily_nav.replay_holdings` 的口径，保留兼容）：
      卖出按股数比例扣减、每股成本不变；买入取加权平均成本。

    现有账本里两种语义**结果完全相同**（无部分卖出、无重复买入），
    所以这个选择**无法被现有数据证伪** → 选与生产一致的那个。
    """
    if semantics not in ("full_liquidation", "moving_average"):
        raise ValueError(f"未知 semantics: {semantics}")

    positions: Dict[str, Dict[str, Any]] = {}
    cash = float(initial_capital)
    snapshots: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "semantics": semantics,
        "n_buy": 0, "n_sell": 0, "n_unpriced": 0,
        "partial_sells": [], "re_buys": [],
        "cash_unavailable_rows": [],
        "initial_capital": float(initial_capital),
    }

    def _adj_cost(code: str, date: str) -> Tuple[Optional[float], Optional[str]]:
        if adj_close is None or last_valid_fn is None:
            return None, None
        try:
            v, d = last_valid_fn(adj_close, code, pd.Timestamp(date))
        except (KeyError, TypeError, ValueError):
            return None, None
        if v is None:
            return None, None
        return round(float(v), 6), (str(d) if d else None)

    for row in rows:
        d_cash = cash_delta(row, cash_model=cash_model)
        if d_cash is None and row.is_sell:
            # SELL 的 gross 兜底：cost×股数×后复权比值（与生产 run_line 的
            # `gross = cost * shares * _adj_ratio(c)` 同口径）。
            # 账本 amount 缺失时用这个兜底（实测真实账本里 SELL 的 amount 都有值，
            # 这条分支只在合成数据/未来异常时兜底）。
            held = positions.get(row.code)
            if held is not None:
                sh = float(row.shares or 0.0)
                a_now, _pd = _adj_cost(row.code, row.date)
                a_cost = held.get("adj_cost")
                ratio = 1.0
                try:
                    if a_now is not None and a_cost is not None:
                        a_cost_f = float(a_cost)
                        if np.isfinite(a_cost_f) and a_cost_f > 0:
                            ratio = float(a_now) / a_cost_f
                except (TypeError, ValueError):
                    ratio = 1.0
                gross = float(held.get("cost") or 0.0) * sh * ratio
                if gross > 0:
                    # 兜底时账本三列可能也缺失 → 按 cash_model 对应的费率重算
                    # （2026-11-05：袖套腿走 ETF 费率，不再沿用 A 股三项费用）
                    if cash_model == "etf_rate":
                        d_cash = gross * (1.0 - ETF_COMMISSION_RATE)
                    else:
                        commission = max(MIN_COMMISSION,
                                         gross * COMMISSION_RATE)
                        slippage = gross * SLIPPAGE_RATE
                        d_cash = (gross - commission - gross * STAMP_TAX_RATE
                                  - slippage)
        if d_cash is None:
            diag["cash_unavailable_rows"].append(row.index)
        else:
            cash += d_cash

        if row.is_buy:
            diag["n_buy"] += 1
            if row.code in positions:
                diag["re_buys"].append({"row": row.index, "code": row.code,
                                        "date": row.date})
            gross = buy_cost_from_amount(row.amount, row.shares, row.price)
            shares = float(row.shares) if row.shares is not None else 0.0
            price = float(row.price) if row.price is not None else 0.0
            if semantics == "moving_average":
                p = positions.setdefault(row.code, {
                    "shares": 0.0, "cost": 0.0, "buy_date": row.date,
                    "adj_cost": None, "adj_cost_date": None})
                old_shares = float(p["shares"])
                new_shares = old_shares + shares
                p["cost"] = (round((float(p["cost"]) * old_shares + price * shares)
                                   / new_shares, 3) if new_shares else 0.0)
                p["shares"] = new_shares
                p["buy_date"] = row.date
                a, ad = _adj_cost(row.code, row.date)
                if a is not None:
                    p["adj_cost"] = a
                    p["adj_cost_date"] = ad
            else:
                # 全清仓语义：买入 = 整体覆盖（与 run_line 的 positions[c] = {...} 一致）
                a, ad = _adj_cost(row.code, row.date)
                positions[row.code] = {
                    "shares": shares,
                    "cost": round(price, 3),
                    "buy_date": row.date,
                    "adj_cost": a,
                    "adj_cost_date": ad,
                }
            if gross is None:
                diag["n_unpriced"] += 1
        elif row.is_sell:
            diag["n_sell"] += 1
            held = positions.get(row.code)
            sold = float(row.shares) if row.shares is not None else 0.0
            if held is None:
                diag["partial_sells"].append({"row": row.index, "code": row.code,
                                               "reason": "卖出时不在持仓"})
            elif abs(float(held["shares"]) - sold) > 1e-9:
                diag["partial_sells"].append({
                    "row": row.index, "code": row.code,
                    "reason": "部分卖出", "held": held["shares"], "sold": sold})
            if semantics == "moving_average":
                if held is not None:
                    held["shares"] = max(0.0, float(held["shares"]) - sold)
                    if held["shares"] <= 0:
                        positions.pop(row.code, None)
            else:
                # 全清仓：整只从 positions 删除
                positions.pop(row.code, None)

        snapshots.append({
            "date": row.date,
            "index": row.index,
            "positions": {k: dict(v) for k, v in positions.items()},
            "cash": cash,
            "initial_capital": float(initial_capital),
        })

    diag["final_cash"] = cash
    diag["n_positions"] = len(positions)
    return snapshots, diag


def state_at(snapshots: Sequence[Dict[str, Any]], as_of: str
             ) -> Tuple[Dict[str, Dict[str, Any]], float, Dict[str, Any]]:
    """重建 **as_of（含当日）** 的持仓与现金。

    返回 `(positions, cash, info)`；`info` 含命中快照的日期与行号。
    as_of 早于第一笔成交 → 空持仓 + 初始现金。
    """
    if not snapshots:
        return {}, 0.0, {"matched": None, "note": "无快照"}
    as_of_s = str(as_of)[:10]
    hit = None
    for snap in snapshots:
        if str(snap["date"])[:10] <= as_of_s:
            hit = snap
        else:
            break
    if hit is None:
        cap = 0.0
        if snapshots:
            cap = float(snapshots[0].get("initial_capital") or 0.0)
        return {}, cap, {
            "matched": None, "note": "as_of 早于第一笔成交"}
    return ({k: dict(v) for k, v in (hit["positions"] or {}).items()},
            float(hit["cash"]),
            {"matched": str(hit["date"]), "row_index": hit["index"]})


# ══════════════════════════════════════════════════════════════════════
# 3. 日频净值点（每交易日一个点，缺价显式标注）
# ══════════════════════════════════════════════════════════════════════

def build_nav_points(snapshots: Sequence[Dict[str, Any]],
                     *,
                     adj_close: Optional[pd.DataFrame],
                     trading_days: Iterable[Any],
                     initial_capital: float = 1_000_000.0,
                     value_fn: Optional[Callable[..., Any]] = None,
                     last_valid_fn: Optional[Callable[..., Any]] = None,
                     start: Optional[str] = None,
                     end: Optional[str] = None,
                     ) -> List[Dict[str, Any]]:
    """逐交易日产出净值点；**缺价的日子显式标注**（不静默用前值填充）。

    每个点的字段：
      - `date` / `value` / `return_pct` / `n_positions`
      - `stale_price`：该点**不是用当日收盘价**的标的代码列表（兼容旧字段）
      - `stale_codes` / `stale_count`：用了**更早**的有效价（停牌 / 涨跌幅未同步）
      - `missing_codes` / `missing_count`：**完全无价**（该日无任何有效后复权价）
      - `frozen_codes` / `frozen_count`：`adj_cost` 无效 → 比值恒为 1.0（市值冻结在买入额）
      - `stale_price_dates`：{code: 实际用的价格日期}

    估值**只实现一次**：调 `paper_portfolio.total_return_portfolio_value`
    （与周频调仓同一份代码）；本模块不另立公式。
    """
    if value_fn is None or last_valid_fn is None:
        pp = load_paper_portfolio()
        value_fn = value_fn or pp.total_return_portfolio_value
        last_valid_fn = last_valid_fn or pp.last_valid_adj

    days = sorted(pd.Timestamp(d) for d in trading_days)
    if start:
        s = pd.Timestamp(start)
        days = [d for d in days if d >= s]
    if end:
        e = pd.Timestamp(end)
        days = [d for d in days if d <= e]

    snaps = sorted(snapshots, key=lambda s: (str(s["date"])[:10], s["index"]))
    points: List[Dict[str, Any]] = []
    i = 0
    cur_pos: Dict[str, Any] = {}
    cur_cash = float(initial_capital)
    first_date = str(snaps[0]["date"])[:10] if snaps else None
    last_date = str(snaps[-1]["date"])[:10] if snaps else None

    for d in days:
        ds = str(d.date())
        # 推进到该日已发生的所有成交（**同一日多行**按行序推进 → 末行状态生效）
        while i < len(snaps) and str(snaps[i]["date"])[:10] <= ds:
            cur_pos = {k: dict(v) for k, v in (snaps[i]["positions"] or {}).items()}
            cur_cash = float(snaps[i]["cash"])
            i += 1
        if first_date and ds < first_date:
            # 首次建仓之前：没有持仓 → 不产出点（否则是假的 0 收益点）
            continue
        if last_date and ds > last_date:
            # 账本最后一次成交之后：持仓与现金**不再变化**，继续盯市（价格还在动）
            pass

        value, details = value_fn(cur_pos, cur_cash, adj_close, d)
        stale_codes: List[str] = []
        missing_codes: List[str] = []
        frozen_codes: List[str] = []
        price_dates: Dict[str, str] = {}
        for code, pos in (cur_pos or {}).items():
            p = pos or {}
            sh = float(p.get("shares", 0) or 0)
            cost = float(p.get("cost", 0) or 0)
            if sh <= 0 or cost <= 0:
                continue
            a_cost = p.get("adj_cost")
            try:
                a_cost_f = float(a_cost) if a_cost is not None else float("nan")
            except (TypeError, ValueError):
                a_cost_f = float("nan")
            if not (np.isfinite(a_cost_f) and a_cost_f > 0):
                frozen_codes.append(code)
            try:
                v, pdate = last_valid_fn(adj_close, code, d)
            except (KeyError, TypeError, ValueError):
                v, pdate = None, None
            if v is None:
                missing_codes.append(code)
            else:
                pdate_s = str(pdate) if pdate else ""
                if pdate_s != ds:
                    stale_codes.append(code)
                    if pdate_s:
                        price_dates[code] = pdate_s

        points.append({
            "date": ds,
            "value": round(float(value), 2),
            "return_pct": (round((float(value) / initial_capital - 1.0) * 100, 4)
                           if initial_capital else 0.0),
            "n_positions": len(cur_pos or {}),
            # 兼容旧字段：所有"该点不是用当日收盘价"的标的
            "stale_price": sorted(set(stale_codes) | set(missing_codes)
                                  | set(frozen_codes)) or None,
            "stale_codes": sorted(stale_codes),
            "stale_count": len(stale_codes),
            "missing_codes": sorted(missing_codes),
            "missing_count": len(missing_codes),
            "frozen_codes": sorted(frozen_codes),
            "frozen_count": len(frozen_codes),
            "stale_price_dates": price_dates,
        })
    return points


# ══════════════════════════════════════════════════════════════════════
# 4. paper_portfolio 加载（与 paper_daily_nav 同一套 importlib 约定）
# ══════════════════════════════════════════════════════════════════════

_PP = None


def load_paper_portfolio():
    """按 importlib 加载 `scripts/paper_portfolio.py`（模块名 `paper_portfolio_mod`）。

    ⚠️ 与 `from scripts import paper_portfolio` **不是同一个模块对象**
    （各自一份模块级状态，如 `PAPER_DIR`）→ 测试 monkeypatch 必须打在实际使用的
    那一个上（台账 §12.52 的"双模块对象"坑）。
    """
    global _PP
    if _PP is not None:
        return _PP
    script = os.path.join(BACKEND_ROOT, "scripts", "paper_portfolio.py")
    spec = importlib.util.spec_from_file_location("paper_portfolio_mod", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # ⚠️ 先注册进 sys.modules 再 exec_module：
    # 被加载的模块若含 `@dataclass` + `from __future__ import annotations`
    # （注解是字符串），dataclasses 装饰器会回 sys.modules 找模块来解析注解
    # → 不注册会抛 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    sys.modules.setdefault("paper_portfolio_mod", mod)
    spec.loader.exec_module(mod)
    _PP = mod
    return mod


# ══════════════════════════════════════════════════════════════════════
# 5. 验收：重放 vs holdings.json
# ══════════════════════════════════════════════════════════════════════

def compare_with_holdings(strategy: str,
                          positions: Dict[str, Dict[str, Any]],
                          cash: float,
                          holdings: Dict[str, Any],
                          ) -> Dict[str, Any]:
    """把重放结果与 holdings.json 对账（锚点 A / B）。

    - A：`shares` / `cost` / `buy_date` **逐位**比较（0 差异才算过）
    - B：`cash` 残差（阈值 0.05 元）
    """
    hp = holdings.get("positions") or {}
    diffs: List[Dict[str, Any]] = []
    for code in sorted(set(hp) | set(positions)):
        a = hp.get(code)
        b = positions.get(code)
        if (a is None) != (b is None):
            diffs.append({"code": code, "field": "presence",
                          "holdings": a, "replay": b})
            continue
        for k in ("shares", "cost", "buy_date"):
            av = a.get(k)
            bv = b.get(k)
            if av != bv:
                diffs.append({"code": code, "field": k,
                              "holdings": av, "replay": bv})
    h_cash = float(holdings.get("cash") or 0.0)
    return {
        "strategy": strategy,
        "n_positions_holdings": len(hp),
        "n_positions_replay": len(positions),
        "position_diffs": diffs,
        "n_position_diffs": len(diffs),
        "position_diff_ok": len(diffs) == 0,
        "holdings_cash": h_cash,
        "replay_cash": float(cash),
        "cash_diff": round(float(cash) - h_cash, 4),
        "cash_diff_ok": abs(float(cash) - h_cash) <= 0.05,
        "adj_cost_diffs": [
            {"code": c,
             "holdings": (hp.get(c) or {}).get("adj_cost"),
             "replay": (positions.get(c) or {}).get("adj_cost"),
             "holdings_date": (hp.get(c) or {}).get("adj_cost_date"),
             "replay_date": (positions.get(c) or {}).get("adj_cost_date")}
            for c in sorted(set(hp) & set(positions))
            if (hp.get(c) or {}).get("adj_cost") != (positions.get(c) or {}).get("adj_cost")
        ],
    }


# ══════════════════════════════════════════════════════════════════════
# 6. CLI（只读盘；不写 data/paper/）
# ══════════════════════════════════════════════════════════════════════

DEFAULT_LINES = ["22f", "honest_42f", "rsi_71421", "factor_ranking",
                 "gbm_voltgt", "honest_42f_neu"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="纸面盘 ledger 重放（方案 D）—— 只读盘，不写 data/paper/")
    ap.add_argument("--strategy", action="append", default=None,
                    help="要重放的线（可重复；默认 6 条单线）")
    ap.add_argument("--as-of", default=None,
                    help="只重建 as_of（含）当日的持仓与现金")
    ap.add_argument("--range", dest="range_", default=None,
                    help="区间回溯，格式 START:END（两个日期都要是交易日）")
    ap.add_argument("--json", default=None, help="把结果写到该 JSON 文件")
    ap.add_argument("--cash-model", default="rate",
                    choices=["rate", "ledger_columns"],
                    help="现金重算模型：rate=按生产费率从 amount 重算（默认）；"
                         "ledger_columns=用账本 commission/stamp_tax/slippage 三列"
                         "（仅作对照，会因 2dp 舍入累积超门禁）")
    ap.add_argument("--lock-timeout", type=int, default=1800,
                    help="互斥锁等待秒数（默认 1800，与 paper_portfolio 一致）")
    ap.add_argument("--no-lock", action="store_true",
                    help="⚠️ 跳过互斥锁（只在纯离线分析时用）")
    args = ap.parse_args(argv)

    strategies = args.strategy or DEFAULT_LINES
    pp = load_paper_portfolio()

    lock_fd = None
    if not args.no_lock:
        # 取纸面盘同一把 fcntl 锁；**必须在加载行情之前**（那 2~4 分钟里排期实例可能已开跑）
        lock_fd = pp.acquire_run_lock(timeout_s=args.lock_timeout)
        print(f"✅ 已取得纸面盘互斥锁（fd={lock_fd}）", flush=True)

    print("加载行情与后复权价格（较慢，约 2~4 分钟）……", flush=True)
    codes = pp.ds.list_universe()
    _frames, _panel, master = pp.ds.load_data(codes)
    adj_close = pp.load_adjusted_close(codes, master)
    print(f"数据末日 = {master[-1].date()}", flush=True)

    out: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "data_end": str(master[-1].date()),
        "basis": "total_return@trade_day (paper_portfolio.total_return_portfolio_value)",
        "cash_model": args.cash_model,
        "cash_model_note": (
            "rate=gross=ledger.amount; BUY: -(gross+max(5,gross*0.0003)+gross*0.001); "
            "SELL: +(gross-max(5,gross*0.0003)-gross*0.0005-gross*0.001)"
            if args.cash_model == "rate" else
            "ledger_columns=gross=ledger.amount; BUY: -(amount+commission+slippage); "
            "SELL: +(amount-commission-stamp_tax-slippage)"),
        "position_semantics": "full_liquidation (与 paper_portfolio.run_line 一致)",
        "reproducibility_note": (
            "后复权序列是递推出来的 → 历史值会随数据更新漂移 → "
            "本曲线是『按当前后复权序列回算的结果』，不是可复现的历史事实"),
        "strategies": {},
    }

    for s in strategies:
        rows, ldiag = parse_ledger(pp.ledger_path(s))
        holdings = pp.load_holdings(s, 1_000_000.0)
        cap0 = float(holdings.get("initial_capital") or 1_000_000.0)
        snaps, rdiag = replay_rows(rows, cap0, adj_close=adj_close,
                                   last_valid_fn=pp.last_valid_adj,
                                   cash_model=args.cash_model)
        pos, cash, info = state_at(snaps, args.as_of) if args.as_of else (
            (snaps[-1]["positions"] if snaps else {}),
            (snaps[-1]["cash"] if snaps else cap0), {})
        cmp = compare_with_holdings(s, pos, cash, holdings)

        nav_hist = list(holdings.get("nav_history") or [])
        nav_last = nav_hist[-1] if nav_hist else None
        end_val = None
        end_diff = None
        if nav_last and nav_last.get("date"):
            as_of = str(nav_last["date"])[:10]
            p2, c2, _ = state_at(snaps, as_of)
            v2, _det = pp.total_return_portfolio_value(p2, c2, adj_close,
                                                       pd.Timestamp(as_of))
            end_val = round(float(v2), 2)
            try:
                end_diff = round(end_val - float(nav_last.get("value") or 0.0), 4)
            except (TypeError, ValueError):
                end_diff = None

        start = end = None
        if args.range_:
            parts = str(args.range_).split(":")
            if len(parts) == 2:
                start, end = parts[0], parts[1]

        points = build_nav_points(
            snaps, adj_close=adj_close, trading_days=master,
            initial_capital=cap0,
            value_fn=pp.total_return_portfolio_value,
            last_valid_fn=pp.last_valid_adj,
            start=start, end=end,
        )
        n_days_missing = sum(1 for p in points if p.get("missing_count"))
        n_days_stale = sum(1 for p in points if p.get("stale_count"))

        out["strategies"][s] = {
            "ledger_diag": ldiag,
            "replay_diag": rdiag,
            "acceptance": cmp,
            "nav_end": {
                "nav_history_last": nav_last,
                "replay_value": end_val,
                "diff": end_diff,
                "ok": (end_diff is not None and abs(end_diff) <= 0.05),
            },
            "nav_points": {
                "n_points": len(points),
                "n_days_with_missing": n_days_missing,
                "n_days_with_stale": n_days_stale,
                "first": points[0]["date"] if points else None,
                "last": points[-1]["date"] if points else None,
            },
            "points": points,
        }
        acc = cmp
        print(f"  [{s}] A(positions)差异 {acc['n_position_diffs']} | "
              f"B(cash)残差 {acc['cash_diff']:+.4f} | "
              f"C(末点净值)差 {end_diff} | 日频点 {len(points)} "
              f"(缺价日 {n_days_missing} / 滞后价日 {n_days_stale})", flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print(f"已写出：{args.json}", flush=True)
    else:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "points"}
                          for k, v in out["strategies"].items()},
                         ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
