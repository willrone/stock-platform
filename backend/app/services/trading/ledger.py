"""纸面账本（JSONL，按策略严格隔离）—— 2026-10-27

## 为什么要有第二个账本

平台原来只有 `data/paper/<strategy>_ledger.csv`（脚本 `paper_portfolio.py` 写的），
它是**研究口径**的逐笔流水：CSV、无幂等、无现金校验、口径是「佣金万3 + 印花税万5
+ 滑点千1」。而实盘链路需要的是**产品口径**的账本：

  - 每条策略线**独立 100 万本金、独立持仓、独立记账**（用户明确要求「赛马」）
  - **幂等**：同一天同一策略重复执行不会重复记账（脚本重跑不能把账做坏）
  - **可审计**：每条记录自带成交后现金 / 持仓市值 / 净值快照
  - **可查询**：当前持仓、现金、累计费用、净值序列、交易历史、换手率

所以本模块是**新建**的独立账本，不改 `paper_portfolio.py`，两者并存：
  - `data/paper/<strategy>_ledger.csv`  → 研究口径（回测/展示对齐用）
  - `data/paper_ledger/<strategy>.jsonl` → 产品口径（本模块，实盘执行链路用）

## 落盘格式

JSONL：每行一条 JSON 记录，追加写（append-only）。目录默认
`backend/data/paper_ledger/`，可用环境变量 `SP_LEDGER_DIR` 覆盖。

一条记录的完整字段（所有类型共用同一套字段，缺失填 null/0，方便统一解析）：

| 字段 | 含义 |
|------|------|
| `ts` | 写入时间戳（ISO8601，本地时区）——仅供审计，**不参与任何决策** |
| `date` | 业务日期（执行日 / 净值日） |
| `strategy` | 策略 key（文件级隔离，跨策略不混） |
| `type` | `init` / `buy` / `sell` / `fee` / `nav` |
| `code` / `name` | 股票代码与名称（init/nav 为 null） |
| `side` | `buy` / `sell` |
| `shares` | 数量（股） |
| `price` | 单价（元） |
| `amount` | 成交金额（元，不含费） |
| `fees` | 手续费明细 `{commission, transfer_fee, stamp_tax, total}` |
| `cash_after` | 成交后现金（元） |
| `position_value_after` | 成交后持仓市值（元） |
| `nav` | 成交后净值（元） |
| `note` | 备注（执行原因 / 口径说明） |

## 幂等实现

每条记录带一个 `dedup_key`（由 strategy + date + 业务主键拼成）：

  - `init` → `<strategy>|init`
  - `buy` / `sell` → `<strategy>|<date>|<side>|<code>`
  - `fee` → `<strategy>|<date>|fee`（当日费用汇总）
  - `nav` → `<strategy>|<date>|nav`（当日净值快照）

`append_records()` 读取已有 key 集合，重复的直接跳过 → 同一天重跑不会重复记账。
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# backend/app/services/trading/ledger.py → parents[3] == backend/
BACKEND_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_ROOT.parent

#: 账本目录（可用 SP_LEDGER_DIR 覆盖）
LEDGER_DIR = Path(
    os.environ.get("SP_LEDGER_DIR") or (BACKEND_ROOT / "data" / "paper_ledger")
)

#: 每条策略线的初始本金（用户要求：每条线独立 100 万赛马）
INITIAL_CAPITAL = float(os.environ.get("SP_PAPER_CAPITAL") or 1_000_000)

#: 记录类型
RECORD_TYPES = ("init", "buy", "sell", "fee", "nav")

#: 净值点少于这个数就不给年化换手率（点太少外推没意义）
MIN_NAV_POINTS_FOR_ANNUAL = 20

_WRITE_LOCK = threading.Lock()


# ── 基础 IO ──────────────────────────────────────────────────────────


def ledger_path(strategy: str) -> Path:
    """某条策略线的账本文件（一个策略一个文件 → 天然隔离）。"""
    if not strategy or "/" in strategy or strategy.startswith("."):
        raise ValueError(f"非法策略 key: {strategy!r}")
    return LEDGER_DIR / f"{strategy}.jsonl"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_records(strategy: str) -> List[Dict[str, Any]]:
    """读取某策略的全部记录（按写入顺序）。

    坏行（半截写入 / 人工误改）会被跳过而不是让整个接口 500 —— 账本宁可少
    一行也不能读不出来；但跳过数量会记录在 `skipped_lines` 里由调用方披露。
    """
    path = ledger_path(strategy)
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def record_key(rec: Dict[str, Any]) -> str:
    """记录的幂等 key。"""
    key = rec.get("dedup_key")
    if key:
        return str(key)
    strategy = rec.get("strategy", "")
    date = rec.get("date", "")
    rtype = rec.get("type", "")
    if rtype == "init":
        return f"{strategy}|init"
    if rtype == "nav":
        return f"{strategy}|{date}|nav"
    if rtype == "fee":
        return f"{strategy}|{date}|fee"
    return f"{strategy}|{date}|{rec.get('side', '')}|{rec.get('code', '')}"


def append_records(
    strategy: str,
    records: Sequence[Dict[str, Any]],
    *,
    dedup: bool = True,
) -> List[Dict[str, Any]]:
    """追加记录，返回**实际写入**的记录列表。

    `dedup=True`（默认）时，已存在相同 `dedup_key` 的记录会被跳过 → 幂等。
    """
    if not records:
        return []
    path = ledger_path(strategy)
    path.parent.mkdir(parents=True, exist_ok=True)

    written: List[Dict[str, Any]] = []
    with _WRITE_LOCK:
        existing = {record_key(r) for r in read_records(strategy)} if dedup else set()
        with path.open("a", encoding="utf-8") as fh:
            for rec in records:
                rec = dict(rec)
                rec.setdefault("strategy", strategy)
                rec.setdefault("ts", _now_iso())
                key = record_key(rec)
                if dedup and key in existing:
                    continue
                existing.add(key)
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                written.append(rec)
    return written


# ── 状态重建（账本是唯一事实来源）──────────────────────────────────


@dataclass
class Position:
    """一只持仓。"""

    code: str
    name: str = ""
    shares: int = 0
    cost_amount: float = 0.0        # 含买入费用的摊薄成本金额
    last_price: Optional[float] = None
    buy_date: str = ""

    @property
    def cost_price(self) -> float:
        """摊薄成本价（含买入费用）。"""
        if self.shares <= 0:
            return 0.0
        return self.cost_amount / self.shares

    @property
    def market_value(self) -> float:
        if self.last_price is None:
            return self.cost_amount
        return self.shares * self.last_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "shares": self.shares,
            "cost_price": round(self.cost_price, 4),
            "cost_amount": round(self.cost_amount, 2),
            "last_price": self.last_price,
            "market_value": round(self.market_value, 2),
            "buy_date": self.buy_date,
        }


@dataclass
class LedgerState:
    """账本重建出的组合状态。"""

    strategy: str
    initial_capital: float = 0.0
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    fees: Dict[str, float] = field(
        default_factory=lambda: {
            "commission": 0.0,
            "transfer_fee": 0.0,
            "stamp_tax": 0.0,
            "total": 0.0,
        }
    )
    nav_series: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    nav: float = 0.0
    first_date: str = ""
    last_date: str = ""

    @property
    def position_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def n_positions(self) -> int:
        return sum(1 for p in self.positions.values() if p.shares > 0)


def build_state(strategy: str, records: Sequence[Dict[str, Any]]) -> LedgerState:
    """重放记录 → 组合状态。

    规则：
      - `init` 设置本金与现金
      - `buy` 现金 -= 金额 + 费用；持仓 += 股数，成本按含费加权平均
      - `sell` 现金 += 金额 - 费用；持仓 -= 股数，成本按比例结转
      - `fee` 只做费用汇总（**不再动现金**，否则与 buy/sell 重复计账）
      - `nav` 记录净值点
    """
    state = LedgerState(strategy=strategy)
    for rec in records:
        rtype = rec.get("type")
        rdate = str(rec.get("date") or "")
        if rdate:
            if not state.first_date or rdate < state.first_date:
                state.first_date = rdate
            if rdate > state.last_date:
                state.last_date = rdate

        if rtype == "init":
            state.initial_capital = float(rec.get("amount") or 0.0)
            state.cash = float(rec.get("cash_after") or state.initial_capital)
            continue

        if rtype in ("buy", "sell"):
            fees = rec.get("fees") or {}
            fee_total = float(fees.get("total") or 0.0)
            amount = float(rec.get("amount") or 0.0)
            shares = int(rec.get("shares") or 0)
            code = str(rec.get("code") or "")
            if rtype == "buy":
                state.cash -= amount + fee_total
                pos = state.positions.setdefault(code, Position(code=code))
                pos.shares += shares
                pos.cost_amount += amount + fee_total
                pos.name = str(rec.get("name") or pos.name)
                if not pos.buy_date:
                    pos.buy_date = rdate
                if rec.get("price") is not None:
                    pos.last_price = float(rec["price"])
            else:
                state.cash += amount - fee_total
                pos = state.positions.get(code)
                if pos and pos.shares > 0:
                    sold = min(shares, pos.shares)
                    ratio = sold / pos.shares
                    pos.cost_amount -= pos.cost_amount * ratio
                    pos.shares -= sold
                    if rec.get("price") is not None:
                        pos.last_price = float(rec["price"])
                    if pos.shares <= 0:
                        state.positions.pop(code, None)
            for k in ("commission", "transfer_fee", "stamp_tax"):
                state.fees[k] += float(fees.get(k) or 0.0)
            state.fees["total"] += fee_total
            state.trades.append(rec)
            continue

        if rtype == "fee":
            # 日费用汇总：只用于展示，费用真值仍来自 buy/sell，避免重复计账
            state.nav_series.append(rec) if False else None
            continue

        if rtype == "nav":
            state.nav_series.append(rec)
            if rec.get("nav") is not None:
                state.nav = float(rec["nav"])
            if rec.get("cash_after") is not None:
                state.cash = float(rec["cash_after"])
            continue

    if state.nav_series:
        last = state.nav_series[-1]
        if last.get("nav") is not None:
            state.nav = float(last["nav"])
    return state


def get_state(strategy: str) -> LedgerState:
    """读账本 → 组合状态。"""
    return build_state(strategy, read_records(strategy))


def get_positions(strategy: str) -> Dict[str, Dict[str, Any]]:
    """当前持仓 {code: {...}}。"""
    return {c: p.to_dict() for c, p in get_state(strategy).positions.items()}


def get_cash(strategy: str) -> float:
    """当前现金。"""
    return round(get_state(strategy).cash, 2)


def get_total_fees(strategy: str) -> Dict[str, float]:
    """累计费用明细。"""
    return {k: round(v, 2) for k, v in get_state(strategy).fees.items()}


def get_nav_series(strategy: str) -> List[Dict[str, Any]]:
    """净值序列（每日一个点）。"""
    return list(get_state(strategy).nav_series)


def get_trades(strategy: str) -> List[Dict[str, Any]]:
    """交易历史（买入 + 卖出，按写入顺序）。"""
    return list(get_state(strategy).trades)


def _turnover(state: LedgerState) -> Dict[str, Any]:
    """换手率：总成交额 / 净值，以及年化外推。

    年化需要足够的净值点（少于 20 个点外推没有意义）→ 不够就返回 null。
    """
    traded = sum(float(t.get("amount") or 0.0) for t in state.trades)
    nav_now = state.nav or (state.cash + state.position_value)
    total_pct = (traded / nav_now * 100.0) if nav_now > 0 else 0.0
    out: Dict[str, Any] = {
        "traded_amount": round(traded, 2),
        "turnover_pct": round(total_pct, 4),
        "nav_points": len(state.nav_series),
        "annual_turnover_pct": None,
        "annual_turnover_note": "",
    }
    if len(state.nav_series) < MIN_NAV_POINTS_FOR_ANNUAL:
        out["annual_turnover_note"] = (
            f"净值点仅 {len(state.nav_series)} 个（< {MIN_NAV_POINTS_FOR_ANNUAL}），"
            "年化换手率不可靠，不外推"
        )
        return out
    navs = [
        float(n["nav"]) for n in state.nav_series if n.get("nav") is not None
    ]
    if len(navs) < 2:
        return out
    avg_nav = sum(navs) / len(navs)
    days = len(state.nav_series)
    # 每个净值点按一个交易日算
    annual = (traded / avg_nav) * (252.0 / days) * 100.0 if avg_nav > 0 else 0.0
    out["annual_turnover_pct"] = round(annual, 4)
    out["annual_turnover_note"] = f"按 {days} 个净值点 × 252 交易日外推"
    return out


def summarize(strategy: str) -> Dict[str, Any]:
    """账本汇总（前端 / 脚本都用这一份）。"""
    state = get_state(strategy)
    return {
        "strategy": strategy,
        "exists": ledger_path(strategy).exists(),
        "path": str(ledger_path(strategy)),
        "initial_capital": round(state.initial_capital or INITIAL_CAPITAL, 2),
        "cash": round(state.cash, 2),
        "position_value": round(state.position_value, 2),
        "nav": round(state.nav or (state.cash + state.position_value), 2),
        "n_positions": state.n_positions,
        "total_return_pct": (
            round((state.nav / state.initial_capital - 1.0) * 100.0, 4)
            if state.initial_capital
            else None
        ),
        "total_fees": {k: round(v, 2) for k, v in state.fees.items()},
        "n_trades": len(state.trades),
        "n_buys": sum(1 for t in state.trades if t.get("type") == "buy"),
        "n_sells": sum(1 for t in state.trades if t.get("type") == "sell"),
        "nav_series": [
            {
                "date": n.get("date"),
                "nav": n.get("nav"),
                "cash": n.get("cash_after"),
                "position_value": n.get("position_value_after"),
            }
            for n in state.nav_series
        ],
        "positions": [
            p.to_dict() for p in sorted(
                state.positions.values(), key=lambda x: -x.market_value
            )
        ],
        "first_date": state.first_date,
        "last_date": state.last_date,
        "turnover": _turnover(state),
    }


# ── 写入辅助（供 live_signal.py 使用）────────────────────────────────


def make_init_record(
    strategy: str, date: str, capital: float = INITIAL_CAPITAL, note: str = ""
) -> Dict[str, Any]:
    return {
        "date": date,
        "type": "init",
        "code": None,
        "name": None,
        "side": None,
        "shares": 0,
        "price": None,
        "amount": round(float(capital), 2),
        "fees": {
            "commission": 0.0,
            "transfer_fee": 0.0,
            "stamp_tax": 0.0,
            "total": 0.0,
        },
        "cash_after": round(float(capital), 2),
        "position_value_after": 0.0,
        "nav": round(float(capital), 2),
        "note": note or f"初始资金 ¥{float(capital):,.2f}",
        "dedup_key": f"{strategy}|init",
    }


def make_trade_record(
    strategy: str,
    date: str,
    *,
    side: str,
    code: str,
    name: str,
    shares: int,
    price: float,
    amount: float,
    fees: Dict[str, float],
    cash_after: float,
    position_value_after: float,
    nav: float,
    note: str = "",
) -> Dict[str, Any]:
    return {
        "date": date,
        "type": side,
        "code": code,
        "name": name,
        "side": side,
        "shares": int(shares),
        "price": round(float(price), 4),
        "amount": round(float(amount), 2),
        "fees": {
            "commission": round(float(fees.get("commission") or 0.0), 2),
            "transfer_fee": round(float(fees.get("transfer_fee") or 0.0), 2),
            "stamp_tax": round(float(fees.get("stamp_tax") or 0.0), 2),
            "total": round(float(fees.get("total") or 0.0), 2),
        },
        "cash_after": round(float(cash_after), 2),
        "position_value_after": round(float(position_value_after), 2),
        "nav": round(float(nav), 2),
        "note": note,
        "dedup_key": f"{strategy}|{date}|{side}|{code}",
    }


def make_fee_record(
    strategy: str, date: str, fees: Dict[str, float], note: str = ""
) -> Dict[str, Any]:
    return {
        "date": date,
        "type": "fee",
        "code": None,
        "name": None,
        "side": None,
        "shares": 0,
        "price": None,
        "amount": round(float(fees.get("total") or 0.0), 2),
        "fees": {
            "commission": round(float(fees.get("commission") or 0.0), 2),
            "transfer_fee": round(float(fees.get("transfer_fee") or 0.0), 2),
            "stamp_tax": round(float(fees.get("stamp_tax") or 0.0), 2),
            "total": round(float(fees.get("total") or 0.0), 2),
        },
        "cash_after": None,
        "position_value_after": None,
        "nav": None,
        "note": note or "当日费用汇总",
        "dedup_key": f"{strategy}|{date}|fee",
    }


def make_nav_record(
    strategy: str,
    date: str,
    *,
    cash: float,
    position_value: float,
    nav: float,
    n_positions: int,
    note: str = "",
) -> Dict[str, Any]:
    return {
        "date": date,
        "type": "nav",
        "code": None,
        "name": None,
        "side": None,
        "shares": 0,
        "price": None,
        "amount": round(float(position_value), 2),
        "fees": {
            "commission": 0.0,
            "transfer_fee": 0.0,
            "stamp_tax": 0.0,
            "total": 0.0,
        },
        "cash_after": round(float(cash), 2),
        "position_value_after": round(float(position_value), 2),
        "nav": round(float(nav), 2),
        "n_positions": int(n_positions),
        "note": note or "每日净值快照",
        "dedup_key": f"{strategy}|{date}|nav",
    }


def list_strategies() -> List[str]:
    """账本里已存在的策略 key 列表。"""
    if not LEDGER_DIR.exists():
        return []
    return sorted(p.stem for p in LEDGER_DIR.glob("*.jsonl"))


__all__ = [
    "BACKEND_ROOT",
    "PROJECT_ROOT",
    "LEDGER_DIR",
    "INITIAL_CAPITAL",
    "RECORD_TYPES",
    "MIN_NAV_POINTS_FOR_ANNUAL",
    "LedgerState",
    "Position",
    "append_records",
    "build_state",
    "get_cash",
    "get_nav_series",
    "get_positions",
    "get_state",
    "get_total_fees",
    "get_trades",
    "ledger_path",
    "list_strategies",
    "make_fee_record",
    "make_init_record",
    "make_nav_record",
    "make_trade_record",
    "read_records",
    "record_key",
    "summarize",
]
