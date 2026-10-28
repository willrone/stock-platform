"""执行计划计算（信号 + 账本状态 → 可照着下单的清单）—— 2026-10-27

## 解决什么

`backend/scripts/daily_signal.py` 是研究脚本，只给「代码 + 分数」，用户没法照着下单：
  1. 它假定「持仓 = 昨天的 top30」，不是从账本读真实持仓；
  2. 没有股数 / 金额；
  3. 没有交易成本；
  4. 没有账本、没有执行记录。

本模块把「策略信号」升级成「**执行计划**」：

    SignalSnapshot（策略想持有什么、想卖什么）
      + LedgerView（账本里真实持有什么、现金多少）
      + Quotes（真实行情价 + 当日涨跌幅）
      → ExecutionPlan（每一笔：数量 / 单价 / 金额 / 费用 / 成本价 / 理由）

## 三条硬规则

1. **持仓感知**：卖出清单里的股票必须是**账本里真实持有**的。信号说卖但账本没持有
   → 不出现在卖出清单里（记在 `skipped` 里说明原因）。反过来，账本里持有但已经不在
   策略目标组合里 → 会进卖出清单（这叫「对齐」，否则组合会越漂越歪）。
2. **股数必须是 100 股整数倍**（A 股最小交易单位）。算出来不足 1 手 → 不买，
   在 `skipped` 里写明「不足一手」。
3. **可复现**：同样的输入（信号 + 账本 + 行情 + 配置）必须得到**逐字节相同**的计划。
   → 不使用随机数、不使用当前时间做决策、遍历顺序全部显式排序、浮点按 2 位落盘。
   时间只出现在账本记录（`ts`）与 CLI 抬头，**不进入计划内容**。

## 费率（可配置，常量写明来源）

| 项目 | 默认值 | 来源 / 说明 |
|------|--------|-------------|
| 佣金 | 0.025%，最低 5 元 | A 股券商常见「万 2.5」档位；`SP_COMMISSION_RATE` / `SP_MIN_COMMISSION` 覆盖 |
| 印花税 | 0.05%，**卖出单向** | 2023-08-28 起减半为 0.05%；`SP_STAMP_TAX_RATE` 覆盖 |
| 过户费 | 0.001%，**双向** | 沪深两市统一按成交金额 0.001% 双向收取；`SP_TRANSFER_FEE_RATE` 覆盖 |

> 与 `backend/scripts/paper_portfolio.py` 的纸面盘成本模型（佣金万 3 + 印花税万 5 +
> 滑点千 1、无过户费）**刻意不同**：纸面盘是研究口径，本模块是产品口径，
> 费用明细必须能让用户逐项核对。滑点**不建模**（实盘滑点无法预估，
> 已在口径声明里写明）。

## 仓位与风控参数（可配置）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SP_BOARD_LOT` | 100 | 最小交易单位（一手） |
| `SP_MAX_POSITION_PCT` | 0.10 | 单只股票市值上限（占组合净值） |
| `SP_MAX_ORDER_AMOUNT` | 空 | 单笔金额上限（元）；空 = 沿用单股市值上限 |
| `SP_CASH_USE_PCT` | 0.95 | 总买入金额 ≤ 可用现金 × 95%（留手续费余量） |
| `SP_CASH_RESERVE_PCT` | 0.05 | 目标现金预留（用于等权目标权重） |
| `SP_PRICE_MOVE_BUFFER` | 0.005 | 涨跌停缓冲 → 主板 ±9.5%、创业板/科创板 ±19.5%、北交所 ±29.5%、ST ±4.5% |
| `SP_MAX_DAILY_MOVE` | 0.095 | 无法判定板块时的兜底涨跌幅阈值 |

### 精确计算公式

设组合净值为 `V`，可用现金 `C`，第 i 个目标标的的参考价 `p_i`，目标组合有 `n` 只：

```
目标权重        w_i     = (1 - cash_reserve_pct) / n
单只市值上限    cap_i   = V * max_position_pct
单笔金额上限    ord_i   = min(cap_i, max_order_amount)          # max_order_amount 为空时取 cap_i
目标买入金额    b_i     = min( V * w_i , ord_i , 剩余现金预算 )
现金预算        budget  = C * cash_use_pct                       # 买入「金额」口径
买入股数        q_i     = floor( b_i / p_i / board_lot ) * board_lot
买入金额        A_i     = q_i * p_i
买入费用        fee_i   = commission(A_i) + transfer_fee(A_i)
   commission(A) = max(A * commission_rate, min_commission)     # A > 0 时
   transfer_fee(A) = A * transfer_fee_rate                      # 双向
买入现金流出    out_i   = A_i + fee_i
买入成本价      cost_i  = (A_i + fee_i) / q_i                    # 含费摊薄
卖出金额        S_i     = q_i * p_i                              # q_i = 账本真实持股数（可不足 100 股）
卖出费用        sfee_i  = commission(S_i) + transfer_fee(S_i) + stamp_tax(S_i)
   stamp_tax(S) = S * stamp_tax_rate                           # 仅卖出
卖出现金流入    in_i    = S_i - sfee_i
卖出盈亏        pnl_i   = in_i - q_i * 持仓摊薄成本价
```

硬约束（不满足就跳过并写明原因）：
  - `A_i <= budget`（累计）与 `out_i <= C`（逐笔、含手续费）
  - `q_i > 0`（不足一手 → 跳过）
  - 买入：`pct_chg < +limit`（涨停买不进）；卖出：`pct_chg > -limit`（跌停卖不出）
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ── 成本模型（可配置）──────────────────────────────────────────────
#: 佣金费率（默认万 2.5 = 0.025%）
COMMISSION_RATE = float(os.environ.get("SP_COMMISSION_RATE", "0.00025"))
#: 佣金最低收费（元）
MIN_COMMISSION = float(os.environ.get("SP_MIN_COMMISSION", "5"))
#: 印花税（卖出单向，万 5 = 0.05%）
STAMP_TAX_RATE = float(os.environ.get("SP_STAMP_TAX_RATE", "0.0005"))
#: 过户费（双向，十万分之一 = 0.001%）
TRANSFER_FEE_RATE = float(os.environ.get("SP_TRANSFER_FEE_RATE", "0.00001"))

# ── 交易规则（可配置）──────────────────────────────────────────────
#: 最小交易单位（一手）
BOARD_LOT = int(os.environ.get("SP_BOARD_LOT", "100"))
#: 单只股票市值上限（占组合净值）
MAX_POSITION_PCT = float(os.environ.get("SP_MAX_POSITION_PCT", "0.10"))
#: 单笔金额上限（元）；空 = 不额外限制
_MAX_ORDER_AMOUNT_ENV = os.environ.get("SP_MAX_ORDER_AMOUNT", "").strip()
MAX_ORDER_AMOUNT: Optional[float] = (
    float(_MAX_ORDER_AMOUNT_ENV) if _MAX_ORDER_AMOUNT_ENV else None
)
#: 总买入金额 ≤ 可用现金 × 该比例（留手续费余量）
CASH_USE_PCT = float(os.environ.get("SP_CASH_USE_PCT", "0.95"))
#: 目标现金预留（等权目标权重用）
CASH_RESERVE_PCT = float(os.environ.get("SP_CASH_RESERVE_PCT", "0.05"))
#: 涨跌停缓冲（主板 10% - 0.5pp = 9.5%）
PRICE_MOVE_BUFFER = float(os.environ.get("SP_PRICE_MOVE_BUFFER", "0.005"))
#: 判定不出板块时的兜底涨跌幅阈值
MAX_DAILY_MOVE = float(os.environ.get("SP_MAX_DAILY_MOVE", "0.095"))

#: 费用构成
FEE_KEYS = ("commission", "transfer_fee", "stamp_tax", "total")


# ── 费率与配置 ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeeSchedule:
    """费率表（默认取 A 股实际费率，见模块文档）。"""

    commission_rate: float = COMMISSION_RATE
    min_commission: float = MIN_COMMISSION
    stamp_tax_rate: float = STAMP_TAX_RATE
    transfer_fee_rate: float = TRANSFER_FEE_RATE

    def to_dict(self) -> Dict[str, float]:
        return {
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
            "stamp_tax_rate": self.stamp_tax_rate,
            "transfer_fee_rate": self.transfer_fee_rate,
        }

    def commission(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return max(amount * self.commission_rate, self.min_commission)

    def transfer_fee(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return amount * self.transfer_fee_rate

    def stamp_tax(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return amount * self.stamp_tax_rate

    def buy_fees(self, amount: float) -> "FeeBreakdown":
        """买入费用：佣金 + 过户费（无印花税）。"""
        commission = self.commission(amount)
        transfer = self.transfer_fee(amount)
        return FeeBreakdown(
            commission=commission,
            transfer_fee=transfer,
            stamp_tax=0.0,
            total=commission + transfer,
        )

    def sell_fees(self, amount: float) -> "FeeBreakdown":
        """卖出费用：佣金 + 过户费 + 印花税。"""
        commission = self.commission(amount)
        transfer = self.transfer_fee(amount)
        stamp = self.stamp_tax(amount)
        return FeeBreakdown(
            commission=commission,
            transfer_fee=transfer,
            stamp_tax=stamp,
            total=commission + transfer + stamp,
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """执行参数（仓位上限、现金比例、交易规则）。"""

    board_lot: int = BOARD_LOT
    max_position_pct: float = MAX_POSITION_PCT
    max_order_amount: Optional[float] = MAX_ORDER_AMOUNT
    cash_use_pct: float = CASH_USE_PCT
    cash_reserve_pct: float = CASH_RESERVE_PCT
    price_move_buffer: float = PRICE_MOVE_BUFFER
    max_daily_move: float = MAX_DAILY_MOVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "board_lot": self.board_lot,
            "max_position_pct": self.max_position_pct,
            "max_order_amount": self.max_order_amount,
            "cash_use_pct": self.cash_use_pct,
            "cash_reserve_pct": self.cash_reserve_pct,
            "price_move_buffer": self.price_move_buffer,
            "max_daily_move": self.max_daily_move,
        }


@dataclass(frozen=True)
class FeeBreakdown:
    """单笔费用明细。"""

    commission: float = 0.0
    transfer_fee: float = 0.0
    stamp_tax: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "commission": round(self.commission, 2),
            "transfer_fee": round(self.transfer_fee, 2),
            "stamp_tax": round(self.stamp_tax, 2),
            "total": round(self.total, 2),
        }


# ── 信号 / 账本 / 行情 的输入结构 ─────────────────────────────────


@dataclass(frozen=True)
class TargetPick:
    """目标组合里的一只股票。"""

    code: str
    name: str = ""
    price: Optional[float] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "price": self.price,
            "score": self.score,
        }


@dataclass(frozen=True)
class SignalSnapshot:
    """策略信号快照（可序列化 → 可落盘 / 可从文件读回）。

    `target` 是策略**想要持有**的组合；`buy_codes` / `sell_codes` 是信号显式
    建议的买卖（用于 `delta` 模式跟随信号，或用于标注理由）。
    """

    strategy: str
    signal_date: str
    display_name: str = ""
    source: str = ""
    target: Tuple[TargetPick, ...] = ()
    buy_codes: Tuple[str, ...] = ()
    sell_codes: Tuple[str, ...] = ()
    topk: int = 0
    ndrop: int = 0
    blocked_buys: Tuple[Dict[str, Any], ...] = ()
    blocked_sells: Tuple[Dict[str, Any], ...] = ()
    notes: Tuple[str, ...] = ()

    @property
    def target_codes(self) -> Tuple[str, ...]:
        return tuple(p.code for p in self.target)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "signal_date": self.signal_date,
            "display_name": self.display_name,
            "source": self.source,
            "target": [p.to_dict() for p in self.target],
            "buy_codes": list(self.buy_codes),
            "sell_codes": list(self.sell_codes),
            "topk": self.topk,
            "ndrop": self.ndrop,
            "blocked_buys": [dict(b) for b in self.blocked_buys],
            "blocked_sells": [dict(b) for b in self.blocked_sells],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalSnapshot":
        return cls(
            strategy=str(data.get("strategy") or ""),
            signal_date=str(data.get("signal_date") or ""),
            display_name=str(data.get("display_name") or ""),
            source=str(data.get("source") or ""),
            target=tuple(
                TargetPick(
                    code=str(p.get("code") or ""),
                    name=str(p.get("name") or ""),
                    price=(float(p["price"]) if p.get("price") is not None else None),
                    score=(float(p["score"]) if p.get("score") is not None else None),
                )
                for p in (data.get("target") or [])
            ),
            buy_codes=tuple(str(c) for c in (data.get("buy_codes") or [])),
            sell_codes=tuple(str(c) for c in (data.get("sell_codes") or [])),
            topk=int(data.get("topk") or 0),
            ndrop=int(data.get("ndrop") or 0),
            blocked_buys=tuple(dict(b) for b in (data.get("blocked_buys") or [])),
            blocked_sells=tuple(dict(b) for b in (data.get("blocked_sells") or [])),
            notes=tuple(str(n) for n in (data.get("notes") or [])),
        )

    @classmethod
    def from_json(cls, text: str) -> "SignalSnapshot":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class Quote:
    """一只股票在某一天的行情（真实数据）。"""

    code: str
    date: str
    price: float
    prev_close: Optional[float] = None
    pct_chg: Optional[float] = None
    pct_reliable: bool = True
    name: str = ""
    amount: Optional[float] = None
    is_st: bool = False
    trade_status: Optional[int] = None
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "date": self.date,
            "price": self.price,
            "prev_close": self.prev_close,
            "pct_chg": self.pct_chg,
            "pct_reliable": self.pct_reliable,
            "name": self.name,
            "amount": self.amount,
            "is_st": self.is_st,
            "trade_status": self.trade_status,
            "source": self.source,
        }


@dataclass
class LedgerPositionView:
    """账本里的一只持仓（真实持仓感知的数据来源）。"""

    code: str
    name: str = ""
    shares: int = 0
    cost_amount: float = 0.0
    last_price: Optional[float] = None
    buy_date: str = ""

    @property
    def cost_price(self) -> float:
        if self.shares <= 0:
            return 0.0
        return self.cost_amount / self.shares


@dataclass
class LedgerView:
    """账本状态快照（现金 + 真实持仓）。"""

    strategy: str = ""
    cash: float = 0.0
    initial_capital: float = 0.0
    positions: Dict[str, LedgerPositionView] = field(default_factory=dict)

    def held_shares(self, code: str) -> int:
        pos = self.positions.get(code)
        return int(pos.shares) if pos else 0

    def market_value(self, quotes: Dict[str, Quote]) -> float:
        total = 0.0
        for code, pos in self.positions.items():
            q = quotes.get(code)
            price = q.price if q is not None else pos.last_price
            total += pos.shares * (price or 0.0)
        return total


# ── 输出结构 ───────────────────────────────────────────────────────


@dataclass
class Order:
    """一笔可执行的委托。"""

    code: str
    name: str
    side: str                    # buy / sell
    shares: int
    price: float
    amount: float
    fees: FeeBreakdown
    total_cost: float            # 买入 = 金额 + 费用；卖出 = 金额 - 费用（净收入）
    cost_price: float            # 含费摊薄成本价
    reason: str = ""
    score: Optional[float] = None
    realized_pnl: Optional[float] = None
    cash_after: float = 0.0
    position_value_after: float = 0.0
    nav_after: float = 0.0
    weight_pct: Optional[float] = None
    price_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "side": self.side,
            "shares": self.shares,
            "price": round(self.price, 4),
            "amount": round(self.amount, 2),
            "fees": self.fees.to_dict(),
            "total_cost": round(self.total_cost, 2),
            "cost_price": round(self.cost_price, 4),
            "reason": self.reason,
            "score": self.score,
            "realized_pnl": (
                round(self.realized_pnl, 2) if self.realized_pnl is not None else None
            ),
            "cash_after": round(self.cash_after, 2),
            "position_value_after": round(self.position_value_after, 2),
            "nav_after": round(self.nav_after, 2),
            "weight_pct": (
                round(self.weight_pct, 4) if self.weight_pct is not None else None
            ),
            "price_source": self.price_source,
        }


@dataclass
class SkippedItem:
    """被跳过的标的 + 原因（用户要知道「信号里有它但为什么没成交」）。"""

    code: str
    name: str
    side: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "side": self.side,
            "reason": self.reason,
        }


@dataclass
class ExecutionPlan:
    """完整执行计划。"""

    strategy: str
    signal_date: str
    mode: str
    display_name: str = ""
    source: str = ""
    buys: List[Order] = field(default_factory=list)
    sells: List[Order] = field(default_factory=list)
    skipped: List[SkippedItem] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    cost_model: Dict[str, Any] = field(default_factory=dict)
    caliber: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "display_name": self.display_name,
            "signal_date": self.signal_date,
            "mode": self.mode,
            "source": self.source,
            "buys": [o.to_dict() for o in self.buys],
            "sells": [o.to_dict() for o in self.sells],
            "skipped": [s.to_dict() for s in self.skipped],
            "summary": self.summary,
            "config": self.config,
            "cost_model": self.cost_model,
            "caliber": list(self.caliber),
            "notes": list(self.notes),
        }

    def all_orders(self) -> List[Order]:
        return list(self.sells) + list(self.buys)


# ── 板块涨跌停幅度 ─────────────────────────────────────────────────

#: 各代码前缀的涨跌停幅度（与 app/services/trading/execution_rules.py 一致）
_LIMIT_BY_PREFIX: Tuple[Tuple[str, float], ...] = (
    ("30", 0.20),   # 创业板
    ("68", 0.20),   # 科创板
    ("92", 0.30),   # 北交所
    ("83", 0.30),
    ("87", 0.30),
    ("43", 0.30),
    ("82", 0.30),
    ("60", 0.10),   # 沪主板
    ("00", 0.10),   # 深主板
)

#: ST 股涨跌停幅度
ST_LIMIT_PCT = 0.05


def board_limit_pct(code: str, is_st: bool = False) -> float:
    """该股的涨跌停幅度（主板 10% / 创业板科创板 20% / 北交所 30% / ST 5%）。"""
    if is_st:
        return ST_LIMIT_PCT
    prefix = str(code)[:2]
    for pfx, val in _LIMIT_BY_PREFIX:
        if prefix == pfx:
            return val
    return 0.10


def price_move_limit(code: str, is_st: bool, config: ExecutionConfig) -> float:
    """可成交的涨跌幅阈值（正数）。

    主板 10% - 0.5pp = 9.5%；创业板/科创板 20% - 0.5pp = 19.5%；
    北交所 30% - 0.5pp = 29.5%；ST 5% - 0.5pp = 4.5%。
    """
    board = board_limit_pct(code, is_st)
    limit = board - config.price_move_buffer
    # 兜底：判定不出板块时用 max_daily_move
    if limit <= 0:
        limit = config.max_daily_move
    return limit


# ── 计划生成 ───────────────────────────────────────────────────────


def _round2(x: float) -> float:
    return round(x + 0.0, 2)


def _lot_floor(shares: float, lot: int) -> int:
    """向下取整到整手。"""
    if lot <= 0:
        return int(shares)
    return int(math.floor(shares / lot)) * lot


def _pct_of(quote: Optional[Quote], fallback_price: Optional[float]) -> Optional[float]:
    """取当日涨跌幅；没有可靠涨跌幅时用价格自比（占位，调用方需判可靠位）。"""
    if quote is None:
        return None
    if quote.pct_chg is not None:
        return float(quote.pct_chg)
    if quote.prev_close and quote.prev_close > 0:
        return quote.price / quote.prev_close - 1.0
    return None


def build_execution_plan(
    snapshot: SignalSnapshot,
    ledger: LedgerView,
    quotes: Dict[str, Quote],
    *,
    fees: Optional[FeeSchedule] = None,
    config: Optional[ExecutionConfig] = None,
    mode: str = "delta",
) -> ExecutionPlan:
    """生成执行计划（**纯函数**：同样输入 → 同样输出）。

    Args:
        snapshot: 策略信号快照（目标组合 + 建议买卖）
        ledger: 账本状态（真实持仓 + 现金）
        quotes: 真实行情（价格 + 当日涨跌幅），键为股票代码
        fees: 费率表
        config: 执行参数
        mode:
          - ``"delta"``：只跟随信号显式的买卖清单（增量调仓，低换手）
          - ``"target"``：把账本对齐到策略目标组合（缺失的补齐、多的卖出）
    """
    fees = fees or FeeSchedule()
    config = config or ExecutionConfig()
    if mode not in ("delta", "target"):
        raise ValueError(f"未知 mode: {mode!r}（支持 delta / target）")

    target_codes = list(snapshot.target_codes)
    target_set = set(target_codes)
    target_by_code = {p.code: p for p in snapshot.target}

    cash = float(ledger.cash)
    initial_capital = float(ledger.initial_capital) or 0.0

    # ── 组合净值（用真实行情价估值；缺价用账本最后成交价）──────────
    position_value = 0.0
    for code, pos in ledger.positions.items():
        if pos.shares <= 0:
            continue
        q = quotes.get(code)
        price = q.price if q is not None else (pos.last_price or 0.0)
        position_value += pos.shares * price
    nav = cash + position_value

    plan = ExecutionPlan(
        strategy=snapshot.strategy,
        signal_date=snapshot.signal_date,
        mode=mode,
        display_name=snapshot.display_name,
        source=snapshot.source,
    )

    # ══ 卖出 ══════════════════════════════════════════════════════
    # 卖出候选 = 信号明确要卖的 ∩ 账本真实持有
    #          ∪ (账本持有但已不在目标组合里)   ← 对齐，防组合漂移
    sell_candidates: List[str] = []
    seen: set[str] = set()
    for code in sorted(set(snapshot.sell_codes)):
        if code not in seen:
            seen.add(code)
            sell_candidates.append(code)
    for code in sorted(ledger.positions):
        if code in target_set or code in seen:
            continue
        if ledger.positions[code].shares <= 0:
            continue
        seen.add(code)
        sell_candidates.append(code)

    cash_running = cash
    # 卖出后的持仓市值（用于净值 / 仓位占比）
    remaining_shares: Dict[str, int] = {
        c: p.shares for c, p in ledger.positions.items()
    }
    held_value: Dict[str, float] = {}
    for code, pos in ledger.positions.items():
        if pos.shares <= 0:
            continue
        q = quotes.get(code)
        price = q.price if q is not None else (pos.last_price or 0.0)
        held_value[code] = pos.shares * price

    for code in sell_candidates:
        pos = ledger.positions.get(code)
        pick = target_by_code.get(code)
        name = (pos.name if pos and pos.name else "") or (pick.name if pick else "")
        if pos is None or pos.shares <= 0:
            plan.skipped.append(
                SkippedItem(
                    code=code,
                    name=name,
                    side="sell",
                    reason="账本无真实持仓，无需卖出",
                )
            )
            continue

        q = quotes.get(code)
        if q is None or not q.price or q.price <= 0:
            plan.skipped.append(
                SkippedItem(
                    code=code,
                    name=name,
                    side="sell",
                    reason="无当日行情价格，无法定价",
                )
            )
            continue

        pct = _pct_of(q, None)
        limit = price_move_limit(code, bool(q.is_st), config)
        if pct is not None and pct <= -limit:
            plan.skipped.append(
                SkippedItem(
                    code=code,
                    name=name,
                    side="sell",
                    reason=f"跌停（{pct*100:.1f}%，阈值 -{limit*100:.1f}%），卖不出",
                )
            )
            continue

        shares = int(pos.shares)
        amount = shares * q.price
        fee = fees.sell_fees(amount)
        net = amount - fee.total
        cash_running += net
        remaining_shares[code] = 0
        held_value[code] = 0.0

        cost_price = pos.cost_price
        pnl = net - shares * cost_price
        reason_bits = []
        if code in set(snapshot.sell_codes):
            reason_bits.append("信号建议卖出")
        else:
            reason_bits.append("已不在策略目标组合，回收到目标")
        reason_bits.append(f"账本持股 {shares} 股全部卖出")

        plan.sells.append(
            Order(
                code=code,
                name=name,
                side="sell",
                shares=shares,
                price=q.price,
                amount=amount,
                fees=fee,
                total_cost=net,
                cost_price=cost_price,
                reason="；".join(reason_bits),
                realized_pnl=pnl,
                cash_after=cash_running,
                price_source=q.source or "quote",
            )
        )

    # ══ 买入 ══════════════════════════════════════════════════════
    n_target = len(target_codes)
    target_weight = (
        (1.0 - config.cash_reserve_pct) / n_target if n_target > 0 else 0.0
    )
    buy_budget_total = cash_running * config.cash_use_pct
    budget_remaining = buy_budget_total

    # 买入候选：delta 模式 = 信号显式买入；target 模式 = 目标组合里没持有的
    if mode == "delta":
        candidates = [c for c in sorted(set(snapshot.buy_codes)) if c not in target_set or True]
    else:
        candidates = [c for c in target_codes if c not in ledger.positions]

    for code in candidates:
        pick = target_by_code.get(code)
        name = (pick.name if pick and pick.name else "") or ""
        q = quotes.get(code)

        if q is None or not q.price or q.price <= 0:
            plan.skipped.append(
                SkippedItem(
                    code=code,
                    name=name,
                    side="buy",
                    reason="无当日行情价格，无法定价",
                )
            )
            continue

        pct = _pct_of(q, None)
        limit = price_move_limit(code, bool(q.is_st), config)
        if pct is not None and pct >= limit:
            plan.skipped.append(
                SkippedItem(
                    code=code,
                    name=name,
                    side="buy",
                    reason=f"涨停（{pct*100:.1f}%，阈值 +{limit*100:.1f}%），买不进",
                )
            )
            continue

        # 目标金额：等权目标 → 单只市值上限 → 单笔金额上限 → 剩余现金预算
        cap_value = nav * config.max_position_pct
        ord_cap = (
            min(cap_value, config.max_order_amount)
            if config.max_order_amount is not None
            else cap_value
        )
        desired = nav * target_weight if n_target > 0 else cap_value
        budget = min(desired, ord_cap, budget_remaining)

        shares = _lot_floor(budget / q.price, config.board_lot)
        if shares <= 0:
            plan.skipped.append(
                SkippedItem(
                    code=code,
                    name=name,
                    side="buy",
                    reason=(
                        f"不足一手（目标金额 ¥{budget:,.2f} < 1 手 "
                        f"{config.board_lot} 股 × ¥{q.price}）"
                    ),
                )
            )
            continue

        amount = shares * q.price
        fee = fees.buy_fees(amount)
        cash_out = amount + fee.total
        if cash_out > cash_running or amount > budget_remaining + 1e-9:
            # 再缩一手试试
            while shares > 0:
                shares -= config.board_lot
                if shares <= 0:
                    break
                amount = shares * q.price
                fee = fees.buy_fees(amount)
                cash_out = amount + fee.total
                if cash_out <= cash_running and amount <= budget_remaining + 1e-9:
                    break
            if shares <= 0:
                plan.skipped.append(
                    SkippedItem(
                        code=code,
                        name=name,
                        side="buy",
                        reason="可用现金不足，不足一手",
                    )
                )
                continue

        budget_remaining -= amount
        cash_running -= cash_out
        remaining_shares[code] = remaining_shares.get(code, 0) + shares
        held_value[code] = held_value.get(code, 0.0) + amount

        cost_price = (amount + fee.total) / shares
        reason_bits = [f"目标权重 {target_weight*100:.2f}%"]
        if code in set(snapshot.buy_codes):
            reason_bits.append("信号建议买入")
        else:
            reason_bits.append("目标组合缺失，补齐")
        reason_bits.append(
            f"¥{q.price} × {shares} 股，费用 ¥{fee.total:.2f}，"
            f"成本价 ¥{cost_price:.4f}"
        )

        plan.buys.append(
            Order(
                code=code,
                name=name,
                side="buy",
                shares=shares,
                price=q.price,
                amount=amount,
                fees=fee,
                total_cost=cash_out,
                cost_price=cost_price,
                reason="；".join(reason_bits),
                score=(pick.score if pick else None),
                cash_after=cash_running,
                price_source=q.source or "quote",
            )
        )

    # ══ 汇总 ══════════════════════════════════════════════════════
    total_buy_amount = sum(o.amount for o in plan.buys)
    total_sell_amount = sum(o.amount for o in plan.sells)
    total_fees = sum(o.fees.total for o in plan.all_orders())
    buy_fees = sum(o.fees.total for o in plan.buys)
    sell_fees = sum(o.fees.total for o in plan.sells)
    position_value_after = sum(held_value.values())
    nav_after = cash_running + position_value_after

    for o in plan.buys:
        o.position_value_after = position_value_after
        o.nav_after = nav_after
        o.weight_pct = (o.amount / nav_after * 100.0) if nav_after > 0 else 0.0
    for o in plan.sells:
        o.position_value_after = position_value_after
        o.nav_after = nav_after

    plan.summary = {
        "cash_before": _round2(cash),
        "cash_after": _round2(cash_running),
        "position_value_before": _round2(position_value),
        "position_value_after": _round2(position_value_after),
        "nav_before": _round2(nav),
        "nav_after": _round2(nav_after),
        "initial_capital": _round2(initial_capital),
        "n_buys": len(plan.buys),
        "n_sells": len(plan.sells),
        "n_skipped": len(plan.skipped),
        "buy_amount": _round2(total_buy_amount),
        "sell_amount": _round2(total_sell_amount),
        "buy_fees": _round2(buy_fees),
        "sell_fees": _round2(sell_fees),
        "total_fees": _round2(total_fees),
        "net_cash_flow": _round2(total_sell_amount - total_buy_amount - total_fees),
        "buy_budget": _round2(buy_budget_total),
        "buy_budget_used_pct": (
            round(total_buy_amount / buy_budget_total * 100.0, 2)
            if buy_budget_total > 0
            else 0.0
        ),
        "n_positions_before": sum(1 for p in ledger.positions.values() if p.shares > 0),
        "n_positions_after": sum(1 for s in remaining_shares.values() if s > 0),
    }
    plan.config = config.to_dict()
    plan.cost_model = {
        **fees.to_dict(),
        "notes": [
            "佣金：max(金额×费率, 最低佣金)，买卖双向",
            "印花税：金额×费率，仅卖出单向",
            "过户费：金额×费率，买卖双向",
            "成本价 = (金额 + 费用) / 股数（含费摊薄）",
            "本模型不含滑点（实盘滑点无法预估）",
        ],
    }
    plan.caliber = [
        f"信号日期 {snapshot.signal_date}；执行计划 mode={mode}",
        "股数按 100 股（1 手）向下取整，不足 1 手不买",
        "买入费用 = 佣金 + 过户费；卖出费用 = 佣金 + 过户费 + 印花税",
        f"单只股票市值 ≤ 组合净值 × {config.max_position_pct*100:.0f}%",
        f"总买入金额 ≤ 可用现金 × {config.cash_use_pct*100:.0f}%（另留 5% 手续费余量）",
        f"涨跌停缓冲 {config.price_move_buffer*100:.1f}pp：主板 ±9.5%、创业板/科创板 ±19.5%、"
        "北交所 ±29.5%、ST ±4.5%",
        "卖出清单只包含账本真实持有的股票",
        "价格与涨跌幅来自真实行情（未复权收盘价 / 交易所口径涨跌幅）",
        "不含滑点",
    ]
    plan.notes = list(snapshot.notes)
    return plan


# ── Markdown 清单 ──────────────────────────────────────────────────


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:,.2f}"


def render_markdown(plan: ExecutionPlan) -> str:
    """把执行计划渲染成人能照着下单的 Markdown 清单。"""
    lines: List[str] = []
    lines.append(f"# 执行计划 · {plan.display_name or plan.strategy}")
    lines.append("")
    lines.append(
        f"- 策略：`{plan.strategy}`　信号日期：**{plan.signal_date}**　模式：`{plan.mode}`"
    )
    if plan.source:
        lines.append(f"- 信号来源：{plan.source}")
    s = plan.summary
    lines.append(
        f"- 组合净值：¥{_fmt_money(s.get('nav_before'))} → ¥{_fmt_money(s.get('nav_after'))}"
        f"　现金：¥{_fmt_money(s.get('cash_before'))} → ¥{_fmt_money(s.get('cash_after'))}"
    )
    lines.append(
        f"- 买入 {s.get('n_buys', 0)} 笔 / 卖出 {s.get('n_sells', 0)} 笔 / "
        f"跳过 {s.get('n_skipped', 0)} 项　"
        f"买入金额 ¥{_fmt_money(s.get('buy_amount'))}　"
        f"卖出金额 ¥{_fmt_money(s.get('sell_amount'))}　"
        f"费用合计 ¥{_fmt_money(s.get('total_fees'))}"
    )
    lines.append("")

    if plan.sells:
        lines.append("## 卖出")
        lines.append("")
        lines.append(
            "| 代码 | 名称 | 方向 | 数量 | 单价 | 金额 | 费用 | 成本价 | 盈亏 | 理由 |"
        )
        lines.append("|------|------|------|------|------|------|------|------|------|------|")
        for o in plan.sells:
            lines.append(
                f"| {o.code} | {o.name} | 卖出 | {o.shares} | {o.price:.3f} | "
                f"{_fmt_money(o.amount)} | {_fmt_money(o.fees.total)} | "
                f"{o.cost_price:.4f} | {_fmt_money(o.realized_pnl)} | {o.reason} |"
            )
        lines.append("")

    if plan.buys:
        lines.append("## 买入")
        lines.append("")
        lines.append(
            "| 代码 | 名称 | 方向 | 数量 | 单价 | 金额 | 费用 | 成本价 | 理由 |"
        )
        lines.append("|------|------|------|------|------|------|------|------|------|")
        for o in plan.buys:
            lines.append(
                f"| {o.code} | {o.name} | 买入 | {o.shares} | {o.price:.3f} | "
                f"{_fmt_money(o.amount)} | {_fmt_money(o.fees.total)} | "
                f"{o.cost_price:.4f} | {o.reason} |"
            )
        lines.append("")

    if plan.skipped:
        lines.append("## 跳过（信号里有，但没成交）")
        lines.append("")
        lines.append("| 代码 | 名称 | 方向 | 原因 |")
        lines.append("|------|------|------|------|")
        for it in plan.skipped:
            lines.append(f"| {it.code} | {it.name} | {it.side} | {it.reason} |")
        lines.append("")

    if plan.caliber:
        lines.append("## 口径说明")
        lines.append("")
        for c in plan.caliber:
            lines.append(f"- {c}")
        lines.append("")

    if plan.notes:
        lines.append("## 备注")
        lines.append("")
        for n in plan.notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "BOARD_LOT",
    "CASH_RESERVE_PCT",
    "CASH_USE_PCT",
    "COMMISSION_RATE",
    "MAX_DAILY_MOVE",
    "MAX_ORDER_AMOUNT",
    "MAX_POSITION_PCT",
    "MIN_COMMISSION",
    "PRICE_MOVE_BUFFER",
    "STAMP_TAX_RATE",
    "TRANSFER_FEE_RATE",
    "ExecutionConfig",
    "ExecutionPlan",
    "FeeBreakdown",
    "FeeSchedule",
    "LedgerPositionView",
    "LedgerView",
    "Order",
    "Quote",
    "SignalSnapshot",
    "SkippedItem",
    "TargetPick",
    "board_limit_pct",
    "build_execution_plan",
    "price_move_limit",
    "render_markdown",
]
