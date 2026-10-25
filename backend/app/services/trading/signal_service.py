"""
可执行信号服务（2026-10-25）

把「策略 → 今天该买什么、买多少」这件事变成一个**产品**：
  - 有真实持仓感知（不是"假设持仓 = 昨天 top-K"）
  - 有股数 / 金额 / 成本（可直接照着下单）
  - 有涨跌停 / 停牌 / ST 可成交性检查
  - 成本模型与纸面账本一致（佣金万3 + 印花税万5 + 滑点千1 + 现金预留 5%）

与 `backend/scripts/paper_portfolio.py` 的关系：
  paper_portfolio.py 是**脚本**（跑 cron、维护纸面账本），本模块是**服务**（对外提供
  同样的计算能力给 API / 前端 / MCP / 推送用）。两边的成本模型常量必须一致，
  改一处要改两处 —— 见下方常量与 scripts/paper_portfolio.py 的注释。

对外接口（app/api/v1/trading.py）：
  GET /api/v1/trading/signal?strategy=honest_42f   今日可执行信号
  GET /api/v1/trading/strategies                    可用信号策略列表
  GET /api/v1/trading/performance?strategy=...      纸面业绩（故事证据）
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── 成本模型（必须与 backend/scripts/paper_portfolio.py 一致）──
COMMISSION = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.0005
SLIPPAGE = 0.001
CASH_RESERVE = 0.05
BOARD_LOT = 100
LIMIT_UP = 0.095

# backend/app/services/trading/signal_service.py → 上溯 4 层到项目根
# （trading → services → app → backend → <项目根>）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SIGNAL_DIR = os.path.join(ROOT, "data", "signals")
PAPER_DIR = os.path.join(ROOT, "data", "paper")
FRONTEND_PUBLIC = os.path.join(ROOT, "frontend", "public")


@dataclass
class SignalStrategy:
    """一个可产出信号的策略定义。"""

    key: str
    display_name: str
    desc: str
    topk: int
    ndrop: int
    # 因子源说明（给用户看的可信度依据）
    factor_note: str = ""


# 纸面盘在跑的 4 条线（周频）。key 与 data/paper/<key>_holdings.json 对应。
SIGNAL_STRATEGIES: Dict[str, SignalStrategy] = {
    "honest_42f": SignalStrategy(
        key="honest_42f",
        display_name="42 因子等权复合",
        desc="23 个基础公式（MIN/MAX/STD/CORR/CORD × 5/10/20/30/60）+ 17 个 Alpha158 面板因子，等权 z-score 复合",
        topk=60,
        ndrop=5,
        factor_note="等权复合，剔除内盘 OHLC 因子，涨停日不可买",
    ),
    "rsi_71421": SignalStrategy(
        key="rsi_71421",
        display_name="RSI 7/14/21 均值回归",
        desc="RSI 7/14/21 超卖 + 低波动 + 量价配合，四类因子等权",
        topk=30,
        ndrop=3,
        factor_note="短周期 RSI 抓「刚跌完马上要弹」，A 股超跌反弹来得快",
    ),
    "gbm_voltgt": SignalStrategy(
        key="gbm_voltgt",
        display_name="LightGBM 元模型 + 波动率目标",
        desc="42 因子喂给 LightGBM 预测次日收益，叠加波动率目标仓位（高波动日自动降仓）",
        topk=60,
        ndrop=5,
        factor_note="元模型学权重，波动率目标控制回撤",
    ),
    "factor_ranking": SignalStrategy(
        key="factor_ranking",
        display_name="动量 / 低波动 / 低振幅",
        desc="126 日动量（跳最近 20 日）+ 60 日低波动 + 10 日低振幅，三者等权",
        topk=50,
        ndrop=5,
        factor_note="纯量价因子，不依赖基本面，回撤最小",
    ),
    "22f": SignalStrategy(
        key="22f",
        display_name="22 因子等权（对照线）",
        desc="MIN +1 / STD,CORR,CORD -1 / MAX30,60 +1，剔除 MAX5/10/20",
        topk=30,
        ndrop=3,
        factor_note="2021-2026 滚动 OOS 夏普 0.685，作为对照基线",
    ),
}


def limit_pct(code: str) -> float:
    """该股的涨跌停幅度（用于可成交性检查）。"""
    c2 = code[:2]
    if c2 in ("30", "68"):
        return 0.20
    if c2 in ("92", "83", "87"):
        return 0.30
    return 0.10


def _read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _load_name_map() -> Dict[str, str]:
    """股票代码 → 名称（来自通达信盘后包最新分区）。"""
    raw_dir = os.path.join(ROOT, "data", "tdx_daily", "raw")
    if not os.path.isdir(raw_dir):
        return {}
    dates = sorted(d for d in os.listdir(raw_dir) if d.startswith("date="))
    if not dates:
        return {}
    try:
        df = pd.read_parquet(
            os.path.join(raw_dir, dates[-1], "part.parquet"),
            columns=["ts_code", "name"],
        )
        return dict(zip(df["ts_code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def _paper_path(strategy: str) -> str:
    return os.path.join(PAPER_DIR, f"{strategy}_holdings.json")


def _ledger_path(strategy: str) -> str:
    return os.path.join(PAPER_DIR, f"{strategy}_ledger.csv")


class TradingSignalService:
    """对外提供可执行信号。"""

    def __init__(self) -> None:
        self._name_map: Optional[Dict[str, str]] = None

    # ── 名称映射 ──
    def name_map(self) -> Dict[str, str]:
        if self._name_map is None:
            self._name_map = _load_name_map()
        return self._name_map

    def stock_name(self, code: str) -> str:
        return self.name_map().get(code, "")

    # ── 策略列表 ──
    def list_strategies(self) -> List[Dict[str, Any]]:
        out = []
        for s in SIGNAL_STRATEGIES.values():
            holdings = _read_json(_paper_path(s.key)) or {}
            out.append({
                "key": s.key,
                "display_name": s.display_name,
                "desc": s.desc,
                "factor_note": s.factor_note,
                "topk": s.topk,
                "ndrop": s.ndrop,
                "has_paper": bool(holdings),
                "last_signal": holdings.get("last_signal"),
                "n_positions": len(holdings.get("positions", {}) or {}),
                "last_value": holdings.get("last_value"),
            })
        return out

    # ── 今日可执行信号 ──
    def daily_signal(self, strategy: str) -> Dict[str, Any]:
        """返回某策略的今日信号（带股数 / 金额 / 成本）。

        数据来源：paper_portfolio.py 生成的 portfolio.json（最近一次调仓结果）
        + <strategy>_holdings.json（当前真实持仓）。
        """
        spec = SIGNAL_STRATEGIES.get(strategy)
        if spec is None:
            raise ValueError(
                f"未知策略 {strategy}，可选: {list(SIGNAL_STRATEGIES)}"
            )

        portfolio = _read_json(os.path.join(PAPER_DIR, "portfolio.json")) or {}
        lines = portfolio.get("lines", {}) or {}
        line = lines.get(strategy)
        holdings = _read_json(_paper_path(strategy)) or {}
        positions = holdings.get("positions", {}) or {}

        if line is None and not positions:
            return {
                "strategy": strategy,
                "display_name": spec.display_name,
                "status": "no_data",
                "message": (
                    "尚无信号数据。平台在每个交易日收盘后自动生成信号"
                    "（约 16:10 完成），请稍后再看，或先跑一次 "
                    "backend/scripts/paper_portfolio.py。"
                ),
                "buys": [],
                "sells": [],
                "holds": [],
            }

        buys = []
        for b in (line or {}).get("buys", []) or []:
            code = b.get("code", "")
            buys.append({
                **b,
                "name": b.get("name") or self.stock_name(code),
                "limit_pct": limit_pct(code),
            })

        sells = []
        for s in (line or {}).get("sells", []) or []:
            code = s.get("code", "")
            sells.append({
                **s,
                "name": s.get("name") or self.stock_name(code),
                "limit_pct": limit_pct(code),
            })

        # 持有：补上股数 / 成本 / 市值，外部用户照着看持仓
        holds = []
        for code in (line or {}).get("holds", []) or list(positions.keys()):
            pos = positions.get(code, {})
            shares = int(pos.get("shares", 0) or 0)
            cost = float(pos.get("cost", 0) or 0)
            holds.append({
                "code": code,
                "name": self.stock_name(code),
                "shares": shares,
                "cost": cost,
                "cost_amount": round(shares * cost, 2),
                "buy_date": pos.get("buy_date", ""),
            })
        holds.sort(key=lambda h: -(h.get("cost_amount") or 0))

        cash = float(holdings.get("cash", 0) or 0)
        value = float(holdings.get("last_value", 0) or 0)
        capital = float(holdings.get("initial_capital", 1_000_000) or 1_000_000)

        return {
            "strategy": strategy,
            "display_name": spec.display_name,
            "desc": spec.desc,
            "factor_note": spec.factor_note,
            "status": "ok",
            "date": (line or {}).get("date") or holdings.get("last_signal"),
            "topk": spec.topk,
            "ndrop": spec.ndrop,
            "cash": round(cash, 2),
            "value": round(value, 2),
            "capital": capital,
            "total_return_pct": round((value / capital - 1) * 100, 3) if capital else 0.0,
            "n_positions": len(positions),
            "buys": buys,
            "sells": sells,
            "holds": holds,
            "cost_model": {
                "commission": COMMISSION,
                "stamp_tax": STAMP_TAX,
                "slippage": SLIPPAGE,
                "cash_reserve": CASH_RESERVE,
                "board_lot": BOARD_LOT,
            },
            "fill_assumption": "信号在收盘后生成，按下一交易日开盘/收盘价成交",
        }

    # ── 纸面业绩（故事证据）──
    def performance(self, strategy: str) -> Dict[str, Any]:
        """纸面组合的业绩与交易记录 —— 对外展示真实 track record。"""
        if strategy not in SIGNAL_STRATEGIES:
            raise ValueError(
                f"未知策略 {strategy}，可选: {list(SIGNAL_STRATEGIES)}"
            )
        holdings = _read_json(_paper_path(strategy)) or {}
        positions = holdings.get("positions", {}) or {}
        capital = float(holdings.get("initial_capital", 1_000_000) or 1_000_000)
        value = float(holdings.get("last_value", 0) or 0)
        cash = float(holdings.get("cash", 0) or 0)

        trades = []
        ledger = _ledger_path(strategy)
        if os.path.exists(ledger):
            try:
                with open(ledger) as fh:
                    import csv

                    trades = list(csv.DictReader(fh))
            except Exception:
                trades = []

        buys = [t for t in trades if t.get("action") == "BUY"]
        sells = [t for t in trades if t.get("action") == "SELL"]
        gross_pl = 0.0
        realized = 0.0
        cost_basis: Dict[str, float] = {}
        for t in trades:
            code = t.get("code", "")
            try:
                amount = float(t.get("amount", 0) or 0)
            except (TypeError, ValueError):
                amount = 0.0
            if t.get("action") == "BUY":
                cost_basis[code] = cost_basis.get(code, 0.0) + amount
            elif t.get("action") == "SELL":
                basis = cost_basis.get(code, 0.0)
                if basis > 0:
                    # 按比例结转成本
                    sold_shares = float(t.get("shares", 0) or 0)
                    total_shares = sum(
                        float(x.get("shares", 0) or 0)
                        for x in positions.values()
                    ) + sold_shares
                    share = min(1.0, sold_shares / total_shares) if total_shares else 0
                    realized += amount - basis * share
                    cost_basis[code] = basis * (1 - share)

        market_value = sum(
            int(p.get("shares", 0) or 0) * float(p.get("cost", 0) or 0)
            for p in positions.values()
        )

        return {
            "strategy": strategy,
            "display_name": SIGNAL_STRATEGIES[strategy].display_name,
            "status": "ok" if positions or trades else "no_data",
            "date": holdings.get("last_signal"),
            "capital": capital,
            "value": round(value, 2),
            "cash": round(cash, 2),
            "position_value": round(market_value, 2),
            "total_return_pct": round((value / capital - 1) * 100, 3) if capital else 0.0,
            "n_positions": len(positions),
            "n_trades": len(trades),
            "n_buys": len(buys),
            "n_sells": len(sells),
            "realized_pnl": round(realized, 2),
            "positions": [
                {
                    "code": code,
                    "name": self.stock_name(code),
                    "shares": int(p.get("shares", 0) or 0),
                    "cost": float(p.get("cost", 0) or 0),
                    "buy_date": p.get("buy_date", ""),
                }
                for code, p in positions.items()
            ],
            "recent_trades": trades[-50:][::-1],
        }
