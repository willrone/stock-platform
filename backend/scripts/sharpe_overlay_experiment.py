#!/usr/bin/env python3
"""夏普提升实验: 文献共识的三类"出场无关"杠杆 (全进全出/入场门, 引擎原生支持)。

背景:
  基线 16 策略夏普 0.09~0.92, 回撤 30-75%。止损 A/B 已证伪"出场端"路线
  (高胜率策略加止损 → 胜率 61-69% 崩到 18-45%, 收益反降)。
  引擎约束: _execute_sell 整仓平仓(quantities[idx]=0), 无部分降仓 →
  杠杆只能做"入场门"(拒新仓) 或"全进全出"(清仓+禁买)。

三类文献杠杆(全部出场无关, 针对"在坏行情里待太久"这个真问题):
  1) 个股 MA200 入场门: 价格 < MA200 拒新仓
     (Mebane-Faber 2007: MA 规则砍回撤不伤收益, 200d 最优; arXiv 2504.10914 理论最优~112d)
  2) 市场 MA200 全进全出: 等权宇宙收盘 < 其 MA200 → 清仓+禁买(防御); 回升 → 恢复
     (quantitativo: 投入时间 20%→11.7% 夏普 1.83→2.25; Medium 动量组合同向; 10-month rule)
  3) 波动率开关: 组合 20d 已实现 vol > 阈值 → 清仓+禁买; < 0.7x 阈值 → 恢复
     (Moreira-Muir 2017 / Harvey-Crawford: 股票类夏普 0.40→0.51, 唯一对风险资产显著;
      单账户无杠杆 → 只能"往低砍", 即高波动时离场)

口径: 500 只宇宙(与止损 A/B 同清单) × 2023-05-12→2026-05-08, 费率/仓位同基线。
实现: monkeypatch PortfolioManagerArray(引擎实际执行路径), 零引擎改动, 实验后恢复。
产物: data/strategy_baseline/sharpe_overlay_experiment.json
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import (  # noqa: E402
    BacktestConfig,
    SignalType,
    TradingSignal,
)
from app.services.backtest.core import portfolio_manager_array as pma_module  # noqa: E402

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "sharpe_overlay_experiment.json")

START, END = datetime(2023, 5, 12), datetime(2026, 5, 8)
_MA = 200

TARGETS = ["stochastic", "kdj", "bollinger", "cci", "rsi"]
# 变体: 入场门(bool) / 防御触发("ma200"|"vol"|None)
VARIANTS = [
    ("A_无overlay", dict(entry_ma200=False, defense=None)),
    ("B_个股MA200门", dict(entry_ma200=True, defense=None)),
    ("C_市场MA200全进全出", dict(entry_ma200=False, defense="ma200")),
    ("D_vol开关25%", dict(entry_ma200=False, defense="vol")),
    ("E_个股门+市场开关", dict(entry_ma200=True, defense="ma200")),
]
VOL_TRIGGER = 0.25  # 组合 20d 已实现 vol 阈值(年化); >25% 防御


class Overlay:
    def __init__(self, entry_ma200=False, defense=None):
        self.entry_ma200 = bool(entry_ma200)
        self.defense = defense
        self.buf = {}  # code -> deque of recent closes (maxlen _MA)
        self.universe_buf = deque(maxlen=_MA)
        self.in_defense = False
        self.active = self.entry_ma200 or self.defense is not None
        self.last_instance = None
        self.n_defense_days = 0

    def _ma_hit(self, code):
        b = self.buf.get(code)
        if not b or len(b) < _MA:
            return None
        m = float(np.mean(np.array(list(b))[-_MA:]))
        if m <= 0:
            return None
        return list(b)[-1] < m

    def _universe_ma_hit(self):
        if len(self.universe_buf) < _MA:
            return None
        arr = list(self.universe_buf)
        m = float(np.mean(arr[-_MA:]))
        if m <= 0:
            return None
        return arr[-1] < m


_orig_snapshot = pma_module.PortfolioManagerArray.record_portfolio_snapshot  # type: ignore[assignment]
_orig_buy = pma_module.PortfolioManagerArray._execute_buy  # type: ignore[assignment]


def _portfolio_realized_vol(pm) -> float | None:
    """组合 20d 已实现 vol(年化)。"""
    if len(pm.equity_curve) >= 25:
        vals = np.array([v for _, v in pm.equity_curve[-21:]], dtype=float)
        base = np.where(vals[:-1] > 0, vals[:-1], 1.0)
        rr = np.diff(vals) / base
        v = float(rr.std() * math.sqrt(252))
        if v > 1e-9:
            return v
    return None


def install(o: Overlay) -> None:
    # 无论是否激活都 stash 实例引用(拿 equity_curve 算真实夏普)
    if not o.active:

        def _stash(self, date, current_prices):
            o.last_instance = self
            _orig_snapshot(self, date, current_prices)  # type: ignore[call-arg]

        pma_module.PortfolioManagerArray.record_portfolio_snapshot = _stash  # type: ignore[assignment]
        pma_module.PortfolioManagerArray._execute_buy = _orig_buy  # type: ignore[assignment]
        return

    def patched_buy(self, idx, stock_code, price, original_price, slip, signal, current_prices):
        # 防御态 → 禁买
        if o.in_defense:
            return None, "防御态: 市场风险过高"
        # 个股 MA200 入场门
        if o.entry_ma200:
            b = o.buf.get(stock_code)
            if b is not None and len(b) >= _MA:
                m = float(np.mean(list(b) + [float(signal.price)]))
                if m > 0 and float(signal.price) < m:
                    return None, "MA200入场门: 价格低于200日线"
        return _orig_buy(self, idx, stock_code, price, original_price, slip, signal, current_prices)  # type: ignore[return-value]

    def patched_snapshot(self, date, current_prices):
        # 1) 记录个股价格缓冲
        if o.entry_ma200 or o.defense is not None:
            for code, price in current_prices.items():
                if price and price > 0:
                    b = o.buf.get(code)
                    if b is None:
                        b = o.buf[code] = deque(maxlen=_MA)
                    b.append(float(price))
        # 2) 等权宇宙收盘(市场 MA200 参考)
        if o.defense == "ma200":
            cps = [p for p in current_prices.values() if p and p > 0]
            if cps:
                o.universe_buf.append(float(np.mean(cps)))
        # 3) 防御触发判定
        if o.defense == "ma200" and len(o.universe_buf) >= _MA:
            hit = o._universe_ma_hit()
            if hit is True and not o.in_defense:
                o.in_defense = True
            elif hit is False and o.in_defense:
                o.in_defense = False
        elif o.defense == "vol":
            rv = _portfolio_realized_vol(self)
            if rv is not None:
                if rv > VOL_TRIGGER and not o.in_defense:
                    o.in_defense = True
                elif rv < VOL_TRIGGER * 0.7 and o.in_defense:
                    o.in_defense = False
        # 4) 防御态 → 清仓所有持仓(全进全出)
        if o.in_defense:
            o.n_defense_days += 1
            for i in np.nonzero(self.quantities > 0)[0]:
                qi = int(i)
                code = self.stock_codes[qi]
                price = current_prices.get(code, self.avg_costs[qi])
                if not price or price <= 0:
                    continue
                self.execute_signal(
                    TradingSignal(
                        timestamp=date,
                        stock_code=code,
                        signal_type=SignalType.SELL,
                        strength=1.0,
                        price=float(price),
                        reason="overlay_defense",
                    ),
                    current_prices,
                )
        o.last_instance = self
        _orig_snapshot(self, date, current_prices)  # type: ignore[call-arg]

    pma_module.PortfolioManagerArray._execute_buy = patched_buy  # type: ignore[assignment]
    pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched_snapshot  # type: ignore[assignment]


def _ret_stats(equity):
    if not equity or len(equity) < 30:
        return None
    vals = np.array([v for _, v in equity], dtype=float)
    base = np.where(vals[:-1] > 0, vals[:-1], 1.0)
    rets = np.diff(vals) / base
    years = max(1.0, len(equity) / 252.0)
    total = vals[-1] / vals[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(rets.std() * math.sqrt(252))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(((vals - peak) / peak).min())
    return {
        "total_return": round(total, 6),
        "annualized_return": round(ann, 6),
        "volatility": round(vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 6),
    }


def _yearly(equity):
    by = {}
    for d, v in equity:
        by.setdefault(d.year, []).append(float(v))
    out, prev = {}, None
    for y in sorted(by):
        vals = np.array(by[y])
        if len(vals) < 20:
            continue
        r0 = prev if prev else vals[0]
        out[str(y)] = round(float(vals[-1] / r0 - 1.0), 4)
        prev = float(vals[-1])
    return out


def load_universe() -> list:
    uni_file = os.path.join(DATA_ROOT, "strategy_baseline", "baseline_universe_500.txt")
    codes = []
    with open(uni_file) as f:
        for line in f:
            c = line.strip()
            if not c or c.startswith("#") or "." not in c:
                continue
            if c not in codes:
                codes.append(c)
    return codes


def make_executor() -> BacktestExecutor:
    return BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )


def base_config() -> BacktestConfig:
    return BacktestConfig(
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        max_position_size=0.2,
        cash_reserve_ratio=0.05,
        board_lot_size=100,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        rebalance_frequency="daily",
        open_cost=0.0,
        close_cost=0.0,
        min_cost=0.0,
    )


def main() -> None:
    codes = load_universe()
    print(f"[universe] {len(codes)} 只", flush=True)
    ex = make_executor()
    results = {}
    for strat in TARGETS:
        results[strat] = {}
        for vname, vparams in VARIANTS:
            o = Overlay(**vparams)
            install(o)
            t0 = time.perf_counter()
            try:
                res = asyncio.run(
                    ex.run_backtest(
                        strategy_name=strat,
                        stock_codes=codes,
                        start_date=START,
                        end_date=END,
                        strategy_config={},
                        backtest_config=base_config(),
                    )
                )
                m = res  # 指标在 res 顶层(同基线 extract_metrics 口径)
                eq = getattr(o.last_instance, "equity_curve", None) if o.last_instance else None
                raw = _ret_stats(eq) if eq else None
                results[strat][vname] = {
                    "overlay": vparams,
                    "metrics_reported": {k: m.get(k) for k in (
                        "total_return", "sharpe_ratio", "max_drawdown",
                        "win_rate", "profit_factor", "total_trades",
                    )},
                    "raw_stats": raw,
                    "yearly_return": _yearly(eq) if eq else None,
                    "defense_days": o.n_defense_days,
                    "wall_s": round(time.perf_counter() - t0, 1),
                }
                r = results[strat][vname]
                rs = r["raw_stats"] or {}
                print(
                    f"[{strat}] {vname}: sharpe={rs.get('sharpe_ratio')} ann={rs.get('annualized_return')} "
                    f"mdd={rs.get('max_drawdown')} trades={r['metrics_reported'].get('total_trades')} "
                    f"defense={o.n_defense_days}d wall={r['wall_s']}s",
                    flush=True,
                )
            finally:
                install(Overlay())  # 摘除全部 patch 防串
            payload = {
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "window": [str(START), str(END)],
                    "universe_size": len(codes),
                    "note": "500只宇宙; 口径同基线; overlay=MA200入场门/市场MA200全进全出/vol开关(出场无关三杠杆)",
                },
                "strategies": results,
            }
            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            with open(OUT_JSON, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
