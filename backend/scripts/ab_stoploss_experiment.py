#!/usr/bin/env python3
"""A/B 实验: 给"高胜率"策略注入价格止损, 验证"加强风控能否让表现更好"。

背景:
  基线发现 5 个策略胜率 67-75%, 但盈亏比 0.29-0.49(赢小亏大, 盈亏因子<1 净亏)。
  关键事实: BacktestConfig.stop_loss_pct/take_profit_pct 在引擎全链路零消费
  (data_models 定义 + API 透传, 但 executor/vectorized_loop/trade_modes 不引用),
  基线里那些策略的回撤 40-60% 是裸奔跑出来的。
  本实验在 PortfolioManager.record_portfolio_snapshot 注入"收盘价跌破 avg_cost*(1-sl)"
  即触发全仓卖出, 作为风险控制的代理, 验证胜率/盈亏因子/回撤的联动变化。

设计(每策略 3 变体):
  A: 无止损(基线口径)   B: 止损 5%   C: 止损 3%   D: 止损 5% + 止盈 10%
  宇宙: 基线 500 只清单(统计套利同款, 与全市场口径分开标注)
  其余口径与基线完全一致(费率/仓位/止盈止损默认关/日频)

用法:
  env -u PYTHONPATH .venv-py313/bin/python3 scripts/ab_stoploss_experiment.py
  # 产物: data/strategy_baseline/ab_stoploss_experiment.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
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
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "ab_stoploss_experiment.json")

START, END = datetime(2023, 5, 12), datetime(2026, 5, 8)

# 高胜率组(基线胜率 67-75%) + 对照
TARGETS = ["stochastic", "kdj", "bollinger", "cci", "rsi"]
# 变体: 名称 -> (stop_loss_pct, take_profit_pct), None = 关闭
VARIANTS = [
    ("A_无止损", None, None),
    ("B_止损5%", 0.05, None),
    ("C_止损3%", 0.03, None),
    ("D_止损5%止盈10%", 0.05, 0.10),
]

# 引擎实际使用 PortfolioManagerArray（run_backtest 走数组化路径）, patch 它
_pma = pma_module.PortfolioManagerArray
_ORIG_SNAPSHOT = _pma.record_portfolio_snapshot
_HIT_LOG: list = []


def install_stoploss(sl: float | None, tp: float | None, current: dict) -> None:
    """注入/摘除价格风控。current_prices 由 runner 每日在快照前填充。"""
    if sl is None and tp is None:
        _pma.record_portfolio_snapshot = _ORIG_SNAPSHOT
        return

    def patched(self, date, current_prices):
        hits = []
        for idx in np.nonzero(self.quantities > 0)[0]:
            i = int(idx)
            cost = float(self.avg_costs[i])
            code = self.stock_codes[i]
            price = current_prices.get(code)
            if price is None or cost <= 0:
                continue
            ret = price / cost - 1.0
            if (sl is not None and ret <= -sl) or (tp is not None and ret >= tp):
                sig = TradingSignal(
                    timestamp=date,
                    stock_code=code,
                    signal_type=SignalType.SELL,
                    strength=1.0,
                    price=float(price),
                    reason="price_stop",
                )
                trade, err = self.execute_signal(sig, current_prices)
                hits.append(code if trade is not None else f"{code}(失败:{err})")
        if hits:
            _HIT_LOG.append((str(date.date()), hits[:8]))
        _ORIG_SNAPSHOT(self, date, current_prices)

    _pma.record_portfolio_snapshot = patched


def load_universe() -> list:
    uni_file = os.path.join(DATA_ROOT, "strategy_baseline", "baseline_universe_500.txt")
    codes = []
    with open(uni_file) as f:
        for line in f:
            c = line.strip()
            # 清洗: 去注释行/空行; 保留合法代码格式
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
        stop_loss_pct=0.05,  # 引擎不消费, 仅保持与基线相同的名义值
        take_profit_pct=0.15,
        rebalance_frequency="daily",
        open_cost=0.0,
        close_cost=0.0,
        min_cost=0.0,
    )


def main() -> None:
    codes = load_universe()
    print(f"[universe] {len(codes)} 只 (与基线 500 只清单一致)")
    ex = make_executor()
    results = {}
    for strat in TARGETS:
        results[strat] = {}
        for vname, sl, tp in VARIANTS:
            _HIT_LOG.clear()
            install_stoploss(sl, tp, current={})
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
                m = res  # 指标在 res 顶层（与基线 runner extract_metrics 同口径）
                n_hits = len(_HIT_LOG)
                results[strat][vname] = {
                    "stop_loss": sl,
                    "take_profit": tp,
                    "metrics": {k: m.get(k) for k in (
                        "total_return", "annualized_return", "sharpe_ratio",
                        "max_drawdown", "volatility", "win_rate", "profit_factor",
                        "total_trades", "stocks_traded",
                    )},
                    "stop_trigger_days": n_hits,
                    "wall_s": round(time.perf_counter() - t0, 1),
                }
                r = results[strat][vname]["metrics"]

                def _pct(v):
                    return f"{v:.1%}" if v is not None else "—"

                print(
                    f"[{strat}] {vname}: ret={_pct(r['total_return'])} wr={_pct(r['win_rate'])} "
                    f"pf={r['profit_factor'] if r['profit_factor'] is not None else '—'} "
                    f"sharpe={r['sharpe_ratio'] if r['sharpe_ratio'] is not None else '—'} "
                    f"mdd={_pct(r['max_drawdown'])} trades={r['total_trades']} "
                    f"stopdays={n_hits} wall={results[strat][vname]['wall_s']}s",
                    flush=True,
                )
            finally:
                install_stoploss(None, None, current={})  # 摘除, 防串
            # 每个策略跑完持久化一次, 防中断丢数据
            payload = {
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "window": [str(START), str(END)],
                    "universe_size": len(codes),
                    "note": "500只宇宙(O(N²)同款清单); 口径同基线; 止损=收盘价跌破avg_cost*(1-sl)触发全仓卖出",
                },
                "strategies": results,
            }
            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            with open(OUT_JSON, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"[saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
