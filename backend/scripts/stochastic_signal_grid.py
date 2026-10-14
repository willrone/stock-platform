#!/usr/bin/env python3
"""策略内部信号质量扫描 (stochastic)。

背景: overlay 四路线(止损/个股门/市场择时/vol开关)全部证伪 —— 外围包装不涨夏普。
stochastic 是现有体系最强腿(夏普 0.77 / 年化 22.6% / 回撤 -36.7%), 本实验改它的
信号本身: 入场/出场阈值 + K 周期, 看能否把"高收益高波动"推向"高收益低波动"。

信号逻辑 (strategies.py StochasticStrategy):
  买 = K<oversold 且 prev_K<oversold 且 K 上穿 D
  卖 = K>overbought 且 prev_K>overbought 且 K 下穿 D

网格 (5 阈值档 × 4 K周期 = 20 变体, 默认 20/80@k14 为基准):
  阈值: 20/80, 15/85, 25/75, 30/70, 35/65
  k_period: 7, 10, 14, 21   (d_period 固定 3)

口径: 500 只宇宙 × 2023-05-12→2026-05-08, 100 万, 费率/仓位同基线, 确定性引擎。
诚实声明: 20 变体单窗口挑选, 最优格存在过拟合风险; 判定标准 =
(a) 最优格是否显著优于默认格 (b) 邻域格是否同向改善(平台 vs 刀锋) (c) 分年稳定性。
产物: data/strategy_baseline/stochastic_signal_grid.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "stochastic_signal_grid.json")
START, END = datetime(2023, 5, 12), datetime(2026, 5, 8)
STRAT = "stochastic"

# (oversold, overbought)
THRESHOLDS = [(20, 80), (15, 85), (25, 75), (30, 70), (35, 65)]
K_PERIODS = [7, 10, 14, 21]


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
    print(f"[universe] {len(codes)} 只, 网格 {len(THRESHOLDS)}x{len(K_PERIODS)}", flush=True)
    ex = make_executor()
    cfg = base_config()

    results = {}
    n_total = len(THRESHOLDS) * len(K_PERIODS)
    n_done = 0
    for os_, ob in THRESHOLDS:
        for kp in K_PERIODS:
            tag = f"ov{os_}_ob{ob}_k{kp}"
            scfg = {"oversold": os_, "overbought": ob, "k_period": kp, "d_period": 3}
            t0 = time.perf_counter()
            try:
                res = asyncio.run(
                    ex.run_backtest(
                        strategy_name=STRAT,
                        stock_codes=codes,
                        start_date=START,
                        end_date=END,
                        strategy_config=scfg,
                        backtest_config=cfg,
                    )
                )
            except Exception as e:  # noqa: BLE001
                print(f"[{tag}] FAILED: {e}", flush=True)
                results[tag] = {"config": scfg, "error": str(e)}
                continue
            # 交易数/胜率从引擎结果取
            trades = res.get("total_trades")
            win = res.get("win_rate")
            n_done += 1
            pf = res.get("profit_factor")
            results[tag] = {
                "config": scfg,
                "trades": trades,
                "win_rate": win,
                "profit_factor": pf,
                "metrics": {k: res.get(k) for k in (
                    "total_return", "sharpe_ratio", "max_drawdown",
                )},
                "wall_s": round(time.perf_counter() - t0, 1),
            }
            m = results[tag]["metrics"]
            print(
                f"[{n_done}/{n_total}] {tag}: ret={m.get('total_return')} "
                f"sharpe={m.get('sharpe_ratio')} mdd={m.get('max_drawdown')} "
                f"trades={trades} win={win} wall={results[tag]['wall_s']}s",
                flush=True,
            )
            payload = {
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "window": [str(START), str(END)],
                    "universe_size": len(codes),
                    "strategy": STRAT,
                    "note": "stochastic 信号质量网格: 阈值×K周期; 确定性引擎",
                },
                "results": results,
            }
            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            with open(OUT_JSON, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
