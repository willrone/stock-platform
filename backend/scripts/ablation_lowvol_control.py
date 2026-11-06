#!/usr/bin/env python
"""低波动对照实验（ablation）：honest_42f 的超额是选股，还是「买最安静的票」这个 beta？

判据（2026-11-06 写死，不许事后调）：
    honest_42f − 低波动对照C 的 3 年夏普差 ≥ +0.15  或  收益差 ≥ +3pp
      → 因子复合有真增量，继续投 42 因子
    两个都不到
      → 因子复合没有增量，A 股腿应被「低波动组合」替代

每条线共用同一宇宙、同一窗口、同一撮合器、同一成本、同一可成交性闸门，
差别只在打分函数。因此差值就是「打分函数的信息量」。

用法（一条进程一条线，跑完即释放内存；串行由外层 shell 循环控制）：
    backend/.venv-py313/bin/python3 scripts/ablation_lowvol_control.py --line lowvol_ctrl
    backend/.venv-py313/bin/python3 scripts/ablation_lowvol_control.py --line all      # 单进程跑全部（内存充裕时用）
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from datetime import datetime

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies.control_strategies import (  # noqa: E402
    RandomTopKControlStrategy,
)
from app.services.backtest.strategies.strategy_factory import StrategyFactory  # noqa: E402

# --lag 1 = 真实盘口径（周五收盘出信号，下一交易日收盘成交），与纸面盘一致；
# --lag 0 = 当日分数当日收盘成交（引擎旧默认，实盘做不到）。由 main 设定。
LAG = 0
# 这些策略没有 lag 实现 —— 传 lag=1 会被静默忽略，必须直接拒绝，不许挂错口径
_STRATEGIES_WITHOUT_LAG = {"low_volatility"}

START = datetime(2023, 5, 12)
END = datetime(2026, 5, 8)
TOPK, NDROP = 60, 5
CASH_RESERVE = 0.05
OUT_DIR = os.path.join(os.path.dirname(BACKEND_ROOT), "data", "strategy_baseline")

# 对照 D 只在本脚本注册，不进生产策略表（生产行为不变）。
StrategyFactory.register_strategy("random_topk_control", RandomTopKControlStrategy)

_WEEKLY = {"topk": TOPK, "n_drop": NDROP, "rebalance_days": "weekly", "rebalance_weekday": 4}

LINES: dict[str, dict] = {
    # 被测对象：纸面盘 honest_42f（42 因子 = 25 公式 + 17 Alpha158 面板）
    "honest_42f": {"strategy": "honest_composite", "config": {**_WEEKLY, "use_panel": True}},
    # 已知更好的 25 因子口径（夏普 0.880 > 42 因子 0.799）
    "honest_25f": {"strategy": "honest_composite", "config": {**_WEEKLY, "use_panel": False}},
    # 对照 C：把 42 个因子全拆掉，只留「过去 60 交易日已实现波动最低」
    "lowvol_ctrl": {"strategy": "factor_ranking", "config": {**_WEEKLY, "weights": {"low_vol": 1.0}}},
    # 参照：动量+低波动+低振幅三因子（现役 factor_ranking 线，改成 topk60）
    "mom_vol_amp": {"strategy": "factor_ranking", "config": {**_WEEKLY}},
    # 对照 D：随机 TopK —— 噪声本身
    "random_ctrl": {"strategy": "random_topk_control", "config": {**_WEEKLY}},
    # 对照 C′：仓库里已注册的 low_volatility 策略（= −波动(63d) 与 −风险调整收益(21d) 两个 z 等权，
    #              不是纯低波动）—— 用来确认「C 的结论」不依赖于我挑了哪个低波动定义
    "lowvol_registered": {"strategy": "low_volatility", "config": {**_WEEKLY}},
    # 现役淘汰候选：42 因子 + 申万一级行业中性（纸面盘 15 个盯市点里 −1.90pp）
    "honest_42f_neu": {"strategy": "honest_composite",
                       "config": {**_WEEKLY, "use_panel": True, "industry_neutral": True}},
}


def list_universe() -> list[str]:
    files = glob.glob(os.path.join(str(settings.DATA_ROOT_PATH), "qlib_data", "features", "day", "*.parquet"))
    return sorted(os.path.basename(f)[: -len(".parquet")].replace("_", ".") for f in files)


def backtest_config() -> BacktestConfig:
    """成本与可成交性口径对齐纸面盘（paper_portfolio.py）：
    佣金 0.0003 / 滑点 0.001 / 印花税 0.0005 只算卖出侧 / 最低佣金 5 元；
    无个股止损止盈（实测紧止损对周频横截面选股净负面）；
    单仓上限显式给成 (1-现金预留)/topk —— 靠默认 0.2 会把 60 只的组合冻结成 9 只（陷阱 21/25）。
    """
    return BacktestConfig(
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        slippage_rate=0.001,
        open_cost=0.0,
        close_cost=0.0005,
        min_cost=5.0,
        max_position_size=(1.0 - CASH_RESERVE) / TOPK,
        cash_reserve_ratio=CASH_RESERVE,
        board_lot_size=100,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        enable_tradability=True,
        participation_rate=0.005,
        min_turnover=5_000_000.0,
        enable_pending_sells=True,
        rebalance_frequency="daily",
        record_portfolio_history=True,
        portfolio_history_stride=1,
        record_positions_in_history=False,
    )


async def run_line(name: str) -> dict:
    spec = LINES[name]
    codes = list_universe()
    print(
        f"[{name}] strategy={spec['strategy']} universe={len(codes)} "
        f"window={START.date()}..{END.date()} topk={TOPK} n_drop={NDROP} weekly(Fri)",
        flush=True,
    )
    cfg = dict(spec["config"])
    if LAG > 0:
        if spec["strategy"] in _STRATEGIES_WITHOUT_LAG:
            raise SystemExit(
                f"[{name}] 策略 {spec['strategy']} 没有 lag 实现，lag={LAG} 会被静默忽略 → 拒绝运行"
            )
        cfg["lag"] = LAG

    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    res = await executor.run_backtest(
        strategy_name=spec["strategy"],
        stock_codes=codes,
        start_date=START,
        end_date=END,
        strategy_config=cfg,
        backtest_config=backtest_config(),
    )
    wall = time.perf_counter() - t0

    def _conv(o):
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(str(type(o)))

    res["__meta__"] = {
        "ablation_line": name,
        "strategy": spec["strategy"],
        "strategy_config": spec["config"],
        "universe_requested": len(codes),
        "window": [START.isoformat(), END.isoformat()],
        "topk": TOPK,
        "n_drop": NDROP,
        "lag": LAG,
        "caliber": "real_execution_lag1" if LAG > 0 else "signal_day_close_fill_lag0",
        "wall_seconds": wall,
        "run_at": datetime.now().isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "" if LAG == 0 else f"_lag{LAG}"
    out = os.path.join(OUT_DIR, f"ablation_2026-11-06_{name}{suffix}.json")
    with open(out, "w") as f:
        json.dump(res, f, default=_conv, ensure_ascii=False)

    for key in ("risk_metrics", "trading_stats", "period"):
        if key in res:
            print(f"[{name}] {key}: {json.dumps(res[key], default=_conv)[:700]}", flush=True)
    perf = res.get("perf_breakdown", {}) or {}
    print(f"[{name}] wall={wall:.1f}s perf={json.dumps(perf, default=_conv)[:300]}", flush=True)
    print(f"[{name}] saved -> {out}", flush=True)
    return res


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True, help="|".join(LINES) + " | all")
    ap.add_argument("--lag", type=int, choices=(0, 1), default=0,
                    help="1 = 真实盘口径（信号日次一交易日成交）；0 = 旧口径（当日收盘成交）")
    args = ap.parse_args()
    global LAG
    LAG = args.lag
    names = list(LINES) if args.line == "all" else [args.line]
    if args.line != "all" and args.line not in LINES:
        raise SystemExit(f"未知 line: {args.line}（可选 {list(LINES)} 或 all）")
    for name in names:
        await run_line(name)


if __name__ == "__main__":
    asyncio.run(main())
