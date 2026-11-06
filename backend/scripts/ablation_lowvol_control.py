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
from app.services.backtest.universe_gate import gate_report_path  # noqa: E402

# --lag 1 = 真实盘口径（周五收盘出信号，下一交易日收盘成交），与纸面盘一致；
# --lag 0 = 当日分数当日收盘成交（引擎旧默认，实盘做不到）。由 main 设定。
LAG = 0
GATE_ENABLED = True  # 可筛选性闸门；--no-gate 关掉（只为复现旧数字）
TAG = ""  # 产物后缀，区分闸门前后两套数字
CAP_TIER = ""  # ""=不分层；low/mid/high = 按估算自由流通市值分位切宇宙（判"是否 size β"用）
CAP_TIER_STATS: dict = {}  # 分层口径随产物一起落盘，便于事后审计
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
    # 2×2 设计的最后一格：25 因子 + 行业中性 —— 用来判断「行业中性」是否对公式因子也有效
    # （只有 42f_neu 高、25f_neu 不高，才说明中性是与面板因子配套的）
    "honest_25f_neu": {"strategy": "honest_composite",
                       "config": {**_WEEKLY, "use_panel": False, "industry_neutral": True}},
    # 现役淘汰候选：42 因子 + 申万一级行业中性（纸面盘 15 个盯市点里 −1.90pp）
    "honest_42f_neu": {"strategy": "honest_composite",
                       "config": {**_WEEKLY, "use_panel": True, "industry_neutral": True}},
}


def list_universe() -> tuple[list[str], object]:
    """宇宙要过「可筛选性闸门」：风控库(baostock_daily)看不见的票不许进宇宙 —— 查不到 = 排除。
    关掉闸门只为了复现旧数字（--no-gate），正常实验一律带闸门。见 app/services/backtest/universe_gate.py。
    --tier low/mid/high 再按估算自由流通市值切一刀（判"因子收益是不是 size β"），查不到市值的票同样排除。"""
    global CAP_TIER_STATS
    from app.services.backtest.universe_gate import gated_universe

    codes, rep = gated_universe(str(settings.DATA_ROOT_PATH), START, END, enabled=GATE_ENABLED)
    if rep is not None:
        print(f"[闸门] {rep.summary()}", flush=True)
    if CAP_TIER:
        from app.services.backtest.cap_estimate import estimate_caps, filter_tier, tier_stats

        cache = os.path.join(OUT_DIR, "cap_cache_2026-11-06.json")
        caps = estimate_caps(str(settings.DATA_ROOT_PATH), START.strftime("%Y-%m-%d"),
                             END.strftime("%Y-%m-%d"), codes, cache)
        kept = filter_tier(codes, caps, CAP_TIER)
        CAP_TIER_STATS = {"tier": CAP_TIER, **tier_stats(caps), "kept": len(kept)}
        print(f"[分层] tier={CAP_TIER} 宇宙 {len(codes)} → {len(kept)} "
              f"（市值中位 {CAP_TIER_STATS['median_yi']} 亿，q25 {CAP_TIER_STATS['q25_yi']} 亿，"
              f"q75 {CAP_TIER_STATS['q75_yi']} 亿）", flush=True)
        codes = kept
    return codes, rep


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
    codes, gate_rep = list_universe()
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
        "universe_requested": gate_rep.universe_size if gate_rep else len(codes),
        "universe_after_gate": len(codes),
        "coverage_gate": None if gate_rep is None else {
            "min_screen_ratio": gate_rep.min_ratio,
            "dropped": len(gate_rep.dropped),
            "dropped_share_pct": round(gate_rep.dropped_share * 100, 2),
            "report": gate_report_path(str(settings.DATA_ROOT_PATH)),
        },
        "cap_tier": CAP_TIER_STATS or None,
        },
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
    # TAG 让「带闸门 / 不带闸门」两套产物并存，别互相覆盖
    out = os.path.join(OUT_DIR, f"ablation_2026-11-06_{name}{TAG}{suffix}.json")
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
    ap.add_argument("--tag", default="", help="产物文件后缀（必须以 _ 开头），例：_gate")
    ap.add_argument("--tier", default="", choices=("", "low", "mid", "high"),
                    help="按估算自由流通市值切宇宙：low=<q25，mid=中间，high=>q75（判因子收益是否 size β）")
    ap.add_argument("--no-gate", action="store_true",
                    help="关掉可筛选性闸门（只为复现 2026-11-06 之前的旧数字，正常实验别用）")
    ap.add_argument("--lag", type=int, choices=(0, 1), default=0,
                    help="1 = 真实盘口径（信号日次一交易日成交）；0 = 旧口径（当日收盘成交）")
    args = ap.parse_args()
    global LAG, GATE_ENABLED, TAG, CAP_TIER
    LAG = args.lag
    TAG = args.tag
    CAP_TIER = args.tier
    if TAG and not TAG.startswith("_"):
        raise SystemExit("--tag 必须以 _ 开头，避免和已有产物混名")
    GATE_ENABLED = not args.no_gate
    print(f"[口径] 可筛选性闸门={'开' if GATE_ENABLED else '关（复现旧数字）'}", flush=True)
    names = list(LINES) if args.line == "all" else [args.line]
    if args.line != "all" and args.line not in LINES:
        raise SystemExit(f"未知 line: {args.line}（可选 {list(LINES)} 或 all）")
    for name in names:
        await run_line(name)


if __name__ == "__main__":
    asyncio.run(main())
