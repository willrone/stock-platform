#!/usr/bin/env python3
"""Layer-2 否决层 A/B（2026-11-06）。

定位：research 线。不进纸面盘 7 条线，不改任何现役策略行为。

要回答的问题：在机械打分之上加一层「可回测的排雷否决」，到底值不值钱。

判据（**跑之前写死，不许事后调**）：
    base 与 base+veto，同宇宙 / 同窗口 / 同成本 / 同 lag1：
        夏普 ≥ +0.05  或  最大回撤改善 ≥ 2.0pp   → 否决层保留
        两者都不到                                  → 砍掉这层
    附加披露义务：年化收益变化必须写进台账 —— 风险削减不许靠"少赚"伪装成更安全。

规则（只用"当时就知道"的信息，源 `data/parquet/baostock_daily/*.parquet`，逐日历史，无未来函数）：
    R1 ST/*ST   当日 is_st == 1                                   → 否决
    R2 流动性   当日之前 60 个交易日日均 amount < 5000 万          → 否决
    R3 停牌     当日 trade_status == 0                            → 否决（引擎交易层已拦，这里连分数都不给）
    未纳入 R4「财务恶化」：`data/fundamentals/financial_ak_full.json` 今天查出两个硬伤 ——
      含未来日期（平安银行末条 2027-01-28，本机今天 2026-11-06）+ roe 单位口径不统一（0.15 与 4.89 混存）。
      用它做回测等于偷看未来，必须先用独立源（baostock 财务查询）重建并核对口径，才能进实验。

跑法：
    python scripts/ablation_veto_layer.py --build-veto              # 先建否决掩码（一次性）
    python scripts/ablation_veto_layer.py --line honest_25f_veto|honest_42f_veto|lowvol_veto|random_veto
    # 对照基线（同口径，无否决）在 ablation_lowvol_control.py：honest_25f / honest_42f / lowvol_ctrl / random_ctrl
对照基线（同口径，已跑过，不必重跑）：`--line honest_25f_base` / `honest_42f_base`（见 ablation_lowvol_control.py 的 *_lag1 产物）。
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

import numpy as np
import pandas as pd
from loguru import logger

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies.control_strategies import (  # noqa: E402
    RandomTopKControlStrategy,
)
from app.services.backtest.strategies.factor_ranking_strategy import (  # noqa: E402
    FactorRankingStrategy,
)
from app.services.backtest.strategies.honest_composite_strategy import (  # noqa: E402
    HonestCompositeStrategy,
)
from app.services.backtest.strategies.strategy_factory import StrategyFactory  # noqa: E402
from app.services.backtest.universe_gate import gate_report_path, gated_universe  # noqa: E402

GATE_ENABLED = True  # 可筛选性闸门：风控库看不见的票不许进宇宙（查不到 = 排除）

START = datetime(2023, 5, 12)
END = datetime(2026, 5, 8)
TOPK, NDROP = 60, 5
CASH_RESERVE = 0.05
LIQ_MIN_AMOUNT = 50_000_000.0  # 日均成交额闸门（元）
LIQ_WINDOW = 60
OUT_DIR = os.path.join(os.path.dirname(BACKEND_ROOT), "data", "strategy_baseline")
VETO_PATH = os.path.join(OUT_DIR, "veto_mask_2026-11-06.parquet")

_WEEKLY_LAG1 = {
    "topk": TOPK, "n_drop": NDROP,
    "rebalance_days": "weekly", "rebalance_weekday": 4, "lag": 1,
}

LINES: dict[str, dict] = {
    "honest_25f_veto": {"strategy": "honest_composite_veto",
                        "config": {**_WEEKLY_LAG1, "use_panel": False}},
    "honest_42f_veto": {"strategy": "honest_composite_veto",
                        "config": {**_WEEKLY_LAG1, "use_panel": True}},
    # 关键对照：同一套否决规则扣在**对照策略**上。
    # 如果对照组加了否决也一样变好，那这层是「宇宙清洗 / 可交易性卫生」，不是我们策略的增量。
    "lowvol_veto": {"strategy": "factor_ranking_veto",
                    "config": {**_WEEKLY_LAG1, "weights": {"low_vol": 1.0}}},
    "random_veto": {"strategy": "random_topk_control_veto", "config": {**_WEEKLY_LAG1}},
    # 2026-11-06 §12.63：2×2 里最合理的候选 = 25 因子 + 行业中性；再叠 Layer2 否决。
    # 注意：两个改良**必须实测叠加效果**，不许假设「各自有效 → 加起来更有效」。
    "honest_25f_neu_veto": {"strategy": "honest_composite_veto",
                            "config": {**_WEEKLY_LAG1, "use_panel": False, "industry_neutral": True}},
    # 机制核查需要「同一套持仓记录」的无否决基线，否则只能看加了否决的两条，无从对照
    "base_25f": {"strategy": "honest_composite", "config": {**_WEEKLY_LAG1, "use_panel": False}},
    "base_25f_neu": {"strategy": "honest_composite",
                     "config": {**_WEEKLY_LAG1, "use_panel": False, "industry_neutral": True}},
}


# ── 否决掩码：建一次，存盘，各次运行只读 ───────────────────────────────
def build_veto_mask() -> None:
    files = sorted(glob.glob(os.path.join(str(settings.DATA_ROOT_PATH), "parquet", "baostock_daily", "*.parquet")))
    rows: list[tuple[str, int, str]] = []
    n_files = 0
    for path in files:
        code = os.path.basename(path)[: -len(".parquet")].replace("_", ".")
        try:
            df = pd.read_parquet(path, columns=["date", "is_st", "trade_status", "amount"])
        except Exception as exc:  # 列缺失/空文件
            logger.warning(f"跳过 {code}: {exc}")
            continue
        if df.empty:
            continue
        n_files += 1
        d = pd.to_datetime(df["date"].astype(str)).values.astype("datetime64[ns]")
        order = np.argsort(d)
        d = d[order]
        st = np.asarray(df["is_st"].to_numpy(), dtype=float)[order]
        status = np.asarray(df["trade_status"].to_numpy(), dtype=float)[order]
        amount = np.asarray(df["amount"].to_numpy(), dtype=float)[order]

        keep = (d >= np.datetime64(START)) & (d <= np.datetime64(END))
        if not keep.any():
            continue
        # R2 用「当日之前 60 个交易日」的均值 → 先在全序列上算，再切窗口，避免用未来
        roll = pd.Series(amount).rolling(LIQ_WINDOW, min_periods=20).mean().shift(1).to_numpy()
        st_bad = st == 1
        halt = status == 0
        liq_bad = np.isfinite(roll) & (roll < LIQ_MIN_AMOUNT)
        for rule, mask in (("ST", st_bad), ("HALT", halt), ("LOWLIQ", liq_bad)):
            idx = np.where(mask & keep)[0]
            for i in idx:
                rows.append((code, int(d[i].astype(np.int64)), rule))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = pd.DataFrame(rows, columns=["code", "date_ns", "rule"])
    out.to_parquet(VETO_PATH, index=False)
    print(f"veto mask: {n_files} 只扫描，{len(out)} 条(只,日,规则)，"
          f"按规则 {out['rule'].value_counts().to_dict()}", flush=True)
    print(f"saved -> {VETO_PATH}", flush=True)


_VETO_CACHE: dict[str, set[int]] | None = None


def load_veto() -> dict[str, set[int]]:
    global _VETO_CACHE
    if _VETO_CACHE is None:
        if not os.path.exists(VETO_PATH):
            raise SystemExit(f"否决掩码不存在：{VETO_PATH}（先跑 --build-veto）")
        df = pd.read_parquet(VETO_PATH)
        cache: dict[str, set[int]] = {}
        for code, grp in df.groupby("code"):
            cache[str(code)] = set(int(v) for v in grp["date_ns"].to_numpy())
        _VETO_CACHE = cache
    return _VETO_CACHE


def apply_veto(stock_data: dict[str, pd.DataFrame]) -> int:
    """把被否决的 (只,日) 分数从策略存的分数表里删掉 → 执行器那天不可能买它。"""
    veto = load_veto()
    removed = 0
    for code, frame in stock_data.items():
        dates = veto.get(code)
        if not dates:
            continue
        for key in [k for k in frame.attrs if "score" in k.lower()]:
            table = frame.attrs[key]
            if not isinstance(table, dict):
                continue
            doomed = [d for d in table if int(pd.Timestamp(d).value) in dates]
            for d in doomed:
                table.pop(d, None)
            removed += len(doomed)
    return removed


class _VetoMixin:
    """打分照旧，被否决的 (只,日) 分数直接删掉 → 执行器那天不可能选到它。"""

    async def prepare_backtest_data(self, stock_data, start_date, end_date) -> None:
        await super().prepare_backtest_data(stock_data, start_date, end_date)
        removed = apply_veto(stock_data)
        print(f"[Layer2 否决层] 删除分数条目 {removed}（规则 ST/LOWLIQ/HALT）", flush=True)


class VetoedFactorRanking(_VetoMixin, FactorRankingStrategy):
    """纯低波动对照 + 同一套否决（对照用）"""


class VetoedRandomTopK(_VetoMixin, RandomTopKControlStrategy):
    """随机对照 + 同一套否决（对照用）"""


class VetoedHonestComposite(_VetoMixin, HonestCompositeStrategy):
    """honest_composite + Layer-2 否决"""


StrategyFactory.register_strategy("honest_composite_veto", VetoedHonestComposite)
StrategyFactory.register_strategy("factor_ranking_veto", VetoedFactorRanking)
StrategyFactory.register_strategy("random_topk_control_veto", VetoedRandomTopK)


def backtest_config(record_positions: bool = False) -> BacktestConfig:
    """与 ablation_lowvol_control.py 完全一致的真实成本口径（不是档案里 slippage 0.0001 + 免印花税那种）。"""
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
        # --record-positions 打开：做「为什么 25f_neu_veto 崩了」的机制核查要看持仓集中度
        record_positions_in_history=record_positions,
    )


async def run_line(name: str, tag: str = "", record_positions: bool = False) -> None:
    spec = LINES[name]
    codes, gate_rep = gated_universe(str(settings.DATA_ROOT_PATH), START, END, enabled=GATE_ENABLED)
    if gate_rep is not None:
        print(f"[闸门] {gate_rep.summary()}", flush=True)
    print(f"[{name}] strategy={spec['strategy']} universe={len(codes)} "
          f"window={START.date()}..{END.date()} topk={TOPK} n_drop={NDROP} weekly(Fri) lag=1 veto=ON",
          flush=True)
    executor = BacktestExecutor(data_dir=str(settings.DATA_ROOT_PATH), enable_parallel=True,
                               max_workers=8, enable_performance_profiling=False,
                               use_multiprocessing=False)
    t0 = time.perf_counter()
    res = await executor.run_backtest(
        strategy_name=spec["strategy"], stock_codes=codes,
        start_date=START, end_date=END,
        strategy_config=dict(spec["config"]), backtest_config=backtest_config(record_positions),
    )
    wall = time.perf_counter() - t0

    def _conv(o):
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
        "ablation_line": name, "strategy": spec["strategy"], "strategy_config": dict(spec["config"]),
        "liq_min_amount": LIQ_MIN_AMOUNT, "veto_mask_path": VETO_PATH, "tag": tag,
        "record_positions": record_positions,
        "veto_rules": [f"ST(is_st=1)", f"LOWLIQ(此前{LIQ_WINDOW}交易日日均 amount<{LIQ_MIN_AMOUNT/1e7:.0f}千万)",
                       "HALT(trade_status=0)"],
        "universe_requested": gate_rep.universe_size if gate_rep else len(codes),
        "universe_after_gate": len(codes),
        "coverage_gate": None if gate_rep is None else {
            "min_screen_ratio": gate_rep.min_ratio, "dropped": len(gate_rep.dropped),
            "dropped_share_pct": round(gate_rep.dropped_share * 100, 2),
            "report": gate_report_path(str(settings.DATA_ROOT_PATH)),
        },
        "window": [START.isoformat(), END.isoformat()],
        "wall_seconds": wall, "run_at": datetime.now().isoformat(),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"ablation_2026-11-06_{name}{tag}_lag1.json")
    with open(out, "w") as f:
        json.dump(res, f, default=_conv, ensure_ascii=False)
    print(f"[{name}] sharpe={res['sharpe_ratio']:.4f} total={res['total_return']*100:.2f}% "
          f"dd={res['max_drawdown']*100:.2f}% vol={res['volatility']*100:.2f}% "
          f"trades={res['total_trades']} cost={res['cost_statistics']['cost_ratio']*100:.2f}% wall={wall:.1f}s",
          flush=True)
    print(f"[{name}] saved -> {out}", flush=True)


def main() -> None:
    global LIQ_MIN_AMOUNT, VETO_PATH, RULES, GATE_ENABLED
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-veto", action="store_true")
    ap.add_argument("--line", help="|".join(LINES))
    # 阈值敏感性用：同一个线跑不同流动性闸门，掩码和产物各自带 tag，不互相覆盖。
    ap.add_argument("--liq-min", type=float, default=LIQ_MIN_AMOUNT,
                    help="R2 流动性闸门（元/日均成交额），默认 5000 万")
    ap.add_argument("--tag", default="", help="产物文件后缀，例如 _liq20m / _pos")
    ap.add_argument("--veto-tag", default=None,
                    help="掩码文件后缀；默认与 --tag 相同。跑 _pos 这种「复用默认掩码」的实验时传空串")
    ap.add_argument("--record-positions", action="store_true", help="记录逐日持仓（机制核查用）")
    ap.add_argument("--rules", default="", help="只用指定规则，逗号分隔，例：ST,HALT 或 LOWLIQ（默认三条全用）")
    ap.add_argument("--no-gate", action="store_true",
                    help="关掉可筛选性闸门（只为复现 2026-11-06 闸门前的旧数字）")
    args = ap.parse_args()

    LIQ_MIN_AMOUNT = args.liq_min
    RULES = {r.strip() for r in args.rules.split(",") if r.strip()}
    GATE_ENABLED = not args.no_gate
    print(f"[口径] 可筛选性闸门={'开' if GATE_ENABLED else '关（复现旧数字）'}", flush=True)
    veto_suffix = args.tag if args.veto_tag is None else args.veto_tag
    VETO_PATH = os.path.join(OUT_DIR, f"veto_mask_2026-11-06{veto_suffix}.parquet")
    if args.tag and not args.tag.startswith("_"):
        raise SystemExit("--tag 必须以 _ 开头，避免和已有产物混名")

    if args.build_veto:
        build_veto_mask()
        return
    if not args.line or args.line not in LINES:
        raise SystemExit(f"用 --build-veto 或 --line {{{'|'.join(LINES)}}}")
    print(f"[口径] liq_min={LIQ_MIN_AMOUNT/1e7:.0f}千万 掩码={os.path.basename(VETO_PATH)} "
          f"rules={sorted(RULES) or '全部'} tag={args.tag}", flush=True)
    asyncio.run(run_line(args.line, tag=args.tag, record_positions=args.record_positions))


if __name__ == "__main__":
    main()
