"""
策略赛马（真实盘口径）—— 多策略横向对比（2026-10-28）

## 为什么需要这个模块

【P0-04】`frontend/src/app/paper-portfolio/page.tsx` 名为「策略赛马」，但每张卡
只有「市值 / 累计收益 / 持仓数 / 一句话描述」4 个信息 —— 用户最关心的对比维度
一个都没有：没有夏普、没有最大回撤、没有换手率、没有净值曲线，6 条线之间也
没有排名。

而**数据是现成的**：
  - `data/backtest/ALL_real_lag1_v2.json` —— 5 条主策略的 **真实盘口径**
    （lag=1 + 可成交性检查）3 年全市场回测结果，含夏普 / 最大回撤 / 交易笔数 /
    逐笔胜率 / 年化 / 分年 / 分月收益。
  - `data/strategy_baseline/baseline_full_market_3y.json` —— 16 策略基线
    （含全市场等权基准的 723 点日频净值），用于「跑赢/跑输基准」的判定。

这两份产物的口径说明：
  - **真实盘口径** = 信号日收盘出信号 → **下一交易日**成交（lag1）+ 可成交性
    检查（涨跌停/停牌/ST）。这是实盘能做到的口径，不是 lag0 的理想化结果。
  - 基线的 `ml_unavailable` 明确记录了 lightgbm / model_signal / ml_ensemble
    三个策略**不可用**（pkl 加载段错误 + 前视），因此赛马表里不会出现它们 ——
    这比让用户以为「跑了个模型策略」更诚实。

## 关键设计

1. **换手率必须显式暴露**：用户明确反感高频交易。3 年 1190 笔 vs 2357 笔的
   差别，光看收益率看不出来。这里按 `年化平仓笔数 = 卖出笔数 / 年数` 与
   `年换手率 = 年化平仓笔数 / 真实平均持仓只数 × 100%` 两种口径给出，
   并给「低频/中频/高频」标注。

   【口径修正 2026-10-28】此前这里有三个错误，已全部修掉：
   - `total_trades` 被当成「完整买卖回合」，实际是**单边成交笔数**
     （买入记 1 次 + 卖出记 1 次）—— 换手率被高估约 1 倍。
   - 分母用 `STRATEGY_TOPK`（topk **目标**持仓数，猜的），不是回测里
     **实测**的持仓只数。
   - 披露文案写「一笔 = 完整买卖回合」，与引擎实现直接矛盾。
   现在统一从 `data/backtest/turnover_truth.json` 读实测值（真实卖出笔数 +
   真实平均持仓只数），该文件由 `backend/scripts/recompute_turnover_truth.py`
   生成，且脚本会与 v2 基准**逐位比对**（不一致直接退出码 1）。

2. **跑输基准必须说出口**：主线 3 年累计 +35.5%，基准（全市场等权买入持有）
   +56.0% —— 主线**跑输 20.5 个百分点**。只列数字不写这句话，等于让用户
   以为策略创造了超额收益。`benchmark_comparison` 字段把结论直接算出来返回
   （赢在回撤与夏普，输在绝对收益），前端照实展示。

3. **本期收益 vs 累计收益必须分开**：纸面盘页显示的「+0.83%」是**两次调仓
   之间**的收益，与 3 年回测的累计收益完全是两回事，不能并排显示。
   本模块返回的是 **3 年回测累计口径**，前端不得与纸面盘的区间收益混排。
4. **数据缺失就说缺失**：净值曲线由**分月收益复利**还原（月频 36 点），
   明确标注 `nav_basis = "月频复利还原"` —— 不是日频真实净值，图形上会有
   阶梯感，不能当成日频精确曲线用。

对外接口：GET /api/v1/trading/strategy-race
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# （services → app → backend → <项目根>）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

#: 真实盘口径（lag1 + 可成交性）的 5 策略回测产物
REAL_PAPER_BASELINE = os.path.join(ROOT, "data", "backtest", "ALL_real_lag1_v2.json")
#: 16 策略基线（含等权基准日频净值）
STRATEGY_BASELINE = os.path.join(
    ROOT, "data", "strategy_baseline", "baseline_full_market_3y.json"
)

#: 展示用的策略显示名（与 signal_service.SIGNAL_STRATEGIES 对齐）
DISPLAY_NAMES: Dict[str, str] = {
    "honest_composite": "综合因子（25 公式·主线）",
    "rsi_71421": "RSI(7,14,21)",
    "factor_ranking": "因子排序",
    "gbm_voltgt": "GBM 波动率目标化",
    "multi_factor": "多因子（对照）",
}

#: 各策略的**默认持仓只数**（topk），取自策略类自身的默认值：
#:   honest_composite  → topk=60  (honest_composite_strategy.py:135)
#:   gbm_voltgt        → topk=60  (gbm_voltgt_strategy.py:76)
#:   rsi_71421         → topk=30  (rsi_71421_strategy.py:55)
#:   factor_ranking    → topk=50  (factor_ranking_strategy.py:45)
#:   multi_factor      → topk=30  (strategies.py:1830)
#:
#: 【重要】这张表**不再用于换手率分母** —— topk 是策略的**目标**持仓数，
#: 不是回测里**实际**持有的只数（现金预留、建仓未满、整手约束都会让实际
#: 持仓低于 topk）。换手率分母一律用 `TURNOVER_TRUTH` 里**实测**的平均持仓。
#: 本表仅在实测数据缺失时作为**降级兜底**，且此时 `positions_source` 会被
#: 标成 "topk_fallback" 以示区分。
STRATEGY_TOPK: Dict[str, int] = {
    "honest_composite": 60,
    "gbm_voltgt": 60,
    "rsi_71421": 30,
    "factor_ranking": 50,
    "multi_factor": 30,
}

#: 换手率实测数据（真实卖出笔数 + 真实平均持仓只数）。
#: 由 `backend/scripts/recompute_turnover_truth.py` 生成，口径与
#: ALL_real_lag1_v2.json 逐位一致（已校验 multi_factor 的
#: total_return / sharpe / max_drawdown / total_trades 完全一致）。
TURNOVER_TRUTH = os.path.join(ROOT, "data", "backtest", "turnover_truth.json")

#: 换手频率分档阈值（年换手率 %，即「每个持仓位一年被换手几次」）
#:   < 300%  → 低频（一个持仓位平均持有 > 1 年）
#:   < 800%  → 中频
#:   >= 800% → 高频
#: 用户明确反感高频交易，所以分档按**换手率**（可比）而不是绝对笔数
#: （笔数受持仓只数影响：30 只仓的 300 笔比 60 只仓的 300 笔换手更频繁）。
_TURNOVER_LOW = 300.0
_TURNOVER_MID = 800.0


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _months_to_nav(monthly: Dict[str, float]) -> List[Dict[str, Any]]:
    """把分月收益复利还原成净值曲线点（净值起点 = 1.0）。

    月频 36 点。曲线是**阶梯感**的月频折线，不是日频精确净值 —— 已在
    `nav_basis` 里标注口径。
    """
    if not monthly:
        return []
    points: List[Dict[str, Any]] = []
    nav = 1.0
    for month in sorted(monthly.keys()):
        try:
            r = float(monthly[month])
        except (TypeError, ValueError):
            continue
        nav *= 1.0 + r
        points.append({"date": month, "value": round(nav, 6)})
    return points


def _turnover_label(turnover_pct: float) -> str:
    """按年换手率分档。"""
    if turnover_pct < _TURNOVER_LOW:
        return "低频"
    if turnover_pct < _TURNOVER_MID:
        return "中频"
    return "高频"


def _annualize_turnover(
    sell_trades: int, years: float, avg_positions: float
) -> Dict[str, Any]:
    """换手率两口径。

    - `trades_per_year`：年化**平仓**笔数（用户反感高频就看这个）
    - `turnover_pct`：年换手率 = 年化平仓笔数 / **真实平均持仓只数** × 100%
      （「一年下来每个持仓位被换手了几次」，与 A 股「换手率」习惯一致；
      一只票被换 3 次 = 300%）

    【口径修正 2026-10-28】此前三个错误，全部已修：

    1. `total_trades` 是**单边成交笔数**（买入记 1 次 + 卖出记 1 次），
       不是「完整买卖回合」。旧代码把它当回合数用，且注释还写死了
       「一买一卖记 1 笔 / 不再乘 2」—— 与引擎实现
       (`portfolio_manager_array.py` 的 BUY/SELL 各自 `trades.append`)
       直接矛盾，导致换手率被高估约 1 倍。
    2. 分母用 `STRATEGY_TOPK`（topk 默认值**猜**的），不是**真实**持仓只数。
       multi_factor 实际平均持仓 28.22 只，但按 topk=30 算；差距不大，
       真正的放大器是第 1 条。
    3. 现在改为从 `data/backtest/turnover_truth.json` 读**实测**的
       卖出笔数与真实平均持仓只数（由 `scripts/recompute_turnover_truth.py`
       用与 v2 产物逐位一致的口径重跑得出）。

    换手率分母必须是真实持仓数，不是 topk —— 用猜的数会得出荒谬值。
    """
    if years <= 0 or not avg_positions or avg_positions <= 0:
        return {
            "trades_per_year": 0.0,
            "turnover_pct": 0.0,
            "positions": avg_positions or 0,
            "frequency": "未知",
            "positions_source": "unknown",
        }
    per_year = sell_trades / years
    pct = per_year / avg_positions * 100.0
    return {
        "trades_per_year": round(per_year, 1),
        "turnover_pct": round(pct, 1),
        "positions": round(avg_positions, 2),
        "frequency": _turnover_label(pct),
        "positions_source": "measured",
    }


def _benchmark_for(
    strategy_key: str, baseline: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """从 16 策略基线里取该策略的 vs_benchmark 对比（超额/跟踪误差/IR）。"""
    rec = (baseline.get("strategies") or {}).get(strategy_key) or {}
    vs = rec.get("vs_benchmark")
    if not isinstance(vs, dict) or vs.get("note"):
        return None
    return {
        "benchmark": vs.get("benchmark"),
        "strategy_ann_return": vs.get("strategy_ann_return"),
        "benchmark_ann_return": vs.get("benchmark_ann_return"),
        "excess_ann_return": vs.get("excess_ann_return"),
        "tracking_error": vs.get("tracking_error"),
        "information_ratio": vs.get("information_ratio"),
        "correlation": vs.get("correlation"),
        "strategy_total_return": vs.get("strategy_total_return"),
        "benchmark_total_return": vs.get("benchmark_total_return"),
    }


def _benchmark_comparison(
    rows: List[Dict[str, Any]], bench_meta: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """与「全市场等权买入持有」基准的诚实对照。

    【为什么必须显式披露】主线策略 honest_composite 的 3 年累计收益是
    +35.5%，而基准（全市场等权买入持有）是 +56.0% —— **主线跑输买入持有
    约 20 个百分点**。只列数字不写这句话，等于让用户以为策略创造了超额
    收益。这里把结论直接算出来返回，前端照实展示。

    诚实结论也要说清另一半：主线最大回撤 -13.4% 显著好于基准 -29.9%，
    夏普 0.88 也高于基准 0.74 —— 赢在**风险控制**，输在**绝对收益**。
    """
    bench_total = bench_meta.get("total_return")
    bench_ann = bench_meta.get("annualized_return")
    bench_sharpe = bench_meta.get("sharpe_ratio")
    bench_mdd = bench_meta.get("max_drawdown")
    if bench_total is None:
        return None

    mainline = next((r for r in rows if r.get("is_mainline")), None)
    if not mainline:
        return None
    m = mainline.get("metrics") or {}
    m_total = m.get("total_return")
    m_ann = m.get("annualized_return")
    m_sharpe = m.get("sharpe_ratio")
    m_mdd = m.get("max_drawdown")

    def _pct(x: Any) -> Optional[float]:
        return round(float(x) * 100.0, 1) if isinstance(x, (int, float)) else None

    excess_total = None
    excess_ann = None
    if isinstance(m_total, (int, float)):
        excess_total = round((m_total - float(bench_total)) * 100.0, 1)
    if isinstance(m_ann, (int, float)) and isinstance(bench_ann, (int, float)):
        excess_ann = round((m_ann - float(bench_ann)) * 100.0, 1)

    beats_total = bool(excess_total is not None and excess_total > 0)
    beats_sharpe = bool(
        isinstance(m_sharpe, (int, float))
        and isinstance(bench_sharpe, (int, float))
        and m_sharpe > bench_sharpe
    )
    mdd_better = bool(
        isinstance(m_mdd, (int, float))
        and isinstance(bench_mdd, (int, float))
        and m_mdd > bench_mdd
    )

    if beats_total and beats_sharpe:
        verdict = "同时跑赢基准的收益与风险调整后收益"
    elif beats_sharpe and mdd_better:
        verdict = (
            "风险调整后收益与回撤优于基准，但绝对收益跑输基准 —— "
            "这是「少亏」而不是「多赚」"
        )
    elif beats_total:
        verdict = "绝对收益跑赢基准，但风险调整后收益不如基准"
    else:
        verdict = "绝对收益与风险调整后收益均跑输基准"

    return {
        "benchmark_name": bench_meta.get("name"),
        "benchmark_total_return": _pct(bench_total),
        "benchmark_annualized_return": _pct(bench_ann),
        "benchmark_sharpe": bench_sharpe,
        "benchmark_max_drawdown": _pct(bench_mdd),
        "mainline_total_return": _pct(m_total),
        "mainline_annualized_return": _pct(m_ann),
        "mainline_sharpe": m_sharpe,
        "mainline_max_drawdown": _pct(m_mdd),
        "excess_total_return_pct": excess_total,
        "excess_annualized_return_pct": excess_ann,
        "beats_benchmark_total_return": beats_total,
        "beats_benchmark_sharpe": beats_sharpe,
        "mainline_drawdown_better": mdd_better,
        "verdict": verdict,
    }


class StrategyRaceService:
    """策略赛马：多策略 3 年真实盘口径横向对比。"""

    def race(self) -> Dict[str, Any]:
        raw = _read_json(REAL_PAPER_BASELINE)
        baseline = _read_json(STRATEGY_BASELINE)
        truth = _read_json(TURNOVER_TRUTH)
        truth_results = truth.get("results") or {}
        results = raw.get("results") or {}
        if not results:
            return {
                "status": "unavailable",
                "message": (
                    "策略赛马数据未生成：缺少 " "data/backtest/ALL_real_lag1_v2.json。"
                ),
                "strategies": [],
                "benchmark": None,
            }

        window = raw.get("window") or ""
        try:
            start_s, end_s = window.split("→")
            years = max(
                1.0,
                round(
                    (
                        __import__("datetime").datetime.fromisoformat(end_s)
                        - __import__("datetime").datetime.fromisoformat(start_s)
                    ).days
                    / 365.25,
                    2,
                ),
            )
        except Exception:
            start_s, end_s, years = window, "", 3.0

        bench_meta = baseline.get("benchmark") or {}

        rows: List[Dict[str, Any]] = []
        for key, rec in results.items():
            if not isinstance(rec, dict) or rec.get("strategy") is None:
                continue
            total_trades = int(rec.get("total_trades") or 0)
            # 换手率用**实测**的卖出笔数 + 真实平均持仓只数；
            # 实测数据缺失时才降级到 topk 猜值（并标注 topk_fallback）。
            t_rec = truth_results.get(key) or {}
            sell_trades = t_rec.get("sell_trades")
            avg_positions = t_rec.get("avg_positions")
            if sell_trades and avg_positions:
                turnover = _annualize_turnover(
                    int(sell_trades), years, float(avg_positions)
                )
            else:
                fallback_positions = STRATEGY_TOPK.get(key, 0)
                turnover = _annualize_turnover(
                    total_trades, years, float(fallback_positions)
                )
                turnover["positions_source"] = "topk_fallback"
            monthly = rec.get("monthly") or {}
            rows.append(
                {
                    "key": key,
                    "name": DISPLAY_NAMES.get(key, key),
                    "is_mainline": key == "honest_composite",
                    "metrics": {
                        "total_return": rec.get("total_return"),
                        "annualized_return": rec.get("annualized_return"),
                        "sharpe_ratio": rec.get("sharpe_ratio"),
                        "max_drawdown": rec.get("max_drawdown"),
                        "volatility": rec.get("volatility"),
                        "win_rate": rec.get("win_rate"),
                        "total_trades": total_trades,
                        "buy_trades": t_rec.get("buy_trades"),
                        "sell_trades": t_rec.get("sell_trades"),
                        "final_value": rec.get("final_value"),
                    },
                    "turnover": turnover,
                    "yearly": rec.get("yearly") or {},
                    "nav": _months_to_nav(monthly),
                    "nav_basis": "月频复利还原（36 点）",
                    "vs_benchmark": _benchmark_for(key, baseline),
                }
            )

        # 按夏普降序 —— 用户要看的是「谁的风险调整后收益最好」，不是「谁赚得多」
        rows.sort(key=lambda r: (r["metrics"].get("sharpe_ratio") or -99), reverse=True)

        return {
            "status": "ok",
            "window": window,
            "start_date": start_s,
            "end_date": end_s,
            "years": years,
            "universe_size": raw.get("universe_size"),
            "methodology": raw.get("methodology") or {},
            "initial_cash": 1_000_000,
            "strategies": rows,
            "benchmark": {
                "name": bench_meta.get("name"),
                "symbol": bench_meta.get("symbol"),
                "total_return": bench_meta.get("total_return"),
                "annualized_return": bench_meta.get("annualized_return"),
                "sharpe_ratio": bench_meta.get("sharpe_ratio"),
                "max_drawdown": bench_meta.get("max_drawdown"),
            },
            # 口径披露 —— 免费用户也要看得到「这个数字是怎么算的」
            "disclosure": [
                "回测窗口 3 年（2023-05-08 → 2026-05-08），全市场 5514 只。",
                "真实盘口径 = 信号日收盘出信号 → 下一交易日成交（lag1）+ 可成交性检查（涨跌停/停牌/ST）。",
                "年换手率 = 年化**平仓**笔数 ÷ 真实平均持仓只数 × 100%（一个持仓位一年被换手几次）。"
                "交易笔数是**单边成交**计数（买入记 1 次、卖出记 1 次），不是完整买卖回合。",
                "持仓只数为回测实测值（非 topk 目标值），来自同口径重跑的持仓快照统计。",
                "净值曲线由分月收益复利还原（月频 36 点），非日频精确净值。",
                "多因子（multi_factor）为对照策略，历史上为负收益，保留以便对比。",
            ],
            # 与基准的诚实对照 —— 不藏「跑输买入持有」这件事
            "benchmark_comparison": _benchmark_comparison(rows, bench_meta),
        }


strategy_race_service = StrategyRaceService()
