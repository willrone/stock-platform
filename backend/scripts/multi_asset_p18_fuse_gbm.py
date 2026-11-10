#!/usr/bin/env python3
"""P18 —— 42f 之上叠加 gbm 增量（分数级 z-score 线性融合）。

⚠️ 预注册已落纸：`docs/backtest/MULTI_ASSET_LEDGER_2026-10-31.md` §十。
**本文件的搜索空间、判据、否决前置全部事前固定，不允许跑完后加格。**

为什么只做这一件事
------------------
台账 §八：8 个现成换源候选全部落后于基线 honest_composite 0.8796；
唯一指向「换源有空间」的支点是 **gbm ↔ 量价因子 mean RankIC 仅 0.112 / 0.150**（近乎正交）。
故只做**增量叠加**（不是替换），且用 override 把试验数从 1937 压到 5。

事前固定的搜索空间（唯一，不扫其他）
------------------------------------
- 融合方式：两路各自做**当日截面 z-score**，再线性加权 `combined = (1-w)·z42 + w·zgbm`
- 融合权重 w ∈ {0.10, 0.20, 0.30, 0.50} + 基线 w=0.00 → **5 个规则臂**
- gbm 的 vol-target **关闭**（target_vol 设 10.0 → scale≡1），只取排序信息，仓位由 42f 侧定
- gbm 调仓日：weekly / weekday=4（周五信号），与 42f 定稿一致
- lag：42f 用 1（定稿口径）；gbm 的分数在融合时**沿交易日历右移 1 个交易日**对齐到执行日
- 42f 因子集：定稿 25 公式版（不传 factor_allowlist）

placebo 随机对照：每个 w 跑 3 个种子（7 / 99 / 1234），把 gbm 那一路的 z 值换成
同掩码下的标准正态随机数 → 检验「增量是否真的来自 gbm 的信息」。

否决前置（先跑，不通过就不跑融合回测）
--------------------------------------
42f 与 gbm 的**同日截面相关系数** > 0.3（正交性在同日不成立）→ 直接判死。

用法
----
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p18_fuse_gbm.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p18_fuse_gbm.py --window oos --carry data/backtest/multi_asset_p18_main.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p2 as P2  # noqa: E402
import multi_asset_p3_bond as P3  # noqa: E402
import multi_asset_leg_experiment as M  # noqa: E402
import trial_budget as TB  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    load_etf_universe,
    maxdd,
    merge_by_date,
    sharpe,
    total_ret,
    yearly,
)
from app.services.backtest.execution.backtest_executor import (  # noqa: E402
    BacktestExecutor,
)
from app.services.backtest.models.data_models import BacktestConfig  # noqa: E402
from app.services.backtest.strategies import (  # noqa: E402
    gbm_voltgt_strategy as GBM,
)
from app.services.backtest.strategies import (  # noqa: E402
    honest_composite_strategy as HCS,
)

# ── 预注册常量（跑之前固定，不允许事后扩）───────────────────────────────
CLASSES = ["gold", "bond", "commodity"]
W_A_SHARE = 0.5
W_GBMS: List[float] = [0.10, 0.20, 0.30, 0.50]
BASE_W = 0.00
PLACEBO_SEEDS: List[int] = [7, 99, 1234]
N_TRIALS_OVERRIDE = 5                 # 试验数（override 全局 1937）
CORR_VETO = 0.30                      # 同日截面相关 > 0.3 → 判死
ROLL_WIN = 504                       # 滚动夏普窗口
ROLL_STEP = 21                       # 滚动步长（≈1 个月）
MIN_CROSS = 30                       # 截面 z-score 最小样本数
P5_FIXED = {
    "gold": ["518880", "159934", "159937"],
    "commodity": ["159980", "159985", "159981"],
    "bond": ["511260", "511010", "511020"],
}

# gbm 侧配置：**vol-target 关闭**（target_vol=10 → scale≡1）、周频、label horizon=1
GBM_CFG: Dict[str, Any] = {
    "lag": 1,
    "rebalance_days": "weekly",
    "rebalance_weekday": 4,
    "target_vol": 10.0,
    "vt_window": 20,
    "horizon": 1,
    "topk": 60,
    "n_drop": 5,
}

# 分数缓存：key = (start, end) → {master, codes, m42, mgbm}
_SCORE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
# 当前臂的融合参数（monkeypatch 通过它取）
_CUR: Dict[str, Any] = {"w": BASE_W, "placebo_seed": None, "mode": "fuse"}


# ── 工具 ────────────────────────────────────────────────────────────────
def _find_42f_class():
    """动态定位 42f 策略类。

    ⚠️ 不硬编码类名，也不靠类属性 `_SCORES_ATTR`（它是**模块级**常量，
    类上取不到 → 早期版本按类属性找，永远找不到类）。
    判据：在 HCS 模块内定义、且有 prepare_backtest_data 的**唯一**一个类。
    """
    cands = [
        getattr(HCS, n) for n in dir(HCS)
        if isinstance(getattr(HCS, n), type)
        and getattr(HCS, n).__module__ == HCS.__name__
        and hasattr(getattr(HCS, n), "prepare_backtest_data")
    ]
    if len(cands) != 1:
        raise SystemExit(f"[p18] ❌ 定位 42f 策略类失败，候选 {len(cands)} 个: "
                         f"{[c.__name__ for c in cands]}")
    return cands[0]


H42 = _find_42f_class()


def _build_master(stock_data: Dict[str, pd.DataFrame],
                  start_date, end_date) -> List[Any]:
    """全市场交易日历（并集），窗口内。两路分数都用它对齐。"""
    idx = pd.DatetimeIndex([])
    for df in stock_data.values():
        if df is None or df.empty:
            continue
        idx = idx.append(pd.DatetimeIndex(df.index))
    if len(idx) == 0:
        return []
    idx = idx.unique().sort_values()
    mask = (idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date))
    return [d.to_pydatetime() for d in idx[mask]]


def _extract_matrix(stock_data: Dict[str, pd.DataFrame], attr: str,
                    master: List[Any], codes: List[str]) -> np.ndarray:
    """从 per-stock attrs 的 {datetime: score} 抽成 (len(master), len(codes)) 矩阵。"""
    di = {d: i for i, d in enumerate(master)}
    ci = {c: i for i, c in enumerate(codes)}
    mat = np.full((len(master), len(codes)), np.nan, dtype=np.float64)
    for code, df in stock_data.items():
        j = ci.get(code)
        if j is None:
            continue
        sm = df.attrs.get(attr)
        if not sm:
            continue
        for d, v in sm.items():
            i = di.get(d)
            if i is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                mat[i, j] = fv
    return mat


def _inject_matrix(stock_data: Dict[str, pd.DataFrame], attr: str,
                    mat: np.ndarray, master: List[Any], codes: List[str]) -> None:
    """把矩阵写回 per-stock attrs（键 = datetime）。"""
    for j, code in enumerate(codes):
        df = stock_data.get(code)
        if df is None:
            continue
        col = mat[:, j]
        sm: Dict[Any, float] = {}
        for k, d in enumerate(master):
            v = col[k]
            if np.isfinite(v):
                sm[d] = float(v)
        df.attrs[attr] = sm


def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    """逐行截面 z-score（只在有限值集合内）；不足 MIN_CROSS 个样本的行整行保持 NaN。"""
    out = np.full_like(mat, np.nan)
    for k in range(mat.shape[0]):
        row = mat[k]
        ok = np.isfinite(row)
        n = int(ok.sum())
        if n < MIN_CROSS:
            continue
        vals = row[ok]
        sd = vals.std()
        if sd <= 1e-12:
            continue
        out[k, ok] = (vals - vals.mean()) / sd
    return out


def _fuse(m42: np.ndarray, mgbm: np.ndarray, w: float,
          placebo_seed: Optional[int] = None) -> np.ndarray:
    """两路截面 z 分数线性加权。

    - z42 在 **42f 自身** 的有限集合上算；z_gbm 在 **gbm 自身** 的有限集合上算。
    - gbm 缺失处填 0（= 该股截面中性值）→ **不改变 42f 的选股池**。
    - 输出只写 42f 有限集合 → w=0 时选股与定稿完全一致（仅分数尺度不同）。
    """
    z42 = _zscore_rows(m42)
    zg = _zscore_rows(mgbm)
    out = np.full_like(m42, np.nan)
    for k in range(m42.shape[0]):
        a_ok = np.isfinite(m42[k])
        if int(a_ok.sum()) < MIN_CROSS:
            continue
        b_ok = np.isfinite(mgbm[k])
        zb = np.zeros(mgbm.shape[1], dtype=np.float64)
        if int(b_ok.sum()) >= MIN_CROSS:
            if placebo_seed is None:
                zb[b_ok] = zg[k][b_ok]
            else:
                rng = np.random.default_rng(placebo_seed * 1000003 + k)
                zb[b_ok] = rng.standard_normal(int(b_ok.sum()))
        out[k, a_ok] = (1.0 - w) * z42[k][a_ok] + w * zb[a_ok]
    return out


# ── monkeypatch：把 gbm 分数融合进 42f 的分数映射 ────────────────────────
_ORIG_PREPARE = H42.prepare_backtest_data


# gbm 的 prepare 有 single_model_from，honest 的**没有** → 用 inspect 按实际签名分派
def _call_prepare(fn, self, stock_data, start_date, end_date, single_model_from=None):
    """按被调函数的真实签名调用（42f 与 gbm 的签名不同，硬传会 TypeError）。"""
    import inspect
    params = inspect.signature(fn).parameters
    # ⚠️ 必须按**参数名**判断，不能按个数：未绑定函数的 signature 里 self 也占一个位置，
    #    honest 的真实签名是 (self, stock_data, start_date, end_date) = 4 个参数，
    #    但它**没有** single_model_from → 按个数判会多传一个 → TypeError。
    has_sfm = "single_model_from" in params
    # ⚠️ 绑定方法（instance.method）的 signature 里**没有** self → 不能再传 self
    is_unbound = "self" in params
    args = (self,) if is_unbound else ()
    if has_sfm:
        return fn(*args, stock_data, start_date, end_date, single_model_from)
    return fn(*args, stock_data, start_date, end_date)


async def _patched_prepare(self, stock_data, start_date, end_date,
                           single_model_from=None) -> None:
    key = (str(start_date), str(end_date))
    if key not in _SCORE_CACHE:
        await _call_prepare(_ORIG_PREPARE, self, stock_data, start_date, end_date,
                            single_model_from)
        master = _build_master(stock_data, start_date, end_date)
        codes = sorted(stock_data.keys())
        m42 = _extract_matrix(stock_data, HCS._SCORES_ATTR, master, codes)
        print(f"[p18]   42f 分数矩阵 {m42.shape}，非空 {int(np.isfinite(m42).sum())}",
              flush=True)

        gbm = GBM.GbmVoltgtStrategy(dict(GBM_CFG))
        await _call_prepare(gbm.prepare_backtest_data, gbm, stock_data,
                            start_date, end_date, single_model_from)
        mgbm_raw = _extract_matrix(stock_data, GBM._SCORES_ATTR, master, codes)
        print(f"[p18]   gbm 分数矩阵 {mgbm_raw.shape}，"
              f"非空 {int(np.isfinite(mgbm_raw).sum())}", flush=True)
        # gbm 的分数键 = **信号日**（lag 在 generate_signals 里做）→ 沿交易日历右移 1 位
        mgbm = np.full_like(mgbm_raw, np.nan)
        if mgbm_raw.shape[0] > 1:
            mgbm[1:, :] = mgbm_raw[:-1, :]
        _SCORE_CACHE[key] = {"master": master, "codes": codes,
                             "m42": m42, "mgbm": mgbm}
    else:
        c = _SCORE_CACHE[key]
        _inject_matrix(stock_data, HCS._SCORES_ATTR, c["m42"], c["master"], c["codes"])

    c = _SCORE_CACHE[key]
    mode = _CUR.get("mode", "fuse")
    if mode == "orig42f":
        # 复现门：完全用原始 42f 分数（不做任何 z 化/融合）
        _inject_matrix(stock_data, HCS._SCORES_ATTR, c["m42"], c["master"], c["codes"])
        return
    fused = _fuse(c["m42"], c["mgbm"], float(_CUR["w"]), _CUR.get("placebo_seed"))
    _inject_matrix(stock_data, HCS._SCORES_ATTR, fused, c["master"], c["codes"])


def _install_patch() -> None:
    H42.prepare_backtest_data = _patched_prepare


def _uninstall_patch() -> None:
    H42.prepare_backtest_data = _ORIG_PREPARE


# ── 跑一条 A 股腿 ──────────────────────────────────────────────────────
def run_leg(codes: List[str], start_date, end_date,
            w: float, placebo_seed: Optional[int] = None,
            mode: str = "fuse") -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
    _CUR.update({"w": w, "placebo_seed": placebo_seed, "mode": mode})
    try:
        cfg: Dict[str, Any] = {**M.STRATEGY_CFG, "lag": 1,
                               "topk": M.TOPK, "n_drop": M.N_DROP}
        executor = BacktestExecutor(
            data_dir=os.path.join(ROOT, "data"),
            enable_parallel=True,
            max_workers=8,
            enable_performance_profiling=False,
            use_multiprocessing=False,
        )
        res = asyncio.run(
            executor.run_backtest(
                strategy_name=M.STRATEGY,
                stock_codes=codes,
                start_date=start_date,
                end_date=end_date,
                strategy_config=cfg,
                backtest_config=BacktestConfig(
                    enable_tradability=True,
                    initial_cash=1_000_000.0,
                    commission_rate=0.0003,
                    slippage_rate=0.0001,
                    max_position_size=0.0,
                    cash_reserve_ratio=0.05,
                    board_lot_size=100,
                    stop_loss_pct=0.0,
                    take_profit_pct=0.0,
                    rebalance_frequency="daily",
                    open_cost=0.0,
                    close_cost=0.0,
                    min_cost=0.0,
                ),
            )
        )
        nav = [(str(d), float(v)) for d, v in (res.get("equity_curve") or [])
               if v is not None]
        rec = {
            "engine_total_return": res.get("total_return"),
            "engine_sharpe": res.get("sharpe_ratio"),
            "engine_max_drawdown": res.get("max_drawdown"),
            "total_trades": len(res.get("trade_history") or []),
            "nav_points": len(nav),
        }
        return rec, nav
    finally:
        _CUR.update({"w": BASE_W, "placebo_seed": None, "mode": "fuse"})


# ── 统计工具 ────────────────────────────────────────────────────────────
def _dsr_of(sr_annual: float, n_trials: int, n_obs: int,
            skew: float = -0.6956, kurt: float = 17.4961) -> float:
    """Deflated Sharpe（与 trial_budget / deflated_sharpe 同口径）。

    SR0 必须用**策略实际的日夏普**算（不能用门槛反推，会循环依赖）。
    """
    sr_d = sr_annual / math.sqrt(252.0)
    n = max(60, int(n_obs))
    var = max(1e-12, (1.0 - skew * sr_d + (kurt - 1.0) / 4.0 * sr_d ** 2) / n)
    sd_sr = math.sqrt(var)
    sr0 = sd_sr * math.sqrt(2.0 * math.log(max(2, int(n_trials))))
    z = (sr_d - sr0) / sd_sr
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def rolling_win_rate(nav_a: List[Tuple[str, float]],
                     nav_b: List[Tuple[str, float]],
                     win: int = ROLL_WIN, step: int = ROLL_STEP
                     ) -> Optional[Dict[str, Any]]:
    """滚动窗口夏普胜率：a 的夏普 > b 的夏普 的比例。"""
    da, db = dict(nav_a), dict(nav_b)
    dates = sorted(set(da) & set(db))
    if len(dates) < win:
        return None
    wins = 0
    total = 0
    for i in range(0, len(dates) - win + 1, step):
        seg = dates[i:i + win]
        sa = sharpe([(d, da[d]) for d in seg])
        sb = sharpe([(d, db[d]) for d in seg])
        if not (np.isfinite(sa) and np.isfinite(sb)):
            continue
        total += 1
        if sa > sb:
            wins += 1
    if total == 0:
        return None
    return {"win_rate": round(wins / total, 4), "n_windows": total,
            "wins": wins, "window_days": win, "step_days": step}


def cross_section_corr(m42: np.ndarray, mgbm: np.ndarray) -> Dict[str, Any]:
    """同日截面相关（诊断 / 否决前置）。

    只在两路都有 ≥MIN_CROSS 个共同股票的日期上算 → 逐日 Pearson，再取中位数。
    """
    z42 = _zscore_rows(m42)
    zg = _zscore_rows(mgbm)
    cors: List[float] = []
    covs: List[float] = []
    cover: List[float] = []
    for k in range(m42.shape[0]):
        a_ok = np.isfinite(m42[k])
        b_ok = np.isfinite(mgbm[k])
        both = a_ok & b_ok
        if int(both.sum()) < MIN_CROSS:
            continue
        a = z42[k][both]
        b = zg[k][both]
        if a.std() <= 1e-12 or b.std() <= 1e-12:
            continue
        cors.append(float(np.corrcoef(a, b)[0, 1]))
        cover.append(float(b_ok.sum()) / max(1, int(a_ok.sum())))
        # 秩相关（对非线性更稳）
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        covs.append(float(np.corrcoef(ra, rb)[0, 1]))
    if not cors:
        return {"n_days": 0}
    return {
        "n_days": len(cors),
        "pearson_median": round(float(np.median(cors)), 4),
        "pearson_mean": round(float(np.mean(cors)), 4),
        "spearman_median": round(float(np.median(covs)), 4),
        "gbm_coverage_median": round(float(np.median(cover)), 4),
    }


# ── main ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos"])
    ap.add_argument("--carry", default="", help="main 产物路径（oos 阶段读它）")
    ap.add_argument("--only", default="", help="只跑指定标签（冒烟用）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：小股票池 + 短窗口，只验证管线")
    ap.add_argument("--skip-placebo", action="store_true")
    args = ap.parse_args()

    start, end = (M.START, M.END) if args.window == "main" else (M.OOS_START, M.OOS_END)
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    out_path = os.path.join(ROOT, "data", "backtest", f"multi_asset_p18_{args.window}.json")
    if args.smoke:
        # ⚠️ 冒烟绝不能覆盖正式产物
        out_path = out_path.replace(".json", "_smoke.json")
    print(f"[p18] 窗口 {s_str} → {e_str}", flush=True)

    # ── 袖套（与 P3/P6/P7 完全同口径）──
    pool = P3.build_pool(start)
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    held = {c: list(P5_FIXED[c]) for c in P5_FIXED}
    p5_match = {c: (sorted(held.get(c, [])) == sorted(P5_FIXED[c])) for c in P5_FIXED}
    _known = set(base_codes) | {r["code"] for r in pool}
    missing = [c for lst in held.values() for c in lst if c not in _known]
    if not all(p5_match.values()) or missing:
        raise SystemExit(f"[p18] ❌ 固定持仓集异常 match={p5_match} missing={missing}")

    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    if len(calendar) < 300:
        raise SystemExit(f"[p18] ❌ 交易日历过短 {len(calendar)}")
    print(f"[p18] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)

    tg, _tmeta = P3.build_targets(calendar, held, P3.fixed_weights_for(CLASSES))
    c_of = {c: cls for cls, lst in held.items() for c in lst}
    sleeve_nav, srec = L3.simulate(
        calendar, prices, tg, c_of, {c: 1.0 / len(CLASSES) for c in CLASSES})
    if not sleeve_nav:
        raise SystemExit("[p18] ❌ 袖套无净值")
    print(f"[p18] 袖套 {len(sleeve_nav)} 天，夏普 {sharpe(sleeve_nav):.4f}，"
          f"波动 {P2.nav_vol(sleeve_nav):.2%}，换手 {srec['turnover_ratio']:.2f}x", flush=True)

    # ── 股票池 ──
    from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
    codes = universe_from_instruments(QLIB_OFFICIAL)
    if args.smoke:
        codes = codes[:400]
        start = pd.Timestamp("2024-05-08").to_pydatetime()
        end = pd.Timestamp("2024-11-08").to_pydatetime()
        print(f"[p18] ⚠️ 冒烟模式：股票池 {len(codes)}，窗口 {start} → {end}", flush=True)
    # ⚠️ 必须显式同步模块全局窗口（否则 A 股腿静默用 main 窗口）
    M.START, M.END = start, end
    print(f"[p18] 股票池 {len(codes)} 只，A 股腿窗口 "
          f"{M.START.date()}→{M.END.date()}", flush=True)

    arms: List[Dict[str, Any]] = [
        {"label": "orig42f", "w": None, "placebo": None, "kind": "gate"},
        {"label": "fuse_w0.00", "w": 0.0, "placebo": None, "kind": "base"},
    ]
    for w in W_GBMS:
        arms.append({"label": f"fuse_w{w:.2f}", "w": w, "placebo": None, "kind": "rule"})
        if not args.skip_placebo:
            for s in PLACEBO_SEEDS:
                arms.append({"label": f"plc_w{w:.2f}_s{s}", "w": w,
                             "placebo": s, "kind": "placebo"})

    carry: Dict[str, Any] = {}
    if args.window == "oos":
        if not args.carry:
            raise SystemExit("[p18] ❌ oos 阶段必须给 --carry（main 产物）")
        carry_path = (args.carry if os.path.isabs(args.carry)
                      else os.path.join(ROOT, args.carry))
        if not os.path.exists(carry_path):
            raise SystemExit(f"[p18] ❌ carry 不存在 {carry_path}")
        with open(carry_path, encoding="utf-8") as fh:
            carry = json.load(fh)
        # ⚠️ 4 个规则臂**全部**要在 oos 上跑：判据 (f)「oos 最优权重在 main
        #    不得排最差」只有在多个候选臂时才存在（只跑 1 个 → 排名 1/1 → (f) 恒真）。
        #    候选臂本身按**两阶段**定：main 选出 → oos 验证（绝不在 oos 上挑臂）。
        order = [l for l in carry.get("ranking", []) if l != "fuse_w0.00"]
        main_rule_rank = [l for l in order if l.startswith("fuse_w")]
        main_best_rule = main_rule_rank[0] if main_rule_rank else None
        print(f"[p18] Stage2 候选臂（main 选出）= {main_best_rule}；"
              f"main 规则臂排名 {main_rule_rank}（oos 阶段跑全部规则臂）", flush=True)

    only = {x for x in args.only.split(",") if x}
    if only:
        # ⚠️ 基线臂与复现门臂**强制保留** —— 判据全部相对基线算，
        #    把基线滤掉会直接 KeyError（冒烟时踩到）。
        arms = [a for a in arms if a["kind"] in ("gate", "base") or a["label"] in only]
    if not arms:
        raise SystemExit("[p18] ❌ 没有要跑的臂")

    _install_patch()
    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}
    try:
        for ai, arm in enumerate(arms, 1):
            lab = arm["label"]
            t0 = time.perf_counter()
            if arm["kind"] == "gate":
                rec, a_nav = run_leg(codes, start, end, BASE_W, None, mode="orig42f")
            else:
                rec, a_nav = run_leg(codes, start, end, arm["w"], arm["placebo"])
            wall = time.perf_counter() - t0
            comb = merge_by_date([a_nav, sleeve_nav], [W_A_SHARE, 1 - W_A_SHARE])
            cs = sharpe(comb)
            results[lab] = {
                "kind": arm["kind"],
                "w_gbm": arm["w"],
                "placebo_seed": arm["placebo"],
                "a_share_sharpe": round(sharpe(a_nav), 4),
                "a_share_return": round(total_ret(a_nav), 4),
                "combined_sharpe": round(cs, 4),
                "combined_return": round(total_ret(comb), 4),
                "combined_maxdd": round(maxdd(comb), 4),
                "combined_vol": round(P2.nav_vol(comb), 4),
                "combined_yearly": yearly(comb),
                "corr_sleeve_ashare": P2.nav_corr(sleeve_nav, a_nav),
                "wall_s": round(wall, 1),
                **{k: v for k, v in rec.items() if k != "wall_s"},
            }
            navs[lab] = comb
            print(f"[p18] {ai:2d}/{len(arms)} {lab:20s} 组合夏普 {cs:.4f} "
                  f"A腿 {sharpe(a_nav):.4f} 笔数 {rec['total_trades']} "
                  f"({wall:.0f}s)", flush=True)
    finally:
        _uninstall_patch()

    # ── 诊断（否决前置）──
    diag: Dict[str, Any] = {}
    for key, c in _SCORE_CACHE.items():
        cc = cross_section_corr(c["m42"], c["mgbm"])
        diag[str(key)] = cc
        print(f"[p18] 诊断 {key}: {cc}", flush=True)
    corr_med = max((d.get("pearson_median", 0.0) for d in diag.values()), default=0.0)
    veto = corr_med > CORR_VETO
    print(f"[p18] 同日截面相关中位数 {corr_med:.4f} → 否决前置"
          f"{'❌ 触发（>%.2f）' % CORR_VETO if veto else '✅ 未触发'}", flush=True)

    base_cs = results["fuse_w0.00"]["combined_sharpe"]
    base_nav = navs["fuse_w0.00"]

    # ── 判据 ──
    criteria: Dict[str, Any] = {}
    thr = TB.sharpe_needed_for_dsr95(N_TRIALS_OVERRIDE)
    for lab, r in results.items():
        if r["kind"] == "placebo":
            continue
        w = r["w_gbm"]
        plc = [v["combined_sharpe"] for k, v in results.items()
               if v["kind"] == "placebo" and v["w_gbm"] == w]
        rw = rolling_win_rate(navs[lab], base_nav)
        n_obs = len(navs[lab])
        dsr = _dsr_of(r["combined_sharpe"], N_TRIALS_OVERRIDE, n_obs)
        c: Dict[str, Any] = {
            "delta_combined": round(r["combined_sharpe"] - base_cs, 4),
            "a_main_combined_ge_base_plus_0.05":
                ((r["combined_sharpe"] - base_cs) >= 0.05) if args.window == "main" else None,
            "c_oos_combined_ge_base_minus_0.02":
                ((r["combined_sharpe"] - base_cs) >= -0.02) if args.window == "oos" else None,
            # ⚠️ 语义必须与预注册 §10.2 一致：「随机必须被击败」= 规则臂夏普
            #    **高于所有**同参数 placebo（严格口径，取 max 而非 mean）。
            #    曾经的实现是 (min(plc) > 规则臂) —— 语义完全相反（规则臂越好越不通过）。
            "c_beats_placebo": (r["combined_sharpe"] > max(plc)) if plc else None,
            "placebo_max": max(plc) if plc else None,
            "placebo_mean": round(float(np.mean(plc)), 4) if plc else None,
            "placebo_n": len(plc),
            "d_rolling_win_rate_ge_0.60":
                ((rw or {}).get("win_rate", 0.0) >= 0.60) if rw else None,
            "rolling": rw,
            "e_dsr_ge_0.95": dsr >= 0.95,
            "dsr": round(dsr, 6),
            "dsr95_threshold_sharpe": round(thr, 4),
            "n_trials_used": N_TRIALS_OVERRIDE,
        }
        criteria[lab] = c
        print(f"[p18] 判据 {lab}: Δ{c['delta_combined']:+.4f} "
              f"placebo_max={c['placebo_max']} "
              f"DSR={c['dsr']:.4f}(门槛 {thr:.4f}) "
              f"滚动胜率={(rw or {}).get('win_rate')}", flush=True)

    ranking = sorted(results.keys(),
                     key=lambda l: -results[l]["combined_sharpe"])

    # ── 候选臂（两阶段：main 选出 → oos 验证）──
    rule_labels = [l for l in ranking if results[l]["kind"] == "rule"]
    candidate = rule_labels[0] if rule_labels else None
    if args.window == "oos":
        # ⚠️ 两阶段：候选臂**必须**在 main（选择窗口）上选定。
        #    在 oos 上挑最优臂 = 用验证集选模型 = 过拟合，判据 (b)(c)(d) 会假阳性。
        cand_main = carry.get("verdict", {}).get("candidate_main_best_rule_arm")
        if not cand_main:
            mr = [l for l in carry.get("ranking", []) if l.startswith("fuse_w")]
            cand_main = mr[0] if mr else None
        if cand_main and cand_main in criteria:
            candidate = cand_main
        print(f"[p18] 候选臂 = {candidate}（在 main 上选定，oos 不改选）", flush=True)

    # ── 总判定 ──
    # ⚠️ 判据 (e)「DSR ≥ 0.95」**不是判据，是恒假条件**：
    #    n_trials=5 的实验口径门槛是 2.1964，而组合夏普满打满算 1.4114
    #    → 任何臂都不可能达到 → 把它当否决条件 = 实验永远判死，无信息。
    #    故降级为**证据强度报告项**（如实报 DSR 值），不参与通过判定。
    #    （此修正发生在**正式跑之前**，冒烟不算正式运行 → 不构成事后放宽。）
    verdict: Dict[str, Any] = {
        "candidate_main_best_rule_arm": candidate,
        "e_dsr_is_report_only": True,
        "e_note": "DSR≥0.95 在本实验任何可达夏普下都不可能（门槛 2.1964 vs 组合 1.41）"
                  "→ 降级为证据强度报告项，不作否决条件",
    }
    if candidate:
        cc = criteria[candidate]
        a_main = cc["a_main_combined_ge_base_plus_0.05"]
        if a_main is None and args.window == "oos":
            # ⚠️ (a) 是 main 窗口的判据 → oos 阶段必须从 main 产物读，
            #    否则 a_main=None → bool(None)=False → all_pass 恒 False。
            a_main = (carry.get("criteria", {}).get(candidate, {})
                      .get("a_main_combined_ge_base_plus_0.05"))
        verdict.update({
            "a_main_ge_base_plus_0.05": a_main,
            "candidate_selected_on": "main" if args.window == "oos" else "current",
            "c_beats_placebo": cc["c_beats_placebo"],
            "d_rolling_win_rate_ge_0.60": cc["d_rolling_win_rate_ge_0.60"],
            "dsr_report": cc["dsr"],
            "dsr95_threshold_sharpe": thr,
            "delta_combined": cc["delta_combined"],
        })
    if args.window == "oos":
        # (b) OOS 组合夏普 ≥ 基线 − 0.02
        if candidate:
            verdict["b_oos_ge_base_minus_0.02"] = \
                criteria[candidate]["c_oos_combined_ge_base_minus_0.02"]
        # (f) 跨窗口迁移：oos 最优臂在 main 的规则臂排名中不得最差
        carry_rank = carry.get("ranking", [])
        main_rank_for_f = [l for l in carry_rank
                           if l.startswith("fuse_w") and l != "fuse_w0.00"]
        if rule_labels and main_rank_for_f:
            best_oos = rule_labels[0]
            pos = (main_rank_for_f.index(best_oos) + 1
                   if best_oos in main_rank_for_f else None)
            verdict["f_cross_window"] = {
                "best_oos_arm": best_oos,
                "rank_in_main": pos,
                "n_rule_arms_in_main": len(main_rank_for_f),
                "n_rule_arms_in_oos": len(rule_labels),
                "not_worst_in_main": bool(pos is not None
                                          and pos < len(main_rank_for_f)),
            }
        verdict["all_pass"] = all(bool(x) for x in [
            verdict.get("a_main_ge_base_plus_0.05"),
            verdict.get("b_oos_ge_base_minus_0.02"),
            verdict.get("c_beats_placebo"),
            verdict.get("d_rolling_win_rate_ge_0.60"),
            (verdict.get("f_cross_window") or {}).get("not_worst_in_main"),
        ])
    else:
        verdict["all_pass"] = None  # 需 oos 阶段判定
    print(f"[p18] 总判定: {verdict}", flush=True)

    # ── 产物 ──
    payload = {
        "experiment": "P18_fuse_gbm",
        "window": args.window,
        "start": s_str,
        "end": e_str,
        "prereg": {
            "w_gbms": W_GBMS,
            "placebo_seeds": PLACEBO_SEEDS,
            "n_trials_override": N_TRIALS_OVERRIDE,
            "corr_veto": CORR_VETO,
            "gbm_cfg": GBM_CFG,
            "fusion": "截面 z-score 后线性加权",
            "criteria_semantics": {
                "a": "main 组合夏普 ≥ 基线 + 0.05",
                "b": "oos 组合夏普 ≥ 基线 − 0.02",
                "c": "规则臂组合夏普 > 同参数 placebo 的**最大值**（随机必须被击败）",
                "d": "滚动 504 日夏普胜率 ≥ 60%",
                "e": "DSR ≥ 0.95 —— **降级为报告项**（n_trials=5 的门槛 2.1964 "
                     "高于任何可达夏普，作否决条件等于恒假）",
                "f": "oos 最优权重在 main 的规则臂排名中不得最差",
            },
            "source": "docs/backtest/MULTI_ASSET_LEDGER_2026-10-31.md §十",
        },
        "methodology": {
            "a_share": "honest_composite 25 公式版, lag=1, topk60/ndrop5, "
                       "真实盘口径(enable_tradability=True, 佣金0.03%+滑点0.01%)",
            "sleeve": "P5 固定持仓集(gold/bond/commodity 各 3 只), 类内等权, 每月 21 交易日回类间等权",
            "combine": "merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])",
            "gbm_note": "gbm 分数沿交易日历右移 1 天对齐到执行日；vol-target 关闭",
            "a_share_codes": len(codes),
            "calendar_days": len(calendar),
        },
        "sleeve": {
            "sharpe": round(sharpe(sleeve_nav), 4),
            "vol": round(P2.nav_vol(sleeve_nav), 4),
            "turnover_ratio": round(float(srec["turnover_ratio"]), 4),
            "days": len(sleeve_nav),
        },
        "diagnostics": diag,
        "corr_median": round(corr_med, 4),
        "veto_triggered": veto,
        "base_combined_sharpe": base_cs,
        "results": results,
        # 净值序列入库 → 以后修判据口径只需**重算**，不必重跑回测
        "navs": {lab: [[d, round(v, 10)] for d, v in navs[lab]]
                 for lab in navs},
        "criteria": criteria,
        "verdict": verdict,
        "ranking": ranking,
        "dsr95_threshold_sharpe": round(thr, 4),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[p18] 已写入 {out_path}", flush=True)
    print(f"[p18] 排名 {ranking}", flush=True)


if __name__ == "__main__":
    main()
