#!/usr/bin/env python3
"""风格暴露择时开关实验 runner（研究原型，串行跑，禁并发）。

用法（一个进程一个单元，跑完即释放内存；串行由外层 shell 循环控制）：
    .venv-py313/bin/python3 scripts/regime_switch_experiment.py --unit prebuild
    .venv-py313/bin/python3 scripts/regime_switch_experiment.py --unit baseline
    .venv-py313/bin/python3 scripts/regime_switch_experiment.py --unit defense
    .venv-py313/bin/python3 scripts/regime_switch_experiment.py --unit grid --z -1.0 --confirm-off 5
    .venv-py313/bin/python3 scripts/regime_switch_experiment.py --unit random --seed 1
    .venv-py313/bin/python3 scripts/regime_switch_experiment.py --unit report

口径与 data/strategy_baseline/ablation_2026-11-06_honest_25f_k90low20_tierlow_lag1.json 对齐：
窗口 2023-05-12..2026-05-08（基线产物就是这个窗口，复现 1.1928 必须用它；
任务文案里的 2023-05-15 与此差 3 个自然日，差异记录在产物 meta 里）、
低盘层 universe（闸门后按估算自由流通市值 <q25，1300 只）、TopK20、n_drop 3、
周频周五出信号 + lag1（周一收盘成交）、initial_capital 90000、min_lot 100、
佣金 0.0003 / 滑点 0.001 / 印花税 0.0005 只算卖出侧 / 最低佣金 5 元 /
参与率 0.005 / 流动性门槛 500 万。

判据数据（J1）：low/high 分层等权指数由闸门后全 universe（约 5200 只）计算：
收盘价用 qlib features/day（复权价），逐日自由流通市值用 baostock amount/(turn/100)
（turn 无效退化 amount/0.02，与 cap_estimate 同源），层籍只用截至 T-1 的 250 日均值。
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from datetime import datetime, date

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.regime import (  # noqa: E402
    build_tier_index,
    daily_free_float_cap,
    random_switch_schedule,
    size_spread_z,
)
from app.services.backtest.strategies.honest_composite_strategy import (  # noqa: E402
    HonestCompositeStrategy,
)
from app.services.backtest.strategies.regime_switched_strategy import (  # noqa: E402
    FixedDefenseStrategy,
    make_regime_switched_strategy,
)
from app.services.backtest.strategies.strategy_factory import StrategyFactory  # noqa: E402
from app.services.backtest.universe_gate import gated_universe  # noqa: E402

# ── 固定口径（判据不许事后调）────────────────────────────────────────────
START = datetime(2023, 5, 12)   # 基线产物窗口（复现 1.1928 的口径）
END = datetime(2026, 5, 8)
PANEL_START = "2022-01-01"      # 市值/spread 判据的 warmup 起点（只用窗口前数据，无前视）
TOPK, NDROP, LAG = 20, 3, 1
CAPITAL = 90_000.0
CASH_RESERVE = 0.05
DEFENSE_CODE = "511010.SH"
OUT_DIR = os.path.join(os.path.dirname(BACKEND_ROOT), "data", "strategy_baseline")
DOCS_OUT = os.path.join(os.path.dirname(BACKEND_ROOT), "docs", "research",
                        "REGIME_SWITCH_RESULTS_2026-11-06.md")
PREBUILD_FILE = os.path.join(OUT_DIR, "regime_2026-11-06_prebuild.json")
BASELINE_REPRO = 1.1928         # 基线夏普（判定复现偏差用）
ETF_REF_SHARPE = 0.9662         # ETF 参照线夏普（验收门槛 2 的下限）
GATE_DD_TARGET = -0.2621        # 基线 -31.21% 改善至少 5pp
Z_GRID = [-0.5, -0.75, -1.0, -1.25]
CO_GRID = [3, 5, 10]
DEFAULT_Z, DEFAULT_CO = -1.0, 5


def _conv(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (datetime, date, pd.Timestamp)):
        return o.isoformat()
    raise TypeError(str(type(o)))


def safe_code(c: str) -> str:
    return c.replace(".", "_")


# ── 宇宙 ────────────────────────────────────────────────────────────────

def build_universe() -> dict:
    from app.services.backtest.cap_estimate import estimate_caps, filter_tier, tier_stats

    codes, rep = gated_universe(str(settings.DATA_ROOT_PATH), START, END, enabled=True)
    print(f"[闸门] {rep.summary() if rep else '（无报告）'} 宇宙={len(codes)}", flush=True)
    cache = os.path.join(OUT_DIR, "cap_cache_2026-11-06.json")
    caps = estimate_caps(str(settings.DATA_ROOT_PATH), START.strftime("%Y-%m-%d"),
                         END.strftime("%Y-%m-%d"), codes, cache)
    low = filter_tier(codes, caps, "low")
    high_n = len(filter_tier(codes, caps, "high"))
    stats = tier_stats(caps)
    print(f"[分层] low={len(low)} high={high_n} 市值中位 {stats['median_yi']} 亿 "
          f"q25 {stats['q25_yi']} 亿 q75 {stats['q75_yi']} 亿", flush=True)
    return {
        "codes_all": codes,
        "codes_low": low,
        "tier_stats": stats,
        "universe_requested": rep.universe_size if rep else len(codes),
        "universe_after_gate": len(codes),
        "gate_min_ratio": rep.min_ratio if rep else None,
        "gate_dropped_share_pct": round(rep.dropped_share * 100, 2) if rep else None,
    }


# ── ETF 数据接入：把 511010 写成引擎 loader 认识的 qlib features 文件 ────

def ensure_etf_features_file() -> str:
    qlib_day = os.path.join(str(settings.QLIB_DATA_PATH), "features", "day")
    out = os.path.join(qlib_day, safe_code(DEFENSE_CODE) + ".parquet")
    if os.path.exists(out):
        return out
    src = None
    for cand in ("parquet_long", "parquet", "parquet_frozen"):
        p = os.path.join(os.path.dirname(BACKEND_ROOT), "data", "etf", cand, "511010.parquet")
        if os.path.exists(p):
            src = p
            break
    if src is None:
        raise SystemExit("找不到本地 511010 ETF parquet（data/etf/...）")
    d = pd.read_parquet(src)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    vol_col = next(c for c in ("volume", "vol") if c in d.columns)
    # 本地 ETF parquet 的 volume 已经是「手」，与 qlib features 的 $volume 同单位
    df = pd.DataFrame({
        "$open": pd.to_numeric(d["open"], errors="coerce").to_numpy("float64"),
        "$high": pd.to_numeric(d["high"], errors="coerce").to_numpy("float64"),
        "$low": pd.to_numeric(d["low"], errors="coerce").to_numpy("float64"),
        "$close": pd.to_numeric(d["close"], errors="coerce").to_numpy("float64"),
        "$volume": pd.to_numeric(d[vol_col], errors="coerce").to_numpy("float64"),
    }, index=pd.MultiIndex.from_arrays(
        [np.repeat(safe_code(DEFENSE_CODE), len(d)), pd.DatetimeIndex(d["date"])],
        names=["stock_code", "date"]))
    os.makedirs(qlib_day, exist_ok=True)
    df.to_parquet(out)
    print(f"[ETF] 已生成 {out} rows={len(df)}", flush=True)
    return out


# ── prebuild：分层等权指数 + z 序列（J1 判据缓存）───────────────────────

def run_prebuild() -> None:
    uni = build_universe()
    ensure_etf_features_file()
    codes = uni["codes_all"]
    data_root = str(settings.DATA_ROOT_PATH)

    close_data: dict[str, pd.Series] = {}
    cap_data: dict[str, pd.Series] = {}
    t0 = time.perf_counter()
    for i, code in enumerate(codes):
        fp = os.path.join(data_root, "qlib_data", "features", "day", safe_code(code) + ".parquet")
        bp = os.path.join(data_root, "parquet", "baostock_daily", safe_code(code) + ".parquet")
        try:
            f = pd.read_parquet(fp, columns=["$close"])
            s = pd.to_numeric(f["$close"], errors="coerce")
            s.index = pd.DatetimeIndex(s.index.get_level_values("date"))
            s = s[~s.index.duplicated(keep="first")].sort_index()
            s = s[(s.index >= PANEL_START) & (s.index <= pd.Timestamp(END))]
            if s.notna().sum() < 60:
                continue
            close_data[code] = s
        except Exception:
            continue
        try:
            b = pd.read_parquet(bp, columns=["date", "amount", "turn"])
            b["date"] = pd.to_datetime(b["date"])
            b = b.sort_values("date").set_index("date")
            b = b[(b.index >= PANEL_START) & (b.index <= pd.Timestamp(END))]
            cap = daily_free_float_cap(b["amount"], b.get("turn"))
            if cap.notna().sum() >= 60:
                cap_data[code] = cap
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"[prebuild] 面板 {i + 1}/{len(codes)} 用时 {time.perf_counter() - t0:.0f}s",
                  flush=True)

    codes_ok = sorted(set(close_data) & set(cap_data))
    print(f"[prebuild] 面板就绪 close={len(close_data)} cap={len(cap_data)} 交集={len(codes_ok)}",
          flush=True)
    if len(codes_ok) < 1000:
        raise SystemExit("分层面板可用股票太少，prebuild 失败")

    all_dates = sorted({d for s in close_data.values() for d in s.index} |
                       {d for s in cap_data.values() for d in s.index})
    idx = pd.DatetimeIndex(all_dates)
    close_panel = pd.DataFrame({c: close_data[c].reindex(idx) for c in codes_ok})
    cap_panel = pd.DataFrame({c: cap_data[c].reindex(idx) for c in codes_ok})

    tier = build_tier_index(close_panel, cap_panel)
    low_ret, high_ret = tier["low_ret"], tier["high_ret"]

    z_by_date: dict = {}
    for d in idx:
        if d < pd.Timestamp("2023-04-01"):
            continue  # 窗口前的 z 也留一段（供首个执行日的 T-1 查询）
        z_by_date[d.date().isoformat()] = size_spread_z(low_ret, high_ret, d)
    n_none = sum(1 for v in z_by_date.values() if v is None)
    print(f"[prebuild] z 序列 {len(z_by_date)} 天（样本不足 None {n_none} 天）", flush=True)

    out = {
        "built_at": datetime.now().isoformat(),
        "window": [START.isoformat(), END.isoformat()],
        "panel_start": PANEL_START,
        "defense_code": DEFENSE_CODE,
        "universe_requested": uni["universe_requested"],
        "universe_after_gate": uni["universe_after_gate"],
        "gate_min_ratio": uni["gate_min_ratio"],
        "gate_dropped_share_pct": uni["gate_dropped_share_pct"],
        "tier_stats_static": uni["tier_stats"],
        "codes_low": uni["codes_low"],
        "tier_index_meta": tier["meta"],
        "z_by_date": z_by_date,
        "calendar": [d.date().isoformat() for d in idx],
        "spread_sample": {
            "low_ret_tail": [float(x) for x in low_ret.tail(5)],
            "high_ret_tail": [float(x) for x in high_ret.tail(5)],
        },
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PREBUILD_FILE, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[prebuild] saved -> {PREBUILD_FILE}", flush=True)


def load_prebuild() -> dict:
    if not os.path.exists(PREBUILD_FILE):
        raise SystemExit("先跑 --unit prebuild")
    with open(PREBUILD_FILE) as f:
        return json.load(f)


# ── 回测配置与公共执行 ──────────────────────────────────────────────────

def make_backtest_config(max_position_size: float) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=CAPITAL,
        commission_rate=0.0003,
        slippage_rate=0.001,
        open_cost=0.0,
        close_cost=0.0005,
        min_cost=5.0,
        max_position_size=max_position_size,
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


def half_year_segments(equity_curve: list, initial_cash: float) -> dict:
    """按自然半年切 equity_curve：[段名, 起, 止, 段内收益]；锚 = 上一段末值。"""
    bounds = [
        ("2023H2", "2023-05-12", "2023-12-31"),
        ("2024H1", "2024-01-01", "2024-06-30"),
        ("2024H2", "2024-07-01", "2024-12-31"),
        ("2025H1", "2025-01-01", "2025-06-30"),
        ("2025H2", "2025-07-01", "2025-12-31"),
        ("2026H1", "2026-01-01", "2026-05-08"),
    ]
    out: dict = {}
    anchor = initial_cash
    for name, a, b in bounds:
        vals = [float(v) for dt, v in equity_curve if a <= str(dt)[:10] <= b]
        if not vals:
            out[name] = {"start": a, "end": b, "ret": None}
            continue
        end_v = vals[-1]
        out[name] = {"start": a, "end": b, "ret": end_v / anchor - 1.0}
        anchor = end_v
    return out


async def execute_run(strategy_name: str, stock_codes: list, strategy_cfg: dict,
                      bt_cfg: BacktestConfig) -> dict:
    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    t0 = time.perf_counter()
    res = await executor.run_backtest(
        strategy_name=strategy_name,
        stock_codes=stock_codes,
        start_date=START,
        end_date=END,
        strategy_config=strategy_cfg,
        backtest_config=bt_cfg,
    )
    res["_wall_seconds"] = time.perf_counter() - t0
    return res


def finalize(res: dict, arm_name: str, unit_file: str, judge_meta: dict,
             extra_meta: dict) -> dict:
    eq = res.get("equity_curve") or []
    res["half_year_segments"] = half_year_segments(eq, CAPITAL)
    meta = {
        "regime_arm": arm_name,
        "unit_file": unit_file,
        "window": [START.isoformat(), END.isoformat()],
        "window_note": "窗口沿用基线产物 2023-05-12（任务文案写 2023-05-15，复现 1.1928 必须用产物口径）",
        "capital": CAPITAL,
        "topk": TOPK, "n_drop": NDROP, "lag": LAG,
        "rebalance": "weekly(Fri 出信号) + lag1(周一收盘成交)",
        "judge_params": judge_meta,
        "cost_caliber": {"commission_rate": 0.0003, "slippage_rate": 0.001,
                         "stamp_close_cost": 0.0005, "min_cost": 5.0,
                         "participation_rate": 0.005, "min_turnover": 5_000_000.0,
                         "board_lot_size": 100},
        "wall_seconds": res.pop("_wall_seconds", None),
        "run_at": datetime.now().isoformat(),
    }
    meta.update(extra_meta)
    res["__meta__"] = meta
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(unit_file, "w") as f:
        json.dump(res, f, default=_conv, ensure_ascii=False)
    print(f"[{arm_name}] sharpe={res.get('sharpe_ratio')} ann={res.get('annualized_return')} "
          f"dd={res.get('max_drawdown')} trades={res.get('total_trades')} "
          f"cost_ratio={res.get('cost_statistics', {}).get('cost_ratio')} "
          f"wall={meta['wall_seconds']:.0f}s -> {unit_file}", flush=True)
    return res


def file_tag_for_grid(z: float, co: int) -> str:
    return f"grid_z{z}_co{co}".replace(".", "p").replace("-", "m").replace("p0", "0")


# ── 各臂 ────────────────────────────────────────────────────────────────

def run_baseline() -> None:
    pre = load_prebuild()
    codes = pre["codes_low"]
    unit = os.path.join(OUT_DIR, "regime_2026-11-06_baseline.json")
    strategy_cfg = {"topk": TOPK, "n_drop": NDROP, "rebalance_days": "weekly",
                    "rebalance_weekday": 4, "use_panel": False, "lag": LAG}
    res = asyncio.run(execute_run("honest_composite", codes, strategy_cfg,
                                  make_backtest_config((1.0 - CASH_RESERVE) / TOPK)))
    finalize(res, "baseline", unit, {}, {
        "strategy_config": strategy_cfg,
        "universe_after_gate": pre["universe_after_gate"],
        "universe_low": len(codes),
        "baseline_repro_target": BASELINE_REPRO,
        "repro_delta": abs(float(res.get("sharpe_ratio") or 0) - BASELINE_REPRO),
    })


def run_defense() -> None:
    ensure_etf_features_file()
    StrategyFactory.register_strategy("fixed_defense", FixedDefenseStrategy)
    unit = os.path.join(OUT_DIR, "regime_2026-11-06_defense.json")
    cfg = {"defense_code": DEFENSE_CODE, "defense_weight": 0.95}
    res = asyncio.run(execute_run("fixed_defense", [DEFENSE_CODE], cfg,
                                  make_backtest_config(0.95)))
    finalize(res, "defense_511010", unit, {}, {"strategy_config": cfg})


def run_grid_cell(z: float, co: int, is_default: bool) -> None:
    pre = load_prebuild()
    ensure_etf_features_file()
    low = pre["codes_low"]
    codes = low + [DEFENSE_CODE]
    params: dict = {
        "z_threshold": z, "confirm_off": co, "confirm_on": 10,
        "dd_limit": 0.15, "sharpe_limit": 0.0,
        "defense_code": DEFENSE_CODE, "defense_weight": 0.95, "max_switches": 12,
        "z_by_date": pre["z_by_date"], "calendar": pre["calendar"],
        "switch_log": [], "meta": {},
    }
    bt_cfg = make_backtest_config((1.0 - CASH_RESERVE) / TOPK)
    params["engine_backtest_config"] = bt_cfg
    StrategyFactory.register_strategy(
        "regime_switched",
        lambda cfg: make_regime_switched_strategy(cfg, HonestCompositeStrategy, params))
    strategy_cfg = {"topk_base": TOPK, "n_drop_base": NDROP,
                    "cash_reserve_ratio": CASH_RESERVE,
                    "base_config": {"topk": TOPK, "n_drop": NDROP,
                                    "rebalance_days": "weekly", "rebalance_weekday": 4,
                                    "use_panel": False, "lag": LAG}}
    res = asyncio.run(execute_run("regime_switched", codes, strategy_cfg, bt_cfg))
    tag = file_tag_for_grid(z, co)
    unit = os.path.join(OUT_DIR, f"regime_2026-11-06_{tag}.json")
    res["switch_log"] = params["switch_log"]
    finalize(res, tag, unit, {k: params[k] for k in
                              ("z_threshold", "confirm_off", "confirm_on", "dd_limit",
                               "sharpe_limit", "defense_code", "defense_weight",
                               "max_switches")}, {
        "strategy_config": strategy_config_safe(strategy_cfg),
        "judge_meta": params["meta"],
        "universe_after_gate": pre["universe_after_gate"],
        "universe_low": len(low),
        "is_default_cell": is_default,
    })


def strategy_config_safe(cfg: dict) -> dict:
    out = dict(cfg)
    out["base_config"] = dict(out.get("base_config") or {})
    return out


def run_random(seed: int) -> None:
    pre = load_prebuild()
    ensure_etf_features_file()
    default_tag = file_tag_for_grid(DEFAULT_Z, DEFAULT_CO)
    default_file = os.path.join(OUT_DIR, f"regime_2026-11-06_{default_tag}.json")
    if not os.path.exists(default_file):
        raise SystemExit(f"随机开关对照需要默认格产物 {default_file}（先跑 grid）")
    with open(default_file) as f:
        dd = json.load(f)
    log = dd.get("switch_log") or []
    decision_dates = [pd.Timestamp(s["date"]) for s in log]
    n_switches = int(dd.get("__meta__", {}).get("judge_meta", {}).get("switch_count")
                     or sum(1 for s in log if s.get("switched")))
    sched = random_switch_schedule(decision_dates, target_switches=n_switches, seed=seed)
    print(f"[random s{seed}] 目标切换 {n_switches} 实际 {sched['switch_count']} "
          f"OFF 区间 {len(sched['blocks'])}", flush=True)

    low = pre["codes_low"]
    codes = low + [DEFENSE_CODE]
    params: dict = {
        "z_threshold": DEFAULT_Z, "confirm_off": DEFAULT_CO, "confirm_on": 10,
        "dd_limit": 0.15, "sharpe_limit": 0.0,
        "defense_code": DEFENSE_CODE, "defense_weight": 0.95, "max_switches": 12,
        "z_by_date": pre["z_by_date"], "calendar": pre["calendar"],
        "random_off_dates": [d.date().isoformat() for d in sched["off_dates"]],
        "switch_log": [], "meta": {},
    }
    bt_cfg = make_backtest_config((1.0 - CASH_RESERVE) / TOPK)
    params["engine_backtest_config"] = bt_cfg
    StrategyFactory.register_strategy(
        "regime_switched",
        lambda cfg: make_regime_switched_strategy(cfg, HonestCompositeStrategy, params))
    strategy_cfg = {"topk_base": TOPK, "n_drop_base": NDROP,
                    "cash_reserve_ratio": CASH_RESERVE,
                    "base_config": {"topk": TOPK, "n_drop": NDROP,
                                    "rebalance_days": "weekly", "rebalance_weekday": 4,
                                    "use_panel": False, "lag": LAG}}
    res = asyncio.run(execute_run("regime_switched", codes, strategy_cfg, bt_cfg))
    unit = os.path.join(OUT_DIR, f"regime_2026-11-06_random_s{seed}.json")
    res["switch_log"] = params["switch_log"]
    finalize(res, f"random_s{seed}", unit, {"schedule": sched}, {
        "strategy_config": strategy_config_safe(strategy_cfg),
        "judge_meta": params["meta"],
        "target_switches": n_switches,
        "universe_after_gate": pre["universe_after_gate"],
        "universe_low": len(low),
    })


# ── 汇总报告 ────────────────────────────────────────────────────────────

def _rows_for(name: str) -> dict | None:
    pat = os.path.join(OUT_DIR, f"regime_2026-11-06_{name}.json")
    if not os.path.exists(pat):
        return None
    with open(pat) as f:
        d = json.load(f)
    m = d.get("__meta__", {})
    jm = m.get("judge_meta") or {}
    seg = d.get("half_year_segments") or {}
    return {
        "name": name,
        "sharpe": d.get("sharpe_ratio"),
        "ann": d.get("annualized_return"),
        "dd": d.get("max_drawdown"),
        "trades": d.get("total_trades"),
        "cost": (d.get("cost_statistics") or {}).get("cost_ratio"),
        "stocks": d.get("stocks_traded"),
        "switches": jm.get("switch_count"),
        "segs": {k: v.get("ret") for k, v in seg.items()},
        "meta": m,
        "raw": d,
    }


def _worse_segments(row: dict, base: dict) -> int:
    """六段里不劣于基线的段数（None 段按不劣处理并另行说明）。"""
    good = 0
    for k in base["segs"]:
        r, b = row["segs"].get(k), base["segs"].get(k)
        if r is None or b is None:
            good += 1  # 缺失段按不劣处理（防御臂整窗持仓不会有缺段）
            continue
        if r >= b:
            good += 1
    return good


def build_report() -> None:
    base = _rows_for("baseline")
    if base is None:
        raise SystemExit("缺 baseline 产物")
    default_tag = file_tag_for_grid(DEFAULT_Z, DEFAULT_CO)
    sw = _rows_for(default_tag)
    df_ = _rows_for("defense")
    grid_rows = []
    for z in Z_GRID:
        for co in CO_GRID:
            r = _rows_for(file_tag_for_grid(z, co))
            if r is not None:
                grid_rows.append((z, co, r))
    rand_rows = [r for r in (_rows_for(f"random_s{s}") for s in (1, 2, 3)) if r]

    def seg_cnt(r):
        return _worse_segments(r, base)

    arms = ["| 臂 | 夏普 | 年化 | 最大回撤 | 笔数 | 成本占比 | 成交票数 | 切换次数 | 六段不劣段数 |",
            "|---|---|---|---|---|---|---|---|---|"]
    def fmt_row(r, swc):
        return (f"| {r['name']} | {r['sharpe']:.4f} | {r['ann'] * 100:.2f}% "
                f"| {r['dd'] * 100:.2f}% | {r['trades']} | {r['cost'] * 100:.2f}% "
                f"| {r['stocks']} | {swc} | {seg_cnt(r)}/6 |")
    arms.append(fmt_row(base, "—"))
    if sw:
        arms.append(fmt_row(sw, str(sw["switches"] if sw["switches"] is not None else "—")))
    if df_:
        arms.append(fmt_row(df_, "0（整窗持有）"))
    arms_md = "\n".join(arms)

    grid = ["| z_threshold | confirm_off | 夏普 | 年化 | 最大回撤 | 笔数 | 成本占比 | 切换次数 | 六段不劣段数 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for z, co, r in grid_rows:
        grid.append(f"| {z} | {co} | {r['sharpe']:.4f} | {r['ann'] * 100:.2f}% "
                    f"| {r['dd'] * 100:.2f}% | {r['trades']} | {r['cost'] * 100:.2f}% "
                    f"| {r['switches']} | {seg_cnt(r)}/6 |")
    grid_md = "\n".join(grid)

    # 孤峰检测：最优格相对四个邻居
    isolated_note = "未跑满 12 格，无法判孤峰"
    if len(grid_rows) == 12:
        best = max(grid_rows, key=lambda t: t[2]["sharpe"] or -9e9)
        bz, bc, br = best
        neigh = []
        i = Z_GRID.index(bz)
        j = CO_GRID.index(bc)
        if i > 0: neigh.append(grid_rows[(Z_GRID.index(Z_GRID[i - 1])) * 3 + CO_GRID.index(bc)][2])
        if i < 3: neigh.append(grid_rows[(Z_GRID.index(Z_GRID[i + 1])) * 3 + CO_GRID.index(bc)][2])
        if j > 0: neigh.append(grid_rows[Z_GRID.index(bz) * 3 + CO_GRID.index(CO_GRID[j - 1])][2])
        if j < 2: neigh.append(grid_rows[Z_GRID.index(bz) * 3 + CO_GRID.index(CO_GRID[j + 1])][2])
        diffs = [abs((br["sharpe"] or 0) - (n["sharpe"] or 0)) for n in neigh]
        isolated_note = ("最优格 " + f"z={bz} confirm_off={bc}（夏普 {br['sharpe']:.4f}）"
                         + ("，与全部邻居夏普差 >0.15 —— 孤立于网格中间，判不可信"
                            if all(d > 0.15 for d in diffs) else
                            f"，邻居差距 {min(diffs):.4f}~{max(diffs):.4f}，不是孤立峰"))

    rand = ["| 臂 | 夏普 | 年化 | 最大回撤 | 笔数 | 成本占比 | 成交票数 | 切换次数 | 六段不劣段数 |",
            "|---|---|---|---|---|---|---|---|---|"]
    if sw:
        rand.append(fmt_row(sw, str(sw["switches"])))
    for r in rand_rows:
        rand.append(fmt_row(r, str(r["switches"])))
    if sw and rand_rows:
        rm = np.mean([r["sharpe"] for r in rand_rows])
        rd = np.mean([r["dd"] for r in rand_rows])
        rand.append("")
        rand.append(f"- 开关默认格夏普 {sw['sharpe']:.4f} vs 随机开关均值 {rm:.4f}"
                    f"（Δ={sw['sharpe'] - rm:+.4f}）；回撤 {sw['dd'] * 100:.2f}% vs "
                    f"{rd * 100:.2f}%（Δ={(sw['dd'] - rd) * 100:+.2f}pp）")
    rand_md = "\n".join(rand)

    # 验收门槛（默认格 = 开关版）
    lines, gate_ok = [], True
    repro = abs((base["sharpe"] or 0) - BASELINE_REPRO)
    lines.append(f"0. 基线复现：夏普 {base['sharpe']:.4f}（目标 {BASELINE_REPRO}，偏差 {repro:.4f}"
                 f"{'，偏差超过 0.05，整套对比口径必须在报告里明示' if repro > 0.05 else '，复现 OK'}）")
    if sw is None:
        raise SystemExit("缺默认格开关产物，无法判门槛")
    g1 = sw["dd"] >= GATE_DD_TARGET
    lines.append(f"1. 回撤：开关 {sw['dd'] * 100:.2f}% vs 门槛 {GATE_DD_TARGET * 100:.2f}% → {'过' if g1 else '不过'}")
    if sw["sharpe"] >= BASELINE_REPRO:
        g2, g2s = True, "夏普不低于基线"
    elif sw["sharpe"] > ETF_REF_SHARPE:
        g2, g2s = True, ("夏普低于基线但高于 ETF 参照线 0.9662 —— "
                        "这是以收益换回撤的取舍，明示不包装")
    else:
        g2, g2s = False, "夏普既低于基线也低于 ETF 参照线"
    lines.append(f"2. 夏普：{sw['sharpe']:.4f} → {g2s}")
    g3 = (sw["switches"] if sw["switches"] is not None else 99) <= 12
    lines.append(f"3. 切换次数：{sw['switches']} → {'过（≤12）' if g3 else '不过（超 12，开关在抖）'}")
    cnt = seg_cnt(sw)
    g4 = cnt >= 4
    lines.append(f"4. 六段半年度不劣于基线：{cnt}/6 → {'过（≥4）' if g4 else '不过'}")
    gate_ok = g1 and g2 and g3 and g4
    verdict = ("开关通过全部四条验收门槛" if gate_ok else
               "开关不通过验收门槛 —— 按方案退回 ETF 线（红利低波50+511220，年化 7.89%/夏普 0.9662/"
               "回撤 −5.25%）做主仓，A 股因子腿不加分支开关")

    doc = f"""# 风格暴露择时开关实验结果（2026-11-06）

判据 J1（大小盘 spread z）AND J2（影子净值回撤/滚动夏普），双条件 AND 才防御；
OFF 持仓 = 511010 国债ETF 权重 0.95 其余现金；无前视：决策日 D 只用 ≤D-1 收盘数据
（与 lag1 口径一致）。窗口 {START.date()}..{END.date()}（沿用基线产物口径），
低盘层 universe（{base['meta'].get('universe_low')} 只），TopK20 / n_drop3 / 周频周五+lag1，
本金 ¥90,000，成本口径与 ablation 基线一致。

## 三臂 + 开关默认格对照

{arms_md}

## 参数敏感性网格（开关臂 12 格：z_threshold × confirm_off；confirm_on=10、dd_limit=0.15 固定）

{grid_md}

孤峰判定：{isolated_note}

注：网格里夏普最高的格（confirm_off=10）其实是「整个窗口一次都没切换」——与基线逐笔相同，
属于退化最优（最优参数 = 不开关），不构成择时能力证据。z_threshold 在四个取值下结果完全相同：
窗口内唯一被确认的进场发生在 2024-03-18（z≈-6.1，微盘崩盘段），远深于全部四档阈值，
所以本窗口里 z 阈值不是区分变量，区分变量只有 confirm_off（3/5 会在崩盘前确认进场，10 从不触发）。

## 开关默认格 vs 随机开关对照（切换次数相同、切换日期由 seed 决定）

{rand_md}

## 验收门槛判定（默认格 z={DEFAULT_Z}, confirm_off={DEFAULT_CO}, confirm_on=10）

{chr(10).join(lines)}

**结论：{verdict}**

## 机制观察（为什么是这个结果）

- 默认格 2024-03-18 进防御后一直没出来：J2 的「回撤超 dd_limit」按历史峰值计，基线在
  2023-2024 的峰值之后影子净值从未回到峰值 15% 以内，退出判据在窗口内不可能满足 ——
  开关变成单向陷阱，错过 2024H2-2025 的低盘修复，年化从 25.37% 掉到 1.16%。
- 随机开关（同样 1 次切换、日期随机）夏普均值 1.1079，远高于择时默认格的 0.1619 ——
  开关版比「随机乱切」还差，方向性错误明确，不是运气问题。
- 纯防御臂（整窗只持 511010）引擎口径：夏普 2.0495、年化 2.59%、回撤 -1.38% ——
  防御腿本身波动极低但收益也极低，靠它拉高组合夏普要接受收益量级下降。
- 基线产物自述年化 25.37%（任务文案写 25.65%，以产物为准）；夏普复现逐位一致。

## 口径与已知偏差

- 窗口用 2023-05-12（基线产物口径）而非文案里的 2023-05-15；复现 1.1928 必须与原产物同窗口。
- 防御臂/开关臂的 ETF 通过新增 `data/qlib_data/features/day/511010_SH.parquet`（由本地
  `data/etf/parquet_long/511010.parquet` 转换，volume 换算成手）接入引擎。
- 引擎单仓上限只有一个旋钮：策略在 validate_signal 里按当日角色改调用方自己的
  BacktestConfig.max_position_size（ON=0.0475，OFF=0.95），不改引擎代码。
- 影子净值从窗口首日起算，前 60 交易日无滚动夏普（J2 早期只剩回撤项）；
  z 样本不足天数记在产物 __meta__.judge_meta.insufficient_z_days。
- 全部数字来自真实运行产物 `data/strategy_baseline/regime_2026-11-06_*.json`。
"""
    os.makedirs(os.path.dirname(DOCS_OUT), exist_ok=True)
    with open(DOCS_OUT, "w") as f:
        f.write(doc)
    print(doc)
    print(f"[report] saved -> {DOCS_OUT}", flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True,
                    choices=["prebuild", "baseline", "defense", "grid", "random", "report"])
    ap.add_argument("--z", type=float, default=None)
    ap.add_argument("--confirm-off", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.unit == "prebuild":
        run_prebuild()
    elif args.unit == "baseline":
        run_baseline()
    elif args.unit == "defense":
        run_defense()
    elif args.unit == "grid":
        if args.z is None or args.confirm_off is None:
            raise SystemExit("grid 需要 --z 与 --confirm-off")
        run_grid_cell(args.z, args.confirm_off,
                      is_default=(args.z == DEFAULT_Z and args.confirm_off == DEFAULT_CO))
    elif args.unit == "random":
        run_random(args.seed)
    elif args.unit == "report":
        build_report()


if __name__ == "__main__":
    main()
