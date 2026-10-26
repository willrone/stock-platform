#!/usr/bin/env python3
"""多策略纸面盘（策略赛马落地版，2026-10-25）。

每条策略一条独立线：各自 ¥1,000,000 起始资金、各自持仓、各自账本，
同一起始日、同一个全市场宇宙、同一成本模型（佣金万3 + 印花税万5 + 滑点千1
+ 现金预留 5% + 100 股整手）。

内置策略（全部为周频：周五收盘打分 → 下周一执行）:
  22f              22 因子等权（MIN +1、STD/CORR/CORD -1、MAX30/60 +1）
  honest_42f       42 因子等权复合（23 基础 + 17 Alpha158 面板），注册策略同源
  rsi_71421        RSI 7/14/21 均值回归 + 低波动 + 量价
  factor_ranking   动量(126d 跳 20d) / 低波动(60d) / 低振幅(10d) 等权 z

3 年全市场回测夏普（2026-10-25 实测，日频→周频）:
  honest_42f   0.718 → 0.965   回撤 -20.3% → -17.6%   交易 6660 → 1390
  rsi_71421    0.328 → 0.803   回撤 -26.7% → -16.3%   交易 4092 → 828
  factor_rank  0.720 → 0.768   回撤 -15.2% → -14.2%   交易 5054 → 1160

用法:
  # 生成所有线的信号（周五跑）
  env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/paper_portfolio.py
  # 只跑某条线
  ... paper_portfolio.py --strategies rsi_71421
  # 非周五强制跑
  ... paper_portfolio.py --force

输出:
  data/paper/<strategy>_holdings.json   每条线的持仓 + 现金
  data/paper/<strategy>_ledger.csv      每条线的逐笔账本
  data/paper/portfolio.json             全部线的汇总（供前端/MCP 读）
  frontend/public/paper-portfolio.json  前端静态 JSON
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

BACKEND_ROOT = "/Users/rone/Projects/stock-platform/backend"
ROOT = os.path.dirname(BACKEND_ROOT)
sys.path.insert(0, BACKEND_ROOT)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ds = _load("ds", os.path.join(BACKEND_ROOT, "scripts", "daily_signal.py"))
ws = _load("ws", os.path.join(BACKEND_ROOT, "scripts", "weekly_signal.py"))

# 实盘可成交性规则引擎（2026-10-25）—— 与回测 / 对外信号共用同一份实现
from app.services.trading.execution_rules import (  # noqa: E402
    DEFAULT_MIN_TURNOVER,
    DEFAULT_PARTICIPATION_RATE,
    check_tradability,
    limit_up_mask_matrix,
    load_st_flags,
)
from scripts.market_data import (  # noqa: E402
    get_extra,
    load_adjusted_close,
    load_market_extras,
)

PAPER_DIR = os.path.join(ROOT, "data", "paper")
RAW_DIR = os.path.join(ROOT, "data", "tdx_daily", "raw")

#: 是否有 baostock 附加数据可用（可成交性精确判定）
BAOSTOCK_READY = os.path.isdir(
    os.path.join(ROOT, "data", "parquet", "baostock_daily")
)

COMMISSION = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.0005
SLIPPAGE = 0.001
# 2026-10-27：主线配置（与回测引擎一致）
#   use_panel=False → 只用 25 个诚实公式因子（实测夏普 0.880 > 42 因子 0.799）
#   exposure_neutral=True → 暴露中性化（原生引擎分段验证：Δ夏普 +0.22~+0.31，
#     回撤每段改善 2~3pp）
USE_PANEL = False
# 暴露中性化：**不开**。
# 原生引擎分段验证（2026-10-27）：
#   seg1 Δ夏普 −0.140 / seg2 +0.200 / seg3 −0.269 → 分段不稳定 = 过拟合
# （子代理的实验室模拟器报 3 段全正 +0.22~+0.31，但模拟器 seg1 基线
#   0.243 vs 原生引擎 0.712 —— 模拟器在该段有重大偏差，不能用于最终判断）
EXPOSURE_NEUTRAL = False
CASH_RESERVE = 0.05
BOARD_LOT = 100
# 涨停阈值：2026-10-26 起按板块动态（见 execution_rules.limit_pct_array），
# 主板 10% / 创业板科创板 20% / 北交所 30% / ST 5%。原来硬编码 0.095。


# ── 打分函数：每条线一个复合分（越大越该买） ──────────────────────────
def score_22f(frames, master, panel=None) -> pd.DataFrame:
    """22 因子等权（与 rolling walk-forward 选出的符号一致）。"""
    return ws.compute_composite_22f(frames, master)


def score_42f(frames, master, panel=None, explain_date=None) -> pd.DataFrame:
    """42 因子等权复合（注册策略 honest_composite 同源）。"""
    if panel is None:
        panel = {}
    return ds.compute_composite(frames, panel, master, explain_date=explain_date,
                                use_panel=USE_PANEL, exposure_neutral=EXPOSURE_NEUTRAL)


def score_42f_neutral(frames, master, panel=None, explain_date=None) -> pd.DataFrame:
    """42 因子 + 行业中性化（新增策略线，与 honest_42f 并行赛马）。

    做法：因子 z-score 先按申万一级行业去均值（消除系统性行业暴露），
    再在行业内重新标准化（保证行业内可比），最后等权合成。
    全部操作**逐日横截面**，无未来函数。

    依据（3 年全市场周频回测，复刻执行器口径）：
      基线 42f        夏普 0.965  年化 +13.25%  回撤 -17.63%
      行业中性 + 重归一 夏普 1.441  年化 +25.32%  回撤 -17.86%
    分年度 4 年全部提升。作为**新线**落地，与原线并行跑纸面盘对比，
    不直接替换主线。
    """
    if panel is None:
        panel = {}
    return ds.compute_composite(
        frames, panel, master,
        explain_date=explain_date,
        industry_neutral=True,
    )


def score_rsi_71421(frames, master, panel=None) -> pd.DataFrame:
    """RSI 7/14/21 + 低波动 + 量价等权复合（与注册策略 rsi_71421 同源）。"""
    close = ds.to_mat(frames, master, "close")
    vol = ds.to_mat(frames, master, "volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close.pct_change(fill_method=None)
        vchg = vol.pct_change(fill_method=None)
        cands: Dict[str, pd.DataFrame] = {}
        for n in (7, 14, 21):
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(n).mean()
            loss = (-delta.clip(upper=0)).rolling(n).mean()
            rsi = 100 - 100 / (1 + gain / loss)
            cands[f"RSI{n}"] = -rsi
            cands[f"STD{n}"] = -(close.rolling(n).std() / close)
            cands[f"CORR{n}"] = -close.rolling(n).corr(vol)
            cands[f"CORD{n}"] = -ret.rolling(n).corr(vchg)
    acc = np.zeros((len(master), len(frames)))
    wsum = np.zeros_like(acc)
    for fname, mat in cands.items():
        ds.zscore_add(acc, wsum, mat.to_numpy(dtype=np.float64), 1.0, ds.MIN_CROSS)
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    # 涨停掩码：按板块动态阈值 + ST 5%（2026-10-26 P0-2）
    # 原来硬编码 0.095：20% 板涨 9.5%~20% 被误剔（砍 alpha），
    # ST 股 5% 涨停拦不住（买入即亏）。
    _codes = list(frames)
    _lu = limit_up_mask_matrix(
        close.pct_change(fill_method=None).to_numpy(), _codes,
        is_st=load_st_flags(_codes),
    )
    comp[_lu] = np.nan
    return pd.DataFrame(comp, index=master, columns=list(frames))


def score_gbm_voltgt(frames, master, panel=None, rebalance_day=None) -> pd.DataFrame:
    """LightGBM 元模型 + 波动率目标（复用注册策略 GbmVoltgtStrategy）。

    2026-10-25 新增：补齐策略赛马第 4 条线 —— 审计报告
    (docs/backtest/FULL_STRATEGY_AUDIT_2026-10-25.md) 入选的 4 条是
    honest_42f / gbm_voltgt / rsi_71421 / factor_ranking，但纸面盘里
    gbm_voltgt 被 22f 顶替了 —— 跑了两个月的纸面账本少了一条进攻线。

    这里**不重写**策略逻辑，直接调注册策略的 prepare_backtest_data，
    分数从 stock_data[code].attrs['_gbm_voltgt_scores'] 取回，
    保证纸面盘与回测/线上永远是同一份代码。
    """
    import asyncio

    from app.services.backtest.strategies.gbm_voltgt_strategy import (
        GbmVoltgtStrategy,
    )

    strategy = GbmVoltgtStrategy({
        "topk": 60,
        "n_drop": 5,
        "rebalance_days": "weekly",
        "rebalance_weekday": 4,   # 周五
        "horizon": 1,
        "limit_up": LIMIT_UP,
        "panel_dir": os.path.join(ROOT, "data", "qlib_data", "features", "day"),
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "warmup_months": 2,
        "target_vol": 0.20,
        "vt_window": 20,
        "cash_reserve_ratio": CASH_RESERVE,
    })

    start = master[0].to_pydatetime()
    end = master[-1].to_pydatetime()
    # 单模型模式：只用调仓日之前的数据训一个模型，预测调仓日当天。
    # 口径与月度滚动一致（该月模型就是月初训的、整月复用），但把 36 次
    # 重训压成 1 次 —— 否则单次纸面盘运行要 20+ 分钟，进不了 cron。
    single_from = None
    if rebalance_day is not None:
        single_from = pd.Timestamp(rebalance_day).to_pydatetime()
    print(
        f"  [gbm_voltgt] 训练元模型（42 因子，单模型模式，"
        f"切点 {single_from}）…",
        flush=True,
    )
    asyncio.run(
        strategy.prepare_backtest_data(
            frames, start, end, single_model_from=single_from
        )
    )

    codes = list(frames)
    out = pd.DataFrame(
        np.nan, index=pd.DatetimeIndex(master), columns=codes, dtype="float64"
    )
    for code in codes:
        score_map = (frames[code].attrs or {}).get("_gbm_voltgt_scores") or {}
        if not score_map:
            continue
        ser = pd.Series(score_map, dtype="float64")
        ser.index = pd.DatetimeIndex(ser.index)
        out[code] = ser.reindex(out.index)
    print(
        f"  [gbm_voltgt] 打分完成，有效分数 {int(np.isfinite(out.to_numpy()).sum())}",
        flush=True,
    )
    return out


def score_factor_ranking(frames, master, panel=None) -> pd.DataFrame:
    """动量(126d 跳 20d) / 低波动(60d) / 低振幅(10d) 等权 z（注册策略同源）。"""
    close = ds.to_mat(frames, master, "close")
    high = ds.to_mat(frames, master, "high")
    low = ds.to_mat(frames, master, "low")
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = close.shift(20) / close.shift(126) - 1.0
        vol = (close.pct_change(fill_method=None).rolling(60).std()
               * np.sqrt(252))
        amp = (high - low) / close
        amp = amp.rolling(10).mean()
    acc = np.zeros((len(master), len(frames)))
    wsum = np.zeros_like(acc)
    for mat in (mom, vol, amp):
        ds.zscore_add(acc, wsum, mat.to_numpy(dtype=np.float64), -1.0, ds.MIN_CROSS)
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    # 涨停掩码：按板块动态阈值 + ST 5%（2026-10-26 P0-2）
    # 原来硬编码 0.095：20% 板涨 9.5%~20% 被误剔（砍 alpha），
    # ST 股 5% 涨停拦不住（买入即亏）。
    _codes = list(frames)
    _lu = limit_up_mask_matrix(
        close.pct_change(fill_method=None).to_numpy(), _codes,
        is_st=load_st_flags(_codes),
    )
    comp[_lu] = np.nan
    return pd.DataFrame(comp, index=master, columns=list(frames))


# ── 策略注册表 ────────────────────────────────────────────────────────
STRATEGIES: Dict[str, Dict[str, Any]] = {
    "22f": {
        "score": score_22f, "topk": 30, "ndrop": 3,
        "desc": "22 因子等权（MIN +1 / STD,CORR,CORD -1 / MAX30,60 +1）",
    },
    "honest_42f": {
        "score": score_42f, "topk": 60, "ndrop": 5,
        "desc": "42 因子等权复合（23 基础 + 17 Alpha158 面板），注册策略同源",
    },
    "rsi_71421": {
        "score": score_rsi_71421, "topk": 30, "ndrop": 3,
        "desc": "RSI 7/14/21 均值回归 + 低波动 + 量价",
    },
    "factor_ranking": {
        "score": score_factor_ranking, "topk": 50, "ndrop": 5,
        "desc": "动量(126d 跳 20d)/低波动(60d)/低振幅(10d) 等权 z",
    },
    "gbm_voltgt": {
        "score": score_gbm_voltgt, "topk": 60, "ndrop": 5,
        "desc": "LightGBM 元模型 + 波动率目标（42 因子，月度滚动重训）",
    },
    "honest_42f_neu": {
        "score": score_42f_neutral, "topk": 60, "ndrop": 5,
        "desc": "42 因子 + 行业中性（申万一级行业内选相对最强的）",
    },
}


def load_name_map() -> Dict[str, str]:
    if not os.path.isdir(RAW_DIR):
        return {}
    dates = sorted(d for d in os.listdir(RAW_DIR) if d.startswith("date="))
    if not dates:
        return {}
    try:
        df = pd.read_parquet(os.path.join(RAW_DIR, dates[-1], "part.parquet"),
                             columns=["ts_code", "name"])
        return dict(zip(df["ts_code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def holdings_path(strategy: str) -> str:
    return os.path.join(PAPER_DIR, f"{strategy}_holdings.json")


def ledger_path(strategy: str) -> str:
    return os.path.join(PAPER_DIR, f"{strategy}_ledger.csv")


def load_holdings(strategy: str, capital: float) -> Dict[str, Any]:
    p = holdings_path(strategy)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {"cash": float(capital), "positions": {},
            "initial_capital": float(capital)}


def save_holdings(strategy: str, data: Dict[str, Any]) -> None:
    os.makedirs(PAPER_DIR, exist_ok=True)
    with open(holdings_path(strategy), "w") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


def append_ledger(strategy: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(PAPER_DIR, exist_ok=True)
    p = ledger_path(strategy)
    exists = os.path.exists(p)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def _num(series, code):
    """从 Series 里取数值，取不到或非有限返回 None。"""
    try:
        v = series.get(code)
    except AttributeError:
        return None
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _tradability(code, date, names, extras, *, close, open_price, high, low,
                  volume, close_prev):
    """调用统一的可成交性规则引擎，组装所需字段。

    优先用 baostock 的交易所口径字段（pct_chg / trade_status / is_st / amount），
    没有时退回本地 close/prev_close 推导（此时涨跌幅可能含除权跳空，
    规则引擎按保守处理，不会把除权误判成涨跌停）。
    """
    def _get(field):
        if not extras:
            return None
        try:
            return get_extra(extras, field, code, date)
        except Exception:
            return None

    pct = _get("pct_chg")
    # pct 来自交易所口径 = 可靠；由未复权 close/prev_close 推导 = 可能含除权跳空
    pct_reliable = pct is not None
    if pct is None:
        c = _num(close, code)
        prev = _num(close_prev, code)
        pct = (c / prev - 1.0) if (c and prev and prev > 0) else None

    ts = _get("trade_status")
    st = _get("is_st")
    amt = _get("amount")

    return check_tradability(
        code,
        name=names.get(code, ""),
        close=_num(close, code),
        open_price=_num(open_price, code),
        high=_num(high, code),
        low=_num(low, code),
        pct_chg=pct,
        volume=_num(volume, code),
        amount=amt,
        trade_status=int(ts) if ts is not None else None,
        is_st=int(st) if st is not None else None,
        pct_reliable=pct_reliable,
    )



def _explain_factors(code: str, columns, n_top: int = 4) -> list:
    """取该股票在调仓日的因子贡献分解（42f 专用）。

    返回 [{factor, contribution, direction}]，按贡献绝对值排序。
    contribution = 该因子的 sign*z 值（已含方向修正：正值=该因子看多）。

    columns 必须是 comp.columns —— z 值矩阵的列序与之一致。
    """
    zmap = getattr(ds, "_LAST_FACTOR_Z", None) or {}
    if not zmap or columns is None:
        return []
    try:
        col = list(columns).index(code)
    except ValueError:
        return []
    items = []
    for fname, row in zmap.items():
        if col >= len(row):
            continue
        v = float(row[col])
        if v != v:   # NaN
            continue
        items.append({
            "factor": fname,
            "contribution": round(v, 4),
            "direction": "看多" if v > 0 else "看空",
        })
    items.sort(key=lambda x: -abs(x["contribution"]))
    return items[:n_top]


def run_line(
    strategy: str,
    spec: Dict[str, Any],
    frames: Dict[str, Any],
    panel: Dict[str, Any],
    master: pd.DatetimeIndex,
    names: Dict[str, str],
    capital: float,
    rebalance_day: pd.Timestamp,
    extras: Optional[Dict[str, Any]] = None,
    adj_close: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """跑一条线的调仓（周频：周五收盘打分 → **下一交易日**成交）。

    2026-10-26 起口径改为 **lag1**（对齐回测的真实盘口径）：
      - `signal_day`（周五收盘）出信号、选股
      - `trade_day`（下一交易日）成交 —— 价格、涨跌停判定都用成交日
    原因：回测已改成 `lag=1`（用当日收盘价算信号又用同一收盘价成交是
    实盘做不到的），纸面盘必须同口径，否则「纸面业绩 vs 回测预期」不可比。
    """
    # 所有 score 函数统一签名 (frames, master, panel=None)
    # gbm_voltgt 额外需要 rebalance_day（单模型模式的切点）
    if spec["score"] is score_gbm_voltgt:
        comp = spec["score"](frames, master, panel, rebalance_day=rebalance_day)
    elif spec["score"] in (score_42f, score_42f_neutral):
        # 42f 顺便收集当日因子 z 值 → 用于「为什么买它」归因
        comp = spec["score"](frames, master, panel, explain_date=rebalance_day)
    else:
        comp = spec["score"](frames, master, panel)

    # ── 信号日 / 成交日 ───────────────────────────────────────────
    signal_day = pd.Timestamp(rebalance_day)
    # 打分矩阵里取**不晚于**信号日的最后一天（不能往后取 → 那是未来数据）
    usable = comp.index[comp.index <= signal_day]
    if len(usable) == 0:
        raise SystemExit(
            f"❌ {strategy}：打分矩阵里没有不晚于 {signal_day.date()} 的数据，"
            f"无法生成信号（矩阵起点 {comp.index[0].date()}）"
        )
    latest = usable[-1]

    # 成交日 = 信号日之后的第一个交易日（数据里必须存在）
    _after = master[master > signal_day]
    if len(_after) == 0:
        raise SystemExit(
            f"❌ {strategy}：信号日 {signal_day.date()} 之后还没有交易日的行情数据。\n"
            f"   lag1 口径需要**下一交易日**才能成交（价格 + 涨跌停判定都在成交日）。\n"
            f"   数据最新只到 {master[-1].date()}，请等下一交易日的日线包发布后重跑。",
        )
    trade_day = _after[0]

    close_mat = ds.to_mat(frames, master, "close")
    # 成交价与涨跌停判定一律用**成交日**
    close = close_mat.loc[trade_day]
    vol = ds.to_mat(frames, master, "volume").loc[trade_day]
    open_price = ds.to_mat(frames, master, "open").loc[trade_day]
    high = ds.to_mat(frames, master, "high").loc[trade_day]
    low = ds.to_mat(frames, master, "low").loc[trade_day]
    close_prev = close_mat.shift(1).loc[trade_day]
    day_ret = close / close_prev - 1.0
    # 本次调仓因可成交性被拦下的标的（写进结果，让用户知道「为什么没买」）
    blocked_buys: List[Dict[str, Any]] = []
    blocked_sells: List[Dict[str, Any]] = []

    paper = load_holdings(strategy, capital)
    positions = paper.get("positions", {})
    cash = float(paper.get("cash", capital))

    # ── 回填历史持仓的 adj_cost（2026-10-25）────────────────────────
    # 复权口径估值需要每只持仓的「买入日后复权价」。老持仓是在加这个
    # 字段之前建的，从 ledger.csv 的买入记录回填。
    if adj_close is not None:
        _filled = 0
        for _code, _pos in positions.items():
            if _pos.get("adj_cost"):
                continue
            _bd = _pos.get("buy_date")
            if not _bd:
                continue
            try:
                _ts = pd.Timestamp(_bd)
                if _ts in adj_close.index and _code in adj_close.columns:
                    _v = float(adj_close.at[_ts, _code])
                    if np.isfinite(_v) and _v > 0:
                        _pos["adj_cost"] = round(_v, 6)
                        _filled += 1
            except (KeyError, TypeError, ValueError):
                continue
        if _filled:
            print(f"  [{strategy}] 回填 adj_cost: {_filled} 只持仓", flush=True)

    rank = comp.loc[latest].dropna().sort_values(ascending=False)
    # 实盘安全过滤（2026-10-25 补退市股）：
    #   ST  —— 5% 涨跌停 + 退市风险，9.5% 涨停掩码对它们失效
    #   退  —— 名称含「退」= 已进入退市整理期 / 已退市，根本买不进
    #   （实测踩到过：002731.SZ 萃华退 被选进买入清单）
    st_codes = {c for c in rank.index if "ST" in names.get(c, "").upper()}
    delist_codes = {
        c for c in rank.index
        if "退" in names.get(c, "") and "退市" not in names.get(c, "")
    }
    # 名称里「退市」是说明性后缀（如「退市整理」），只有裸「退」字结尾才算退市整理期
    delist_codes = {
        c for c in rank.index
        if names.get(c, "").endswith("退")
    }
    unsafe = (st_codes | delist_codes)
    if unsafe:
        rank = rank.drop(index=[c for c in unsafe if c in rank.index])
    rank_index = {c: i for i, c in enumerate(rank.index)}
    topk = int(spec["topk"])
    ndrop = int(spec["ndrop"])
    top = list(rank.index[:topk])

    held = list(positions.keys())
    total_value = cash + sum(
        float(positions[c].get("shares", 0)) * float(close.get(c, np.nan))
        for c in held
    )
    target_weight = (1.0 - CASH_RESERVE) / topk

    # ── 持仓估值（总回报口径）───────────────────────────────────────
    # 关键：未复权价在除权日跳空（实测 -50%），用它算持仓市值会
    # **系统性低估收益**（实测 1193 只等权组合年化差 5.45 个百分点）。
    #
    # 正确公式（标准总回报 / total-return 口径）：
    #   持仓市值 = **买入金额** × (当前后复权价 / 买入日后复权价)
    #
    # 为什么不是「股数 × 当前未复权价 × 后复权比值」：
    #   后复权比值里**已经包含了普通跌幅**（实测 adj_ratio 与真实未复权
    #   价比几乎完全相等）。再乘一次当前未复权价 = 跌幅被算两次
    #   （实测会让所有线的净值凭空少 1%）。
    #   写成「买入金额 × 后复权比值」则：
    #     - 无除权 → 后复权比值 = 真实跌幅 → 正确
    #     - 有送转（10 送 10，价腰斩）→ 后复权比值 = 1.0（pct_chg 已调整）
    #       → 市值不变 → 正确（送转后股数翻倍，总市值不变）
    def _adj_ratio(code: str) -> float:
        """当前后复权价 / 买入日后复权价。"""
        if adj_close is None or code not in adj_close.columns:
            return 1.0
        try:
            a_now = float(adj_close.at[latest, code])
            a_cost = float(positions.get(code, {}).get("adj_cost", 0) or 0)
        except (KeyError, TypeError, ValueError):
            return 1.0
        if not np.isfinite(a_now) or not np.isfinite(a_cost) or a_cost <= 0:
            return 1.0
        return a_now / a_cost

    def _pos_value(code: str) -> float:
        """总回报口径的持仓市值 = 买入金额 × 后复权比值。"""
        p = positions.get(code) or {}
        sh = float(p.get("shares", 0) or 0)
        cost = float(p.get("cost", 0) or 0)
        if sh <= 0 or cost <= 0:
            return 0.0
        return sh * cost * _adj_ratio(code)

    def px(c: str) -> float:
        v = close.get(c, np.nan)
        return float(v) if np.isfinite(v) else float("nan")

    # ── 卖出: 持仓中掉出 topk 的, 最差优先, 最多 ndrop ──
    fell_out = [c for c in held if c not in set(top)]
    fell_out.sort(key=lambda c: rank_index.get(c, len(rank) + 1), reverse=True)
    sells = []
    for c in fell_out[:ndrop]:
        p = px(c)
        if not np.isfinite(p) or p <= 0:
            continue
        # ── 可成交性判定（统一规则引擎）──
        tr = _tradability(c, latest, names, extras, close=close,
                          open_price=open_price, high=high, low=low,
                          volume=vol, close_prev=close_prev)
        if not tr.can_sell:
            blocked_sells.append({"code": c, "name": names.get(c, ""),
                                  "reason": tr.blocked_reason or tr.reason})
            continue
        shares = int(positions[c].get("shares", 0))
        if shares <= 0:
            continue
        # 卖出金额用总回报口径（与持仓估值同源，保证账本自洽）；
        # real_gross 记录「按当前未复权价实际能拿到的钱」供用户参考。
        real_gross = shares * p
        gross = float(positions[c].get("cost", 0) or 0) * shares * _adj_ratio(c)
        commission = max(MIN_COMMISSION, gross * COMMISSION)
        stamp = gross * STAMP_TAX
        slip = gross * SLIPPAGE
        cash += gross - commission - stamp - slip
        sells.append({"code": c, "name": names.get(c, ""), "price": p,
                      "shares": shares, "amount": round(gross, 2),
                      "real_amount": round(real_gross, 2),
                      "commission": round(commission, 2),
                      "stamp_tax": round(stamp, 2), "slippage": round(slip, 2),
                      "net": round(gross - commission - stamp - slip, 2),
                      "score": round(float(rank.get(c, np.nan)), 4)
                      if c in rank else "",
                      "factors": _explain_factors(c, comp.columns)})
        del positions[c]

    # ── 买入: 首次建仓买满 topk；之后每次只买 ndrop 只 ──
    held_set = set(positions.keys())
    new_in = [c for c in top if c not in held_set]
    # 首次建仓（或组合被清空）→ 一次买满 topk；日常调仓 → 只换 ndrop 只
    max_buys = max(0, topk - len(positions)) if len(positions) < topk else ndrop
    buys = []
    for c in new_in[:max_buys]:
        p = px(c)
        if not np.isfinite(p) or p <= 0:
            continue
        # ── 可成交性判定（统一规则引擎）──
        tr = _tradability(c, latest, names, extras, close=close,
                          open_price=open_price, high=high, low=low,
                          volume=vol, close_prev=close_prev)
        if not tr.can_buy:
            blocked_buys.append({"code": c, "name": names.get(c, ""),
                                 "reason": tr.blocked_reason or tr.reason})
            continue
        budget = min(
            total_value * target_weight,
            max(0.0, cash - total_value * CASH_RESERVE),
        )
        # 流动性上限：买入金额不能超过该股当日成交额的参与率
        if tr.max_buy_amount is not None:
            budget = min(budget, tr.max_buy_amount)
        shares = int(budget / p / BOARD_LOT) * BOARD_LOT
        if shares <= 0:
            continue
        gross = shares * p
        commission = max(MIN_COMMISSION, gross * COMMISSION)
        slip = gross * SLIPPAGE
        cost = gross + commission + slip
        if cost > cash:
            continue
        cash -= cost
        a_cost = None
        if adj_close is not None and c in adj_close.columns:
            try:
                a_cost = float(adj_close.at[latest, c])
            except (KeyError, TypeError, ValueError):
                a_cost = None
        positions[c] = {"shares": shares, "cost": round(p, 3),
                        "buy_date": str(latest.date()),
                        # 买入日的后复权价 —— 用于计算持有期的真实收益（无除权跳空）
                        "adj_cost": round(a_cost, 6) if a_cost else None}
        buys.append({"code": c, "name": names.get(c, ""), "price": p,
                     "shares": shares, "amount": round(gross, 2),
                     "commission": round(commission, 2), "stamp_tax": 0.0,
                     "slippage": round(slip, 2), "total_cost": round(cost, 2),
                     "score": round(float(rank.get(c, np.nan)), 4),
                     "factors": _explain_factors(c, comp.columns)})

    end_value = cash + sum(_pos_value(c) for c in positions)
    paper.update({"cash": round(cash, 2), "positions": positions,
                  # last_signal = **信号日**（周五出信号）
                  # last_trade  = **成交日**（下一交易日成交，lag1 口径）
                  "last_signal": str(signal_day.date()),
                  "last_trade": str(trade_day.date()),
                  "last_value": round(end_value, 2),
                  "fill_assumption": "Friday close"})
    save_holdings(strategy, paper)

    rows = []
    for s in sells:
        rows.append({"date": str(trade_day.date()), "action": "SELL",
                     "code": s["code"], "name": s["name"], "price": s["price"],
                     "shares": s["shares"], "amount": s["amount"],
                     "commission": s["commission"], "stamp_tax": s["stamp_tax"],
                     "slippage": s["slippage"], "score": s["score"],
                     "cash_after": round(cash, 2),
                     "portfolio_value": round(end_value, 2)})
    for b in buys:
        rows.append({"date": str(trade_day.date()), "action": "BUY",
                     "code": b["code"], "name": b["name"], "price": b["price"],
                     "shares": b["shares"], "amount": b["amount"],
                     "commission": b["commission"], "stamp_tax": 0.0,
                     "slippage": b["slippage"], "score": b["score"],
                     "cash_after": round(cash, 2),
                     "portfolio_value": round(end_value, 2)})
    append_ledger(strategy, rows)

    return {
        "strategy": strategy,
        "signal_date": str(signal_day.date()),
        "trade_date": str(trade_day.date()),
        "desc": spec["desc"],
        "date": str(latest.date()),
        "topk": topk,
        "ndrop": ndrop,
        "cash": round(cash, 2),
        "value": round(end_value, 2),
        "n_positions": len(positions),
        "target_weight": round(target_weight, 6),
        "sells": sells,
        "buys": buys,
        # 因可成交性被拦下的标的（涨停买不进 / 跌停卖不出 / 停牌 / 流动性不足）
        # —— 写进结果是为了让用户知道「为什么信号里有它但没成交」
        "blocked_buys": blocked_buys,
        "blocked_sells": blocked_sells,
        "holds": [c for c in top if c in positions
                  and c not in {s["code"] for s in sells}],
    }



def rebuild_line_from_files(strategy: str, capital: float) -> Dict[str, Any]:
    """从持仓文件 + ledger 重建一条线的展示数据（不重新调仓）。

    为什么需要：portfolio.json 是给前端看的汇总，而它原来只写「本次运行」的
    结果 → 一旦某条线因为「重复调仓」被跳过、或某条线中途报错导致整个
    进程退出，portfolio.json 就会停留在旧数据 —— 而持仓文件已经是新的，
    **前端展示的买卖价与实际成交对不上**（实测 2026-10-26 就是这个情况）。

    这里从 ledger 的最近一个成交日取买卖明细，从持仓文件取现金/市值。
    """
    spec = STRATEGIES.get(strategy) or {}
    h = load_holdings(strategy, capital)
    positions = h.get("positions") or {}
    cash = float(h.get("cash", capital))
    # 市值：优先用持仓文件里的现值字段
    value = float(h.get("last_value") or h.get("value") or 0)
    if not value:
        value = cash + sum(
            float((pos or {}).get("market_value") or
                  float((pos or {}).get("quantity") or 0) *
                  float((pos or {}).get("last_price") or
                        (pos or {}).get("avg_cost") or 0))
            for pos in positions.values()
        )
    buys: List[Dict[str, Any]] = []
    sells: List[Dict[str, Any]] = []
    trade_date = h.get("last_trade") or h.get("last_signal")
    _lp = ledger_path(strategy)
    if trade_date and os.path.exists(_lp):
        try:
            import csv as _csv
            with open(_lp, newline="") as fh:
                rows = list(_csv.DictReader(fh))
            for r in rows:
                if str(r.get("date")) != str(trade_date):
                    continue
                item = {
                    "code": r.get("code", ""),
                    "name": r.get("name", ""),
                    "price": _tofloat(r.get("price")),
                    "quantity": int(_tofloat(r.get("quantity")) or 0),
                    "amount": _tofloat(r.get("amount")),
                    "fee": _tofloat(r.get("fee")) or 0.0,
                }
                if str(r.get("action", "")).upper() == "BUY":
                    buys.append(item)
                else:
                    sells.append(item)
        except Exception:
            pass
    return {
        "strategy": strategy,
        "desc": spec.get("desc", strategy),
        "signal_date": h.get("last_signal"),
        "trade_date": trade_date,
        "date": trade_date,
        "cash": round(cash, 2),
        "value": round(value, 2),
        "n_positions": len(positions),
        "sells": sells,
        "buys": buys,
        "blocked_buys": [],
        "blocked_sells": [],
        "holds": list(positions.keys()),
    }


def _tofloat(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0

def main() -> None:
    ap = argparse.ArgumentParser(description="多策略纸面盘（每策略一条独立线）")
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--force", action="store_true", help="非周五也跑")
    args = ap.parse_args()

    codes = ds.list_universe()
    frames, panel, master = ds.load_data(codes)
    names = load_name_map()

    # 2026-10-25：加载 baostock 附加字段（交易所口径涨跌幅 / 成交额 / 停牌 / ST）
    # 用于可成交性判定 —— 避免除权跳空被误判成涨跌停
    extras: Dict[str, Any] = {}
    adj_close: Optional[pd.DataFrame] = None
    if BAOSTOCK_READY:
        extras = load_market_extras(codes, master)
        # 后复权价格 —— 消除除权跳空对组合净值的系统性低估
        # （实测 1193 只等权组合 2021-2026：未复权年化 +6.90%，
        #   交易所口径年化 +12.35%，差 5.45 个百分点）
        adj_close = load_adjusted_close(codes, master)

    # ── 选调仓日 ─────────────────────────────────────────────────────
    # 2026-10-25 修正（关键 bug）：
    #   原逻辑只看 master（数据）最后一天是不是周五，从不核对**真实日期**。
    #   叠加 TDX 当日日线包 17:00 后才发布，16:05 跑时数据最新日永远是周四
    #   → 每周五都判成「不是周五」→ 周频调仓整整两周没跑过（信号停在 10-16）。
    #   现在：真实日期决定「该不该调仓」，数据决定「用哪天调仓」，
    #   两者必须对齐 —— 数据没到本周五就明确报错，绝不静默用旧周五。
    #
    # 规则（按优先级）：
    #   1. master 最后一天恰好是周五 → 用它（正常周五收盘后的情况）
    #   2. 否则用 master 里最后一个周五（数据还没更新到本周五）
    #   3. --force → 用 master 最后一天（人工指定，但只能向前）
    # 关键约束：**调仓日不得早于组合已记录的 last_signal**。
    #   否则用 --force 建仓（用了 master[-1]）后再自动跑（用更早的周五）
    #   会拿过去的分数调仓 = 时间倒流。
    latest_all = master[-1]
    today = pd.Timestamp(date.today())
    _skipped = False

    # 真实今天（或之前）最近的周五
    days_since_friday = (today.dayofweek - 4) % 7
    expected_friday = today - pd.Timedelta(days=days_since_friday)

    # 2026-10-26（lag1 口径）：信号日必须是**最后一个周五**，
    # 成交日 = 它的下一个交易日。数据到周一/周二时，信号日仍取上周五，
    # 成交日取周一 —— 与回测 `lag=1` 完全一致。
    fridays = [d for d in master if d.dayofweek == 4]
    if fridays:
        rebalance_day = fridays[-1]
    else:
        rebalance_day = latest_all
    trade_day = None
    _after = master[master > rebalance_day]
    if len(_after):
        trade_day = _after[0]

    # 数据新鲜度硬校验：真实日期已过本周五，但数据还没到 → 报错退出，不静默用旧周五
    if today.dayofweek == 4 and not args.force and latest_all < expected_friday:
        print(
            f"❌ 数据陈旧：真实今天是 {today.date()}（周五），"
            f"但数据最新只到 {latest_all.date()}。\n"
            f"   TDX 当日日线包通常 17:00 后发布，请确认拉取是否成功"
            f"（scripts/daily_tdx_update.sh）。\n"
            f"   不会用旧的 {rebalance_day.date()} 冒充本周信号。",
            flush=True,
        )
        sys.exit(2)

    if args.force and latest_all > rebalance_day:
        # --force：仍保持「信号日 = 周五」的口径，只把信号日推到最近的周五
        _f = [d for d in master if d.dayofweek == 4 and d <= latest_all]
        rebalance_day = _f[-1] if _f else latest_all
        _after = master[master > rebalance_day]
        trade_day = _after[0] if len(_after) else None

    # 已有持仓的线：**成交日**不得早于上次成交日
    #
    # 2026-10-26 修正：原来比较的是「信号日」，在 lag1 口径下会误伤 ——
    # 信号日恒为上周五（10-23），而成交日每周都在变（10-26、11-02…）。
    # 用信号日比较 → 第二周开始就会被判成「重复调仓」而永远跳过。
    # 正确判据是**成交日**（last_trade）。
    # 老数据没有 last_trade 时回退到 last_signal。
    existing_signals = []
    for s in args.strategies:
        h = load_holdings(s, args.capital)
        _lt = h.get("last_trade") or h.get("last_signal")
        if _lt:
            existing_signals.append(pd.Timestamp(_lt))
    if existing_signals:
        last_used = max(existing_signals)
        # 用 `<=` 而不是 `<`：同一天重复跑（手动重跑 / cron 补跑）时
        # 信号日会相等 —— 原来只挡「更早」，会**重复调仓**（同一周买两次）。
        _trade_for_check = trade_day if trade_day is not None else rebalance_day
        if _trade_for_check <= last_used and not args.force:
            print(
                f"跳过：成交日 {_trade_for_check.date()} 不晚于已成交的 "
                f"{last_used.date()}（已调仓过，重复执行会重复下单）。"
                f"如需重跑请加 --force。",
                flush=True,
            )
            # 跳过也要更新 meta —— 否则推送脚本读不到 signal_date/trade_date，
            # 标题会退回旧的「每日信号 X」格式，看不出该哪天操作。
            _ppath = os.path.join(PAPER_DIR, "portfolio.json")
            if os.path.exists(_ppath):
                try:
                    with open(_ppath) as _fh:
                        _pd = json.load(_fh)
                    _md = _pd.get("meta") or {}
                    _md["signal_date"] = str(rebalance_day.date())
                    _md["trade_date"] = (
                        str(trade_day.date()) if trade_day is not None else None
                    )
                    _md["data_latest"] = str(latest_all.date())
                    _md["last_run_at"] = datetime.now().isoformat()
                    _pd["meta"] = _md
                    with open(_ppath, "w") as _fh:
                        json.dump(_pd, _fh, ensure_ascii=False, indent=1)
                    print("  （已更新 portfolio.json 的 meta）", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  meta 更新失败（不影响）: {e}", flush=True)
            # 不 return —— 继续往下走，让「从持仓文件重建所有线」的逻辑执行，
            # 否则 portfolio.json 会停留在旧数据（前端展示的买卖价与实际成交对不上）
            _skipped = True
    if not args.force:
        print(
            f"信号日（周五）= {rebalance_day.date()}"
            + (f"  成交日（下一交易日）= {trade_day.date()}" if trade_day is not None
               else "  ⚠️ 还没有下一交易日的行情数据 → 本次不成交"),
            flush=True,
        )

    results = []
    if locals().get("_skipped"):
        print("（本次未调仓，仅重建 portfolio.json）", flush=True)
    for s in (args.strategies if not locals().get("_skipped") else []):
        spec = STRATEGIES[s]
        print(f"\n{'='*60}\n[{s}] {spec['desc']}", flush=True)
        r = run_line(s, spec, frames, panel, master, names, args.capital,
                     rebalance_day, extras=extras, adj_close=adj_close)
        results.append(r)
        print(f"  组合总值 ¥{r['value']:,.0f} | 现金 ¥{r['cash']:,.0f} | "
              f"{r['n_positions']} 只 | 买 {len(r['buys'])} 卖 {len(r['sells'])}",
              flush=True)

    os.makedirs(PAPER_DIR, exist_ok=True)
    portfolio_path = os.path.join(PAPER_DIR, "portfolio.json")

    # 2026-10-25 修正：原来直接覆盖整个 portfolio.json，
    # 用 --strategies 跑单条线时会把其他线的信号全清掉
    # （实测跑 gbm_voltgt 后 4 条线消失）。改成**合并写入**：
    # 本次跑到的线覆盖，其余线保留上一次的结果。
    existing = {}
    if os.path.exists(portfolio_path):
        try:
            with open(portfolio_path) as fh:
                existing = json.load(fh) or {}
        except Exception:
            existing = {}

    prev_meta = existing.get("meta") or {}
    # 2026-10-27：**总是从持仓文件 + ledger 重建所有线**。
    # 原来只写「本次运行」的结果 → 某条线被跳过（重复调仓）或某条线报错
    # 导致进程退出时，portfolio.json 会停留在旧数据，而持仓文件已是新的
    # → **前端展示的买卖价与实际成交对不上**（2026-10-26 实测到这个）。
    merged_lines = dict(existing.get("lines") or {})
    for _name in (list(STRATEGIES.keys()) + [r["strategy"] for r in results]):
        try:
            merged_lines[_name] = rebuild_line_from_files(_name, args.capital)
        except Exception as e:  # noqa: BLE001
            print(f"  重建 {_name} 失败（保留旧值）: {e}", flush=True)
    # 本次实际成交的明细（blocked_buys 等只有本次运行才有）覆盖回去
    for r in results:
        _prev = merged_lines.get(r["strategy"]) or {}
        _prev.update({
            "buys": r.get("buys", []),
            "sells": r.get("sells", []),
            "blocked_buys": r.get("blocked_buys", []),
            "blocked_sells": r.get("blocked_sells", []),
            "signal_date": r.get("signal_date"),
            "trade_date": r.get("trade_date"),
            "desc": r.get("desc", _prev.get("desc")),
        })
        merged_lines[r["strategy"]] = _prev

    summary = {
        "meta": {
            # signal_date = 周五出信号；trade_date = 下一交易日成交（lag1 口径）
            "date": str(rebalance_day.date()),
            "signal_date": str(rebalance_day.date()),
            "trade_date": str(trade_day.date()) if trade_day is not None else None,
            "data_latest": str(latest_all.date()),
            "capital_per_line": args.capital,
            "costs": {"commission": COMMISSION, "stamp_tax": STAMP_TAX,
                      "slippage": SLIPPAGE, "cash_reserve": CASH_RESERVE},
            "fill_assumption": (
                "lag1：周五收盘出信号，下一交易日成交（价格与涨跌停判定均在成交日）"
            ),
            "generated_at": datetime.now().isoformat(),
            "lines_updated": [r["strategy"] for r in results],
        },
        "lines": merged_lines,
    }
    with open(portfolio_path, "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    frontend_public = os.path.join(ROOT, "frontend", "public")
    os.makedirs(frontend_public, exist_ok=True)
    with open(os.path.join(frontend_public, "paper-portfolio.json"), "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    print(f"\n{'='*60}", flush=True)
    print(f"{'strategy':<20}{'value':>14}{'cash':>12}{'pos':>6}{'buy':>5}{'sell':>6}")
    for r in results:
        print(f"{r['strategy']:<20}¥{r['value']:>13,.0f}¥{r['cash']:>11,.0f}"
              f"{r['n_positions']:>6}{len(r['buys']):>5}{len(r['sells']):>6}")
    print(f"\n[saved] {os.path.join(PAPER_DIR, 'portfolio.json')}", flush=True)
    print(f"[saved] {os.path.join(frontend_public, 'paper-portfolio.json')}",
          flush=True)


if __name__ == "__main__":
    main()
