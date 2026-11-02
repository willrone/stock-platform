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
  data/paper/portfolio.json             全部线的汇总（供后端 / MCP 读）

注意：脚本**不再**输出到 frontend/public/ —— 静态文件无鉴权可下载，
会泄露付费信号的持仓与逐笔明细。前端统一读受保护端点
GET /api/v1/trading/signal（需 Bearer token）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# 多资产组合线（2026-11-01）：A股腿 + ETF 袖套腿
# 袖套的核心逻辑放在 app/services/trading/ 里（可被 API / MCP / 前端复用），
# 这里只做「纸面盘接线」。
from app.services.trading.multi_asset_service import (  # noqa: E402
    SLEEVE_CLASSES,
    SLEEVE_HELD,
    SLEEVE_REB_EVERY,
    SleeveError,
    is_rebalance_day,
    load_etf_bars,
    missing_codes,
    rebalance_sleeve,
    sleeve_class_weights_now,
    sleeve_positions_value,
    sleeve_target_weights,
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
        # ⚠️ 2026-11-01 删掉 "limit_up": LIMIT_UP（**NameError: LIMIT_UP 未定义**）：
        #   commit 3d3e34e5 删了 `LIMIT_UP = 0.095` 硬编码（可成交性检查下沉引擎），
        #   却漏删这一处引用 → 纸面盘跑到 gbm_voltgt 这条线就崩（RC=1），
        #   **同一次运行里后面的 multi_asset_final 旗舰线根本轮不到执行**。
        #   删掉它与「最后一次能跑的状态」**逐位等价**：当时传的就是 0.095，
        #   而 GbmVoltgtStrategy 自己的默认值也是 0.095。
        #   注：gbm 侧现在走 limit_up_mask_matrix（逐票按板块判定），
        #   这个 config 只用于日志打印，不参与掩码计算。
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
    # 多资产组合线（2026-11-01）：A股腿 + ETF 袖套腿，静态 50/50 划分。
    # 这是研究侧 P3 定稿口径（组合夏普 1.4114 / OOS 1.0691）的产品化实现，
    # 也是唯一同时覆盖 A 股与黄金/债券/商品的差异化信号线。
    "multi_asset_final": {
        "score": None, "topk": 60, "ndrop": 5, "multi_asset": True,
        "desc": ("多资产组合：A股 42 因子 Top60（50%）+ "
                 "ETF 袖套 黄金/债券/商品 各 1/3（50%）"),
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


def last_valid_adj(adj_close: Optional[pd.DataFrame], code: str,
                   as_of: Any) -> Tuple[Optional[float], Optional[str]]:
    """取 **≤ as_of 的最后一个有效后复权价** → (值, 价格日期字符串)。

    为什么需要"最后有效"而不是 `adj_close.at[as_of, code]`（2026-11-02 实测）：
      后复权价由**交易所口径涨跌幅**合成（`load_adjusted_close`），
      而当日涨跌幅要等 **baostock（T+1）** 或 **TDX raw 包** 同步到位
      —— 实测 2026-11-02 当天 sync_pct_chg 报「没有可用的 TDX raw 分区」
      → adj_close 最后一行**整行 NaN**。
      若直接取 at[as_of] → NaN → 比值兜底成 1.0 → **该票市值被冻结在买入
      金额、永不反映涨跌**（实测 2026-11-02 首跑：multi_asset_final 69/69 只、
      其余各线 3~12 只中招）。
      → 统一改成「≤ as_of 的最后有效价」（停牌期价格本就不变，语义正确）。

    三处复用（口径单一来源）：买入成本、adj_cost 回填、组合估值。
    """
    if adj_close is None or code not in adj_close.columns:
        return None, None
    try:
        s_ = adj_close.loc[:as_of, code].dropna()
    except (KeyError, TypeError, ValueError):
        return None, None
    if not len(s_):
        return None, None
    try:
        v = float(s_.iloc[-1])
        d = str(pd.Timestamp(s_.index[-1]).date())
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(v) or v <= 0:
        return None, None
    return v, d


# ── 总回报口径估值（**周频与日频共用的唯一实现**，2026-11-02）──────────
# 为什么提取成模块级：日频盯市（paper_daily_nav.py）必须与周频调仓**逐位一致**
# —— 方案书 §3 门禁 a 把这条定为"不过就不许上线"的硬门禁。
# 若两边各写一份估值，口径迟早漂移（台账 P13：袖套指标 ≠ 组合指标；
# P12：w 曲线不可识别 —— 口径漂移是本项目最贵的错误类型）。
# → 口径**只实现一次**，周频与日频都调它。
#
# 口径本身（与原 run_line 内闭包 _pos_value 完全等价）：
#   持仓市值 = 买入金额(shares × cost) × 后复权比值(adj_close[as_of] / adj_cost)
#   组合市值 = 现金 + Σ 持仓市值
# 注意：**不是**「股数 × 当前未复权价 × 复权因子」—— 那会把跌幅算两次
# （原代码注释实测：会让所有线的净值凭空少 1%）。

def total_return_portfolio_value(
    positions: Dict[str, Any],
    cash: float,
    adj_close: Optional[pd.DataFrame],
    as_of: Any,
) -> Tuple[float, List[Dict[str, Any]]]:
    """总回报口径的组合市值 + 逐只明细（**只实现一次**，周频/日频共用）。

    参数：
      positions  {code: {shares, cost, adj_cost, ...}}
      cash       现金
      adj_close  后复权价矩阵（行=交易日）
      as_of      估值日（矩阵行索引）

    返回：(总市值, 明细列表)
      明细项：code / shares / cost / adj_ratio / market_value / stale_price
      `stale_price=True` 表示该票在 as_of 当日**没有后复权价**
      （停牌/数据缺失）→ 估值按**买入金额**计价（ratio=1.0），
      这与周频 `_adj_ratio` 的兜底**完全一致**（不能另立口径）。

    ⚠️ 浮点运算顺序与原闭包保持一致（sh * cost * ratio）→ 逐位相等。
    """
    total = float(cash)
    details: List[Dict[str, Any]] = []
    for code, p in (positions or {}).items():
        p = p or {}
        sh = float(p.get("shares", 0) or 0)
        cost = float(p.get("cost", 0) or 0)
        if sh <= 0 or cost <= 0:
            continue
        ratio = 1.0
        stale = False
        price_date = None
        if adj_close is not None and code in adj_close.columns:
            try:
                a_cost = float(p.get("adj_cost", 0) or 0)
            except (TypeError, ValueError):
                a_cost = 0.0
            # ⚠️ 2026-11-02 修正（第二个真 bug）：分子不能用 `adj_close.at[as_of, code]`
            #   —— 当日没有有效后复权价时（停牌；或**交易所口径涨跌幅还没同步**，
            #   实测 2026-11-02 当天 sync_pct_chg 报"没有可用的 TDX raw 分区"
            #   → adj_close 最后一行整行 NaN）→ 旧代码落到 ratio=1.0
            #   → **该票市值被冻结在买入金额，永不反映涨跌**。
            #   现在改成：取 **≤ as_of 的最后一个有效后复权价**
            #   （方案书 §2.2 的 last_valid_close 兜底；停牌期价格本就不变）。
            a_now, price_date = last_valid_adj(adj_close, code, as_of)
            if a_now is None:
                a_now = float("nan")
                price_date = None
            if not np.isfinite(a_now):
                ratio = 1.0
                stale = True
            elif np.isfinite(a_cost) and a_cost > 0:
                ratio = a_now / a_cost
                # 用了**滞后**的价格（停牌 / 涨跌幅未同步 → 最后有效价早于 as_of）
                # → 必须标记 stale（方案书 §2.6）：否则"用旧价冒充当日价"，
                # 会造出假的 0 收益点 → 压低波动、稀释夏普。
                stale = (price_date != str(pd.Timestamp(as_of).date()))
            else:
                ratio = 1.0
                stale = True
        else:
            stale = True
        mv = sh * cost * ratio
        total += mv
        details.append({
            "code": code,
            "shares": sh,
            "cost": cost,
            "adj_ratio": ratio,
            "market_value": mv,
            "stale_price": stale,
            "price_date": price_date,
        })
    return total, details


# ── 进程互斥锁（2026-11-02）──────────────────────────────────────────
# 为什么要：纸面盘今天有**三个排期**都会调 scripts/paper_portfolio.py
#   ① launchd com.rone.tdx-update  16:05（daily_tdx_update.sh 第 5 步，周一）
#   ② Hermes cron tdx-daily-update 17:30（同一个 daily_tdx_update.sh）
#   ③ launchd com.rone.paper-portfolio 17:40（+ 周二 09:20 兜底）
# 实测 2026-11-02：② 跑到 17:44:55，而 ③ 在 17:40:04 就启动了
#   → 两个进程**并发写**同一批 holdings/ledger/portfolio.json
#   → 后写者赢 → 前者的成交明细可能被**静默覆盖**。
#   今晚没出事是运气（② 正好漏了 honest_42f_neu，③ 补上了）。
#
# 语义：**等待，不跳过** —— ③ 若直接跳过，② 漏掉的线就永远补不回来
#   （honest_42f_neu 今晚正是靠 ③ 才补成交成功的）。
LOCK_FILENAME = ".paper_portfolio.lock"


def lock_path() -> str:
    """锁文件路径（**函数内**算 → 测试可重定向 PAPER_DIR）。"""
    return os.path.join(PAPER_DIR, LOCK_FILENAME)


def acquire_run_lock(timeout_s: int = 1800, poll_s: float = 5.0) -> int:
    """取纸面盘互斥锁；返回 fd（调用方须保持它开着直到进程结束）。

    - 拿不到就**等**（最多 timeout_s 秒），不是跳过；
    - 超时 → 打印原因并 sys.exit(75)（EX_TEMPFAIL：临时性失败，值得重试）。
    """
    import fcntl
    import time as _time
    os.makedirs(PAPER_DIR, exist_ok=True)
    fd = os.open(lock_path(), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = _time.time() + max(0, timeout_s)
    waited = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if _time.time() >= deadline:
                os.close(fd)
                print(
                    f"❌ 纸面盘互斥锁等待超时（{timeout_s}s）：另有实例正在跑"
                    f"（锁文件 {lock_path()}）。本次不启动，避免并发写账本。",
                    flush=True,
                )
                sys.exit(75)
            if not waited:
                print(
                    "⏳ 另有纸面盘实例正在跑（可能是 17:30 数据任务那一档）"
                    f"→ 等待它结束（最多 {timeout_s}s）……",
                    flush=True,
                )
                waited = True
            _time.sleep(poll_s)
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, f"pid={os.getpid()} started={_time.strftime('%F %T')}\n".encode())
    return fd


def ledger_buy_dates(strategy: str) -> Dict[str, str]:
    """从 ledger.csv 取每只票**最近一次买入的成交日**（{code: "YYYY-MM-DD"}）。

    为什么需要（2026-11-02）：ledger 的 `date` 列记的是**成交日**（权威），
    而持仓文件里的 `buy_date` 在 2026-11-02 之前记的是**信号日**
    （lag1 口径下买入发生在成交日）→ 用它回填 adj_cost 会取错日期 →
    后复权比值失真。现在统一以 ledger 为准。
    """
    out: Dict[str, str] = {}
    p = ledger_path(strategy)
    if not os.path.exists(p):
        return out
    try:
        import csv as _csv
        with open(p, newline="") as fh:
            for r in _csv.DictReader(fh):
                if str(r.get("action") or "").upper() != "BUY":
                    continue
                code = str(r.get("code") or "")
                d = str(r.get("date") or "").strip()
                if not code or not d:
                    continue
                # ledger 按时间顺序写入 → 后面出现的 BUY 覆盖前面的（最近一次买入）
                if d >= out.get(code, ""):
                    out[code] = d
    except Exception:  # noqa: BLE001
        return out
    return out


# ── 观测口径层（方案书附录 A，2026-11-02）────────────────────────────
# 为什么要有这一层：7 条线的暴露度差 5 倍（17.5% ~ 92.1%），**总盈亏在暴露度
# 不同的情况下不可比**。实测两个已发生的误读：
#   ① 从未交易的线（0 持仓/100% 现金）裸 PnL = 0.00% → 在裸 PnL 榜上排第 2，
#      高于 4 条真正亏了钱的线；
#   ② 付费档 gbm_voltgt 裸 PnL -0.05%（像打平），但它 82.5% 现金 →
#      投入的那 17.5% 实际亏 0.30%。
# 修法：**每条线同时给出「投入段收益」**（已投资部分本身的收益率）+
# 「暴露度」+「as-of 数据日」+「成交笔数」+「状态」，并让**从未交易的线
# 不再显示 0.00%**（显示 — 且 status=未建仓）。

def ledger_stats(strategy: str) -> Dict[str, Any]:
    """该线 ledger 的统计：记录数 / 去重成交日数 / 最后成交日（只读）。

    ledger.csv 的行 = 一笔成交记录（买、卖各算一笔）。
    """
    p = ledger_path(strategy)
    n_rows = 0
    days = set()
    if os.path.exists(p):
        try:
            import csv as _csv
            with open(p, newline="") as fh:
                for r in _csv.DictReader(fh):
                    n_rows += 1
                    d = str(r.get("date") or "").strip()
                    if d:
                        days.add(d)
        except Exception:  # noqa: BLE001
            pass
    days_sorted = sorted(days)
    return {"n_trades": n_rows,
            "n_trade_days": len(days_sorted),
            "last_ledger_date": days_sorted[-1] if days_sorted else None}


def line_observation_metrics(strategy: str, line: Dict[str, Any],
                              capital: float) -> Dict[str, Any]:
    """单条线的观测口径指标。

    **只实现一次**：重建路径（rebuild_line_from_files）与新跑路径的行 dict
    都交给它 → 口径不会漂移。本项目吃过"同套逻辑写两份 → 互相共错"的亏
    （台账 §12.28 的审计器/回填器共错教训），所以这里刻意收口。

    字段：
      as_of               该线估值所用的数据日（= 成交日/最后估值日）
      exposure            暴露度 = 1 − cash/value
      invested_return_pct **投入段收益** = (value−initial)/(value−cash)×100
                          ← 消除暴露度差异，让不同仓位量的线可比
      pnl_pct             总盈亏%；**从未交易过时为 None**（不是 0.00）
      n_trades            累计成交记录数
      n_trade_days        累计成交日数
      status              未建仓 / 持仓中 / 空仓
    """
    h = load_holdings(strategy, capital)
    cash = float(line.get("cash") or 0.0)
    value = float(line.get("value") or 0.0)
    initial = float(h.get("initial_capital") or line.get("initial_capital") or capital)
    st = ledger_stats(strategy)
    n_pos = int(line.get("n_positions") or 0)
    n_trades = int(st["n_trades"])

    exposure = (value - cash) / value if value > 0 else None
    invested = value - cash
    invested_return_pct = ((value - initial) / invested * 100.0) if invested > 1e-9 else None
    # 从未交易过 → 该线的盈亏**没有意义**（0.00% 是"什么都没做"，不是"打平"）
    pnl_pct = ((value - initial) / initial * 100.0) if (initial > 0 and n_trades > 0) else None
    if n_trades == 0:
        status = "未建仓"
    elif n_pos > 0:
        status = "持仓中"
    else:
        status = "空仓"
    # as_of 优先级：本次写盘时记的 value_as_of → last_trade → ledger 最后成交日
    as_of = h.get("value_as_of") or h.get("last_trade") or st["last_ledger_date"]
    return {
        "as_of": as_of,
        "exposure": round(exposure, 6) if exposure is not None else None,
        "invested_return_pct": (round(invested_return_pct, 4)
                                if invested_return_pct is not None else None),
        "pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
        # 展示用字符串：**显示规则只在这里实现一次**，MCP / API / 前端都读它
        # （三处各写一遍=迟早漂移，本项目已吃过这个亏）。
        "pnl_display": ("—" if pnl_pct is None else f"{pnl_pct:.2f}%"),
        "n_trades": n_trades,
        "n_trade_days": int(st["n_trade_days"]),
        "last_trade": h.get("last_trade"),
        "status": status,
        "initial_capital": round(initial, 2),
    }


# ── ledger 列契约（§12.48）──────────────────────────────────────────────
# 固定 15 列 = SELL 行与 BUY 行键集的并集。
#
# 为什么不能继续用 list(rows[0].keys())：
#   fieldnames 由**本批次首行**决定，而表头只在文件不存在时写一次 →
#   ① 首次是 BUY(13 键)、后来追加 SELL(15 键) → 新行 15 列、表头仍 13 列
#      → csv.DictReader 按列名读会**静默错位**：cash_after 读到 cost_amount、
#        portfolio_value 读到 realized_pnl，脏值直接进净值曲线。
#   ② 首次是 SELL(15 键)、后来追加 BUY(13 键) → 新行 13 列，比表头少 2 列。
#   ③ 最隐蔽：列数取决于**批次内首行是谁** → 同一批代码，行数可能不同。
#   （这三条已在 §12.48 用合成数据复现；当前磁盘上还是 13/13 整齐，
#    因为现存 SELL 行是老代码写的 13 键版本 —— **下一次有卖出的调仓即触发**。）
LEDGER_FIELDS = [
    "date", "action", "code", "name", "price", "shares", "amount",
    "commission", "stamp_tax", "slippage", "score",
    "cost_amount", "realized_pnl", "cash_after", "portfolio_value",
]


def _normalize_ledger_file(path: str) -> bool:
    """把旧 ledger 迁移到固定的 LEDGER_FIELDS 列契约。返回是否发生了迁移。

    · 旧表头缺列 → 对应列补空串（老 SELL 行本来就没写 cost_amount/realized_pnl）
    · 旧行长度不齐 → 按**表头位置**取值，多余列丢弃、缺失列补空
    · 先备份原文件到 <path>.bak，再 os.replace 原子替换
    """
    try:
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
    except (OSError, csv.Error):
        return False
    if not rows:
        return False
    header = [h.strip() for h in rows[0]]
    if header == LEDGER_FIELDS:
        return False
    idx = {name: i for i, name in enumerate(header)}
    out_rows: List[List[str]] = []
    for r in rows[1:]:
        if not r:
            continue
        out_rows.append([
            (r[idx[f]] if f in idx and idx[f] < len(r) else "")
            for f in LEDGER_FIELDS
        ])
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
            w.writeheader()
            for r in out_rows:
                w.writerow(dict(zip(LEDGER_FIELDS, r)))
        shutil.copy2(path, f"{path}.bak")
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False
    return True


def append_ledger(strategy: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(PAPER_DIR, exist_ok=True)
    p = ledger_path(strategy)
    exists = os.path.exists(p)
    if exists:
        # 旧文件表头与列契约不一致 → 先原子迁移，避免追加出长短不一的行
        _normalize_ledger_file(p)
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, restval="", extrasaction="ignore")
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
        _lbd = ledger_buy_dates(strategy)
        for _code, _pos in positions.items():
            # ⚠️ 2026-11-02 修正：原来用 truthy 判断 —— **NaN 是 truthy**
            #   → adj_cost 是 NaN 的持仓永远被跳过回填 → 永远修不好。
            #   现在判"是否有效"（非空且有限）。
            # buy_date 应是**成交日**（2026-11-02 起修）；老数据里它记的是
            # 信号日 → 优先用 ledger 的成交日（权威），没有才退回 buy_date。
            _bd = _lbd.get(_code) or _pos.get("buy_date")
            if not _bd:
                continue
            try:
                _ts = pd.Timestamp(_bd)
            except (TypeError, ValueError):
                continue
            # ⚠️ 2026-11-02 再修正：原来"adj_cost 是有效值就跳过" ——
            #   但旧值可能是**信号日**写的（日期错位，分子分母不同日）→
            #   跳过后永远修不好 → 日频/周频对账实测差 300~850 元。
            #   现在：只有当 adj_cost 的**价格日**就是成交日才跳过。
            _ac = _pos.get("adj_cost")
            _acd = str(_pos.get("adj_cost_date") or "")
            if (_ac is not None and np.isfinite(float(_ac))
                    and _acd == str(_ts.date())):
                continue
            # ⚠️ 用「≤ 买入**成交日**的最后有效后复权价」，不是 at[_ts] ——
            #   ① 买入日当天可能还没涨跌幅（见 last_valid_adj 的说明）；
            #   ② 旧代码建仓时按**信号日**写 adj_cost → 分子分母落在不同日期。
            #   → 一律按成交日重算（不看旧值是否"有效"）。
            _v, _vd = last_valid_adj(adj_close, _code, _ts)
            if _v is not None:
                _pos["adj_cost"] = round(_v, 6)
                if _vd:
                    _pos["adj_cost_date"] = _vd
                _filled += 1
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

    def _last_valid_close(code: str) -> float:
        """成交日停牌/数据缺失 → 沿用**最近一个有效收盘价**。

        为什么需要：停牌期间价格不变，沿用最后有效价是正确估值。
        不做这个兜底 → 该股 close=NaN → total_value 变 NaN →
        `int(budget/p/100)` 抛 ValueError，整条线当周直接崩。
        （parity 回测 2023-05 起跑时抓到：持仓里有一只停牌股。）
        """
        s_ = close_mat.loc[:trade_day, code].dropna()
        return float(s_.iloc[-1]) if len(s_) else float("nan")

    _held_values: List[float] = []
    for c in held:
        v = close.get(c, np.nan)
        v = float(v) if np.isfinite(v) else _last_valid_close(c)
        if not np.isfinite(v):
            continue  # 连历史都没有 → 无法估值，剔除（保守）
        _held_values.append(float(positions[c].get("shares", 0)) * v)
    total_value = cash + sum(_held_values)
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
        """成交日后复权价 / 买入日后复权价（口径见 total_return_portfolio_value）。

        ⚠️ 2026-11-02 修正：原来分子用的是 `latest`（**信号日**）→ 净值滞后
        一个交易日（lag1 口径下买入在成交日，分子却停在信号日）。
        """
        _v, _d = total_return_portfolio_value(
            {code: positions.get(code) or {}}, 0.0, adj_close, trade_day)
        return float(_d[0]["adj_ratio"]) if _d else 1.0

    def _pos_value(code: str) -> float:
        """总回报口径的持仓市值 = 买入金额 × 后复权比值。

        2026-11-02：改为调**模块级** total_return_portfolio_value ——
        日频盯市（paper_daily_nav.py）必须与周频**逐位一致**，
        两边各写一份估值迟早漂移（方案书 §3 门禁 a）。
        """
        _v, _d = total_return_portfolio_value(
            {code: positions.get(code) or {}}, 0.0, adj_close, trade_day)
        return float(_d[0]["market_value"]) if _d else 0.0

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
        # 已实现盈亏（总回报口径，扣费用）—— 用于算逐笔胜率
        _cost_amt = float(positions[c].get("cost", 0) or 0) * shares
        _realized = gross - commission - stamp - slip - _cost_amt
        sells.append({"code": c, "name": names.get(c, ""), "price": p,
                      "shares": shares, "amount": round(gross, 2),
                      "real_amount": round(real_gross, 2),
                      "cost_amount": round(_cost_amt, 2),
                      "realized_pnl": round(_realized, 2),
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

    def _record_buy(c: str, p: float, shares: int) -> None:
        """记一笔买入（扣现金 + 写持仓 + 追加成交明细）。"""
        nonlocal cash
        gross = shares * p
        commission = max(MIN_COMMISSION, gross * COMMISSION)
        slip = gross * SLIPPAGE
        cost = gross + commission + slip
        if cost > cash:
            return
        cash -= cost
        a_cost = None
        if adj_close is not None and c in adj_close.columns:
            try:
                # ⚠️ 2026-11-02 修正（真 bug，污染了今晚的账本）：
                #   原来取 `latest`（**信号日**）的后复权价，但买入发生在**成交日**
                #   （lag1 口径）→ 分子分母落在不同日期 → 比值失真；
                #   更糟的是信号日该票没有后复权价时得到 NaN，
                #   而 `if a_cost else None` 里 **NaN 是 truthy** → NaN 被写进持仓
                #   → `_adj_ratio` 因 `not np.isfinite(a_cost)` 恒返回 1.0
                #   → **该票市值被冻结在买入金额，永不反映涨跌**。
                #   实测 2026-11-02 首跑：multi_asset_final **69/69 只**、
                #   其余各线 3~12 只的 adj_cost 无效 → 那条线当时显示的
                #   "-0.10%" 其实是**手续费损耗**，不含任何市场波动。
                #   现在统一用**成交日** trade_day。
                a_cost, _a_date = last_valid_adj(adj_close, c, trade_day)
            except (KeyError, TypeError, ValueError):
                a_cost = None
        positions[c] = {"shares": shares, "cost": round(p, 3),
                        "buy_date": str(trade_day.date()),
                        # 买入日（**成交日**）的后复权价 —— 持有期真实收益（无除权跳空）
                        "adj_cost": (round(a_cost, 6)
                                     if a_cost is not None and np.isfinite(a_cost)
                                     else None)}
        buys.append({"code": c, "name": names.get(c, ""), "price": p,
                     "shares": shares, "amount": round(gross, 2),
                     "commission": round(commission, 2), "stamp_tax": 0.0,
                     "slippage": round(slip, 2), "total_cost": round(cost, 2),
                     "score": round(float(rank.get(c, np.nan)), 4),
                     "factors": _explain_factors(c, comp.columns)})

    if spec.get("lot_remainder"):
        # ── lot_remainder 模式：整手零头分配 ─────────────────────────
        # 为什么需要（2026-11-01 多资产线实测）：
        #   整手约束（100 股）下，`int(budget/p/100)*100` 每只都要**丢掉零头**，
        #   丢掉的部分变成**闲置现金**。A 腿预算只有 50万×0.95/60 ≈ 7,900 元，
        #   股价 10~40 元时一手 1,000~4,000 元 → 每只平均丢 ~1,000 元，
        #   60 只合计丢 ~6 万 = **A 腿 12% 的资金躺在现金里**（实测投入率 86.1%，
        #   预期 95%）。研究端用小数股，没有这个问题 → 产品端白丢 12% 的仓位。
        #   做法：先按整手向下取整记下零头，再把剩余现金按**零头最大优先**
        #   补成整手（每只最多补一手，且不能跌破现金储备线）。
        #   ⚠️ 只对开了这个开关的线生效：其余 5 条线是**既有纸面账户**，
        #      改它们的历史口径会让已展示的净值无法复现 —— 要不要一起改，
        #      走「先给方案、等拍板」的流程。
        reserve = total_value * CASH_RESERVE
        sim_cash = cash
        plan: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for c in new_in[:max_buys]:
            p = px(c)
            if not np.isfinite(p) or p <= 0:
                continue
            tr = _tradability(c, latest, names, extras, close=close,
                              open_price=open_price, high=high, low=low,
                              volume=vol, close_prev=close_prev)
            if not tr.can_buy:
                blocked_buys.append({"code": c, "name": names.get(c, ""),
                                     "reason": tr.blocked_reason or tr.reason})
                continue
            budget = min(
                total_value * target_weight,
                max(0.0, sim_cash - reserve),
            )
            if tr.max_buy_amount is not None:
                budget = min(budget, tr.max_buy_amount)
            shares = int(budget / p / BOARD_LOT) * BOARD_LOT
            lot_cost = BOARD_LOT * p
            if shares <= 0:
                # 买不起一手 → 整只跳过（默认行为）。
                # lot_cover 模式：只要一手不超过目标预算的 lot_cover_max 倍，
                # 就**买一手**（宁可这只超配，也不要整只空着）。
                # 为什么需要：Top60 策略在产品端显示 58 只会让人以为坏了；
                # 一手超配 50% → 该股权重 +0.83%（组合口径），可接受。
                cover_max = float(spec.get("lot_cover_max", 1.0))
                if lot_cost <= 0 or cover_max <= 1.0:
                    continue
                if lot_cost > budget * cover_max + 1e-6:
                    continue
                if lot_cost + max(MIN_COMMISSION, lot_cost * COMMISSION) \
                        + lot_cost * SLIPPAGE > sim_cash:
                    continue
                shares = BOARD_LOT
                budget = lot_cost  # 零头 = 0（已经买满一手）
            gross = shares * p
            cost = gross + max(MIN_COMMISSION, gross * COMMISSION) + gross * SLIPPAGE
            if cost > sim_cash:
                continue
            sim_cash -= cost
            plan[c] = {"price": p, "shares": shares,
                       "remainder": budget / p - shares, "boosted": False,
                       # 零头按**金额**排序（不是股数）：一手金额差 20 倍
                       # （511020 ≈1万 vs 518880 ≈500），按股数排会让高价
                       # 标的继续欠配。
                       "remainder_value": (budget / p - shares) * p}
            order.append(c)
        # 零头最大优先补一手（每只最多一次，且不跌破储备线）
        for _ in range(len(order)):
            best: Optional[str] = None
            for c in order:
                it = plan[c]
                if it["boosted"]:
                    continue
                extra = BOARD_LOT * it["price"]
                extra_cost = (extra + max(MIN_COMMISSION, extra * COMMISSION)
                              + extra * SLIPPAGE)
                if sim_cash - extra_cost < reserve - 1e-6:
                    continue
                if (best is None
                        or it["remainder_value"] > plan[best]["remainder_value"]):
                    best = c
            if best is None:
                break
            extra = BOARD_LOT * plan[best]["price"]
            sim_cash -= (extra + max(MIN_COMMISSION, extra * COMMISSION)
                         + extra * SLIPPAGE)
            plan[best]["shares"] += BOARD_LOT
            plan[best]["boosted"] = True
        for c in order:
            _record_buy(c, plan[c]["price"], plan[c]["shares"])
    else:
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
            _record_buy(c, p, shares)

    end_value = cash + sum(_pos_value(c) for c in positions)

    # ── 净值历史（2026-10-27）────────────────────────────────────
    # 为什么加：6 窗口稳定性检验显示策略**最近一年（2025Q4）夏普只有 0.221**
    # （而 2025 上半年是 2.887）→ 策略当下处于低效期，必须能对用户诚实披露
    # 「这一年赚了多少」，而不是只报累计收益。
    _cap0 = float(paper.get("initial_capital") or 1_000_000)
    _nav_hist = list(paper.get("nav_history") or [])
    _nav_hist.append({
        "date": str(trade_day.date()),
        "value": round(end_value, 2),
        "return_pct": round((end_value / _cap0 - 1.0) * 100, 4) if _cap0 else 0.0,
    })
    # 只保留最近 3 年，避免文件无限增长
    if len(_nav_hist) > 800:
        _nav_hist = _nav_hist[-800:]

    paper.update({"cash": round(cash, 2), "positions": positions,
                  "nav_history": _nav_hist,
                  # last_signal = **信号日**（周五出信号）
                  # last_trade  = **成交日**（下一交易日成交，lag1 口径）
                  "last_signal": str(signal_day.date()),
                  "last_trade": str(trade_day.date()),
                  "last_value": round(end_value, 2),
                  # value_as_of：这份市值是用**哪一天**的收盘价算的
                  # （as-of 日期是观测口径层的硬要求：没有它就无法知道
                  #  各条线在拿不同日期的市值做比较）
                  "value_as_of": str(trade_day.date()),
                  "fill_assumption": "Friday close"})
    save_holdings(strategy, paper)

    rows = []
    for s in sells:
        rows.append({"date": str(trade_day.date()), "action": "SELL",
                     "code": s["code"], "name": s["name"], "price": s["price"],
                     "shares": s["shares"], "amount": s["amount"],
                     "commission": s["commission"], "stamp_tax": s["stamp_tax"],
                     "slippage": s["slippage"], "score": s["score"],
                     "cost_amount": s.get("cost_amount", 0.0),
                     "realized_pnl": s.get("realized_pnl", 0.0),
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



# ── 多资产组合线（multi_asset_final）────────────────────────────────
# 定稿口径见 docs/backtest/MULTI_ASSET_LEDGER_2026-10-31.md §十 / §十一：
#   A股腿 42 因子 Top60/ndrop5 占 50%，ETF 袖套占 50%，**静态划分、腿间不再平衡**。
# 研究值：组合夏普 main 1.4114 / oos 1.0691（噪声带 ±0.05，见台账 §11.4）。
#
# 为什么拆成两个 state key：
#   A 腿要复用 `run_line` 的**唯一一份**撮合实现（复制撮合器必然漂移），
#   但它不能覆盖既有 `honest_42f` 那条线的持仓文件 → 独立 state key。
#   袖套腿是 ETF，规则完全不同（无印花税、无涨跌停、目标权重再平衡）→ 独立实现。
MULTI_ASSET_A_KEY = "multi_asset_final_a"
MULTI_ASSET_SLEEVE_KEY = "multi_asset_final_sleeve"
MULTI_ASSET_A_WEIGHT = 0.50
MULTI_ASSET_SLEEVE_WEIGHT = 0.50

#: 袖套 ETF 显示名（取自 data/etf/etf_universe.json，2026-11-01 核对）
ETF_NAMES: Dict[str, str] = {
    "518880": "华安黄金ETF",
    "159934": "易方达黄金ETF",
    "159937": "博时黄金ETF",
    "511260": "上证10年期国债ETF",
    "511010": "国泰上证5年期国债ETF",
    "511020": "平安5-10年期国债活跃券ETF",
    "159980": "大成有色金属期货ETF",
    "159985": "华夏饲料豆粕期货ETF",
    "159981": "建信能源化工期货ETF",
}


def _sleeve_codes() -> List[str]:
    return [c for lst in SLEEVE_HELD.values() for c in lst]


def _ledger_cash_timeline(strategy: str) -> Dict[str, float]:
    """从 ledger 重建「日期 → 现金」时间线（逐日前向填充）。

    合并账本需要每条腿在同一日期的现金，才能算合并后的 cash_after。
    """
    out: Dict[str, float] = {}
    p = ledger_path(strategy)
    if not os.path.exists(p):
        return out
    try:
        import csv as _csv
        with open(p, newline="") as fh:
            for r in _csv.DictReader(fh):
                d = str(r.get("date") or "")
                ca = _tofloat(r.get("cash_after"))
                if d and ca is not None:
                    out[d] = ca
    except Exception:  # noqa: BLE001
        pass
    return out


def _nav_value_map(strategy: str) -> Dict[str, float]:
    """持仓文件里的 nav_history → {日期: 净值}。"""
    p = holdings_path(strategy)
    out: Dict[str, float] = {}
    if not os.path.exists(p):
        return out
    try:
        with open(p) as fh:
            h = json.load(fh)
        for row in (h.get("nav_history") or []):
            d = str(row.get("date") or "")
            v = _tofloat(row.get("value"))
            if d and v is not None:
                out[d] = v
    except Exception:  # noqa: BLE001
        pass
    return out


def _write_multi_asset_ledger(
    trade_date: str,
    merged_value: float,
) -> None:
    """把两条腿的账本合并成 multi_asset_final_ledger.csv（每次全量重写，幂等）。

    为什么全量重写而不是追加：两条子账本是「追加式」的，重复跑会重复下单；
    合并账本若也追加，同一个成交日会出现两行。**重写 = 天然幂等**。
    """
    import csv as _csv

    a_cash = _ledger_cash_timeline(MULTI_ASSET_A_KEY)
    s_cash = _ledger_cash_timeline(MULTI_ASSET_SLEEVE_KEY)
    a_nav = _nav_value_map(MULTI_ASSET_A_KEY)
    s_nav = _nav_value_map(MULTI_ASSET_SLEEVE_KEY)

    def _fill_last(dct: Dict[str, float], date: str) -> float:
        """取 date 当天或之前最近的一个值（没有则 0）。"""
        if date in dct:
            return dct[date]
        prev = [d for d in dct if d <= date]
        return dct[max(prev)] if prev else 0.0

    rows: List[Dict[str, Any]] = []
    for strategy, leg in ((MULTI_ASSET_A_KEY, "a"), (MULTI_ASSET_SLEEVE_KEY, "sleeve")):
        p = ledger_path(strategy)
        if not os.path.exists(p):
            continue
        with open(p, newline="") as fh:
            for r in _csv.DictReader(fh):
                d = str(r.get("date") or "")
                if not d:
                    continue
                row = dict(r)
                row["leg"] = leg
                # 合并口径：现金 = 两腿之和（缺一侧则前向填充最近值）
                row["cash_after"] = round(
                    _fill_last(a_cash, d) + _fill_last(s_cash, d), 2
                )
                # 组合净值 = 两腿净值之和
                row["portfolio_value"] = round(
                    _fill_last(a_nav, d) + _fill_last(s_nav, d), 2
                )
                rows.append(row)
    if not rows:
        return
    rows.sort(key=lambda x: (str(x.get("date")), str(x.get("action")),
                             str(x.get("code"))))
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    os.makedirs(PAPER_DIR, exist_ok=True)
    with open(ledger_path("multi_asset_final"), "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _invested_ratio(value: float, cash: float) -> float:
    """投入率 = (总市值 - 现金) / 总市值。0 视为未建仓。"""
    if not value:
        return 0.0
    return round(max(0.0, min(1.0, (value - cash) / value)), 6)


def _append_nav(nav: List[Dict[str, Any]], point: Dict[str, Any]) -> List[Dict[str, Any]]:
    """净值历史追加一个点；**同日期则覆盖**（不是追加）。

    为什么：同一成交日重跑（幂等重试 / 手动补跑）时，如果直接 append，
    净值历史会出现两个同日期的点 → 收益率序列出现一个假的 0% 区间，
    夏普/年化被稀释。实测 2026-11-01 E2E 抓到：重跑后 nav_history 有 2 个
    点但日期相同。
    """
    out = list(nav or [])
    if out and str(out[-1].get("date")) == str(point.get("date")):
        out[-1] = point
    else:
        out.append(point)
    return out[-800:]


def run_multi_asset_line(
    frames: Dict[str, Any],
    panel: Dict[str, Any],
    master: pd.DatetimeIndex,
    names: Dict[str, str],
    capital: float,
    rebalance_day: pd.Timestamp,
    extras: Optional[Dict[str, Any]] = None,
    adj_close: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """多资产组合线：A 股腿（42 因子 Top60）+ ETF 袖套（gold/bond/commodity 各 1/3）。

    两条腿**静态 50/50 划分、腿间不再平衡**——这正是研究侧的
    `merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])` 在账户层面的等价物。
    """
    a_capital = capital * MULTI_ASSET_A_WEIGHT
    s_capital = capital * MULTI_ASSET_SLEEVE_WEIGHT

    # ── 成交日（信号日之后的下一个交易日，lag1）────────────────────────
    # 先算成交日：A 腿要不要「复用」取决于它在这个成交日是否已经调过仓。
    signal_day = pd.Timestamp(rebalance_day)
    _after = master[master > signal_day]
    if len(_after) == 0:
        raise SystemExit(
            f"❌ 多资产线：信号日 {signal_day.date()} 之后还没有交易日的行情数据"
            f"（lag1 口径需要下一交易日成交）。"
        )
    trade_day = _after[0]
    trade_idx = int(master.get_loc(trade_day))

    # ── A 腿：复用 honest_42f 的同一套打分 + 撮合实现 ──────────────────
    a_spec = {
        "score": score_42f, "topk": 60, "ndrop": 5,
        # 整手零头分配：A 腿只有 50 万预算，不开这个开关会有 ~12% 的资金
        # 因整手取整闲置（实测投入率 86.1% vs 预期 95%），与研究端
        # （小数股、100 万）不可比。详见 run_line 里的说明。
        "lot_remainder": True,
        # 买不起一手的标的可买一手（最多超目标预算 50%），
        # 避免 Top60 在产品端只显示 58 只。
        "lot_cover_max": 1.5,
        "desc": "多资产组合 · A股腿（42 因子 Top60 / n_drop 5，占 50%）",
    }
    # 2026-11-01：**A 腿的幂等保护**。
    #   场景：本周袖套（ETF 数据没到）失败 → A 腿已经成交并写了持仓/账本 →
    #   下一周补跑同一成交日时，如果再跑一次 run_line，就会**重复撮合**：
    #   同一天多出一批买卖记录、佣金重复扣、合并账本出现同日期两行。
    #   规则：A 腿在这个成交日已经成交过 → 直接复用持仓文件 + 账本，
    #   只补袖套腿，不重复下单。
    _a_pre = load_holdings(MULTI_ASSET_A_KEY, a_capital)
    _a_pre_trade = _a_pre.get("last_trade")
    _a_done = (
        bool(_a_pre_trade)
        and str(pd.Timestamp(_a_pre_trade).date()) == str(trade_day.date())
    )
    if _a_done:
        print(f"  [A腿] 成交日 {trade_day.date()} 已调过仓 → 复用现有持仓，不重复撮合",
              flush=True)
        a_res = rebuild_line_from_files(MULTI_ASSET_A_KEY, a_capital)
        # 复用时 a_res 的 buys/sells 来自 A 腿账本里**这个成交日**的成交记录。
        # 什么时候该把它们当成本次明细报出去？
        #   合并线还没覆盖这个成交日（_m_catching_up）→ 说明上周那次跑挂在袖套腿，
        #   合并持仓文件根本没写 → **用户从没收到过这份信号** → 必须补报。
        #   合并线已经覆盖这个成交日 → 上次已经报过 → 不重复报（否则重复推送）。
        _m_pre = load_holdings("multi_asset_final", capital)
        _m_pre_trade = _m_pre.get("last_trade")
        _m_catching_up = not (
            _m_pre_trade
            and str(pd.Timestamp(_m_pre_trade).date()) == str(trade_day.date())
        )
        if not _m_catching_up:
            a_res["buys"] = []
            a_res["sells"] = []
    else:
        a_res = run_line(
            MULTI_ASSET_A_KEY, a_spec, frames, panel, master, names,
            a_capital, rebalance_day, extras=extras, adj_close=adj_close,
        )

    # ── 袖套腿 ────────────────────────────────────────────────────────
    codes = _sleeve_codes()
    miss = missing_codes(codes)
    if miss:
        raise SystemExit(
            f"❌ 多资产线：袖套 ETF 缺少本地行情 {miss}。\n"
            f"   持仓集退化会让权重静默摊到别的票上（9 只变 8 只 → 权重全错），"
            f"必须先补数据：\n"
            f"   cd backend && .venv-py313/bin/python3 scripts/etf_data_pipeline.py "
            f"--codes {','.join(codes)}"
        )
    etf_bars = load_etf_bars(codes)

    s_paper = load_holdings(MULTI_ASSET_SLEEVE_KEY, s_capital)
    s_positions = dict(s_paper.get("positions") or {})
    s_cash = float(s_paper.get("cash", s_capital))
    anchor = s_paper.get("reb_anchor_idx")
    last_reb_idx = s_paper.get("last_reb_idx")
    first_build = (anchor is None) or (not s_positions)

    target = sleeve_target_weights()
    if first_build:
        anchor = trade_idx
        reb = True
    else:
        # ⚠️ 必须排除「同一个成交日已经调过仓」的情况：
        #    anchor 是在首次建仓时设成 trade_idx 的，而
        #    is_rebalance_day(trade_idx, anchor) = (trade_idx-anchor)%21 == 0
        #    在 anchor 当天**恒为真** → 同一成交日重跑（幂等重试 / 手动补跑）
        #    会再调一次仓，多出一堆买卖记录（实测账本 76 → 85 行）。
        #    用 last_reb_idx 记住「上次调仓发生在哪个交易日索引」来去重。
        reb = (
            is_rebalance_day(trade_idx, int(anchor))
            and (last_reb_idx is None or int(last_reb_idx) != trade_idx)
        )

    s_buys: List[Dict[str, Any]] = []
    s_sells: List[Dict[str, Any]] = []
    if reb:
        try:
            s_positions, s_cash, s_buys, s_sells = rebalance_sleeve(
                s_positions, s_cash, etf_bars, trade_day, target,
                # 产品侧严格：成交价必须是成交当日收盘价（研究侧允许容忍 3 天，
                # 但那会让 ETF 数据晚到时拿上周五的价成交 → 净值失真）
                require_exact_date=True,
            )
        except SleeveError as e:
            raise SystemExit(f"❌ 多资产线：{e}")

    s_value = s_cash + sleeve_positions_value(s_positions, etf_bars, trade_day)
    s_classes = sleeve_class_weights_now(s_positions, etf_bars, trade_day)

    s_paper.update({
        "cash": round(s_cash, 2),
        "positions": s_positions,
        "reb_anchor_idx": int(anchor),
        "last_reb_idx": int(trade_idx) if reb else last_reb_idx,
        "last_signal": str(signal_day.date()),
        "last_trade": str(trade_day.date()),
        "last_value": round(s_value, 2),
        "value_as_of": str(trade_day.date()),
        "initial_capital": s_capital,
        "fill_assumption": "lag1：周五收盘出信号，下一交易日收盘成交",
    })
    s_nav = _append_nav(s_paper.get("nav_history"), {
        "date": str(trade_day.date()),
        "value": round(s_value, 2),
        "return_pct": round((s_value / s_capital - 1.0) * 100, 4) if s_capital else 0.0,
    })
    s_paper["nav_history"] = s_nav
    save_holdings(MULTI_ASSET_SLEEVE_KEY, s_paper)

    s_rows: List[Dict[str, Any]] = []
    for s in s_sells:
        s_rows.append({
            "date": str(trade_day.date()), "action": "SELL",
            "code": s["code"], "name": ETF_NAMES.get(s["code"], s["code"]),
            "price": s["price"], "shares": s["shares"], "amount": s["amount"],
            "commission": s["commission"], "stamp_tax": s["stamp_tax"],
            "slippage": s["slippage"], "score": "",
            "cost_amount": s.get("cost_amount", 0.0),
            "realized_pnl": s.get("realized_pnl", 0.0),
            "cash_after": round(s_cash, 2), "portfolio_value": round(s_value, 2),
        })
    for b in s_buys:
        s_rows.append({
            "date": str(trade_day.date()), "action": "BUY",
            "code": b["code"], "name": ETF_NAMES.get(b["code"], b["code"]),
            "price": b["price"], "shares": b["shares"], "amount": b["amount"],
            "commission": b["commission"], "stamp_tax": b["stamp_tax"],
            "slippage": b["slippage"], "score": "",
            "cash_after": round(s_cash, 2), "portfolio_value": round(s_value, 2),
        })
    append_ledger(MULTI_ASSET_SLEEVE_KEY, s_rows)

    # ── 合并 ──────────────────────────────────────────────────────────
    # ⚠️ A 腿的持仓**不在 run_line 的返回值里**（run_line 只返回汇总字段，
    #    持仓由它自己写进 holdings 文件）。直接 a_res.get("positions") 永远
    #    拿到空 dict → 合并持仓只剩袖套 9 只（实测 2026-11-01 E2E 抓到）。
    #    正确做法：从 A 腿的持仓文件读（run_line 刚刚保存过）。
    a_holdings = load_holdings(MULTI_ASSET_A_KEY, a_capital)
    a_positions = a_holdings.get("positions") or {}
    if not a_positions and int(a_res.get("n_positions") or 0) > 0:
        raise SystemExit(
            f"❌ 多资产线：A 腿持仓文件与本次调仓结果不一致"
            f"（持仓文件 {len(a_positions)} 只，撮合结果 {a_res.get('n_positions')} 只）"
            f"—— 拒绝写出一份持仓残缺的合并账本。"
        )
    merged_positions: Dict[str, Any] = {}
    for c, p in a_positions.items():
        q = dict(p)
        q["leg"] = "a"
        merged_positions[c] = q
    for c, p in s_positions.items():
        q = dict(p)
        q["leg"] = "sleeve"
        q["name"] = ETF_NAMES.get(c, c)
        merged_positions[c] = q

    merged_cash = float(a_res.get("cash", 0.0)) + s_cash
    merged_value = float(a_res.get("value", 0.0)) + s_value

    m_paper = load_holdings("multi_asset_final", capital)
    m_nav = _append_nav(m_paper.get("nav_history"), {
        "date": str(trade_day.date()),
        "value": round(merged_value, 2),
        "return_pct": round((merged_value / capital - 1.0) * 100, 4) if capital else 0.0,
    })
    m_paper.update({
        "cash": round(merged_cash, 2),
        "positions": merged_positions,
        "nav_history": m_nav,
        "last_signal": str(signal_day.date()),
        "last_trade": str(trade_day.date()),
        "last_value": round(merged_value, 2),
        "value_as_of": str(trade_day.date()),
        "initial_capital": capital,
        "fill_assumption": "lag1：周五收盘出信号，下一交易日收盘成交",
        "legs": {
            "a": {
                "name": "A股腿（42 因子 Top60）", "weight": MULTI_ASSET_A_WEIGHT,
                "value": round(float(a_res.get("value", 0.0)), 2),
                "cash": round(float(a_res.get("cash", 0.0)), 2),
                "n_positions": len(a_positions),
                # 投入率 = 持仓市值 / 总市值。**不是 1 就说明有闲置现金**：
                # A 腿的每只预算只有 50万×0.95/60 ≈ 7,900 元，
                # 股价 >79 元的股票买不起一手（100 股）→ 整只跳过、预算闲置。
                # 这个数直接决定产品端能不能复现研究端的夏普，必须显式暴露。
                "invested_ratio": _invested_ratio(
                    float(a_res.get("value", 0.0)), float(a_res.get("cash", 0.0))),
            },
            "sleeve": {
                "name": "ETF 袖套（黄金/债券/商品 各 1/3）",
                "weight": MULTI_ASSET_SLEEVE_WEIGHT,
                "value": round(s_value, 2), "cash": round(s_cash, 2),
                "n_positions": len(s_positions),
                "invested_ratio": _invested_ratio(s_value, s_cash),
                "rebalanced": reb,
                # ⚠️ 调仓当天要显示"距下次调仓 21 个交易日"，不是 0。
                #   公式本身就够：(trade_idx - anchor) % 21 == 0（刚调过）→ 21。
                #   之前多写了个 `0 if reb`，把刚调完仓显示成"0 天后还要调"，
                #   前端会显示成"今天还要调仓"。
                "next_rebalance_in_trading_days": (
                    SLEEVE_REB_EVERY - ((trade_idx - int(anchor)) % SLEEVE_REB_EVERY)
                ),
                "class_weights_now": {k: round(v, 6) for k, v in s_classes.items()},
                "target_weights": {k: round(v, 6) for k, v in target.items()},
            },
        },
    })
    save_holdings("multi_asset_final", m_paper)
    _write_multi_asset_ledger(str(trade_day.date()), merged_value)

    # 袖套的买卖明细补上 ETF 名称（合并进总账本 / 推送时要显示名字，
    # 否则前端只能显示 6 位代码）；A 腿的明细自带名称。
    for b in s_buys:
        b.setdefault("name", ETF_NAMES.get(str(b.get("code", "")), ""))
    for s in s_sells:
        s.setdefault("name", ETF_NAMES.get(str(s.get("code", "")), ""))
    buys = list(a_res.get("buys") or []) + s_buys
    sells = list(a_res.get("sells") or []) + s_sells
    for b in buys:
        b.setdefault("name", "")
    for s in sells:
        s.setdefault("name", "")

    return {
        "strategy": "multi_asset_final",
        "signal_date": str(signal_day.date()),
        "trade_date": str(trade_day.date()),
        "desc": STRATEGIES["multi_asset_final"]["desc"],
        "date": str(trade_day.date()),
        "topk": 60, "ndrop": 5,
        "cash": round(merged_cash, 2),
        "value": round(merged_value, 2),
        "n_positions": len(merged_positions),
        "target_weight": round(1.0 / 60, 6),
        "sells": sells, "buys": buys,
        "blocked_buys": list(a_res.get("blocked_buys") or []),
        "blocked_sells": list(a_res.get("blocked_sells") or []),
        "holds": list(merged_positions.keys()),
        "legs": m_paper["legs"],
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

    # ── 守卫（2026-11-02）：ledger 有成交、但 holdings 文件缺失 → 拒绝重建 ──
    # 为什么必须挡：load_holdings() 在文件缺失时返回**空壳**
    # （cash=capital、positions={}），而"没有 holdings 文件"对 multi_asset_final
    # 来说是**合法状态**（它从没建过仓）→ 对这两件事必须区别对待：
    #   · ledger 也没有成交 → 空壳是对的（真的没建仓）；
    #   · ledger **有**成交 → holdings 文件丢了 = 数据残缺，
    #     照空壳重建会让这条线在 portfolio.json 里被静默改成"未建仓"
    #     （展示层看起来像"这条线没做过"，实际它做过，ledger 还在）。
    # → 抛异常：上层 main 循环会打印告警并**保留 portfolio.json 里的旧值**，
    #   ledger / 其它文件一概不动（可人工从 ledger 恢复）。
    _hp = holdings_path(strategy)
    if not os.path.exists(_hp):
        _st = ledger_stats(strategy)
        if _st.get("n_trades"):
            raise RuntimeError(
                f"holdings 文件缺失但 ledger 有 {_st['n_trades']} 笔成交"
                f"（{os.path.basename(_hp)} 不存在）→ 拒绝重建，"
                f"保留 portfolio.json 旧值；请从 ledger 恢复持仓文件"
            )

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
                # ⚠️ 2026-10-27：ledger.csv 的列名是 **shares**（不是 quantity）。
                #    原来读 r.get("quantity") → 永远 None → 股数 0、费用 0，
                #    用户看到的执行清单是「× 0 股 = 0 元」→ 无法照着下单。
                _sh = _tofloat(r.get("shares"))
                if _sh is None:
                    _sh = _tofloat(r.get("quantity"))
                # 费用 = 佣金 + 印花税（仅卖出）+ 滑点
                # （ledger.csv 没有 fee 列 → 不能读，直接按明细求和）
                _fee = ((_tofloat(r.get("commission")) or 0.0)
                        + (_tofloat(r.get("stamp_tax")) or 0.0)
                        + (_tofloat(r.get("slippage")) or 0.0))
                item = {
                    "code": r.get("code", ""),
                    "name": r.get("name", ""),
                    "price": _tofloat(r.get("price")),
                    "shares": int(_sh or 0),
                    "quantity": int(_sh or 0),      # 向后兼容旧字段名
                    "amount": _tofloat(r.get("amount")),
                    "fee": _fee,
                    "commission": _tofloat(r.get("commission")) or 0.0,
                    "stamp_tax": _tofloat(r.get("stamp_tax")) or 0.0,
                    "slippage": _tofloat(r.get("slippage")) or 0.0,
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


def rebuild_portfolio_lines(capital: float = 1_000_000.0,
                            strict: bool = True) -> Dict[str, Any]:
    """从持仓文件 + ledger 重建**所有线**的展示数据，并补齐观测口径层。

    **为什么单独抽出来**（2026-11-03）：重建路径与周频主路径曾各走各的 ——
    周频主路径会 `update(line_observation_metrics(...))` 补
    `exposure / invested_return_pct / pnl_display / status / as_of`，
    而重建路径只调 `rebuild_line_from_files` → **字段全丢**，
    门禁 a/b/c 读到 None 全挂、前端 `signal_service` 的暴露度/投入段收益也变 None。
    → 同套逻辑必须只留一份（台账 §12.28 的共错教训）。

    strict=True：任一条线重建/观测失败 → **抛异常**（守卫：结果不可信就别出数）。
    """
    out: Dict[str, Any] = {}
    for name in list(STRATEGIES.keys()):
        try:
            ln = rebuild_line_from_files(name, capital)
            ln.update(line_observation_metrics(name, ln, capital))
        except Exception as e:  # noqa: BLE001
            if strict:
                raise RuntimeError(
                    f"重建 {name} 失败：{type(e).__name__}: {e}") from e
            print(f"  重建 {name} 失败（保留旧值）: {e}", flush=True)
            continue
        out[name] = ln
    return out


def _tofloat(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0


# ── 等 TDX 当日数据落地（2026-11-02 新增）─────────────────────────────
# 项目根 = backend/scripts/ → 上三级（backend/scripts → backend → 项目根）
RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "tdx_daily", "raw")


def _raw_partitions() -> List[str]:
    """raw 分区里的日期字符串（升序）—— TDX 包落地就会多一个分区。"""
    try:
        names = os.listdir(RAW_DIR)
    except OSError:
        return []
    out = []
    for n in names:
        if n.startswith("date="):
            out.append(n.split("=", 1)[1])
    return sorted(out)


def wait_for_trade_data(signal_day: pd.Timestamp, max_min: int = 45) -> str:
    """等"信号日的下一个交易日"的 TDX 包落地。

    为什么需要（2026-11-02 实测排期竞态）：
      · 纸面盘排期 `com.rone.paper-portfolio` 是 **周一 17:40**；
      · TDX 当日日线包实测 **~17:30~18:00 才落地**（今天 17:35 到位，运气好）；
      · 数据更新脚本已改为"≥17:00 失败则每 30 分钟重试，最晚 18:30"；
      · 若 17:40 时数据还没到 → `run_line` 会 `SystemExit`
        → **这一周的调仓直接作废**（只能等周二 09:20 兜底，用滞后数据）。

    做法：只等 **raw 分区里出现晚于 signal_day 的日期**（= TDX 包落地的标志），
    每 3 分钟查一次，最多 max_min 分钟。
    触发条件收紧（避免非交易日白等）：
      · 当前是**周一~周五**（调仓日）；
      · 当前时间在 **17:00~19:30**（盘后数据落地窗口）；
      · raw 分区里目前**还没有**晚于 signal_day 的日期。
    返回 "ready"（本来就绪）/ "waited"（等到）/ "timeout"（超时）。
    """
    import datetime as _dt
    import time as _time
    now = _dt.datetime.now()
    if not (1 <= now.weekday() <= 5):          # 0=周一
        return "ready"
    if not (17 <= now.hour <= 19):
        return "ready"
    today = now.date().isoformat()
    last = str(pd.Timestamp(signal_day).date())
    parts = _raw_partitions()
    if last >= today or any(p >= today for p in parts):
        return "ready"                     # 当天数据已到位 → 不用等
    deadline = _time.time() + max(0, max_min) * 60
    waited = 0
    while _time.time() < deadline:
        parts = _raw_partitions()
        if any(p > last for p in parts):
            print(f"  [wait] TDX 数据已到位（最新分区 {max(parts)}，"
                  f"等了 {waited // 60} 分钟）", flush=True)
            return "waited"
        if waited == 0:
            print(f"  [wait] 当日（{today}）的 TDX 日线包还没落地，"
                  f"最多等 {max_min} 分钟（排期 17:40 vs 包落地 ~17:30~18:00）",
                  flush=True)
        _time.sleep(180)
        waited += 180
    print(f"  [wait] 等待超时（{max_min} 分钟）→ 数据仍缺，本次可能拒绝成交", flush=True)
    return "timeout"


def main() -> None:
    ap = argparse.ArgumentParser(description="多策略纸面盘（每策略一条独立线）")
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--force", action="store_true", help="非周五也跑")
    args = ap.parse_args()

    # 互斥锁：防止 16:05 / 17:30 / 17:40 三档并发写同一批账本文件
    # （拿不到就等待，等待超时才退出 —— 见 acquire_run_lock 的说明）
    _lock_fd = acquire_run_lock()

    codes = ds.list_universe()
    frames, panel, master = ds.load_data(codes)

    # ── 等当日 TDX 数据落地（2026-11-02 新增）──────────────────────
    #   排期 17:40 vs TDX 包落地 ~17:30~18:00 → 数据没到时 run_line 会
    #   直接 SystemExit → **这一周的调仓作废**（等周二 09:20 兜底用滞后数据）。
    #   → 这里等到包落地后**重新加载**，让成交日/价格用上最新数据。
    _wait_min = int(os.environ.get("PAPER_WAIT_DATA_MIN", "45") or 0)
    if _wait_min > 0:
        _st = wait_for_trade_data(master, max_min=_wait_min)
        if _st == "waited":
            _f2, _p2, _m2 = ds.load_data(codes)
            if len(_m2) > len(master):
                frames, panel, master = _f2, _p2, _m2
                print(f"  [wait] 已重新加载数据，最新日 {master[-1].date()}", flush=True)

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
    #
    # 2026-11-01 修正（关键）：原来用「所有线的 max(last_trade)」做**全局**判重
    # —— 只要**任何一条**线在该成交日调过仓，其余线全部被跳过。
    # 对多资产线这种「新上线、还没建仓」的线是致命的：
    #   本周它因为袖套 ETF 数据没到而失败 → 其他 6 条线照常成交 →
    #   下一次补跑时全局判重把它们全跳过 → **这条线永远建不上仓**，
    #   而且每次补跑都被跳，连报错都没有。
    # 现在改成**每条线各自判重**：已成交的线跳过，没成交的线照常跑。
    line_last_used: Dict[str, Optional[pd.Timestamp]] = {}
    for s in args.strategies:
        h = load_holdings(s, args.capital)
        _lt = h.get("last_trade") or h.get("last_signal")
        line_last_used[s] = pd.Timestamp(_lt) if _lt else None
    # 用 `<=` 而不是 `<`：同一天重复跑（手动重跑 / cron 补跑）时
    # 成交日会相等 —— 原来只挡「更早」，会**重复调仓**（同一周买两次）。
    _trade_for_check = trade_day if trade_day is not None else rebalance_day
    skip_lines: List[str] = []
    for s in args.strategies:
        _lu = line_last_used[s]
        if _lu is not None and _trade_for_check <= _lu and not args.force:
            skip_lines.append(s)
    if skip_lines:
        _desc_parts: List[str] = []
        for _s in skip_lines:
            _lu = line_last_used.get(_s)
            _desc_parts.append(f"{_s}(上次 {_lu.date()})" if _lu is not None else f"{_s}(无记录)")
        _skipped_desc = ", ".join(_desc_parts)
        print(
            f"跳过（该成交日已调过仓，重复执行会重复下单）：{_skipped_desc}\n"
            f"  成交日 {_trade_for_check.date()}\n"
            f"  如需重跑请加 --force。",
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
    if not args.force:
        print(
            f"信号日（周五）= {rebalance_day.date()}"
            + (f"  成交日（下一交易日）= {trade_day.date()}" if trade_day is not None
               else "  ⚠️ 还没有下一交易日的行情数据 → 本次不成交"),
            flush=True,
        )

    results = []
    line_errors: List[Dict[str, str]] = []
    todo = [s for s in args.strategies if s not in skip_lines]
    if not todo:
        print("（本次未调仓，仅重建 portfolio.json）", flush=True)
    for s in todo:
        spec = STRATEGIES[s]
        print(f"\n{'='*60}\n[{s}] {spec['desc']}", flush=True)
        # 2026-11-01：**单条线失败不许拖垮其他线**。
        #   原来线里 raise SystemExit 会直接退出整个进程 → portfolio.json
        #   停留在旧数据（前端展示的买卖价与实际成交对不上，2026-10-26 实测过），
        #   而且其他线虽然已经成交、写了持仓文件，汇总文件却没更新。
        #   现在：捕获 → 记账 → 继续跑其他线 → 最后仍以非 0 退出码报失败。
        try:
            if spec.get("multi_asset"):
                r = run_multi_asset_line(frames, panel, master, names,
                                         args.capital, rebalance_day,
                                         extras=extras, adj_close=adj_close)
            else:
                r = run_line(s, spec, frames, panel, master, names, args.capital,
                             rebalance_day, extras=extras, adj_close=adj_close)
        except SystemExit as e:
            msg = str(e)
            print(f"❌ [{s}] 本次调仓失败：{msg}", flush=True)
            line_errors.append({"strategy": s, "error": msg})
            continue
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

    # ── 观测口径层（附录 A）：给**每条线**补 6 个指标 ─────────────────
    # 放在这里（所有行合并完成之后）→ 新跑路径与重建路径拿到的是**同一个**
    # 函数算出的口径，不会出现"两条路径各算一套"的漂移。
    for _name in list(merged_lines):
        try:
            merged_lines[_name].update(
                line_observation_metrics(_name, merged_lines[_name], args.capital))
        except Exception as e:  # noqa: BLE001
            # 观测层失败**不能**让整条线失败：宁可少几个字段，也不能不出汇总
            print(f"  ⚠️ 观测指标计算失败 {_name}: {e}", flush=True)

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
            "lines_skipped": skip_lines,
            # 本次失败的线（单条线失败不影响其他线，但必须留痕）
            "errors": line_errors,
        },
        "lines": merged_lines,
    }
    with open(portfolio_path, "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    # 【P1 · security】不再写 frontend/public/。
    # public/ 目录下的静态文件**无需鉴权**即可 GET，等于把付费信号的
    # 持仓/金额/逐笔明细免费公开（前端审计 P1-4 实测：curl 直接拿到 200）。
    # 前端现在统一走受保护端点 GET /api/v1/trading/signal（需 token），
    # 这里只保留 data/paper/ 下的受控副本供后端 / MCP 读取。

    print(f"\n{'='*60}", flush=True)
    # ── 观测口径层（附录 A）：表格必须让人看出"暴露度不同 → 总盈亏不可比" ──
    # 排序用**投入段收益**（消除暴露度差异）；从未交易的线排最后且不参与排名。
    def _fmt_pct(v, width=8):
        return f"{'—':>{width}}" if v is None else f"{v:>{width}.2f}"

    _rows = []
    for _n, _ln in merged_lines.items():
        _rows.append((
            _ln.get("invested_return_pct"),
            _n, _ln.get("value"), _ln.get("cash"), _ln.get("exposure"),
            _ln.get("pnl_pct"), _ln.get("invested_return_pct"),
            _ln.get("as_of"), _ln.get("n_positions"), _ln.get("n_trades"),
            _ln.get("status"),
        ))
    _rows.sort(key=lambda r: (r[0] is None, -(r[0] if r[0] is not None else 0.0)))
    _asofs = {r[7] for r in _rows if r[7]}
    _asof_note = "" if len(_asofs) <= 1 else \
        f"  ⚠️ 各线 as-of 不一致（{sorted(_asofs)}）→ 排名仅供参考，不是同一时点的比较"
    print(f"{'strategy':<20}{'value':>13}{'cash':>12}{'暴露':>7}{'PnL%':>8}"
          f"{'投入段%':>9}{'as_of':>12}{'pos':>5}{'笔':>5}  状态{_asof_note}")
    for (_ir, _n, _val, _cash, _exp, _pnl, _inv, _asof, _npos, _nt, _st) in _rows:
        print(f"{_n:<20}¥{float(_val or 0):>12,.0f}¥{float(_cash or 0):>11,.0f}"
              f"{(float(_exp)*100 if _exp is not None else 0):>6.1f}%"
              f"{_fmt_pct(_pnl)}{_fmt_pct(_inv, 9)}"
              f"{str(_asof or '—'):>12}{int(_npos or 0):>5}{int(_nt or 0):>5}  {_st or '—'}")
    print("  （PnL% 为「—」= 该线从未成交，0.00% 不代表打平；投入段% = 已投资部分本身的收益率）")
    print(f"\n[saved] {os.path.join(PAPER_DIR, 'portfolio.json')}", flush=True)

    if line_errors:
        # 有线失败 → 非 0 退出码（launchd / 定时任务会看到失败），
        # 但 portfolio.json 已经写好了（其他线的成交结果不丢）。
        print(
            f"\n❌ {len(line_errors)} 条线本次失败："
            + ", ".join(e["strategy"] for e in line_errors),
            flush=True,
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
