"""回测引擎的可成交性矩阵（2026-10-26）

## 为什么需要
子代理调研（GitHub 优秀项目对照）指出**当前最大的回测-实盘口径裂缝**：

> 纸面盘（`paper_portfolio.py`）有 5 项可成交性检查（ST/退市、停牌、
> 涨停不可买、跌停不可卖、印花税），**回测引擎一项都没有**。
> → 回测能买涨停、卖跌停 → 回测夏普虚高，纸面盘跑出来的收益永远
>   达不到回测预期。

本模块把 `app/services/trading/execution_rules.py` 里那套规则**下沉到回测引擎**，
让回测与纸面盘、对外信号三处共用同一份实现（RQAlpha `BaseMatcher.match`
的思路：可成交性是**撮合层**的事，不该散落在策略代码里）。

## 与 execution_rules 的分工
- `execution_rules.check_tradability()`：**单标的**判定，适合对外信号和纸面盘
  （有 name / trade_status / amount 等外部信息）
- 本模块：**批量、向量化**地给出一整段回测期的判定矩阵，回测引擎查表用
  （回测里没有 name/trade_status 这些外部字段，只有 OHLCV）

## 规则（与 execution_rules 保持一致）
  1. 停牌：volume == 0                     → 买卖都不可
  2. 涨停：day_ret >= 涨停幅度（按板块）    → 买不进
  3. 一字涨停：high == low 且涨停          → 买不进
  4. 跌停：day_ret <= -涨停幅度            → 卖不出
  5. 一字跌停：high == low 且跌停          → 卖不出
  6. ST：涨停幅度按 5%（需外部 ST 表，否则按板块）
  7. 流动性：买入金额 <= 当日成交额 × 参与率
  8. 成交额门槛：当日成交额 < 门槛 → 不进信号

## 数据来源
回测的 `stock_data` 是 qlib 特征 parquet，列名经过 `_to_qlib_columns` 映射，
本模块接受**已经在内存里的 DataFrame**，列名兼容两种写法
（`$close` / `close`）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.services.trading.execution_rules import (
    DEFAULT_MIN_TURNOVER,
    DEFAULT_PARTICIPATION_RATE,
    limit_pct_array,
)

#: qlib 特征列 → 常见别名（回测里两种都可能出现）
_COL_ALIASES = {
    "close": ("$close", "close"),
    "open": ("$open", "open"),
    "high": ("$high", "high"),
    "low": ("$low", "low"),
    "volume": ("$volume", "volume"),
    "amount": ("$amount", "amount"),
}


def _pick_col(df: pd.DataFrame, key: str) -> Optional[str]:
    for cand in _COL_ALIASES[key]:
        if cand in df.columns:
            return cand
    return None


@dataclass
class TradabilityMatrix:
    """回测期可成交性矩阵。

    索引：行 = 交易日（与回测循环的 trading_dates 对齐），列 = 股票代码。
    """

    dates: List[pd.Timestamp]
    codes: List[str]
    can_buy: np.ndarray  # (n_dates, n_stocks) bool
    can_sell: np.ndarray  # (n_dates, n_stocks) bool
    max_buy_amount: np.ndarray  # (n_dates, n_stocks) float，nan = 不限
    reason_buy: Optional[np.ndarray] = None  # (n_dates, n_stocks) object
    reason_sell: Optional[np.ndarray] = None

    def _row_of(self, date) -> int:
        ts = pd.Timestamp(date)
        # trading_dates 已是交易日；用字典做 O(1) 定位
        return self._date_index.get(ts.normalize(), -1)

    def __post_init__(self) -> None:
        self._date_index: Dict[pd.Timestamp, int] = {
            pd.Timestamp(d).normalize(): i for i, d in enumerate(self.dates)
        }
        self._code_index: Dict[str, int] = {str(c): i for i, c in enumerate(self.codes)}

    def check(self, code: str, date) -> tuple:
        """返回 (can_buy, can_sell, max_buy_amount, reason)"""
        r = self._row_of(date)
        c = self._code_index.get(str(code))
        if r < 0 or c is None:
            return True, True, None, ""
        can_b = bool(self.can_buy[r, c])
        can_s = bool(self.can_sell[r, c])
        mb = self.max_buy_amount[r, c]
        mb = None if (mb is None or not np.isfinite(mb)) else float(mb)
        reason = ""
        if self.reason_buy is not None and not can_b:
            reason = str(self.reason_buy[r, c])
        elif self.reason_sell is not None and not can_s:
            reason = str(self.reason_sell[r, c])
        return can_b, can_s, mb, reason


def load_st_codes(stock_codes: Optional[Sequence[str]] = None) -> Dict[str, bool]:
    """读 ST / 退市标记表 → {code: True}。

    主数据源：`data/stocks/st_names.json`（由 `backend/scripts/fetch_st_names.py`
    用 baostock 的 `query_stock_basic` 抓取，实测 9022 只里 375 只 ST、
    94 只退市整理）。**必须用这份** —— qlib 官方的 instruments/all.txt
    只有 symbol/start/end 三列，没有名称，判不了 ST。
    """
    import json

    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    path = os.path.join(root, "data", "stocks", "st_names.json")
    out: Dict[str, bool] = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                raw = json.load(fh)
            for code, info in raw.items():
                if info.get("is_st") or info.get("is_delisting"):
                    out[str(code)] = True
        except Exception:
            out = {}
    if out:
        return out
    # 兜底：qlib 官方 instruments（若某天补了 name 列也能用）
    path2 = os.path.join(root, "data", "qlib_official_data", "instruments", "all.txt")
    if not os.path.exists(path2):
        return out
    try:
        with open(path2) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                sym, name = parts[0], parts[3]
                if "ST" in name.upper() or name.strip().endswith("退"):
                    out[sym] = True
    except Exception:
        return out
    return out


def build_tradability(
    stock_data: Dict[str, pd.DataFrame],
    trading_dates: Sequence,
    codes: Optional[Sequence[str]] = None,
    participation_rate: float = DEFAULT_PARTICIPATION_RATE,
    min_turnover: float = DEFAULT_MIN_TURNOVER,
    st_codes: Optional[Dict[str, bool]] = None,
    prev_close: Optional[Dict[str, float]] = None,
) -> TradabilityMatrix:
    """构建回测期的可成交性矩阵。

    Args:
        stock_data: {code: DataFrame}（索引为 DatetimeIndex，列含 OHLCV）
        trading_dates: 回测循环使用的交易日列表（顺序必须一致）
        codes: 列顺序；默认用 stock_data 的 key 顺序
        participation_rate: 单日买入额占当日成交额的最大比例
        min_turnover: 当日成交额门槛
        st_codes: {code: bool} ST/退市标记；None 时自动读 instruments
        prev_close: {code: 窗口前一日收盘价}。**必须提供** ——
            回测的 stock_data 是按窗口裁剪的，窗口第一天的涨跌幅
            `close.pct_change()` 是 NaN，早期版本直接 `continue` 跳过 →
            **窗口首日的涨停股没被拦住**（实测 2025-04-03 这天
            24 只买入判定分歧全部集中在这里）。
    """
    if codes is None:
        codes = list(stock_data.keys())
    codes = list(codes)
    n_d, n_s = len(trading_dates), len(codes)
    can_buy = np.ones((n_d, n_s), dtype=bool)
    can_sell = np.ones((n_d, n_s), dtype=bool)
    max_buy = np.full((n_d, n_s), np.nan, dtype=np.float64)
    r_buy: List[List[str]] = [[""] * n_s for _ in range(n_d)]
    r_sell: List[List[str]] = [[""] * n_s for _ in range(n_d)]

    if st_codes is None:
        st_codes = load_st_codes(codes)

    # 涨跌停幅度（按板块 + ST）
    limits = limit_pct_array(codes)
    st_flag = np.array([1.0 if st_codes.get(str(c)) else 0.0 for c in codes])
    limits = np.where(st_flag > 0, 0.05, limits)

    dates = [pd.Timestamp(d).normalize() for d in trading_dates]
    date_pos = {d: i for i, d in enumerate(dates)}

    for j, code in enumerate(codes):
        df = stock_data.get(code)
        if df is None or df.empty:
            continue
        c_col = _pick_col(df, "close")
        if c_col is None:
            continue
        o_col = _pick_col(df, "open")
        h_col = _pick_col(df, "high")
        l_col = _pick_col(df, "low")
        v_col = _pick_col(df, "volume")
        a_col = _pick_col(df, "amount")

        close = pd.to_numeric(df[c_col], errors="coerce")
        ret = close.pct_change()
        open_ = pd.to_numeric(df[o_col], errors="coerce") if o_col else None
        high = pd.to_numeric(df[h_col], errors="coerce") if h_col else None
        low = pd.to_numeric(df[l_col], errors="coerce") if l_col else None
        vol = pd.to_numeric(df[v_col], errors="coerce") if v_col else None

        # 成交额：优先用 amount，否则 close × volume
        #
        # ⚠️ 单位陷阱：qlib 预计算特征里的 `$volume` 单位是**手**（1 手 = 100 股），
        # 实测 `qlib_volume / baostock_volume = 0.0100`（恒定），
        # 而 baostock 的 volume 是股。不乘 100 会让成交额少算 100 倍 →
        # 77% 的股票被误判「流动性不足」而剔除（实测：240 笔 → 28 笔）。
        if a_col:
            amount = pd.to_numeric(df[a_col], errors="coerce")
        elif vol is not None:
            amount = close * vol * 100.0
        else:
            amount = None

        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)
        # 窗口首日的 ret 是 NaN（数据按窗口裁剪，没有前一行）→ 用 prev_close 补
        _ret = ret.to_numpy()
        if prev_close:
            _pc = prev_close.get(str(code))
            if _pc is not None and np.isfinite(_pc) and _pc > 0 and len(_ret):
                if not np.isfinite(_ret[0]):
                    _c0 = close.to_numpy()
                    if np.isfinite(_c0[0]) and _c0[0] > 0:
                        _ret[0] = _c0[0] / _pc - 1.0
        for t, (ts, r) in enumerate(zip(idx, _ret)):
            row = date_pos.get(pd.Timestamp(ts).normalize())
            if row is None:
                continue
            if not np.isfinite(r):
                continue
            # 停牌
            if vol is not None:
                v = vol.iloc[t]
                if np.isfinite(v) and v == 0:
                    can_buy[row, j] = False
                    can_sell[row, j] = False
                    r_buy[row][j] = "停牌（无成交）"
                    r_sell[row][j] = "停牌（无成交）"
                    continue
            lp = limits[j]
            thr = lp - 0.001
            if r >= thr:
                can_buy[row, j] = False
                msg = f"涨停 {r*100:.1f}%（上限 {lp*100:.0f}%）"
                if open_ is not None and high is not None and low is not None:
                    o = open_.iloc[t]
                    h = high.iloc[t]
                    lo = low.iloc[t]
                    if all(np.isfinite(x) for x in (o, h, lo)) and h == lo:
                        msg = "一字涨停，全天封死"
                r_buy[row][j] = msg
            elif r <= -thr:
                can_sell[row, j] = False
                msg = f"跌停 {r*100:.1f}%，卖不出"
                if open_ is not None and high is not None and low is not None:
                    o = open_.iloc[t]
                    h = high.iloc[t]
                    lo = low.iloc[t]
                    if all(np.isfinite(x) for x in (o, h, lo)) and h == lo:
                        msg = "一字跌停，卖不出"
                r_sell[row][j] = msg
            # 流动性
            if amount is not None:
                amt = amount.iloc[t]
                if np.isfinite(amt) and amt > 0:
                    if amt < min_turnover:
                        can_buy[row, j] = False
                        if not r_buy[row][j]:
                            r_buy[row][j] = f"成交额 {amt/1e4:.0f} 万低于门槛"
                    else:
                        max_buy[row, j] = amt * participation_rate

    return TradabilityMatrix(
        dates=dates,
        codes=codes,
        can_buy=can_buy,
        can_sell=can_sell,
        max_buy_amount=max_buy,
        reason_buy=np.array(r_buy, dtype=object),
        reason_sell=np.array(r_sell, dtype=object),
    )
