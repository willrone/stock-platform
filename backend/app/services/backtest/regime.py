"""风格暴露择时开关的判据层（纯函数，可单测，严禁前视）。

口径（与 docs/research/REGIME_SWITCH_DESIGN_2026-11-06.md 对齐）：

- 分层（build_tier_index）：某日 t 属于低盘/高盘层，只能用到 **t-1 为止** 的历史市值。
  市值 = 自由流通市值估算（口径同 cap_estimate：amount / (turn/100)，turn 无效时
  退化为 amount / 0.02），先取截至 t-1 的 250 交易日滚动均值（含 t-1 当日），
  再按 t-1 截面的 q25/q75 分位划分。层内成员用 t 日收盘对 t-1 收盘做等权日收益。
- spread z（size_spread_z）：spread_t = 低盘过去 lookback 个交易日累计收益
  − 高盘同期累计收益；z = (spread_asof − mean / std) 相对 asof 之前（含 asof 当日）
  hist_window 个交易日内的 spread 分布。只用 asof 当日及以前的数据；
  样本不足 min_samples（默认 120）返回 None，上层默认「不防御」并自行计数记录。
- ShadowNav：策略自己记账的影子净值。update(weights, prev_date, asof, closes)
  用只含过去（prev_date、asof 两点）的收盘价算上一期目标篮子从 prev_date 到
  asof 的加权实现收益并滚动更新净值序列；暴露 drawdown 与 rolling_sharpe
  （窗口默认 60 个交易日，年化按 244 交易日）。
- run_switch_machine：把逐日判据布尔序列变成状态序列（进防御需连续 confirm_off
  个判据日、出防御需连续 confirm_on 个「两判据都好」的判据日；达到 max_switches
  后冻结，不再切换，并在返回的 meta 里记录）。

无前视契约：所有滚动窗口都以「当日及以前」的数据结尾；任何以 asof 截断的输入
panel，其在 ≤asof 各点的输出必须与用完整 panel 计算逐位相等（有单测保证）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# cap 缺失（turn 无效）时的退化换手假设（口径沿用设计稿：市值 = 成交额/换手率）
FALLBACK_TURN_PCT = 2.0  # 2% → 市值 = amount / 0.02

TIER_LOW_Q = 0.25
TIER_HIGH_Q = 0.75


def _as_index(dates: Iterable) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    return idx


def _iso(d) -> str:
    if isinstance(d, str):
        return d[:10]
    return pd.Timestamp(d).date().isoformat()


# ──────────────────────────────────────────────────────────────────────
# 分层等权指数
# ──────────────────────────────────────────────────────────────────────


def daily_free_float_cap(
    amount: pd.Series, turn: Optional[pd.Series], close: Optional[pd.Series] = None
) -> pd.Series:
    """单日自由流通市值估算（元）：amount / (turn/100)；turn 无效退化为 amount/0.02。

    与 cap_estimate.py 同源：cap_estimate 取窗口中位数做静态分层，
    这里保留逐日序列供 build_tier_index 做「只用截至 t-1 历史」的动态分层。
    """
    amount = pd.to_numeric(amount, errors="coerce")
    out = pd.Series(np.nan, index=amount.index, dtype="float64")
    ok = np.isfinite(amount) & (amount > 0)
    if turn is not None:
        turn = pd.to_numeric(turn, errors="coerce").reindex(amount.index)
        good = ok & np.isfinite(turn) & (turn > 0)
        out[good] = amount[good] / (turn[good] / 100.0)
    fb = ok & ~np.isfinite(out)
    out[fb] = amount[fb] / (FALLBACK_TURN_PCT / 100.0)
    return out


def build_tier_index(
    close: pd.DataFrame,
    cap_daily: pd.DataFrame,
    hist_window: int = 250,
    min_periods: int = 60,
    low_q: float = TIER_LOW_Q,
    high_q: float = TIER_HIGH_Q,
) -> Dict[str, Any]:
    """低盘/高盘等权日收益指数（无前视）。

    Args:
        close: 收盘价 panel（index=交易日, columns=code）
        cap_daily: 逐日自由流通市值估算 panel（与 close 同 shape 或可 reindex）

    分层规则：t 日的层籍由「截至 t-1 的 hist_window 日市值滚动均值」在 t-1 截面的
    分位决定（low < q25，high > q75）；t 日层内收益 = 成员 close[t]/close[t-1]-1 等权。
    层籍严禁使用 t 日（含）之后的数据。

    Returns:
        {"low_ret": Series, "high_ret": Series, "low_n": Series, "high_n": Series,
         "meta": {...}}  Series index 与 close.index 对齐。
    """
    close = close.sort_index()
    cap_daily = cap_daily.reindex(index=close.index, columns=close.columns)

    # 滚动均值只含截至当日的历史；层籍用 shift(1) → 只含 t-1 及以前
    cap_ma = cap_daily.rolling(hist_window, min_periods=min_periods).mean()
    assign_cap = cap_ma.shift(1)

    n_dates, n_codes = close.shape
    ret = close / close.shift(1) - 1.0

    low_ret = pd.Series(np.nan, index=close.index, dtype="float64")
    high_ret = pd.Series(np.nan, index=close.index, dtype="float64")
    low_n = pd.Series(0, index=close.index, dtype="int64")
    high_n = pd.Series(0, index=close.index, dtype="int64")

    assign_vals = assign_cap.to_numpy(dtype="float64")
    ret_vals = ret.to_numpy(dtype="float64")

    for i in range(n_dates):
        row = assign_vals[i]
        good = np.isfinite(row)
        if good.sum() < 20:
            continue
        q_lo = np.nanquantile(row[good], low_q)
        q_hi = np.nanquantile(row[good], high_q)
        is_low = good & (row < q_lo)
        is_high = good & (row > q_hi)
        r_row = ret_vals[i]
        ok_r = np.isfinite(r_row)
        lm = is_low & ok_r
        hm = is_high & ok_r
        if lm.any():
            low_ret.iloc[i] = float(np.nanmean(r_row[lm]))
            low_n.iloc[i] = int(lm.sum())
        if hm.any():
            high_ret.iloc[i] = float(np.nanmean(r_row[hm]))
            high_n.iloc[i] = int(hm.sum())

    meta = {
        "hist_window": hist_window,
        "min_periods": min_periods,
        "low_q": low_q,
        "high_q": high_q,
        "n_dates": int(n_dates),
        "n_codes": int(n_codes),
        "low_ret_valid_days": int(low_ret.notna().sum()),
        "high_ret_valid_days": int(high_ret.notna().sum()),
        "assign_uses_data_through": "t-1（层籍只用截至 t-1 的市值滚动均值）",
    }
    return {
        "low_ret": low_ret,
        "high_ret": high_ret,
        "low_n": low_n,
        "high_n": high_n,
        "meta": meta,
    }


# ──────────────────────────────────────────────────────────────────────
# spread z 分数（J1）
# ──────────────────────────────────────────────────────────────────────


def size_spread_z(
    low_ret: pd.Series,
    high_ret: pd.Series,
    asof,
    lookback: int = 20,
    hist_window: int = 250,
    min_samples: int = 120,
) -> Optional[float]:
    """asof 日的大小盘相对强弱 z 分数（只用 asof 当日及以前的数据）。

    spread_t = Σ_{k=t-lookback+1..t} (low_ret_k − high_ret_k)
    z = (spread_asof − mean(spread_hist)) / std(spread_hist)
        hist = 截至 asof 的 hist_window 个交易日内的 spread（含 asof）。
    样本不足 min_samples → None（上层默认不防御并记录样本不足天数）。
    """
    asof_ts = pd.Timestamp(asof)
    idx = low_ret.index.union(high_ret.index)
    lo = low_ret.reindex(idx).sort_index()
    hi = high_ret.reindex(idx).sort_index()

    # 截断：asof 之后的数据一概不看
    lo = lo[lo.index <= asof_ts]
    hi = hi[hi.index <= asof_ts]
    if lo.empty:
        return None

    daily = lo - hi
    # 只用 ≤asof 的窗口算 spread：rolling 只回看过去 → 因果
    spread = daily.rolling(lookback, min_periods=lookback).sum()
    spread = spread.dropna()
    if spread.empty:
        return None

    window = spread.tail(hist_window).to_numpy(dtype="float64")
    n = int(np.isfinite(window).sum())
    if n < min_samples:
        return None
    mean = float(np.nanmean(window))
    std = float(np.nanstd(window, ddof=1))
    last = float(window[-1])
    if not np.isfinite(last) or std <= 0 or not np.isfinite(std):
        return None
    return (last - mean) / std


# ──────────────────────────────────────────────────────────────────────
# 影子净值（J2）
# ──────────────────────────────────────────────────────────────────────


class ShadowNav:
    """策略自己记账的影子净值：只按过去的收盘价复现「上一期目标篮子」的实现收益。

    典型用法：每个交易日调用一次 update(prev_basket_weights, prev_date, today, closes)，
    得到逐日影子净值序列；drawdown() 相对历史峰值，rolling_sharpe() 默认 60 日窗口、
    244 交易日年化。update 只读取 closes 里 prev_date 与 asof 两个截面的价格 →
    asof 之后的价格改动不影响本次结果（有单测保证）。
    """

    def __init__(self, window: int = 60, annualize: int = 244):
        self.window = int(window)
        self.annualize = int(annualize)
        self.dates: List[pd.Timestamp] = []
        self.navs: List[float] = []
        self.missing_price_events = 0

    def seed(self, date, value: float = 1.0) -> None:
        self.dates.append(pd.Timestamp(date))
        self.navs.append(float(value))

    @property
    def last_date(self) -> Optional[pd.Timestamp]:
        return self.dates[-1] if self.dates else None

    def update(
        self,
        prev_weights: Optional[Dict[str, float]],
        prev_date,
        asof,
        closes: pd.DataFrame,
    ) -> float:
        """用 prev_date→asof 两点收盘价更新一期影子净值，返回本期收益。

        prev_weights: {code: weight}（等权篮子 1/topk；防御篮子 defense_weight，
        其余视为现金、收益记 0）。缺价（停牌等）的票按收益 0 处理并计数。
        """
        prev_ts, asof_ts = pd.Timestamp(prev_date), pd.Timestamp(asof)
        if self.last_date is not None and asof_ts < self.last_date:
            raise ValueError(
                f"ShadowNav.update 时间倒退: asof={asof_ts} < {self.last_date}"
            )
        if self.last_date is not None and asof_ts == self.last_date:
            raise ValueError(f"ShadowNav.update 重复日期: {asof_ts}")

        step = 0.0
        if prev_weights:
            for code, w in prev_weights.items():
                if code not in closes.columns:
                    self.missing_price_events += 1
                    continue
                try:
                    c0 = closes.at[prev_ts, code]
                    c1 = closes.at[asof_ts, code]
                except KeyError:
                    self.missing_price_events += 1
                    continue
                if not (np.isfinite(c0) and np.isfinite(c1)) or c0 <= 0 or c1 <= 0:
                    self.missing_price_events += 1
                    continue
                step += float(w) * (float(c1) / float(c0) - 1.0)
        new_nav = (self.navs[-1] if self.navs else 1.0) * (1.0 + step)
        self.dates.append(asof_ts)
        self.navs.append(new_nav)
        return step

    def drawdown(self) -> float:
        """当前净值相对历史峰值的回撤（≤0）。"""
        if not self.navs:
            return 0.0
        peak = max(self.navs)
        if peak <= 0:
            return 0.0
        return self.navs[-1] / peak - 1.0

    def rolling_sharpe(self) -> Optional[float]:
        """最近 window 个交易日影子净值日收益的年化夏普（年化 √annualize）。"""
        if len(self.navs) < self.window + 1:
            return None
        navs = np.asarray(self.navs[-(self.window + 1) :], dtype="float64")
        rets = navs[1:] / navs[:-1] - 1.0
        rets = rets[np.isfinite(rets)]
        if len(rets) < max(2, self.window):
            return None
        sd = float(np.std(rets, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return None
        return float(np.mean(rets)) / sd * math.sqrt(self.annualize)

    def series(self) -> List[List[str]]:
        return [[d.date().isoformat(), float(v)] for d, v in zip(self.dates, self.navs)]


# ──────────────────────────────────────────────────────────────────────
# 确认制状态机（进/出防御都要连续确认）
# ──────────────────────────────────────────────────────────────────────


def run_switch_machine(
    dates: List,
    enter_flags: List[bool],
    exit_flags: List[bool],
    confirm_off: int,
    confirm_on: int,
    max_switches: int,
    initial_state: str = "ON",
) -> Dict[str, Any]:
    """把逐判据日的 (enter_flag, exit_flag) 序列转成状态序列。

    规则：
      - enter_flag 连续 ≥ confirm_off → 翻到 OFF（计一次切换）
      - exit_flag 连续 ≥ confirm_on → 翻回 ON（计一次切换）
      - 连续 = 相邻判据日都成立；单日刺穿即断
      - 切换次数达到 max_switches 后冻结在当前状态，不再切换，meta 记录
    enter_flag/exit_flag 由调用方按「J1 且 J2 都坏 / J1 且 J2 都好」预先算好，
    状态机本身不碰判据口径。
    """
    if len(dates) != len(enter_flags) or len(dates) != len(exit_flags):
        raise ValueError("dates/enter_flags/exit_flags 长度必须一致")
    state = initial_state
    enter_streak = 0
    exit_streak = 0
    switch_count = 0
    frozen = False
    steps: List[Dict[str, Any]] = []
    for d, e, x in zip(dates, enter_flags, exit_flags):
        before = state
        switched = False
        frozen_attempt = False
        enter_streak = enter_streak + 1 if e else 0
        exit_streak = exit_streak + 1 if x else 0
        if not frozen:
            if state == "ON" and enter_streak >= confirm_off:
                if switch_count >= max_switches:
                    frozen = True
                    frozen_attempt = True
                else:
                    state = "OFF"
                    switched = True
                    switch_count += 1
                    enter_streak = 0
                    exit_streak = 0
            elif state == "OFF" and exit_streak >= confirm_on:
                if switch_count >= max_switches:
                    frozen = True
                    frozen_attempt = True
                else:
                    state = "ON"
                    switched = True
                    switch_count += 1
                    enter_streak = 0
                    exit_streak = 0
        steps.append(
            {
                "date": _iso(d),
                "state_before": before,
                "state": state,
                "enter_flag": bool(e),
                "exit_flag": bool(x),
                "enter_streak": enter_streak,
                "exit_streak": exit_streak,
                "switched": switched,
                "switch_count": switch_count,
                "frozen": frozen,
                "frozen_attempt": frozen_attempt,
            }
        )
    meta = {
        "switch_count": switch_count,
        "frozen_at_max_switches": frozen,
        "max_switches": max_switches,
        "confirm_off": confirm_off,
        "confirm_on": confirm_on,
        "off_days": int(sum(1 for s in steps if s["state"] == "OFF")),
        "n_decision_days": len(steps),
    }
    return {"steps": steps, "meta": meta}


def random_switch_schedule(
    decision_dates: List,
    target_switches: int,
    seed: int,
    min_block: int = 2,
    max_block: int = 6,
) -> Dict[str, Any]:
    """随机开关对照：切换次数与目标一致、OFF 区间由 seed 决定。

    target_switches 偶数 → 若干个互不重叠的 OFF 区块（每块 进+出 各 1 次）；
    奇数 → 最后一块延伸到窗口末尾（只有进、没有出 → 计 1 次），保证总次数相等。
    """
    import random as _random

    dates = list(decision_dates)
    n = len(dates)
    rng = _random.Random(int(seed))
    n_closed = target_switches // 2
    tail_open = target_switches % 2 == 1

    blocks: List[tuple] = []
    attempts = 0
    while len(blocks) < n_closed and attempts < 400:
        attempts += 1
        length = rng.randint(min_block, max_block)
        start = rng.randrange(0, max(1, n - length - 1))
        end_excl = start + length
        if any(not (end_excl <= s or start >= e) for s, e in blocks):
            continue
        blocks.append((start, end_excl))
    if tail_open and n > 0:
        last_start = max(0, n - rng.randint(min_block, max_block))
        if all(last_start >= e for _, e in blocks):
            blocks.append((last_start, n))
        elif blocks:
            s0, e0 = blocks[-1]
            blocks[-1] = (s0, n)
    blocks.sort()

    off_idx = set()
    for s, e in blocks:
        off_idx.update(range(s, e))
    switches = 2 * sum(1 for s, e in blocks if e < n) + sum(
        1 for s, e in blocks if e == n
    )
    return {
        "off_dates": [dates[i] for i in sorted(off_idx)],
        "blocks": [[_iso(dates[s]), _iso(dates[min(e, n) - 1])] for s, e in blocks],
        "switch_count": switches,
        "target_switches": int(target_switches),
        "seed": int(seed),
    }
