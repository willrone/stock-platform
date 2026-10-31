"""诊断：多资产线 A 腿首笔建仓的钱去哪了。

背景：E2E 测试发现 A 腿（50 万）只买到 58 只、现金闲置 18.8%（本应只有 5% 储备）。
怀疑是两类约束把预算吃掉了：
  ① 整手约束：一手 100 股买不起就整只跳过（`int(budget/p/100)*100 <= 0`）
  ② 流动性上限：`tr.max_buy_amount`（参与率 × 当日成交额）
本脚本把 top-60 每只的「名义预算 / 实际到手 / 约束来源」逐条打出来。

用法：python3 scripts/diag_a_leg_budget.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import daily_signal as ds  # noqa: E402
import paper_portfolio as PP  # noqa: E402
SIGNAL_DAY = pd.Timestamp("2026-10-16")
CAPITAL = 500_000.0
TOPK = 60


def main() -> int:
    codes = ds.list_universe()
    frames, panel, master = ds.load_data(codes)
    names = PP.load_name_map()
    extras = PP.load_market_extras(codes, master) if PP.BAOSTOCK_READY else None

    comp = PP.score_42f(frames, master, panel, explain_date=SIGNAL_DAY)
    if comp is None or comp.empty:
        print("❌ 42f 打分失败")
        return 1
    # ⚠️ comp 是**日期 × 代码**矩阵：取「不晚于信号日的最后一天」那一行，
    #    再按分数排序拿前 60 —— 直接用 comp.iloc[:,0] 会拿到日期列（错的）。
    latest = comp.index[comp.index <= SIGNAL_DAY][-1]
    row = comp.loc[latest].dropna()
    rank = row.sort_values(ascending=False)
    top = list(rank.index[:TOPK])

    # 成交日 = 信号日之后第一个交易日（与 run_line 同口径）
    trade_day = master[master > SIGNAL_DAY][0]
    close_mat = ds.to_mat(frames, master, "close")
    close = close_mat.loc[trade_day]
    vol = ds.to_mat(frames, master, "volume").loc[trade_day]
    open_price = ds.to_mat(frames, master, "open").loc[trade_day]
    high = ds.to_mat(frames, master, "high").loc[trade_day]
    low = ds.to_mat(frames, master, "low").loc[trade_day]
    close_prev = close_mat.shift(1).loc[trade_day]

    target_weight = (1.0 - PP.CASH_RESERVE) / TOPK
    budget = CAPITAL * target_weight

    print(f"信号日 {SIGNAL_DAY.date()} → 成交日 {trade_day.date()}")
    print(f"名义预算/只 = {budget:,.2f}  ({CAPITAL:,.0f} × (1-{PP.CASH_RESERVE}) / {TOPK})")
    print()
    print(f"{'代码':<10}{'收盘价':>9}{'一手成本':>11}{'流动性上限':>12}"
          f"{'到手股数':>10}{'买入额':>11}  备注")
    n_skip_lot = 0
    n_liq = 0
    n_blocked = 0
    total_spent = 0.0
    cash = CAPITAL
    for c in top:
        p = float(close.get(c, np.nan))
        tr = PP._tradability(c, trade_day, names, extras, close=close,
                             open_price=open_price, high=high, low=low,
                             volume=vol, close_prev=close_prev)
        b = budget
        note = ""
        if not tr.can_buy:
            note = f"不可买：{tr.blocked_reason or tr.reason}"
            n_blocked += 1
            shares = 0
        else:
            if tr.max_buy_amount is not None and tr.max_buy_amount < b:
                b = tr.max_buy_amount
                note = "流动性上限"
                n_liq += 1
            shares = int(b / p / PP.BOARD_LOT) * PP.BOARD_LOT if np.isfinite(p) and p > 0 else 0
            lot_cost = p * PP.BOARD_LOT
            if shares <= 0:
                note = f"买不起一手（需 {lot_cost:,.0f} > 预算 {b:,.0f}）"
                n_skip_lot += 1
        spent = shares * p if (np.isfinite(p) and shares > 0) else 0.0
        total_spent += spent
        cash -= spent
        print(f"{c:<10}{p:>9.2f}{p*PP.BOARD_LOT:>11,.0f}"
              f"{(tr.max_buy_amount if tr.max_buy_amount is not None else float('nan')):>12,.0f}"
              f"{shares:>10d}{spent:>11,.0f}  {note}")

    print()
    print(f"不可买（涨跌停/ST/停牌）：{n_blocked} 只")
    print(f"被流动性上限压缩：{n_liq} 只")
    print(f"整手买不起被跳过：{n_skip_lot} 只")
    print(f"实际投入 {total_spent:,.0f} / {CAPITAL:,.0f} = {total_spent/CAPITAL:.1%}"
          f"  → 现金闲置 {(CAPITAL-total_spent)/CAPITAL:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
