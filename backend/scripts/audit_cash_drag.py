"""量化「回测里未记账的现金仓位」对夏普/回撤的影响（台账 §12.66，产数脚本）。

思路（可自证）：
  B_t        = Σ 持仓数量 × 当日收盘价            （纯市场市值，用本地不复权 close）
  flow_in_t  = 当日「从现金搬进股票」的净钱（买入为正、卖出为负；成交价含千1滑点，已还原成收盘价口径）
  marketPnL_t = B_t − B_{t−1} − flow_in_t          （持仓部分的纯市场盈亏，不含成本）
                 ⚠️ 符号别写反：写成「卖出 − 买入」会把买入日双倍计入，残差会虚高到 70%+（2026-11-06 踩过）。
  r_pure_t    = marketPnL_t / B_{t−1}             （满仓、无成本口径的日收益）
对照：
  r_act       = equity_curve 的日收益              （有现金、有成本）
  r_act_noc   = portfolio_value_without_cost 的日收益（有现金、无成本）
→ Sharpe(r_act_noc) vs Sharpe(r_pure) 之差 = **纯现金缓冲造成的夏普抬升**（不含成本口径混淆）。

恒等式校验：Σ marketPnL ≈ (final_value − initial_cash) + total_cost
（引擎口径：终值 − 初始 = 市场盈亏 − 成本；本仓库行情不复权，两边都不含分红）。
"""
import os
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "data/strategy_baseline")
# 估值必须用回测真正读的库：stock_data（全历史，不复权）。
# baostock_daily 只是带 is_st/trade_status/amount 的镜像，且**北交所票只有 2026-05-11 起的 122 行** →
# 用它算市值会把 2023 年持有的 BJ 票算成 0，市场盈亏全错（2026-11-06 踩过）。
PQ = os.path.join(ROOT, "data/parquet/stock_data")
PQ_FALLBACK = os.path.join(ROOT, "data/parquet/baostock_daily")

# 只有带 --record-positions 跑出来的产物才有逐日持仓；没有持仓就只能算暴露（现金占比），算不了满仓口径。
LINES = [
    ("25f 无否决", "ablation_2026-11-06_base_25f_pos_lag1.json"),
    ("25f+中性", "ablation_2026-11-06_base_25f_neu_pos_lag1.json"),
    ("25f+否决", "ablation_2026-11-06_honest_25f_veto_pos_lag1.json"),
    ("25f+中性+否决", "ablation_2026-11-06_honest_25f_neu_veto_pos_lag1.json"),
    ("否决闸门¥2000万", "ablation_2026-11-06_honest_25f_veto_liq20m_lag1.json"),
    ("否决闸门¥1亿", "ablation_2026-11-06_honest_25f_veto_liq100m_lag1.json"),
]

_cache: dict[str, pd.Series] = {}


def closes(code: str) -> pd.Series:
    if code not in _cache:
        f = os.path.join(PQ, code.replace(".", "_") + ".parquet")
        if not os.path.exists(f):
            f = os.path.join(PQ_FALLBACK, code.replace(".", "_") + ".parquet")
        if not os.path.exists(f):
            _cache[code] = pd.Series(dtype=float)
        else:
            cols = ["date", "close"]
            df = pd.read_parquet(f, columns=cols)
            s = pd.Series(df["close"].astype(float).values,
                          index=pd.to_datetime(df["date"]).dt.normalize())
            s = s[~s.index.duplicated(keep="last")].sort_index()
            # 停牌日 parquet 里没有行：持仓还在，市值必须沿用最后一个收盘价（前向填充），
            # 否则那只票在停牌日被算成 0，市值凭空跳水，拆出来的市场盈亏全是假的。
            _cache[code] = s.resample("D").last().ffill()
    return _cache[code]


def stats(r: pd.Series) -> tuple[float, float, float]:
    """日收益序列 -> (年化夏普, 年化收益%, 最大回撤%)。年化按 252 交易日。"""
    r = r.dropna()
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else float("nan")
    nav = (1 + r).cumprod()
    yrs = len(r) / 252
    ann = float((nav.iloc[-1] ** (1 / yrs) - 1) * 100) if yrs > 0 else float("nan")
    dd = float(((nav / nav.cummax()) - 1).min() * 100)
    return sharpe, ann, dd


def analyze(label: str, path: str) -> dict:
    d = json.load(open(os.path.join(DIR, path)))
    if not any(r.get("positions") for r in d["portfolio_history"]):
        # 没有逐日持仓（跑的时候没开 --record-positions）→ 只能给暴露，不能给满仓口径
        ph = d["portfolio_history"]
        idx = pd.DatetimeIndex([pd.Timestamp(r["date"]).normalize() for r in ph])
        pv = pd.Series([float(r["portfolio_value"]) for r in ph], index=idx)
        cash = pd.Series([float(r["cash"]) for r in ph], index=idx)
        s, a, dd = stats(pv.pct_change())
        return dict(label=label, codes=0, miss=0, expo_mean=float((1 - cash / pv).mean() * 100),
                    expo_min=float((1 - cash / pv).min() * 100), B_eng_mean=float("nan"),
                    B_mine_mean=float("nan"), valuation_gap=float("nan"), exp_pnl=float("nan"),
                    got_pnl=float("nan"), resid_pct=float("nan"), s_act=s, a_act=a, dd_act=dd,
                    s_noc=float("nan"), a_noc=float("nan"), dd_noc=float("nan"),
                    s_pure=float("nan"), a_pure=float("nan"), dd_pure=float("nan"),
                    d_sharpe=float("nan"), d_dd=float("nan"), trades=int(d["total_trades"]),
                    days=len(idx), no_positions=True)
    ph = d["portfolio_history"]
    dates = [pd.Timestamp(r["date"]).normalize() for r in ph]
    idx = pd.DatetimeIndex(dates)

    # 引擎自己的市值与现金
    pv = pd.Series([float(r["portfolio_value"]) for r in ph], index=idx)
    pv_noc = pd.Series([float(r["portfolio_value_without_cost"]) for r in ph], index=idx)
    cash = pd.Series([float(r["cash"]) for r in ph], index=idx)
    B_eng = pd.Series([sum(float(v["market_value"]) for v in (r.get("positions") or {}).values())
                       for r in ph], index=idx)

    # 我用本地收盘价重算市值（顺带查引擎估值是否就是原始收盘）
    qty_map = [{c: float(v["quantity"]) for c, v in (r.get("positions") or {}).items() if float(v["quantity"]) > 0}
               for r in ph]
    codes = sorted({c for m in qty_map for c in m})
    close = {c: closes(c) for c in codes}
    B_mine, mism = [], 0
    for dt, q in zip(idx, qty_map):
        tot = 0.0
        for c, n in q.items():
            s = close[c]
            if dt in s.index and not np.isnan(float(s.loc[dt])):
                tot += n * float(s.loc[dt])
            else:
                mism += 1
        B_mine.append(tot)
    B_mine = pd.Series(B_mine, index=idx)

    # 成交流水（纯市场口径：剔除佣金与滑点）
    tr = pd.DataFrame(d["trade_history"])
    tr["dt"] = pd.to_datetime(tr["timestamp"]).dt.normalize()
    tr["notional"] = tr["quantity"].astype(float) * tr["price"].astype(float)
    tr["slip"] = tr["slippage_cost"].astype(float)
    buy = tr[tr["action"].str.upper() == "BUY"]
    sell = tr[tr["action"].str.upper() == "SELL"]
    # 成交价 = 收盘×1.001(买) / ×0.999(卖)，slippage_cost 是已含在成交价里的记录值（不是再扣一遍）。
    # 这里剔掉滑点，还原成「按收盘价的纯市场现金流」。
    f_buy = (buy.groupby("dt")["notional"].sum() - buy.groupby("dt")["slip"].sum()).reindex(idx).fillna(0.0)
    f_sell = (sell.groupby("dt")["notional"].sum() + sell.groupby("dt")["slip"].sum()).reindex(idx).fillna(0.0)
    # 符号：flow_in = 从现金搬进股票的净钱（买入为正、卖出为负）。
    #   B_t = B_{t-1} + marketPnL_t + flow_in_t  →  marketPnL_t = B_t − B_{t-1} − flow_in_t
    flow = f_buy - f_sell

    # 纯市场盈亏
    B = B_mine.where(B_mine > 0, np.nan)
    pnl = B - B.shift(1) - flow
    pnl = pnl.where(B.shift(1).notna(), np.nan)
    r_pure = pnl / B.shift(1)

    r_act = pv.pct_change()
    r_noc = pv_noc.pct_change()

    init = float(d["initial_cash"]); fin = float(d["final_value"])
    cost = float(d["cost_statistics"]["total_cost"])
    exp_pnl = (fin - init) + cost
    got_pnl = float(pnl.sum(skipna=True))

    s_act, a_act, dd_act = stats(r_act)
    s_noc, a_noc, dd_noc = stats(r_noc)
    s_pure, a_pure, dd_pure = stats(r_pure.dropna())

    expo = (1 - cash / pv)
    return dict(
        label=label, codes=len(codes), miss=mism,
        expo_mean=float(expo.mean() * 100), expo_min=float(expo.min() * 100),
        B_eng_mean=float(B_eng.mean()), B_mine_mean=float(B_mine.mean()),
        valuation_gap=float(abs(B_eng.mean() - B_mine.mean()) / B_eng.mean() * 100),
        exp_pnl=exp_pnl, got_pnl=got_pnl, resid_pct=abs(got_pnl - exp_pnl) / exp_pnl * 100,
        s_act=s_act, a_act=a_act, dd_act=dd_act,
        s_noc=s_noc, a_noc=a_noc, dd_noc=dd_noc,
        s_pure=s_pure, a_pure=a_pure, dd_pure=dd_pure,
        d_sharpe=s_pure - s_noc, d_dd=dd_pure - dd_act,
        trades=int(d["total_trades"]), days=len(idx),
    )


def main() -> None:
    bad = []
    rows = []
    for label, path in LINES:
        r = analyze(label, path)
        rows.append(r)
        if r.get("no_positions"):
            print(f"[{label}] 无逐日持仓（未开 --record-positions）→ 只报暴露 {r['expo_mean']:.1f}% 最低 {r['expo_min']:.1f}%", flush=True)
            continue
        print(f"[{label}] 校验：期望Σ市场盈亏 {r['exp_pnl']:,.0f} vs 拆解 {r['got_pnl']:,.0f} "
              f"残差 {r['resid_pct']:.2f}% | 估值差(引擎市值 vs 我算) {r['valuation_gap']:.3f}% "
              f"| 缺收盘价 {r['miss']} 处", flush=True)
        # 硬闸门：拆解不精确就不许产出数字
        if r["resid_pct"] > 0.5 or r["valuation_gap"] > 0.5 or r["miss"] > 0:
            bad.append((label, round(r["resid_pct"], 2), round(r["valuation_gap"], 3), r["miss"]))
    df = pd.DataFrame(rows)
    print()
    print(f"{'线':16s} {'平均暴露':>8s} {'最低':>6s} | {'实际夏普':>8s} {'无成本夏普':>9s} {'满仓夏普':>8s} "
          f"{'现金抬升':>8s} | {'实际年化':>8s} {'满仓年化':>8s} | {'实际回撤':>8s} {'满仓回撤':>8s} {'回撤变差':>8s}")
    for r in rows:
        print(f"{r['label']:16s} {r['expo_mean']:7.1f}% {r['expo_min']:5.1f}% | "
              f"{r['s_act']:8.4f} {r['s_noc']:9.4f} {r['s_pure']:8.4f} {r['d_sharpe']:+8.4f} | "
              f"{r['a_act']:7.1f}% {r['a_pure']:7.1f}% | {r['dd_act']:7.1f}% {r['dd_pure']:7.1f}% {r['d_dd']:+7.1f}pp")
    if bad:
        raise SystemExit(f"拆解未通过校验，数字不可用：{bad}")
    out = os.path.join(DIR, "cash_drag_2026-11-06.json")
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=2)
    print("\nsaved ->", out)


if __name__ == "__main__":
    sys.exit(main())
