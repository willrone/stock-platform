#!/usr/bin/env python3
"""ETF 核心虚拟账户（¥90,000）—— 把 §12.70 拍定的可下单清单变成一条持续盯市的线。

## 为什么存在
2026-11-06 判定：本金 <10 万时，A 股因子选股线（低盘 TopK20）在真实可执行性下回撤 −31%，
风格择时开关判死（§12.76）。预拍决定：**真钱主仓 = ETF 线**，因子腿只跑纸面。
本脚本就是这条主仓的**虚拟账**：按真实"手"（100 份/手）建仓、按真实佣金规则记账、
每日盯市，让你能看到"如果今天按这张单子下单，账户会长什么样"。

## 账本规则（与 §12.70 回测口径一致，不得随意改）
- 本金 90,000 元；目标权重 红利低波50(515450) 50% + 城投债ETF(511220) 50%
- 整数手：1 手 = 100 份，只买整数手，余额留现金
- 佣金 万 3、单笔最低 5 元；ETF 免印花税
- **未计滑点/冲击成本**（3 年 8 笔、单边 0.05% 的话约 0.04% 总拖累，报告里会披露）
- 再平衡：**每年第一个交易日**一次（与回测口径一致）；不做漂移触发的临时再平衡，
  漂移只在报告里披露，避免"看盘手痒"改规则
- 盯市：用当日收盘价（前复权，腾讯源 → akshare 兜底），数据来自 data/etf/parquet/

## 用法
    cd backend
    # 建仓（按最新收盘换算整数手）
    env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 scripts/etf_virtual_account.py --open
    # 盯市（默认用数据里最新一天；可 --date 指定）
    env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 scripts/etf_virtual_account.py --mark
    # 出报告（含真实下单清单）
    env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 scripts/etf_virtual_account.py --report
    # 三合一（cron 用这个）
    ... scripts/etf_virtual_account.py --auto     # = --mark --report

## 产出
- `data/paper/etf_core_90k.json`      账本（持仓/成交/逐日净值/规则）
- `data/paper/etf_core_90k_nav.csv`   净值序列（画图/对比用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../backend
REPO = os.path.dirname(ROOT)
ETF_DIR = os.path.join(REPO, "data", "etf", "parquet")
PAPER_DIR = os.path.join(REPO, "data", "paper")
STATE_PATH = os.path.join(PAPER_DIR, "etf_core_90k.json")
NAV_CSV = os.path.join(PAPER_DIR, "etf_core_90k_nav.csv")

LINE_ID = "etf_core_90k"
CAPITAL = 90_000.0
TARGET_WEIGHTS = {"515450": 0.50, "511220": 0.50}
NAMES = {"515450": "红利低波50", "511220": "城投债ETF", "511010": "国债ETF", "510300": "沪深300"}
LOT = 100
COMMISSION_RATE, MIN_COMMISSION = 0.0003, 5.0
# §12.70 整数手回测参照（同本金、同佣金规则、2023-05-12→2026-05-08）
REFERENCE = {"annualized": 0.0789, "sharpe": 0.9662, "max_dd": -0.0525, "window": "2023-05-12~2026-05-08"}
BENCH_CODE = "510300"   # 沪深300 ETF，作对照基准


def commission(amount: float) -> float:
    """券商常见规则：万 3，单笔最低 5 元。ETF 免印花税。"""
    return max(amount * COMMISSION_RATE, MIN_COMMISSION)


def load_close(code: str, upto: str | None = None) -> pd.Series:
    p = os.path.join(ETF_DIR, f"{code}.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"[{LINE_ID}] 缺 ETF 数据 {code} → 先跑 scripts/etf_data_pipeline.py --codes {code} --incremental")
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    s = pd.Series(pd.to_numeric(d["close"], errors="coerce").values, index=d["date"], dtype=float)
    s = s[s.notna() & (s > 0)]
    if upto:
        s = s[s.index <= upto]
    if s.empty:
        raise SystemExit(f"[{LINE_ID}] {code} 在 {upto} 之前无可用收盘价")
    return s


def latest_data_date(codes: list[str]) -> str:
    return min(max(load_close(c).index) for c in codes)   # 取所有腿都有的最新一天


def blank_state() -> dict:
    return {
        "line": LINE_ID,
        "capital": CAPITAL,
        "status": "not_opened",
        "opened_date": None,
        "rules": {
            "target_weights": TARGET_WEIGHTS,
            "lot": LOT,
            "commission_rate": COMMISSION_RATE,
            "min_commission": MIN_COMMISSION,
            "stamp_tax": 0.0,
            "slippage_modelled": 0.0,
            "rebalance": "每年第一个交易日一次（与 §12.70 回测口径一致）",
        },
        "holdings": {},          # code -> {name, shares, entry_price, entry_amount, last_price}
        "cash": CAPITAL,
        "fees_paid": 0.0,
        "trades": [],
        "nav_daily": [],         # {date, nav, cash, position_value, prices, weights}
        "last_rebalance_date": None,
        "reference_backtest": REFERENCE,
    }


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return blank_state()
    with open(STATE_PATH) as fh:
        return json.load(fh)


def save_state(st: dict) -> None:
    os.makedirs(PAPER_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(st, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)
    navs = st.get("nav_daily") or []
    if navs:
        with open(NAV_CSV, "w") as fh:
            fh.write("date,nav,cash,position_value,nav_vs_capital\n")
            for r in navs:
                fh.write(f"{r['date']},{r['nav']:.2f},{r['cash']:.2f},{r['position_value']:.2f},{r['nav']/st['capital']-1:+.6f}\n")


def target_lots(total: float, prices: dict[str, float]) -> dict[str, int]:
    """把目标权重换算成整数手（向下取整，余额进现金）。"""
    out = {}
    for code, w in TARGET_WEIGHTS.items():
        unit = prices[code] * LOT
        out[code] = int((total * w) // unit)
    return out


def record_trade(st: dict, d: str, code: str, delta_shares: int, price: float, reason: str) -> None:
    amount = abs(delta_shares) * price
    fee = commission(amount) if delta_shares else 0.0
    st["trades"].append({
        "date": d, "code": code, "name": NAMES.get(code, code),
        "side": "buy" if delta_shares > 0 else ("sell" if delta_shares < 0 else "noop"),
        "shares": abs(delta_shares), "lots": abs(delta_shares) // LOT, "price": round(price, 4),
        "amount": round(amount, 2), "commission": round(fee, 2), "reason": reason,
    })
    st["fees_paid"] = round(st.get("fees_paid", 0.0) + fee, 2)


def apply_targets(st: dict, d: str, prices: dict[str, float], reason: str) -> None:
    """按目标权重换到整数手：先卖后买，现金不够就少买（如实记录未成交量）。"""
    total = st["cash"] + sum(h["shares"] * prices[c] for c, h in st["holdings"].items())
    want = target_lots(total, prices)
    # 先卖（腾现金）
    for code, h in st["holdings"].items():
        want_shares = want.get(code, 0) * LOT
        delta = want_shares - h["shares"]
        if delta < 0:
            record_trade(st, d, code, delta, prices[code], reason)
            h["shares"] = want_shares
            st["cash"] += abs(delta) * prices[code] - commission(abs(delta) * prices[code])
    # 再买
    for code, lots in want.items():
        want_shares = lots * LOT
        h = st["holdings"].setdefault(code, {"name": NAMES.get(code, code), "shares": 0, "entry_price": None, "entry_amount": 0.0, "last_price": None})
        delta = want_shares - h["shares"]
        if delta <= 0:
            continue
        cost = delta * prices[code]
        fee = commission(cost)
        affordable = st["cash"] - MIN_COMMISSION > 0
        if cost + fee > st["cash"]:
            affordable_lots = int((st["cash"] - MIN_COMMISSION) // (prices[code] * LOT))
            if affordable_lots <= 0:
                st["trades"].append({"date": d, "code": code, "side": "unfilled", "shares": delta, "price": round(prices[code], 4),
                                     "amount": round(cost, 2), "commission": 0.0, "reason": reason + "｜现金不足，未成交"})
                continue
            delta = affordable_lots * LOT
            cost = delta * prices[code]
            fee = commission(cost)
            record_trade(st, d, code, delta, prices[code], reason + f"｜现金不足，只买 {affordable_lots} 手")
        else:
            record_trade(st, d, code, delta, prices[code], reason)
        if affordable or delta:
            st["cash"] -= cost + fee
            h["shares"] += delta
            if h["shares"]:
                h["entry_price"] = prices[code] if h["entry_price"] is None else (
                    (h["entry_price"] * (h["shares"] - delta) + prices[code] * delta) / h["shares"])
                h["entry_amount"] = round(h["shares"] * h["entry_price"], 2)
    st["holdings"] = {c: h for c, h in st["holdings"].items() if h["shares"] > 0}
    st["last_rebalance_date"] = d


def position_value(st: dict, prices: dict[str, float]) -> float:
    return sum(h["shares"] * prices.get(c, h.get("last_price") or 0.0) for c, h in st["holdings"].items())


def is_first_trading_day_of_year(nav_dates: list[str], d: str) -> bool:
    prev = [x for x in nav_dates if x < d]
    return bool(prev) and prev[-1][:4] != d[:4]


def cmd_open(st: dict, when: str | None) -> dict:
    if st["status"] == "opened":
        print(f"[{LINE_ID}] 已建仓（{st['opened_date']}），不重复建仓。要看现状用 --report")
        return st
    codes = list(TARGET_WEIGHTS)
    d = when or latest_data_date(codes)
    prices = {c: float(load_close(c, d).iloc[-1]) for c in codes}
    st["status"], st["opened_date"] = "opened", d
    apply_targets(st, d, prices, "建仓")
    pv = position_value(st, prices)
    st["nav_daily"].append({"date": d, "nav": round(st["cash"] + pv, 2), "cash": round(st["cash"], 2),
                            "position_value": round(pv, 2), "prices": {c: round(p, 4) for c, p in prices.items()},
                            "weights": {c: round(h["shares"] * prices[c] / (pv + st["cash"]), 4) for c, h in st["holdings"].items()}})
    save_state(st)
    print(f"[{LINE_ID}] 建仓完成 @{d}｜投入 {pv:,.2f} 元｜现金 {st['cash']:,.2f} 元｜佣金 {st['fees_paid']:.2f} 元")
    for t in st["trades"]:
        print(f"  {t['side']:5s} {t['name']:10s} {t['code']} {t['lots']} 手 @ {t['price']} = {t['amount']:,.2f} 元（佣金 {t['commission']:.2f}）")
    return st


def cmd_mark(st: dict, when: str | None) -> dict:
    if st["status"] != "opened":
        raise SystemExit(f"[{LINE_ID}] 未建仓，先跑 --open")
    codes = sorted(set(list(TARGET_WEIGHTS) + list(st["holdings"])))
    d = when or latest_data_date(codes)
    dates = [r["date"] for r in st["nav_daily"]]
    if d in dates:
        print(f"[{LINE_ID}] {d} 已盯市，跳过")
        return st
    last = max(dates) if dates else None
    if last and d < last:
        raise SystemExit(f"[{LINE_ID}] --date {d} 早于账本最后一天 {last}（不回填历史，避免乱序）")
    prices = {c: float(load_close(c, d).iloc[-1]) for c in codes}
    rebal = ""
    if is_first_trading_day_of_year(dates, d):
        apply_targets(st, d, prices, "年度再平衡")
        prices = {c: float(load_close(c, d).iloc[-1]) for c in codes}
        rebal = "｜已执行年度再平衡"
    pv = position_value(st, prices)
    nav = st["cash"] + pv
    st["nav_daily"].append({"date": d, "nav": round(nav, 2), "cash": round(st["cash"], 2), "position_value": round(pv, 2),
                            "prices": {c: round(p, 4) for c, p in prices.items()},
                            "weights": {c: round(h["shares"] * prices[c] / nav, 4) for c, h in st["holdings"].items()}})
    save_state(st)
    n = len(st["nav_daily"])
    print(f"[{LINE_ID}] 盯市 {d}{rebal}｜净值 {nav:,.2f} 元（{(nav/st['capital']-1)*100:+.2f}%）｜现金 {st['cash']:,.2f}{'' if n>1 else ''}")
    return st


def cmd_report(st: dict, asof: str | None) -> None:
    if st["status"] != "opened":
        raise SystemExit(f"[{LINE_ID}] 未建仓，先跑 --open")
    navs = st["nav_daily"]
    last = navs[-1]
    d = last["date"]
    prices = {c: float(load_close(c, d).iloc[-1]) for c in sorted(set(list(TARGET_WEIGHTS) + list(st["holdings"])))}
    nav, cash, pv = last["nav"], last["cash"], last["position_value"]
    ret = nav / st["capital"] - 1
    n = len(navs)
    years = max((pd.Timestamp(d) - pd.Timestamp(st["opened_date"])).days, 1) / 365.25
    ann = (nav / st["capital"]) ** (1.0 / years) - 1
    peak, dd = nav, 0.0
    for r in navs:
        peak = max(peak, r["nav"])
        dd = min(dd, r["nav"] / peak - 1)
    print(f"\n=== ETF 核心虚拟账户 {LINE_ID}（本金 {st['capital']:,.0f} 元｜状态 {st['status']}）===")
    print(f"建仓日 {st['opened_date']}｜最新盯市 {d}（{n} 个净值点，已跟踪 {years*365.25:.0f} 天）")
    ann_txt = f"{ann*100:+.2f}%" if years >= 30 / 365.25 else "（跟踪不足 30 天，年化不外推）"
    print(f"净值 {nav:,.2f} 元｜累计 {ret*100:+.2f}%｜年化 {ann_txt}｜最大回撤 {dd*100:+.2f}%｜已付佣金 {st['fees_paid']:.2f} 元｜现金 {cash:,.2f} 元")
    print(f"\n持仓（按 {d} 收盘）：")
    print(f"  {'标的':14s} {'代码':8s} {'手数':>5s} {'成本价':>8s} {'现价':>8s} {'市值':>11s} {'实际权重':>9s} {'目标':>7s} {'浮动盈亏':>10s}")
    for c, h in sorted(st["holdings"].items()):
        mv = h["shares"] * prices[c]
        pl = (prices[c] - (h["entry_price"] or prices[c])) * h["shares"]
        print(f"  {NAMES.get(c,c):14s} {c:8s} {h['shares']//LOT:5d} {h['entry_price']:8.3f} {prices[c]:8.3f} {mv:11,.2f} {mv/nav*100:8.1f}% {TARGET_WEIGHTS[c]*100:6.0f}% {pl:+10,.2f}")
    drift = {c: last["weights"].get(c, 0.0) - TARGET_WEIGHTS[c] for c in TARGET_WEIGHTS}
    print(f"  权重漂移：{', '.join(f'{NAMES.get(c,c)} {drift[c]*100:+.1f}pp' for c in TARGET_WEIGHTS)}"
          f"｜下次年度再平衡：下一个自然年首个交易日（上次 {(st.get('last_rebalance_date') or st['opened_date'])}）")
    # 对照基准
    try:
        b = load_close(BENCH_CODE)
        b0 = b[b.index >= st["opened_date"]].iloc[0]
        br = float(b.iloc[-1]) / b0 - 1
        print(f"对照：沪深300ETF(510300) 同期 {br*100:+.2f}%（自 {st['opened_date']}）")
    except SystemExit:
        pass
    ref = st["reference_backtest"]
    print(f"参照（§12.70 整数手回测 {ref['window']}）：年化 {ref['annualized']*100:+.2f}% 夏普 {ref['sharpe']:.4f} 回撤 {ref['max_dd']*100:+.2f}%")
    print("口径披露：整数手 + 万3佣金(最低5元) + 免印花税；**未计滑点/冲击**（本线交易极少，单边 0.05% 对应总拖累 <0.05%）；ETF 价源为前复权收盘。")
    # 真实下单清单（若要真钱执行，照这个下）
    tl = target_lots(nav, prices)
    print(f"\n=== 若按当前净值真钱下单（{d} 收盘基准，年度再平衡口径）===")
    for c in sorted(TARGET_WEIGHTS):
        cur = st["holdings"].get(c, {}).get("shares", 0) // LOT
        delta = tl[c] - cur
        amt = abs(delta) * prices[c]
        act = ("买" if delta > 0 else ("卖" if delta < 0 else "不动"))
        print(f"  {NAMES.get(c,c):14s} {c} 目标 {tl[c]} 手 / 现有 {cur} 手 → {act} {abs(delta)} 手"
              + (f" ≈ {amt:,.2f} 元（佣金 {commission(amt):.2f}）" if delta else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="ETF 核心虚拟账户（¥90,000 整数手盯市）")
    ap.add_argument("--open", action="store_true", help="建仓（按最新收盘换算整数手）")
    ap.add_argument("--mark", action="store_true", help="盯市一天（含年度再平衡）")
    ap.add_argument("--report", action="store_true", help="出报告 + 真钱下单清单")
    ap.add_argument("--auto", action="store_true", help="= --mark --report（cron 用）")
    ap.add_argument("--date", default=None, help="指定盯市日期 YYYY-MM-DD（不得早于已有最后一天）")
    a = ap.parse_args()
    if not (a.open or a.mark or a.report or a.auto):
        ap.error("至少给一个动作：--open / --mark / --report / --auto")
    st = load_state()
    if a.open:
        st = cmd_open(st, a.date)
    if a.mark or a.auto:
        st = cmd_mark(st, a.date)
    if a.report or a.auto:
        cmd_report(st, a.date)


if __name__ == "__main__":
    main()
