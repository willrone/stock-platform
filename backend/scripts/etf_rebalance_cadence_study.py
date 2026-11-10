#!/usr/bin/env python3
"""ETF 主仓的**再平衡节奏**实测：几天动一次到底值不值（¥90,000 真钱口径）。

背景：§12.75 已在 A 股因子线上测过节奏（daily 0.8288 / weekly 1.1928 / monthly 0.7473，
daily 三年 3408 笔、成本 35.6% 本金）。但真钱主仓（515450/511220 五五开）从没测过节奏，
只有 §12.77/§12.80 的「每年一次」。这一节把主仓的节奏补齐，并把「每周看一眼、按阈值才动手」
这个折中方案量化。

口径：
- 价源 data/etf/parquet（前复权，收益含分红，§12.80 ⓪ 审计确认）。
- 整数手（100 份），佣金万 3、单笔最低 5 元，ETF 免印花税，未计滑点（本线单边流动性极好）。
- 目标权重 50/50；「再平衡」= 把两只腿拉回 50/50（整数手，向下取整，余额留现金）。
- 阈值口径（band）= 每 cadence 日检查，只有 |权重偏离| > band 才交易。

用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/etf_rebalance_cadence_study.py
产出：../data/etf/rebalance_cadence_study.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ETF_DIR = os.path.join(ROOT, "data", "etf", "parquet")
OUT = os.path.join(ROOT, "data", "etf", "rebalance_cadence_study.json")

CODES = ["515450", "511220"]
TARGET = {"515450": 0.5, "511220": 0.5}
CAPITAL = 90_000.0
LOT = 100
COMMISSION, MIN_COMMISSION = 0.0003, 5.0


def load_closes() -> pd.DataFrame:
    frames = []
    for c in CODES:
        p = os.path.join(ETF_DIR, f"{c}.parquet")
        if not os.path.exists(p):
            raise SystemExit(f"缺 {p} → 先跑 scripts/etf_data_pipeline.py --codes {','.join(CODES)} --incremental")
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["date"])
        s = pd.Series(pd.to_numeric(d["close"], errors="coerce").values, index=d["date"], dtype=float)
        frames.append(s[s.notna() & (s > 0)].rename(c))
    px = pd.concat(frames, axis=1).dropna()
    return px


def fee(amount: float) -> float:
    return max(abs(amount) * COMMISSION, MIN_COMMISSION)


def _cadence_dates(cal: pd.DatetimeIndex, cadence: str) -> set:
    out = set()
    if cadence == "daily":
        return set(cal)
    if cadence == "annual":
        for y, grp in pd.DataFrame(index=cal).groupby(cal.year):
            out.add(grp.index[0])
        return out
    if cadence == "quarterly":
        per = pd.Series(cal, index=cal).groupby([cal.year, (cal.month - 1) // 3]).min()
        return set(per.tolist())
    if cadence == "monthly":
        per = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).min()
        return set(per.tolist())
    if cadence == "weekly":
        per = pd.Series(cal, index=cal).groupby([cal.isocalendar().year, cal.isocalendar().week]).min()
        return set(per.tolist())
    raise ValueError(f"未知 cadence {cadence}")


def simulate(px: pd.DataFrame, cadence: str, band: float | None = None) -> dict:
    """整数手账户：cadence 决定「检查」频率，band 决定检查后动不动手（None=只要偏离就动）。

    一遍走完，不在第二遍里重复逻辑：成交笔数/成交额/佣金/再平衡日/净值序列都来自同一次模拟。
    """
    cal = px.index
    check_days = _cadence_dates(cal, cadence) if cadence != "hold" else set()

    prices0 = {c: float(px.iloc[0][c]) for c in CODES}
    lots = {c: int((CAPITAL * TARGET[c]) // (prices0[c] * LOT)) for c in CODES}
    invested0 = sum(lots[c] * prices0[c] * LOT for c in CODES)
    fees = sum(fee(lots[c] * prices0[c] * LOT) for c in CODES)      # 建仓佣金
    cash = CAPITAL - invested0 - fees

    nav_dates, nav_vals = [], []
    n_trades = 0
    traded_value = 0.0
    n_check = n_act = 0

    for d in cal:
        prices = {c: float(px.loc[d][c]) for c in CODES}
        pv = sum(lots[c] * LOT * prices[c] for c in CODES)
        nav = cash + pv
        nav_dates.append(d)
        nav_vals.append(nav)

        if d not in check_days or nav <= 0:
            continue
        n_check += 1
        w_now = {c: lots[c] * LOT * prices[c] / nav for c in CODES}
        if band is not None and max(abs(w_now[c] - TARGET[c]) for c in CODES) <= band:
            continue                                    # 看一眼，不动手，不花钱
        n_act += 1
        for c in CODES:
            price = prices[c]
            want_c = int((nav * TARGET[c]) // (price * LOT))
            delta = want_c - lots[c]
            if delta > 0:                                  # 买不动就少买，不许透支
                while delta > 0 and delta * price * LOT + fee(delta * price * LOT) > cash:
                    delta -= 1
            if delta == 0:
                continue
            gross = delta * LOT * price
            cash -= gross
            f = fee(gross)
            cash -= f                                      # 佣金必须真的离开账户，否则成本口径是假的
            fees += f
            traded_value += abs(gross)
            lots[c] = lots[c] + delta
            n_trades += 1
            assert cash >= -1e-9, f"现金透支 {cash} @ {d}"

    s = pd.Series(nav_vals, index=pd.DatetimeIndex(nav_dates))
    years = len(s) / 252.0
    total = s.iloc[-1] / CAPITAL - 1
    ann = (s.iloc[-1] / CAPITAL) ** (1.0 / years) - 1
    vol = float(s.pct_change().dropna().std() * np.sqrt(252))
    dd = float(((s - s.cummax()) / s.cummax()).min())
    return {
        "cadence": cadence if band is None else f"{cadence}+band{int(band * 100)}pp",
        "checks": n_check,
        "acted": n_act,
        "n_trades": n_trades,
        "traded_value": round(traded_value, 2),
        "turnover_pct_of_capital": round(traded_value / CAPITAL, 4),
        "fees_total": round(fees, 2),
        "fees_pct_of_capital": round(fees / CAPITAL, 4),
        "initial_invested_pct": round(invested0 / CAPITAL, 4),
        "years": round(years, 2),
        "total_pct": round(total * 100, 2),
        "annualized_pct": round(ann * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "sharpe": round(ann / vol, 4) if vol > 0 else 0.0,
        "max_dd_pct": round(dd * 100, 2),
        "final_nav": round(float(s.iloc[-1]), 2),
    }


def main() -> None:
    px = load_closes()
    start, end = px.index[0].strftime("%Y-%m-%d"), px.index[-1].strftime("%Y-%m-%d")
    plans = [
        ("hold", None),          # 一次性建仓，永远不动
        ("annual", None),        # §12.77 现行口径
        ("quarterly", None),
        ("monthly", None),
        ("weekly", None),        # 「几天动一次」
        ("daily", None),         # 每天动
        ("weekly", 0.05),        # 每周检查，偏离 >5pp 才动手
        ("weekly", 0.03),        # 每周检查，偏离 >3pp 才动手
        ("daily", 0.05),         # 每天检查，偏离 >5pp 才动手
    ]
    rows = [simulate(px, cad, band) for cad, band in plans]

    # 分段稳健性：全样本的排序可能是风格赌注（§12.90 教训：先分段再看）
    windows = [("前 1/3", "2020-02-26", "2022-05-31"),
               ("中 1/3", "2022-01-03", "2024-04-30"),
               ("后 1/3", "2024-01-02", "2026-11-09"),
               ("台账窗口", "2023-05-12", "2026-05-08")]
    seg = {}
    for name, a, b in windows:
        sub = px.loc[a:b]
        if len(sub) < 120:
            continue
        seg[name] = {
            "window": [sub.index[0].strftime("%Y-%m-%d"), sub.index[-1].strftime("%Y-%m-%d")],
            "rows": {r["cadence"]: {"annualized_pct": r["annualized_pct"], "sharpe": r["sharpe"],
                                    "max_dd_pct": r["max_dd_pct"], "n_trades": r["n_trades"],
                                    "fees_pct_of_capital": r["fees_pct_of_capital"]}
                     for r in (simulate(sub, cad, band) for cad, band in plans)},
        }
    rank = {}
    for cad, band in plans:
        key = cad if band is None else f"{cad}+band{int(band * 100)}pp"
        wins = sum(1 for name in seg if seg[name]["rows"][key]["sharpe"] >= seg[name]["rows"]["annual"]["sharpe"])
        rank[key] = {"段数": len(seg), "夏普不劣于 annual 的段数": wins}

    out = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "window": [start, end],
        "capital": CAPITAL,
        "codes": CODES,
        "caliber": "前复权收盘价（含分红）；整数手 100 份；佣金万3最低5元；ETF 免印花税；未计滑点",
        "rows": rows,
        "segments": seg,
        "sharpe_rank_vs_annual": rank,
        "noise_scale": 0.18,   # §12.65：同策略改一个参数的夏普跨度噪声尺度，小于它就是噪声
        "caveats": [
            "同一份价、同一批腿，只改节奏 → 行间差异就是节奏本身的效果（配对比较，不是独立样本）",
            "前复权锚点重算对本模拟无影响：账户只用同一批载入的价格",
            "band 口径的「检查」不花钱，只有真成交才花钱；所以 band 是把『盯市频率』和『交易频率』拆开",
            "未计滑点/冲击；周频 6.4 年成交额=本金的 519%，按单边 0.1% 算约再多 0.5% 本金",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    hdr = ("节奏", "检查次数", "成交笔数", "成交额/本金", "佣金", "佣金占本金", "年化", "波动", "夏普", "最大回撤", "终值")
    print(f"\n=== ETF 主仓 50/50 再平衡节奏实测（{start} → {end}，本金 {CAPITAL:,.0f} 元）===")
    print("  " + " | ".join(h.rjust(11) for h in hdr))
    for r in rows:
        print("  " + " | ".join(str(v).rjust(11) for v in (
            r["cadence"], r["checks"], r["n_trades"], f"{r['turnover_pct_of_capital'] * 100:.1f}%",
            f"{r['fees_total']:.0f}", f"{r['fees_pct_of_capital'] * 100:.2f}%",
            f"{r['annualized_pct']:.2f}%", f"{r['vol_pct']:.2f}%", f"{r['sharpe']:.4f}",
            f"{r['max_dd_pct']:.2f}%", f"{r['final_nav']:.0f}")))
    print(f"\n→ {OUT}")

    print("\n=== 分段稳健性（每段独立账户，本金同为 9 万）===")
    for name, blk in seg.items():
        print(f"\n[{name}] {blk['window'][0]} → {blk['window'][1]}")
        print("  " + " | ".join(h.rjust(11) for h in ("节奏", "年化", "夏普", "最大回撤", "笔数", "佣金%")))
        for key, r in blk["rows"].items():
            print("  " + " | ".join(str(v).rjust(11) for v in (
                key, f"{r['annualized_pct']:.2f}%", f"{r['sharpe']:.4f}", f"{r['max_dd_pct']:.2f}%",
                r["n_trades"], f"{r['fees_pct_of_capital'] * 100:.2f}%")))
    print("\n=== 夏普不劣于 annual 的段数（噪声尺度 0.18，差值小于它不算优势）===")
    for key, r in rank.items():
        print(f"  {key:18s} {r['夏普不劣于 annual 的段数']}/{r['段数']}")


if __name__ == "__main__":
    main()
