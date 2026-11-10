#!/usr/bin/env python3
"""§12.84 「买了迟早会涨」长窗口无偏复核：2013–2022 入场 × 含退市股 × 真退市结局。

和 §12.83 的区别（这次把上界变成结论）：
1. 宇宙来自 baostock 注册表 5,565 只（含 341 只已退市），价格来自 akshare 后复权 → **退市股在样本里**；
2. 退市不再是「折价敏感性假设」，而是**真实结局**：持仓期内被摘牌的，给两个口径
   A 乐观=按摘牌前最后一根收盘价卖出；B 悲观=这份仓位归零。真值在两者之间（实务上多在退市整理期出，
   累计已跌 6–8 成），两个口径都报，差值就是诚实边界；
3. 入场年份 2013–2022 共 10 期 × K=20 = 200 只/规则（§12.83 只有 40 只，SE≈7.8pp；这次 SE≈3.5pp）；
4. 回报一律用**自建总回报面板** tr_panel（未复权价 × 分红再投 × 送转扩股，§12.84 口径审计的结论）。
   `close_hfq`（东财后复权）与 `pct_chg_raw`（实测就是未复权）都**禁止**用来算收益。

规则只保留「价格行为类」：随机、低波动、12 月动量、大成交额、低波动∩大成交额。
PE/PB/ROE/股息率类规则在这个宇宙里**凑不齐无偏数据**（qlib 层只有 OHLCV 且不含退市股；
akshare 财务/分红接口对退市股直接返回空）——这本身就是一条结论：
基本面选股在含退市样本上连复现条件都没有。

成本照旧：万3 佣金（最低 5 元）、卖出印花税 0.05%、整数手（100 股）、买入价用未复权真实价。

用法：cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/buy_and_hold_hitrate_long.py
产出：../data/stocks/buy_and_hold_hitrate_long.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRICE_DIR = os.path.join(ROOT, "data", "parquet", "tr_panel")   # 自建总回报面板（§12.84 口径）
REGISTRY = os.path.join(ROOT, "data", "parquet", "baostock_full", "_registry.parquet")
OUT = os.path.join(ROOT, "data", "stocks", "buy_and_hold_hitrate_long.json")

CAPITAL = 90_000.0
LOT = 100
COMM_RATE, COMM_MIN = 3e-4, 5.0
STAMP = 5e-4                      # 卖出印花税
ENTRY_YEARS = list(range(2013, 2023))
HORIZONS_Y = (1, 2, 3)
KS = (10, 20)
MIN_BARS_AT_ENTRY = 250           # 要有 250 日波动率
AMOUNT_FLOOR = 1e7                # 入场日附近日均成交额 ≥1000 万（太死的票不进池）
RANDOM_SEEDS = (7, 131, 20240)
RULES = ("random", "low_vol", "mom12", "big_amount", "low_vol_big")
RULE_CN = {"random": "随机", "low_vol": "低波动", "mom12": "12月动量",
           "big_amount": "大成交额", "low_vol_big": "低波动∩大成交额"}


def fee(buy_value: float, sell_value: float | None) -> float:
    f = max(buy_value * COMM_RATE, COMM_MIN)
    if sell_value is not None:
        f += max(sell_value * COMM_RATE, COMM_MIN) + sell_value * STAMP
    return f


class Panel:
    """全宇宙价格面板：每只票存 (日期序列, 未复权收盘, 自建总回报指数, 成交额)。float32 省内存。"""

    def __init__(self) -> None:
        self.close: dict[str, np.ndarray] = {}
        self.tr: dict[str, np.ndarray] = {}
        self.amount: dict[str, np.ndarray] = {}
        self.dates: dict[str, np.DatetimeIndex] = {}
        self.cal = None

    def load(self) -> None:
        reg = pd.read_parquet(REGISTRY)
        reg["status"] = pd.to_numeric(reg["status"], errors="coerce")
        self.reg = reg
        alld = []
        files = {}
        for f in os.listdir(PRICE_DIR):
            if not f.endswith(".parquet"):
                continue
            num, mkt = f[: -len(".parquet")].rsplit("_", 1)
            files[f"{num}.{mkt}"] = os.path.join(PRICE_DIR, f)
        self.files = files
        for key, path in files.items():
            d = pd.read_parquet(path, columns=["date", "close", "tr_index", "amount"])
            d["date"] = pd.to_datetime(d["date"])
            d = d.sort_values("date")
            self.dates[key] = pd.DatetimeIndex(d["date"])
            self.close[key] = d["close"].astype("float32").to_numpy()
            self.tr[key] = d["tr_index"].astype("float32").to_numpy()
            self.amount[key] = d["amount"].astype("float32").to_numpy()
            alld.append(d["date"].to_numpy())
        self.cal = pd.DatetimeIndex(sorted(np.unique(np.concatenate(alld))))
        idx = {r["code"].split(".")[1] + "." + r["code"].split(".")[0].upper(): r for _, r in reg.iterrows()}
        self.meta = {}
        for key in self.dates:
            row = idx.get(key)
            out_raw = str(row["outDate"]) if row is not None else ""
            out = pd.Timestamp(out_raw) if out_raw not in ("", "NaT", "nan") else None
            self.meta[key] = {"delisted": bool(row is not None and row["status"] != 1), "outDate": out}
        print(f"面板：{len(self.dates)} 只，日历 {self.cal[0].date()}→{self.cal[-1].date()}，"
              f"退市标记 {sum(1 for m in self.meta.values() if m['delisted'])} 只")

    def asof_index(self, key: str, entry: pd.Timestamp) -> int:
        d = self.dates[key]
        return int(np.searchsorted(d.values, np.datetime64(entry), side="right") - 1)

    def tradable_at(self, key: str, entry: pd.Timestamp) -> bool:
        """入场日这只票必须真实在架：有 bar、bar 落在 entry 当天或前 5 个交易日内、且摘牌日在 entry 之后。"""
        i = self.asof_index(key, entry)
        if i < MIN_BARS_AT_ENTRY - 1:
            return False
        if self.dates[key][i] < entry - pd.Timedelta(days=10):
            return False
        m = self.meta[key]
        if m["outDate"] is not None and m["outDate"] <= entry:
            return False
        return True


def features_at(p: Panel, key: str, i: int) -> dict | None:
    """点位特征：只用 ≤ 入场日的 bar（后视数据一律不许用）。"""
    c = p.close[key][: i + 1].astype(float)
    h = p.tr[key][: i + 1].astype(float)
    a = p.amount[key][: i + 1].astype(float)
    if len(c) < MIN_BARS_AT_ENTRY or c[-1] <= 0:
        return None
    r = h[1:] / h[:-1] - 1.0
    if not np.isfinite(r).all() or len(r) < 250:
        return None
    vol250 = float(np.std(r[-250:], ddof=1))
    mom12 = float(h[-1] / h[-252] - 1.0) if len(h) > 252 else np.nan
    amt250 = float(np.mean(a[-250:]))
    if not np.isfinite(amt250) or amt250 < AMOUNT_FLOOR:
        return None
    return {"price": float(c[-1]), "vol250": vol250, "mom12": mom12, "amount": amt250}


def pick(p: Panel, entry: pd.Timestamp, rule: str, k: int, seed: int) -> list[str]:
    cands = []
    for key in p.dates:
        if not p.tradable_at(key, entry):
            continue
        f = features_at(p, key, p.asof_index(key, entry))
        if f is None:
            continue
        f["code"] = key
        cands.append(f)
    df = pd.DataFrame(cands)
    if df.empty:
        return []
    budget_each = CAPITAL / k
    price_cap = budget_each / LOT                      # 一手都买不起的直接剔除
    df = df[df["price"] <= price_cap]
    if rule == "random":
        rs = np.random.default_rng(seed)
        idx = rs.choice(len(df), size=min(k, len(df)), replace=False)
        return list(df.iloc[idx]["code"])
    if rule == "low_vol":
        return list(df.nsmallest(k, "vol250")["code"])
    if rule == "big_amount":
        return list(df.nlargest(k, "amount")["code"])
    if rule == "mom12":
        d2 = df.dropna(subset=["mom12"])
        return list(d2.nlargest(k, "mom12")["code"])
    if rule == "low_vol_big":
        big = set(df.nlargest(max(len(df) // 2, k), "amount")["code"])
        d2 = df[df["code"].isin(big)]
        return list(d2.nsmallest(k, "vol250")["code"])
    raise ValueError(rule)


def settle(p: Panel, codes: list[str], entry: pd.Timestamp, horizon_y: int) -> dict:
    """整数手买入 → 到期（或退市）卖出，费用真的从现金里扣（§12.82 的坑：费用必须离开净值）。"""
    exit_target = entry + pd.DateOffset(years=horizon_y)
    each = CAPITAL / len(codes) if codes else 0.0
    cash = CAPITAL
    fees = 0.0
    wins = 0
    per_stock = []
    delisted_recovered = 0.0            # A 口径里退市仓位拿回的钱（B 口径要抹掉）
    delisted_in_horizon = 0

    for code in codes:
        i = p.asof_index(code, entry)
        price = float(p.close[code][i])
        h_in = float(p.tr[code][i])
        lots = int(each // (price * LOT))
        if lots <= 0 or price <= 0 or h_in <= 0:
            continue
        buy_val = lots * LOT * price
        f_buy = max(buy_val * COMM_RATE, COMM_MIN)
        cash -= buy_val + f_buy
        fees += f_buy

        d = p.dates[code]
        last = len(d) - 1
        out = p.meta[code]["outDate"]
        # 判摘牌只能看「面板是否提前结束」：最后一根 bar 早于到期日 = 这只票没 bar 了。
        # （这里曾写反成 `j < last` —— 那是「面板比到期日还长」= 活得好好的正常情况，
        #   结果全部持仓被误判成退市、结算日被推到面板最后一根，3 年持有变成 4~14 年，整份结果作废。）
        ends_early = bool(d[last] < exit_target)
        j = last if ends_early else int(np.searchsorted(d.values, np.datetime64(exit_target), side="right") - 1)
        if out is not None and out <= exit_target:       # 注册表说到期前摘牌：结算不许越过摘牌日
            k = int(np.searchsorted(d.values, np.datetime64(out), side="left") - 1)
            j = min(j, max(k, i))
            ends_early = True
        assert i <= j <= last, f"结算日跑到了买入日之前 {code} {entry.date()}"
        caliber = "退市A(末根卖出)" if ends_early else "到期卖出"
        if ends_early:
            delisted_in_horizon += 1

        h_out = float(p.tr[code][j])
        gross = h_out / h_in - 1.0
        sell_val = lots * LOT * price * (1.0 + gross)
        f_sell = max(sell_val * COMM_RATE, COMM_MIN) + sell_val * STAMP
        cash += sell_val - f_sell
        fees += f_sell
        if caliber != "到期卖出":
            delisted_recovered += sell_val - f_sell
        assert cash >= -1e-6, f"现金被写到负数 {cash} @ {code} {entry.date()}"

        pnl_a = sell_val - f_sell - buy_val - f_buy
        wins += int(pnl_a > 0)
        per_stock.append({"code": code, "lots": lots, "buy_price": round(price, 3),
                          "ret_pct": round(gross * 100, 2), "caliber": caliber,
                          "pnl_a": round(pnl_a, 2)})

    nav_a = cash                                        # A 口径净值：逐笔结清后的现金
    nav_b = nav_a - delisted_recovered                  # B 口径：退市仓位一分钱拿不回来
    years = horizon_y
    ann_a = (nav_a / CAPITAL) ** (1.0 / years) - 1.0 if nav_a > 0 else -1.0
    ann_b = (nav_b / CAPITAL) ** (1.0 / years) - 1.0 if nav_b > 0 else -1.0
    rets = [x["ret_pct"] for x in per_stock]
    return {"n": len(per_stock), "hit_pct": round(100.0 * wins / max(len(per_stock), 1), 1),
            "ann_a_pct": round(ann_a * 100, 2), "ann_b_pct": round(ann_b * 100, 2),
            "median_ret_pct": round(float(np.median(rets)), 2) if rets else None,
            "worst_ret_pct": round(float(np.min(rets)), 2) if rets else None,
            "best_ret_pct": round(float(np.max(rets)), 2) if rets else None,
            "delisted_in_horizon": delisted_in_horizon, "fees": round(fees, 2),
            "nav_a": round(nav_a, 2), "nav_b": round(nav_b, 2), "picks": per_stock}


def main() -> int:
    for d in (PRICE_DIR, REGISTRY):
        if not os.path.exists(d):
            print(f"缺 {d} → 先跑 fetch_akshare_full_history.py")
            return 1
    p = Panel()
    p.load()
    results = []
    for year in ENTRY_YEARS:
        entry = p.cal[p.cal.year == year][0]
        for rule in RULES:
            for k in KS:
                seeds = RANDOM_SEEDS if rule == "random" else (7,)
                runs = []
                for s in seeds:
                    codes = pick(p, entry, rule, k, s)
                    if not codes:
                        continue
                    r = settle(p, codes, entry, 3)      # 3 年是主口径
                    r["seed"] = s
                    runs.append(r)
                if not runs:
                    continue
                agg = {"entry": str(entry.date()), "year": year, "rule": rule, "k": k,
                       "hit_pct": round(float(np.mean([r["hit_pct"] for r in runs])), 1),
                       "ann_a_pct": round(float(np.mean([r["ann_a_pct"] for r in runs])), 2),
                       "ann_b_pct": round(float(np.mean([r["ann_b_pct"] for r in runs])), 2),
                       "delisted_in_horizon": int(np.sum([r["delisted_in_horizon"] for r in runs])),
                       "n": int(np.sum([r["n"] for r in runs])),
                       "runs": runs}
                results.append(agg)
                print(f"{year} {RULE_CN[rule]:>12} K={k:>2} n={agg['n']:>2} "
                      f"命中{agg['hit_pct']:>5}% 年化A{agg['ann_a_pct']:>7}% 年化B{agg['ann_b_pct']:>7}% "
                      f"期内退市{agg['delisted_in_horizon']}", flush=True)

    # 规则级汇总（跨 10 期）
    summary = []
    for rule in RULES:
        for k in KS:
            rows = [r for r in results if r["rule"] == rule and r["k"] == k]
            if not rows:
                continue
            hits = [r["hit_pct"] for r in rows]
            ann_a = [r["ann_a_pct"] for r in rows]
            ann_b = [r["ann_b_pct"] for r in rows]
            summary.append({"rule": rule, "cn": RULE_CN[rule], "k": k, "cohorts": len(rows),
                            "n_stocks": int(np.sum([r["n"] for r in rows])),
                            "hit_mean_pct": round(float(np.mean(hits)), 1),
                            "hit_min_pct": round(float(np.min(hits)), 1),
                            "hit_max_pct": round(float(np.max(hits)), 1),
                            "hit_sd_pp": round(float(np.std(hits, ddof=1)), 1),
                            "ann_a_mean_pct": round(float(np.mean(ann_a)), 2),
                            "ann_b_mean_pct": round(float(np.mean(ann_b)), 2),
                            "ann_a_win_cohorts": int(np.sum(np.array(ann_a) > 0)),
                            "delisted_total": int(np.sum([r["delisted_in_horizon"] for r in rows]))})
    summary.sort(key=lambda x: -x["ann_a_mean_pct"])
    print("\n=== 规则级汇总（2013-2022 十期，3 年持有） ===")
    print(f"{'规则':>16} {'K':>3} {'只数':>5} {'命中均值':>9} {'命中区间':>14} {'年化A':>8} {'年化B':>8} {'正收益期数':>10} {'期内退市':>8}")
    for s in summary:
        print(f"{s['cn']:>16} {s['k']:>3} {s['n_stocks']:>5} {s['hit_mean_pct']:>8}% "
              f"{s['hit_min_pct']:>5}-{s['hit_max_pct']:>5}% {s['ann_a_mean_pct']:>7}% {s['ann_b_mean_pct']:>7}% "
              f"{s['ann_a_win_cohorts']:>6}/{s['cohorts']} {s['delisted_total']:>8}")

    out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "universe": {"price_dir": PRICE_DIR, "registry": REGISTRY,
                        "panel_stocks": len(p.dates),
                        "panel_delisted": sum(1 for m in p.meta.values() if m["delisted"])},
           "calibers": {"A_乐观": "退市持仓按摘牌前最后一根收盘价卖出",
                        "B_悲观": "退市持仓归零",
                        "说明": "真实结果在两者之间（退市整理期出局通常已跌 6-8 成）"},
           "excluded_rules": ["低PE/低PB/高ROE/股息率：含退市宇宙里凑不齐无偏数据（qlib 层仅 OHLCV 且不含退市股；akshare 财务/分红接口对退市股返回空）"],
           "cost": {"commission_rate": COMM_RATE, "commission_min": COMM_MIN, "stamp_sell": STAMP, "lot": LOT},
           "summary": summary, "detail": results}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
