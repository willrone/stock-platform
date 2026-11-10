#!/usr/bin/env python3
"""「买了迟早会涨」的可测版本：一次性买入、持有 1/2/3 年不动，命中率到底多少（¥90,000 口径）。

用户的问法："找出那些买了以后迟早会涨能赚钱的股票……不用每天都能找到，但是要找的准。"
把它变成可证伪的命题：
  在入场日 D，用**当时拿得到的信息**选 K 只，等权买入（整数手），持有 H 年不动，
  问：单只正收益的概率（个股命中率）是多少？组合正收益的概率（胜率）是多少？
  比随机、比宽基 ETF 买入持有，高多少？高出来的部分是否超过噪声？

数据口径（全部实测确认，不靠文档措辞）：
- 收益率用 baostock_daily 的 `pct_chg` 递推链（**含分红**）。已审计：`stock_data.close` 与
  `baostock.close` 逐日相同 = 两者都是未复权价；只有除息日 `pct_chg` 与未复权日收益不同
  （600000 五年 6 天，正是年度分红）→ 所以**只有 pct_chg 链是总回报口径**，直接用 close 算长期收益会漏分红。
- 估值 pe_ttm / pb_mrq / is_st / trade_status 来自 baostock_daily，2021-01→2025-12 覆盖率 100%（抽样 40 只）。
- 财务质量 roe 等来自 data/fundamentals/financial_ak_full.json，记录带**披露日期**，只用披露日 ≤ 入场日的最新一条（点位口径）。
- **池只有存续股（5,597 只），没有任何退市股**（300104 乐视、000018 神城等文件不存在）→
  所有命中率都是**上界**。台账 §十一 实测窗口内退市股占 2.27%，本脚本另给退市折价敏感性档。
- 成本：佣金万 3、单笔最低 5 元；卖出印花税 0.05%；整数手 100 股。
- 9 万硬约束：买 K 只 → 每只预算 9万/K → 单价必须 ≤ 9万/K/100 才凑得够一手。
  K=20 → 只买得起 ≤45 元/股；K=10 → ≤90 元/股。高价股（茅台、茅台类）在这个本金下**结构上不可选**。

用法：
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/buy_and_hold_hitrate_study.py
产出：../data/stocks/buy_and_hold_hitrate_study.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)
BAOSTOCK_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")
ETF_DIR = os.path.join(ROOT, "data", "etf", "parquet")
FUND_JSON = os.path.join(ROOT, "data", "fundamentals", "financial_ak_full.json")
OUT = os.path.join(ROOT, "data", "stocks", "buy_and_hold_hitrate_study.json")
PANEL_CACHE = os.path.join(ROOT, "data", "stocks", "buy_and_hold_panel.npz")

CAPITAL = 90000.0
LOT = 100
FEE_RATE = 0.0003
MIN_FEE = 5.0
STAMP_RATE = 0.0005                 # 卖出印花税（2023-08 起 0.05%）；买入不征
HOLDS = {"1y": 1, "2y": 2, "3y": 3}
ENTRY_YEARS = [2021, 2022, 2023, 2024, 2025]
KS = [10, 20]
MIN_HISTORY = 120                   # 入场前至少 120 根 bar，剔次新股
HAIRCUT_SCENARIOS = [0.015, 0.03, 0.05]   # 池内退市占比（§十一 实测 2.27%）→ 折 -70%
HAIRCUT_RETURN = -0.70
NOISE_PP = 15.0                     # 30~60 个样本下命中率的标准误 ≈ 9pp，小于 15pp 的差不可信


# ---------------------------------------------------------------- panel

def _file_code(fname: str) -> str | None:
    if not fname.endswith(".parquet"):
        return None
    stem = os.path.basename(fname)[: -len(".parquet")]
    if "_" not in stem:
        return None
    num, mkt = stem.rsplit("_", 1)
    if mkt.upper() not in ("SH", "SZ", "BJ"):
        return None
    return f"{num}.{mkt.upper()}"


def load_panel(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """把 baostock_daily 摊成宽矩阵：行=全局日历，列=代码。字段: adj/ret/raw/amount/pe/pb/st/halt/bars。"""
    cols = ["date", "close", "pct_chg", "amount", "pe_ttm", "pb_mrq", "is_st", "trade_status"]
    files = sorted(f for f in os.listdir(BAOSTOCK_DIR) if f.endswith(".parquet"))
    codes = [c for f in files if (c := _file_code(f))]
    if os.path.exists(PANEL_CACHE) and use_cache:
        z = np.load(PANEL_CACHE, allow_pickle=True)
        cal = pd.DatetimeIndex(z["cal"])
        out = {k: pd.DataFrame(z[k], index=cal, columns=list(z["codes"])) for k in
               ("adj", "ret", "raw", "amount", "pe", "pb", "st", "halt", "bars")}
        return out

    cal = pd.DatetimeIndex(sorted({d for f in files for d in
                                   pd.read_parquet(os.path.join(BAOSTOCK_DIR, f), columns=["date"])["date"]}))
    fields = {k: np.full((len(cal), len(codes)), np.nan, dtype=np.float32)
              for k in ("adj", "ret", "raw", "amount", "pe", "pb", "st", "halt", "bars")}
    for j, f in enumerate(files):
        code = _file_code(f)
        if code is None:
            continue
        d = pd.read_parquet(os.path.join(BAOSTOCK_DIR, f), columns=cols)
        d["date"] = pd.to_datetime(d["date"])
        d = d.drop_duplicates("date").set_index("date").reindex(cal)
        ret = pd.to_numeric(d["pct_chg"], errors="coerce").to_numpy(dtype=np.float64) / 100.0
        adj = np.cumprod(1.0 + np.nan_to_num(ret, nan=0.0))
        adj[np.isnan(ret)] = np.nan
        bars = np.cumsum(~np.isnan(ret))
        fields["adj"][:, j] = adj
        fields["ret"][:, j] = ret
        fields["raw"][:, j] = pd.to_numeric(d["close"], errors="coerce").to_numpy(dtype=np.float32)
        fields["amount"][:, j] = pd.to_numeric(d["amount"], errors="coerce").to_numpy(dtype=np.float32)
        fields["pe"][:, j] = pd.to_numeric(d["pe_ttm"], errors="coerce").to_numpy(dtype=np.float32)
        fields["pb"][:, j] = pd.to_numeric(d["pb_mrq"], errors="coerce").to_numpy(dtype=np.float32)
        fields["st"][:, j] = pd.to_numeric(d["is_st"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        fields["halt"][:, j] = (pd.to_numeric(d["trade_status"], errors="coerce") != 1).astype(np.float32)
        fields["bars"][:, j] = bars
    out = {k: pd.DataFrame(v, index=cal, columns=codes) for k, v in fields.items()}
    np.savez_compressed(PANEL_CACHE, cal=cal.values, codes=np.array(codes), **{k: v.to_numpy() for k, v in out.items()})
    return out


def load_roe() -> dict[str, list[tuple[pd.Timestamp, float]]]:
    """code → [(披露日, roe)]，只用披露日 ≤ 入场日的记录（点位口径，不回灌未来）。"""
    raw = json.load(open(FUND_JSON))
    out: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for num, recs in raw.items():
        seq = []
        for item in recs:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            day, data = item[0], item[1]
            if not isinstance(data, dict) or data.get("roe") is None:
                continue
            try:
                seq.append((pd.Timestamp(day), float(data["roe"])))
            except Exception:
                continue
        if seq:
            seq.sort(key=lambda x: x[0])
            out[f"{num[:6]}"] = seq
    return out


def roe_as_of(roe_map: dict, code: str, entry: pd.Timestamp) -> float | None:
    seq = roe_map.get(code.split(".")[0])
    if not seq:
        return None
    val = None
    for day, v in seq:
        if day <= entry:
            val = v
        else:
            break
    return val


# ---------------------------------------------------------------- rules

RULES = {
    "随机(3种子)": "random",
    "低PE": "low_pe",
    "低PB": "low_pb",
    "高ROE": "high_roe",
    "高ROE+低PB(好公司便宜买)": "quality_value",
    "低波动": "low_vol",
    "12月动量": "momentum",
    "大成交额(蓝筹proxy)": "big_amount",
}


def features_at(panel: dict, entry: pd.Timestamp, roe_map: dict, price_cap: float, seed: int = 7) -> pd.DataFrame:
    """入场日 D 的可选宇宙 + 当时可得的所有特征（严格只用 ≤D 的数据）。"""
    cal = panel["adj"].index
    i = cal.get_loc(entry)
    lo = max(0, i - 250)
    sl = slice(lo, i + 1)
    adj = panel["adj"].iloc[sl]
    ok_entry = panel["adj"].iloc[i].notna()
    hist = adj.notna().sum()
    px = panel["raw"].iloc[i]
    df = pd.DataFrame({
        "adj_entry": panel["adj"].iloc[i],
        "raw_entry": px,
        "pe": panel["pe"].iloc[i],
        "pb": panel["pb"].iloc[i],
        "st": panel["st"].iloc[i],
        "halt": panel["halt"].iloc[i],
        "hist_bars": hist,
        "mom12": (panel["adj"].iloc[i] / adj.iloc[0] - 1.0) if len(adj) >= 251 else np.nan,
        "vol12": panel["ret"].iloc[sl].std(),
        "amount60": panel["amount"].iloc[max(0, i - 59):i + 1].mean(),
    })
    df["roe"] = pd.to_numeric(pd.Series([roe_as_of(roe_map, c, entry) for c in df.index], index=df.index),
                              errors="coerce")          # 必须转成 float：全 None 时 object 列会让 nlargest 直接报错
    elig = (ok_entry & (df["hist_bars"] >= MIN_HISTORY) & (df["st"] == 0) & (df["halt"] == 0)
            & (df["raw_entry"] > 0) & (df["raw_entry"] <= price_cap))
    df = df[elig].copy()
    rng = np.random.default_rng(seed)
    df["rand_key"] = rng.random(len(df))
    return df


def pick(df: pd.DataFrame, rule: str, k: int) -> list[str]:
    if rule == "random":
        return list(df["rand_key"].nsmallest(k).index)
    if rule == "low_pe":
        x = df[(df["pe"] > 0) & np.isfinite(df["pe"])].nsmallest(k, "pe")
    elif rule == "low_pb":
        x = df[(df["pb"] > 0) & np.isfinite(df["pb"])].nsmallest(k, "pb")
    elif rule == "high_roe":
        x = df[df["roe"].notna() & (df["roe"] > 0)].nlargest(k, "roe")
    elif rule == "quality_value":
        q = df[df["roe"].notna() & (df["roe"] >= 8)]        # 质量闸门：ROE ≥ 8%
        x = q[(q["pb"] > 0) & np.isfinite(q["pb"])].nsmallest(k, "pb")
    elif rule == "low_vol":
        x = df[df["vol12"].notna()].nsmallest(k, "vol12")
    elif rule == "momentum":
        x = df[df["mom12"].notna()].nlargest(k, "mom12")
    elif rule == "big_amount":
        x = df[df["amount60"].notna()].nlargest(k, "amount60")
    else:
        raise ValueError(rule)
    return list(x.index)


def settle(panel: dict, entry: pd.Timestamp, hold_years: int, codes: list[str], k: int) -> dict:
    """整数手买入 → 持有 H 年 → 结算。收益用 pct_chg 递推链（含分红），手数受单价约束。"""
    cal = panel["adj"].index
    target_exit = entry + pd.DateOffset(years=hold_years)
    exit_candidates = cal[cal <= target_exit]
    if len(exit_candidates) == 0:
        return {}
    exit_day = exit_candidates[-1]
    per_stock, cash_in, idle, vanished, skipped = [], 0.0, 0.0, 0, 0
    budget = CAPITAL / k
    for c in codes:
        px_in = float(panel["raw"].loc[entry, c])
        adj_in = float(panel["adj"].loc[entry, c])
        if not np.isfinite(px_in) or not np.isfinite(adj_in) or px_in <= 0:
            skipped += 1
            continue
        lots = int(budget // (px_in * LOT))
        if lots <= 0:
            skipped += 1
            continue
        cost = lots * LOT * px_in
        buy_fee = max(cost * FEE_RATE, MIN_FEE)
        row = panel["adj"][c].loc[:exit_day]
        adj_out = row.dropna()
        if len(adj_out) == 0:
            skipped += 1
            continue
        last_day = adj_out.index[-1]
        r = float(adj_out.iloc[-1] / adj_in - 1.0)
        if last_day < exit_day - pd.Timedelta(days=30):
            vanished += 1                      # 中途消失：按最后可得价结算（偏乐观）
        gross_out = cost * (1.0 + r)
        sell_fee = max(gross_out * FEE_RATE, MIN_FEE)
        stamp = gross_out * STAMP_RATE
        per_stock.append({"code": c, "entry_price": round(px_in, 3), "lots": lots,
                          "cost": round(cost, 2), "total_return_pct": round(r * 100, 2),
                          "net_return_pct": round((gross_out - buy_fee - sell_fee - stamp) / cost * 100 - 100, 2),
                          "vanished": vanished})
        cash_in += cost + buy_fee
        idle += 0.0
    if not per_stock:
        return {}
    invested = sum(s["cost"] for s in per_stock)
    exit_cash = sum(s["cost"] * (1 + s["net_return_pct"] / 100) for s in per_stock)
    # cash_in 已含买入佣金 → 闲置现金就是没买进去、也没花成佣金的那部分
    fees = sum(max(s["cost"] * FEE_RATE, MIN_FEE) for s in per_stock) + \
        sum(max(s["cost"] * (1 + s["net_return_pct"] / 100) * (FEE_RATE + STAMP_RATE), MIN_FEE) for s in per_stock)
    rets = np.array([s["net_return_pct"] / 100 for s in per_stock], dtype=float)
    years = (exit_day - entry).days / 365.25
    port_total = exit_cash / CAPITAL - 1.0
    return {
        "entry": entry.strftime("%Y-%m-%d"), "exit": exit_day.strftime("%Y-%m-%d"), "years": round(years, 2),
        "n_picks": len(per_stock), "skipped_by_lot_or_data": skipped,
        "invested": round(invested, 2),        "idle_cash_pct": round((CAPITAL - cash_in) / CAPITAL * 100, 2),
        "fees": round(fees, 2),
        "stock_hit_rate_pct": round(float((rets > 0).mean()) * 100, 1),
        "stock_median_pct": round(float(np.median(rets)) * 100, 2),
        "stock_p10_pct": round(float(np.percentile(rets, 10)) * 100, 2),
        "stock_worst_pct": round(float(rets.min()) * 100, 2),
        "portfolio_total_pct": round(port_total * 100, 2),
        "portfolio_annualized_pct": round(((1 + port_total) ** (1 / years) - 1) * 100, 2) if port_total > -1 else None,
        "portfolio_win": bool(port_total > 0),
        "vanished_mid_hold": vanished,
        "stocks": per_stock,
    }


def etf_benchmark(entry: pd.Timestamp, hold_years: int, codes: list[str], weights: dict) -> dict:
    """宽基/主仓买入持有（前复权含分红），同入场同出场日。用比值口径（§12.81），不另计现金红利。"""
    frames = {}
    for c in codes:
        p = os.path.join(ETF_DIR, f"{c}.parquet")
        if not os.path.exists(p):
            return {}
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["date"])
        s = pd.Series(pd.to_numeric(d["close"], errors="coerce").values, index=d["date"], dtype=float)
        frames[c] = s[s.notna() & (s > 0)]
    cal = frames[codes[0]].index
    target_exit = entry + pd.DateOffset(years=hold_years)
    ex = cal[cal <= target_exit]
    if len(ex) == 0:
        return {}
    exit_day = ex[-1]
    port = 0.0
    legs = {}
    for c in codes:
        s = frames[c]
        s = s[(s.index >= entry) & (s.index <= exit_day)]
        if len(s) < 2:
            return {}
        r = float(s.iloc[-1] / s.iloc[0] - 1)
        legs[c] = round(r * 100, 2)
        port += weights[c] * r
    years = (exit_day - entry).days / 365.25
    return {"entry": entry.strftime("%Y-%m-%d"), "exit": exit_day.strftime("%Y-%m-%d"),
            "legs_pct": legs, "portfolio_total_pct": round(port * 100, 2),
            "portfolio_annualized_pct": round(((1 + port) ** (1 / years) - 1) * 100, 2)}


# ---------------------------------------------------------------- main

def audit_adj_chain() -> dict:
    """自证：close 是未复权、pct_chg 才含分红（否则整份研究的收益口径是假的）。"""
    f = os.path.join(BAOSTOCK_DIR, "600000_SH.parquet")
    d = pd.read_parquet(f, columns=["date", "close", "pct_chg"])
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    raw_ret = pd.to_numeric(d["close"], errors="coerce").pct_change()
    adj_ret = pd.to_numeric(d["pct_chg"], errors="coerce") / 100.0
    gap = (adj_ret - raw_ret).dropna()
    big = gap[gap.abs() > 0.0005]
    return {"code": "600000.SH", "days": int(len(gap)), "gap_days": int(len(big)),
            "gap_mean_pct_on_gap_days": round(float(big.mean()) * 100, 3),
            "verdict": "close 未复权、pct_chg 含分红" if len(big) >= 2 else "存疑：没找到除息日"}


def haircut_one(cohort: dict, q: float, mode: str) -> dict | None:
    """把 q 比例的持仓强制成 HAIRCUT_RETURN（退市情景），重算命中率与年化。"""
    stocks = cohort.get("stocks") or []
    n = len(stocks)
    if n == 0:
        return None
    m = max(1, int(round(q * n)))
    pairs = [(s["cost"], s["net_return_pct"] / 100.0) for s in stocks]
    if mode == "悲观":
        idx = sorted(range(n), key=lambda i: -pairs[i][1])[:m]      # 挑最赚钱的换成 -70%
    else:
        idx = [int(i * n / m) for i in range(m)]                     # 等距挑 → 无偏近似
    for j in idx:
        pairs[j] = (pairs[j][0], HAIRCUT_RETURN)
    exit_cash = sum(c * (1 + r) for c, r in pairs)
    years = max(cohort["years"], 0.25)
    port = exit_cash / CAPITAL - 1.0
    return {"hit_rate_pct": float(sum(1 for _, r in pairs if r > 0) / n * 100),
            "annualized_pct": ((1 + port) ** (1 / years) - 1) * 100 if port > -1 else -100.0}


def main() -> int:
    sys.stdout.flush()
    audit = audit_adj_chain()
    print("=== 收益口径自证 ===")
    print(json.dumps(audit, ensure_ascii=False))
    if audit["gap_days"] < 2:
        raise SystemExit("收益口径无法确认是否含分红 —— 研究先别跑")

    panel = load_panel()
    roe_map = load_roe()
    print(f"池: {panel['adj'].shape[1]} 只 × {len(panel['adj'].index)} 个交易日 "
          f"({panel['adj'].index[0].date()} → {panel['adj'].index[-1].date()})")

    results: dict = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "audit_adj_chain": audit, "capital": CAPITAL,
                     "caliber": {
                         "收益": "baostock pct_chg 递推链（含分红）",
                         "估值": "baostock pe_ttm/pb_mrq（入场日当时值）",
                         "质量": "financial_ak roe（披露日 ≤ 入场日）",
                         "宇宙": "存续股，剔 ST/停牌/次新（<120 bar）/买不起整手的",
                         "退市": "池内 0 只退市股 → 命中率是上界；另给折价敏感性",
                         "成本": "佣金万3最低5元 + 卖出印花税0.05%，整数手100股"},
                     "rows": [], "baselines": [], "pick_examples": []}

    for k in KS:
        cap = CAPITAL / k / LOT
        for hold_name, h in HOLDS.items():
            for rule_name in RULES:
                cohorts = []
                uni_sizes = []
                for y in ENTRY_YEARS:
                    cal = panel["adj"].index
                    entry_days = cal[cal.year == y]
                    if len(entry_days) == 0:
                        continue
                    entry = entry_days[0]
                    if entry + pd.DateOffset(years=h) > cal[-1]:
                        continue
                    for seed in ((1, 2, 3) if rule_name.startswith("随机") else (7,)):
                        df = features_at(panel, entry, roe_map, price_cap=cap, seed=seed)
                        uni_sizes.append(len(df))
                        codes = pick(df, RULES[rule_name], k)
                        res = settle(panel, entry, h, codes, k)
                        if not res:
                            continue
                        res["seed"] = seed
                        res["universe"] = len(df)
                        if y == ENTRY_YEARS[0] and hold_name == "3y":
                            results["pick_examples"].append(
                                {"rule": rule_name, "k": k, "entry": res["entry"],
                                 "codes": [s["code"] for s in res["stocks"]][:12]})
                        cohorts.append(res)
                if not cohorts:
                    continue
                tot = np.array([c["portfolio_total_pct"] / 100 for c in cohorts])
                ann = np.array([c["portfolio_annualized_pct"] for c in cohorts], dtype=float)
                hit = np.array([c["stock_hit_rate_pct"] for c in cohorts], dtype=float)
                # 退市折价敏感性：池内 0 只退市股 → 命中率是上界。强制 q 比例的持仓为 -70%，
                # 给两个界：悲观（挑表现最好的那几只换成 -70%）与随机（等距挑）。
                hc = {}
                for q in HAIRCUT_SCENARIOS:
                    for mode in ("随机", "悲观"):
                        h2, a2 = [], []
                        for c in cohorts:
                            v = haircut_one(c, q, mode)
                            if v:
                                h2.append(v["hit_rate_pct"]); a2.append(v["annualized_pct"])
                        hc[f"q{int(q*1000)/10}pp_{mode}"] = {
                            "hit_rate_pct": round(float(np.mean(h2)), 1) if h2 else None,
                            "annualized_pct": round(float(np.mean(a2)), 2) if a2 else None}
                results["rows"].append({
                    "k": k, "hold": hold_name, "rule": rule_name, "cohorts": len(cohorts),
                    "price_cap": round(cap, 1),
                    "eligible_universe_median": int(np.median(uni_sizes)),
                    "hit_rate_mean_pct": round(float(hit.mean()), 1),
                    "hit_rate_min_pct": round(float(hit.min()), 1),
                    "portfolio_win_rate_pct": round(float(np.mean([c["portfolio_win"] for c in cohorts])) * 100, 0),
                    "annualized_mean_pct": round(float(ann.mean()), 2),
                    "annualized_min_pct": round(float(ann.min()), 2),
                    "worst_single_pct": round(float(min(c["stock_worst_pct"] for c in cohorts)), 2),
                    "p10_single_pct": round(float(np.median([c["stock_p10_pct"] for c in cohorts])), 2),
                    "median_single_pct": round(float(np.median([c["stock_median_pct"] for c in cohorts])), 2),
                    "idle_cash_pct_mean": round(float(np.mean([c["idle_cash_pct"] for c in cohorts])), 2),
                    "fees_total": round(float(sum(c["fees"] for c in cohorts)), 1),
                    "delisting_haircut": hc,
                    "cohorts_detail": [{kk: vv for kk, vv in c.items() if kk != "stocks"} for c in cohorts],
                })
        # ETF 基准（同入场日/同出场日）
        for hold_name, h in HOLDS.items():
            for y in ENTRY_YEARS:
                cal = panel["adj"].index
                eds = cal[cal.year == y]
                if len(eds) == 0 or eds[0] + pd.DateOffset(years=h) > cal[-1]:
                    continue
                b1 = etf_benchmark(eds[0], h, ["510300"], {"510300": 1.0})
                b2 = etf_benchmark(eds[0], h, ["515450", "511220"], {"515450": 0.5, "511220": 0.5})
                if b1:
                    results["baselines"].append({"bench": "510300 买入持有", "k": k, "hold": hold_name, **b1})
                if b2:
                    results["baselines"].append({"bench": "主仓 515450+511220 50/50 买入持有", "k": k, "hold": hold_name, **b2})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n→ {OUT}")

    print("\n=== 命中率与年化（本金 9 万，等权整数手，持有期不动）===")
    for k in KS:
        for hold_name in HOLDS:
            sub = [r for r in results["rows"] if r["k"] == k and r["hold"] == hold_name]
            if not sub:
                continue
            print(f"\n[K={k} 单价上限 {CAPITAL/k/LOT:.0f} 元 | 持有 {hold_name}]")
            print("  " + " | ".join(x.rjust(11) for x in ("规则", "期数", "命中率", "最差命中", "组合胜率",
                                                          "年化均值", "年化最差", "最差单只", "闲置现金",
                                                          "退市折3pp后命中率", "退市折3pp后年化")))
            for r in sorted(sub, key=lambda x: -x["annualized_mean_pct"]):
                hcp = r["delisting_haircut"]["q3.0pp_悲观"]
                print("  " + " | ".join(str(v).rjust(11) for v in (
                    r["rule"][:10], r["cohorts"], f"{r['hit_rate_mean_pct']:.1f}%", f"{r['hit_rate_min_pct']:.1f}%",
                    f"{r['portfolio_win_rate_pct']:.0f}%", f"{r['annualized_mean_pct']:.2f}%",
                    f"{r['annualized_min_pct']:.2f}%", f"{r['worst_single_pct']:.1f}%", f"{r['idle_cash_pct_mean']:.1f}%",
                    f"{hcp['hit_rate_pct']:.1f}%", f"{hcp['annualized_pct']:.2f}%")))
            for bench in {b["bench"] for b in results["baselines"] if b["k"] == k and b["hold"] == hold_name}:
                bs = [b for b in results["baselines"] if b["bench"] == bench and b["k"] == k and b["hold"] == hold_name]
                anns = [b["portfolio_annualized_pct"] for b in bs]
                wins = sum(1 for b in bs if b["portfolio_total_pct"] > 0)
                print("  " + " | ".join(str(v).rjust(11) for v in (
                    bench[:10], len(bs), "-", "-", f"{wins/len(bs)*100:.0f}%",
                    f"{np.mean(anns):.2f}%", f"{min(anns):.2f}%", "-", "-", "-")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
