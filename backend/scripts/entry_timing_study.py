"""入场节奏实测：一次性 all-in vs 分批（§12.80）。

台账证死了"买什么"（ETF 主仓 515450/511220 五五开，§12.70–§12.77），
但从未量化"怎么进场"。这是 9 万真钱上唯一还没被数字回答过的问题：
一次性建仓在期望上通常更优（钱早点进场吃收益），但它把首笔回撤集中到一个交易日。
小账户真正的死亡方式不是选错资产，而是**进场后第一个月就跌穿心理线然后割肉**。

做法：对每一个可能的建仓日（按月取第一个交易日），用同一套主仓、
同一套手数/佣金口径，比较三种节奏的**未来 12 个月**结果：
  lump  : 建仓日一次买满 50/50
  drip3 : 每 21 个交易日买 1/3（约 3 个月进满）
  drip6 : 每 21 个交易日买 1/6（约 6 个月进满）
未投入的钱按 0 趴在现金里（保守口径，不假设货基收益，不假设抄底）。
每个建仓日都算：期末收益、期间最大回撤、期间最差单日浮亏。
然后看分布，不是看某一次。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ETF_DIR = ROOT / "data/etf/parquet"
CODES = ("515450", "511220")
WEIGHTS = {"515450": 0.50, "511220": 0.50}
CAPITAL = 90_000.0
LOT = 100                      # ETF 1 手 = 100 份
FEE_RATE = 3e-4
MIN_FEE = 5.0
STEP = 21                      # 批次间隔（交易日）
HORIZON = 252                  # 观测期：12 个月
ENTRY_FIRST = "2020-03-02"     # 515450 首根日线 2020-02-26，留出建仓可用日


def load_closes() -> pd.DataFrame:
    out = {}
    for c in CODES:
        df = pd.read_parquet(ETF_DIR / f"{c}.parquet")
        out[c] = pd.Series(df["close"].astype(float).values,
                           index=pd.to_datetime(df["date"].astype(str).str[:10]))
    px = pd.DataFrame(out).sort_index()
    return px.dropna(how="any")


def buy_batch(cash: float, price: float, budget: float) -> tuple[int, float, float]:
    """按预算买整数手；买不满的零头留在现金里，不追价。"""
    lots = int(budget // (price * LOT))
    amount = lots * price * LOT
    if amount <= 0:
        return 0, 0.0, 0.0
    return lots, amount, max(MIN_FEE, FEE_RATE * amount)


def simulate(px: pd.DataFrame, start_i: int, tranches: int, horizon: int) -> dict:
    """观测终点固定 = 建仓日 + horizon，三种节奏共用同一段日历。
    （若按"最后一批 + horizon"算，分批会因为多活了几十天而白赚一段行情——那是口径漏洞。）"""
    dates = px.index
    batch_idx = [start_i + k * STEP for k in range(tranches)]
    end_i = start_i + horizon
    if batch_idx[-1] >= end_i or end_i >= len(dates):
        return {}

    cash = CAPITAL
    shares = {c: 0 for c in CODES}
    fees = 0.0
    nav: list[float] = []
    cursor = start_i

    for idx in batch_idx:
        # 批次之间按市价盯市（未投入部分趴在现金，0 收益）
        for j in range(cursor, idx):
            row = dates[j]
            nav.append(cash + sum(shares[c] * px.at[row, c] for c in CODES))
        budget = CAPITAL / tranches
        for c in CODES:
            price = float(px.at[dates[idx], c])
            lots, amount, fee = buy_batch(cash, price, budget * WEIGHTS[c])
            if amount <= 0:
                continue
            cash -= amount + fee
            fees += fee
            shares[c] += lots * LOT   # 记账用份额，不是手数（漏乘 LOT 会把净值算成千把块）
        cursor = idx + 1

    for j in range(cursor, end_i + 1):
        row = dates[j]
        nav.append(cash + sum(shares[c] * px.at[row, c] for c in CODES))

    arr = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(arr)
    day_change = np.diff(arr) / arr[:-1] if len(arr) > 1 else np.array([0.0])
    return {
        "start": str(dates[start_i].date()),
        "end": str(dates[end_i].date()),
        "invested_pct": round((CAPITAL - cash) / CAPITAL * 100, 2),
        "ret_pct": round((arr[-1] / CAPITAL - 1) * 100, 2),
        "max_dd_pct": round(float(((arr - peak) / peak).min()) * 100, 2),
        "worst_day_pct": round(float(day_change.min()) * 100, 2),
        "days_below_90k": int((arr < CAPITAL).sum()),
        "fees": round(fees, 2),
        "cash_left": round(cash, 2),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ret = np.array([r["ret_pct"] for r in rows])
    dd = np.array([r["max_dd_pct"] for r in rows])
    below = np.array([r["days_below_90k"] for r in rows])
    return {
        "n_entry_dates": len(rows),
        "ret_mean_pp": round(float(ret.mean()), 2),
        "ret_median_pp": round(float(np.median(ret)), 2),
        "ret_p10_pp": round(float(np.percentile(ret, 10)), 2),
        "ret_worst_pp": round(float(ret.min()), 2),
        "ret_best_pp": round(float(ret.max()), 2),
        "share_of_entries_negative": round(float((ret < 0).mean() * 100), 1),
        "dd_median_pp": round(float(np.median(dd)), 2),
        "dd_worst_pp": round(float(dd.min()), 2),
        "median_days_below_90k": int(np.median(below)),
        "max_days_below_90k": int(below.max()),
    }


def main() -> None:
    px = load_closes()
    horizons = {"12m": 252, "36m": 756}

    def monthly_starts(horizon: int) -> list[int]:
        """每个观测期各自取尽可能多的建仓日：12m 视图不必被 36m 的长度截断
        （被截断会让样本偏向早期顺风段，这是上一版的口径漏洞）。"""
        out: list[int] = []
        seen: set[str] = set()
        for i, d in enumerate(px.index):
            if str(d.date()) < ENTRY_FIRST:
                continue
            if i + (5 * STEP) + horizon >= len(px.index):
                continue
            key = str(d.date())[:7]
            if key not in seen:
                seen.add(key)
                out.append(i)
        return out

    print(f"主仓日线 {px.index[0].date()} → {px.index[-1].date()}")

    modes = {"lump": 1, "drip3": 3, "drip6": 6}
    summary, detail = {}, {}
    for hz_name, hz in horizons.items():
        monthly_first = monthly_starts(hz)
        print(f"\n########## 观测期 {hz_name}（{hz} 交易日）｜建仓日 {len(monthly_first)} 个 ##########")
        summary[hz_name], detail[hz_name] = {}, {}
        for label, tranches in modes.items():
            rows = [r for i in monthly_first if (r := simulate(px, i, tranches, hz))]
            summary[hz_name][label] = summarize(rows)
            detail[hz_name][label] = rows
            print(f"\n=== {label}（分 {tranches} 批）===")
            for k, v in summary[hz_name][label].items():
                print(f"  {k:>28}: {v}")

    lump = {r["start"]: r for r in detail["12m"]["lump"]}
    drip = {r["start"]: r for r in detail["12m"]["drip6"]}
    common = sorted(set(lump) & set(drip))
    d_ret = np.array([lump[s]["ret_pct"] - drip[s]["ret_pct"] for s in common])
    d_dd = np.array([lump[s]["max_dd_pct"] - drip[s]["max_dd_pct"] for s in common])
    pairwise = {
        "n": len(common),
        "lump_minus_drip6_ret_mean_pp": round(float(d_ret.mean()), 2),
        "lump_minus_drip6_ret_median_pp": round(float(np.median(d_ret)), 2),
        "lump_minus_drip6_ret_p10_pp": round(float(np.percentile(d_ret, 10)), 2),
        "lump_minus_drip6_ret_p90_pp": round(float(np.percentile(d_ret, 90)), 2),
        "share_lump_return_better": round(float((d_ret > 0).mean() * 100), 1),
        "share_lump_dd_deeper": round(float((d_dd < 0).mean() * 100), 1),
        "lump_dd_worse_by_more_than_3pp_share": round(float((d_dd < -3.0).mean() * 100), 1),
    }
    print("\n=== 同一建仓日配对：lump − drip6 ===")
    for k, v in pairwise.items():
        print(f"  {k:>38}: {v}")

    worst_entries = sorted(detail["12m"]["lump"], key=lambda r: r["ret_pct"])[:8]
    print("\n一次性建仓最难的 8 个进场日：")
    for r in worst_entries:
        print(f"  {r['start']}  12月收益 {r['ret_pct']:>7.2f}%  期间最大回撤 {r['max_dd_pct']:>7.2f}%  "
              f"跌破本金 {r['days_below_90k']} 天")

    payload = {
        "window": [str(px.index[0].date()), str(px.index[-1].date())],
        "capital": CAPITAL, "codes": list(CODES), "weights": WEIGHTS,
        "horizons": horizons, "batch_step_days": STEP,
        "caveats": [
            "ETF 日线是前复权价（§12.80 ⓪ 审计确认）→ 收益已含分红",
            "未投入资金按 0 收益计算（不假设货基/逆回购收益）",
            "未计滑点与冲击成本",
        ],
        "summary": summary, "pairwise": pairwise, "detail_36m_lump": detail["36m"]["lump"],
        "worst_lump_entries": worst_entries,
    }
    out = ROOT / "data/etf/entry_timing_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
