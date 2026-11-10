"""转债腿可行性研究（台账 §12.79 的 C 方案）。

要回答的唯一问题：把转债作为一条腿加进现有的 ETF 主仓（515450+511220 五五开），
在 3 年窗口里能不能改善风险调整收益与回撤；以及它和主仓/宽基的相关性有多低。

明确不做：不碰转股价值/溢价率（缺转股价调整史，见 §12.78），因此这不是双低策略，
只是「价格 + 剩余期限」两条最朴素的可成交腿，用来判断值不值得为它补事件数据。

口径（全部写死，不许中途换）：
- 可交易宇宙 = 当日有真实 bar 的可转债（instrument_class=="可转债"）。停牌/摘牌/老三板自动出局。
- 无 bar 最多平 20 个交易日（≈1 个月），超过则该券当日不贡献 —— 不许把死券救活。
- lag1：选券用调仓日前一交易日收盘，收益从调仓日当天起算。
- 月频调仓（每月第一个交易日），等权。
- 成本：佣金 max(0.0003, 5 元/单笔金额) + 滑点 0.001，转债无印花税。
- 夏普 = mean(日收益)/std(日收益)*sqrt(252)，rf=0（与平台 portfolio_manager 同公式）。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CAPITAL = 90_000.0
COMMISSION = 0.0003
MIN_COMMISSION = 5.0
SLIPPAGE = 0.001
FFILL_CAP = 20           # 无 bar 最多平 20 个交易日
LOW_PRICE_N = 30         # 低价腿取价格最低的 30 只
MAX_YEARS_LEFT = 3.0     # 剩余期限闸门（契约事实，不是未来信息）
# 默认窗口 = 台账基准窗口（§12.61 起所有结论都用这段），可 --start/--end 换窗做稳健性
START = "2023-05-12"
END = "2026-05-08"
CORE_ETFS = ("515450", "511220")
EQUITY_PROXY = "510300"

CB_DIR = "data/cb"
ETF_DIR = "data/etf/parquet"
OUT_JSON = f"{CB_DIR}/cb_leg_study.json"


# ── 数据装配 ─────────────────────────────────────────────────────────
def load_cb_panel(bars: pd.DataFrame, univ: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """转债收盘价面板 + 可成交掩码。非可转债一律不进面板。"""
    keep = set(univ.loc[univ.instrument_class == "可转债", "code"])
    b = bars[bars.code.isin(keep)].copy()
    b["date"] = pd.to_datetime(b["date"])
    close = b.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    mask = b.pivot_table(index="date", columns="code", values="close", aggfunc="size").notna()
    return close.sort_index(), mask.sort_index()


def load_etf_close(codes: tuple[str, ...]) -> pd.DataFrame:
    out = {}
    for c in codes:
        df = pd.read_parquet(f"{ETF_DIR}/{c}.parquet")
        s = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
        out[c] = s
    return pd.DataFrame(out).sort_index()


def master_calendar(cb_dates: pd.DatetimeIndex, etf_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """主日历取 ETF 侧：腿能不能照着下单取决于 ETF 有交易的日子。"""
    return etf_dates[(etf_dates >= pd.Timestamp(START)) & (etf_dates <= pd.Timestamp(END))]


def aligned_close_panel(close: pd.DataFrame, mask: pd.DataFrame,
                        cal: pd.DatetimeIndex) -> pd.DataFrame:
    """把转债收盘价对齐到主日历，停牌只允许平 FFILL_CAP 个交易日。

    直接 ffill(limit=...) 会把「已摘牌但登记表没日期」的券无限平下去，
    所以先按真实 bar 掩码把超过闸门的天置 NaN，再向前平。
    """
    px = close.reindex(cal.union(close.index)).ffill(limit=FFILL_CAP)
    px = px.reindex(cal)
    known = mask.reindex(cal).fillna(False)
    carried = 0
    out = px.copy()
    out[:] = np.nan
    live: dict[str, tuple[float, int]] = {}
    for i, d in enumerate(cal):
        for code in px.columns:
            v = px.at[d, code]
            if known.at[d, code] and not pd.isna(v):
                live[code] = (float(v), 0)
                out.at[d, code] = float(v)
            elif code in live:
                price, gap = live[code]
                if gap + 1 > FFILL_CAP:
                    del live[code]
                else:
                    live[code] = (price, gap + 1)
                    out.at[d, code] = price
    return out


def years_left(expire: pd.Timestamp, d: pd.Timestamp) -> float:
    if pd.isna(expire):
        return np.inf
    return (expire - d).days / 365.25


def select_leg(px_row: pd.Series, expire: dict[str, pd.Timestamp], d: pd.Timestamp,
               mode: str, leg_weight: float = 1.0,
               known_row: pd.Series | None = None) -> list[str]:
    """调仓日选券。只用当日（含）之前可观察到的信息。

    手数闸门：1 手 = 10 张 = 1000 元面值，成交价 p 时单手占用 10*p 元。
    腿资金 = CAPITAL*leg_weight，则持仓只数上限受「每只至少 1 手」约束。
    """
    # 只有当日真的有成交记录（真实 bar）的券才有资格入选：
    # 被 ffill 平出来的价格不是可成交价，用它选券 = 买进已经停牌的死券
    if known_row is not None:
        avail = [c for c in px_row.index if not pd.isna(px_row[c]) and bool(known_row.get(c, False))]
    else:
        avail = [c for c in px_row.index if not pd.isna(px_row[c])]
    budget = CAPITAL * leg_weight

    def apply_lot_gate(pool: list[str]) -> list[str]:
        picked: list[str] = []
        for c in pool:
            per_pos = budget / (len(picked) + 1)
            if 10.0 * float(px_row[c]) > per_pos:
                break
            picked.append(c)
        return picked

    if mode == "all":
        # 全宇宙等权是「转债指数」参照物，不是可成交腿：9 万账户连 1 手都铺不开，
        # 手数可行性在 main 里单独披露，不在这里偷偷裁剪宇宙。
        return sorted(avail)
    if mode == "low_price":
        near = [c for c in avail if years_left(expire.get(c, pd.NaT), d) <= MAX_YEARS_LEFT]
        pool = sorted(near, key=lambda c: px_row[c])[:LOW_PRICE_N]
        return apply_lot_gate(pool)        # 价格升序，越后面单手越贵
    raise ValueError(f"未知腿: {mode}")


# ── 组合与指标 ───────────────────────────────────────────────────────
def rebalance_days(cal: pd.DatetimeIndex) -> list[pd.Timestamp]:
    days, seen = [], set()
    for d in cal:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            days.append(d)
    return days


def leg_daily_returns(px: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """给定逐日权重（行=调仓日，只在该日到下一次调仓日前有效）与收盘价，出腿的日收益。"""
    ret = px.pct_change(fill_method=None)
    out = pd.Series(0.0, index=px.index)
    for code in px.columns:
        col = ret[code]
        w = weights[code]
        out += (col * w).fillna(0.0)
    return out


def build_leg(px: pd.DataFrame, expire: dict[str, pd.Timestamp], cal: pd.DatetimeIndex,
              mode: str, leg_weight: float = 1.0,
              known: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[float]]:
    """返回 (逐日权重面板, 毛收益序列, 每次调仓持仓只数, 每次调仓换手)。lag1：用前一日收盘选券。"""
    reb = rebalance_days(cal)
    rows = {}
    turnovers = []
    prev_w: dict[str, float] = {}
    for i, d in enumerate(reb):
        pick_day = cal[max(0, cal.get_loc(d) - 1)]        # lag1
        codes = select_leg(px.loc[pick_day], expire, pick_day, mode, leg_weight,
                           None if known is None else known.loc[pick_day])
        if not codes:
            rows[d] = {}
            turnovers.append(0.0)
            prev_w = {}
            continue
        w = {c: 1.0 / len(codes) for c in codes}
        rows[d] = w
        # 双边换手（买入+卖出都计佣金），从零建仓 = 1.0
        turnover = sum(abs(w.get(c, 0.0) - prev_w.get(c, 0.0)) for c in set(w) | set(prev_w))
        turnovers.append(turnover)
        prev_w = w
    weight_panel = pd.DataFrame(0.0, index=cal, columns=px.columns)
    for i, d in enumerate(reb):
        end = reb[i + 1] if i + 1 < len(reb) else cal[-1] + pd.Timedelta(days=1)
        span = weight_panel.loc[(weight_panel.index >= d) & (weight_panel.index < end)]
        for c, wv in rows[d].items():
            weight_panel.loc[span.index, c] = wv
    gross = leg_daily_returns(px, weight_panel)
    counts = pd.Series({d: len(rows[d]) for d in reb})
    return weight_panel, gross, counts, turnovers


def annual_rebalanced_core(etf: pd.DataFrame, cal: pd.DatetimeIndex,
                           weights: tuple[float, ...] = (0.5, 0.5)) -> pd.Series:
    """主仓 = 两只 ETF 等权，仅在每年第一个交易日再平衡（§12.77 口径）。"""
    ret = etf.reindex(cal).pct_change(fill_method=None).fillna(0.0)
    w = pd.Series(0.0, index=cal, dtype=float)                    # 主仓内 515450 的权重
    cur = weights[0]
    for i, d in enumerate(cal):
        if i == 0 or d.year != cal[i - 1].year:
            cur = weights[0]
        w.loc[d] = cur
        r0 = float(ret.iloc[i][etf.columns[0]])
        r1 = float(ret.iloc[i][etf.columns[1]])
        if i + 1 < len(cal):
            nav0 = 1.0 + r0
            new_w = cur * nav0 / (cur * nav0 + (1 - cur) * (1 + r1))
            cur = new_w
    out = w * ret.iloc[:, 0] + (1 - w) * ret.iloc[:, 1]
    return out


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    return float((nav / peak - 1).min())


def perf(daily: pd.Series) -> dict:
    daily = daily.dropna()
    nav = (1 + daily).cumprod()
    yrs = len(daily) / 252.0
    ann = float(nav.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan
    sd = float(daily.std(ddof=1))
    return {
        "days": int(len(daily)),
        "total_return_pct": round(float(nav.iloc[-1] - 1) * 100, 2),
        "annualized_pct": round(ann * 100, 2),
        "ann_vol_pct": round(sd * math.sqrt(252) * 100, 2),
        "sharpe": round(float(daily.mean() / sd * math.sqrt(252)), 4) if sd > 0 else None,
        "max_drawdown_pct": round(max_drawdown(nav) * 100, 2),
    }


def abrupt_exit_events(known: pd.DataFrame, weight_panel: pd.DataFrame,
                       reb: list[pd.Timestamp], cal: pd.DatetimeIndex) -> list[dict]:
    """找出「持仓期内券消失」事件：某券在某个持仓区间内没有任何真实 bar。

    这类券在净值里被按最后一根 bar 的收盘价退出（最多平 20 天），
    真实世界里它多半是强赎（回收≈最后价）或违约摘牌（回收打折）。
    必须逐笔列出来，因为低价腿最容易踩到的正是违约券。
    """
    events = []
    codes = list(weight_panel.columns)
    for i, d in enumerate(reb):
        end = reb[i + 1] if i + 1 < len(reb) else cal[-1] + pd.Timedelta(days=1)
        span = weight_panel.loc[(weight_panel.index >= d) & (weight_panel.index < end)]
        if span.empty:
            continue
        sub = known.reindex(span.index).reindex(columns=codes).fillna(False)
        for code in codes:
            w = float(weight_panel.at[d, code])
            if w <= 0:
                continue
            if not sub[code].any():          # 整个持仓区间没有一根真实 bar
                events.append({"exit_date": str(d.date()), "code": code, "weight": round(w, 4)})
    return events


def apply_recovery_haircut(gross: pd.Series, events: list[dict], weight_in_account: float,
                           recovery_ratio: float) -> pd.Series:
    """把消失券按回收率打折：假设最后可见价 = 面值附近，真实回收 = ratio × 最后价。"""
    out = gross.copy()
    for e in events:
        d = pd.Timestamp(e["exit_date"])
        if d in out.index:
            out.loc[d] -= e["weight"] * weight_in_account * (1.0 - recovery_ratio)
    return out


def half_year_sharpes(daily: pd.Series) -> dict:
    """按自然半年分段算夏普——单窗数字必须先看分段（§12.65 教训）。"""
    idx = daily.index
    key = idx.year * 10 + ((idx.month - 1) // 6 + 1)
    out = {}
    for k, seg in daily.groupby(key):
        sd = seg.std(ddof=1)
        if len(seg) > 30 and sd > 0:
            y, h = divmod(int(k), 10)
            out[f"{str(y)[2:]}H{h}"] = round(float(seg.mean() / sd * math.sqrt(252)), 4)
    return out


def main() -> None:
    global START, END, OUT_JSON
    ap = argparse.ArgumentParser(description="转债腿可行性研究（台账 §12.79）")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    START, END, OUT_JSON = args.start, args.end, args.out

    bars = pd.read_parquet(f"{CB_DIR}/cb_daily_bars.parquet")
    univ = pd.read_parquet(f"{CB_DIR}/cb_universe.parquet")
    close_raw, mask = load_cb_panel(bars, univ)
    etf = load_etf_close(CORE_ETFS + (EQUITY_PROXY,))
    cal = master_calendar(close_raw.index, etf.index)
    known = mask.reindex(cal).reindex(columns=close_raw.columns).fillna(False)
    px = aligned_close_panel(close_raw, mask, cal)
    expire = {r.code: pd.to_datetime(r.expire_date) for r in univ.itertuples()
              if getattr(r, "expire_date", None) and str(r.expire_date) != "NaT"}

    # 宇宙规模闸门
    counts_all = px.notna().sum(axis=1)
    reb = rebalance_days(cal)
    print(f"窗口 {cal[0].date()} → {cal[-1].date()} 共 {len(cal)} 个主日历日 / {len(reb)} 次月调仓")
    print(f"转债宇宙只数：中位 {int(counts_all.median())}，最少 {int(counts_all.min())}，最多 {int(counts_all.max())}")
    if counts_all.median() < 100:
        raise SystemExit(f"闸门：可交易转债宇宙中位只有 {int(counts_all.median())} 只，样本不足以出结论")

    def build_one(mode: str, w: float):
        """按账户权重建腿：手数闸门与佣金门槛都随 w 变，所以腿要逐权重重建。"""
        wpanel, gross, counts, turnovers = build_leg(px, expire, cal, mode, leg_weight=w,
                                                     known=known)
        gross = gross.reindex(cal).fillna(0.0)
        leg_cost = pd.Series(0.0, index=cal)
        for d, t, n in zip(reb, turnovers, counts):
            per_pos = CAPITAL * w / max(int(n), 1)
            comm = max(COMMISSION, MIN_COMMISSION / max(per_pos, 1.0))
            leg_cost.loc[d] = t * (comm + SLIPPAGE)
        return wpanel, gross, counts, turnovers, leg_cost

    # 主仓口径与 §12.77 一致：515450 50% + 511220 50%，只在每年第一个交易日再平衡
    core_ret = annual_rebalanced_core(etf, cal)
    equity = etf[EQUITY_PROXY].pct_change().reindex(cal).fillna(0.0)

    report: dict = {"window": {"start": str(cal[0].date()), "end": str(cal[-1].date()),
                               "days": int(len(cal)),
                               "cb_universe_median_count": int(counts_all.median())},
                    "cost_model": {"capital": CAPITAL, "commission": COMMISSION,
                                   "min_commission": MIN_COMMISSION, "slippage": SLIPPAGE,
                                   "stamp_tax": 0.0, "lot_rule": "1 手 = 10 张 = 1000 元面值",
                                   "note": "转债无印花税；佣金按 5 元最低门槛逐笔算"},
                    "legs": {}, "combos": {}, "half_year_sharpe": {}, "stress": {}}

    core_net = core_ret
    core_perf = perf(core_net)
    report["legs"]["core_etf_50_50"] = {**core_perf, "avg_holdings": 2,
                                        "corr_with_equity": round(float(np.corrcoef(core_net, equity)[0, 1]), 4)}
    report["half_year_sharpe"]["core_etf_50_50"] = half_year_sharpes(core_net)
    report["legs"]["equity_510300"] = perf(equity)

    legs_store = {}
    for mode in ("all", "low_price"):
        for w in (0.10, 0.20, 0.30, 0.50, 1.00):
            key = f"{mode}@{int(w*100)}"
            wpanel, gross, counts, turnovers, leg_cost = build_one(mode, w)
            net = gross - leg_cost
            # 手数可行性：该腿若要每只持 1 手（10 张）需要多少钱 vs 腿实际拿到的钱
            need = []
            for d in reb:
                held = [c for c in px.columns if float(wpanel.at[d, c]) > 0]
                row = px.loc[cal[max(0, cal.get_loc(d) - 1)]].reindex(held).dropna()
                need.append(float((10.0 * row).sum()))
            need_mean = float(np.mean(need)) if need else 0.0
            leg_budget = CAPITAL * w
            executable = bool(need_mean <= leg_budget)
            if not executable:
                leg_cost = pd.Series(0.0, index=cal)   # 不可执行的腿只作指数参照，不计成本
                net = gross
            legs_store[key] = (wpanel, gross, counts, turnovers, leg_cost)
            report["legs"][key] = {
                **perf(net),
                "可执行性": "可成交" if executable else "不可成交（每只1手就超出腿资金，仅作指数参照）",
                "每只1手所需资金": round(need_mean),
                "腿资金": round(leg_budget),
                "手数可行": bool(need_mean <= leg_budget),
                "avg_holdings": int(counts.mean()),
                "min_holdings": int(counts.min()),
                "双边换手_每次调仓pct": round(float(np.mean(turnovers)) * 100, 2),
                "双边换手_年pct": round(float(np.mean(turnovers)) * 12 * 100, 1),
                "cost_drag_pp_total": round(float(leg_cost.sum()) * 100, 2),
                "corr_with_core": round(float(np.corrcoef(net, core_net)[0, 1]), 4),
                "corr_with_equity": round(float(np.corrcoef(net, equity)[0, 1]), 4),
            }
            report["half_year_sharpe"][key] = half_year_sharpes(net)
            combo = (1 - w) * core_net + w * net
            cp = perf(combo)
            report["combos"][f"core{int((1-w)*100)}_cb_{mode}{int(w*100)}"] = {
                **cp,
                "sharpe_gain_vs_core": round(cp["sharpe"] - core_perf["sharpe"], 4),
                "dd_gain_vs_core_pp": round(cp["max_drawdown_pct"] - core_perf["max_drawdown_pct"], 2),
                "holdings_at_this_weight": int(counts.mean()),
                "手数可行": executable,
                "成本拖累pct_占账户": round(float(leg_cost.sum()) * w * 100, 2),
                "half_year_sharpe": half_year_sharpes(combo),
            }

    # 违约回收压力测试：持仓区间内完全没有真实 bar 的券，按回收率打折
    for key in ("low_price@20", "low_price@30", "all@100"):
        wpanel, gross, counts, turnovers, leg_cost = legs_store[key]
        events = abrupt_exit_events(known, wpanel, reb, cal)
        wgt = {"low_price@20": 0.20, "low_price@30": 0.30, "all@100": 1.0}[key]
        mode_name = key.split("@")[0]
        combo_key = f"core{int((1-wgt)*100)}_cb_{mode_name}{int(wgt*100)}"
        entry = {"事件笔数": len(events),
                 "事件全表": events,
                 "事件样例": events[:8],
                 "组合夏普_不打折": report["combos"][combo_key]["sharpe"],
                 "组合回撤_不打折pct": report["combos"][combo_key]["max_drawdown_pct"]}
        for ratio in (0.7, 0.5):
            net = apply_recovery_haircut(gross - leg_cost, events, 1.0, ratio)
            combo = (1 - wgt) * core_net + wgt * net
            entry[f"回收{int(ratio*100)}成"] = {
                "腿夏普": perf(net)["sharpe"],
                "组合夏普": perf(combo)["sharpe"],
                "组合回撤pct": perf(combo)["max_drawdown_pct"],
                "组合年化pct": perf(combo)["annualized_pct"],
            }
        report["stress"][key] = entry

    # 四道门槛（§12.66 口径）：三档同向 / 分段不劣于核心-0.05 的段数 / 领先 > 0.18 噪声
    verdict = {}
    for mode in ("all", "low_price"):
        trio = [report["combos"][f"core{int((1-w)*100)}_cb_{mode}{int(w*100)}"]["sharpe_gain_vs_core"]
                for w in (0.10, 0.20, 0.30)]
        best_w = max((0.10, 0.20, 0.30), key=lambda w: report["combos"][f"core{int((1-w)*100)}_cb_{mode}{int(w*100)}"]["sharpe"])
        seg_combo = report["combos"][f"core{int((1-best_w)*100)}_cb_{mode}{int(best_w*100)}"]["half_year_sharpe"]
        seg_core = report["half_year_sharpe"]["core_etf_50_50"]
        segs = [k for k in seg_combo if k in seg_core]
        wins = sum(1 for k in segs if seg_combo[k] >= seg_core[k] - 0.05)
        verdict[mode] = {
            "门槛1_三档夏普同向增益": {"gains": trio, "通过": all(g > 0 for g in trio)},
            "门槛2_分段不劣于核心": {"通过段数": f"{wins}/{len(segs)}",
                              "通过": wins >= math.ceil(len(segs) * 2 / 3)},
            "门槛3_领先超过噪声0.18": {"最大增益": max(trio),
                               "通过": max(trio) > 0.18},
        }
    report["四道门槛判定"] = verdict

    Path(OUT_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({"legs": report["legs"], "combos": {k: {kk: v[kk] for kk in
          ("annualized_pct", "sharpe", "max_drawdown_pct", "sharpe_gain_vs_core",
           "成本拖累pct_占账户", "手数可行")}
          for k, v in report["combos"].items()},
          "stress": report["stress"], "四道门槛判定": verdict},
          ensure_ascii=False, indent=1))
    print("\n产物:", OUT_JSON)


if __name__ == "__main__":
    main()
