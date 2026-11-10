"""成本与市值归因：把外部结论落到自己的数据上（台账 §12.68 用）。

外部依据：
  1) 冲击平方根定律  ΔP/P = σ · γ · sqrt(Q/V)   (Kyle 1985; Almgren et al. 2005)
  2) 小盘股单边总成本 27–125 bps；业界普遍报告实盘夏普比回测低 30–50%
  3) 引擎当前成本口径：滑点固定 0.1%（10bp）+ 佣金 0.03% + 印花税 0.1% 卖出

本地数据：baostock_daily 有 amount(元) / turn(换手率%) / pct_chg / volume →
  成交额/换手率 = 价格×自由流通股本 = 自由流通市值（一个纯推导，不需要额外数据）
"""
import os
import json, math, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BS = os.path.join(ROOT, "data/parquet/baostock_daily")
D = os.path.join(ROOT, "data/strategy_baseline")


def load_bs(code: str) -> pd.DataFrame | None:
    f = os.path.join(BS, code.replace(".", "_") + ".parquet")
    if not os.path.exists(f):
        return None
    d = pd.read_parquet(f)
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return d


def unit_check() -> None:
    """先验单位：amount 应为元、volume 应为股（amount/volume ≈ close）。不通过就别用。"""
    d = load_bs("000001.SZ")
    s = d.tail(200)
    ratio = (s["amount"] / s["volume"]).astype(float) / s["close"].astype(float)
    med = float(np.median(ratio))
    print(f"[单位校验] 000001.SZ amount/(volume×close) 中位数 = {med:.4f}（应≈1）")
    if not 0.7 < med < 1.4:
        raise SystemExit(f"单位假设不成立（{med:.4f}），本脚本结论作废")


def main():
    unit_check()

    lines = {
        "25f（闸门后）": "ablation_2026-11-06_honest_25f_gate_lag1.json",
        "25f+中性（闸门后）": "ablation_2026-11-06_honest_25f_neu_gate_lag1.json",
        "随机 TopK60（闸门后）": "ablation_2026-11-06_random_ctrl_gate_lag1.json",
    }

    # ---- 宇宙市值分布（用全池 baostock 窗口内中位 amount/turn 估自由流通市值）----
    cap_of: dict[str, pd.DataFrame] = {}

    def caps(code):
        if code not in cap_of:
            d = load_bs(code)
            if d is None:
                cap_of[code] = None
            else:
                t = pd.to_numeric(d["turn"], errors="coerce")
                a = pd.to_numeric(d["amount"], errors="coerce")
                d["mcap"] = np.where(t > 0, a / (t / 100.0), np.nan)
                d["sigma_day"] = pd.to_numeric(d["pct_chg"], errors="coerce").abs() / 100.0
                cap_of[code] = d.set_index("date")
        return cap_of[code]

    gate = json.load(open(os.path.join(D, "universe_gate_2026-11-06.json")))
    uni = gate["kept_codes"]
    med_cap = {}
    for i, c in enumerate(uni):
        d = caps(c)
        if d is None:
            continue
        v = d.loc[(d.index >= "2023-05-12") & (d.index <= "2026-05-08"), "mcap"].dropna()
        if len(v):
            med_cap[c] = float(np.median(v))
    caps_arr = np.array(sorted(med_cap.values()))
    print(f"\n[宇宙市值] 可估 {len(caps_arr)} 只｜自由流通市值中位数 {np.median(caps_arr)/1e8:.1f} 亿 "
          f"｜25分位 {np.percentile(caps_arr,25)/1e8:.1f} 亿｜75分位 {np.percentile(caps_arr,75)/1e8:.1f} 亿")

    print("\n=== A. 冲击成本核算（平方根定律，γ=0.7） ===")
    print(f"{'线':22s} {'笔数':>6s} {'平均单笔':>10s} {'订单/日成交':>12s} {'估算单边冲击':>13s} {'引擎假设':>10s} {'3年成本差':>11s}")
    gamma = 0.7
    for name, fn in lines.items():
        d = json.load(open(os.path.join(D, fn)))
        th = d["trade_history"]
        rows, miss = [], 0
        for t in th:
            code = t.get("stock_code") or t.get("code")
            date = str(t.get("timestamp") or t.get("date") or t.get("trade_date"))[:10]
            qty = float(t.get("quantity", 0) or 0)
            px = float(t.get("price", 0) or 0)
            Q = qty * px
            sf = caps(code)
            if sf is None or date not in sf.index or Q <= 0:
                miss += 1
                continue
            row = sf.loc[date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            V = float(row["amount"]); sg = float(row["sigma_day"])
            if not np.isfinite(V) or V <= 0 or not np.isfinite(sg) or sg <= 0:
                miss += 1
                continue
            frac = Q / V
            impact = sg * gamma * math.sqrt(frac)
            rows.append((frac, impact, Q))
        if not rows:
            print(f"{name:22s} 无可核算交易"); continue
        a = np.array(rows)
        avgQ = a[:, 2].mean()
        medfrac = float(np.median(a[:, 0]))
        # 用逐笔真算的成本（不是中位数×笔数），再换成等效单边 bps
        tot_cost = float((a[:, 2] * a[:, 1]).sum())
        tot_notional = float(a[:, 2].sum())
        eff_bps = tot_cost / tot_notional * 1e4
        engine_bps = 10.0
        init = float(d.get("initial_cash") or 1_000_000)
        eng_cost = sum(float(t.get("commission", 0) or 0) + float(t.get("slippage_cost", 0) or 0) for t in th)
        print(f"{name:22s} {len(a):6d} {avgQ:10,.0f} {medfrac*100:11.3f}% {eff_bps:12.1f}bp {engine_bps:9.1f}bp "
              f"{(tot_cost - tot_notional*engine_bps/1e4)/init*100:+10.2f}%账户")
        print(f"   引擎已扣成本 {(eng_cost/init*100):.2f}% 账户｜平方根模型应扣 {(tot_cost/init*100):.2f}% "
              f"→ 少扣 {(tot_cost-eng_cost)/init*100:+.2f}pp（3 年累计）")
        if miss:
            print(f"   （{miss} 笔无法核算：无风控行或成交额/波动缺失）")

        # B. 持仓市值画像
        held = [c for c in {t.get("stock_code") or t.get("code") for t in th} if c in med_cap]
        hc = np.array([med_cap[c] for c in held])
        pct = [float((caps_arr[:, None] < x).mean() * 100) for x in hc]
        small = float(np.mean([p < 25 for p in pct])) * 100
        tiny = float(np.mean([p < 10 for p in pct])) * 100
        print(f"   交易过的 {len(hc)} 只：市值中位数 {np.median(hc)/1e8:.1f} 亿 "
              f"（宇宙中位 {np.median(caps_arr)/1e8:.1f} 亿）｜落在宇宙最小25%的分位 {small:.1f}% "
              f"｜落在最小10% {tiny:.1f}%")


if __name__ == "__main__":
    main()
