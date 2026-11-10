"""市值分层报告：25f vs 随机对照（多种子）+ 国债压舱组合 + ETF 锚。

口径全部对齐闸门后 lag1（信号日次一交易日成交），窗口 2023-05-12→2026-05-08。
随机对照必须多种子取均值 —— 单种子的夏普极差实测 0.19~0.46，单种子对照不可判（§12.69）。
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data/strategy_baseline")
ETF_DIR = os.path.join(ROOT, "data/etf/parquet")
A, B = "2023-05-12", "2026-05-08"
TIERS = {"low": "低盘 <q25", "mid": "中盘 q25-q75", "high": "高盘 >q75", "": "全宇宙"}
ETF_ANCHORS = {"515450": "红利低波50", "511010": "国债ETF", "510300": "沪深300ETF", "510500": "中证500ETF"}


def load(fn):
    return json.load(open(os.path.join(D, fn)))


def curve(d):
    return pd.Series({str(a)[:10]: float(b) for a, b in d["equity_curve"]}).sort_index()


def etf_curve(code):
    d = pd.read_parquet(os.path.join(ETF_DIR, f"{code}.parquet"))
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d = d[(d["date"] >= A) & (d["date"] <= B)]
    p = pd.to_numeric(d["close"], errors="coerce")
    ok = np.isfinite(p) & (p > 0)
    s = pd.Series(p[ok].values, index=d.loc[ok, "date"]).sort_index()
    return s / s.iloc[0]


def metrics_from_ret(r, n):
    ann = float((1 + r).prod() ** (244.0 / n) - 1)
    vol = float(r.std(ddof=1) * math.sqrt(244))
    nav = (1 + r).cumprod()
    return ann, vol, (ann / vol if vol > 0 else float("nan")), float((nav / nav.cummax() - 1).min())


def blend(s_norm, bond_norm, w=0.6):
    """60/40 年初再平衡（每年首个交易日拉回目标权重）——比日频静态权重可执行。"""
    idx = s_norm.index
    b = bond_norm.reindex(idx).astype(float).ffill()
    rs = s_norm.pct_change(fill_method=None).fillna(0.0)
    rb = b.pct_change(fill_method=None).fillna(0.0)
    rebal = {idx[0]} | {g.iloc[0] for _, g in s_norm.groupby(pd.to_datetime(idx).year)}
    cur_s, cur_b, vals = w, 1.0 - w, []
    for dt in idx:
        if dt in rebal and dt != idx[0]:
            tot = cur_s + cur_b
            cur_s, cur_b = tot * w, tot * (1 - w)
        cur_s *= 1 + float(rs.loc[dt])
        cur_b *= 1 + float(rb.loc[dt])
        vals.append(cur_s + cur_b)
    nav = pd.Series(vals, index=idx)
    r = nav.pct_change(fill_method=None).fillna(0.0)
    return metrics_from_ret(r, len(r)) + (float(nav.iloc[-1] - 1),)


def main():
    bond = etf_curve("511010")
    print("=== 1) 25f（因子线）× 市值层，闸门后 lag1 ===")
    print(f"{'层':16s} {'宇宙':>6s} {'夏普':>8s} {'累计':>9s} {'年化':>8s} {'波动':>8s} {'回撤':>9s} {'笔数':>6s}")
    for t, label in TIERS.items():
        tag = f"_cap{t}" if t else "_gate"
        d = load(f"ablation_2026-11-06_honest_25f{tag}_lag1.json")
        print(f"{label:16s} {d['__meta__']['universe_after_gate']:6d} {d['sharpe_ratio']:8.4f} "
              f"{d['total_return']*100:+8.2f}% {d['annualized_return']*100:+7.2f}% "
              f"{d['volatility']*100:7.2f}% {d['max_drawdown']*100:+8.2f}% {d['total_trades']:6d}")

    print("\n=== 2) 每层随机对照 3 种子：Δ 是不是噪声 ===")
    print(f"{'层':16s} {'25f':>7s} {'随机均值':>9s} {'随机极差':>9s} {'均值标准误':>10s} {'Δ夏普':>8s} {'约几σ':>7s} {'随机回撤均值':>12s}")
    for t in ["low", "mid", "high"]:
        f = load(f"ablation_2026-11-06_honest_25f_cap{t}_lag1.json")["sharpe_ratio"]
        runs = [load(p) for p in sorted(glob.glob(os.path.join(D, f"ablation_2026-11-06_random_ctrl_cap{t}*_lag1.json")))]
        sh = [r["sharpe_ratio"] for r in runs]
        dd = [r["max_drawdown"] for r in runs]
        m = st.mean(sh)
        se = st.stdev(sh) / math.sqrt(len(sh)) if len(sh) > 1 else float("nan")
        delta = f - m
        print(f"{TIERS[t]:16s} {f:7.4f} {m:9.4f} {max(sh)-min(sh):9.4f} {se:10.4f} {delta:+8.4f} "
              f"{(delta/se if se == se and se > 0 else float('nan')):7.2f} {st.mean(dd)*100:+11.2f}%")

    print("\n=== 3) 半年切段（25f vs 该层随机种子#1）：超额是不是只有一段行情 ===")
    SEG = [("2023H2", "2023-05-12", "2023-12-31"), ("2024H1", "2024-01-01", "2024-06-30"),
           ("2024H2", "2024-07-01", "2024-12-31"), ("2025H1", "2025-01-01", "2025-06-30"),
           ("2025H2", "2025-07-01", "2025-12-31"), ("2026H1", "2026-01-01", "2026-05-08")]
    print(f"{'层':10s} " + " | ".join(f"{n:>12s}" for n, _, _ in SEG) + "   25f跑赢段数")
    for t in ["low", "mid", "high"]:
        s = curve(load(f"ablation_2026-11-06_honest_25f_cap{t}_lag1.json"))
        r = curve(load(f"ablation_2026-11-06_random_ctrl_cap{t}_lag1.json"))
        cells, wins = [], 0
        for _, a, b in SEG:
            def seg(x):
                x = x.loc[(x.index >= a) & (x.index <= b)]
                return x.iloc[-1] / x.iloc[0] - 1
            fs, fr = seg(s), seg(r)
            if fs > fr:
                wins += 1
            cells.append(f"{fs*100:>+6.1f}/{fr*100:>+5.1f}")
        print(f"{t:10s} " + " | ".join(cells) + f"   {wins}/6   （25f/随机 同段收益%）")

    print("\n=== 4) 因子线 + 40% 国债（年初再平衡）vs ETF 锚（同窗口同口径）===")
    print(f"{'候选':28s} {'累计':>9s} {'年化':>8s} {'波动':>8s} {'夏普':>8s} {'回撤':>9s}")
    for t, label in TIERS.items():
        tag = f"_cap{t}" if t else "_gate"
        s = curve(load(f"ablation_2026-11-06_honest_25f{tag}_lag1.json"))
        s = s / s.iloc[0]
        ann, vol, shp, dd, tot = blend(s, bond)
        print(f"{label+' +40%国债':28s} {tot*100:+8.2f}% {ann*100:+7.2f}% {vol*100:7.2f}% {shp:8.4f} {dd*100:+8.2f}%")
    for code, name in ETF_ANCHORS.items():
        c = etf_curve(code)
        r = c.pct_change(fill_method=None).fillna(0.0)
        ann, vol, shp, dd = metrics_from_ret(r, len(r))
        print(f"{name:28s} {(c.iloc[-1]-1)*100:+8.2f}% {ann*100:+7.2f}% {vol*100:7.2f}% {shp:8.4f} {dd*100:+8.2f}%")
    for w, label in [(0.6, "红利低波50 +40%国债"), (0.5, "红利低波50 +50%国债")]:
        ann, vol, shp, dd, tot = blend(etf_curve("515450"), bond, w)
        print(f"{label:28s} {tot*100:+8.2f}% {ann*100:+7.2f}% {vol*100:7.2f}% {shp:8.4f} {dd*100:+8.2f}%")


if __name__ == "__main__":
    main()
