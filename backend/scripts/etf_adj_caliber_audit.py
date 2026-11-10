"""ETF 复权口径审计（§12.80 前置审计）。

台账 §12.70 写着"ETF 不复权价 → 分红未计入 → 历史收益低估 6.6pp"，
而 etf_data_pipeline.py 写"主源腾讯 fqkline，qfq 前复权"。同一份文档自相矛盾。
主仓（515450/511220）的所有收益数字都挂在这个序列上，所以先用外部同源双口径对齐定论：

  东财 push2his kline: fqt=0 不复权 / fqt=1 前复权
判据：存量文件与哪个口径逐位一致，口径就是哪个。

顺带量化混锚风险：前复权每次分红都会把历史整体重锚，
增量拉取的序列如果锚在不同日期，历史段就会留下系统性偏差。
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UA = {"User-Agent": "Mozilla/5.0"}
SECID = {"515450": "1.515450", "511220": "1.511220", "510300": "1.510300"}
WINDOW = ("2023-05-12", "2026-05-08")


def em_kline(secid: str, fqt: int) -> pd.DataFrame:
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
           "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
           f"&klt=101&fqt={fqt}&beg={WINDOW[0].replace('-','')}&end=20261110")
    obj = json.loads(urllib.request.urlopen(
        urllib.request.Request(url, headers=UA)).read().decode())
    rows = (obj.get("data") or {}).get("klines") or []
    out = []
    for r in rows:
        p = r.split(",")
        out.append({"date": p[0], "close": float(p[2])})
    if not out:
        raise RuntimeError(f"东财无数据 secid={secid} fqt={fqt}")
    return pd.DataFrame(out)


def stored(code: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/etf/parquet" / f"{code}.parquet")
    df = df.copy()
    df["date"] = df["date"].astype(str).str[:10]
    return df[["date", "close"]].sort_values("date")


def stored_windowed(code: str) -> pd.DataFrame:
    st = stored(code)
    return st[(st["date"] >= WINDOW[0]) & (st["date"] <= WINDOW[1])]


def caliber_gap(code: str) -> dict:
    st = stored_windowed(code)
    res = {}
    for tag, fqt in (("raw_fqt0", 0), ("qfq_fqt1", 1)):
        src = em_kline(SECID[code], fqt)
        m = st.merge(src, on="date", suffixes=("_st", "_src"))
        rel = (m["close_st"] / m["close_src"] - 1) * 100
        res[tag] = {
            "n": int(len(m)),
            "abs_max_pp": round(float(rel.abs().max()), 3),
            "mean_pp": round(float(rel.mean()), 3),
            "identical_rows": int((rel.abs() < 1e-6).sum()),
        }
    res["verdict"] = judge_caliber(res["raw_fqt0"]["abs_max_pp"], res["qfq_fqt1"]["abs_max_pp"])
    return res


def judge_caliber(abs_max_raw: float, abs_max_qfq: float,
                  tol: float = 1e-6, ambiguous_tol: float = 1e-9) -> str:
    """复权口径只由"与哪个参考口径更近"决定，不许由脚本注释决定（注释错过一次）。"""
    if abs_max_raw <= tol and abs_max_qfq <= tol:
        return "两口径同价（窗内无复权差异）"
    if abs(abs_max_raw - abs_max_qfq) <= ambiguous_tol:
        raise ValueError(f"口径无法判定：raw={abs_max_raw} qfq={abs_max_qfq} 差异过小")
    return "前复权" if abs_max_qfq < abs_max_raw else "不复权"


def align_window(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把两个序列裁到同一批交易日。
    上一版让 716 天的存量去比 848 天的东财序列，直接造出 -2.65pp 的假"锚点漂移"。"""
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("两个序列没有共同交易日，无法比较")
    return a.loc[common], b.loc[common]


def core_stats(close: pd.DataFrame) -> dict:
    """主仓 50/50，年度再平衡。入参必须是 DatetimeIndex 的两列收盘价。"""
    close = close.sort_index()
    cal = close.index
    ret = close.pct_change(fill_method=None).fillna(0.0)
    nav, w = [], 0.5
    for i, d in enumerate(cal):
        if i == 0 or d.year != cal[i - 1].year:
            w = 0.5
        r0, r1 = float(ret.iloc[i, 0]), float(ret.iloc[i, 1])
        nav.append(1.0 if i == 0 else nav[-1] * (1 + w * r0 + (1 - w) * r1))
        if i + 1 < len(cal):
            w = w * (1 + r0) / (w * (1 + r0) + (1 - w) * (1 + r1))
    s = pd.Series(nav, index=cal)
    assert len(s) > 100, "样本太短不足以谈年化"
    ann = float(s.iloc[-1] ** (252.0 / len(s)) - 1) * 100
    vol = float(s.pct_change().std() * np.sqrt(252)) * 100
    dd = float(((s - s.cummax()) / s.cummax()).min()) * 100
    rf0 = (ann / 100) / (vol / 100)          # 平台/portfolio_manager 口径：rf=0
    rf1 = (ann / 100 - 0.0136) / (vol / 100)  # 扣无风险利率口径
    return {"total_pct": round((s.iloc[-1] - 1) * 100, 2), "annualized_pct": round(ann, 2),
            "vol_pct": round(vol, 2), "sharpe_rf0": round(rf0, 4), "sharpe_rf136": round(rf1, 4),
            "max_dd_pct": round(dd, 2), "n_days": len(s)}


def main() -> None:
    out: dict = {"window": list(WINDOW), "per_code": {}, "core": {}}
    print(f"窗口 {WINDOW[0]} → {WINDOW[1]}\n")
    for code in ("515450", "511220", "510300"):
        out["per_code"][code] = caliber_gap(code)
        g = out["per_code"][code]
        print(f"{code}: 判定 {g['verdict']}  "
              f"vs不复权 abs_max {g['raw_fqt0']['abs_max_pp']:7.3f}pp  "
              f"vs前复权 abs_max {g['qfq_fqt1']['abs_max_pp']:7.3f}pp "
              f"(逐位一致 {g['qfq_fqt1']['identical_rows']}/{g['qfq_fqt1']['n']})")

    print("\n主仓口径对照（同窗口、同年度再平衡）")
    combos = {}
    stored_core = pd.DataFrame({
        c: pd.Series(stored_windowed(c)["close"].values,
                     index=pd.to_datetime(stored_windowed(c)["date"]))
        for c in ("515450", "511220")}).dropna()
    for tag in ("stored", "em_qfq_fqt1", "em_raw_fqt0"):
        frames = []
        for code in ("515450", "511220"):
            if tag == "stored":
                df = stored_windowed(code)
            else:
                fqt = 1 if tag.endswith("fqt1") else 0
                df = em_kline(SECID[code], fqt)
            s = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
            frames.append(s)
        px = pd.DataFrame({c: f for c, f in zip(("515450", "511220"), frames)}).dropna()
        px = px.loc[WINDOW[0]:WINDOW[1]]   # 三方必须同窗口，否则"漂移"是窗口差造出来的
        if tag != "stored":
            px, _ = align_window(px, stored_core)
        combos[tag] = core_stats(px)
        combos[tag]["window"] = [str(px.index[0].date()), str(px.index[-1].date())]
        print(f"  {tag:>12}: 总收益 {combos[tag]['total_pct']:6.2f}%  年化 {combos[tag]['annualized_pct']:6.2f}%  "
              f"夏普(rf=0) {combos[tag]['sharpe_rf0']:6.4f}  夏普(rf1.36%) {combos[tag]['sharpe_rf136']:6.4f}  "
              f"回撤 {combos[tag]['max_dd_pct']:7.2f}%  n={combos[tag]['n_days']}")
    out["core"] = combos
    delta = combos["em_qfq_fqt1"]["sharpe_rf0"] - combos["stored"]["sharpe_rf0"]
    out["anchor_drift_sharpe"] = round(delta, 4)
    print(f"\n同窗口下 东财当前前复权 − 存量序列：夏普差 {delta:+.4f}")
    print("→ 结论：主仓序列是前复权（分红已计入）；台账 §12.70 'ETF 不复权价、分红低估 6.6pp' 的说法作废。")

    path = ROOT / "data/etf/adj_caliber_audit.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n→ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
