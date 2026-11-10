#!/usr/bin/env python3
"""P3 勘察 2：债券类候选池 —— 久期/信用结构 + 窗口前流动性 + 窗口前 ρ。

## 为什么要单独勘察债券池

P3 勘察 1 发现 ρ(候选, 510300) 最低的前几名**全是长久期利率债**
（511260/511010/511020，ρ 为负），而当前定稿的 bond 类持有的是
**短融/政金债/短债**（511360/511380/511520）。

如果长债真的更分散，最直接的改法就是**把 bond 类换成久期更长的债**。
但在下结论前必须先验三件事：
1. 这些长债 ETF 在**窗口前**的日均成交额是否够（能不能实盘买到）？
2. 债券池里到底有多少只可用的候选（够不够做随机对照）？
3. 「久期长的债 ρ 更低」是**资产性质**（长债 = 利率避险），还是**碰巧选中了这几只**？
   → 必须用同池随机对照区分。

## 口径

- ρ：窗口**开始前** 250 个交易日 vs 510300（代理），严格 point-in-time
- 流动性：窗口**开始前** 120 个交易日日均成交额（volume × close × 100）
- 窗口：main 2023-05-08 / oos 2021-05-11

## 用法
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/p3_bond_recon.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
sys.path.insert(0, BACKEND_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ETF_DIR = os.path.join(ROOT, "data", "etf")
PARQUET_DIR = os.path.join(ETF_DIR, "parquet")
PROXY = "510300"

LOOKBACK = 250
LIQ_LOOKBACK = 120
MIN_HISTORY = 200
START_TOLERANCE_DAYS = 7
WINDOWS = {"main": "2023-05-08", "oos": "2021-05-11"}


def load_code(code: str) -> pd.DataFrame:
    path = os.path.join(PARQUET_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df


def rets(df: pd.DataFrame) -> pd.Series:
    return df.set_index("date")["close"].pct_change().dropna()


def main() -> None:
    universe = json.load(open(os.path.join(ETF_DIR, "etf_universe.json"), encoding="utf-8"))
    etfs = universe["etfs"]
    proxy = load_code(PROXY)
    proxy_ret = rets(proxy)
    have_data = {os.path.basename(p)[:-8]
                 for p in glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))}

    out: Dict[str, object] = {"windows": {}}
    for wname, wstart in WINDOWS.items():
        ws = pd.Timestamp(wstart)
        w0 = ws - pd.Timedelta(days=START_TOLERANCE_DAYS)
        pre = proxy_ret[proxy_ret.index < ws]
        if len(pre) < MIN_HISTORY:
            print(f"{wname}: 代理窗口前数据不足")
            continue
        pre = pre.iloc[-LOOKBACK:]

        rows: List[Dict[str, object]] = []
        for code in sorted(have_data):
            tags = etfs.get(code, {}).get("tags", [])
            if "bond" not in tags:
                continue
            df = load_code(code)
            if df.empty:
                continue
            if df["date"].iloc[0] > w0:
                continue
            if df["date"].iloc[-1] < ws:
                continue
            w = df[(df["date"] >= ws) & (df["date"] <= ws + pd.Timedelta(days=1500))]
            if len(w) < MIN_HISTORY:
                continue
            r = rets(df)
            r_pre = r[(r.index >= pre.index[0]) & (r.index <= pre.index[-1])]
            joined = pd.concat([r_pre, pre], axis=1, join="inner").dropna()
            if len(joined) < MIN_HISTORY:
                continue
            a, b = joined.iloc[:, 0], joined.iloc[:, 1]
            if a.std() == 0 or b.std() == 0:
                continue
            rho = float(np.corrcoef(a, b)[0, 1])
            liq_df = df[df["date"] < ws].tail(LIQ_LOOKBACK)
            turnover = float((liq_df["volume"].fillna(0) * liq_df["close"] * 100).mean()) \
                if len(liq_df) else 0.0
            rows.append({
                "code": code, "name": etfs.get(code, {}).get("name", ""),
                "rho": round(rho, 4),
                "vol": round(float(a.std() * np.sqrt(244)), 4),
                "turnover": round(turnover, 0),
                "first": str(df["date"].iloc[0].date()),
                "n": int(len(joined)),
            })

        rows.sort(key=lambda x: x["rho"])
        out["windows"][wname] = rows  # type: ignore[index]
        print(f"\n===== {wname}  窗口起点 {wstart}  债券池合格 {len(rows)} 只 =====")
        print(f"{'code':8s} {'ρ':>8s} {'vol':>7s} {'窗口前日均成交额':>16s}  {'上市日':10s} 名称")
        for r in rows:
            print(f"{r['code']:8s} {r['rho']:+8.4f} {r['vol']:7.2%} "
                  f"{r['turnover']/1e4:14,.0f}万  {r['first']:10s}  {r['name']}")

    path = os.path.join(ROOT, "data", "backtest", "p3_bond_recon.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n✓ 已写入 {path}")


if __name__ == "__main__":
    main()
