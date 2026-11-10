#!/usr/bin/env python3
"""P3 勘察：候选资产池里到底有没有"与 A 股低相关"的资产。

## 为什么先勘察

P2 已经证明 **ρ(袖套, A 股腿) 是组合夏普的决定性变量**（bond 低波但 ρ 0.55/0.62，
按波动率配权 = 反分散）。那自然的追问是：**能不能找到 ρ 更低的新资产？**

但直接"挑 ρ 最低的几只"有两个坑，必须先看清数据再定实验设计：
1. **前视**：ρ 必须用**窗口开始前**的数据算，否则就是未来函数。
2. **赢家选择偏差**：从几百只里挑 ρ 最低的 3 只，本身就是一次极端选择，
   必须配**同池随机对照**（同一候选池随机抽 3 只）才能判断是机制还是运气。

## 关键设计决策：选腿依据用「510300 代理」而不是 A 股腿

A 股腿（honest_composite 25 因子）的面板数据只从 **2021-05-11** 起。
样本外窗口起点正好是 2021-05-11 → **窗口前 250 天没有 A 股腿数据**，
无法在 OOS 窗口做 point-in-time 的 ρ 选腿（用推导窗口算就等于用未来选资产）。

→ 选腿依据改用 **510300（沪深300 ETF，数据可回溯到 2012 年）** 作为 A 股 beta 代理：
窗口前 250 个交易日算 ρ(ETF, 510300)，完全 point-in-time，且两个窗口都能算。
A 股腿只用于**事后验证**（组合 ρ 是否真下降、夏普是否真提升）。

## 本脚本只做勘察，不下结论

输出：候选池规模、ρ 分布、ρ 最低的 50 只都是什么资产、现有三类在池子里的位置。

## 用法
    cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/p3_recon.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
sys.path.insert(0, BACKEND_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ETF_DIR = os.path.join(ROOT, "data", "etf")
PARQUET_DIR = os.path.join(ETF_DIR, "parquet")
PROXY = "510300"  # 沪深300 ETF，作为 A 股 beta 代理

# 窗口起点（与 P2 一致）。ρ 只用窗口**开始前**的数据。
WINDOWS = {
    "main": "2023-05-08",
    "oos": "2021-05-11",
}
LOOKBACK = 250      # 算 ρ 的回看交易日数
MIN_HISTORY = 200   # 窗口前至少要有这么多天数据
START_TOLERANCE_DAYS = 7


def load_code(code: str) -> pd.DataFrame:
    path = os.path.join(PARQUET_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df


def daily_returns(df: pd.DataFrame) -> pd.Series:
    return df.set_index("date")["close"].pct_change().dropna()


def main() -> None:
    universe = json.load(open(os.path.join(ETF_DIR, "etf_universe.json"), encoding="utf-8"))
    names = {k: v.get("name", "") for k, v in universe["etfs"].items()}
    tags = {k: v.get("tags", []) for k, v in universe["etfs"].items()}

    proxy_df = load_code(PROXY)
    if proxy_df.empty:
        print("!! 代理 510300 无数据，无法勘察")
        return
    proxy_ret = daily_returns(proxy_df)

    codes = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    print(f"候选池（有 parquet 数据）: {len(codes)} 只")

    out: Dict[str, object] = {"lookback": LOOKBACK, "proxy": PROXY, "windows": {}}

    for wname, wstart in WINDOWS.items():
        w0 = pd.Timestamp(wstart) - pd.Timedelta(days=START_TOLERANCE_DAYS)
        # 窗口前 LOOKBACK 个交易日（严格 < 窗口起点）
        pre = proxy_ret[proxy_ret.index < pd.Timestamp(wstart)]
        if len(pre) < MIN_HISTORY:
            print(f"  {wname}: 代理在窗口前只有 {len(pre)} 天数据，样本不足")
            continue
        pre = pre.iloc[-LOOKBACK:]

        rows: List[Dict[str, object]] = []
        for code in codes:
            if code == PROXY:
                continue
            df = load_code(code)
            if df.empty:
                continue
            first = df["date"].iloc[0]
            if first > w0:
                continue  # 上市太晚
            if df["date"].iloc[-1] < pd.Timestamp(wstart):
                continue  # 已退市/数据太短
            r = daily_returns(df)
            r = r[(r.index >= pre.index[0]) & (r.index <= pre.index[-1])]
            joined = pd.concat([r, pre], axis=1, join="inner").dropna()
            if len(joined) < MIN_HISTORY:
                continue
            a, b = joined.iloc[:, 0], joined.iloc[:, 1]
            if a.std() == 0 or b.std() == 0:
                continue
            rho = float(np.corrcoef(a, b)[0, 1])
            vol = float(a.std() * np.sqrt(244))
            rows.append({
                "code": code, "name": names.get(code, ""), "tags": tags.get(code, []),
                "rho": round(rho, 4), "vol": round(vol, 4), "n": int(len(joined)),
            })

        rows.sort(key=lambda x: x["rho"])
        out["windows"][wname] = rows  # type: ignore[index]
        print(f"\n===== {wname}  窗口起点 {wstart}  合格候选 {len(rows)} 只 =====")
        print(f"  ρ 分布: 最低 {rows[0]['rho']:.4f} / 中位 {rows[len(rows)//2]['rho']:.4f} "
              f"/ 最高 {rows[-1]['rho']:.4f}")
        print("  ρ 最低的 40 只：")
        for r in rows[:40]:
            print(f"    {r['code']}  ρ={r['rho']:+.4f}  vol={r['vol']:.2%}  {r['name']}")

    path = os.path.join(ROOT, "data", "backtest", "p3_recon.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n✓ 已写入 {path}")


if __name__ == "__main__":
    main()
