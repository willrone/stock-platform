"""自由流通市值估算 + 分层（给回测做"我是不是在做 size β"的判定用）。

推导（不需要任何额外数据源，只用 baostock 日线已有的两列）：
    换手率 turn(%) = 成交量 / 自由流通股本
    成交额 amount(元) = 成交量 × 价格
    ⇒ 自由流通市值 = 价格 × 自由流通股本 = amount / (turn / 100)

口径声明：
- 取窗口内逐日估值的**中位数**（避开单日异动与涨跌停日的失真）。
- 单位在加载时校验：`amount / (volume × close)` 必须 ≈ 1，否则直接抛错而不是产出错数。
- 这是"估算"，不是官方流通市值；用途只做**相对分层**（ quartile 分位 ），不做绝对值结论。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

TIERS = ("low", "mid", "high")
UNIT_RATIO_TOL = (0.7, 1.4)  # amount/(volume×close) 的可接受区间


def _parquet(data_root: str, code: str) -> str:
    return os.path.join(
        data_root, "parquet", "baostock_daily", code.replace(".", "_") + ".parquet"
    )


def estimate_caps(
    data_root: str,
    start: str,
    end: str,
    codes: list[str],
    cache_path: str | None = None,
) -> dict[str, float]:
    """code -> 窗口内自由流通市值中位数（元）。命中缓存则直接读缓存。"""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        hit = {c: cached["caps"][c] for c in codes if c in cached.get("caps")}
        if len(hit) >= int(len(codes) * 0.9):
            return hit

    caps: dict[str, float] = {}
    ratios: list[float] = []
    for code in codes:
        path = _parquet(data_root, code)
        if not os.path.exists(path):
            continue
        d = pd.read_parquet(path)
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if d.empty:
            continue
        amount = pd.to_numeric(d["amount"], errors="coerce")
        volume = pd.to_numeric(d["volume"], errors="coerce")
        close = pd.to_numeric(d["close"], errors="coerce")
        turn = pd.to_numeric(d["turn"], errors="coerce")
        ok = (
            np.isfinite(amount)
            & np.isfinite(volume)
            & np.isfinite(close)
            & (amount > 0)
            & (volume > 0)
            & (close > 0)
        )
        if ok.sum() >= 5:
            ratios.append(float(np.median((amount[ok] / (volume[ok] * close[ok])))))
        good = ok & np.isfinite(turn) & (turn > 0)
        if good.sum() < 5:
            continue
        est = amount[good] / (turn[good] / 100.0)
        est = est[np.isfinite(est) & (est > 0)]
        if len(est):
            caps[code] = float(np.median(est))

    if ratios:
        med = float(np.median(ratios))
        if not (UNIT_RATIO_TOL[0] <= med <= UNIT_RATIO_TOL[1]):
            raise ValueError(
                f"单位假设不成立：amount/(volume×close) 中位数={med:.4f}（应在 {UNIT_RATIO_TOL}）"
                "——字段口径变了，市值估算作废，别用这套分层结论"
            )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "built_at": datetime.now().isoformat(timespec="seconds"),
                    "window": [start, end],
                    "method": "median(amount / (turn/100))",
                    "n_codes": len(caps),
                    "caps": caps,
                },
                f,
            )
    return caps


def tier_bounds(caps: dict[str, float]) -> tuple[float, float]:
    """返回 (q25, q75)：low = <q25，high = >q75，mid = 中间。宇宙太小则报错。"""
    if len(caps) < 100:
        raise ValueError(f"可估市值的票太少（{len(caps)}），分层无意义")
    vals = np.array(sorted(caps.values()))
    return float(np.percentile(vals, 25)), float(np.percentile(vals, 75))


def filter_tier(codes: list[str], caps: dict[str, float], tier: str) -> list[str]:
    """tier=low/mid/high；查不到市值的票一律排除（沿用闸门原则：查不到=排除，不是放行）。"""
    if tier not in TIERS:
        raise ValueError(f"未知 tier={tier}，可选 {TIERS}")
    q25, q75 = tier_bounds(caps)
    out: list[str] = []
    for c in codes:
        v = caps.get(c)
        if v is None:
            continue
        if tier == "low" and v < q25:
            out.append(c)
        elif tier == "high" and v > q75:
            out.append(c)
        elif tier == "mid" and q25 <= v <= q75:
            out.append(c)
    return out


def tier_stats(caps: dict[str, float]) -> dict[str, float]:
    vals = np.array(sorted(caps.values()))
    return {
        "n": int(len(vals)),
        "median_yi": round(float(np.median(vals)) / 1e8, 2),
        "q25_yi": round(float(np.percentile(vals, 25)) / 1e8, 2),
        "q75_yi": round(float(np.percentile(vals, 75)) / 1e8, 2),
    }
