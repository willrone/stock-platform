#!/usr/bin/env python3
"""每日交易信号: honest_composite (42f 等权复合) → TopK-Dropout 当日买卖清单。

用途: 收盘后跑一遍, 输出"今天该买/卖/留哪 60 只", 直接可操作。
口径与正式策略 honest_composite 完全一致:
  - 23 个诚实公式因子 (MIN/MAX/STD/CORR/CORD × n∈{5,10,20,30,60})
  - 17 个 Alpha158 面板因子 (data/qlib_data/features/day/)
  - 等权截面 z-score, 涨停掩码 (当日涨幅 >= 9.5% 不可买, 分数置 NaN)
  - TopK=60, n_drop=5: 今天进 top60 的是买入, 掉出的是卖出, 留下的是持有。

数据要求: 先用 tdx_pipeline 拉当日数据 + merge_to_stock_data 落 parquet。
输出: 终端表格 + data/strategy_baseline/daily_signal_<date>.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STOCK_DATA_DIR = os.path.join(ROOT, "data", "parquet", "stock_data")
PANEL_DIR = os.path.join(ROOT, "data", "qlib_data", "features", "day")
OUT_DIR = os.path.join(ROOT, "data", "strategy_baseline")

LIMIT_UP = 0.095
LOOKBACK = 80  # 交易日, 覆盖 n=60 的滚动窗
MIN_CROSS = 30

# 与 honest_composite_strategy 完全一致的因子定义
_HONEST_FORMULAS: dict[str, float] = {}
for _n in (5, 10, 20, 30, 60):
    _HONEST_FORMULAS[f"MIN{_n}"] = 1.0
    _HONEST_FORMULAS[f"MAX{_n}"] = -1.0
    _HONEST_FORMULAS[f"STD{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORR{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORD{_n}"] = -1.0

_ALPHAPANEL_FACTORS: dict[str, float] = {
    "VSUMP5": -1.0, "VSUMP10": -1.0, "VSUMP20": -1.0,
    "VSUMN5": 1.0, "VSUMN10": 1.0, "VSUMN20": 1.0,
    "VSUMD5": -1.0, "VSUMD10": -1.0, "VSUMD20": -1.0,
    "QTLD5": 1.0, "QTLD10": 1.0, "QTLD20": 1.0,
    "IMAX5": -1.0, "IMIN5": -1.0, "WVMA5": -1.0,
    "VMA5": 1.0, "VSTD5": -1.0,
}


def list_universe() -> list[str]:
    """所有有 stock_data parquet 的 A 股代码。

    2026-10-25 修正：原来只判断 `.parquet` 后缀，把
    `000001.parquet` / `000858.parquet` 这类**无交易所后缀的旧命名残留**
    也收进了股票池（stock_data 里同时存在 `000001_SZ.parquet` 和
    `000001.parquet` 两套命名）。后果：
      - 股票池多出重复标的（同一只股票被算两次）
      - 旧的 `000001.parquet` 数据只到 2023-06-30，之后全是 NaN
      - 污染横截面统计（市场宽度、等权指数波动率 → 直接影响
        gbm_voltgt 的波动率目标仓位）

    现在只接受 `<code>_<SH|SZ|BJ>.parquet`，交易所后缀是硬要求。
    """
    codes = []
    for f in sorted(os.listdir(STOCK_DATA_DIR)):
        if not f.endswith(".parquet"):
            continue
        stem = f[: -len(".parquet")]
        # 必须是 <6位数字>_<SH|SZ|BJ> 形式
        if "_" not in stem:
            continue
        code, _, market = stem.partition("_")
        if market not in ("SH", "SZ", "BJ") or not code.isdigit():
            continue
        codes.append(f"{code}.{market}")
    return codes


def load_data(codes: list[str]) -> tuple[dict, dict, pd.DatetimeIndex]:
    """加载 close/high/low/volume, 返回 (frames, panel, master)。"""
    t0 = time.time()
    frames: dict[str, pd.DataFrame] = {}
    panel: dict[str, pd.DataFrame | None] = {}
    for c in codes:
        safe = c.replace(".", "_")
        try:
            df = pd.read_parquet(os.path.join(STOCK_DATA_DIR, f"{safe}.parquet"),
                                 columns=["date", "open", "high", "low", "close", "volume"])
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        if len(df) < 60:
            continue
        frames[c] = df
        try:
            pf = pd.read_parquet(os.path.join(PANEL_DIR, f"{safe}.parquet"))
            pf = pf.reset_index().set_index("date").sort_index()
            pf = pf[~pf.index.duplicated(keep="first")]
            panel[c] = pf
        except Exception:
            panel[c] = None
    codes = list(frames)
    master = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
    print(f"[load] {len(codes)} stocks, {len(master)} days, "
          f"panel={sum(1 for c in codes if panel.get(c) is not None)} {time.time()-t0:.0f}s", flush=True)
    return frames, panel, master


def to_mat(frames: dict, master: pd.DatetimeIndex, col: str) -> pd.DataFrame:
    n_d, n_s = len(master), len(frames)
    codes = list(frames)
    out = np.full((n_d, n_s), np.nan)
    for j, c in enumerate(codes):
        pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
        ok = pos >= 0
        out[pos[ok], j] = pd.to_numeric(frames[c][col], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan)[ok]
    return pd.DataFrame(out, index=master, columns=codes)


def to_panel_mat(panel: dict, master: pd.DatetimeIndex, col: str) -> np.ndarray:
    codes = list(panel)
    out = np.full((len(master), len(codes)), np.nan)
    for j, c in enumerate(codes):
        if panel.get(c) is None:
            continue
        pos = master.get_indexer(pd.DatetimeIndex(panel[c].index))
        ok = pos >= 0
        out[pos[ok], j] = pd.to_numeric(panel[c][col], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan)[ok]
    return out


def zscore_add(acc: np.ndarray, wsum: np.ndarray, mat: np.ndarray,
               sign: float, min_cross: int) -> None:
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan) * sign
    z = np.where(cnt >= min_cross, z, np.nan)
    good = np.isfinite(z)
    acc[good] += np.nan_to_num(z)[good]
    wsum[good] += 1


def compute_composite(frames: dict, panel: dict, master: pd.DatetimeIndex) -> pd.DataFrame:
    """返回 composite DataFrame (index=date, columns=code), 已应用涨停掩码。"""
    n_d, n_s = len(master), len(frames)
    close = to_mat(frames, master, "close")
    vol = to_mat(frames, master, "volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close.pct_change(fill_method=None)
        vchg = vol.pct_change(fill_method=None)
        cands: dict[str, pd.DataFrame] = {}
        for n in (5, 10, 20, 30, 60):
            cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
            cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
            cands[f"STD{n}"] = close.rolling(n).std() / close
            cands[f"CORR{n}"] = close.rolling(n).corr(vol)
            cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)

    acc = np.zeros((n_d, n_s))
    wsum = np.zeros((n_d, n_s))
    for fname, sign in _HONEST_FORMULAS.items():
        zscore_add(acc, wsum, cands[fname].to_numpy(), sign, MIN_CROSS)
    for fname, sign in _ALPHAPANEL_FACTORS.items():
        zscore_add(acc, wsum, to_panel_mat(panel, master, fname), sign, MIN_CROSS)

    composite = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    limit_up = close.pct_change(fill_method=None).to_numpy() >= LIMIT_UP
    composite[limit_up] = np.nan  # 涨停掩码: 当日不可买
    return pd.DataFrame(composite, index=master, columns=list(frames))


def main() -> None:
    ap = argparse.ArgumentParser(description="honest_composite 每日买卖信号")
    ap.add_argument("--topk", type=int, default=30, help="持仓数量 (默认 30)")
    ap.add_argument("--ndrop", type=int, default=3, help="每天最多换几只 (默认 3)")
    args = ap.parse_args()
    topk, ndrop = args.topk, args.ndrop

    codes = list_universe()
    frames, panel, master = load_data(codes)
    comp = compute_composite(frames, panel, master)
    comp = comp.tail(LOOKBACK)
    today = comp.index[-1]
    prev = comp.index[-2]
    close_today = to_mat(frames, master, "close").tail(2).iloc[-1]

    # 排名 (分数降序), 涨停掩码已把涨停票分数置 NaN → 自动落榜不可买
    rank_today = comp.iloc[-1].dropna().sort_values(ascending=False)
    rank_prev = comp.iloc[-2].dropna().sort_values(ascending=False)
    top_today = list(rank_today.index[:topk])
    top_prev = list(rank_prev.index[:topk])

    # TopK-Dropout 执行: 持仓 = top_prev。今天掉出 topk 的持仓里,
    # 只卖最差的 ndrop 只; topk 里新进的, 只买最强的 ndrop 只。
    held = set(top_prev)
    fell_out = [c for c in top_prev if c not in set(top_today)]   # 掉出持仓的
    new_in = [c for c in top_today if c not in held]              # 新进 topk 的
    sells = fell_out[:ndrop]
    buys = new_in[:ndrop]
    holds = [c for c in top_today if c in held and c not in sells]

    result = {
        "meta": {
            "date": str(today.date()), "prev_date": str(prev.date()),
            "n_stocks": len(frames), "topk": topk, "n_drop": ndrop,
            "max_position_size": round((1.0 - 0.05) / topk, 6),
            "generated_at": datetime.now().isoformat(),
        },
        "buys": [{"code": c, "score": round(float(rank_today[c]), 4),
                  "close": round(float(close_today[c]), 2)} for c in buys],
        "sells": [{"code": c, "prev_score": round(float(rank_prev[c]), 4)} for c in sells],
        "holds": [{"code": c, "score": round(float(rank_today[c]), 4)} for c in holds],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"daily_signal_{today.date()}_topk{topk}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)

    print(f"\n{'='*60}", flush=True)
    print(f"每日信号  {today.date()}  (vs {prev.date()})", flush=True)
    print(f"宇宙 {len(frames)} 只 | 持仓 top{topk} | 每天最多换 {ndrop} 只 "
          f"| 单仓上限 {(1.0-0.05)/topk*100:.1f}%", flush=True)
    print(f"\n◆ 买入 ({len(buys)} 只)", flush=True)
    for b in result["buys"]:
        print(f"  {b['code']}  score={b['score']:+.3f}  收盘 {b['close']}", flush=True)
    print(f"\n◆ 卖出 ({len(sells)} 只)", flush=True)
    for s in result["sells"]:
        print(f"  {s['code']}  前日 score={s['prev_score']:+.3f}", flush=True)
    print(f"\n◆ 持有 ({len(holds)} 只)", flush=True)
    print("  " + "  ".join(holds[:14]) + (" ..." if len(holds) > 14 else ""), flush=True)
    print(f"\n[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
