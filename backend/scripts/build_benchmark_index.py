#!/usr/bin/env python3
"""零技巧基准构建器（自用 KPI 的对照组，2026-11-05 立项，建设项 B）。

为什么必须有它：自用版的验收线是"定稿线扣完成本后相对**自己的零技巧基准**超额 ≥ +3pp"。
没有这条基准，"跑赢/跑输"都只是情绪。

口径（写死在这里，别在调用方各自解释）：
  A 股腿基准 = **全池等权买入持有** = mean_i [ P_i(t) / P_i(t0) ]
      - 全池 = `data/parquet/stock_data` ∩ `data/parquet/baostock_daily` 的代码集合
      - P 是**后复权全收益**，用交易所口径涨跌幅链 `pct_chg`（百分数）连乘得到
      - ⚠️ 是"买入持有"（每只各自连乘后取均值），**不是**"每日再平衡等权"（日均收益连乘）。
        两者在长期上差别不小，混用就是口径漂移。
  袖套基准 = 定稿 9 只 ETF 等权买入持有 = mean_j [ P_j(t) / P_j(t0) ]
  母线基准 = 0.5 × A 基准 + 0.5 × 袖套基准（与定稿配置的权重一致）

基准序列只算一次（以 `--start` 为 1.0 的**电平**存盘），
任何线的 t0 都用 `L(t)/L(t0)` 重新锚定 → 不会出现"每条线自己算一遍基准"。

产出：
  data/paper/benchmark_levels.parquet   日期 × 代码 的电平矩阵（float32，start=1.0）
  data/paper/benchmark_sleeve.json      日期 → 袖套基准电平
  data/paper/benchmark_meta.json        池子大小、每日覆盖率、口径声明

⚠️ 已知偏差（必须写出来，不许装作没有）：
  代码中途停牌/退市后不再出现在矩阵里 → 当天的均值只在"还活着"的样本上取
  → **基准被系统性抬高**（退市的票通常先暴跌）→ 策略超额被低估。
  台账 §十一 实测 main 窗口 114/5031 = 2.27% 退市股。所以这道闸同时输出**覆盖率**：
  覆盖率跌破 `--min-coverage` 就把 `coverage_ok=false` 写进 meta 并非零退出。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PCT_DIR = os.path.join(ROOT, "data", "parquet", "baostock_daily")
STOCK_DIR = os.path.join(ROOT, "data", "parquet", "stock_data")
ETF_DIR = os.path.join(ROOT, "data", "etf", "parquet")
OUT_DIR = os.path.join(ROOT, "data", "paper")
LEVELS_FP = os.path.join(OUT_DIR, "benchmark_levels.parquet")
SLEEVE_FP = os.path.join(OUT_DIR, "benchmark_sleeve.json")
META_FP = os.path.join(OUT_DIR, "benchmark_meta.json")

# 定稿袖套（与 multi_asset_service 一致；这里 import 而不是重抄名单，防口径漂移）
sys.path.insert(0, os.path.join(ROOT, "backend"))
try:
    from app.services.trading.multi_asset_service import (  # type: ignore
        SLEEVE_CLASSES, SLEEVE_HELD)
except Exception as exc:  # pragma: no cover - 兜底路径必须显式可见
    print(f"❌ 无法从 multi_asset_service 导入 SLEEVE_HELD：{exc}")
    sys.exit(2)

# SLEEVE_HELD 是 {类: [代码...]}。定稿每类正好 3 只 →
# "类 1/3、类内等权" 与 "9 只等权" 在数学上完全相同，所以基准直接取 9 只均值。
SLEEVE_CODES: List[str] = [c for cls in SLEEVE_CLASSES for c in SLEEVE_HELD[cls]]


def _codes() -> List[str]:
    stock = {os.path.splitext(n)[0] for n in os.listdir(STOCK_DIR) if n.endswith(".parquet")}
    pct = {os.path.splitext(n)[0] for n in os.listdir(PCT_DIR) if n.endswith(".parquet")}
    return sorted(stock & pct)


def build_levels(start: str, incremental: bool) -> pd.DataFrame:
    codes = _codes()
    if not codes:
        print("❌ 全池为空（stock_data ∩ baostock_daily）")
        sys.exit(1)

    prev: pd.DataFrame | None = None
    if incremental and os.path.exists(LEVELS_FP):
        prev = pd.read_parquet(LEVELS_FP)
        prev_start = str(prev.index.min())[:10]
        # 既有起点允许比 --start 晚（元旦/长假不开盘），只要晚得不多。
        # ⚠️ 原来写成 `prev_start > start → 重建`：起点 2026-01-05 vs --start 2026-01-01
        #    天天白白全量重建一次（145 秒）。真正需要重建的只有"要更早的历史"。
        gap_days = (pd.Timestamp(prev_start) - pd.Timestamp(start)).days
        if gap_days > 10:
            print(f"[bench] 既有电平起点 {prev_start} 比 --start {start} 晚 {gap_days} 天"
                  " → 需要更早历史，全量重建")
            prev = None
        else:
            print(f"[bench] 既有电平起点 {prev_start} 已覆盖 --start {start}"
                  f"（差 {gap_days} 个自然日，节假日不开盘）→ 走增量")

    if prev is None:
        print(f"[bench] 全量构建：{len(codes)} 只，起点 {start}", flush=True)
        cols: Dict[str, pd.Series] = {}
        for i, code in enumerate(codes):
            try:
                df = pd.read_parquet(os.path.join(PCT_DIR, f"{code}.parquet"),
                                     columns=["date", "pct_chg"])
            except Exception:
                continue
            df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
            df = df[df["date"] >= pd.Timestamp(start)]
            if df.empty:
                continue
            r = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0   # pct_chg 是百分数
            s = pd.Series(r.values, index=df["date"]).groupby(level=0).last()
            s = s[s.notna()]
            if s.empty:
                continue
            cols[code] = (1.0 + s).cumprod()
            if (i + 1) % 1000 == 0:
                print(f"  ... {i + 1}/{len(codes)}", flush=True)
        if not cols:
            print("❌ 一只都读不出来")
            sys.exit(1)
        lvl = pd.DataFrame(cols).astype("float32").sort_index()
    else:
        lvl = prev
        last_ts = lvl.index.max()
        last = str(last_ts)[:10]
        print(f"[bench] 增量：既有电平到 {last}，{lvl.shape[1]} 只", flush=True)

        # ⚠️ 基准必须锚在**已知完整的那一行** `last_ts`，不能用 `iloc[-1]`。
        # 旧写法 `base = float(lvl[code].iloc[-1])` + 逐格 `lvl.loc[d, code] = ...`：
        # 列序第一个 code 会把新日期那一整行先插出来（整行 NaN），此后每个 code 的
        # `iloc[-1]` 读到的就是那行空值 → base=NaN → 整条链变 NaN。
        # 实测后果：新追加那天 5596 只里只有 1 只有值（000001_SZ），且因为
        # `last = lvl.index.max()` 已是该日，之后每次增量 `date > last` 恒为空
        # → **坏尾巴永不自愈**，所有 excess_pp 都是拿一只平安银行算的。
        # 修法：锚点取 `lvl.at[last_ts, code]`；新值先攒进 dict 再**一次性** concat
        # （逐格 .loc 插入还会让每次赋值复制整张矩阵，5577 次）。
        new_cols: Dict[str, pd.Series] = {}
        for code in lvl.columns:
            try:
                df = pd.read_parquet(os.path.join(PCT_DIR, f"{code}.parquet"),
                                     columns=["date", "pct_chg"])
            except Exception:
                continue
            df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
            df = df[df["date"] > last_ts]
            if df.empty:
                continue
            r = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
            s = pd.Series(r.values, index=df["date"]).groupby(level=0).last()
            anchor = lvl.at[last_ts, code] if last_ts in lvl.index else np.nan
            if not np.isfinite(anchor):
                known = lvl[code].loc[:last_ts].dropna()
                if known.empty:
                    continue               # 这只票在锚点之前就没有电平 → 无法连乘，跳过
                anchor = known.iloc[-1]
            base = float(anchor)
            vals: Dict[pd.Timestamp, "np.float32"] = {}
            for d, v in s.items():         # 按日期顺序连乘，基准 = 上一条电平 × (1+r)
                if not np.isfinite(v):
                    continue               # 停牌/缺涨跌幅：不连乘，该日留 NaN
                base *= 1.0 + float(v)
                vals[d] = np.float32(base)
            if vals:
                new_cols[code] = pd.Series(vals)

        if new_cols:
            # 新日期是**新行**，不是新列：reindex 到既有列集合再按行堆叠。
            # ⚠️ 别写成 `concat([lvl, added], axis=1)` —— 那会按列做外连接，
            # 同名列被复制成两份（实测 4 列变 8 列）。
            added = pd.DataFrame(new_cols).astype("float32").reindex(columns=lvl.columns)
            lvl = pd.concat([lvl, added], axis=0).sort_index()
            lvl = lvl[~lvl.index.duplicated(keep="first")]
        new_dates = [d for d in lvl.index if d > last_ts]
        appended = len(new_dates)
        lvl = lvl.astype("float32")
        print(f"[bench] 追加交易日 {appended} 个", flush=True)

        # 尾巴自检：新追加的每一行都应当和锚点行一样"几乎全员有价"。
        # 覆盖率塌了 = 追加逻辑又坏了（正是 11-05 那个 NaN 中毒的样子）。
        # 直接非零退出，绝不把 1/5596 的尾巴写进盘让下游算出假超额。
        anchor_n = int(lvl.loc[last_ts].notna().sum()) if last_ts in lvl.index else 0
        for d in new_dates:
            n = int(lvl.loc[d].notna().sum())
            ratio = n / max(anchor_n, 1)
            if ratio < 0.5:
                print(f"❌ 追加日 {str(d)[:10]} 覆盖率异常：{n}/{anchor_n} 只有值"
                      f"（{ratio:.2%}）→ 增量追加逻辑有问题，拒绝落盘")
                sys.exit(1)

    return lvl


def build_sleeve(start: str) -> Dict[str, float]:
    """9 只定稿 ETF 等权买入持有电平（起点 = start 处 = 1.0）。

    ⚠️ 区分两种"缺价"（2026-11-05 实测踩到）：
      - **末尾未完成日**：ETF 管道盘中也会写行，于是 9 只里只有几只 has 11-05
        → 这不是数据坏了，是当天还没收盘 → **截掉**，并且打印说明；
      - **窗口中间的缺价**（早于最后一个完整交易日）：那是真缺数据 → 直接失败，
        绝不用"有价的子集"求均值（那等于把 9 只等权基准偷偷换成 6 只基准）。
    """
    per_code: Dict[str, pd.Series] = {}
    for code in SLEEVE_CODES:
        path = os.path.join(ETF_DIR, f"{code}.parquet")
        if not os.path.exists(path):
            print(f"❌ 袖套 ETF 数据缺失：{code}（{path}）")
            sys.exit(1)
        df = pd.read_parquet(path)
        dcol = "date" if "date" in df.columns else df.index.name or "date"
        if dcol not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
        df = df[df["date"] >= pd.Timestamp(start)].sort_values("date")
        close = pd.to_numeric(df["close"], errors="coerce")
        ok = close.notna()
        s = pd.Series(close.values[ok.values], index=df["date"][ok]).groupby(level=0).last()
        if s.empty:
            print(f"❌ 袖套 ETF {code} 在 {start} 之后没有价格")
            sys.exit(1)
        per_code[code] = s / float(s.iloc[0])
    frame = pd.DataFrame(per_code).sort_index()

    complete = frame.notna().all(axis=1)
    if not complete.any():
        print("❌ 袖套基准没有任何一天是 9 只齐全的")
        sys.exit(1)
    last_complete = frame.index[complete].max()
    tail = [str(d)[:10] for d in frame.index[frame.index > last_complete]]
    if tail:
        print(f"[bench] 截掉未完成交易日 {tail}（9 只 ETF 未全部有价 —— 盘中/未收盘）")
    frame = frame.loc[frame.index <= last_complete]

    # 窗口中间必须齐全：任何一天缺价都是真缺数据
    holes = frame.isna().sum().sum()
    if holes:
        bad = [str(d)[:10] for d in frame.index[frame.isna().any(axis=1)]]
        print(f"❌ 袖套等权基准在 {len(bad)} 个已收盘交易日上缺价（{bad[:5]}）"
              "—— 不许用有价子集蒙过去（那会把 9 只等权换成 6 只等权）")
        sys.exit(1)
    eqw = frame.mean(axis=1)
    return {str(d)[:10]: float(v) for d, v in eqw.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01", help="基准电平起点（=1.0）")
    ap.add_argument("--incremental", action="store_true",
                    help="只追加既有电平之后的交易日")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="任一交易日覆盖率（当日有价代码数 / 起点池子数）低于此值 → 判失败")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    lvl = build_levels(args.start, args.incremental)
    sleeve = build_sleeve(args.start)

    pool_size = int(lvl.shape[1])
    present = lvl.notna().sum(axis=1)

    # ⚠️ 覆盖率怎么算才算对（2026-11-05 自己踩的坑）：
    # 拿「当日有价代码数 / 全池 5596」当覆盖率是**错的**——
    # 全池里有 2026 年新上市的票，它们在 1 月没有行，于是 1~4 月的"覆盖率"
    # 被新上市拉到 92%，看着像数据坏了，其实是池子定义问题（实测 92.05% @ 2026-04-30）。
    # 要防的是**幸存者偏差**：从某条线的起点往后，池子成员还在不在。
    # 所以门禁只看**最近 60 个交易日**（覆盖实际锚点 10-16 / 10-23 / 11-02），
    # 分母 = 窗口首日的有价代码数；全窗口最小值只作诊断打印。
    horizon_idx = lvl.index[max(0, len(lvl.index) - 60)]
    horizon_start = str(horizon_idx)[:10]
    horizon_base = int(present.loc[horizon_idx])
    recent = present.loc[lvl.index >= horizon_idx] / max(horizon_base, 1)
    cov_min = float(recent.min())
    cov_min_date = str(recent.idxmin())[:10]
    cov_min_all = float((present / max(pool_size, 1)).min())
    cov_min_all_date = str(present.idxmin())[:10]
    coverage_ok = bool(cov_min >= args.min_coverage)

    lvl.to_parquet(LEVELS_FP)
    with open(SLEEVE_FP, "w", encoding="utf-8") as fh:
        json.dump({"start": args.start, "levels": sleeve,
                   "codes": list(SLEEVE_CODES),
                   "basis": "9 只定稿 ETF 等权买入持有 mean_j[P_j(t)/P_j(start)]"}, fh,
                  ensure_ascii=False, indent=1)
    meta = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start": args.start,
        "pool_size": pool_size,
        "dates": int(lvl.shape[0]),
        "first_date": str(lvl.index.min())[:10],
        "last_date": str(lvl.index.max())[:10],
        "coverage_ok": coverage_ok,
        "gate_horizon_start": horizon_start,
        "gate_horizon_base_codes": horizon_base,
        "coverage_min": round(cov_min, 6),
        "coverage_min_date": cov_min_date,
        "coverage_note": f"门禁口径：{horizon_start}（{horizon_base} 只）起，"
                         "最近 60 个交易日的最小覆盖率",
        "coverage_min_all_diagnostic": round(cov_min_all, 6),
        "coverage_min_all_date_diagnostic": cov_min_all_date,
        "coverage_min_all_note": "全窗口最小覆盖率（分母=全池）只作诊断：低值通常是新上市票",
        "basis_a": "全池等权买入持有 mean_i[P_i(t)/P_i(start)]，后复权 pct_chg 链（不是日频再平衡等权）",
        "basis_sleeve": "9 只定稿 ETF 等权买入持有",
        "bias_note": "中途停牌/退市的代码不再出现在矩阵里 → 基准被抬高 → 超额被低估（台账 §十一：2.27% 退市股）",
    }
    with open(META_FP, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    print(f"[bench] A 基准：{pool_size} 只，{meta['first_date']} → {meta['last_date']}"
          f"（{meta['dates']} 个交易日）")
    print(f"[bench] 门禁覆盖率（{horizon_start} 起，分母 {horizon_base} 只）："
          f"最低 {cov_min:.4%} @ {cov_min_date} "
          f"（门槛 {args.min_coverage:.2%}）{'✓' if coverage_ok else '✗'}")
    print(f"[bench] 诊断：全窗口最小覆盖率 {cov_min_all:.4%} @ {cov_min_all_date}"
          f"（分母=全池 {pool_size}，含新上市票）")
    print(f"[bench] 袖套基准 {min(sleeve.values()):.6f}~{max(sleeve.values()):.6f}，"
          f"{len(sleeve)} 天")
    if not coverage_ok:
        print("❌ 锚定窗口覆盖率不达标：基准只在幸存者子集上算，不能用它谈超额")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
