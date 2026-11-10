#!/usr/bin/env python3
"""qlib_data (188 特征面板) 增量补齐 — 基于 stock_data 全历史重算。

为什么不用现成的 OfflineFactorPrecomputeService.precompute_single_stock:
  它内部走 load_stock_data()，而后者**优先读 features/day 的旧预计算特征**
  （含 $open/$close 就直接用，完全不读 stock_data）。于是形成闭环:
    precompute → load_stock_data → 读 features/day 旧文件 → 算出的仍是旧日期
  实测 600000.SH 走这条路 shape=(2608,188) 末日仍是 2026-05-08。

本脚本绕过 load_stock_data，直接读 data/parquet/stock_data（已到 2026-10-16，
且无 NaN），然后:
  1. convert_parquet_to_qlib + _compute_qlib_base_indicators  → 9 列
  2. Alpha158 用**表达式引擎**（force_expression_engine=True） → 158 列
     必须用表达式引擎: handler 读 features/day 旧文件；且表达式引擎在
     2021 起的短历史上会失败 75 个长窗口因子（MA/STD/BETA/RSQR/RESI/MAX/MIN
     的 20/30/60 窗口），必须喂**全历史**（1999 起）才能 158/158 全成功。
  3. _compute_technical_indicators（stock_data 无 NaN → 不触发
     StockData(volume=NaN) 的 int() 崩溃）                   → 16 列
  4. _compute_fundamental_features                            → 9 列
  5. LABEL0 = Ref($close,-2)/Ref($close,-1) - 1              → 1 列

实测 600000.SH: 187/188 列与现有完全对齐（只差 LABEL0），非 NaN 率从
现有 48% 提升到 100%，日期覆盖到 2026-10-16。

用法:
    cd ~/Projects/stock-platform
    env -u PYTHONPATH backend/.venv/bin/python3 \
        back_test_data_service/tdx_pipeline/rebuild_qlib_features.py \
        --dry-run --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from app.services.qlib import native_env  # noqa: E402
native_env.ensure_libomp_env_before_lightgbm_import()

from app.services.data.offline_factor_precompute import (  # noqa: E402
    OfflineFactorPrecomputeService,
)

STOCK_DATA_DIR = Path("/Users/rone/Projects/stock-platform/data/parquet/stock_data")
FEATURES_DIR = Path("/Users/rone/Projects/stock-platform/data/qlib_data/features/day")
CALENDAR = Path("/Users/rone/Projects/stock-platform/data/qlib_data/calendars/day.txt")
INSTRUMENTS = Path("/Users/rone/Projects/stock-platform/data/qlib_data/instruments/all.txt")

# 现有 features 用的 index level0 格式是大写后缀 (000001_SZ)
CODE_FORMAT = "upper_suffix"


def _code_to_ts(stem: str) -> Optional[str]:
    """600000_SH → 600000.SH；非三市场后缀返回 None。"""
    if "_" not in stem:
        return None
    code, suffix = stem.rsplit("_", 1)
    if suffix.upper() in ("SH", "SZ", "BJ") and code.isdigit():
        return f"{code}.{suffix.upper()}"
    return None


def _ts_to_stem(ts_code: str) -> str:
    """600000.SH → 600000_SH"""
    code, suffix = ts_code.split(".")
    return f"{code}_{suffix}"


def compute_label0(q: pd.DataFrame) -> pd.Series:
    """LABEL0 = Ref($close,-2)/Ref($close,-1) - 1（qlib 官方定义）。

    -2/-1 是**未来**偏移，即 T+1 收盘到 T+2 收盘的收益率。
    末两行必为 NaN（没有未来数据）。
    """
    close = q["$close"]
    fwd1 = close.shift(-1)
    fwd2 = close.shift(-2)
    return fwd2 / fwd1 - 1.0


async def rebuild_one(svc: OfflineFactorPrecomputeService,
                      ts_code: str) -> Optional[pd.DataFrame]:
    """重算单只股票的完整特征面板。返回 MultiIndex(stock_code, date) DataFrame。"""
    stem = _ts_to_stem(ts_code)
    path = STOCK_DATA_DIR / f"{stem}.parquet"
    if not path.exists():
        return None

    raw = pd.read_parquet(path)
    if raw.empty:
        return None
    raw = raw.set_index("date").sort_index()

    # 1. qlib 格式 + 基础指标
    q = svc.format_converter.convert_parquet_to_qlib(raw, ts_code)
    q = await svc._compute_qlib_base_indicators(q, ts_code)

    # 2. Alpha158（表达式引擎，全历史）
    idx = q.index
    dates = idx if not isinstance(idx, pd.MultiIndex) else idx.get_level_values(1)
    alpha = await svc.alpha_calculator.calculate_alpha_factors(
        qlib_data=q, stock_codes=[ts_code],
        date_range=(dates.min().to_pydatetime(), dates.max().to_pydatetime()),
        use_cache=False, force_expression_engine=True)
    if not alpha.empty:
        q = svc.format_converter.add_indicators_to_qlib(q, alpha, ts_code)

    # 3. 技术指标（stock_data 无 NaN，不会触发 int(NaN) 崩溃）
    ti = await svc._compute_technical_indicators(raw, ts_code)
    if not ti.empty:
        q = svc.format_converter.add_indicators_to_qlib(q, ti, ts_code)

    # 4. 基本面特征
    ff = await svc._compute_fundamental_features(raw, ts_code)
    if not ff.empty:
        q = svc.format_converter.add_indicators_to_qlib(q, ff, ts_code)

    # 5. LABEL0
    if "$close" in q.columns:
        q["LABEL0"] = compute_label0(q)

    # index level0 统一成 600000_SH（与现有 features 一致）
    if isinstance(q.index, pd.MultiIndex):
        q.index = q.index.set_levels([stem], level=0)
    return q


async def main() -> None:
    ap = argparse.ArgumentParser(description="重建 qlib_data 特征面板")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 只（测试用）")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    stocks = sorted(s for s in (_code_to_ts(p.stem) for p in STOCK_DATA_DIR.glob("*.parquet")) if s)
    if args.limit:
        stocks = stocks[: args.limit]
    print(f"目标股票: {len(stocks)} 只  模式: {'DRY-RUN' if args.dry_run else '写盘'}")

    svc = OfflineFactorPrecomputeService(batch_size=50, max_workers=args.workers)
    t0 = time.time()
    await svc.initialize_qlib()
    print(f"qlib 初始化: {time.time()-t0:.1f}s", flush=True)

    ok = fail = 0
    t0 = time.time()
    for i, ts_code in enumerate(stocks):
        try:
            q = await rebuild_one(svc, ts_code)
            if q is None:
                fail += 1
                continue
            ok += 1
            if not args.dry_run:
                out = FEATURES_DIR / f"{_ts_to_stem(ts_code)}.parquet"
                q.to_parquet(out, compression="snappy")
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  失败 {ts_code}: {type(e).__name__}: {e}")
        if (i + 1) % 25 == 0 or (i + 1) == len(stocks):
            el = time.time() - t0
            rate = (i + 1) / el
            eta = (len(stocks) - i - 1) / rate if rate else 0
            print(f"  [{i+1}/{len(stocks)}] ok={ok} fail={fail}  "
                  f"{rate:.1f}只/s  ETA={eta/60:.0f}分钟", flush=True)

    print(f"\n完成: ok={ok} fail={fail}  总耗时={time.time()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
