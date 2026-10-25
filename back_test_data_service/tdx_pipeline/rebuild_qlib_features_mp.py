#!/usr/bin/env python3
"""qlib_data 特征面板全量重建 — 多进程版。

单只耗时 4.2s（其中 Alpha158 表达式引擎 3.2s），5513 只串行约 6.4 小时。
用多进程并行，每个 worker 独立初始化 qlib（0.2s 开销，可忽略）。

用法:
    env -u PYTHONPATH backend/.venv/bin/python3 \
        back_test_data_service/tdx_pipeline/rebuild_qlib_features_mp.py \
        --workers 8

断点续传: 跳过已存在且末日 >= 目标末日的文件。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

STOCK_DATA_DIR = Path("/Users/rone/Projects/stock-platform/data/parquet/stock_data")
FEATURES_DIR = Path("/Users/rone/Projects/stock-platform/data/qlib_data/features/day")


def _code_to_ts(stem: str) -> Optional[str]:
    if "_" not in stem:
        return None
    code, suffix = stem.rsplit("_", 1)
    if suffix.upper() in ("SH", "SZ", "BJ") and code.isdigit():
        return f"{code}.{suffix.upper()}"
    return None


def _ts_to_stem(ts_code: str) -> str:
    code, suffix = ts_code.split(".")
    return f"{code}_{suffix}"


def compute_label0(q: pd.DataFrame) -> pd.Series:
    close = q["$close"]
    return close.shift(-2) / close.shift(-1) - 1.0


def worker(worker_id: int, codes: List[str], out_q: Queue) -> None:
    """子进程: 处理一批股票，把结果回传主进程。"""
    # 注意: 不要 chdir —— 父进程已 chdir 到 BACKEND，子进程继承即可；
    # 再 chdir 会让 multiprocessing 重新解析脚本路径时找不到文件。
    if BACKEND not in sys.path:
        sys.path.insert(0, BACKEND)

    from app.services.qlib import native_env
    native_env.ensure_libomp_env_before_lightgbm_import()
    from app.services.data.offline_factor_precompute import (
        OfflineFactorPrecomputeService,
    )

    async def _run() -> Tuple[int, int, List[str]]:
        svc = OfflineFactorPrecomputeService(batch_size=50, max_workers=1)
        await svc.initialize_qlib()
        ok = fail = 0
        errors: List[str] = []
        for ts_code in codes:
            try:
                path = STOCK_DATA_DIR / f"{_ts_to_stem(ts_code)}.parquet"
                raw = pd.read_parquet(path)
                if raw.empty:
                    fail += 1
                    continue
                raw = raw.set_index("date").sort_index()

                q = svc.format_converter.convert_parquet_to_qlib(raw, ts_code)
                q = await svc._compute_qlib_base_indicators(q, ts_code)

                idx = q.index
                dates = idx if not isinstance(idx, pd.MultiIndex) \
                    else idx.get_level_values(1)
                alpha = await svc.alpha_calculator.calculate_alpha_factors(
                    qlib_data=q, stock_codes=[ts_code],
                    date_range=(dates.min().to_pydatetime(), dates.max().to_pydatetime()),
                    use_cache=False, force_expression_engine=True)
                if not alpha.empty:
                    q = svc.format_converter.add_indicators_to_qlib(q, alpha, ts_code)

                ti = await svc._compute_technical_indicators(raw, ts_code)
                if not ti.empty:
                    q = svc.format_converter.add_indicators_to_qlib(q, ti, ts_code)

                ff = await svc._compute_fundamental_features(raw, ts_code)
                if not ff.empty:
                    q = svc.format_converter.add_indicators_to_qlib(q, ff, ts_code)

                if "$close" in q.columns:
                    q["LABEL0"] = compute_label0(q)

                if isinstance(q.index, pd.MultiIndex):
                    q.index = q.index.set_levels([_ts_to_stem(ts_code)], level=0)

                out = FEATURES_DIR / f"{_ts_to_stem(ts_code)}.parquet"
                q.to_parquet(out, compression="snappy")
                ok += 1
            except Exception as e:
                fail += 1
                if len(errors) < 5:
                    errors.append(f"{ts_code}: {type(e).__name__}: {e}")
        return ok, fail, errors

    ok, fail, errors = asyncio.run(_run())
    out_q.put((worker_id, ok, fail, errors, len(codes)))


def is_up_to_date(stem: str, target_end: pd.Timestamp) -> bool:
    """断点续传: 文件存在且末日 >= 目标末日则跳过。"""
    f = FEATURES_DIR / f"{stem}.parquet"
    if not f.exists():
        return False
    try:
        df = pd.read_parquet(f, columns=None)
        last = df.index[-1]
        last_date = last[1] if isinstance(last, tuple) else last
        return pd.Timestamp(last_date) >= target_end
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    stocks = sorted(s for s in (_code_to_ts(p.stem) for p in STOCK_DATA_DIR.glob("*.parquet")) if s)

    # 目标末日 = stock_data 里最新的日期（扫前 20 只估算，够准）
    sample_end = pd.Timestamp.min
    for s in stocks[:20]:
        p = STOCK_DATA_DIR / f"{_ts_to_stem(s)}.parquet"
        d = pd.read_parquet(p, columns=["date"])
        sample_end = max(sample_end, pd.Timestamp(d["date"].max()))
    print(f"目标末日: {sample_end.date()}")

    # 断点续传筛选
    todo = []
    skipped = 0
    for s in stocks:
        if is_up_to_date(_ts_to_stem(s), sample_end):
            skipped += 1
        else:
            todo.append(s)
    print(f"总计 {len(stocks)} 只, 已最新 {skipped} 只, 待处理 {len(todo)} 只")
    if args.limit:
        todo = todo[: args.limit]
        print(f"  (limit={args.limit} → 实际处理 {len(todo)} 只)")
    if not todo:
        print("无需处理"); return

    # 分片
    n = args.workers
    chunks = [todo[i::n] for i in range(n)]
    chunks = [c for c in chunks if c]
    print(f"开 {len(chunks)} 个 worker，各自处理 {[len(c) for c in chunks]}")

    q: Queue = Queue()
    procs = []
    t0 = time.time()
    for i, ch in enumerate(chunks):
        p = Process(target=worker, args=(i, ch, q), daemon=True)
        p.start()
        procs.append(p)

    total_ok = total_fail = 0
    done_stocks = 0
    all_errors: List[str] = []
    for _ in procs:
        wid, ok, fail, errors, n_done = q.get()
        total_ok += ok
        total_fail += fail
        done_stocks += n_done
        all_errors.extend(errors)
        el = time.time() - t0
        rate = done_stocks / el if el else 0
        eta = (len(todo) - done_stocks) / rate / 60 if rate else 0
        print(f"  worker{wid} 完成: ok={ok} fail={fail}  "
              f"累计 {done_stocks}/{len(todo)}  {rate:.1f}只/s  ETA={eta:.0f}分钟",
              flush=True)

    for p in procs:
        p.join(timeout=30)

    print(f"\n完成: ok={total_ok} fail={total_fail}  总耗时={(time.time()-t0)/60:.1f}分钟")
    if all_errors:
        print("错误样例:")
        for e in all_errors[:8]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
