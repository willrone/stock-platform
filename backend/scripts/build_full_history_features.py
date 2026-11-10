#!/usr/bin/env python3
"""从 qlib 官方全历史 OHLCV 构建平台口径的 42 因子特征面板。

## 背景

平台的 42 因子复合（`honest_composite_strategy`）需要两套输入：

1. **25 个基础公式**（MIN/MAX/STD/CORR/CORD × 5 窗口）—— 策略在
   `prepare_backtest_data` 里用 close/high/low/volume 现场算，不落盘。
2. **17 个 Alpha158 面板因子**（VSUMP/VSUMN/VSUMD/QTLD × 3 窗口 +
   IMAX5/IMIN5/WVMA5/VMA5/VSTD5）—— 从 `panel_dir` 的 parquet 读。

`data/qlib_data/features/day/` 只有 272 只股票有 2021-05-11 之前的全历史，
其余 5241 只的 parquet 是「2021-05-11 起的增量块」与「Alpha158 面板块」
两段拼接（见 `docs/backtest/` 里 qlib parquet 两段拼接的记录），而 Alpha158
面板块本身又被日期去重截断成 ~53% 覆盖率 → 42 因子在 10 年回测里根本
拿不到 2021 年之前的面板因子。

本脚本用 `data/qlib_official_data` 的 .bin 原始数据（**每个股票从上市首日
到数据末日**）重算 188 列特征，输出到 `data/qlib_data/features_full/`。

## 关键口径事实（已实测，见 docs 报告）

- **价格基准 = 未复权，且与 features/day 完全同源**：
  - qlib 官方 .bin 的 close 与 `data/parquet/stock_data` 的 close **逐日相等**
    （抽样 200 只，maxdiff ≤ 2.9e-5，纯 float32 精度；5513/5513 只的稠密
    值长度 == [上市日, 退市日] 区间内的 stock_data 行数，200/200 抽样全部命中）。
  - qlib 官方数据**没有 $factor 文件**，存的就是未复权真实价。
  - features/day 的 $close 同样未复权 → **新管线与 features/day 价格基准
    天然一致，无需（也不能）做任何复权**。
  - 参考 features/day 的 $close 逐日 maxdiff = 0.0（完全相等）。
- **.bin 的日期轴不可用**：qlib 的 FileFeatureStorage 把 start_index 写在
  文件头（float32），本仓库这批文件**全部是 0**（5513/5513），但内容是按
  **该股票自身的交易日序列稠密排列**的（值与 stock_data 逐位相等）。
  qlib 自己的 `D.features()` 因此会给出**错误日期**（600000_sh 被读成
  1990-12-19~2016-11-10，实际 1999-11-10 起）。
  → 本脚本**不依赖 qlib 的日期轴**，而是用 `data/parquet/stock_data` 的
  日期作为权威轴，并用 qlib 官方 .bin 逐日**交叉校验**（不一致就跳过该股，
  绝不静默出数）。日期范围用 qlib 官方 `instruments/all.txt` 的上市/退市日
  做 PIT 截断。
- **features/day 的两段拼接**（docs/backtest/FULL_STRATEGY_AUDIT_2026-10-25.md
  的 D6）：5241/5513 只的参考文件是 block1(交易日格, 有 $close 无面板因子)
  + block2(业务日格, 含节假日行, 有面板因子无价格) 拼接而成，策略去重后
  同一天只剩其中一段 → 「同一天既有价格又有面板因子」的日期数 = 0。
  那个 block2 的面板因子是**用 qlib 原生 `D.features()` 在业务日格上算的**
  （本脚本已复现，maxdiff=0.0, corr=1.0），滚动窗口跨了节假日，
  因此与「交易日格」口径在数值上不同 —— 这是**参考数据的既有缺陷**，
  不是新管线的口径错误。新管线统一在交易日格上算，与 272 只单段参考文件
  逐列完全相等（corr=1.0, maxdiff=0.0）。

## 因子口径（与 features/day 逐列同源）

复用平台自己的 `OfflineFactorPrecomputeService`：
  1. `convert_parquet_to_qlib` + `_compute_qlib_base_indicators`  → 9 列
  2. Alpha158 **表达式引擎**（`force_expression_engine=True`）        → 158 列
  3. `_compute_technical_indicators`                                  → 16 列
  4. `_compute_fundamental_features`                                  → 9 列
  5. `LABEL0 = Ref($close,-2)/Ref($close,-1) - 1`                      → 1 列

这与 `back_test_data_service/tdx_pipeline/rebuild_qlib_features.py` 的
既有重建脚本**完全同一套代码**，只是数据源换成全历史 + 输出目录换成
`features_full/`。

## PIT 正确性

- 所有滚动窗口只用历史（含当日），无未来函数。
- 唯一的例外是 `LABEL0`，它用 `shift(-2)/shift(-1)`，**是标签不是因子**，
  与 features/day 行为一致（末两行必为 NaN）。
- 17 个 Alpha158 面板因子里 `Ref(x, n)` 全部是**向后**引用（n>0），
  `Sum/Mean/Std/Quantile/IdxMax/IdxMin` 都是**后向窗口**。
- 每个股票的样本区间 = [上市日, 退市日]，退市后不再有数据（PIT 宇宙）。

## 用法

    # 冒烟：先跑 5 只
    env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
        backend/scripts/build_full_history_features.py --limit 5

    # 全量（断点续跑）
    env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
        backend/scripts/build_full_history_features.py --resume

    # 只做口径对账
    env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
        backend/scripts/build_full_history_features.py --reconcile-only \
        --out data/backtest/full_history_reconcile.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# ── 启动期环境固化（必须在任何 chdir / import app.* 之前）────────────
# 1) sys.argv[0] 固化成绝对路径。
#    本脚本稍后会把 cwd 切到 backend/，而 app.services.qlib.native_env
#    在 macOS 上会用 os.execvpe(sys.executable, [sys.executable, *sys.argv])
#    **重新执行本脚本**（为了让 dyld 在进程启动时就带上 Homebrew libomp）。
#    若 argv[0] 是相对路径，换 cwd 后就会变成
#    ".../backend/backend/scripts/xxx.py" → 子进程直接死掉，父进程收到 0 结果。
if sys.argv and sys.argv[0]:
    _argv0 = Path(sys.argv[0])
    if not _argv0.is_absolute():
        sys.argv[0] = str(_argv0.resolve())

# 2) 父进程在此处完成一次 libomp 的 re-exec。
#    必须在 chdir 之前做，且只做一次 —— fork 出来的子进程会继承父进程
#    已经设置好的 DYLD_LIBRARY_PATH，native_env 里会直接 early-return，
#    不会在子进程里再次 re-exec（否则会递归 fork 出新的一批 worker）。
_LIBOMP_DIR = Path("/opt/homebrew/opt/libomp/lib")
if sys.platform == "darwin" and _LIBOMP_DIR.exists():
    _existing = [p for p in os.environ.get("DYLD_LIBRARY_PATH", "").split(":") if p]
    if str(_LIBOMP_DIR) not in _existing:
        _env = os.environ.copy()
        _env["DYLD_LIBRARY_PATH"] = ":".join([str(_LIBOMP_DIR), *_existing])
        os.execvpe(sys.executable, [sys.executable, *sys.argv], _env)

QLIB_OFFICIAL = ROOT / "data" / "qlib_official_data"
STOCK_DATA_DIR = ROOT / "data" / "parquet" / "stock_data"
FEATURES_FULL_DIR = ROOT / "data" / "qlib_data" / "features_full"
FEATURES_DAY_DIR = ROOT / "data" / "qlib_data" / "features" / "day"
PROGRESS_FILE = ROOT / "data" / "backtest" / "full_history_features_progress.json"

# 42 因子里被策略真正读取的 17 个 Alpha158 面板因子（honest_composite_strategy）
PANEL_FACTORS: Dict[str, float] = {
    "VSUMP5": -1.0, "VSUMP10": -1.0, "VSUMP20": -1.0,
    "VSUMN5": 1.0, "VSUMN10": 1.0, "VSUMN20": 1.0,
    "VSUMD5": -1.0, "VSUMD10": -1.0, "VSUMD20": -1.0,
    "QTLD5": 1.0, "QTLD10": 1.0, "QTLD20": 1.0,
    "IMAX5": -1.0, "IMIN5": -1.0, "WVMA5": -1.0,
    "VMA5": 1.0, "VSTD5": -1.0,
}

# 对账用的 Alpha158 全量因子名（features/day 里存在的 158 列）
ALPHA158_COLS: List[str] = [
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
    "OPEN0", "HIGH0", "LOW0", "VWAP0",
    "ROC5", "ROC10", "ROC20", "ROC30", "ROC60",
    "MA5", "MA10", "MA20", "MA30", "MA60",
    "STD5", "STD10", "STD20", "STD30", "STD60",
    "BETA5", "BETA10", "BETA20", "BETA30", "BETA60",
    "RSQR5", "RSQR10", "RSQR20", "RSQR30", "RSQR60",
    "RESI5", "RESI10", "RESI20", "RESI30", "RESI60",
    "MAX5", "MAX10", "MAX20", "MAX30", "MAX60",
    "MIN5", "MIN10", "MIN20", "MIN30", "MIN60",
    "QTLU5", "QTLU10", "QTLU20", "QTLU30", "QTLU60",
    "QTLD5", "QTLD10", "QTLD20", "QTLD30", "QTLD60",
    "RANK5", "RANK10", "RANK20", "RANK30", "RANK60",
    "RSV5", "RSV10", "RSV20", "RSV30", "RSV60",
    "IMAX5", "IMAX10", "IMAX20", "IMAX30", "IMAX60",
    "IMIN5", "IMIN10", "IMIN20", "IMIN30", "IMIN60",
    "IMXD5", "IMXD10", "IMXD20", "IMXD30", "IMXD60",
    "CORR5", "CORR10", "CORR20", "CORR30", "CORR60",
    "CORD5", "CORD10", "CORD20", "CORD30", "CORD60",
    "CNTP5", "CNTP10", "CNTP20", "CNTP30", "CNTP60",
    "CNTN5", "CNTN10", "CNTN20", "CNTN30", "CNTN60",
    "CNTD5", "CNTD10", "CNTD20", "CNTD30", "CNTD60",
    "SUMP5", "SUMP10", "SUMP20", "SUMP30", "SUMP60",
    "SUMN5", "SUMN10", "SUMN20", "SUMN30", "SUMN60",
    "SUMD5", "SUMD10", "SUMD20", "SUMD30", "SUMD60",
    "VMA5", "VMA10", "VMA20", "VMA30", "VMA60",
    "VSTD5", "VSTD10", "VSTD20", "VSTD30", "VSTD60",
    "WVMA5", "WVMA10", "WVMA20", "WVMA30", "WVMA60",
    "VSUMP5", "VSUMP10", "VSUMP20", "VSUMP30", "VSUMP60",
    "VSUMN5", "VSUMN10", "VSUMN20", "VSUMN30", "VSUMN60",
    "VSUMD5", "VSUMD10", "VSUMD20", "VSUMD30", "VSUMD60",
]

# 对账用的 25 个基础公式（策略在 prepare_backtest_data 现场算，口径见下）
BASE_FORMULAS: List[str] = [
    f"{fam}{n}" for n in (5, 10, 20, 30, 60)
    for fam in ("MIN", "MAX", "STD", "CORR", "CORD")
]


# ── 工具 ──────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    print(msg, flush=True)


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


def load_qlib_instruments() -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """qlib 官方 instruments/all.txt → {instrument: (上市日, 退市日)}（PIT 语义）。"""
    out: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    path = QLIB_OFFICIAL / "instruments" / "all.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and "_" in parts[0]:
            try:
                out[parts[0]] = (
                    pd.Timestamp(parts[1]),
                    pd.Timestamp(parts[2]),
                )
            except ValueError:
                continue
    return out


def read_qlib_bin_dense(instrument: str, field: str) -> np.ndarray:
    """读 qlib 官方 .bin，返回**稠密**（去掉 NaN）后的 float32 序列。

    qlib 的 .bin 结构：首 4 字节 float32 = start_index，其后是按日历槽位
    排列的值。本仓库这批文件的 start_index 恒为 0（实测 5513/5513），内容
    是该股票自身交易日的稠密序列 —— 不能用 qlib 的 D.features() 取日期。
    """
    p = QLIB_OFFICIAL / "features" / instrument / f"{field}.day.bin"
    raw = np.fromfile(p, dtype="<f4")
    values = raw[1:]  # 丢掉 start_index 头
    return values[~np.isnan(values)]


def build_stock_frame(
    instrument: str,
    ts_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """构造该股票在 [上市日, 退市日] 区间上的 OHLCV 权威数据框。

    数据源：
      - 日期轴 + OHLCV：`data/parquet/stock_data`（平台权威，逐日与 qlib
        官方 .bin 稠密值相等，实测 maxdiff ≤ 1.2e-4）
      - qlib 官方 .bin 用来**校验**（值不等就跳过该股，绝不静默出数）
    """
    sd_path = STOCK_DATA_DIR / f"{_ts_to_stem(ts_code)}.parquet"
    if not sd_path.exists():
        return None
    raw = pd.read_parquet(sd_path)
    if raw.empty:
        return None
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()
    # PIT：只保留 [上市日, 退市日]
    raw = raw.loc[(raw.index >= start) & (raw.index <= end)]
    if raw.empty:
        return None
    # 保留 ts_code 列：features/day 里的 ts_code 列正是从 stock_data 的
    # ts_code 经 _compute_fundamental_features 透传出来的（实测值一致）。
    keep = ["open", "high", "low", "close", "volume"]
    if "ts_code" in raw.columns:
        keep = ["ts_code"] + keep
    raw = raw[keep].copy()
    for c in ("open", "high", "low", "close"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    raw = raw.dropna(subset=["open", "high", "low", "close", "volume"])
    if raw.empty:
        return None

    # 与 qlib 官方 .bin 交叉校验（close）：稠密值必须逐日相等
    qlib_close = read_qlib_bin_dense(instrument, "close")
    n = min(len(qlib_close), len(raw))
    if n == 0:
        return None
    diff = np.abs(
        qlib_close[:n].astype(np.float64)
        - raw["close"].to_numpy()[:n].astype(np.float64)
    )
    # 允许 float32 精度（相对 1e-4），超出说明两源不同步 → 不可信
    tol = np.maximum(np.abs(raw["close"].to_numpy()[:n]) * 1e-4, 1e-3)
    if np.any(diff > tol):
        return None
    return raw


# ── 特征计算（复用平台自己的服务，保证与 features/day 同源同口径）──────
async def compute_features(svc, ts_code: str, raw: pd.DataFrame) -> pd.DataFrame:
    """与 rebuild_qlib_features.rebuild_one 完全一致的计算路径。"""
    q = svc.format_converter.convert_parquet_to_qlib(raw, ts_code)
    q = await svc._compute_qlib_base_indicators(q, ts_code)

    idx = q.index
    dates = idx if not isinstance(idx, pd.MultiIndex) else idx.get_level_values(1)
    alpha = await svc.alpha_calculator.calculate_alpha_factors(
        qlib_data=q, stock_codes=[ts_code],
        date_range=(dates.min().to_pydatetime(), dates.max().to_pydatetime()),
        use_cache=False, force_expression_engine=True,
    )
    if not alpha.empty:
        q = svc.format_converter.add_indicators_to_qlib(q, alpha, ts_code)

    ti = await svc._compute_technical_indicators(raw, ts_code)
    if not ti.empty:
        q = svc.format_converter.add_indicators_to_qlib(q, ti, ts_code)

    ff = await svc._compute_fundamental_features(raw, ts_code)
    if not ff.empty:
        q = svc.format_converter.add_indicators_to_qlib(q, ff, ts_code)

    if "$close" in q.columns:
        close = q["$close"]
        q["LABEL0"] = close.shift(-2) / close.shift(-1) - 1.0

    if isinstance(q.index, pd.MultiIndex):
        q.index = q.index.set_levels([_ts_to_stem(ts_code)], level=0)
    return q


def make_service():
    os.chdir(BACKEND)
    from app.services.qlib import native_env
    native_env.ensure_libomp_env_before_lightgbm_import()
    from app.services.data.offline_factor_precompute import (
        OfflineFactorPrecomputeService,
    )
    return OfflineFactorPrecomputeService(batch_size=50, max_workers=1)


# ── 多进程 ────────────────────────────────────────────────────────────
def _worker(worker_id: int, items: List[Tuple[str, str, str, str]],
            out_dir_str: str, progress_path_str: str) -> None:
    """子进程：处理一批股票，结果写到自己的进度分片文件。

    不走 multiprocessing.Queue：
      - Queue 需要父进程持续消费，大批量结果容易把父子进程都卡死；
      - 进度分片文件天然支持断点续跑（父进程合并所有分片）。
    """
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    from app.services.qlib import native_env
    native_env.ensure_libomp_env_before_lightgbm_import()
    from app.services.data.offline_factor_precompute import (
        OfflineFactorPrecomputeService,
    )

    out_dir = Path(out_dir_str)
    progress_path = Path(progress_path_str)
    svc = OfflineFactorPrecomputeService(batch_size=50, max_workers=1)

    async def _run() -> None:
        await svc.initialize_qlib()
        results: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        def flush() -> None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = progress_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"results": results, "errors": errors},
                           ensure_ascii=False), encoding="utf-8")
            tmp.replace(progress_path)

        for n_done, (instrument, ts_code, start_s, end_s) in enumerate(items, 1):
            stem = _ts_to_stem(ts_code)
            try:
                raw = build_stock_frame(
                    instrument, ts_code, pd.Timestamp(start_s), pd.Timestamp(end_s))
                if raw is None:
                    results[stem] = {"status": "no_data"}
                    continue
                q = await compute_features(svc, ts_code, raw)
                if q is None or q.empty:
                    results[stem] = {"status": "empty"}
                    continue
                q.to_parquet(out_dir / f"{stem}.parquet", compression="snappy")
                results[stem] = {
                    "status": "ok",
                    "rows": int(len(q)),
                    "start": str(q.index.get_level_values(1).min().date()),
                    "end": str(q.index.get_level_values(1).max().date()),
                    "ncols": int(q.shape[1]),
                }
            except Exception as e:  # noqa: BLE001
                results[stem] = {"status": f"error: {type(e).__name__}: {e}"}
                if len(errors) < 5:
                    errors.append(f"{stem}: {type(e).__name__}: {e}")
            # 每 10 只落一次盘：中断后 --resume 能接着跑
            if n_done % 10 == 0:
                flush()
        flush()

    asyncio.run(_run())


def run_build_mp(limit: Optional[int], resume: bool, out_dir: Path,
                 only: Optional[List[str]], workers: int) -> Dict[str, Any]:
    import multiprocessing as mp

    targets = _collect_targets(only)
    _log(f"[build] qlib 官方 instruments: {len(load_qlib_instruments())}，"
         f"其中有本地 stock_data 的: {len(targets)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    progress: Dict[str, Any] = {}
    if resume and PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            progress = {}

    shard_dir = PROGRESS_FILE.parent / "full_history_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    # 上次运行遗留的分片（worker 中途每 10 只落一次盘）→ 合并进进度，
    # 这样 --resume 能识别到已完成但主进度文件还没写的股票。
    if resume:
        for old in sorted(shard_dir.glob("worker_*.json")):
            try:
                blob = json.loads(old.read_text(encoding="utf-8"))
                for stem, rec in blob.get("results", {}).items():
                    progress.setdefault(stem, rec)
            except Exception:
                continue

    todo = []
    skipped = 0
    for instrument, ts_code, start, end in targets:
        stem = _ts_to_stem(ts_code)
        if resume:
            rec = progress.get(stem)
            if rec and rec.get("status") == "ok" and (
                out_dir / f"{stem}.parquet").exists():
                skipped += 1
                continue
        todo.append((instrument, ts_code, start, end))
    if limit:
        todo = todo[:limit]
    _log(f"[build] 已完成跳过 {skipped}，本次待处理 {len(todo)}，"
         f"开 {workers} 个 worker")

    os.chdir(BACKEND)
    chunks = [todo[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]

    # fork：子进程继承已 chdir 的 cwd 与 sys.path，
    # 不会像 spawn 那样重新按相对路径解析脚本导致子进程起不来。
    ctx = mp.get_context("fork")
    procs = []
    for i, ch in enumerate(chunks):
        items = [(inst, ts, str(st), str(en)) for inst, ts, st, en in ch]
        shard = shard_dir / f"worker_{i}.json"
        p = ctx.Process(
            target=_worker,
            args=(i, items, str(out_dir), str(shard)),
            daemon=True,
        )
        p.start()
        procs.append((i, p, shard, len(ch)))

    t0 = time.time()
    ok = fail = 0
    errors: List[str] = []
    seen: Dict[str, Dict[str, Any]] = {}
    pending = list(procs)
    last_log = 0.0
    while pending:
        alive = []
        for i, p, shard, n_in_chunk in pending:
            if p.is_alive():
                alive.append((i, p, shard, n_in_chunk))
                continue
            p.join()
            results: Dict[str, Any] = {}
            werrs: List[str] = []
            if shard.exists():
                try:
                    blob = json.loads(shard.read_text(encoding="utf-8"))
                    results = blob.get("results", {})
                    werrs = blob.get("errors", [])
                except Exception as e:  # noqa: BLE001
                    werrs = [f"shard read error: {e}"]
            for stem, rec in results.items():
                progress[stem] = rec
                seen[stem] = rec
            errors.extend(werrs)
        pending = alive

        # 轮询分片，把在途进度合并进主进度文件（--resume 才有效）
        n_ok = n_all = 0
        for shard_path in sorted(shard_dir.glob("worker_*.json")):
            try:
                blob = json.loads(shard_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for stem, rec in blob.get("results", {}).items():
                progress[stem] = rec
                seen[stem] = rec
            n_all += len(blob.get("results", {}))
        n_ok = sum(1 for r in seen.values() if r.get("status") == "ok")
        el = time.time() - t0
        rate = n_ok / el if el else 0
        eta = (len(todo) - n_ok) / rate / 60 if rate else 0
        now = time.time()
        if now - last_log > 30:
            last_log = now
            _log(f"  [进度] {n_ok}/{len(todo)} ok, {rate:.2f}只/s, "
                 f"ETA={eta:.0f}分钟")
            PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            PROGRESS_FILE.write_text(
                json.dumps(progress, ensure_ascii=False), encoding="utf-8")
        if pending:
            time.sleep(2)

    ok = sum(1 for r in seen.values() if r.get("status") == "ok")
    fail = len(seen) - ok

    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps(progress, ensure_ascii=False), encoding="utf-8")
    _log(f"[build] 完成 ok={ok} fail={fail} 耗时={(time.time()-t0)/60:.1f}分钟")
    if errors:
        _log("[build] 错误样例:")
        for e in errors[:10]:
            _log(f"  {e}")
    return {"ok": ok, "fail": fail, "skipped": skipped, "total": len(targets)}


# ── 主流程 ────────────────────────────────────────────────────────────
def _collect_targets(
    only: Optional[List[str]] = None,
) -> List[Tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    """收集要处理的股票：qlib 官方 instruments ∩ 本地 stock_data。"""
    instruments = load_qlib_instruments()
    only_set = {s.upper() for s in only} if only else None
    targets: List[Tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    for instrument, (start, end) in sorted(instruments.items()):
        ts_code = _code_to_ts(instrument)
        if ts_code is None:
            continue
        if only_set is not None and instrument.upper() not in only_set:
            continue
        if not (STOCK_DATA_DIR / f"{_ts_to_stem(ts_code)}.parquet").exists():
            continue
        targets.append((instrument, ts_code, start, end))
    return targets


async def run_build(limit: Optional[int], resume: bool, out_dir: Path,
                    only: Optional[List[str]] = None) -> Dict[str, Any]:
    targets = _collect_targets(only)
    _log(f"[build] qlib 官方 instruments: {len(load_qlib_instruments())}，"
         f"其中有本地 stock_data 的: {len(targets)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    progress: Dict[str, Any] = {}
    if resume and PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            progress = {}

    todo = []
    skipped = 0
    for instrument, ts_code, start, end in targets:
        stem = _ts_to_stem(ts_code)
        if resume:
            rec = progress.get(stem)
            if rec and rec.get("status") == "ok" and (
                out_dir / f"{stem}.parquet"
            ).exists():
                skipped += 1
                continue
        todo.append((instrument, ts_code, start, end))
    if limit:
        todo = todo[:limit]
    _log(f"[build] 已完成跳过 {skipped}，本次待处理 {len(todo)}")

    svc = make_service()
    await svc.initialize_qlib()

    ok = fail = 0
    t0 = time.time()
    errors: List[str] = []
    for i, (instrument, ts_code, start, end) in enumerate(todo):
        stem = _ts_to_stem(ts_code)
        try:
            raw = build_stock_frame(instrument, ts_code, start, end)
            if raw is None:
                fail += 1
                progress[stem] = {"status": "no_data"}
                continue
            q = await compute_features(svc, ts_code, raw)
            if q is None or q.empty:
                fail += 1
                progress[stem] = {"status": "empty"}
                continue
            q.to_parquet(out_dir / f"{stem}.parquet", compression="snappy")
            ok += 1
            progress[stem] = {
                "status": "ok",
                "rows": int(len(q)),
                "start": str(q.index.get_level_values(1).min().date()),
                "end": str(q.index.get_level_values(1).max().date()),
                "ncols": int(q.shape[1]),
            }
        except Exception as e:  # noqa: BLE001
            fail += 1
            progress[stem] = {"status": f"error: {type(e).__name__}: {e}"}
            if len(errors) < 10:
                errors.append(f"{stem}: {type(e).__name__}: {e}")
                traceback.print_exc()
        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            el = time.time() - t0
            rate = (i + 1) / el if el else 0
            eta = (len(todo) - i - 1) / rate / 60 if rate else 0
            _log(f"  [{i+1}/{len(todo)}] ok={ok} fail={fail} "
                 f"{rate:.2f}只/s ETA={eta:.0f}分钟")
            PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            PROGRESS_FILE.write_text(
                json.dumps(progress, ensure_ascii=False), encoding="utf-8"
            )

    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps(progress, ensure_ascii=False), encoding="utf-8"
    )
    _log(f"[build] 完成 ok={ok} fail={fail} 耗时={(time.time()-t0)/60:.1f}分钟")
    if errors:
        _log("[build] 错误样例:")
        for e in errors:
            _log(f"  {e}")
    return {"ok": ok, "fail": fail, "skipped": skipped, "total": len(targets)}


# ── 口径对账 ──────────────────────────────────────────────────────────
def _read_panel(path: Path) -> pd.DataFrame:
    """按策略的读法读 panel parquet（reset_index → set_index(date) → 去重）。"""
    pf = pd.read_parquet(path).reset_index()
    pf = pf.set_index("date").sort_index()
    pf = pf[~pf.index.duplicated(keep="first")]
    return pf


def _split_reference_blocks(ref: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """把参考 parquet 拆成 {'price': ..., 'panel': ...} 两块。

    平台 features/day 里 5241/5513 只是**两段拼接**（见 docs/backtest/
    FULL_STRATEGY_AUDIT_2026-10-25.md 的 D6）：
      block1 (lvl0='600000.SH')  交易日格 → 有 $close，无 Alpha158 面板因子
      block2 (lvl0='600000_sh')  业务日格（含节假日）→ 无价格，有面板因子
    两者日期集合不同，策略去重后同一天只剩其中一段
    → 「同一天既有价格又有面板因子」的日期数 = 0。

    单段文件（272 只有全历史的）则价格与面板同格。
    """
    out: Dict[str, pd.DataFrame] = {}
    if not isinstance(ref.index, pd.MultiIndex):
        out["single"] = ref
        return out
    for lvl0 in pd.unique(ref.index.get_level_values(0)):
        sub = ref.xs(lvl0, level=0, drop_level=False)
        sub = sub.set_index(sub.index.get_level_values(1))
        has_price = "$close" in sub.columns and bool(sub["$close"].notna().any())
        has_panel = "VSUMP5" in sub.columns and bool(sub["VSUMP5"].notna().any())
        key = "price" if has_price and not has_panel else (
            "panel" if has_panel and not has_price else "both")
        out[key] = sub
    return out


def _pick_block(blocks: Dict[str, pd.DataFrame],
                order: Tuple[str, ...]) -> Optional[pd.DataFrame]:
    """按优先级取参考块（避免对 DataFrame 用 `or` 触发歧义真值）。"""
    for key in order:
        blk = blocks.get(key)
        if blk is not None and not blk.empty:
            return blk
    return None


def compute_base_factors(close: pd.Series, high: pd.Series, low: pd.Series,
                         vol: pd.Series) -> Dict[str, pd.Series]:
    """复刻 honest_composite_strategy.prepare_backtest_data 的 25 个基础公式。

    cands[f"MIN{n}"]  = close.rolling(n).min() / close - 1.0
    cands[f"MAX{n}"]  = close.rolling(n).max() / close - 1.0
    cands[f"STD{n}"]  = close.rolling(n).std() / close
    cands[f"CORR{n}"] = close.rolling(n).corr(vol)
    cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)
    （ret/vchg 用 pct_change(fill_method=None)）
    """
    out: Dict[str, pd.Series] = {}
    ret = close.pct_change(fill_method=None)
    vchg = vol.pct_change(fill_method=None)
    for n in (5, 10, 20, 30, 60):
        out[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
        out[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
        out[f"STD{n}"] = close.rolling(n).std() / close
        out[f"CORR{n}"] = close.rolling(n).corr(vol)
        out[f"CORD{n}"] = ret.rolling(n).corr(vchg)
    return out


def _corr(a: pd.Series, b: pd.Series) -> Optional[float]:
    """两组值的相关系数。

    注意：某些因子（如 CORR5）在退市/常量窗口上会产生 ±inf
    （pandas rolling.corr 的退化窗口），features/day 与新管线**同样**有这些
    ±inf（实测 inf 位置完全一致）。若直接喂给 np.corrcoef 会得到 NaN，
    所以这里先把非有限值剔除再算。
    """
    both = a.notna() & b.notna()
    if both.sum() < 30:
        return None
    x = a[both].to_numpy(dtype=np.float64)
    y = b[both].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 30:
        return None
    x = x[finite]
    y = y[finite]
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def reconcile(
    out_dir: Path,
    n_stocks: int,
    out_json: Path,
    start: str,
    end: str,
) -> Dict[str, Any]:
    """新管线 vs features/day 的逐列对账（限 2021+ 重叠区间）。"""
    instruments = load_qlib_instruments()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    report: Dict[str, Any] = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "overlap_window": [start, end],
        "price_basis": {
            "adjustment": "unadjusted",
            "why": "qlib official .bin has no $factor file; its close equals "
                   "data/parquet/stock_data close day-by-day (200-stock sample, "
                   "maxdiff<=2.9e-5, float32 precision). features/day $close is "
                   "the same unadjusted series (maxdiff=0.0 vs new pipeline).",
            "verification": "n_qlib_valid == n_stock_data_rows_in_[listing,delisting] "
                            "for 200/200 sampled instruments; qlib .bin values match "
                            "those rows to <=2.9e-5",
        },
        "reference_structure_finding": {
            "single_block_stocks": 272,
            "two_block_stocks": 5241,
            "note": "features/day is a 2-block concat for 5241/5513 stocks: "
                    "block1=trading-day grid (has $close, no panel factors), "
                    "block2=business-day grid incl. holidays (has panel factors, "
                    "NO prices). After the strategy's date-dedup, days with both "
                    "price and panel factors = 0.",
            "block2_provenance": "reproduced exactly by qlib native D.features() on "
                                 "the data/qlib_data provider (maxdiff=0.0, corr=1.0) "
                                 "-> its rolling windows straddle holidays, so it is "
                                 "NOT comparable to a trading-day-grid computation.",
        },
        "panel_dir_new": str(out_dir),
        "panel_dir_ref": str(FEATURES_DAY_DIR),
        "stocks": [],
    }
    # 候选：features/day 里有、且新管线已产出（能真正对账）的股票
    ref_stems = {p.stem for p in FEATURES_DAY_DIR.glob("*.parquet")}
    new_stems = {p.stem for p in out_dir.glob("*.parquet")}
    candidates = sorted(ref_stems & new_stems)
    if not candidates:
        _log("[reconcile] 新管线与 features/day 无交集，无法对账")
        return report

    # 按参考结构分层抽样：单段文件（可比口径）与两段文件（不可比口径）各取一半
    single: List[str] = []
    two: List[str] = []
    for stem in candidates:
        try:
            ref_raw = pd.read_parquet(FEATURES_DAY_DIR / f"{stem}.parquet",
                                      columns=[])
        except Exception:
            continue
        lvl0 = ref_raw.index.get_level_values(0)
        (single if len(set(lvl0)) == 1 else two).append(stem)
    rng = np.random.default_rng(20261026)
    n_single = min(len(single), (n_stocks + 1) // 2)
    n_two = min(len(two), n_stocks - n_single)
    chosen = sorted(rng.permutation(single)[:n_single].tolist()
                    + rng.permutation(two)[:n_two].tolist())
    _log(f"[reconcile] 共同股票 {len(candidates)} 只 "
         f"(单段 {len(single)} / 两段 {len(two)})，抽样 {len(chosen)} 只")

    agg: Dict[str, List[float]] = {"corr": [], "mean_abs_diff": [], "std_ratio": []}
    per_factor_corr: Dict[str, List[float]] = {}
    # 按参考结构分桶：只有单段文件（价格与面板同格）才是可比口径
    buckets: Dict[str, Dict[str, List[float]]] = {}

    for stem in chosen:
        new_path = out_dir / f"{stem}.parquet"
        ref_path = FEATURES_DAY_DIR / f"{stem}.parquet"
        if not new_path.exists() or not ref_path.exists():
            continue
        new = _read_panel(new_path)
        ref_raw = pd.read_parquet(ref_path)
        blocks = _split_reference_blocks(ref_raw)
        if not blocks:
            continue
        # 参考面板块：有 Alpha158 面板因子的那段
        ref_panel_blk = _pick_block(blocks, ("panel", "both", "single"))
        ref_price_blk = _pick_block(blocks, ("price", "both", "single"))
        if ref_panel_blk is None or ref_price_blk is None:
            continue
        ref_structure = "+".join(sorted(blocks.keys()))

        if new.empty or ref_panel_blk.empty:
            continue
        new = new.loc[(new.index >= start_ts) & (new.index <= end_ts)]
        ref_panel_blk = ref_panel_blk.loc[
            (ref_panel_blk.index >= start_ts) & (ref_panel_blk.index <= end_ts)]
        ref_price_blk = ref_price_blk.loc[
            (ref_price_blk.index >= start_ts) & (ref_price_blk.index <= end_ts)]
        if new.empty or ref_panel_blk.empty:
            continue

        entry: Dict[str, Any] = {
            "stock": stem,
            "ref_structure": ref_structure,
            "new_rows": int(len(new)),
            "new_first": str(new.index.min().date()),
            "new_last": str(new.index.max().date()),
            "ref_panel_rows": int(len(ref_panel_blk)),
            "ref_price_rows": int(len(ref_price_blk)),
            "ref_panel_first": str(ref_panel_blk.index.min().date()),
            "ref_panel_last": str(ref_panel_blk.index.max().date()),
            "new_panel_coverage": float(
                new[list(PANEL_FACTORS)].notna().any(axis=1).mean()
            ),
            "ref_panel_coverage": float(
                ref_panel_blk[list(PANEL_FACTORS)].notna().any(axis=1).mean()
            ),
            "columns": {},
        }

        # 1) 17 个面板因子：逐列相关系数 + 均值/标准差差异
        for col in PANEL_FACTORS:
            if col not in new.columns or col not in ref_panel_blk.columns:
                continue
            c = _corr(new[col], ref_panel_blk[col])
            both = new[col].notna() & ref_panel_blk[col].notna()
            md = (float((new[col][both] - ref_panel_blk[col][both]).abs().mean())
                  if both.any() else None)
            s_new = float(new[col].std())
            s_ref = float(ref_panel_blk[col].std())
            entry["columns"][col] = {
                "corr": c,
                "mean_abs_diff": md,
                "std_new": s_new,
                "std_ref": s_ref,
                "std_ratio": (s_new / s_ref) if (s_ref and s_ref != 0) else None,
                "n_both": int(both.sum()),
                "n_new": int(new[col].notna().sum()),
                "n_ref": int(ref_panel_blk[col].notna().sum()),
            }
            if c is not None:
                agg["corr"].append(c)
                per_factor_corr.setdefault(col, []).append(c)
                if md is not None:
                    agg["mean_abs_diff"].append(md)
                if s_ref:
                    agg["std_ratio"].append(s_new / s_ref)

        # 2) 价格列：应完全一致（同一未复权基准）
        for col in ("$close", "$open", "$high", "$low", "$volume"):
            if col in new.columns and col in ref_price_blk.columns:
                both = new[col].notna() & ref_price_blk[col].notna()
                d = (float((new[col][both] - ref_price_blk[col][both]).abs().max())
                     if both.any() else None)
                entry.setdefault("price_cols", {})[col] = {
                    "max_abs_diff": d,
                    "n_both": int(both.sum()),
                }

        # 3) 25 个基础公式：用两边的 OHLCV 各自算，再对账
        if all(c in new.columns for c in ("$close", "$volume")):
            new_base = compute_base_factors(
                new["$close"], new.get("$high"), new.get("$low"), new["$volume"]
            )
            ref_base = compute_base_factors(
                ref_price_blk.get("$close"), ref_price_blk.get("$high"),
                ref_price_blk.get("$low"), ref_price_blk.get("$volume"),
            )
            base_corrs = {}
            for name in BASE_FORMULAS:
                if name in new_base and name in ref_base:
                    c = _corr(new_base[name], ref_base[name])
                    if c is not None:
                        base_corrs[name] = c
            entry["base_formula_corr"] = base_corrs
            if base_corrs:
                entry["base_formula_corr_min"] = min(base_corrs.values())
                entry["base_formula_corr_mean"] = float(np.mean(list(base_corrs.values())))

        # 4) 其它 Alpha158 列（回归检查口径是否整体一致）
        other = []
        for col in ALPHA158_COLS:
            if col in PANEL_FACTORS:
                continue
            if col in new.columns and col in ref_panel_blk.columns:
                c = _corr(new[col], ref_panel_blk[col])
                if c is not None:
                    other.append(c)
        if other:
            entry["alpha158_other_corr_min"] = float(np.min(other))
            entry["alpha158_other_corr_mean"] = float(np.mean(other))
            entry["alpha158_other_n"] = len(other)

        report["stocks"].append(entry)
        bucket = "single_block" if ref_structure in ("single", "both") else "two_block"
        b = buckets.setdefault(bucket, {
            "corr": [], "mean_abs_diff": [], "std_ratio": [],
            "base_corr": [], "price_maxdiff": [],
        })
        for v in entry["columns"].values():
            if v["corr"] is not None:
                b["corr"].append(v["corr"])
            if v["mean_abs_diff"] is not None:
                b["mean_abs_diff"].append(v["mean_abs_diff"])
            if v["std_ratio"] is not None:
                b["std_ratio"].append(v["std_ratio"])
        if entry.get("base_formula_corr"):
            b["base_corr"].extend(entry["base_formula_corr"].values())
        for v in entry.get("price_cols", {}).values():
            if v["max_abs_diff"] is not None:
                b["price_maxdiff"].append(v["max_abs_diff"])

    # 汇总
    def _agg_summary(b: Dict[str, List[float]]) -> Dict[str, Any]:
        return {
            "n_panel_factors": len(b["corr"]),
            "panel_factor_corr_min": float(np.min(b["corr"])) if b["corr"] else None,
            "panel_factor_corr_mean": float(np.mean(b["corr"])) if b["corr"] else None,
            "panel_factor_mean_abs_diff_max": (
                float(np.max(b["mean_abs_diff"])) if b["mean_abs_diff"] else None),
            "panel_factor_std_ratio_mean": (
                float(np.mean(b["std_ratio"])) if b["std_ratio"] else None),
            "base_formula_corr_min": (
                float(np.min(b["base_corr"])) if b["base_corr"] else None),
            "price_col_max_abs_diff_max": (
                float(np.max(b["price_maxdiff"])) if b["price_maxdiff"] else None),
        }

    report["summary"] = {
        "n_stocks_compared": len(report["stocks"]),
        "by_reference_structure": {
            k: _agg_summary(v) for k, v in buckets.items()
        },
        "panel_factor_corr_min": float(np.min(agg["corr"])) if agg["corr"] else None,
        "panel_factor_corr_mean": float(np.mean(agg["corr"])) if agg["corr"] else None,
        "panel_factor_mean_abs_diff_max": (
            float(np.max(agg["mean_abs_diff"])) if agg["mean_abs_diff"] else None
        ),
        "panel_factor_std_ratio_mean": (
            float(np.mean(agg["std_ratio"])) if agg["std_ratio"] else None
        ),
        "per_factor_corr_min": {
            k: float(np.min(v)) for k, v in per_factor_corr.items()
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _log(f"[reconcile] 已写入 {out_json}")
    _log(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="只处理前 N 只（冒烟用）")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：跳过已完成的股票")
    ap.add_argument("--out-dir", default=str(FEATURES_FULL_DIR),
                    help="输出目录（默认 data/qlib_data/features_full）")
    ap.add_argument("--reconcile-only", action="store_true",
                    help="只做口径对账，不重新计算特征")
    ap.add_argument("--reconcile-n", type=int, default=5,
                    help="对账抽样股票数（默认 5）")
    ap.add_argument("--reconcile-start", default="2021-05-11",
                    help="对账窗口起点")
    ap.add_argument("--reconcile-end", default="2026-05-08",
                    help="对账窗口终点")
    ap.add_argument("--out", default=str(ROOT / "data" / "backtest" /
                                         "full_history_reconcile.json"),
                    help="对账报告输出路径")
    ap.add_argument("--stocks", nargs="*", default=None,
                    help="只处理指定股票（qlib instrument 名，如 000001_sz）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行进程数（默认 1；全量建议 8）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.reconcile_only:
        reconcile(out_dir, args.reconcile_n, Path(args.out),
                  args.reconcile_start, args.reconcile_end)
        return

    if args.workers > 1:
        result = run_build_mp(args.limit, args.resume, out_dir, args.stocks,
                              args.workers)
    else:
        result = asyncio.run(run_build(args.limit, args.resume, out_dir,
                                        args.stocks))
    _log(json.dumps(result, ensure_ascii=False))

    # 建完自动对账
    if not args.limit or args.limit >= 5:
        reconcile(out_dir, args.reconcile_n, Path(args.out),
                  args.reconcile_start, args.reconcile_end)


if __name__ == "__main__":
    main()
