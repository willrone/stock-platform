"""通达信盘后包增量拉取 + 存储。

存储布局:
    data/tdx_daily/raw/date=YYYY-MM-DD/part.parquet   每日全市场不复权价
    data/tdx_daily/adjust_factors/factors.parquet      复权因子长表
    data/tdx_daily/state.json                          断点续传状态

关键设计:
- **存不复权价，不存前复权价**。现有 data/parquet/stock_data 已验证是不复权价
  （600000 2026-05-08: stock_data=9.07，通达信不复权=9.07，完全一致）。
  存不复权价 + 因子表 → 增量拉取不会让历史数据漂移。
- volume 存"手"（现有 stock_data 是手: 892690.33 = 通达信 89269033股/100），
  但本模块 raw 层存**股**（通达信原值），merge 到 stock_data 时再换算。
- 按日期分区，每天一个 parquet，天然支持增量与断点续传。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from .tdx_parser import TdxBar, fetch_daily, is_trading_day

# 项目根目录（本文件在 back_test_data_service/tdx_pipeline/ 下）
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "tdx_daily", "raw")
FACTOR_DIR = os.path.join(PROJECT_ROOT, "data", "tdx_daily", "adjust_factors")
STATE_PATH = os.path.join(PROJECT_ROOT, "data", "tdx_daily", "state.json")


@dataclass
class PipelineState:
    """断点续传状态。记录已成功拉取的日期与失败日期。"""
    fetched_dates: List[str] = field(default_factory=list)
    failed_dates: Dict[str, str] = field(default_factory=dict)  # date -> 错误信息
    last_run_at: Optional[str] = None

    def is_fetched(self, date_str: str) -> bool:
        return date_str in self.fetched_dates

    def mark_fetched(self, date_str: str) -> None:
        if date_str not in self.fetched_dates:
            self.fetched_dates.append(date_str)
        self.failed_dates.pop(date_str, None)

    def mark_failed(self, date_str: str, error: str) -> None:
        self.failed_dates[date_str] = error


def load_state() -> PipelineState:
    """加载状态，不存在则返回空状态。"""
    if not os.path.exists(STATE_PATH):
        return PipelineState()
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PipelineState(**data)


def save_state(state: PipelineState) -> None:
    """保存状态（原子写）。"""
    state.last_run_at = datetime.now().isoformat()
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)


def bars_to_dataframe(bars: List[TdxBar]) -> pd.DataFrame:
    """TdxBar 列表 → DataFrame，列对齐现有 stock_data 口径。

    注意: 此处 volume 为**股**（通达信原值），merge 到 stock_data 时需 /100 转手。
    """
    rows = []
    for b in bars:
        rows.append({
            "ts_code": b.ts_code,
            "date": pd.Timestamp(b.date),
            "market": b.market,
            "code": b.code,
            "name": b.name,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "prev_close": b.prev_close,
            "volume": b.volume,      # 股
            "amount": b.amount,      # 元
            "is_suspended": b.is_suspended,
        })
    return pd.DataFrame(rows)


def date_partition_path(date_str: str) -> str:
    """某交易日的分区 parquet 路径。"""
    return os.path.join(RAW_DIR, f"date={date_str}", "part.parquet")


def has_partition(date_str: str) -> bool:
    """该日期分区是否已存在。"""
    return os.path.exists(date_partition_path(date_str))


def fetch_and_store(date_str: str, force: bool = False) -> int:
    """拉取某交易日并存为分区 parquet。返回行数。

    Args:
        date_str: YYYY-MM-DD
        force: 已存在时是否强制重拉

    Returns:
        解析出的行数（含停牌记录）

    Raises:
        FileNotFoundError: 非交易日/包未发布
        RuntimeError: 包残缺/格式变化
        ConnectionError: 网络错误
    """
    if has_partition(date_str) and not force:
        # 已存在，直接读行数
        df = pd.read_parquet(date_partition_path(date_str))
        return len(df)

    bars = fetch_daily(date_str)
    df = bars_to_dataframe(bars)

    out = date_partition_path(date_str)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False, compression="snappy")
    return len(df)


def trading_days_between(start: str, end: str) -> List[str]:
    """生成 [start, end] 闭区间内的工作日列表（周一至周五）。

    注意: 这只是粗筛，节假日需实际请求盘后包验证（404 = 非交易日）。
    """
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    days = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:  # 周一至周五
            days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def run_incremental(
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_days: Optional[int] = None,
    force: bool = False,
    sleep_between: float = 0.3,
    verbose: bool = True,
) -> Dict[str, float]:
    """增量拉取主入口。

    Args:
        start: 起始日 YYYY-MM-DD，缺省用已拉取的最后一天 +1
        end: 结束日，缺省用今天
        max_days: 最多处理多少个工作日（防误操作拉爆）
        force: 强制重拉已存在的分区
        sleep_between: 每天之间间隔秒数（防封 IP）
        verbose: 是否打印进度

    Returns:
        {"fetched": N, "skipped": N, "failed": N, "non_trading": N}
    """
    state = load_state()

    if end is None:
        end = date.today().isoformat()
    if start is None:
        if state.fetched_dates:
            last = max(state.fetched_dates)
            start = (datetime.strptime(last, "%Y-%m-%d").date()
                     + timedelta(days=1)).isoformat()
        else:
            # 未拉过，默认从盘后包可用起点开始
            start = "2021-08-16"

    candidates = trading_days_between(start, end)
    if max_days is not None:
        candidates = candidates[:max_days]

    stats: Dict[str, float] = {"fetched": 0, "skipped": 0, "failed": 0, "non_trading": 0}
    t0 = time.time()

    for i, day in enumerate(candidates):
        if not force and state.is_fetched(day) and has_partition(day):
            stats["skipped"] += 1
            continue

        try:
            n = fetch_and_store(day, force=force)
            state.mark_fetched(day)
            stats["fetched"] += 1
            if verbose:
                print(f"  [{i+1}/{len(candidates)}] {day}: {n} 行  "
                      f"(累计耗时 {time.time()-t0:.0f}s)")
        except FileNotFoundError as e:
            # 非交易日/包未发布 — 正常情况，不算失败
            stats["non_trading"] += 1
            if verbose and "非交易日" in str(e):
                pass  # 静默跳过节假日
            else:
                if verbose:
                    print(f"  [{i+1}/{len(candidates)}] {day}: 非交易日/未发布")
        except Exception as e:
            stats["failed"] += 1
            state.mark_failed(day, f"{type(e).__name__}: {e}")
            if verbose:
                print(f"  [{i+1}/{len(candidates)}] {day}: 失败 {type(e).__name__}: {e}")

        # 每处理 20 天保存一次状态，中断可续
        if (i + 1) % 20 == 0:
            save_state(state)

        if sleep_between > 0:
            time.sleep(sleep_between)

    save_state(state)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    if verbose:
        print(f"\n完成: 拉取 {stats['fetched']} 天, 跳过 {stats['skipped']} 天, "
              f"失败 {stats['failed']} 天, 非交易日 {stats['non_trading']} 天, "
              f"耗时 {stats['elapsed_s']}s")
    return stats


def loaded_dates() -> List[str]:
    """列出已存在的分区日期（升序）。"""
    if not os.path.isdir(RAW_DIR):
        return []
    dates = []
    for name in os.listdir(RAW_DIR):
        if name.startswith("date="):
            d = name[5:]
            if has_partition(d):
                dates.append(d)
    return sorted(dates)


def load_range(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """加载 [start, end] 区间的全部日线数据。"""
    dates = loaded_dates()
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]

    frames = []
    for d in dates:
        df = pd.read_parquet(date_partition_path(d))
        # 丢掉辅助列，只留核心
        frames.append(df[["ts_code", "date", "market", "code", "name",
                          "open", "high", "low", "close", "volume", "amount",
                          "is_suspended"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
