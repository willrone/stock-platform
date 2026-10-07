#!/usr/bin/env python3
"""
数据完整性校验脚本 - Data Integrity Verifier for Stock Data

用途：检查 parquet 格式存储的股票历史数据完整性。
覆盖文件可读性、字段完整性、日期连续性、缺失值、异常值等检查项。

返回码：
  0 - 数据完整，无错误
  1 - 存在数据问题

使用方式：
  python3 backend/scripts/verify_data_integrity.py
  python3 backend/scripts/verify_data_integrity.py --data-dir data/parquet/stock_data
  python3 backend/scripts/verify_data_integrity.py --data-dir data/parquet/stock_data --gap-threshold 10
  python3 backend/scripts/verify_data_integrity.py --verbose
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
"""必需字段列表。data 列可能名称不同（ts_code / stock_code），单独处理。"""

CODE_COLUMN_NAMES = {"ts_code", "stock_code"}
"""股票代码列的候选名称。"""

OHLC_COLUMNS = ["open", "high", "low", "close"]
"""OHLC 列名，用于一致性检查。"""

PRICE_COLUMNS = ["open", "high", "low", "close"]
"""价格列，用于零负值检查。"""

DEFAULT_GAP_THRESHOLD_DAYS = 5
"""自然日间隔超过此阈值视为 gap。"""

MAX_REPORT_SAMPLE = 10
"""每类问题在报告中最多展示的股票示例数。"""

# ---------------------------------------------------------------------------
# Report data structure
# ---------------------------------------------------------------------------


class StockIssue:
    """单只股票的数据问题记录。"""

    __slots__ = (
        "file",
        "ts_code",
        "rows",
        "file_size_kb",
        "missing_columns",
        "null_counts",
        "zero_neg_prices",
        "zero_neg_volume",
        "ohlc_inconsistencies",
        "gap_days",
        "max_gap",
        "errors",
    )

    def __init__(self, file: str, ts_code: str = ""):
        self.file = file
        self.ts_code = ts_code
        self.rows = 0
        self.file_size_kb = 0.0

        self.missing_columns: List[str] = []
        self.null_counts: Dict[str, int] = {}
        self.zero_neg_prices: List[str] = []
        self.zero_neg_volume: bool = False
        self.ohlc_inconsistencies: List[str] = []
        self.gap_days: int = 0
        self.max_gap: int = 0
        self.errors: List[str] = []

    def has_any_issue(self) -> bool:
        return bool(
            self.missing_columns
            or self.null_counts
            or self.zero_neg_prices
            or self.zero_neg_volume
            or self.ohlc_inconsistencies
            or self.max_gap > DEFAULT_GAP_THRESHOLD_DAYS
            or self.errors
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "file": self.file,
            "ts_code": self.ts_code,
            "rows": self.rows,
            "file_size_kb": round(self.file_size_kb, 1),
        }
        if self.missing_columns:
            d["missing_columns"] = self.missing_columns
        if self.null_counts:
            d["nulls"] = self.null_counts
        if self.zero_neg_prices:
            d["zero_neg_prices"] = self.zero_neg_prices[:MAX_REPORT_SAMPLE]
        if self.zero_neg_volume:
            d["zero_neg_volume"] = True
        if self.ohlc_inconsistencies:
            d["ohlc_inconsistencies"] = self.ohlc_inconsistencies[:MAX_REPORT_SAMPLE]
        if self.max_gap > DEFAULT_GAP_THRESHOLD_DAYS:
            d["gap_days"] = self.gap_days
            d["max_gap_days"] = self.max_gap
        if self.errors:
            d["errors"] = self.errors[:MAX_REPORT_SAMPLE]
        return d


class Report:
    """汇总报告。"""

    def __init__(self):
        self.start_time = time.time()
        self.data_dir = ""
        self.total_files = 0
        self.total_rows = 0
        self.readable_files = 0
        self.unreadable_files: List[str] = []
        self.stocks_with_issues: List[StockIssue] = []
        self.global_min_date: Optional[date] = None
        self.global_max_date: Optional[date] = None
        self.corrupt_files: List[str] = []

    def add_issue(self, si: StockIssue) -> None:
        self.stocks_with_issues.append(si)

    @property
    def files_with_issues(self) -> int:
        return sum(1 for s in self.stocks_with_issues if s.has_any_issue())

    def to_json(self, verbose: bool = False) -> str:
        elapsed = time.time() - self.start_time

        # Build a clean result dict
        issues_list = [
            s.to_dict() for s in self.stocks_with_issues if s.has_any_issue()
        ]

        # Summarize issue categories
        gap_stocks = [s for s in self.stocks_with_issues if s.max_gap > DEFAULT_GAP_THRESHOLD_DAYS]
        missing_col_stocks = [s for s in self.stocks_with_issues if s.missing_columns]
        null_stocks = [s for s in self.stocks_with_issues if s.null_counts]
        zero_price_stocks = [s for s in self.stocks_with_issues if s.zero_neg_prices]
        zero_vol_stocks = [s for s in self.stocks_with_issues if s.zero_neg_volume]
        ohlc_bad_stocks = [s for s in self.stocks_with_issues if s.ohlc_inconsistencies]

        result: Dict[str, Any] = {
            "summary": {
                "data_dir": self.data_dir,
                "total_files": self.total_files,
                "readable_files": self.readable_files,
                "unreadable_files": len(self.unreadable_files),
                "corrupt_files": len(self.corrupt_files),
                "total_rows": self.total_rows,
                "stocks_with_issues": len(issues_list),
                "issues_by_category": {
                    "missing_columns": len(missing_col_stocks),
                    "null_values": len(null_stocks),
                    "zero_neg_prices": len(zero_price_stocks),
                    "zero_neg_volume": len(zero_vol_stocks),
                    "ohlc_inconsistency": len(ohlc_bad_stocks),
                    "date_gaps": len(gap_stocks),
                },
                "date_range": {
                    "min": str(self.global_min_date) if self.global_min_date else None,
                    "max": str(self.global_max_date) if self.global_max_date else None,
                },
                "elapsed_seconds": round(elapsed, 2),
            },
            "details": {
                "unreadable_files": self.unreadable_files[:MAX_REPORT_SAMPLE],
                "corrupt_files": self.corrupt_files[:MAX_REPORT_SAMPLE],
                "stocks_with_issues": issues_list,
            },
        }

        # Add top-gap list if any
        if gap_stocks:
            top_gaps = sorted(gap_stocks, key=lambda s: -s.max_gap)[:MAX_REPORT_SAMPLE]
            result["details"]["top_gaps"] = [
                {"ts_code": s.ts_code, "file": s.file, "max_gap_days": s.max_gap, "total_gaps": s.gap_days}
                for s in top_gaps
            ]

        if not verbose:
            # Limit detail length in non-verbose mode
            if len(issues_list) > 50:
                result["details"]["stocks_with_issues"] = issues_list[:50]
                result["summary"]["note"] = (
                    f"Displaying 50 of {len(issues_list)} stocks with issues. "
                    "Use --verbose for full list."
                )

        return json.dumps(result, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_file_readable(filepath: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
    """尝试用 pandas 读取 parquet 文件。返回 (ok, df_or_None, error_or_None)。"""
    try:
        df = pd.read_parquet(filepath)
        return True, df, None
    except Exception as e:
        return False, None, str(e)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化列名：统一 stock_code / ts_code 为 ts_code。
    也处理可能的列名大小写问题。
    """
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in CODE_COLUMN_NAMES:
            rename_map[col] = "ts_code"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def check_fields(df: pd.DataFrame, si: StockIssue) -> None:
    """检查必需字段完整性。"""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        si.missing_columns = missing


def check_nulls(df: pd.DataFrame, si: StockIssue) -> None:
    """检查任何 NaN / None 值。"""
    nulls = df[REQUIRED_COLUMNS].isnull().sum()
    null_dict = {col: int(v) for col, v in nulls.items() if v > 0}
    if null_dict:
        si.null_counts = null_dict


def check_prices(df: pd.DataFrame, si: StockIssue) -> None:
    """检查价格 <= 0 或 volume <= 0。"""
    # Zero/negative prices
    for col in PRICE_COLUMNS:
        bad = df[col] <= 0
        if bad.any():
            count = int(bad.sum())
            si.zero_neg_prices.append(f"{col}: {count} rows")

    # Zero/negative volume
    if "volume" in df.columns:
        bad_vol = df["volume"] <= 0
        if bad_vol.any():
            si.zero_neg_volume = True


def check_ohlc_consistency(df: pd.DataFrame, si: StockIssue) -> None:
    """
    检查 OHLC 关系一致性：
    - high >= low
    - high >= open
    - high >= close
    - low <= open
    - low <= close
    """
    if not all(c in df.columns for c in OHLC_COLUMNS):
        return

    if not (df["high"] >= df["low"]).all():
        count = int((df["high"] < df["low"]).sum())
        si.ohlc_inconsistencies.append(f"high<low: {count} rows")

    if not (df["high"] >= df["open"]).all():
        count = int((df["high"] < df["open"]).sum())
        si.ohlc_inconsistencies.append(f"high<open: {count} rows")

    if not (df["high"] >= df["close"]).all():
        count = int((df["high"] < df["close"]).sum())
        si.ohlc_inconsistencies.append(f"high<close: {count} rows")

    if not (df["low"] <= df["open"]).all():
        count = int((df["low"] > df["open"]).sum())
        si.ohlc_inconsistencies.append(f"low>open: {count} rows")

    if not (df["low"] <= df["close"]).all():
        count = int((df["low"] > df["close"]).sum())
        si.ohlc_inconsistencies.append(f"low>close: {count} rows")


def check_date_continuity(df: pd.DataFrame, si: StockIssue, gap_threshold: int) -> None:
    """
    检查交易日间隔，标记连续自然日间隔大于 gap_threshold 的 gap。
    注意：A 股因停牌存在合法的大间隔，会在报告中标注。
    """
    if "date" not in df.columns:
        return

    if len(df) < 2:
        return

    df_sorted = df.sort_values("date")
    dates = df_sorted["date"]

    deltas = (dates[1:].values - dates[:-1].values).astype("timedelta64[D]").astype(int)
    big_gaps = deltas[deltas > gap_threshold]

    if len(big_gaps) > 0:
        si.gap_days = int(len(big_gaps))
        si.max_gap = int(big_gaps.max())


def verify_stock_file(filepath: str, gap_threshold: int) -> Optional[StockIssue]:
    """
    检查单只股票 parquet 数据的完整性。
    返回 StockIssue（始终返回，不论是否有问题），异常时返回 None。
    """
    filename = os.path.basename(filepath)
    si = StockIssue(file=filename)

    # File size
    try:
        si.file_size_kb = os.path.getsize(filepath) / 1024.0
    except OSError:
        si.file_size_kb = 0.0

    # Read
    ok, df, err = check_file_readable(filepath)
    if not ok:
        si.errors.append(f"无法读取: {err}")
        return si

    si.rows = len(df)

    # Normalize columns
    df = normalize_columns(df)

    # Extract ts_code if present, else derive from filename
    if "ts_code" in df.columns:
        # Use most common value
        si.ts_code = str(df["ts_code"].mode().iloc[0]) if not df["ts_code"].isnull().all() else ""
    else:
        # Derive from filename
        si.ts_code = filename.replace(".parquet", "").replace("_", ".")

    # Run checks
    check_fields(df, si)
    check_nulls(df, si)
    check_prices(df, si)
    check_ohlc_consistency(df, si)
    check_date_continuity(df, si, gap_threshold)

    return si


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def scan_data_dir(data_dir: str, gap_threshold: int = DEFAULT_GAP_THRESHOLD_DAYS) -> Report:
    """扫描数据目录，检查所有 parquet 文件的完整性。"""
    report = Report()
    report.data_dir = data_dir

    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"错误: 数据目录不存在: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all parquet files
    parquet_files = sorted(data_path.glob("*.parquet"))
    # Also check for files without .parquet extension if they look like data files
    # (Some files might be named 000001.SZ without extension)
    all_files = sorted(data_path.iterdir())
    non_parquet_data = [
        f.name
        for f in all_files
        if f.is_file()
        and not f.name.endswith(".parquet")
        and not f.name.startswith(".")
    ]

    report.total_files = len(parquet_files)
    print(f"📂 扫描 {data_dir}", file=sys.stderr)
    print(f"   发现 {len(parquet_files)} 个 parquet 文件", file=sys.stderr, end="")
    if non_parquet_data:
        print(f", {len(non_parquet_data)} 个非 parquet 数据文件（已跳过）", file=sys.stderr)
        if len(non_parquet_data) <= 10:
            print(f"   非 parquet 文件: {', '.join(non_parquet_data)}", file=sys.stderr)
        else:
            print(f"   非 parquet 文件 (前10): {', '.join(non_parquet_data[:10])}...", file=sys.stderr)
    else:
        print(file=sys.stderr)

    if not parquet_files:
        print("⚠️  没有找到 parquet 文件，请确认数据目录路径。", file=sys.stderr)
        return report

    # Process each file
    for i, pf in enumerate(parquet_files):
        # Progress indicator for large directories
        if (i + 1) % 500 == 0 or (i + 1) == len(parquet_files):
            print(f"   进度: {i + 1}/{len(parquet_files)} ...", file=sys.stderr)

        filepath = str(pf)

        si = verify_stock_file(filepath, gap_threshold)
        if si is None:
            report.corrupt_files.append(pf.name)
            report.unreadable_files.append(pf.name)
            continue

        if si.errors:
            report.unreadable_files.append(pf.name)
        else:
            report.readable_files += 1

        report.total_rows += si.rows
        report.add_issue(si)

        # Track global date range if we have the data
        if not si.errors and si.rows > 0:
            try:
                df = pd.read_parquet(filepath)
                df = normalize_columns(df)
                if "date" in df.columns:
                    d_min = df["date"].min()
                    d_max = df["date"].max()
                    if isinstance(d_min, pd.Timestamp):
                        d_min = d_min.date()
                    if isinstance(d_max, pd.Timestamp):
                        d_max = d_max.date()
                    if report.global_min_date is None or d_min < report.global_min_date:
                        report.global_min_date = d_min
                    if report.global_max_date is None or d_max > report.global_max_date:
                        report.global_max_date = d_max
            except Exception:
                pass

    return report


def print_summary(report: Report, verbose: bool = False) -> None:
    """在终端输出人类可读的摘要。"""
    issues = [s for s in report.stocks_with_issues if s.has_any_issue()]
    gap_stocks = [s for s in issues if s.max_gap > DEFAULT_GAP_THRESHOLD_DAYS]
    missing_col = [s for s in issues if s.missing_columns]
    null_stocks = [s for s in issues if s.null_counts]
    zero_prices = [s for s in issues if s.zero_neg_prices]
    zero_vols = [s for s in issues if s.zero_neg_volume]
    ohlc_bad = [s for s in issues if s.ohlc_inconsistencies]
    unreadable = report.unreadable_files
    corrupt = report.corrupt_files

    print()
    print("=" * 60)
    print("📊  数据完整性校验报告")
    print("=" * 60)
    print(f"  数据目录:      {report.data_dir}")
    print(f"  总 parquet 数: {report.total_files}")
    print(f"  可读取:        {report.readable_files}")
    print(f"  不可读取:      {len(unreadable)}")
    print(f"  总数据行数:    {report.total_rows:,}")
    print(f"  日期范围:      {report.global_min_date} ~ {report.global_max_date}")
    print(f"  检查耗时:      {time.time() - report.start_time:.2f}s")
    print()

    # Issue summary
    print("── 问题分类 ──")
    print(f"  缺失字段:          {len(missing_col)} 只股票")
    print(f"  空值:              {len(null_stocks)} 只股票")
    print(f"  异常价格(<=0):     {len(zero_prices)} 只股票")
    print(f"  异常成交量(<=0):   {len(zero_vols)} 只股票")
    print(f"  OHLC 不一致:       {len(ohlc_bad)} 只股票")
    print(f"  日期不连续(>{DEFAULT_GAP_THRESHOLD_DAYS}d): {len(gap_stocks)} 只股票")
    print(f"  无法读取:          {len(unreadable)} 只股票")
    print()

    # Unreadable / corrupt
    if unreadable:
        print("── 无法读取的文件 ──")
        for f in unreadable[:MAX_REPORT_SAMPLE]:
            print(f"  ⛔ {f}")
        if len(unreadable) > MAX_REPORT_SAMPLE:
            print(f"  ... 还有 {len(unreadable) - MAX_REPORT_SAMPLE} 个")
        print()

    # Missing columns
    if missing_col:
        print("── 缺失必需字段 ──")
        for s in missing_col[:MAX_REPORT_SAMPLE]:
            print(f"  ⚠️  {s.file} ({s.ts_code}): 缺失 {s.missing_columns}")
        if len(missing_col) > MAX_REPORT_SAMPLE:
            print(f"  ... 还有 {len(missing_col) - MAX_REPORT_SAMPLE} 只股票")
        print()

    # Nulls
    if null_stocks:
        print("── 含空值 ──")
        for s in null_stocks[:MAX_REPORT_SAMPLE]:
            print(f"  ⚠️  {s.file} ({s.ts_code}): 空值 {s.null_counts}")
        if len(null_stocks) > MAX_REPORT_SAMPLE:
            print(f"  ... 还有 {len(null_stocks) - MAX_REPORT_SAMPLE} 只股票")
        print()

    # Zero/negative prices
    if zero_prices:
        print("── 异常价格(<=0) ──")
        for s in zero_prices[:MAX_REPORT_SAMPLE]:
            print(f"  ⚠️  {s.file} ({s.ts_code}): {s.zero_neg_prices}")
        if len(zero_prices) > MAX_REPORT_SAMPLE:
            print(f"  ... 还有 {len(zero_prices) - MAX_REPORT_SAMPLE} 只股票")
        print()

    # Zero volume
    if zero_vols:
        print("── 异常成交量(<=0) ──")
        for s in zero_vols[:MAX_REPORT_SAMPLE]:
            print(f"  ⚠️  {s.file} ({s.ts_code}): 存在零/负成交量")
        if len(zero_vols) > MAX_REPORT_SAMPLE:
            print(f"  ... 还有 {len(zero_vols) - MAX_REPORT_SAMPLE} 只股票")
        print()

    # OHLC inconsistency
    if ohlc_bad:
        print("── OHLC 不一致 ──")
        for s in ohlc_bad[:MAX_REPORT_SAMPLE]:
            for line in s.ohlc_inconsistencies:
                print(f"  ⚠️  {s.file} ({s.ts_code}): {line}")
        if len(ohlc_bad) > MAX_REPORT_SAMPLE:
            print(f"  ... 还有 {len(ohlc_bad) - MAX_REPORT_SAMPLE} 只股票")
        print()

    # Date gaps - top 10
    if gap_stocks:
        print(f"── 日期不连续 (top {min(MAX_REPORT_SAMPLE, len(gap_stocks))}) ──")
        top = sorted(gap_stocks, key=lambda s: -s.max_gap)[:MAX_REPORT_SAMPLE]
        for s in top:
            print(f"  ⚠️  {s.file} ({s.ts_code}): {s.gap_days} 次 gap，最大 {s.max_gap} 天")
        print("  ℹ️  注：部分 gap 可能由股票停牌引起，不一定表示数据错误。")
        print()

    # Final verdict
    total_issues = report.files_with_issues
    if total_issues == 0 and not unreadable:
        print("✅ 数据完整性检查通过，未发现问题。")
    elif total_issues == 0 and unreadable:
        print("⚠️  有无法读取的文件，但可读取的数据未发现问题。")
    else:
        print(f"❌ 发现 {total_issues} 只股票存在数据问题。")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="股票 Parquet 数据完整性校验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  %(prog)s                                        # 默认路径\n"
            "  %(prog)s --data-dir data/parquet/stock_data     # 指定目录\n"
            "  %(prog)s --gap-threshold 10                     # 允许更大的 gap\n"
            "  %(prog)s --verbose                              # 完整报告\n"
            "  %(prog)s --json-only                            # 仅输出 JSON\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/stocks/",
        help="数据目录（默认: data/stocks/）",
    )
    parser.add_argument(
        "--gap-threshold",
        type=int,
        default=DEFAULT_GAP_THRESHOLD_DAYS,
        help=f"日期 gap 阈值（自然日，默认 {DEFAULT_GAP_THRESHOLD_DAYS}）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出完整详细信息",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="仅输出 JSON 报告，不输出人类可读摘要",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="项目根目录（用于解析相对路径，默认自动检测）",
    )
    args = parser.parse_args()

    # Resolve data directory path
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        if args.project_root:
            base = Path(args.project_root)
        else:
            # Auto-detect project root: this script's parent is backend/scripts/
            base = Path(__file__).resolve().parent.parent.parent
        data_dir = str((base / data_dir).resolve())

    # Resolve relative to CWD if path doesn't exist
    if not os.path.isdir(data_dir):
        cwd_candidate = str((Path.cwd() / args.data_dir).resolve())
        if os.path.isdir(cwd_candidate):
            data_dir = cwd_candidate
        else:
            print(f"错误: 找不到数据目录 '{args.data_dir}' (尝试过: {data_dir}, {cwd_candidate})", file=sys.stderr)
            return 1

    # Run verification
    report = scan_data_dir(data_dir, args.gap_threshold)

    # Output human-readable summary
    if not args.json_only:
        # Redirect stdout -> stderr so human-readable summary doesn't pollute JSON pipe
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        print_summary(report, verbose=args.verbose)
        sys.stdout = old_stdout

    # JSON report always goes to stdout for clean piping
    json_report = report.to_json(verbose=args.verbose)
    print(json_report)

    # Determine exit code
    has_issues = (
        report.files_with_issues > 0
        or len(report.unreadable_files) > 0
        or len(report.corrupt_files) > 0
    )
    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
