#!/usr/bin/env python3
"""Fetch and save a subset of stock data for benchmarking.

Default: 100 stocks × 3 years.

Usage:
  cd back_test_data_service
  export TUSHARE_TOKEN=...
  ./venv/bin/python scripts/fetch_test_data_subset.py --n 100 --years 3
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_service.fetcher import DataFetcher
from data_service.config import Config


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--start", type=str, default=None, help="YYYYMMDD (override years)")
    ap.add_argument("--end", type=str, default=None, help="YYYYMMDD")
    args = ap.parse_args()

    if not Config.validate():
        print("Config validate failed", file=sys.stderr)
        return 1

    fetcher = DataFetcher()
    stock_list = fetcher.dao.get_stock_list()
    if not stock_list:
        print("Stock list empty; run init_stock_list.py first", file=sys.stderr)
        return 1

    n = min(int(args.n), len(stock_list))
    subset = stock_list[:n]

    if args.start and args.end:
        start_date_str = args.start
        end_date_str = args.end
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=float(args.years) * 365)
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

    print(f"Fetching {n} stocks, range {start_date_str}-{end_date_str}")

    ok = 0
    fail = 0
    for i, stock_info in enumerate(subset, 1):
        ts_code = stock_info["ts_code"]
        name = stock_info.get("name", "")
        try:
            r = fetcher.fetch_and_save_stock_data(
                ts_code=ts_code, start_date=start_date_str, end_date=end_date_str
            )
            if r is True:
                ok += 1
            elif r is False:
                fail += 1
            # None: no data -> not counted as fail
        except Exception as e:
            fail += 1
            print(f"[{i}/{n}] FAIL {ts_code} {name}: {e}", file=sys.stderr)
        if i % 10 == 0:
            print(f"progress {i}/{n} ok={ok} fail={fail}")

    print(f"done ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
