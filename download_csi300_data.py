"""
Download CSI300 stock data using akshare and save as JSON files
for the training system to use.
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime

import akshare as ak
import pandas as pd

# Target: download 80 CSI300 stocks, 2020-01-01 to 2024-12-31
TARGET_COUNT = 80
START_DATE = "20200101"
END_DATE = "20241231"
OUTPUT_DIR = "backend/data/stocks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get CSI300 components
print("Fetching CSI300 components...")
df_cons = ak.index_stock_cons_csindex(symbol="000300")
codes_raw = df_cons['成分券代码'].tolist()
exchanges = df_cons['交易所'].tolist()

# Convert to ts_code format
all_ts_codes = []
code_to_akcode = {}  # ts_code -> akshare code (6-digit)
for code, exchange in zip(codes_raw, exchanges):
    if '深圳' in exchange:
        ts_code = f"{code}.SZ"
    else:
        ts_code = f"{code}.SH"
    all_ts_codes.append(ts_code)
    code_to_akcode[ts_code] = code

# Pick first TARGET_COUNT stocks (they're already sorted by weight roughly)
selected = all_ts_codes[:TARGET_COUNT]
print(f"Selected {len(selected)} stocks for download")

success_count = 0
fail_count = 0
failed_codes = []

for i, ts_code in enumerate(selected):
    output_file = os.path.join(OUTPUT_DIR, f"{ts_code}.json")
    
    # Skip if already downloaded
    if os.path.exists(output_file):
        # Check if file has enough data
        try:
            with open(output_file) as f:
                existing = json.load(f)
            if len(existing) > 500:
                print(f"[{i+1}/{len(selected)}] {ts_code}: already exists ({len(existing)} records), skipping")
                success_count += 1
                continue
        except:
            pass
    
    ak_code = code_to_akcode[ts_code]
    print(f"[{i+1}/{len(selected)}] Downloading {ts_code} (akshare: {ak_code})...", end=" ")
    
    try:
        # Use akshare to get daily data
        # stock_zh_a_hist returns: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
        df = ak.stock_zh_a_hist(
            symbol=ak_code,
            period="daily",
            start_date=START_DATE,
            end_date=END_DATE,
            adjust="qfq"  # 前复权
        )
        
        if df is None or df.empty:
            print(f"EMPTY")
            fail_count += 1
            failed_codes.append(ts_code)
            continue
        
        # Convert to the format expected by the system
        records = []
        for _, row in df.iterrows():
            records.append({
                "stock_code": ts_code,
                "date": pd.Timestamp(row['日期']).isoformat(),
                "open": float(row['开盘']),
                "high": float(row['最高']),
                "low": float(row['最低']),
                "close": float(row['收盘']),
                "volume": int(row['成交量']),
                "adj_close": float(row['收盘'])  # already adjusted
            })
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(records, f, ensure_ascii=False)
        
        print(f"OK ({len(records)} records, {df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]})")
        success_count += 1
        
        # Rate limiting - akshare may throttle
        time.sleep(0.5)
        
    except Exception as e:
        print(f"FAILED: {e}")
        fail_count += 1
        failed_codes.append(ts_code)
        time.sleep(1)

print(f"\n=== Download Summary ===")
print(f"Success: {success_count}/{len(selected)}")
print(f"Failed: {fail_count}")
if failed_codes:
    print(f"Failed codes: {failed_codes}")
print(f"Data saved to: {OUTPUT_DIR}/")
