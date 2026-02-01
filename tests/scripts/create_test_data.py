#!/usr/bin/env python3
"""
创建测试数据文件
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_test_stock_data():
    """创建测试股票数据"""
    
    # 创建数据目录
    data_dir = Path("data/daily/000001.SZ")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成2024年的测试数据
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # 生成日期序列（只包含工作日）
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    # 生成模拟股票数据
    np.random.seed(42)  # 确保可重复性
    
    n_days = len(dates)
    base_price = 10.0
    
    # 生成价格数据（随机游走）
    returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
    prices = [base_price]
    
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 0.1))  # 确保价格不为负
    
    # 创建DataFrame
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        # 生成开盘价、最高价、最低价
        open_price = close * (1 + np.random.normal(0, 0.005))
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.lognormal(15, 0.5))  # 成交量
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume,
            'adj_close': round(close, 2)
        })
    
    df = pd.DataFrame(data)
    
    # 保存为Parquet文件
    output_file = data_dir / "2024.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"✅ 创建测试数据成功: {output_file}")
    print(f"   数据条数: {len(df)}")
    print(f"   日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"   价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")

if __name__ == "__main__":
    create_test_stock_data()