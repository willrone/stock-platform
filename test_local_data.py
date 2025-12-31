#!/usr/bin/env python3
"""
测试本地数据加载功能
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# 添加backend路径
sys.path.insert(0, 'backend')

# 设置环境变量
os.environ['PYTHONPATH'] = 'backend'

from app.services.data_service import StockDataService
import asyncio

async def test_local_data():
    """测试本地数据加载"""
    
    service = StockDataService()
    
    print(f"数据路径: {service.data_path}")
    
    # 检查文件是否存在
    stock_code = "000001.SZ"
    year = 2024
    parquet_path = service.get_local_parquet_path(stock_code, year)
    
    print(f"Parquet文件路径: {parquet_path}")
    print(f"文件是否存在: {parquet_path.exists()}")
    
    if parquet_path.exists():
        print(f"文件大小: {parquet_path.stat().st_size} bytes")
    
    # 尝试加载数据
    start_date = datetime(2024, 12, 1)
    end_date = datetime(2024, 12, 31)
    
    try:
        df = await service.load_from_local(stock_code, start_date, end_date)
        if df is not None:
            print(f"✅ 成功加载数据: {len(df)} 条记录")
            print(f"日期范围: {df.index.min()} 至 {df.index.max()}")
        else:
            print("❌ 加载数据失败: 返回None")
    except Exception as e:
        print(f"❌ 加载数据异常: {e}")

if __name__ == "__main__":
    asyncio.run(test_local_data())