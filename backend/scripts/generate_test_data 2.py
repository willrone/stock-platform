#!/usr/bin/env python3
"""
生成模拟股票数据用于性能测试
"""

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def generate_stock_data(stock_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    生成单只股票的模拟 OHLCV 数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    # 生成交易日（排除周末）
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]  # 只保留工作日
    
    n = len(dates)
    
    # 生成随机价格（模拟真实股票走势）
    np.random.seed(hash(stock_code) % (2**32))  # 每只股票有不同的随机种子
    
    base_price = np.random.uniform(10, 100)
    returns = np.random.normal(0.0005, 0.02, n)  # 日收益率
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # 生成 OHLC
    open_prices = close_prices * np.random.uniform(0.98, 1.02, n)
    high_prices = np.maximum(open_prices, close_prices) * np.random.uniform(1.0, 1.03, n)
    low_prices = np.minimum(open_prices, close_prices) * np.random.uniform(0.97, 1.0, n)
    
    # 生成成交量
    volumes = np.random.uniform(1e6, 1e8, n)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
    })
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成模拟股票数据")
    parser.add_argument('--num-stocks', type=int, default=500, help='股票数量')
    parser.add_argument('--years', type=int, default=3, help='数据年数')
    parser.add_argument('--output-dir', type=str, default='../data/parquet/stock_data', help='输出目录')
    
    args = parser.parse_args()
    
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"生成 {args.num_stocks} 只股票，{args.years} 年数据")
    print(f"日期范围: {start_date.date()} ~ {end_date.date()}")
    print(f"输出目录: {output_dir}")
    
    # 生成股票代码列表
    stock_codes = []
    for i in range(args.num_stocks):
        if i < args.num_stocks // 2:
            # 深圳股票
            code = f"{i:06d}.SZ"
        else:
            # 上海股票
            code = f"{(i - args.num_stocks // 2):06d}.SH"
        stock_codes.append(code)
    
    # 生成数据
    success = 0
    failed = 0
    
    for i, stock_code in enumerate(stock_codes, 1):
        try:
            df = generate_stock_data(stock_code, start_date, end_date)
            
            # 保存为 parquet
            safe_code = stock_code.replace('.', '_')
            output_file = output_dir / f"{safe_code}.parquet"
            df.to_parquet(output_file, index=False)
            
            success += 1
            
            if i % 50 == 0:
                print(f"进度: {i}/{args.num_stocks} ({i/args.num_stocks*100:.1f}%)")
                
        except Exception as e:
            print(f"生成失败 {stock_code}: {e}")
            failed += 1
    
    print(f"✅ 完成: 成功 {success}, 失败 {failed}")
    print(f"数据目录: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
