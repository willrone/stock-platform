#!/usr/bin/env python3
"""测试数据路径迁移后的数据加载"""

from pathlib import Path
from backend.app.services.data.stock_data_loader import StockDataLoader
from datetime import datetime

def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试数据路径迁移后的数据加载")
    print("=" * 60)
    
    # 测试数据加载器
    loader = StockDataLoader()
    print(f'\n数据根目录: {loader.data_root}')
    print(f'数据目录是否存在: {loader.data_root.exists()}')
    
    stock_data_dir = loader.data_root / "parquet" / "stock_data"
    print(f'股票数据目录: {stock_data_dir}')
    print(f'股票数据目录是否存在: {stock_data_dir.exists()}')
    
    if stock_data_dir.exists():
        files = list(stock_data_dir.glob("*.parquet"))
        print(f'股票数据文件数量: {len(files)}')
        if files:
            print(f'示例文件: {files[0].name}')
    
    # 测试加载一个股票
    print("\n" + "=" * 60)
    stock_code = '000001.SZ'
    print(f'测试加载股票: {stock_code}')
    print("=" * 60)
    
    try:
        data = loader.load_stock_data(
            stock_code, 
            start_date=datetime(2023, 1, 1), 
            end_date=datetime(2023, 12, 31)
        )
        
        if len(data) > 0:
            print(f'✅ 数据加载成功！')
            print(f'数据行数: {len(data)}')
            print(f'日期范围: {data.index[0]} 到 {data.index[-1]}')
            print(f'列名: {list(data.columns)}')
            print(f'\n前3行数据:')
            print(data.head(3))
        else:
            print('❌ 数据为空')
    except Exception as e:
        print(f'❌ 数据加载失败: {e}')
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_data_loading()
