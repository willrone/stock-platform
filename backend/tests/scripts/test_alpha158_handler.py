"""
测试Alpha158 handler能否计算所有158个因子
"""
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.insert(0, '/Users/ronghui/Projects/willrone/backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')

from app.services.qlib.enhanced_qlib_provider import Alpha158Calculator
from app.core.config import settings


async def test_handler():
    """测试handler计算所有158个因子"""
    # 加载真实数据
    parquet_path = Path('/Users/ronghui/Projects/willrone/backend/data/parquet/stock_data/002463_SZ.parquet')
    data = pd.read_parquet(parquet_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data[['open', 'high', 'low', 'close', 'volume']]
    
    # 转换为qlib格式（带$前缀）
    data.columns = [f'${col}' for col in data.columns]
    
    # 保存到qlib features目录
    qlib_features_dir = Path(settings.QLIB_DATA_PATH) / "features" / "day"
    qlib_features_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据（使用标准qlib格式：日期索引，列名为open/high/low/close/volume，不带$前缀）
    to_save = data.copy()
    to_save.columns = [col.lstrip('$') for col in to_save.columns]
    safe_code = '002463.SZ'.replace('.', '_')
    temp_file = qlib_features_dir / f"{safe_code}.parquet"
    to_save.to_parquet(temp_file, compression='snappy', index=True)
    print(f"✓ 保存数据到: {temp_file}")
    
    # 确保instruments/all.txt包含该股票
    instruments_file = Path(settings.QLIB_DATA_PATH) / "instruments" / "all.txt"
    instruments_file.parent.mkdir(parents=True, exist_ok=True)
    if instruments_file.exists():
        with open(instruments_file, 'r') as f:
            instruments = set(line.strip() for line in f if line.strip())
    else:
        instruments = set()
    
    instruments.add('002463.SZ')
    with open(instruments_file, 'w') as f:
        for inst in sorted(instruments):
            f.write(f"{inst}\n")
    print(f"✓ 更新instruments文件: {instruments_file}")
    
    # 使用handler计算
    calculator = Alpha158Calculator()
    start_date = data.index.min().to_pydatetime()
    end_date = data.index.max().to_pydatetime()
    
    print(f"\n使用handler计算因子...")
    factors = await calculator.calculate_alpha_factors(
        qlib_data=data,
        stock_codes=['002463.SZ'],
        date_range=(start_date, end_date),
        use_cache=False,
        force_expression_engine=False  # 使用handler
    )
    
    print(f"\n计算结果:")
    print(f"  总因子数: {len(factors.columns)}")
    print(f"  期望因子数: 158")
    print(f"  成功率: {len(factors.columns) / 158 * 100:.2f}%")
    
    if len(factors.columns) < 158:
        print(f"  ⚠ 因子数不足，缺失: {158 - len(factors.columns)} 个")
        # 列出缺失的因子
        if hasattr(calculator, 'alpha_names') and calculator.alpha_names:
            missing_factors = set(calculator.alpha_names) - set(factors.columns)
            if missing_factors:
                print(f"  缺失的因子: {list(missing_factors)[:20]}")
    else:
        print(f"  ✓ 所有因子计算成功！")
    
    return factors


if __name__ == '__main__':
    factors = asyncio.run(test_handler())
    print(f"\n测试完成，因子数据形状: {factors.shape}")
