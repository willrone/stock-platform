"""
调试数据服务
"""

import tempfile
from datetime import datetime
from app.services.data_service_simple import SimpleStockDataService

# 创建临时目录和服务
with tempfile.TemporaryDirectory() as temp_dir:
    service = SimpleStockDataService(data_path=temp_dir)
    
    # 生成测试数据
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 7)
    stock_code = "000001.SZ"
    
    print(f"生成数据: {start_date.date()} - {end_date.date()}")
    
    data = service.generate_mock_data(stock_code, start_date, end_date)
    print(f"生成了 {len(data)} 条记录")
    
    if data:
        print("第一条记录:", data[0])
        print("最后一条记录:", data[-1])
    
    # 保存数据
    service.save_to_local(data, stock_code)
    
    # 检查数据存在性
    exists = service.check_local_data_exists(stock_code, start_date, end_date)
    print(f"数据存在性检查: {exists}")
    
    # 加载数据检查
    loaded_data = service.load_from_local(stock_code, start_date, end_date)
    if loaded_data:
        print(f"加载了 {len(loaded_data)} 条记录")
        print("第一条加载记录:", loaded_data[0])
        print("最后一条加载记录:", loaded_data[-1])
    else:
        print("加载数据失败")