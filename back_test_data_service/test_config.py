#!/usr/bin/env python3
"""
测试配置是否正确
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_service.config import Config
from data_service.parquet_dao import create_dao

def test_config():
    """测试配置"""
    print("=" * 60)
    print("测试数据服务配置")
    print("=" * 60)
    
    # 测试Tushare Token
    print(f"\n1. Tushare Token配置:")
    if Config.TUSHARE_TOKEN:
        print(f"   ✅ Token已配置: {Config.TUSHARE_TOKEN[:20]}...")
    else:
        print("   ❌ Token未配置")
        return False
    
    # 验证配置
    print(f"\n2. 配置验证:")
    if Config.validate():
        print("   ✅ 配置验证通过")
    else:
        print("   ❌ 配置验证失败")
        return False
    
    # 测试Parquet DAO
    print(f"\n3. Parquet DAO测试:")
    try:
        dao = create_dao()
        print(f"   ✅ Parquet DAO创建成功")
        print(f"   📁 数据目录: {dao.data_dir}")
    except Exception as e:
        print(f"   ❌ Parquet DAO创建失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！服务可以独立运行")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = test_config()
    sys.exit(0 if success else 1)

