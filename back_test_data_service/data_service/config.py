"""
数据服务配置
独立运行的数据服务，提供股票数据获取能力
"""
import os
from typing import Optional
from pathlib import Path


class Config:
    """数据服务配置管理"""
    
    # Tushare配置
    # 优先从环境变量获取，如果没有则使用默认token
    TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '3bb09d7c81ac1f83a90e57b505626391739a93bd02c717bdcb987da4')
    
    # MySQL配置（已废弃，使用Parquet存储）
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'stock_data')
    
    # Redis配置（已废弃，使用Parquet存储）
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
    
    # 缓存配置（已废弃）
    CACHE_EXPIRY_HOURS = int(os.getenv('CACHE_EXPIRY_HOURS', 24))
    CACHE_EXPIRY_SECONDS = CACHE_EXPIRY_HOURS * 3600
    
    # 并发配置
    MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
    
    # 批处理配置
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))
    
    # 默认日期范围
    DEFAULT_START_DATE = '19700101'
    DEFAULT_END_DATE = '20250601'
    
    # Parquet存储配置
    PARQUET_DATA_DIR = os.getenv('PARQUET_DATA_DIR', None)  # 如果为None，使用默认路径

    # 服务发现配置（已废弃，不再使用）
    # ENABLE_AUTO_DISCOVERY = os.getenv('ENABLE_AUTO_DISCOVERY', 'false').lower() == 'true'
    # DISCOVERY_TIMEOUT = int(os.getenv('DISCOVERY_TIMEOUT', 30))  # 自动发现超时时间（秒）

    # 自动发现的服务地址（已废弃，不再使用）
    # DISCOVERED_REDIS_HOST = None
    # DISCOVERED_MYSQL_HOST = None

    @classmethod
    def validate(cls) -> bool:
        """验证配置是否完整"""
        if not cls.TUSHARE_TOKEN:
            print("⚠️  警告: TUSHARE_TOKEN 环境变量未设置，数据获取服务将无法启动")
            print("   请设置环境变量: export TUSHARE_TOKEN='your_token'")
            return False
        return True

    # 以下方法已废弃，不再使用服务发现功能
    # @classmethod
    # def auto_discover_services(cls):
    #     """自动发现服务地址"""
    #     pass
    #
    # @classmethod
    # def get_redis_host(cls):
    #     """获取Redis主机地址（优先使用自动发现的地址）"""
    #     return cls.DISCOVERED_REDIS_HOST or cls.REDIS_HOST
    #
    # @classmethod
    # def get_mysql_host(cls):
    #     """获取MySQL主机地址（优先使用自动发现的地址）"""
    #     return cls.DISCOVERED_MYSQL_HOST or cls.MYSQL_HOST
