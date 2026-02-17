"""
应用程序配置管理
"""

from pathlib import Path
from typing import List, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取 backend 目录的绝对路径（config.py 所在目录的父级的父级）
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """应用程序设置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略额外的环境变量
    )

    # 应用配置
    APP_NAME: str = "Stock Prediction Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"  # 添加环境配置

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # 数据库配置
    DATABASE_URL: str = "sqlite:///./data/app.db"

    # 远端数据服务配置
    REMOTE_DATA_SERVICE_URL: str = "http://192.168.3.62:5002"
    REMOTE_DATA_SERVICE_TIMEOUT: int = 30

    # 数据同步（SFTP）配置（用于分布式部署：从数据服务器拉取parquet到本机）
    SFTP_SYNC_ENABLED: bool = False
    SFTP_HOST: Optional[str] = None
    SFTP_PORT: int = 22
    SFTP_USERNAME: Optional[str] = None
    SFTP_PASSWORD: Optional[str] = None
    SFTP_REMOTE_LIST_PATH: Optional[str] = None
    SFTP_REMOTE_DATA_DIR: Optional[str] = None

    # 数据存储配置
    DATA_ROOT_PATH: str = "../data"  # 相对于backend目录
    PARQUET_DATA_PATH: str = "../data/stocks"
    MODEL_STORAGE_PATH: str = "../data/models"

    # Qlib 配置
    QLIB_DATA_PATH: str = "../data/qlib_data"
    QLIB_CACHE_PATH: str = "../data/qlib_cache"
    # 性能：如果 all_stocks.parquet 很大且经常 miss，会带来大量 I/O 与日志；默认关闭
    QLIB_USE_ALL_STOCKS_FILE: bool = False  # 默认关闭从 all_stocks.parquet 走合并文件路径

    # 性能：组合快照 sanity 日志会在持仓数较大时刷屏；默认关闭
    ENABLE_PORTFOLIO_SNAPSHOT_SANITY_LOG: bool = False

    # 任务配置
    MAX_CONCURRENT_TASKS: int = 3
    TASK_TIMEOUT_SECONDS: int = 3600
    CLEANUP_INTERVAL_HOURS: int = 24

    # 进程池配置（用于任务执行，独立于WORKERS配置）
    PROCESS_POOL_SIZE: int = 3  # 进程池大小，建议为CPU核心数
    TASK_EXECUTION_TIMEOUT: int = 3600  # 任务执行超时（秒）

    # 回测并行化配置（单个回测任务内部的并行化）
    BACKTEST_PARALLEL_ENABLED: bool = True  # 是否启用回测并行化（多股票信号生成并行）
    BACKTEST_MAX_WORKERS: int = 10  # 回测并行化工作线程数（建议为CPU核心数）

    # API 配置
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: str = (
        "http://localhost:3000,http://127.0.0.1:3000,http://192.168.3.62:3000"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """将CORS_ORIGINS字符串转换为列表"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    # 监控配置
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    @property
    def database_url_sync(self) -> str:
        """同步数据库URL"""
        return self.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite:///")

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "Settings":
        """将相对路径转换为绝对路径，确保进程池子进程也能正确找到数据"""
        path_fields = [
            "DATA_ROOT_PATH",
            "PARQUET_DATA_PATH",
            "MODEL_STORAGE_PATH",
            "QLIB_DATA_PATH",
            "QLIB_CACHE_PATH",
        ]
        for field in path_fields:
            value = getattr(self, field)
            if value and value.startswith(".."):
                # 相对路径，转换为绝对路径
                abs_path = str((_BACKEND_DIR / value).resolve())
                object.__setattr__(self, field, abs_path)
        return self


settings = Settings()
