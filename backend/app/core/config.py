<<<<<<< HEAD
"""
应用程序配置管理
"""

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用程序设置"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 应用配置
    APP_NAME: str = "Stock Prediction Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # 数据库配置
    DATABASE_URL: str = "sqlite:///./data/app.db"

    # 远端数据服务配置
    REMOTE_DATA_SERVICE_URL: str = "http://192.168.3.62"
    REMOTE_DATA_SERVICE_TIMEOUT: int = 30

    # 数据存储配置
    DATA_ROOT_PATH: str = "./data"
    PARQUET_DATA_PATH: str = "./data/stocks"
    MODEL_STORAGE_PATH: str = "./data/models"

    # Qlib 配置
    QLIB_DATA_PATH: str = "./data/qlib_data"
    QLIB_CACHE_PATH: str = "./data/qlib_cache"

    # 任务配置
    MAX_CONCURRENT_TASKS: int = 3
    TASK_TIMEOUT_SECONDS: int = 3600
    CLEANUP_INTERVAL_HOURS: int = 24

    # API 配置
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # 监控配置
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    @property
    def database_url_sync(self) -> str:
        """同步数据库URL"""
        return self.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite:///")


=======
"""
应用程序配置管理
"""

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用程序设置"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
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

    # 数据存储配置
    DATA_ROOT_PATH: str = "./data"
    PARQUET_DATA_PATH: str = "./data/stocks"
    MODEL_STORAGE_PATH: str = "./data/models"

    # Qlib 配置
    QLIB_DATA_PATH: str = "./data/qlib_data"
    QLIB_CACHE_PATH: str = "./data/qlib_cache"

    # 任务配置
    MAX_CONCURRENT_TASKS: int = 3
    TASK_TIMEOUT_SECONDS: int = 3600
    CLEANUP_INTERVAL_HOURS: int = 24

    # API 配置
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

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


>>>>>>> a6754c6 (feat(platform): Complete stock prediction platform with deployment and monitoring)
settings = Settings()