"""
日志配置
"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    """设置日志配置"""
    # 移除默认处理器
    logger.remove()
    
    # 控制台输出
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
    )
    
    # 文件输出
    log_path = Path(settings.DATA_ROOT_PATH) / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path / "app.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )
    
    # 错误日志单独文件
    logger.add(
        log_path / "error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )