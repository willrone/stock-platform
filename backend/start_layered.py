"""
股票平台 - 启动脚本

包含 Laya 模型自动加载和商业化计费初始化。
"""

import asyncio
import sys
import os

# 设置项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from app.core.config import settings
from app.services.commerce.commission_service import CommerceService


async def initialize_services():
    """初始化所有服务。"""
    logger.info("=" * 60)
    logger.info("正在启动股票平台...")
    logger.info("=" * 60)
    
    # 1. 初始化计费规则
    logger.info("初始化计费系统...")
    try:
        commerce = CommerceService()
        await commerce.record_usage(
            user_id="system",
            event_type="api_call",
            quantity=1,
            metadata={"action": "platform_startup"}
        )
        logger.info("✅ 计费系统就绪")
    except Exception as e:
        logger.warning(f"计费系统初始化失败: {e}")
    
    # 2. 初始化 Laya 模型
    if settings.LAYA_ENABLED:
        logger.info("正在加载 Laya 决策模型...")
        try:
            from app.services.laya import LayaService
            
            laya_svc = LayaService()
            success = await laya_svc.initialize()
            
            if success:
                logger.info("✅ Laya 模型就绪")
            else:
                logger.warning("⚠️ Laya 模型加载失败，将使用备用策略")
        except Exception as e:
            logger.error(f"Laya 模型加载失败: {e}")
    
    logger.info("所有服务初始化完成!")
    logger.info("=" * 60)


async def main():
    """主入口。"""
    await initialize_services()
    
    # 启动 FastAPI 服务
    from uvicorn import Config
    from app.main import app
    
    config = Config(
        app=app,
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
    
    logger.info(f"服务启动: http://{settings.HOST}:{settings.PORT}")
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
