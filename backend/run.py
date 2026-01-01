"""
开发服务器启动脚本
"""

import uvicorn
from pathlib import Path

from app.core.config import settings

if __name__ == "__main__":
    # 如果启用reload，只监控app目录，避免监控大量数据文件导致文件监控限制
    reload_dirs = None
    if settings.DEBUG:
        # 只监控app目录下的代码文件，排除数据目录
        backend_dir = Path(__file__).parent
        reload_dirs = [str(backend_dir / "app")]
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        reload_dirs=reload_dirs if settings.DEBUG else None,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )