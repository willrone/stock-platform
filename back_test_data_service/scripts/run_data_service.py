#!/usr/bin/env python3
"""
数据服务启动脚本
用于启动Mac Mini上的数据服务
"""
import sys
import os
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_service.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 验证是否在虚拟环境中运行
project_root = Path(__file__).parent.parent
venv_python = project_root / "venv" / "bin" / "python3"

# 检查当前Python路径是否在虚拟环境中
current_python_str = str(sys.executable)
is_in_venv = 'venv' in current_python_str or current_python_str.startswith(str(project_root / "venv"))

logger.info(f"Current Python executable: {current_python_str}")
logger.info(f"Is in venv: {is_in_venv}")
logger.info(f"Venv Python exists: {venv_python.exists()}")

# 如果不在虚拟环境中，且虚拟环境存在，则重新执行
if not is_in_venv and venv_python.exists():
    logger.info("Not in venv, restarting with venv Python...")
    # 不在虚拟环境中，使用虚拟环境的Python重新执行
    os.chdir(str(project_root))  # 切换到项目根目录
    # 使用execv替换当前进程
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_service.scheduler import DataScheduler


def main():
    """主函数"""
    try:
        logger.info("=" * 60)
        logger.info("股票数据服务启动")
        logger.info("=" * 60)
        
        # 创建调度服务
        scheduler = DataScheduler()
        
        # 启动服务
        scheduler.start()
        
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在退出...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"服务启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
