#!/usr/bin/env python3
"""
数据服务API启动脚本
启动数据状态查询API服务
"""
import sys
import os
import logging
import signal
import socket
from pathlib import Path

# 验证是否在虚拟环境中运行
project_root = Path(__file__).parent.parent
venv_python = project_root / "venv" / "bin" / "python3"

# 检查当前Python路径是否在虚拟环境中
current_python_str = str(sys.executable)
is_in_venv = 'venv' in current_python_str or current_python_str.startswith(str(project_root / "venv"))

# 如果不在虚拟环境中，且虚拟环境存在，则重新执行
if not is_in_venv and venv_python.exists():
    # 不在虚拟环境中，使用虚拟环境的Python重新执行
    os.chdir(str(project_root))  # 切换到项目根目录
    # 使用execv替换当前进程
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_service.data_status_api import create_app

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_port_available(host, port):
    """检查端口是否可用"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result != 0  # 如果连接失败，说明端口可用
    except Exception:
        sock.close()
        return True

def find_available_port(start_port, max_attempts=10):
    """查找可用端口"""
    for i in range(max_attempts):
        port = start_port + i
        if check_port_available('0.0.0.0', port):
            return port
    return None

def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info("收到停止信号，正在关闭服务...")
    sys.exit(0)

def main():
    """主函数"""
    try:
        logger.info("=" * 60)
        logger.info("股票数据API服务启动")
        logger.info("=" * 60)

        # 设置信号处理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 检查端口是否可用
        port = 5002
        if not check_port_available('0.0.0.0', port):
            logger.warning(f"端口 {port} 已被占用，尝试查找可用端口...")
            available_port = find_available_port(port)
            if available_port:
                logger.info(f"使用端口 {available_port} 替代 {port}")
                port = available_port
            else:
                logger.error(f"无法找到可用端口（尝试范围: {port}-{port+9}）")
                logger.error("请手动停止占用端口的进程或修改端口配置")
                sys.exit(1)

        # 创建应用
        app = create_app()

        # 启动服务
        logger.info(f"数据API服务启动在 http://0.0.0.0:{port}")
        logger.info("可用的API端点:")
        logger.info("  GET /api/data/health - 健康检查")
        logger.info("  GET /api/data/stock_data_status - 获取所有股票数据状态")
        logger.info("  GET /api/data/stock_data_status/<ts_code> - 获取单个股票数据状态")
        logger.info("  GET /api/data/data_summary - 获取数据汇总统计")

        app.run(host='0.0.0.0', port=port, debug=False)

    except KeyboardInterrupt:
        logger.info("收到停止信号，正在退出...")
    except OSError as e:
        if "Address already in use" in str(e) or "端口" in str(e):
            logger.error(f"端口 {port} 已被占用")
            logger.error("请执行以下命令查找并停止占用端口的进程:")
            logger.error(f"  lsof -i :{port}  # macOS/Linux")
            logger.error(f"  或 netstat -ano | findstr :{port}  # Windows")
        else:
            logger.error(f"服务启动失败: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"服务启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

