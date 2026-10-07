#!/usr/bin/env python3
"""
股票数据服务主入口
独立运行的数据服务，提供股票数据获取能力

使用方法:
    python main.py [service|api|all]
    
    service - 仅启动数据获取服务（定时任务）
    api     - 仅启动数据API服务（RESTful API）
    all     - 同时启动数据获取服务和API服务（默认）
"""
import sys
import os
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def start_api():
    """启动 Flask API 服务（阻塞，运行在主线程或子线程）"""
    from data_service.data_status_api import create_app
    app = create_app()
    print("🚀 数据API服务启动中...")
    print("🌐 API服务地址: http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=False)


def start_scheduler():
    """启动 APScheduler 定时任务（阻塞）"""
    from data_service.scheduler import start_scheduler as _start
    _start()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='股票数据服务 - 独立运行的数据服务，提供股票数据获取能力'
    )
    parser.add_argument(
        'service',
        nargs='?',
        default='all',
        choices=['service', 'api', 'all'],
        help='要启动的服务类型: service(数据获取), api(API服务), all(全部)'
    )

    args = parser.parse_args()

    if args.service == 'api':
        print("🚀 启动数据API服务...")
        print("📋 日志文件: logs/data_api.log")
        print("=" * 60)
        start_api()

    elif args.service == 'service':
        print("🚀 启动数据获取服务（定时任务）...")
        print("📋 日志文件: logs/data_service.log")
        print("=" * 60)
        start_scheduler()

    else:  # all
        print("🚀 启动股票数据服务（数据获取 + API）...")
        print("📋 API服务地址: http://localhost:5002")
        print("=" * 60)

        import threading

        # 后台线程启动调度器
        scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
        scheduler_thread.start()
        print(f"✅ 数据获取服务已启动 (线程: {scheduler_thread.name})")

        import time
        time.sleep(1)  # 给调度器一点启动时间

        # 主线程运行 API 服务
        try:
            start_api()
        except KeyboardInterrupt:
            print("\n收到停止信号，服务已停止")


if __name__ == '__main__':
    main()
