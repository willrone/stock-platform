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

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    import argparse
    import subprocess
    
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
    
    # 获取脚本目录
    scripts_dir = project_root / 'scripts'
    
    if args.service == 'service':
        print("🚀 启动数据获取服务（定时任务）...")
        print("📋 日志文件: logs/data_service.log")
        print("=" * 60)
        script = scripts_dir / 'run_data_service.py'
        subprocess.run([sys.executable, str(script)])
    elif args.service == 'api':
        print("🚀 启动数据API服务...")
        print("📋 日志文件: logs/data_api.log")
        print("🌐 API服务地址: http://localhost:5002")
        print("=" * 60)
        script = scripts_dir / 'run_data_api.py'
        subprocess.run([sys.executable, str(script)])
    else:  # all
        print("🚀 启动股票数据服务（数据获取 + API）...")
        print("📋 数据服务日志: logs/data_service.log")
        print("📋 API服务日志: logs/data_api.log")
        print("🌐 API服务地址: http://localhost:5002")
        print("=" * 60)
        
        # 后台启动数据获取服务
        service_script = scripts_dir / 'run_data_service.py'
        service_process = subprocess.Popen(
            [sys.executable, str(service_script)],
            stdout=open('logs/data_service.log', 'a'),
            stderr=subprocess.STDOUT
        )
        print(f"✅ 数据获取服务已启动 (PID: {service_process.pid})")
        
        # 等待一下让数据服务启动
        import time
        time.sleep(2)
        
        # 前台启动API服务
        api_script = scripts_dir / 'run_data_api.py'
        try:
            subprocess.run([sys.executable, str(api_script)])
        except KeyboardInterrupt:
            print("\n收到停止信号，正在关闭服务...")
            service_process.terminate()
            service_process.wait()
            print("✅ 服务已停止")

if __name__ == '__main__':
    main()

