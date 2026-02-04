#!/usr/bin/env python
"""性能分析脚本 - 使用 cProfile 找到回测的性能瓶颈
直接调用 bench_backtest_500_3y.py 的逻辑，避免路径问题
"""

import asyncio
import cProfile
import os
import pstats
import sys
from io import StringIO

# 添加 backend 到 sys.path
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

# 切换工作目录到 backend
os.chdir(BACKEND_ROOT)

# 导入 bench 脚本的主函数
from scripts.bench_backtest_500_3y import main as bench_main


def run_bench():
    """运行 bench 主函数"""
    # ��置命令行参数
    sys.argv = ["bench_backtest_500_3y.py", "--sizes", "50"]
    # 运行 async main
    asyncio.run(bench_main())


def main():
    """主函数"""
    print("开始性能分析...")
    print("=" * 80)
    print("使用 bench_backtest_500_3y.py 运行 50 只股票的回测")
    print("=" * 80)
    
    # 创建 profiler
    profiler = cProfile.Profile()
    
    # 启动 profiling
    profiler.enable()
    
    # 运行回测（50只股票）
    run_bench()
    
    # 停止 profiling
    profiler.disable()
    
    print("\n" + "=" * 80)
    print("性能分析结果（按累计时间排序，前 50 个函数）：")
    print("=" * 80)
    
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(50)
    print(s.getvalue())
    
    # 保存详细报告
    output_file = os.path.join(BACKEND_ROOT, "profile_result.txt")
    with open(output_file, "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(200)  # 保存前 200 个函数
    
    print(f"\n详细性能报告已保存到: {output_file}")
    print("=" * 80)
    
    # 按总时间排序也输���一份
    print("\n性能分析结果（按总时间排序，前 30 个函数）：")
    print("=" * 80)
    s2 = StringIO()
    ps2 = pstats.Stats(profiler, stream=s2)
    ps2.strip_dirs()
    ps2.sort_stats('tottime')
    ps2.print_stats(30)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
