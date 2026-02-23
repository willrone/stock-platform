"""重新提交卡住的优化任务到进程池"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from concurrent.futures import ProcessPoolExecutor
from app.api.v1.optimization import execute_optimization_task_simple

task_ids = [
    '9b6ebefe-9e72-485b-bd34-65f2a3b739ff',   # 均值回归
    '5bcad3cc-e674-4701-b75a-d3697f6b0a92',   # 配对交易
    '55530c47-6b2c-4871-90a7-b4989d1d85e5',   # KDJ
    '54f14258-5406-4f3f-b854-2dce24dd9710',   # RSI
]

if __name__ == '__main__':
    print("启动进程池，提交4个任务...", flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}
        for tid in task_ids:
            f = executor.submit(execute_optimization_task_simple, tid)
            futures[tid] = f
            print(f"  已提交: {tid[:8]}...", flush=True)
        
        print("等待所有任务完成...", flush=True)
        for tid, f in futures.items():
            try:
                f.result()
                print(f"  ✅ {tid[:8]}... 完成", flush=True)
            except Exception as e:
                print(f"  ❌ {tid[:8]}... 失败: {e}", flush=True)

    print("全部完成", flush=True)
