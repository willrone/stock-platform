"""恢复 KDJ 和 RSI 优化任务的续跑"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from concurrent.futures import ProcessPoolExecutor
from app.api.v1.optimization import execute_optimization_task_simple

task_ids = [
    '55530c47-6b2c-4871-90a7-b4989d1d85e5',   # KDJ 368/400
    '54f14258-5406-4f3f-b854-2dce24dd9710',   # RSI 1678/6400
]

if __name__ == '__main__':
    print("恢复 KDJ + RSI 优化任务...", flush=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {}
        for tid in task_ids:
            f = executor.submit(execute_optimization_task_simple, tid)
            futures[tid] = f
            print(f"  已提交: {tid[:8]}...", flush=True)
        
        print("等待任务完成...", flush=True)
        for tid, f in futures.items():
            try:
                f.result()
                print(f"  ✅ {tid[:8]}... 完成", flush=True)
            except Exception as e:
                print(f"  ❌ {tid[:8]}... 失败: {e}", flush=True)

    print("全部完成", flush=True)
