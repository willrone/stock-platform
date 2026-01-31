#!/usr/bin/env python3
"""
回测任务诊断脚本：检查任务配置、结果中的信号数，以及 signal_records 表中的记录数。
用于排查「任务结果里没有信号记录」等问题。

用法:
  cd backend && python -m scripts.diagnose_backtest_task 13cab251-228e-4c80-b462-7764cccbe7ee
  或
  python scripts/diagnose_backtest_task.py 13cab251-228e-4c80-b462-7764cccbe7ee
"""

import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        print("用法: python diagnose_backtest_task.py <task_id>")
        print("示例: python diagnose_backtest_task.py 13cab251-228e-4c80-b462-7764cccbe7ee")
        sys.exit(1)

    task_id = sys.argv[1].strip()

    # 确保能导入 app（从 backend 目录运行）
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    os.chdir(backend_dir)

    from app.core.database import SessionLocal
    from app.models.backtest_detailed_models import SignalRecord
    from app.models.task_models import Task
    from sqlalchemy import func

    session = SessionLocal()
    try:
        task = session.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            print(f"未找到任务: {task_id}")
            sys.exit(2)

        print("=" * 60)
        print(f"任务 ID: {task_id}")
        print("=" * 60)
        print(f"状态: {task.status}")
        print(f"类型: {task.task_type}")
        print(f"创建时间: {task.created_at}")
        print()

        config = task.config or {}
        print("--- 任务配置 ---")
        print(f"  策略: {config.get('strategy_name', 'N/A')}")
        print(f"  股票: {config.get('stock_codes', [])}")
        print(f"  开始日期: {config.get('start_date', 'N/A')}")
        print(f"  结束日期: {config.get('end_date', 'N/A')}")
        print(f"  初始资金: {config.get('initial_cash', 'N/A')}")
        print()

        result = task.result
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                result = None

        if result:
            print("--- 任务结果摘要 ---")
            total_signals = result.get("total_signals")
            trading_days = result.get("trading_days")
            total_trades = result.get("total_trades")
            print(f"  total_signals (报告): {total_signals if total_signals is not None else '未在报告中'}")
            print(f"  trading_days:         {trading_days if trading_days is not None else '未在报告中'}")
            print(f"  total_trades:         {total_trades}")
            if total_signals is not None:
                if total_signals == 0:
                    print()
                    print("  => 策略在本回测周期内未产生任何信号（total_signals=0）。")
                    print("     可能原因：回测区间过短、策略 warmup 未满足（如需至少 20/130 个交易日）、")
                    print("     或策略条件从未触发（如 RSI/MACD 阈值未触及）。")
                else:
                    print()
                    print(f"  => 报告中有 {total_signals} 个信号，若前端仍无信号记录，请检查 signal_records 表是否写入成功。")
            print()
        else:
            print("--- 任务结果 ---")
            print("  (无 result 或无法解析)")
            print()

        # 查询 signal_records 表中该 task_id 的记录数
        try:
            count_result = (
                session.query(func.count(SignalRecord.id))
                .filter(SignalRecord.task_id == task_id)
                .scalar()
            )
            signal_count = count_result or 0
        except Exception as e:
            print("--- signal_records 表 ---")
            print(f"  查询失败（可能表不存在或未迁移）: {e}")
            signal_count = None
        else:
            print("--- signal_records 表 ---")
            print(f"  该 task_id 下的信号记录数: {signal_count}")
            if signal_count == 0 and result and result.get("total_signals", 0) > 0:
                print("  => 报告中有信号但表中无记录，可能是保存时异常（请查后端日志「保存信号记录」）。")
            elif signal_count == 0:
                print("  => 与报告一致，当前无信号记录。")

        print("=" * 60)
    finally:
        session.close()


if __name__ == "__main__":
    main()
