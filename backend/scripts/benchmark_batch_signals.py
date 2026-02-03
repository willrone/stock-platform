"""
批量信号生成性能测试

测试目标：
- 50 只股票 × 750 天：从 143.96s 降到 < 50s
- 500 只股票 × 750 天：从 ~1440s 降到 < 180s (3分钟)
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from loguru import logger
import sys

from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.models import BacktestConfig


async def benchmark_backtest(
    stock_count: int = 50,
    days: int = 750,
    strategy_name: str = "MACD",
    enable_batch: bool = True
):
    """
    回测性能基准测试

    Args:
        stock_count: 股票数量
        days: 交易日数量
        strategy_name: 策略名称
        enable_batch: 是否启用批量信号生成
    """
    # 降低日志噪声（500 股票时交易日志非常多，会影响真实性能）
    # - 全局仅输出 WARNING+（抑制大量交易日志）
    # - 本 benchmark 脚本自身输出 INFO（通过 filter 限定来源）
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    logger.add(
        sys.stderr,
        level="INFO",
        filter=lambda r: r.get("name", "").startswith("scripts.benchmark_batch_signals"),
    )

    logger.info(f"=" * 80)
    logger.info(f"回测性能测试: {stock_count} 只股票 × {days} 天")
    logger.info(f"策略: {strategy_name}, 批量模式: {enable_batch}")
    logger.info(f"=" * 80)

    # 准备测试数据
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=days + 100)  # 多加100天用于指标预热

    # 生成测试股票代码（优先使用本地真实数据文件，保证是真实回测流程）
    data_root = Path(__file__).parent.parent / "data"
    parquet_dir = data_root / "parquet" / "stock_data"

    stock_codes: List[str] = []
    if parquet_dir.exists():
        files = sorted(parquet_dir.glob("*.parquet"))
        # 文件名格式: 600361_SH.parquet -> 600361.SH
        for fp in files[: stock_count * 3]:  # 预留一些过滤空间
            name = fp.stem
            if "_" in name:
                code, ex = name.split("_", 1)
                stock_codes.append(f"{code}.{ex}")
            else:
                stock_codes.append(name)
        stock_codes = stock_codes[:stock_count]

    if len(stock_codes) < stock_count:
        # Fallback：使用合成代码（可能无数据，会被 DataLoader 过滤）
        stock_codes = [f"{i:06d}.SH" for i in range(600000, 600000 + stock_count)]

    # 策略配置
    strategy_configs = {
        "MACD": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        },
        "RSI": {
            "period": 14,
            "overbought": 70,
            "oversold": 30
        },
        "MA": {
            "short_period": 5,
            "long_period": 20
        }
    }

    strategy_config = strategy_configs.get(strategy_name, strategy_configs["MACD"])

    # 回测配置
    backtest_config = BacktestConfig(
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        max_position_size=0.3,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
    )

    # 创建执行器
    data_dir = Path(__file__).parent.parent / "data"
    executor = BacktestExecutor(
        data_dir=str(data_dir),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=True,
        use_multiprocessing=False  # 批量模式下多线程即可
    )

    # 如果不启用批量模式，需要禁用预计算
    if not enable_batch:
        # 临时禁用批量预计算
        original_method = executor._precompute_strategy_signals
        executor._precompute_strategy_signals = lambda strategy, stock_data: None

    # 执行回测
    start_time = time.time()
    
    try:
        result = await executor.run_backtest(
            strategy_name=strategy_name,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            strategy_config=strategy_config,
            backtest_config=backtest_config
        )

        elapsed = time.time() - start_time

        # 输出结果
        logger.info(f"\n" + "=" * 80)
        logger.info(f"✅ 回测完成")
        logger.info(f"=" * 80)
        logger.info(f"总耗时: {elapsed:.2f} 秒")
        logger.info(f"股票数: {stock_count}")
        logger.info(f"交易日: {result.get('total_trading_days', 'N/A')}")
        logger.info(f"信号数: {result.get('total_signals', 'N/A')}")
        logger.info(f"交易数: {result.get('executed_trades', 'N/A')}")
        logger.info(f"吞吐量: {result.get('total_trading_days', 0) / elapsed:.2f} 天/秒")
        
        # 性能分析
        if executor.enable_performance_profiling and executor.performance_profiler:
            profiler = executor.performance_profiler
            report = profiler.generate_report()

            logger.info(f"\n性能分析:")
            stages = report.get('stages', {})
            logger.info(f"  数据加载: {stages.get('data_loading', {}).get('duration', 0):.2f}s")
            logger.info(f"  回测执行: {stages.get('backtest_execution', {}).get('duration', 0):.2f}s")
            logger.info(f"  报告生成: {stages.get('report_generation', {}).get('duration', 0):.2f}s")

        # 计算加速比
        if enable_batch:
            # 基准：50只股票 143.96s
            baseline_time = 143.96 * (stock_count / 50)
            speedup = baseline_time / elapsed
            logger.info(f"\n加速比: {speedup:.2f}x (基准: {baseline_time:.2f}s)")

        return {
            'elapsed': elapsed,
            'stock_count': stock_count,
            'days': result.get('total_trading_days', 0),
            'signals': result.get('total_signals', 0),
            'trades': result.get('executed_trades', 0),
            'throughput': result.get('total_trading_days', 0) / elapsed if elapsed > 0 else 0
        }

    except Exception as e:
        logger.error(f"❌ 回测失败: {e}", exc_info=True)
        return None

    finally:
        # 恢复原方法
        if not enable_batch:
            executor._precompute_strategy_signals = original_method


async def run_benchmark_suite():
    """运行完整的性能测试套件"""
    
    logger.info("🚀 开始批量信号生成性能测试套件")
    logger.info("=" * 80)
    
    test_cases = [
        # (stock_count, days, strategy, enable_batch, description)
        (50, 750, "MACD", False, "基准测试 - 传统模式"),
        (50, 750, "MACD", True, "批量模式 - 50只股票"),
        (100, 750, "MACD", True, "批量模式 - 100只股票"),
        (200, 750, "MACD", True, "批量模式 - 200只股票"),
        (500, 750, "MACD", True, "批量模式 - 500只股票 (目标)"),
    ]
    
    results = []
    
    for stock_count, days, strategy, enable_batch, description in test_cases:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"测试用例: {description}")
        logger.info(f"{'=' * 80}\n")
        
        result = await benchmark_backtest(
            stock_count=stock_count,
            days=days,
            strategy_name=strategy,
            enable_batch=enable_batch
        )
        
        if result:
            result['description'] = description
            results.append(result)
        
        # 等待一下，避免资源竞争
        await asyncio.sleep(2)
    
    # 输出汇总报告
    logger.info(f"\n{'=' * 80}")
    logger.info("📊 性能测试汇总报告")
    logger.info(f"{'=' * 80}\n")
    
    logger.info(f"{'描述':<30} {'股票数':<10} {'耗时(s)':<12} {'吞吐量(天/s)':<15} {'目标达成'}")
    logger.info("-" * 80)
    
    for r in results:
        target_met = "✅" if (
            (r['stock_count'] == 50 and r['elapsed'] < 50) or
            (r['stock_count'] == 500 and r['elapsed'] < 180)
        ) else "⏳"
        
        logger.info(
            f"{r['description']:<30} "
            f"{r['stock_count']:<10} "
            f"{r['elapsed']:<12.2f} "
            f"{r['throughput']:<15.2f} "
            f"{target_met}"
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("测试完成！")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(run_benchmark_suite())
