"""
诊断脚本：排查 Web 环境 precompute 5x 性能差距
"""
import gc
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = "../data"  # 相对于 backend/


def run_precompute_test(label, setup_logging_full, disable_gc, stock_data_cache, strategy_factory_fn):
    from loguru import logger
    logger.remove()

    if setup_logging_full:
        from app.core.logging import setup_logging
        setup_logging()
        handler_count = len(logger._core.handlers)
    else:
        logger.add(sys.stderr, level="WARNING")
        handler_count = 1

    print(f"\n{'='*60}")
    print(f"测试: {label}")
    print(f"  handlers: {handler_count}, GC_off: {disable_gc}")
    print(f"  threads: {threading.active_count()} {[t.name for t in threading.enumerate()]}")

    from app.services.backtest.execution.backtest_executor import BacktestExecutor

    stock_data = {}
    for code, df in stock_data_cache.items():
        new_df = df.copy()
        new_df.attrs = dict(df.attrs)
        new_df.attrs.pop("_precomputed_signals", None)
        new_df.attrs.pop("_strategy_indicators_cache", None)
        stock_data[code] = new_df

    strategy = strategy_factory_fn()
    executor = BacktestExecutor(data_dir=DATA_DIR, enable_parallel=True, max_workers=8)

    if disable_gc:
        gc.disable()
        gc.collect()

    t0 = time.perf_counter()
    executor._precompute_strategy_signals(strategy, stock_data)
    t_precompute = time.perf_counter() - t0

    if disable_gc:
        gc.enable()

    print(f"  ⏱️  precompute: {t_precompute:.2f}s")
    print(f"{'='*60}")
    logger.remove()
    return t_precompute


def main():
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("加载数据...")
    from app.services.backtest.execution.data_loader import DataLoader
    loader = DataLoader(data_dir=DATA_DIR)

    parquet_dir = Path(DATA_DIR).resolve() / "parquet" / "stock_data"
    stock_codes = []
    for fp in sorted(parquet_dir.glob("*.parquet"))[:1000]:
        name = fp.stem
        if "_" in name:
            code, ex = name.split("_", 1)
            stock_codes.append(f"{code}.{ex}")
    print(f"  找到 {len(stock_codes)} 只股票")

    start_date = datetime(2021, 1, 1)
    end_date = datetime(2026, 2, 23)

    t0 = time.perf_counter()
    stock_data_cache = loader.load_multiple_stocks(stock_codes, start_date, end_date)
    print(f"  加载完成: {len(stock_data_cache)} 只, {time.perf_counter()-t0:.1f}s")

    def make_strategy():
        from app.services.backtest.strategies import StrategyFactory
        return StrategyFactory.create_strategy("rsi", {
            "rsi_period": 14, "oversold_threshold": 30, "overbought_threshold": 70
        })

    results = {}
    results["full_log+gc"] = run_precompute_test(
        "Web环境 (full logging + GC)", True, False, stock_data_cache, make_strategy)
    results["full_log+no_gc"] = run_precompute_test(
        "Full logging + no GC", True, True, stock_data_cache, make_strategy)
    results["min_log+gc"] = run_precompute_test(
        "Min logging + GC", False, False, stock_data_cache, make_strategy)
    results["min_log+no_gc"] = run_precompute_test(
        "独立进程 (min logging + no GC)", False, True, stock_data_cache, make_strategy)

    print(f"\n{'='*60}")
    print("📊 汇总")
    print(f"{'='*60}")
    baseline = results["min_log+no_gc"]
    for label, t in results.items():
        ratio = t / baseline if baseline > 0 else 0
        print(f"  {label:25s}: {t:6.2f}s  ({ratio:.2f}x)")

    gc_impact = results["min_log+gc"] - results["min_log+no_gc"]
    log_impact = results["full_log+no_gc"] - results["min_log+no_gc"]
    combined = results["full_log+gc"] - results["min_log+no_gc"]
    interaction = combined - gc_impact - log_impact

    print(f"\n📊 因素分解:")
    print(f"  GC 影响:      {gc_impact:+.2f}s")
    print(f"  Logging 影响: {log_impact:+.2f}s")
    print(f"  交互效应:     {interaction:+.2f}s")
    print(f"  总差距:       {combined:+.2f}s")


if __name__ == "__main__":
    main()
