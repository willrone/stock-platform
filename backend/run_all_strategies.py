#!/usr/bin/env python3
"""串行回测所有策略 - 直接调用引擎，不走 FastAPI"""
import asyncio
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.services.backtest import BacktestConfig, BacktestExecutor

STRATEGIES = [
    "bollinger", "stochastic", "cci", "pairs_trading", "mean_reversion",
    "cointegration", "value_factor", "momentum_factor", "low_volatility", "multi_factor",
]

START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 1, 1)
NUM_STOCKS = 100
INITIAL_CASH = 1_000_000.0


def get_stock_codes(n: int) -> list[str]:
    parquet_dir = Path(settings.DATA_ROOT_PATH).resolve() / "parquet" / "stock_data"
    if not parquet_dir.exists():
        parquet_dir = Path(__file__).resolve().parent.parent / "data" / "parquet" / "stock_data"
    codes = sorted(f.stem.replace("_", ".") for f in parquet_dir.glob("*.parquet"))
    return codes[:n]


async def run_strategy(strategy_name: str, stock_codes: list[str]) -> dict:
    executor = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=False,
        max_workers=1,
        enable_performance_profiling=False,
    )
    config = BacktestConfig(
        initial_cash=INITIAL_CASH,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        unlimited_buying=True,
    )
    result = await executor.run_backtest(
        strategy_name=strategy_name,
        stock_codes=stock_codes,
        start_date=START_DATE,
        end_date=END_DATE,
        strategy_config={},
        backtest_config=config,
    )
    return result


def extract_metrics(result) -> dict:
    """从回测结果提取关键指标"""
    if result is None:
        return {"error": "结果为空"}
    # 尝试不同的结果结构
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    return {"raw": str(result)[:500]}


def main():
    stock_codes = get_stock_codes(NUM_STOCKS)
    print(f"股票数量: {len(stock_codes)}")
    print(f"回测区间: {START_DATE.date()} ~ {END_DATE.date()}")
    print(f"初始资金: {INITIAL_CASH:,.0f}")
    print(f"模式: 不限制买入")
    print("=" * 80)

    results = {}
    for i, strategy in enumerate(STRATEGIES, 1):
        print(f"\n[{i}/{len(STRATEGIES)}] 正在运行: {strategy}")
        t0 = time.time()
        try:
            result = asyncio.run(run_strategy(strategy, stock_codes))
            elapsed = time.time() - t0
            metrics = extract_metrics(result)
            results[strategy] = {"status": "completed", "elapsed": round(elapsed, 1), "metrics": metrics}
            print(f"  ✅ 完成 ({elapsed:.1f}s)")
            # 打印关键指标
            if isinstance(metrics, dict):
                for k in ["total_return", "annual_return", "sharpe_ratio", "max_drawdown", "total_trades", "win_rate"]:
                    if k in metrics:
                        print(f"     {k}: {metrics[k]}")
        except Exception as e:
            elapsed = time.time() - t0
            results[strategy] = {"status": "failed", "elapsed": round(elapsed, 1), "error": str(e)}
            print(f"  ❌ 失败 ({elapsed:.1f}s): {e}")
        finally:
            gc.collect()

    # 保存结果
    output_path = Path(__file__).parent / "strategy_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n{'=' * 80}")
    print(f"结果已保存到: {output_path}")

    # 汇总表
    print(f"\n{'策略':<20} {'状态':<10} {'耗时':>8} {'总收益':>10} {'夏普':>8} {'交易数':>8}")
    print("-" * 70)
    for s, r in results.items():
        if r["status"] == "completed" and isinstance(r.get("metrics"), dict):
            m = r["metrics"]
            tr = m.get("total_return", "N/A")
            sr = m.get("sharpe_ratio", "N/A")
            tt = m.get("total_trades", "N/A")
            if isinstance(tr, (int, float)):
                tr = f"{tr:.2%}"
            if isinstance(sr, (int, float)):
                sr = f"{sr:.3f}"
            print(f"{s:<20} {'✅':<10} {r['elapsed']:>7.1f}s {tr:>10} {sr:>8} {tt:>8}")
        else:
            print(f"{s:<20} {'❌':<10} {r['elapsed']:>7.1f}s {'--':>10} {'--':>8} {'--':>8}")


if __name__ == "__main__":
    main()
