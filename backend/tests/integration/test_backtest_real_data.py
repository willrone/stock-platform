"""
回测流程（真实数据）集成测试
使用真实数据验证回测主流程（优先预计算，回退 Parquet）。
"""

from pathlib import Path

import pytest

from app.core.config import settings
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.execution.data_loader import DataLoader
from app.services.data.stock_data_loader import StockDataLoader


def _pick_date_range(df, min_days: int = 30):
    if df is None or df.empty or len(df) < min_days:
        return None, None
    df = df.sort_index()
    window = df.tail(max(min_days * 2, 60))
    return window.index.min(), window.index.max()


@pytest.mark.asyncio
async def test_backtest_flow_with_real_data(monkeypatch):
    parquet_file = Path("backend/data/parquet/stock_data/000001_SZ.parquet")
    precomputed_file = Path("backend/data/qlib_data/features/day/000001_SZ.parquet")

    if not parquet_file.exists():
        pytest.skip(f"真实数据不存在: {parquet_file}")
    if not precomputed_file.exists():
        pytest.skip(f"预计算数据不存在: {precomputed_file}")

    # 确保测试进程使用正确的数据路径
    monkeypatch.setattr(settings, "QLIB_DATA_PATH", "backend/data/qlib_data", raising=False)

    loader = StockDataLoader(data_root="backend/data")
    raw_df = loader.load_stock_data("000001.SZ")
    start_date, end_date = _pick_date_range(raw_df)
    if not start_date or not end_date:
        pytest.skip("真实数据交易日不足，无法运行回测")

    # 预计算数据优先加载
    data_loader = DataLoader(data_dir="backend/data")
    precomputed_df = data_loader.load_stock_data("000001.SZ", start_date, end_date)
    assert not precomputed_df.empty
    assert precomputed_df.attrs.get("from_precomputed") is True

    # 执行回测
    executor = BacktestExecutor(data_dir="backend/data", enable_parallel=False)
    report = await executor.run_backtest(
        strategy_name="moving_average",
        stock_codes=["000001.SZ"],
        start_date=start_date,
        end_date=end_date,
        strategy_config={"short_window": 5, "long_window": 20, "signal_threshold": 0.01},
        backtest_config=None,
        task_id=None,
    )

    assert isinstance(report, dict)
    assert "strategy_name" in report
    assert "metrics" in report
    assert "portfolio_history" in report

    # 强制回退 Parquet
    monkeypatch.setattr(settings, "QLIB_DATA_PATH", "backend/data/qlib_data_missing", raising=False)
    fallback_loader = DataLoader(data_dir="backend/data")
    fallback_df = fallback_loader.load_stock_data("000001.SZ", start_date, end_date)
    assert not fallback_df.empty
    assert not fallback_df.attrs.get("from_precomputed", False)
