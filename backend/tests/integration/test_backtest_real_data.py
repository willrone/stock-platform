"""
回测流程（真实数据）集成测试
使用真实数据验证回测主流程（优先预计算，回退 Parquet）。
"""

from pathlib import Path

import pandas as pd
import pytest

from app.core.config import settings
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.execution.data_loader import DataLoader
from app.services.data.qlib_format_converter import QlibFormatConverter
from app.services.data.stock_data_loader import StockDataLoader


def _write_minimal_backtest_fixture(
    data_root: Path, stock_code: str = "000001.SZ"
) -> None:
    """Create deterministic OHLCV + qlib feature parquet files for this smoke test."""
    dates = pd.bdate_range(start="2024-01-01", periods=90)
    steps = pd.Series(range(len(dates)), index=dates, dtype="float64")
    close = 10.0 + steps * 0.04 + (steps % 5) * 0.01
    raw_df = pd.DataFrame(
        {
            "date": dates,
            "open": close.values - 0.02,
            "high": close.values + 0.05,
            "low": close.values - 0.05,
            "close": close.values,
            "volume": (100000 + steps.astype("int64") * 100).values,
            "adj_close": close.values,
        }
    )

    safe_code = stock_code.replace(".", "_")
    parquet_dir = data_root / "parquet" / "stock_data"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(parquet_dir / f"{safe_code}.parquet", index=False)

    qlib_dir = data_root / "qlib_data" / "features" / "day"
    qlib_dir.mkdir(parents=True, exist_ok=True)
    qlib_df = raw_df.set_index("date")
    qlib_features = QlibFormatConverter().convert_parquet_to_qlib(
        qlib_df, stock_code.replace(".", "_")
    )
    qlib_features.to_parquet(qlib_dir / f"{safe_code}.parquet")


def _pick_date_range(df, min_days: int = 30):
    if df is None or df.empty or len(df) < min_days:
        return None, None
    df = df.sort_index()
    window = df.tail(max(min_days * 2, 60))
    return window.index.min(), window.index.max()


@pytest.mark.asyncio
async def test_backtest_flow_with_real_data(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    _write_minimal_backtest_fixture(data_root)

    # 确保测试进程使用隔离的确定性数据路径。
    monkeypatch.setattr(
        settings, "QLIB_DATA_PATH", str(data_root / "qlib_data"), raising=False
    )

    loader = StockDataLoader(data_root=str(data_root))
    raw_df = loader.load_stock_data("000001.SZ")
    start_date, end_date = _pick_date_range(raw_df)
    if not start_date or not end_date:
        pytest.skip("真实数据交易日不足，无法运行回测")

    # 预计算数据优先加载
    data_loader = DataLoader(data_dir=str(data_root))
    precomputed_df = data_loader.load_stock_data("000001.SZ", start_date, end_date)
    assert not precomputed_df.empty
    assert precomputed_df.attrs.get("from_precomputed") is True

    # 执行回测
    executor = BacktestExecutor(data_dir=str(data_root), enable_parallel=False)
    report = await executor.run_backtest(
        strategy_name="moving_average",
        stock_codes=["000001.SZ"],
        start_date=start_date,
        end_date=end_date,
        strategy_config={
            "short_window": 5,
            "long_window": 20,
            "signal_threshold": 0.01,
        },
        backtest_config=None,
        task_id=None,
    )

    assert isinstance(report, dict)
    assert "strategy_name" in report
    assert "metrics" in report
    assert "portfolio_history" in report

    # 强制回退 Parquet
    monkeypatch.setattr(
        settings, "QLIB_DATA_PATH", str(data_root / "qlib_data_missing"), raising=False
    )
    fallback_loader = DataLoader(data_dir=str(data_root))
    fallback_df = fallback_loader.load_stock_data("000001.SZ", start_date, end_date)
    assert not fallback_df.empty
    assert not fallback_df.attrs.get("from_precomputed", False)
