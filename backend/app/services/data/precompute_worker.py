"""
预计算多进程 Worker

独立进程中执行因子计算，避免 asyncio event loop 问题。
每个 worker 处理一批股票的完整因子计算流程。
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


# ── 常量 ──────────────────────────────────────────────
MIN_DATA_ROWS = 30  # 最少需要 30 行数据才能计算因子
QLIB_REQUIRED_COLS = ("$close", "$high", "$low", "$volume", "$open")


def compute_stock_factors(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    在独立进程中计算一只股票的所有因子。

    Args:
        args: 包含 stock_code, data_root, qlib_data_path,
              start_date, end_date, incremental, last_computed_date 的字典

    Returns:
        {"stock_code": str, "success": bool, "error": str | None}
    """
    stock_code = args["stock_code"]
    try:
        result = _compute_single_stock(args)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Worker 计算 {stock_code} 失败: {e}\n{tb}")
        return {"stock_code": stock_code, "success": False, "error": str(e)}


def _compute_single_stock(args: Dict[str, Any]) -> Dict[str, Any]:
    """单只股票的完整计算流程（同步，在子进程中运行）"""
    import asyncio

    stock_code: str = args["stock_code"]
    data_root: str = args["data_root"]
    qlib_data_path: str = args["qlib_data_path"]
    start_date: Optional[datetime] = args.get("start_date")
    end_date: Optional[datetime] = args.get("end_date")
    incremental: bool = args.get("incremental", True)
    last_computed_date: Optional[str] = args.get("last_computed_date")

    # 在子进程中创建新的 event loop（关键！避免 fork 后 loop 损坏）
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        success = loop.run_until_complete(
            _async_compute_stock(
                stock_code=stock_code,
                data_root=data_root,
                qlib_data_path=qlib_data_path,
                start_date=start_date,
                end_date=end_date,
                incremental=incremental,
                last_computed_date=last_computed_date,
            )
        )
        return {"stock_code": stock_code, "success": success, "error": None}
    finally:
        loop.close()


async def _async_compute_stock(
    stock_code: str,
    data_root: str,
    qlib_data_path: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    incremental: bool,
    last_computed_date: Optional[str],
) -> bool:
    """异步计算单只股票因子（在子进程的新 event loop 中运行）"""
    from app.services.data.stock_data_loader import StockDataLoader
    from app.services.data.qlib_format_converter import QlibFormatConverter
    from app.services.data.qlib_bin_converter import QlibBinConverter
    from app.services.data.incremental_updater import IncrementalUpdater
    from app.services.qlib.enhanced_qlib_provider import (
        Alpha158Calculator,
        EnhancedQlibDataProvider,
    )
    from app.models.stock_simple import StockData
    from app.services.models.feature_engineering import FeatureCalculator
    from app.services.prediction.technical_indicators import (
        TechnicalIndicatorCalculator,
    )

    data_loader = StockDataLoader()
    format_converter = QlibFormatConverter()
    bin_converter = QlibBinConverter()
    incremental_updater = IncrementalUpdater()
    indicator_calculator = TechnicalIndicatorCalculator()
    feature_calculator = FeatureCalculator()

    qlib_path = Path(qlib_data_path)
    output_dir = qlib_path / "features" / "day"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载基础数据
    stock_data = data_loader.load_stock_data(stock_code, start_date, end_date)
    if stock_data.empty or len(stock_data) < MIN_DATA_ROWS:
        logger.warning(f"股票 {stock_code} 数据不足 ({len(stock_data)} 行)")
        return False

    # 2. 日期级别增量：只计算新增日期的数据
    compute_start_date = start_date
    if incremental and last_computed_date:
        lcd = pd.Timestamp(last_computed_date)
        # 因子计算需要回看窗口（如 MA60 需要 60 天），所以往前多取一些
        LOOKBACK_DAYS = 120
        lookback_start = lcd - pd.Timedelta(days=LOOKBACK_DAYS)
        # 只保留 lookback_start 之后的数据用于计算
        stock_data_full = stock_data.copy()
        stock_data = stock_data[stock_data.index >= lookback_start]
        if stock_data.empty:
            logger.info(f"股票 {stock_code} 无新增数据")
            return True

    # 3. 转换为 Qlib 格式
    qlib_base = format_converter.convert_parquet_to_qlib(stock_data, stock_code)

    # 4. 计算基础统计指标
    qlib_base = _compute_base_indicators(qlib_base, stock_code, format_converter)

    # 5. 计算 Alpha158 因子
    try:
        provider = EnhancedQlibDataProvider()
        await provider.initialize_qlib()
        alpha_calculator = provider.alpha_calculator

        if alpha_calculator:
            # 生成 bin 数据
            try:
                bin_converter.convert_parquet_to_bin(
                    qlib_base, stock_code, qlib_path
                )
            except Exception as bin_err:
                logger.warning(f"{stock_code} bin 转换失败: {bin_err}")

            dates = (
                qlib_base.index.get_level_values(1)
                if isinstance(qlib_base.index, pd.MultiIndex)
                else qlib_base.index
            )
            date_range = (
                dates.min().to_pydatetime(),
                dates.max().to_pydatetime(),
            )

            alpha_factors = await alpha_calculator.calculate_alpha_factors(
                qlib_data=qlib_base,
                stock_codes=[stock_code],
                date_range=date_range,
                use_cache=False,
                force_expression_engine=False,
            )
            if not alpha_factors.empty:
                qlib_base = format_converter.add_indicators_to_qlib(
                    qlib_base, alpha_factors, stock_code
                )
    except Exception as alpha_err:
        logger.warning(f"{stock_code} Alpha158 计算失败: {alpha_err}")

    # 6. 计算技术指标
    try:
        tech_indicators = _compute_technical_indicators(
            stock_data, stock_code, indicator_calculator
        )
        if not tech_indicators.empty:
            qlib_base = format_converter.add_indicators_to_qlib(
                qlib_base, tech_indicators, stock_code
            )
    except Exception as tech_err:
        logger.warning(f"{stock_code} 技术指标计算失败: {tech_err}")

    # 7. 计算基本面特征
    try:
        fund_features = feature_calculator.calculate_fundamental_features(
            stock_data.copy()
        )
        feature_cols = [
            c for c in fund_features.columns
            if c not in ("open", "high", "low", "close", "volume")
        ]
        if feature_cols:
            fund_df = fund_features[feature_cols]
            qlib_base = format_converter.add_indicators_to_qlib(
                qlib_base, fund_df, stock_code
            )
    except Exception as fund_err:
        logger.warning(f"{stock_code} 基本面特征计算失败: {fund_err}")

    # 8. 增量合并 & 保存
    output_file = output_dir / f"{stock_code.replace('.', '_')}.parquet"

    if incremental and last_computed_date and output_file.exists():
        # 只保留新增日期的计算结果
        lcd = pd.Timestamp(last_computed_date)
        if isinstance(qlib_base.index, pd.MultiIndex):
            dates_idx = qlib_base.index.get_level_values(1)
        else:
            dates_idx = qlib_base.index
        new_data = qlib_base[dates_idx > lcd]

        if new_data.empty:
            logger.info(f"{stock_code} 无新增因子数据")
            return True

        # 加载现有数据并合并
        existing = format_converter.load_qlib_data(output_file, stock_code)
        if not existing.empty:
            qlib_base = incremental_updater.merge_incremental_data(
                existing, new_data, stock_code
            )
        else:
            qlib_base = new_data

    # 保存
    format_converter.save_qlib_data(qlib_base, output_file, stock_code)
    logger.info(f"✓ {stock_code} 预计算完成，指标数: {len(qlib_base.columns)}")
    return True


def _compute_base_indicators(
    qlib_data: pd.DataFrame,
    stock_code: str,
    format_converter: Any,
) -> pd.DataFrame:
    """计算基础统计指标（RET 系列）"""
    try:
        if isinstance(qlib_data.index, pd.MultiIndex):
            stock_data = qlib_data.xs(stock_code, level=0, drop_level=False)
        else:
            stock_data = qlib_data.copy()

        indicators = pd.DataFrame(index=stock_data.index)
        close = stock_data["$close"]

        for period in (1, 5, 10, 20):
            indicators[f"RET{period}"] = close.pct_change(period)

        if not isinstance(indicators.index, pd.MultiIndex):
            indicators["stock_code"] = stock_code
            indicators = indicators.set_index("stock_code", append=True)
            indicators = indicators.swaplevel(0, 1)

        return format_converter.add_indicators_to_qlib(
            qlib_data, indicators, stock_code
        )
    except Exception as e:
        logger.error(f"基础指标计算失败 {stock_code}: {e}")
        return qlib_data


def _compute_technical_indicators(
    stock_data: pd.DataFrame,
    stock_code: str,
    calculator: Any,
) -> pd.DataFrame:
    """计算技术指标"""
    from app.models.stock_simple import StockData as SD

    stock_list = []
    for date, row in stock_data.iterrows():
        stock_list.append(
            SD(
                stock_code=stock_code,
                date=date.to_pydatetime() if hasattr(date, "to_pydatetime") else date,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=int(row["volume"]),
            )
        )

    indicators = pd.DataFrame(index=stock_data.index)

    # MA
    for period in (5, 10, 20, 30, 50, 60):
        indicators[f"MA{period}"] = list(
            calculator.calculate_moving_average(stock_list, period)
        )

    # RSI
    indicators["RSI14"] = list(calculator.calculate_rsi(stock_list))

    # MACD
    macd = calculator.calculate_macd(stock_list)
    indicators["MACD"] = list(macd.get("macd", []))
    indicators["MACD_SIGNAL"] = list(macd.get("signal", []))
    indicators["MACD_HIST"] = list(macd.get("histogram", []))

    # Bollinger
    bb = calculator.calculate_bollinger_bands(stock_list)
    indicators["BOLLINGER_UPPER"] = list(bb.get("upper", []))
    indicators["BOLLINGER_MIDDLE"] = list(bb.get("middle", []))
    indicators["BOLLINGER_LOWER"] = list(bb.get("lower", []))

    # KDJ
    kdj = calculator.calculate_stochastic(stock_list)
    indicators["KDJ_K"] = list(kdj.get("stoch_k", []))
    indicators["KDJ_D"] = list(kdj.get("stoch_d", []))

    # ATR
    indicators["ATR14"] = list(calculator.calculate_atr(stock_list))

    indicators = indicators.bfill().fillna(0)
    return indicators
