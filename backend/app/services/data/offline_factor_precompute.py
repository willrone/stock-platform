"""
离线因子预计算服务
采用混合预计算模式（Qlib计算基础指标+Alpha158因子，Pandas计算技术指标+基本面特征）
全部离线预计算后存储为Qlib格式，供回测/训练直接读取
"""

from __future__ import annotations  # 延迟评估类型注解，避免在独立进程中类型解析问题

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from app.core.config import settings
from app.models.stock_simple import StockData
from app.services.data.incremental_updater import IncrementalUpdater
from app.services.data.indicator_registry import IndicatorRegistry
from app.services.data.precompute_validator import PrecomputeValidator
from app.services.data.qlib_bin_converter import QlibBinConverter
from app.services.data.qlib_format_converter import QlibFormatConverter
from app.services.data.stock_data_loader import StockDataLoader
from app.services.data.version_manager import VersionManager
from app.services.models.feature_engineering import FeatureCalculator
from app.services.prediction.technical_indicators import TechnicalIndicatorCalculator
from app.services.qlib.enhanced_qlib_provider import (
    Alpha158Calculator,
    EnhancedQlibDataProvider,
)


class OfflineFactorPrecomputeService:
    """离线因子预计算服务"""

    def __init__(
        self,
        batch_size: int = 50,
        max_workers: int = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        初始化预计算服务

        Args:
            batch_size: 每批处理的股票数
            max_workers: 最大并发数（None则自动选择）
            progress_callback: 进度回调函数 (progress: float, message: str) -> None
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or min(8, (os.cpu_count() or 4))
        self.progress_callback = progress_callback

        # 初始化组件
        self.data_loader = StockDataLoader()
        self.format_converter = QlibFormatConverter()
        self.validator = PrecomputeValidator()
        self.incremental_updater = IncrementalUpdater()
        self.version_manager = VersionManager()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.feature_calculator = FeatureCalculator()
        self.bin_converter = QlibBinConverter()

        # Qlib相关组件（延迟初始化）
        self.qlib_provider: Optional[EnhancedQlibDataProvider] = None
        self.alpha_calculator: Optional[Alpha158Calculator] = None

        # Qlib数据路径
        self.qlib_data_path = Path(settings.QLIB_DATA_PATH)
        self.qlib_data_path.mkdir(parents=True, exist_ok=True)

        # 确保交易日历文件存在
        try:
            from app.services.data.qlib_calendar_generator import QlibCalendarGenerator

            calendar_generator = QlibCalendarGenerator()
            calendar_generator.ensure_calendar_exists()
        except Exception as cal_error:
            logger.warning(f"生成交易日历文件失败: {cal_error}")

        logger.info(f"离线因子预计算服务初始化完成，批次大小: {batch_size}, 最大并发数: {self.max_workers}")

    async def initialize_qlib(self):
        """初始化Qlib环境（延迟初始化）"""
        if self.qlib_provider is None:
            self.qlib_provider = EnhancedQlibDataProvider()
            await self.qlib_provider.initialize_qlib()
            self.alpha_calculator = self.qlib_provider.alpha_calculator
            logger.info("Qlib环境初始化完成")

    def get_all_stock_codes(self) -> List[str]:
        """
        获取所有可用的股票代码列表

        Returns:
            股票代码列表
        """
        try:
            from pathlib import Path

            data_root = Path(settings.DATA_ROOT_PATH)
            if not data_root.is_absolute():
                # 如果是相对路径，从backend目录开始解析
                backend_dir = Path(__file__).parent.parent.parent.parent
                data_root = (backend_dir / data_root).resolve()

            stock_data_dir = data_root / "parquet" / "stock_data"

            if not stock_data_dir.exists():
                logger.warning(f"股票数据目录不存在: {stock_data_dir}")
                return []

            stock_codes = []
            for file_path in stock_data_dir.glob("*.parquet"):
                file_name = file_path.stem  # 例如: 000001_SZ

                # 文件名格式: {code}_{market}.parquet
                if "_" in file_name:
                    parts = file_name.split("_")
                    if len(parts) >= 2:
                        code = parts[0]
                        market = parts[1].upper()
                        if market in ["SZ", "SH", "BJ"]:
                            stock_code = f"{code}.{market}"
                            stock_codes.append(stock_code)

            stock_codes = sorted(set(stock_codes))
            logger.info(f"找到 {len(stock_codes)} 只股票")

            return stock_codes

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    def get_stock_date_range(
        self, stock_code: str
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        获取股票的日期范围

        Args:
            stock_code: 股票代码

        Returns:
            (start_date, end_date) 或 None
        """
        try:
            stock_data = self.data_loader.load_stock_data(stock_code)
            if stock_data.empty:
                return None

            dates = stock_data.index
            return (dates.min().to_pydatetime(), dates.max().to_pydatetime())

        except Exception as e:
            logger.warning(f"获取股票 {stock_code} 日期范围失败: {e}")
            return None

    async def precompute_single_stock(
        self,
        stock_code: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        预计算单只股票的所有指标

        Args:
            stock_code: 股票代码
            start_date: 开始日期（可选，None则使用所有可用数据）
            end_date: 结束日期（可选，None则使用所有可用数据）

        Returns:
            Qlib格式的DataFrame（MultiIndex: stock_code, date），包含所有指标，失败返回None
        """
        try:
            # 1. 加载基础数据
            stock_data = self.data_loader.load_stock_data(
                stock_code, start_date, end_date
            )
            if stock_data.empty:
                logger.warning(f"股票 {stock_code} 无数据")
                return None

            # 2. 转换为Qlib格式（基础OHLCV数据）
            qlib_base = self.format_converter.convert_parquet_to_qlib(
                stock_data, stock_code
            )

            # 3. 计算基础统计指标（使用Qlib表达式引擎）
            qlib_base = await self._compute_qlib_base_indicators(qlib_base, stock_code)

            # 4. 计算Alpha158因子（使用Qlib内置Alpha158）
            # 改进：先将基础数据临时保存到Qlib数据目录，然后使用Alpha158 handler计算
            # 确保Qlib已初始化
            if self.alpha_calculator is None:
                await self.initialize_qlib()

            if self.alpha_calculator:
                alpha_factors = await self._compute_alpha158_factors_with_handler(
                    qlib_base, stock_code
                )
                if not alpha_factors.empty:
                    qlib_base = self.format_converter.add_indicators_to_qlib(
                        qlib_base, alpha_factors, stock_code
                    )

            # 5. 计算技术指标（使用Pandas）
            technical_indicators = await self._compute_technical_indicators(
                stock_data, stock_code
            )
            if not technical_indicators.empty:
                qlib_base = self.format_converter.add_indicators_to_qlib(
                    qlib_base, technical_indicators, stock_code
                )

            # 6. 计算基本面特征（使用Pandas）
            fundamental_features = await self._compute_fundamental_features(
                stock_data, stock_code
            )
            if not fundamental_features.empty:
                qlib_base = self.format_converter.add_indicators_to_qlib(
                    qlib_base, fundamental_features, stock_code
                )

            logger.info(f"股票 {stock_code} 预计算完成，指标数: {len(qlib_base.columns)}")

            return qlib_base

        except Exception as e:
            logger.error(f"预计算股票 {stock_code} 失败: {e}", exc_info=True)
            return None

    async def _compute_qlib_base_indicators(
        self, qlib_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """
        使用Qlib表达式引擎计算基础统计指标

        Args:
            qlib_data: Qlib格式的基础数据
            stock_code: 股票代码

        Returns:
            添加了基础指标的DataFrame
        """
        try:
            # 提取单股票数据
            if isinstance(qlib_data.index, pd.MultiIndex):
                stock_data = qlib_data.xs(stock_code, level=0, drop_level=False)
            else:
                stock_data = qlib_data.copy()

            indicators = pd.DataFrame(index=stock_data.index)

            # 使用pandas计算基础统计指标（Qlib表达式引擎需要完整的数据集，这里先用pandas实现）
            # 注意：避免与后续技术指标计算重复，这里只计算基础统计指标，不计算MA、STD等（这些会在技术指标中计算）
            close = stock_data["$close"]
            stock_data["$volume"]

            # 价格变化率（这些是基础指标，不会与技术指标重复）
            indicators["RET1"] = close.pct_change(1)
            indicators["RET5"] = close.pct_change(5)
            indicators["RET10"] = close.pct_change(10)
            indicators["RET20"] = close.pct_change(20)

            # 注意：MA、STD、CORR、MAX、MIN等指标会在技术指标计算中生成，这里不重复计算
            # 如果需要，可以在这里计算其他不重复的基础指标

            # 添加stock_code到索引（如果是单股票数据）
            if not isinstance(indicators.index, pd.MultiIndex):
                indicators["stock_code"] = stock_code
                indicators = indicators.set_index("stock_code", append=True)
                indicators = indicators.swaplevel(0, 1)

            return self.format_converter.add_indicators_to_qlib(
                qlib_data, indicators, stock_code
            )

        except Exception as e:
            logger.error(f"计算基础指标失败 {stock_code}: {e}")
            return qlib_data

    async def _compute_alpha158_factors_with_handler(
        self, qlib_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """
        使用Qlib内置Alpha158 handler计算158个标准因子

        改进方案：先将基础数据临时保存到Qlib数据目录，然后使用Alpha158 handler计算

        Args:
            qlib_data: Qlib格式的基础数据
            stock_code: 股票代码

        Returns:
            Alpha158因子DataFrame
        """
        try:
            if not self.alpha_calculator:
                return pd.DataFrame()

            # 提取单股票数据
            if isinstance(qlib_data.index, pd.MultiIndex):
                stock_data = qlib_data.xs(stock_code, level=0, drop_level=False)
            else:
                stock_data = qlib_data.copy()

            if stock_data.empty:
                return pd.DataFrame()

            # 获取日期范围
            dates = (
                stock_data.index.get_level_values(1)
                if isinstance(stock_data.index, pd.MultiIndex)
                else stock_data.index
            )
            start_date = dates.min().to_pydatetime()
            end_date = dates.max().to_pydatetime()

            # 生成Qlib bin数据（方案A）
            try:
                self.bin_converter.convert_parquet_to_bin(
                    stock_data, stock_code, self.qlib_data_path
                )
            except Exception as bin_error:
                logger.warning(f"股票 {stock_code} 生成Qlib bin失败: {bin_error}")

            # 优先使用handler计算
            try:
                alpha_factors = await self.alpha_calculator.calculate_alpha_factors(
                    qlib_data=qlib_data,
                    stock_codes=[stock_code],
                    date_range=(start_date, end_date),
                    use_cache=False,
                    force_expression_engine=False,
                )
                if not alpha_factors.empty and len(alpha_factors.columns) >= 158:
                    logger.info(
                        f"股票 {stock_code} Alpha158 handler计算成功，因子数: {len(alpha_factors.columns)}"
                    )
                    return alpha_factors
                factor_count = (
                    len(alpha_factors.columns) if not alpha_factors.empty else 0
                )
                logger.warning(
                    f"股票 {stock_code} Alpha158 handler因子数不足: {factor_count}，期望158个"
                )
                raise ValueError(f"Handler因子数不足: {factor_count} < 158")
            except Exception as handler_error:
                logger.warning(
                    f"股票 {stock_code} Alpha158 handler计算失败: {handler_error}，回退到表达式引擎"
                )
                alpha_factors = await self.alpha_calculator.calculate_alpha_factors(
                    qlib_data=qlib_data,
                    stock_codes=[stock_code],
                    date_range=(start_date, end_date),
                    use_cache=False,
                    force_expression_engine=True,
                )
                if not alpha_factors.empty:
                    logger.warning(
                        f"股票 {stock_code} 表达式引擎计算完成，因子数: {len(alpha_factors.columns)}（可能不足158个）"
                    )
                return alpha_factors

        except Exception as e:
            logger.warning(f"计算Alpha158因子失败 {stock_code}: {e}", exc_info=True)
            return pd.DataFrame()

    async def _compute_alpha158_factors(
        self, qlib_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """
        使用Qlib内置Alpha158计算158个标准因子（旧方法，保留作为备用）

        Args:
            qlib_data: Qlib格式的基础数据
            stock_code: 股票代码

        Returns:
            Alpha158因子DataFrame
        """
        # 调用新方法
        return await self._compute_alpha158_factors_with_handler(qlib_data, stock_code)

    async def _compute_technical_indicators(
        self, stock_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """
        使用Pandas计算技术指标

        Args:
            stock_data: 单股票DataFrame（索引为日期）
            stock_code: 股票代码

        Returns:
            技术指标DataFrame（索引为日期）
        """
        try:
            # 转换为StockData列表
            stock_data_list = []
            for date, row in stock_data.iterrows():
                stock_data_list.append(
                    StockData(
                        stock_code=stock_code,
                        date=date.to_pydatetime()
                        if isinstance(date, pd.Timestamp)
                        else date,
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=int(row["volume"]),
                    )
                )

            # 计算所有技术指标
            indicators = pd.DataFrame(index=stock_data.index)

            # 移动平均线（包括策略常用的周期）
            indicators["MA5"] = [
                v
                for v in self.indicator_calculator.calculate_moving_average(
                    stock_data_list, 5
                )
            ]
            indicators["MA10"] = [
                v
                for v in self.indicator_calculator.calculate_moving_average(
                    stock_data_list, 10
                )
            ]
            indicators["MA20"] = [
                v
                for v in self.indicator_calculator.calculate_moving_average(
                    stock_data_list, 20
                )
            ]
            indicators["MA30"] = [
                v
                for v in self.indicator_calculator.calculate_moving_average(
                    stock_data_list, 30
                )
            ]
            indicators["MA50"] = [
                v
                for v in self.indicator_calculator.calculate_moving_average(
                    stock_data_list, 50
                )
            ]  # RSI策略的trend_ma需要
            indicators["MA60"] = [
                v
                for v in self.indicator_calculator.calculate_moving_average(
                    stock_data_list, 60
                )
            ]

            # RSI
            rsi_values = self.indicator_calculator.calculate_rsi(stock_data_list)
            indicators["RSI14"] = [v for v in rsi_values]

            # MACD
            macd_result = self.indicator_calculator.calculate_macd(stock_data_list)
            indicators["MACD"] = [v for v in macd_result.get("macd", [])]
            indicators["MACD_SIGNAL"] = [v for v in macd_result.get("signal", [])]
            indicators["MACD_HIST"] = [v for v in macd_result.get("histogram", [])]

            # 布林带
            bollinger_result = self.indicator_calculator.calculate_bollinger_bands(
                stock_data_list
            )
            indicators["BOLLINGER_UPPER"] = [
                v for v in bollinger_result.get("upper", [])
            ]
            indicators["BOLLINGER_MIDDLE"] = [
                v for v in bollinger_result.get("middle", [])
            ]
            indicators["BOLLINGER_LOWER"] = [
                v for v in bollinger_result.get("lower", [])
            ]

            # KDJ
            kdj_result = self.indicator_calculator.calculate_stochastic(stock_data_list)
            indicators["KDJ_K"] = [v for v in kdj_result.get("stoch_k", [])]
            indicators["KDJ_D"] = [v for v in kdj_result.get("stoch_d", [])]

            # ATR
            atr_values = self.indicator_calculator.calculate_atr(stock_data_list)
            indicators["ATR14"] = [v for v in atr_values]

            # 填充缺失值
            indicators = indicators.bfill().fillna(0)

            logger.debug(f"股票 {stock_code} 技术指标计算完成，指标数: {len(indicators.columns)}")

            return indicators

        except Exception as e:
            logger.error(f"计算技术指标失败 {stock_code}: {e}")
            return pd.DataFrame()

    async def _compute_fundamental_features(
        self, stock_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """
        使用Pandas计算基本面特征

        Args:
            stock_data: 单股票DataFrame（索引为日期）
            stock_code: 股票代码

        Returns:
            基本面特征DataFrame（索引为日期）
        """
        try:
            features = self.feature_calculator.calculate_fundamental_features(
                stock_data.copy()
            )

            # 只返回新增的特征列（排除原有的OHLCV列）
            feature_cols = [
                col
                for col in features.columns
                if col not in ["open", "high", "low", "close", "volume"]
            ]
            feature_df = features[feature_cols].copy()

            logger.debug(f"股票 {stock_code} 基本面特征计算完成，特征数: {len(feature_df.columns)}")

            return feature_df

        except Exception as e:
            logger.error(f"计算基本面特征失败 {stock_code}: {e}")
            return pd.DataFrame()

    async def precompute_all_stocks(
        self,
        stock_codes: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        incremental: bool = True,
        force_update: bool = False,
    ) -> "Dict[str, Any]":
        """
        预计算所有股票的所有指标

        Args:
            stock_codes: 股票代码列表（None则自动获取所有股票）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            包含统计信息的字典
        """
        try:
            # 初始化Qlib
            await self.initialize_qlib()

            # 获取股票列表
            if stock_codes is None:
                stock_codes = self.get_all_stock_codes()

            if not stock_codes:
                logger.error("没有可用的股票代码")
                return {
                    "success": False,
                    "message": "没有可用的股票代码",
                    "total_stocks": 0,
                    "success_stocks": 0,
                    "failed_stocks": [],
                }

            # 增量更新：只处理需要更新的股票
            if incremental and not force_update:
                stocks_to_update = self.incremental_updater.get_stocks_to_update(
                    stock_codes, force_update=False
                )
                if stocks_to_update:
                    logger.info(f"增量更新模式: 检测到 {len(stocks_to_update)} 只股票需要更新")
                    stock_codes = stocks_to_update
                else:
                    logger.info("增量更新模式: 没有股票需要更新")
                    return {
                        "success": True,
                        "message": "所有股票数据已是最新，无需更新",
                        "total_stocks": len(self.get_all_stock_codes()),
                        "success_stocks": len(self.get_all_stock_codes()),
                        "failed_stocks": [],
                        "incremental": True,
                    }

            total_stocks = len(stock_codes)
            logger.info(f"开始预计算 {total_stocks} 只股票的所有指标")

            # 分批处理
            success_stocks = []
            failed_stocks = []
            all_precomputed_data = []

            # 创建输出目录
            output_dir = self.qlib_data_path / "features" / "day"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 分批处理股票
            for batch_start in range(0, total_stocks, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_stocks)
                batch_codes = stock_codes[batch_start:batch_end]

                logger.info(
                    f"处理批次 {batch_start // self.batch_size + 1}: 股票 {batch_start + 1}-{batch_end}/{total_stocks}"
                )

                # 并行处理批次内的股票（使用异步方式）
                tasks = [
                    self.precompute_single_stock(code, start_date, end_date)
                    for code in batch_codes
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for stock_code, result in zip(batch_codes, results):
                    try:
                        if isinstance(result, Exception):
                            failed_stocks.append(stock_code)
                            logger.error(f"✗ 股票 {stock_code} 预计算失败: {result}")
                        elif result is not None and not result.empty:
                            # 增量更新：如果文件已存在，合并数据
                            output_file = (
                                output_dir / f"{stock_code.replace('.', '_')}.parquet"
                            )

                            if incremental and output_file.exists():
                                # 加载现有数据并合并
                                existing_data = self.format_converter.load_qlib_data(
                                    output_file, stock_code
                                )
                                if not existing_data.empty:
                                    result = (
                                        self.incremental_updater.merge_incremental_data(
                                            existing_data, result, stock_code
                                        )
                                    )
                                    logger.debug(f"股票 {stock_code} 增量更新完成")

                            # 保存到文件
                            self.format_converter.save_qlib_data(
                                result, output_file, stock_code
                            )

                            success_stocks.append(stock_code)
                            all_precomputed_data.append(result)

                            logger.info(f"✓ 股票 {stock_code} 预计算完成")
                        else:
                            failed_stocks.append(stock_code)
                            logger.warning(f"✗ 股票 {stock_code} 预计算失败（无数据）")
                    except Exception as e:
                        failed_stocks.append(stock_code)
                        logger.error(f"✗ 股票 {stock_code} 预计算失败: {e}")

                # 更新进度
                progress = (batch_end / total_stocks) * 100
                if self.progress_callback:
                    self.progress_callback(
                        progress, f"已处理 {batch_end}/{total_stocks} 只股票"
                    )

            # 合并所有数据（可选，用于创建统一的数据文件）
            if all_precomputed_data:
                logger.info("合并所有预计算数据...")
                combined_data = pd.concat(all_precomputed_data, axis=0)
                combined_data = combined_data.sort_index()

                # 保存合并后的数据
                combined_file = output_dir / "all_stocks.parquet"
                self.format_converter.save_qlib_data(combined_data, combined_file)
                logger.info(f"合并数据已保存: {combined_file}, 形状: {combined_data.shape}")

            # 更新版本信息
            try:
                # 获取所有股票和日期范围
                all_stocks = self.get_all_stock_codes()
                if all_precomputed_data:
                    # 从合并数据获取日期范围
                    dates = (
                        combined_data.index.get_level_values(1)
                        if isinstance(combined_data.index, pd.MultiIndex)
                        else combined_data.index
                    )
                    date_range = {
                        "start": dates.min().strftime("%Y-%m-%d"),
                        "end": dates.max().strftime("%Y-%m-%d"),
                    }
                else:
                    date_range = {"start": None, "end": None}

                # 获取指标列表
                from app.services.data.indicator_registry import IndicatorCategory

                indicators = {
                    "technical": list(
                        IndicatorRegistry.get_indicators_by_category(
                            IndicatorCategory.TECHNICAL
                        ).keys()
                    ),
                    "alpha": list(
                        IndicatorRegistry.get_indicators_by_category(
                            IndicatorCategory.ALPHA
                        ).keys()
                    ),
                    "fundamental": list(
                        IndicatorRegistry.get_indicators_by_category(
                            IndicatorCategory.FUNDAMENTAL
                        ).keys()
                    ),
                    "base": list(
                        IndicatorRegistry.get_indicators_by_category(
                            IndicatorCategory.BASE
                        ).keys()
                    ),
                }

                self.version_manager.update_version_info(
                    stock_count=len(all_stocks),
                    date_range=date_range,
                    indicators=indicators,
                )
            except Exception as e:
                logger.warning(f"更新版本信息失败: {e}")

            # 生成统计信息
            result = {
                "success": True,
                "message": f"预计算完成: 成功 {len(success_stocks)}/{total_stocks} 只股票",
                "total_stocks": total_stocks,
                "success_stocks": len(success_stocks),
                "failed_stocks": failed_stocks,
                "output_dir": str(output_dir),
                "incremental": incremental,
            }

            logger.info(f"预计算完成: 成功 {len(success_stocks)}/{total_stocks} 只股票")

            return result

        except Exception as e:
            logger.error(f"预计算所有股票失败: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"预计算失败: {str(e)}",
                "total_stocks": 0,
                "success_stocks": 0,
                "failed_stocks": [],
            }
