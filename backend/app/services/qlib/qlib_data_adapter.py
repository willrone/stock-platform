"""Qlib 数据适配与格式校验模块。"""

from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger


class QlibDataAdapter:
    """负责 DataFrame 与 Qlib 标准格式之间的转换与校验。"""

    def _convert_to_qlib_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为Qlib标准格式 - 优化版本"""
        if df.empty:
            return pd.DataFrame()

        logger.debug(f"开始转换Qlib格式: 输入数据 {df.shape}")

        # 1. 处理索引格式
        df_qlib = self._ensure_multiindex_format(df)

        # 2. 标准化列名
        df_qlib = self._standardize_column_names(df_qlib)

        # 3. 数据类型优化
        df_qlib = self._optimize_data_types(df_qlib)

        # 4. 处理缺失值
        df_qlib = self._handle_missing_values(df_qlib)

        # 5. 排序和去重
        df_qlib = self._sort_and_deduplicate(df_qlib)

        logger.info(f"Qlib格式转换完成: {df_qlib.shape}, 列: {list(df_qlib.columns)}")
        return df_qlib

    def _ensure_multiindex_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保数据使用MultiIndex格式 (instrument, datetime)"""
        if isinstance(df.index, pd.MultiIndex):
            # 已经是MultiIndex，检查层级名称
            if len(df.index.names) == 2:
                # 标准化索引名称
                df.index.names = ["instrument", "datetime"]
                return df
            else:
                logger.warning(f"MultiIndex层级数不正确: {len(df.index.names)}")

        # 需要创建MultiIndex
        if "stock_code" in df.columns and "date" in df.columns:
            # 确保date列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                df["date"] = pd.to_datetime(df["date"])

            # 设置MultiIndex
            df_indexed = df.set_index(["stock_code", "date"])
            df_indexed.index.names = ["instrument", "datetime"]
            return df_indexed

        elif isinstance(df.index, pd.DatetimeIndex) and "stock_code" in df.columns:
            # 日期在索引中，股票代码在列中
            df_reset = df.reset_index()
            df_reset.rename(columns={"index": "date"}, inplace=True)
            df_reset["date"] = pd.to_datetime(df_reset["date"])
            df_indexed = df_reset.set_index(["stock_code", "date"])
            df_indexed.index.names = ["instrument", "datetime"]
            return df_indexed

        else:
            logger.warning("无法创建MultiIndex，缺少必要的股票代码或日期信息")
            return df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名为Qlib格式"""
        # Qlib标准列名映射
        column_mapping = {
            # 基础OHLCV数据
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
            "adj_close": "$close",  # 如果有复权价格，使用它作为收盘价
            # 技术指标（保持原名或添加前缀）
            "MA5": "MA5",
            "MA10": "MA10",
            "MA20": "MA20",
            "MA60": "MA60",
            "EMA": "EMA20",
            "WMA": "WMA20",
            "RSI": "RSI14",
            "MACD": "MACD",
            "MACD_SIGNAL": "MACD_SIGNAL",
            "MACD_HISTOGRAM": "MACD_HIST",
            "BOLLINGER_UPPER": "BOLL_UPPER",
            "BOLLINGER_MIDDLE": "BOLL_MIDDLE",
            "BOLLINGER_LOWER": "BOLL_LOWER",
            "ATR": "ATR14",
            "VWAP": "VWAP",
            "OBV": "OBV",
            "STOCH_K": "STOCH_K",
            "STOCH_D": "STOCH_D",
            "WILLIAMS_R": "WILLIAMS_R",
            "CCI": "CCI20",
            "KDJ_K": "KDJ_K",
            "KDJ_D": "KDJ_D",
            "KDJ_J": "KDJ_J",
            # 基本面特征
            "price_change": "RET1",
            "price_change_5d": "RET5",
            "price_change_20d": "RET20",
            "volume_change": "VOLUME_RET1",
            "volume_ma_ratio": "VOLUME_MA_RATIO",
            "volatility_5d": "VOLATILITY5",
            "volatility_20d": "VOLATILITY20",
            "price_position": "PRICE_POSITION",
        }

        # 只重命名存在的列
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_renamed = df.rename(columns=existing_mapping)

        # 重命名后可能产生重复列（例如 RSI -> RSI14，而原始数据已存在 RSI14）。
        # Pandas 在按列名取值时会返回 DataFrame 而不是 Series，后续 dtype/填充逻辑会报错。
        if df_renamed.columns.duplicated().any():
            duplicate_columns = df_renamed.columns[
                df_renamed.columns.duplicated(keep=False)
            ].tolist()
            logger.warning(f"列名标准化后发现重复列，保留最后一个: {duplicate_columns}")
            df_renamed = df_renamed.loc[:, ~df_renamed.columns.duplicated(keep="last")]

        # 确保基础OHLCV列存在
        required_base_cols = ["$open", "$high", "$low", "$close", "$volume"]
        missing_base_cols = [
            col for col in required_base_cols if col not in df_renamed.columns
        ]

        if missing_base_cols:
            logger.warning(f"缺少基础OHLCV列: {missing_base_cols}")

        logger.debug(f"列名标准化完成: {len(existing_mapping)} 个列被重命名")
        return df_renamed

    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型以节省内存"""
        df_optimized = df.copy()

        # 价格相关列使用float32
        price_cols = ["$open", "$high", "$low", "$close"]
        for col in price_cols:
            if col in df_optimized.columns:
                df_optimized[col] = pd.to_numeric(
                    df_optimized[col], errors="coerce"
                ).astype("float32")

        # 成交量使用int64（可能很大）
        if "$volume" in df_optimized.columns:
            df_optimized["$volume"] = pd.to_numeric(
                df_optimized["$volume"], errors="coerce"
            ).astype("int64")

        # 技术指标使用float32
        indicator_cols = [
            col for col in df_optimized.columns if col not in price_cols + ["$volume"]
        ]
        for col in indicator_cols:
            if df_optimized[col].dtype in ["float64", "object"]:
                df_optimized[col] = pd.to_numeric(
                    df_optimized[col], errors="coerce"
                ).astype("float32")

        logger.debug("数据类型优化完成")
        return df_optimized

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值 - 改进版：区分缺失值类型，使用更智能的填充策略"""
        df_filled = df.copy()

        # 确保数据按时间排序（避免未来信息泄漏）
        if isinstance(df_filled.index, pd.MultiIndex):
            df_filled = df_filled.sort_index()
        elif df_filled.index.name in ["datetime", "date", "time"] or isinstance(
            df_filled.index, pd.DatetimeIndex
        ):
            df_filled = df_filled.sort_index()

        # 基础价格数据：前向填充（停牌等情况）
        price_cols = ["$open", "$high", "$low", "$close", "$volume"]
        for col in price_cols:
            if col in df_filled.columns:
                # 前向填充，然后后向填充（处理开头缺失）
                df_filled[col] = df_filled[col].ffill().bfill()

        # 技术指标：区分缺失原因
        indicator_cols = [
            col for col in df_filled.columns if col not in price_cols + ["label"]
        ]

        for col in indicator_cols:
            if col not in df_filled.columns:
                continue

            col_data = df_filled[col]
            missing_mask = col_data.isna()

            if not missing_mask.any():
                continue

            missing_count = missing_mask.sum()
            total_count = len(col_data)
            missing_ratio = missing_count / total_count if total_count > 0 else 0

            # 判断缺失原因：
            # 1. 如果缺失比例很高（>50%），可能是计算窗口不足，使用中位数填充
            # 2. 如果缺失比例较低，可能是数据缺失，使用前向填充
            # 3. 对于技术指标，如果开头缺失（计算窗口不足），使用NaN或中位数
            # 4. 对于中间缺失（数据缺失），使用前向填充

            if missing_ratio > 0.5:
                # 高缺失率：可能是计算窗口不足，使用中位数填充
                median_value = col_data.median()
                if pd.notna(median_value):
                    df_filled[col] = col_data.fillna(median_value)
                else:
                    # 如果中位数也是NaN，使用0（作为最后手段）
                    df_filled[col] = col_data.fillna(0)
                logger.debug(f"列 {col} 缺失率 {missing_ratio:.2%}，使用中位数填充")
            else:
                # 低缺失率：可能是数据缺失，使用前向填充
                # 先前向填充，然后后向填充（处理开头缺失）
                df_filled[col] = col_data.ffill().bfill()

                # 如果仍有缺失（开头），使用中位数
                if df_filled[col].isna().any():
                    median_value = df_filled[col].median()
                    if pd.notna(median_value):
                        df_filled[col] = df_filled[col].fillna(median_value)
                    else:
                        df_filled[col] = df_filled[col].fillna(0)

                logger.debug(
                    f"列 {col} 缺失率 {missing_ratio:.2%}，使用前向填充+中位数"
                )

        # 记录缺失值处理情况
        missing_counts_before = df.isnull().sum()
        missing_counts_after = df_filled.isnull().sum()

        if missing_counts_before.sum() > 0:
            logger.debug(
                f"缺失值处理完成 - 处理前: {missing_counts_before[missing_counts_before > 0].to_dict()}, "
                f"处理后: {missing_counts_after[missing_counts_after > 0].to_dict()}"
            )

        return df_filled

    def _sort_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """排序和去重"""
        if not isinstance(df.index, pd.MultiIndex):
            return df

        # 按instrument和datetime排序
        df_sorted = df.sort_index()

        # 去除重复的索引
        if df_sorted.index.duplicated().any():
            logger.warning(f"发现重复索引，去重前: {len(df_sorted)}")
            df_sorted = df_sorted[~df_sorted.index.duplicated(keep="last")]
            logger.warning(f"去重后: {len(df_sorted)}")

        return df_sorted

    def _add_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基本面特征"""
        # 价格变化率
        df["price_change"] = df["close"].pct_change()
        df["price_change_5d"] = df["close"].pct_change(periods=5)
        df["price_change_20d"] = df["close"].pct_change(periods=20)

        # 成交量变化率
        df["volume_change"] = df["volume"].pct_change()
        df["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # 波动率
        df["volatility_5d"] = df["price_change"].rolling(5).std()
        df["volatility_20d"] = df["price_change"].rolling(20).std()

        # 价格位置
        df["price_position"] = (df["close"] - df["low"].rolling(20).min()) / (
            df["high"].rolling(20).max() - df["low"].rolling(20).min()
        )

        return df

    async def create_qlib_model_config(
        self, model_type: str, hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建Qlib模型配置。"""
        kwargs: Dict[str, Any] = {
            # 使用 Huber 损失提高对异常标签的鲁棒性。
            "loss": "huber",
            "huber_delta": 0.1,
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        }
        base_config: Dict[str, Any] = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": kwargs,
        }

        if model_type.lower() == "lightgbm":
            base_config["class"] = "LGBModel"
            base_config["module_path"] = "qlib.contrib.model.gbdt"
        elif model_type.lower() == "xgboost":
            base_config["class"] = "XGBModel"
            base_config["module_path"] = "qlib.contrib.model.xgboost"
        elif model_type.lower() == "mlp":
            base_config["class"] = "DNNModelPytorch"
            base_config["module_path"] = "qlib.contrib.model.pytorch_nn"

        if hyperparameters:
            kwargs.update(hyperparameters)

        return base_config

    async def validate_and_fix_qlib_format(
        self, data: pd.DataFrame
    ) -> Tuple[bool, pd.DataFrame]:
        """验证并修复Qlib数据格式。"""
        try:
            logger.info("开始验证和修复Qlib数据格式")
            if data.empty:
                logger.warning("数据为空")
                return False, data

            if not isinstance(data.index, pd.MultiIndex):
                logger.info("修复MultiIndex格式")
                data = self._ensure_multiindex_format(data)

            if len(data.index.names) != 2:
                logger.warning(f"索引层级数不正确: {len(data.index.names)}")
                return False, data

            required_cols = ["$open", "$high", "$low", "$close", "$volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"缺少必要的列: {missing_cols}")
                data = self._fix_missing_columns(data, missing_cols)
                still_missing = [
                    col for col in required_cols if col not in data.columns
                ]
                if still_missing:
                    logger.error(f"无法修复缺少的列: {still_missing}")
                    return False, data

            data = self._fix_data_types(data)
            quality_issues = self._check_data_quality(data)
            if quality_issues:
                logger.warning(f"数据质量问题: {quality_issues}")
                data = self._fix_data_quality_issues(data, quality_issues)

            is_valid = await self.validate_qlib_data_format(data)
            logger.info(
                f"Qlib格式验证和修复完成: 有效={is_valid}, 数据形状={data.shape}"
            )
            return is_valid, data
        except Exception as exc:
            logger.error(f"Qlib格式验证和修复失败: {exc}")
            return False, data

    def _fix_missing_columns(
        self, data: pd.DataFrame, missing_cols: List[str]
    ) -> pd.DataFrame:
        """修复缺少的列。"""
        data_fixed = data.copy()
        column_alternatives = {
            "$open": ["open", "Open", "OPEN"],
            "$high": ["high", "High", "HIGH"],
            "$low": ["low", "Low", "LOW"],
            "$close": ["close", "Close", "CLOSE", "adj_close", "Adj_Close"],
            "$volume": ["volume", "Volume", "VOLUME", "vol", "Vol"],
        }

        for missing_col in missing_cols:
            if missing_col not in column_alternatives:
                continue
            for alternative in column_alternatives[missing_col]:
                if alternative in data_fixed.columns:
                    data_fixed[missing_col] = data_fixed[alternative]
                    logger.info(f"从 {alternative} 推导出 {missing_col}")
                    break
            else:
                if missing_col == "$volume":
                    data_fixed[missing_col] = 1000000
                elif "$close" in data_fixed.columns:
                    data_fixed[missing_col] = data_fixed["$close"]
                else:
                    data_fixed[missing_col] = 100.0
                logger.warning(f"使用默认值填充 {missing_col}")

        return data_fixed

    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """修复数据类型。"""
        data_fixed = data.copy()
        price_cols = ["$open", "$high", "$low", "$close"]
        for col in price_cols:
            if col in data_fixed.columns:
                data_fixed[col] = pd.to_numeric(data_fixed[col], errors="coerce")

        if "$volume" in data_fixed.columns:
            data_fixed["$volume"] = pd.to_numeric(
                data_fixed["$volume"], errors="coerce"
            )
            data_fixed["$volume"] = data_fixed["$volume"].fillna(0).astype("int64")

        for col in data_fixed.columns:
            if (
                col not in price_cols + ["$volume"]
                and data_fixed[col].dtype == "object"
            ):
                data_fixed[col] = pd.to_numeric(data_fixed[col], errors="coerce")

        return data_fixed

    def _check_data_quality(self, data: pd.DataFrame) -> List[str]:
        """检查数据质量问题。"""
        issues: List[str] = []
        if all(col in data.columns for col in ["$open", "$high", "$low", "$close"]):
            invalid_high_low = (data["$high"] < data["$low"]).sum()
            if invalid_high_low > 0:
                issues.append(f"high < low: {invalid_high_low} 条记录")
            for col in ["$open", "$high", "$low", "$close"]:
                negative_prices = (data[col] <= 0).sum()
                if negative_prices > 0:
                    issues.append(f"{col} <= 0: {negative_prices} 条记录")

        if "$volume" in data.columns:
            negative_volume = (data["$volume"] < 0).sum()
            if negative_volume > 0:
                issues.append(f"负成交量: {negative_volume} 条记录")

        missing_counts = data.isnull().sum()
        critical_missing = missing_counts[missing_counts > len(data) * 0.1]
        if not critical_missing.empty:
            issues.append(f"高缺失率列: {critical_missing.to_dict()}")

        return issues

    def _fix_data_quality_issues(
        self, data: pd.DataFrame, issues: List[str]
    ) -> pd.DataFrame:
        """修复数据质量问题。"""
        data_fixed = data.copy()
        if all(
            col in data_fixed.columns for col in ["$open", "$high", "$low", "$close"]
        ):
            invalid_mask = data_fixed["$high"] < data_fixed["$low"]
            if invalid_mask.sum() > 0:
                data_fixed.loc[invalid_mask, ["$high", "$low"]] = data_fixed.loc[
                    invalid_mask, ["$low", "$high"]
                ].values
                logger.info(f"修复了 {invalid_mask.sum()} 条 high < low 的记录")

            for col in ["$open", "$high", "$low", "$close"]:
                negative_mask = data_fixed[col] <= 0
                if negative_mask.sum() > 0:
                    data_fixed.loc[negative_mask, col] = data_fixed[col].ffill()
                    still_negative = data_fixed[col] <= 0
                    if still_negative.sum() > 0:
                        mean_price = data_fixed[col][data_fixed[col] > 0].mean()
                        data_fixed.loc[still_negative, col] = mean_price
                    logger.info(f"修复了 {negative_mask.sum()} 条 {col} <= 0 的记录")

        if "$volume" in data_fixed.columns:
            negative_mask = data_fixed["$volume"] < 0
            if negative_mask.sum() > 0:
                data_fixed.loc[negative_mask, "$volume"] = 0
                logger.info(f"修复了 {negative_mask.sum()} 条负成交量记录")

        missing_counts = data_fixed.isnull().sum()
        high_missing_cols = missing_counts[missing_counts > len(data_fixed) * 0.5].index
        for col in high_missing_cols:
            if col in ["$open", "$high", "$low", "$close"]:
                data_fixed[col] = data_fixed[col].ffill().bfill()
            elif col == "$volume":
                data_fixed[col] = data_fixed[col].fillna(0)
            else:
                data_fixed[col] = data_fixed[col].fillna(0)

        return data_fixed

    async def validate_qlib_data_format(self, data: pd.DataFrame) -> bool:
        """验证Qlib数据格式。"""
        try:
            if not isinstance(data.index, pd.MultiIndex):
                logger.warning("数据索引不是MultiIndex格式")
                return False
            if len(data.index.names) != 2:
                logger.warning("数据索引应该有两个层级")
                return False

            required_cols = ["$close", "$high", "$low", "$open", "$volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"缺少必要的列: {missing_cols}")
                return False

            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    logger.warning(f"列 {col} 不是数值类型")
                    return False

            logger.info("Qlib数据格式验证通过")
            return True
        except Exception as exc:
            logger.error(f"Qlib数据格式验证失败: {exc}")
            return False

    async def convert_dataframe_to_qlib(
        self, df: pd.DataFrame, validate: bool = True, fix_issues: bool = True
    ) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """将DataFrame转换为Qlib格式的主要接口。"""
        conversion_info = {
            "input_shape": df.shape,
            "input_columns": list(df.columns),
            "conversion_steps": [],
            "issues_found": [],
            "issues_fixed": [],
        }

        try:
            logger.info(f"开始Qlib格式转换: 输入 {df.shape}")
            converted_df = self._convert_to_qlib_format(df)
            conversion_info["conversion_steps"].append("基本格式转换")
            conversion_info["output_shape"] = converted_df.shape
            conversion_info["output_columns"] = list(converted_df.columns)

            if validate or fix_issues:
                is_valid, fixed_df = await self.validate_and_fix_qlib_format(
                    converted_df
                )
                conversion_info["is_valid_before_fix"] = is_valid
                if fix_issues and not is_valid:
                    converted_df = fixed_df
                    conversion_info["conversion_steps"].append("问题修复")
                    is_valid, _ = await self.validate_and_fix_qlib_format(converted_df)
                    conversion_info["is_valid_after_fix"] = is_valid
            else:
                is_valid = True

            conversion_info["final_shape"] = converted_df.shape
            conversion_info["final_columns"] = list(converted_df.columns)
            conversion_info["memory_usage_mb"] = (
                converted_df.memory_usage(deep=True).sum() / 1024 / 1024
            )
            logger.info(
                f"Qlib格式转换完成: {conversion_info['final_shape']}, 有效={is_valid}"
            )
            return is_valid, converted_df, conversion_info
        except Exception as exc:
            logger.error(f"Qlib格式转换失败: {exc}")
            conversion_info["error"] = str(exc)
            return False, df, conversion_info

    async def get_qlib_format_example(self) -> Dict[str, Any]:
        """获取Qlib格式示例和说明。"""
        return {
            "description": "Qlib数据格式要求",
            "index_format": {
                "type": "MultiIndex",
                "levels": ["instrument", "datetime"],
                "example": "('000001.SZ', '2023-01-01')",
            },
            "required_columns": {
                "$open": "开盘价",
                "$high": "最高价",
                "$low": "最低价",
                "$close": "收盘价",
                "$volume": "成交量",
            },
            "optional_columns": {
                "technical_indicators": "技术指标 (RSI, MACD, etc.)",
                "alpha_factors": "Alpha因子 (RESI5, MA10, etc.)",
                "fundamental_features": "基本面特征 (RET1, VOLATILITY5, etc.)",
            },
            "data_types": {
                "prices": "float32",
                "volume": "int64",
                "indicators": "float32",
            },
            "quality_requirements": [
                "价格必须为正数",
                "最高价 >= 最低价",
                "成交量 >= 0",
                "无重复的时间戳",
                "按时间排序",
            ],
        }
