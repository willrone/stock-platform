"""
增强版 Qlib 数据提供器

基于现有 QlibDataProvider，添加 Alpha158 因子计算和缓存机制。
此模块作为协调器，整合各个子模块的功能。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

# 导入子模块
from .alpha158 import Alpha158Calculator
from .cache import FactorCache
from .converters import QlibFormatConverter
from .patches import (
    ALPHA158_AVAILABLE,
    QLIB_AVAILABLE,
    apply_qlib_patches,
)
from .validators import DataQualityValidator, ValidationReport

# 应用 Qlib 兼容性补丁
apply_qlib_patches()

# 导入 Qlib（如果可用）
if QLIB_AVAILABLE:
    import qlib
    from qlib.config import REG_CN

# 导入项目内部模块
from ...core.config import settings
from ..data.simple_data_service import SimpleDataService
from ..prediction.technical_indicators import TechnicalIndicatorCalculator

# 全局 Qlib 初始化状态
_QLIB_GLOBAL_INITIALIZED = False


class EnhancedQlibDataProvider:
    """增强版 Qlib 数据提供器

    整合以下子模块功能：
    - Alpha158Calculator: Alpha158 因子计算
    - FactorCache: 因子缓存管理
    - QlibFormatConverter: 数据格式转换
    - DataQualityValidator: 数据质量验证
    """

    def __init__(self, data_service: Optional[SimpleDataService] = None):
        self.data_service = data_service or SimpleDataService()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.alpha_calculator = Alpha158Calculator()
        self.format_converter = QlibFormatConverter()
        self.data_validator = DataQualityValidator()
        self._qlib_initialized = False

        logger.info("增强版 Qlib 数据提供器初始化完成")

    async def initialize_qlib(self):
        """初始化 Qlib 环境"""
        global _QLIB_GLOBAL_INITIALIZED

        if _QLIB_GLOBAL_INITIALIZED or not QLIB_AVAILABLE:
            self._qlib_initialized = True
            return

        try:
            qlib_data_path = Path(settings.QLIB_DATA_PATH).resolve()
            qlib_data_path.mkdir(parents=True, exist_ok=True)

            # 确保交易日历文件存在
            self._ensure_calendar_exists()

            # 配置路径
            qlib_data_path_str = qlib_data_path.as_posix()
            path_config = {"day": qlib_data_path_str, "1min": qlib_data_path_str}

            qlib.init(
                region=REG_CN,
                provider_uri=path_config,
                mount_path=path_config,
                auto_mount=False,
            )

            _QLIB_GLOBAL_INITIALIZED = True
            self._qlib_initialized = True
            logger.info("Qlib 环境初始化成功")

        except Exception as e:
            logger.error(f"Qlib 初始化失败: {e}")
            raise

    def _ensure_calendar_exists(self):
        """确保交易日历文件存在"""
        try:
            from app.services.data.qlib_calendar_generator import QlibCalendarGenerator
            calendar_generator = QlibCalendarGenerator()
            calendar_generator.ensure_calendar_exists()
        except Exception as e:
            logger.warning(f"生成交易日历文件失败: {e}")

    async def prepare_qlib_dataset(
        self,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        include_alpha_factors: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """准备 Qlib 标准格式的数据集"""
        logger.info(f"准备 Qlib 数据集: {len(stock_codes)} 只股票")

        # 1. 获取基础特征数据
        base_features = await self._prepare_base_features(stock_codes, start_date, end_date)
        if base_features.empty:
            logger.warning("基础特征数据为空")
            return pd.DataFrame()

        # 2. 转换为 Qlib 标准格式
        qlib_data = self.format_converter.convert(
            base_features,
            date_column="date",
            instrument_column="stock_code",
        )

        # 3. 计算 Alpha158 因子
        if include_alpha_factors and QLIB_AVAILABLE:
            qlib_data = await self._add_alpha_factors(
                qlib_data, stock_codes, start_date, end_date, use_cache
            )

        self._log_dataset_info(qlib_data)
        return qlib_data

    async def _prepare_base_features(
        self, stock_codes: List[str], start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """准备基础特征数据"""
        all_features = []

        for stock_code in stock_codes:
            try:
                stock_data = await self._load_stock_data(stock_code, start_date, end_date)
                if stock_data.empty:
                    continue

                # 计算技术指标
                indicators = await self.indicator_calculator.calculate_all_indicators(stock_data)
                features = stock_data.merge(
                    indicators, left_index=True, right_index=True, how="left"
                ) if not indicators.empty else stock_data.copy()

                features["stock_code"] = stock_code
                features = self._ensure_date_column(features)
                all_features.append(features)

            except Exception as e:
                logger.error(f"处理股票 {stock_code} 时出错: {e}")

        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features, ignore_index=True)
        return combined.sort_values(["stock_code", "date"])

    async def _load_stock_data(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """加载股票数据"""
        from app.services.data.stock_data_loader import StockDataLoader

        loader = StockDataLoader(data_root=settings.DATA_ROOT_PATH)
        stock_data = loader.load_stock_data(stock_code, start_date=start_date, end_date=end_date)

        if stock_data.empty:
            stock_data = await self._fetch_remote_data(stock_code, start_date, end_date)

        return stock_data

    async def _fetch_remote_data(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """从远端服务获取数据"""
        stock_data_list = await self.data_service.get_stock_data(stock_code, start_date, end_date)
        if not stock_data_list:
            return pd.DataFrame()

        df = pd.DataFrame([
            {"date": item.date, "open": item.open, "high": item.high,
             "low": item.low, "close": item.close, "volume": item.volume}
            for item in stock_data_list
        ])
        return df.set_index("date")

    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保 DataFrame 有 date 列"""
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={"index": "date"}, inplace=True)
            else:
                df = df.reset_index()
                if "date" not in df.columns:
                    df["date"] = df.index
        return df

    async def _add_alpha_factors(
        self,
        qlib_data: pd.DataFrame,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool,
    ) -> pd.DataFrame:
        """添加 Alpha158 因子"""
        try:
            alpha_factors = await self.alpha_calculator.calculate_alpha_factors(
                qlib_data, stock_codes, (start_date, end_date), use_cache
            )
            if not alpha_factors.empty:
                qlib_data = pd.concat([qlib_data, alpha_factors], axis=1)
                logger.info(f"成功添加 {len(alpha_factors.columns)} 个 Alpha 因子")
        except Exception as e:
            logger.error(f"Alpha 因子计算失败: {e}")
        return qlib_data

    def _log_dataset_info(self, df: pd.DataFrame):
        """记录数据集信息"""
        logger.info("=" * 40)
        logger.info(f"Qlib 数据集准备完成: {df.shape}")
        if isinstance(df.index, pd.MultiIndex):
            logger.info(f"MultiIndex 级别: {df.index.names}")
        logger.info(f"特征数: {len(df.columns)}")
        logger.info("=" * 40)

    async def validate_and_fix_qlib_format(
        self, data: pd.DataFrame
    ) -> Tuple[bool, pd.DataFrame]:
        """验证并修复 Qlib 数据格式"""
        if data.empty:
            return False, data

        # 使用验证器检查数据质量
        report = self.data_validator.validate(data)

        if not report.is_valid:
            logger.warning(f"数据验证发现 {len(report.issues)} 个问题")
            # 尝试修复
            data = self._fix_data_issues(data, report)

        is_valid = await self.validate_qlib_data_format(data)
        return is_valid, data

    def _fix_data_issues(self, data: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
        """修复数据问题"""
        df = data.copy()

        # 修复价格逻辑问题
        price_cols = ["$open", "$high", "$low", "$close"]
        if all(col in df.columns for col in price_cols):
            # 修复 high < low
            invalid_mask = df["$high"] < df["$low"]
            if invalid_mask.any():
                df.loc[invalid_mask, ["$high", "$low"]] = df.loc[invalid_mask, ["$low", "$high"]].values

            # 修复负价格
            for col in price_cols:
                negative_mask = df[col] <= 0
                if negative_mask.any():
                    df.loc[negative_mask, col] = df[col].ffill()

        # 修复负成交量
        if "$volume" in df.columns:
            df.loc[df["$volume"] < 0, "$volume"] = 0

        return df

    async def validate_qlib_data_format(self, data: pd.DataFrame) -> bool:
        """验证 Qlib 数据格式"""
        try:
            if not isinstance(data.index, pd.MultiIndex) or len(data.index.names) != 2:
                return False

            required_cols = ["$close", "$high", "$low", "$open", "$volume"]
            if not all(col in data.columns for col in required_cols):
                return False

            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    return False

            return True
        except Exception as e:
            logger.error(f"格式验证失败: {e}")
            return False

    async def convert_dataframe_to_qlib(
        self, df: pd.DataFrame, validate: bool = True, fix_issues: bool = True
    ) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """将 DataFrame 转换为 Qlib 格式"""
        info = {"input_shape": df.shape, "conversion_steps": []}

        converted = self.format_converter.convert(df)
        info["conversion_steps"].append("格式转换")

        if validate or fix_issues:
            is_valid, converted = await self.validate_and_fix_qlib_format(converted)
            info["is_valid"] = is_valid

        info["output_shape"] = converted.shape
        return info.get("is_valid", True), converted, info

    async def create_qlib_model_config(
        self, model_type: str, hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建 Qlib 模型配置"""
        config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "huber",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
            },
        }

        model_configs = {
            "lightgbm": ("LGBModel", "qlib.contrib.model.gbdt"),
            "xgboost": ("XGBModel", "qlib.contrib.model.xgboost"),
            "mlp": ("DNNModelPytorch", "qlib.contrib.model.pytorch_nn"),
        }

        if model_type.lower() in model_configs:
            config["class"], config["module_path"] = model_configs[model_type.lower()]

        if hyperparameters:
            config["kwargs"].update(hyperparameters)

        return config

    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            cache = self.alpha_calculator.factor_cache
            return {
                **cache.get_cache_stats(),
                "qlib_available": QLIB_AVAILABLE,
                "qlib_initialized": self._qlib_initialized,
            }
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {"error": str(e)}

    async def clear_cache(self):
        """清空缓存"""
        self.alpha_calculator.factor_cache.clear_cache()
        logger.info("缓存已清空")

    async def get_qlib_format_example(self) -> Dict[str, Any]:
        """获取 Qlib 格式示例"""
        return {
            "index": "MultiIndex (instrument, datetime)",
            "required_columns": ["$open", "$high", "$low", "$close", "$volume"],
            "optional_columns": ["$factor", "$vwap", "$change"],
            "example": {
                "index": [("000001.SZ", "2024-01-01"), ("000001.SZ", "2024-01-02")],
                "columns": ["$open", "$close", "$high", "$low", "$volume"],
            },
        }
