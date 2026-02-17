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
from .converters import QlibFormatConverter
from .data_processing import (
    DataTypeOptimizer,
    FundamentalFeatureCalculator,
    MissingValueHandler,
)
from .model import QlibModelConfigBuilder, QlibModelPredictor
from .patches import QLIB_AVAILABLE, apply_qlib_patches
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
    - DataTypeOptimizer: 数据类型优化
    - MissingValueHandler: 缺失值处理
    - FundamentalFeatureCalculator: 基本面特征计算
    - QlibModelConfigBuilder: 模型配置构建
    - QlibModelPredictor: 模型预测
    """

    def __init__(self, data_service: Optional[SimpleDataService] = None):
        """初始化增强版 Qlib 数据提供器

        Args:
            data_service: 数据服务实例，用于获取股票数据
        """
        self.data_service = data_service or SimpleDataService()
        self.indicator_calculator = TechnicalIndicatorCalculator()

        # 初始化子模块
        self.alpha_calculator = Alpha158Calculator()
        self.format_converter = QlibFormatConverter()
        self.data_validator = DataQualityValidator()
        self.data_type_optimizer = DataTypeOptimizer()
        self.missing_value_handler = MissingValueHandler()
        self.fundamental_calculator = FundamentalFeatureCalculator()
        self.model_config_builder = QlibModelConfigBuilder()

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
            error_msg = str(e)
            if "reinitialize" in error_msg.lower() or "already" in error_msg.lower():
                logger.warning(f"Qlib 已初始化，跳过: {error_msg}")
                _QLIB_GLOBAL_INITIALIZED = True
                self._qlib_initialized = True
            else:
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
        include_fundamental_features: bool = True,
        use_cache: bool = True,
        optimize_dtypes: bool = True,
    ) -> pd.DataFrame:
        """准备 Qlib 标准格式的数据集

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            include_alpha_factors: 是否包含 Alpha158 因子
            include_fundamental_features: 是否包含基本面特征
            use_cache: 是否使用缓存
            optimize_dtypes: 是否优化数据类型

        Returns:
            Qlib 标准格式的 DataFrame
        """
        logger.info(f"准备 Qlib 数据集: {len(stock_codes)} 只股票")

        # 1. 获取基础特征数据
        base_features = await self._prepare_base_features(
            stock_codes, start_date, end_date
        )
        if base_features.empty:
            logger.warning("基础特征数据为空")
            return pd.DataFrame()

        # 2. 添加基本面特征
        if include_fundamental_features:
            base_features = self.fundamental_calculator.calculate(base_features)

        # 3. 转换为 Qlib 标准格式
        qlib_data = self.format_converter.convert(
            base_features,
            date_column="date",
            instrument_column="stock_code",
        )

        # 4. 处理缺失值
        qlib_data = self.missing_value_handler.handle(qlib_data)

        # 5. 优化数据类型
        if optimize_dtypes:
            qlib_data = self.data_type_optimizer.optimize(qlib_data)

        # 6. 计算 Alpha158 因子
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
                stock_data = await self._load_stock_data(
                    stock_code, start_date, end_date
                )
                if stock_data.empty:
                    continue

                # 计算技术指标
                indicators = await self.indicator_calculator.calculate_all_indicators(
                    stock_data
                )
                features = (
                    stock_data.merge(
                        indicators, left_index=True, right_index=True, how="left"
                    )
                    if not indicators.empty
                    else stock_data.copy()
                )

                # 删除原始的 ts_code 列（避免与 stock_code 同时被映射为 instrument）
                if "ts_code" in features.columns:
                    features = features.drop(columns=["ts_code"])
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
        stock_data = loader.load_stock_data(
            stock_code, start_date=start_date, end_date=end_date
        )

        if stock_data.empty:
            stock_data = await self._fetch_remote_data(stock_code, start_date, end_date)

        return stock_data

    async def _fetch_remote_data(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """从远端服务获取数据"""
        stock_data_list = await self.data_service.get_stock_data(
            stock_code, start_date, end_date
        )
        if not stock_data_list:
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    "date": item.date,
                    "open": item.open,
                    "high": item.high,
                    "low": item.low,
                    "close": item.close,
                    "volume": item.volume,
                }
                for item in stock_data_list
            ]
        )
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
        """添加 Alpha158 因子

        Alpha158 handler 返回的 instrument 格式（如 000001_SZ）
        可能与 qlib_data 的格式（如 SH600036）不一致。
        在 concat 前统一 instrument 命名，避免行数翻倍。
        """
        try:
            alpha_factors = await self.alpha_calculator.calculate_alpha_factors(
                qlib_data, stock_codes, (start_date, end_date), use_cache
            )
            if alpha_factors.empty:
                return qlib_data

            alpha_factors = self._align_alpha_index(
                qlib_data,
                alpha_factors,
            )
            if alpha_factors.empty:
                logger.warning("Alpha 因子索引对齐后为空，跳过")
                return qlib_data

            # 确保 MultiIndex level 顺序一致，避免 concat 行数膨胀
            if (
                isinstance(qlib_data.index, pd.MultiIndex)
                and isinstance(alpha_factors.index, pd.MultiIndex)
                and qlib_data.index.names != alpha_factors.index.names
            ):
                logger.info(
                    f"对齐 index level 顺序: "
                    f"qlib={qlib_data.index.names} → "
                    f"alpha={alpha_factors.index.names}",
                )
                alpha_factors = alpha_factors.reorder_levels(
                    qlib_data.index.names,
                ).sort_index()

            before_rows = len(qlib_data)
            qlib_data = pd.concat(
                [qlib_data, alpha_factors],
                axis=1,
            )
            after_rows = len(qlib_data)

            if after_rows > before_rows * 1.1:
                logger.warning(
                    f"⚠️ Alpha concat 后行数异常增长: "
                    f"{before_rows} → {after_rows} "
                    f"(+{after_rows - before_rows})，"
                    f"执行索引去重修复",
                )
                # 安全网：去除因索引不对齐产生的重复行
                qlib_data = qlib_data[~qlib_data.index.duplicated(keep="first")]
                logger.info(
                    f"去重后行数: {len(qlib_data)}",
                )

            logger.info(
                f"成功添加 {len(alpha_factors.columns)} 个 Alpha 因子",
            )
        except Exception as e:
            logger.error(f"Alpha 因子计算失败: {e}")
        return qlib_data

    def _align_alpha_index(
        self,
        qlib_data: pd.DataFrame,
        alpha_factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """对齐 Alpha158 因子的索引到 qlib_data 的索引格式

        解决 instrument 命名不一致问题：
        - qlib_data: (SH600036, 2024-01-01)
        - alpha_factors: (600036_SH, 2024-01-01) 或 (000001_SZ, ...)

        通过将两边的 instrument 都归一化为相同格式来对齐。
        """
        if not isinstance(qlib_data.index, pd.MultiIndex):
            return alpha_factors
        if not isinstance(alpha_factors.index, pd.MultiIndex):
            return alpha_factors

        qlib_instruments = set(
            qlib_data.index.get_level_values(0).unique(),
        )
        alpha_instruments = set(
            alpha_factors.index.get_level_values(0).unique(),
        )

        # 如果已经有交集，不需要对齐
        overlap = qlib_instruments & alpha_instruments
        if len(overlap) >= min(
            len(qlib_instruments),
            len(alpha_instruments),
        ):
            return alpha_factors

        logger.info(
            f"instrument 命名不一致，尝试对齐: "
            f"qlib={list(qlib_instruments)[:3]}, "
            f"alpha={list(alpha_instruments)[:3]}",
        )

        # 构建映射: alpha_instrument → qlib_instrument
        mapping = _build_instrument_mapping(
            qlib_instruments,
            alpha_instruments,
        )
        if not mapping:
            logger.warning("无法建立 instrument 映射，跳过 Alpha 因子")
            return pd.DataFrame()

        # 重命名 alpha_factors 的 instrument level
        inst_level = alpha_factors.index.get_level_values(0)
        date_level = alpha_factors.index.get_level_values(1)
        mapped_inst = inst_level.map(lambda x: mapping.get(x, x))

        alpha_factors.index = pd.MultiIndex.from_arrays(
            [mapped_inst, date_level],
            names=alpha_factors.index.names,
        )

        # 只保留映射成功的行
        valid_mask = alpha_factors.index.get_level_values(0).isin(
            qlib_instruments,
        )
        alpha_factors = alpha_factors[valid_mask]

        logger.info(
            f"instrument 对齐完成: 映射 {len(mapping)} 只股票，" f"保留 {len(alpha_factors)} 行",
        )
        return alpha_factors

    def _log_dataset_info(self, df: pd.DataFrame):
        """记录数据集信息"""
        logger.info("=" * 40)
        logger.info(f"Qlib 数据集准备完成: {df.shape}")
        if isinstance(df.index, pd.MultiIndex):
            logger.info(f"MultiIndex 级别: {df.index.names}")
        logger.info(f"特征数: {len(df.columns)}")
        if not df.empty:
            logger.info(f"缺失值总数: {df.isnull().sum().sum()}")
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

    def _fix_data_issues(
        self, data: pd.DataFrame, report: ValidationReport
    ) -> pd.DataFrame:
        """修复数据问题"""
        df = data.copy()

        # 修复价格逻辑问题
        price_cols = ["$open", "$high", "$low", "$close"]
        if all(col in df.columns for col in price_cols):
            # 修复 high < low
            invalid_mask = df["$high"] < df["$low"]
            if invalid_mask.any():
                df.loc[invalid_mask, ["$high", "$low"]] = df.loc[
                    invalid_mask, ["$low", "$high"]
                ].values

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
        """创建 Qlib 模型配置

        Args:
            model_type: 模型类型（lightgbm, xgboost, mlp）
            hyperparameters: 自定义超参数

        Returns:
            Qlib 模型配置字典
        """
        return self.model_config_builder.build(model_type, hyperparameters)

    async def get_qlib_model_predictions(
        self,
        model_config: Dict[str, Any],
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """使用 Qlib 模型进行预测

        Args:
            model_config: 模型配置
            dataset: 数据集

        Returns:
            预测结果 DataFrame
        """
        if not QLIB_AVAILABLE:
            logger.warning("Qlib 不可用，无法进行预测")
            return pd.DataFrame()

        try:
            await self.initialize_qlib()

            predictor = QlibModelPredictor(model_config)
            return predictor.fit_predict(dataset)

        except Exception as e:
            logger.error(f"Qlib 模型预测失败: {e}")
            return pd.DataFrame()

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
            "index": "MultiIndex (datetime, instrument)",
            "required_columns": ["$open", "$high", "$low", "$close", "$volume"],
            "optional_columns": ["$factor", "$vwap", "$change"],
            "example": {
                "index": [
                    ("2024-01-01", "000001.SZ"),
                    ("2024-01-02", "000001.SZ"),
                ],
                "columns": ["$open", "$close", "$high", "$low", "$volume"],
            },
        }

    def get_supported_models(self) -> Dict[str, str]:
        """获取支持的模型类型

        Returns:
            模型类型及其描述
        """
        return self.model_config_builder.get_supported_models()


def _normalize_to_canonical(code: str) -> str:
    """将各种 instrument 格式归一化为统一的 (数字, 交易所) 元组字符串

    支持的格式：
    - SH600036 / SZ000001（QlibFormatConverter 输出）
    - 600036_SH / 000001_SZ（Qlib 文件名格式）
    - 600036.SH / 000001.SZ（Tushare 格式）

    Returns:
        归一化后的 "数字_交易所" 格式，如 "600036_SH"
    """
    code = str(code).strip().upper()

    # SH600036 / SZ000001 格式
    if len(code) >= 8 and code[:2] in ("SH", "SZ"):
        return f"{code[2:]}_{code[:2]}"

    # 600036_SH / 000001_SZ 格式
    if "_" in code:
        parts = code.split("_")
        if len(parts) == 2 and parts[1] in ("SH", "SZ", "SS"):
            exchange = "SH" if parts[1] == "SS" else parts[1]
            return f"{parts[0]}_{exchange}"

    # 600036.SH / 000001.SZ 格式
    if "." in code:
        parts = code.split(".")
        if len(parts) == 2 and parts[1] in ("SH", "SZ", "SS"):
            exchange = "SH" if parts[1] == "SS" else parts[1]
            return f"{parts[0]}_{exchange}"

    return code


def _build_instrument_mapping(
    qlib_instruments: set,
    alpha_instruments: set,
) -> Dict[str, str]:
    """构建 alpha instrument → qlib instrument 的映射

    通过将两���都归一化为 canonical 格式来匹配。

    Args:
        qlib_instruments: qlib_data 中的 instrument 集合
        alpha_instruments: alpha_factors 中的 instrument 集合

    Returns:
        映射字典 {alpha_instrument: qlib_instrument}
    """
    # 构建 canonical → qlib_instrument 的反向索引
    canonical_to_qlib: Dict[str, str] = {}
    for inst in qlib_instruments:
        canonical = _normalize_to_canonical(inst)
        canonical_to_qlib[canonical] = inst

    # 对每个 alpha instrument，找到对应的 qlib instrument
    mapping: Dict[str, str] = {}
    for alpha_inst in alpha_instruments:
        canonical = _normalize_to_canonical(alpha_inst)
        if canonical in canonical_to_qlib:
            mapping[alpha_inst] = canonical_to_qlib[canonical]

    logger.info(
        f"instrument 映射: {len(mapping)}/{len(alpha_instruments)} 匹配成功",
    )
    return mapping
