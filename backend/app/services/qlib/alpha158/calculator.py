"""
Alpha158 因子计算器

提供 Alpha158 因子的计算功能，支持：
- Qlib 内置 Alpha158 Handler
- 表达式引擎计算
- 简化版因子计算
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..cache.factor_cache import FactorCache
from .expression_parser import QlibExpressionParser
from .simplified_factors import calculate_basic_factors_for_stock

# 检测 Qlib 可用性
try:
    from qlib.contrib.data.handler import Alpha158 as Alpha158Handler
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D

    QLIB_AVAILABLE = True
    ALPHA158_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    ALPHA158_AVAILABLE = False
    Alpha158DL = None
    Alpha158Handler = None
    D = None


class Alpha158Calculator:
    """
    Alpha158 因子计算器

    使用 Qlib 内置的 Alpha158 实现计算 158 个标准因子。
    当 Qlib 不可用时，回退到简化版本。
    """

    def __init__(self):
        self.factor_cache = FactorCache()
        self.expression_parser = QlibExpressionParser()
        self.max_workers = min(mp.cpu_count(), 8)

        # 初始化 Alpha158 配置
        self.alpha_fields: List[str] = []
        self.alpha_names: List[str] = []

        if ALPHA158_AVAILABLE and Alpha158DL is not None:
            self._init_alpha158_config()
        else:
            logger.warning("Qlib 内置 Alpha158 不可用，将使用简化版本")

    def _init_alpha158_config(self):
        """初始化 Alpha158 配置"""
        default_config = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        try:
            config_result = Alpha158DL.get_feature_config(default_config)
            if isinstance(config_result, tuple):
                if len(config_result) >= 2:
                    self.alpha_fields, self.alpha_names = (
                        config_result[0],
                        config_result[1],
                    )
                else:
                    self.alpha_fields, self.alpha_names = config_result
            else:
                logger.warning(
                    f"Alpha158DL.get_feature_config 返回非预期类型: {type(config_result)}"
                )
                self.alpha_fields, self.alpha_names = [], []
            logger.info(f"Alpha158 计算器初始化，支持 {len(self.alpha_fields)} 个因子")
        except Exception as e:
            logger.warning(f"获取 Alpha158 配置失败: {e}，使用简化版本")
            self.alpha_fields, self.alpha_names = [], []

    async def calculate_alpha_factors(
        self,
        qlib_data: pd.DataFrame,
        stock_codes: List[str],
        date_range: Tuple[datetime, datetime],
        use_cache: bool = True,
        force_expression_engine: bool = False,
    ) -> pd.DataFrame:
        """
        计算 Alpha158 因子

        优先使用 Alpha158 handler，如果不可用则使用表达式引擎计算。

        Args:
            qlib_data: Qlib 格式的数据
            stock_codes: 股票代码列表
            date_range: 日期范围
            use_cache: 是否使用缓存
            force_expression_engine: 是否强制使用表达式引擎

        Returns:
            包含 Alpha158 因子的 DataFrame
        """
        if not QLIB_AVAILABLE or not ALPHA158_AVAILABLE:
            logger.warning("Qlib 或 Alpha158 不可用，跳过 Alpha 因子计算")
            return pd.DataFrame(index=qlib_data.index)

        if len(self.alpha_fields) == 0 or len(self.alpha_names) == 0:
            logger.warning("Alpha158 配置不可用，跳过 Alpha 因子计算")
            return pd.DataFrame(index=qlib_data.index)

        # 尝试从缓存获取
        if use_cache:
            cache_key = self.factor_cache.get_cache_key(stock_codes, date_range)
            cached_factors = self.factor_cache.get_cached_factors(cache_key)
            if cached_factors is not None:
                logger.info(f"从缓存加载 Alpha158 因子: {len(cached_factors.columns)} 个因子")
                return cached_factors

        try:
            logger.info(
                f"开始计算 Alpha158 因子: {len(stock_codes)} 只股票, "
                f"目标因子数: {len(self.alpha_names)}"
            )

            if qlib_data.empty:
                logger.warning("输入数据为空，无法计算因子")
                return pd.DataFrame(index=qlib_data.index)

            # 方法1：尝试使用 Alpha158 handler
            if not force_expression_engine:
                try:
                    alpha_factors = await self._calculate_using_alpha158_handler(
                        stock_codes, date_range
                    )
                    if not alpha_factors.empty and len(alpha_factors.columns) >= 158:
                        logger.info(
                            f"使用 Alpha158 handler 计算完成: {len(alpha_factors.columns)} 个因子"
                        )
                        if use_cache:
                            cache_key = self.factor_cache.get_cache_key(
                                stock_codes, date_range
                            )
                            self.factor_cache.save_factors(cache_key, alpha_factors)
                        return alpha_factors
                    else:
                        factor_count = (
                            len(alpha_factors.columns) if not alpha_factors.empty else 0
                        )
                        logger.warning(
                            f"Alpha158 handler 计算因子数不足: {factor_count}，尝试表达式引擎"
                        )
                except Exception as e:
                    logger.debug(f"使用 Alpha158 handler 失败: {e}，尝试表达式引擎")

            # 方法2：使用表达式引擎计算
            logger.info("使用表达式引擎计算 Alpha158 因子")
            alpha_factors = await self._calculate_using_expression_engine(
                qlib_data, stock_codes, date_range
            )

            if not alpha_factors.empty:
                factor_count = len(alpha_factors.columns)
                logger.info(
                    f"Alpha158 因子计算完成: {len(alpha_factors)} 条记录, {factor_count} 个因子"
                )
                if factor_count < 158:
                    logger.warning(f"表达式引擎计算的因子数不足: {factor_count} < 158")
                if use_cache:
                    cache_key = self.factor_cache.get_cache_key(stock_codes, date_range)
                    self.factor_cache.save_factors(cache_key, alpha_factors)
                return alpha_factors
            else:
                logger.warning("没有计算出任何因子")
                return pd.DataFrame(index=qlib_data.index)

        except Exception as e:
            logger.error(f"Alpha 因子计算失败: {e}", exc_info=True)
            logger.warning("回退到简化版 Alpha 因子计算")
            try:
                return await self._calculate_simplified_alpha_factors(qlib_data)
            except Exception as e2:
                logger.error(f"简化版 Alpha 因子计算也失败: {e2}")
                return pd.DataFrame(index=qlib_data.index)

    async def _calculate_using_alpha158_handler(
        self, stock_codes: List[str], date_range: Tuple[datetime, datetime]
    ) -> pd.DataFrame:
        """使用 Qlib 内置的 Alpha158 handler 计算因子"""
        try:
            if Alpha158Handler is None:
                raise ValueError("Alpha158Handler 不可用")

            from app.core.config import settings

            start_date, end_date = date_range
            qlib_data_path = Path(settings.QLIB_DATA_PATH).resolve()
            qlib_features_dir = qlib_data_path / "features" / "day"

            if not qlib_features_dir.exists():
                raise ValueError(f"Qlib 数据目录不存在: {qlib_features_dir}")

            # 获取可用文件列表
            available_files = {
                f.stem
                for f in qlib_features_dir.glob("*.parquet")
                if f.stat().st_size > 0
            }

            # 解析股票代码到文件名
            resolved_instruments = []
            instrument_map = {}

            for code in stock_codes:
                selected = self._resolve_instrument_name(code, available_files)
                if selected:
                    instrument_map[code] = selected
                    resolved_instruments.append(selected)
                else:
                    logger.warning(f"未找到 Qlib 数据文件: {code}")

            if not resolved_instruments:
                raise ValueError("没有找到任何有效的 Qlib 数据文件")

            # 创建 Alpha158 handler
            handler = Alpha158Handler(
                instruments=resolved_instruments,
                start_time=start_date.strftime("%Y-%m-%d"),
                end_time=end_date.strftime("%Y-%m-%d"),
                fit_start_time=start_date.strftime("%Y-%m-%d"),
                fit_end_time=end_date.strftime("%Y-%m-%d"),
            )

            alpha_factors = handler.fetch()

            if alpha_factors is not None and not alpha_factors.empty:
                # 映射回原始股票代码
                try:
                    reverse_map = {v: k for k, v in instrument_map.items()}
                    if isinstance(alpha_factors.index, pd.MultiIndex):
                        inst_level = alpha_factors.index.get_level_values(0)
                        mapped_inst = inst_level.map(lambda x: reverse_map.get(x, x))
                        alpha_factors.index = pd.MultiIndex.from_arrays(
                            [mapped_inst, alpha_factors.index.get_level_values(1)],
                            names=alpha_factors.index.names,
                        )
                except Exception as map_error:
                    logger.debug(f"映射 instrument 名称失败: {map_error}")

                # 过滤掉 Qlib Alpha158 handler 自动附带的 LABEL 列，
                # 防止未来收益率（如 LABEL0）作为特征泄漏到训练数据中
                label_cols = [
                    c for c in alpha_factors.columns if c.upper().startswith("LABEL")
                ]
                if label_cols:
                    logger.warning(
                        f"Alpha158 handler 返回了 {len(label_cols)} 个标签列，已过滤: {label_cols}"
                    )
                    alpha_factors = alpha_factors.drop(columns=label_cols)

                logger.info(f"Alpha158 handler 计算完成: {alpha_factors.shape}")
                return alpha_factors
            else:
                raise ValueError("Alpha158 handler 返回空数据")

        except Exception as e:
            logger.warning(f"使用 Alpha158 handler 计算失败: {e}")
            raise

    def _resolve_instrument_name(
        self, code: str, available_files: set
    ) -> Optional[str]:
        """解析股票代码到文件名"""
        raw_code = str(code).strip()
        norm_code = raw_code.upper()
        candidates = [norm_code]

        if "." in norm_code:
            candidates.append(norm_code.replace(".", "_"))
            try:
                sym, exch = norm_code.split(".")
                candidates.append(f"{exch}{sym}")
                candidates.append(f"{sym}_{exch}")
            except ValueError:
                pass

        if len(norm_code) >= 8 and norm_code[:2] in ("SZ", "SH"):
            sym = norm_code[2:]
            exch = norm_code[:2]
            candidates.append(f"{sym}_{exch}")

        if norm_code.isdigit() and len(norm_code) == 6:
            candidates.append(f"{norm_code}_SZ")
            candidates.append(f"{norm_code}_SH")

        available_map = {c.lower(): c for c in available_files}
        for cand in candidates:
            if cand.lower() in available_map:
                return available_map[cand.lower()]

        return None

    async def _calculate_using_expression_engine(
        self,
        qlib_data: pd.DataFrame,
        stock_codes: List[str],
        date_range: Tuple[datetime, datetime],
    ) -> pd.DataFrame:
        """使用表达式引擎计算 Alpha158 因子"""
        try:
            if not (
                isinstance(qlib_data.index, pd.MultiIndex)
                and qlib_data.index.nlevels == 2
            ):
                logger.warning("数据格式不支持，无法使用表达式引擎计算")
                return pd.DataFrame()

            # 按股票分割数据
            stock_groups = {}
            for stock_code in stock_codes:
                try:
                    stock_data = qlib_data.xs(stock_code, level=0, drop_level=False)
                    if not stock_data.empty:
                        stock_groups[stock_code] = stock_data
                except KeyError:
                    logger.warning(f"股票 {stock_code} 不在数据中")

            factors_list = []

            # 使用多进程并行计算
            if len(stock_groups) > 1 and self.max_workers > 1:
                logger.info(f"使用 {self.max_workers} 个进程并行计算因子")
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    for stock_code, stock_data in stock_groups.items():
                        future = executor.submit(
                            self._calculate_alpha_factors_from_expressions,
                            stock_data,
                            stock_code,
                        )
                        futures[future] = stock_code

                    for future in as_completed(futures):
                        stock_code = futures[future]
                        try:
                            stock_factors = future.result()
                            if not stock_factors.empty:
                                factors_list.append(stock_factors)
                        except Exception as e:
                            logger.error(f"计算股票 {stock_code} 的因子时发生错误: {e}")
            else:
                logger.info("使用单进程计算因子")
                for stock_code, stock_data in stock_groups.items():
                    stock_factors = self._calculate_alpha_factors_from_expressions(
                        stock_data, stock_code
                    )
                    if not stock_factors.empty:
                        factors_list.append(stock_factors)

            if factors_list:
                return pd.concat(factors_list)
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"使用表达式引擎计算失败: {e}")
            raise

    def _calculate_alpha_factors_from_expressions(
        self, stock_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """从 Alpha158 表达式计算因子"""
        try:
            required_cols = ["$close", "$high", "$low", "$volume", "$open"]
            missing_cols = [
                col for col in required_cols if col not in stock_data.columns
            ]
            if missing_cols:
                logger.warning(f"股票 {stock_code} 缺少必要列: {missing_cols}")
                return pd.DataFrame(index=stock_data.index)

            factors = pd.DataFrame(index=stock_data.index)

            if (
                len(self.alpha_fields) > 0
                and len(self.alpha_names) > 0
                and len(self.alpha_fields) == len(self.alpha_names)
            ):
                logger.info(f"股票 {stock_code} 使用表达式解析计算 {len(self.alpha_fields)} 个因子")

                success_count = 0
                fail_count = 0
                total_factors = len(self.alpha_fields)

                for idx, (field_expr, factor_name) in enumerate(
                    zip(self.alpha_fields, self.alpha_names)
                ):
                    if idx % 10 == 0:
                        logger.info(f"股票 {stock_code} 计算进度: {idx+1}/{total_factors}")

                    try:
                        factor_series = self.expression_parser.evaluate(
                            stock_data, field_expr
                        )
                        if factor_series is not None and len(factor_series) > 0:
                            valid_count = factor_series.notna().sum()
                            if valid_count > 0:
                                factors[factor_name] = factor_series.fillna(0)
                                success_count += 1
                            else:
                                factors[factor_name] = 0
                                fail_count += 1
                        else:
                            factors[factor_name] = 0
                            fail_count += 1
                    except Exception as e:
                        logger.debug(f"计算因子 {factor_name} 失败: {e}")
                        factors[factor_name] = 0
                        fail_count += 1

                logger.info(
                    f"股票 {stock_code} 表达式解析完成: 成功 {success_count}, 失败 {fail_count}"
                )
            else:
                # 回退到简化版本
                logger.warning("Alpha158 配置不可用，使用简化版本")
                factors = calculate_basic_factors_for_stock(stock_data, stock_code)

            factors = factors.bfill().fillna(0)
            return factors

        except Exception as e:
            logger.error(f"计算股票 {stock_code} 的因子失败: {e}", exc_info=True)
            return pd.DataFrame(index=stock_data.index)

    async def _calculate_simplified_alpha_factors(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """计算简化版 Alpha 因子"""
        if data.empty:
            return pd.DataFrame()

        required_cols = ["$close", "$high", "$low", "$volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"缺少必要列: {missing_cols}")
            return pd.DataFrame(index=data.index)

        factors = pd.DataFrame(index=data.index)

        try:
            for period in [5, 10, 20, 30]:
                factors[f"RESI{period}"] = data["$close"].pct_change(periods=period)
                factors[f"MA{period}"] = data["$close"].rolling(period).mean()
                factors[f"STD{period}"] = data["$close"].rolling(period).std()
                factors[f"VSTD{period}"] = data["$volume"].rolling(period).std()
                factors[f"CORR{period}"] = (
                    data["$close"].rolling(period).corr(data["$volume"])
                )
                factors[f"MAX{period}"] = data["$high"].rolling(period).max()
                factors[f"MIN{period}"] = data["$low"].rolling(period).min()
                q80 = data["$close"].rolling(period).quantile(0.8)
                q20 = data["$close"].rolling(period).quantile(0.2)
                factors[f"QTLU{period}"] = q80 / (q20 + 1e-8)

            factors = factors.replace([np.inf, -np.inf], np.nan)
            factors = factors.ffill().fillna(0)

            logger.debug(f"计算了 {len(factors.columns)} 个简化 Alpha 因子")
            return factors

        except Exception as e:
            logger.error(f"简化 Alpha 因子计算失败: {e}")
            return pd.DataFrame(index=data.index)

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.factor_cache.get_cache_stats()

    def clear_cache(self, memory_only: bool = False):
        """清除缓存"""
        self.factor_cache.clear_cache(memory_only=memory_only)
