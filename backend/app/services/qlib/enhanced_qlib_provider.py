"""
增强版Qlib数据提供器

基于现有QlibDataProvider，添加Alpha158因子计算和缓存机制
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

# 检测Qlib可用性
try:
    import qlib
    from qlib.config import REG_CN, C
    from qlib.contrib.data.handler import Alpha158 as Alpha158Handler

    # 导入Qlib内置的Alpha158
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.loader import QlibDataLoader
    from qlib.data.filter import ExpressionDFilter, NameDFilter
    from qlib.utils import init_instance_by_config

    QLIB_AVAILABLE = True
    ALPHA158_AVAILABLE = True

    # 修复 qlib 内部的 Path 对象问题
    # qlib 的 _mount_nfs_uri 函数在处理 C.dpm.get_data_uri() 返回的路径时，
    # 如果返回的是 Path 对象，会导致 ' '.join() 失败
    # 我们需要 monkey patch 来修复这个问题
    try:
        # 尝试 monkey patch _mount_nfs_uri 函数
        if hasattr(qlib, "_mount_nfs_uri"):
            _original_mount_nfs_uri = qlib._mount_nfs_uri

            def _patched_mount_nfs_uri(provider_uri, mount_path, auto_mount):
                """修复后的 _mount_nfs_uri，处理 Path 对象和路径格式问题"""

                # 清理路径的函数
                def clean_path(path_val):
                    """清理路径，移除末尾的 :\ 等异常字符"""
                    if isinstance(path_val, Path):
                        path_str = path_val.resolve().as_posix()
                    elif isinstance(path_val, str):
                        path_str = path_val
                    else:
                        return path_val
                    # 清理路径末尾的异常字符
                    path_str = (
                        path_str.rstrip(":\\").rstrip(":/").rstrip("\\").rstrip("/")
                    )
                    # 如果路径中有 :\/，替换为 /
                    path_str = path_str.replace(":\/", "/").replace(":\\/", "/")
                    return path_str

                # 确保 mount_path 是字符串，而不是 Path 对象
                if isinstance(mount_path, Path):
                    mount_path = clean_path(mount_path)
                elif isinstance(mount_path, (list, tuple)):
                    # 如果 mount_path 是列表，确保所有元素都是字符串
                    mount_path = [clean_path(item) for item in mount_path]
                elif isinstance(mount_path, dict):
                    # 如果 mount_path 是字典，确保所有值都是正确的路径格式
                    mount_path = {k: clean_path(v) for k, v in mount_path.items()}

                # 如果路径不存在，创建它（避免 FileNotFoundError）
                if isinstance(mount_path, str):
                    Path(mount_path).mkdir(parents=True, exist_ok=True)
                elif isinstance(mount_path, dict):
                    # 如果是字典，为每个路径创建目录
                    for path_val in mount_path.values():
                        if isinstance(path_val, str):
                            Path(path_val).mkdir(parents=True, exist_ok=True)

                # 调用原始函数
                result = _original_mount_nfs_uri(provider_uri, mount_path, auto_mount)

                # 在调用后，再次修复 C.dpm.data_path（Qlib 内部可能会修改它，添加 :\ 字符）
                try:
                    from qlib.config import C

                    if hasattr(C, "dpm") and hasattr(C.dpm, "data_path"):
                        data_path = C.dpm.data_path
                        if isinstance(data_path, dict):
                            fixed_data_path = {}
                            needs_fix = False
                            for freq, path_val in data_path.items():
                                path_str = str(path_val)
                                fixed_path = clean_path(path_val)
                                # 如果路径有问题，记录并修复
                                if (
                                    ":\\" in path_str
                                    or ":\/" in path_str
                                    or path_str.endswith(":\\")
                                    or path_str.endswith(":/")
                                ):
                                    needs_fix = True
                                    logger.info(
                                        f"检测到 data_path[{freq}] 路径问题: {path_str} -> {fixed_path}"
                                    )
                                fixed_data_path[freq] = fixed_path

                            # 如果检测到问题，尝试设置修复后的路径
                            if needs_fix:
                                try:
                                    C.dpm.data_path = fixed_data_path
                                    logger.info(
                                        f"已修复 C.dpm.data_path: {fixed_data_path}"
                                    )
                                except Exception as set_error:
                                    try:
                                        # 尝试通过 __dict__ 设置
                                        if hasattr(C.dpm, "__dict__"):
                                            C.dpm.__dict__[
                                                "data_path"
                                            ] = fixed_data_path
                                            logger.info(f"通过 __dict__ 修复 data_path")
                                    except Exception:
                                        logger.warning(f"无法修复 data_path: {set_error}")
                except Exception as fix_error:
                    logger.debug(f"修复 data_path 时出错: {fix_error}")

                return result

            qlib._mount_nfs_uri = _patched_mount_nfs_uri

        # 尝试 patch CalendarProvider 的日历查找逻辑，修复路径拼接问题
        try:
            from qlib.data import CalendarProvider

            if hasattr(CalendarProvider, "load_calendar"):
                _original_load_calendar = CalendarProvider.load_calendar

                def _patched_load_calendar(self, freq, future=False):
                    """修复后的 load_calendar，修复路径拼接问题（:\/ 字符）"""
                    try:
                        # 调用原始方法
                        return _original_load_calendar(self, freq, future)
                    except Exception as e:
                        error_msg = str(e)
                        # 如果错误信息中包含路径问题（:\/ 或 :\\/），尝试修复
                        if "calendar not exists" in error_msg.lower() and (
                            ":\\/" in error_msg
                            or ":\/" in error_msg
                            or ":\\\\/" in error_msg
                        ):
                            # 提取错误的路径
                            import re

                            match = re.search(
                                r"calendar not exists:\s*(.+)", error_msg, re.IGNORECASE
                            )
                            if match:
                                wrong_path = match.group(1).strip()
                                # 修复路径：将 :\/ 或 :\\/ 替换为 /
                                # 处理多种可能的转义形式
                                fixed_path = (
                                    wrong_path.replace(":\\\\/", "/")
                                    .replace(":\\/", "/")
                                    .replace(":\/", "/")
                                )
                                fixed_path_obj = Path(fixed_path)

                                # 如果修复后的路径存在，直接读取并返回
                                if fixed_path_obj.exists():
                                    logger.info(
                                        f"检测到日历路径问题，已修复: {wrong_path} -> {fixed_path}"
                                    )
                                    try:
                                        with open(fixed_path_obj, "r") as f:
                                            dates = [
                                                line.strip()
                                                for line in f
                                                if line.strip()
                                            ]
                                        # 返回日期列表（转换为 Qlib 期望的格式）
                                        import pandas as pd

                                        calendar_dates = pd.to_datetime(
                                            dates, format="%Y%m%d"
                                        )
                                        logger.info(
                                            f"成功从修复后的路径读取日历，包含 {len(calendar_dates)} 个交易日"
                                        )
                                        return calendar_dates
                                    except Exception as read_error:
                                        logger.warning(f"读取日历文件失败: {read_error}")
                                        raise
                                else:
                                    logger.warning(f"修复后的路径也不存在: {fixed_path}")
                                    # 尝试使用配置中的路径
                                    try:
                                        from app.core.config import settings

                                        config_path = (
                                            Path(settings.QLIB_DATA_PATH).resolve()
                                            / "calendars"
                                            / "day.txt"
                                        )
                                        if config_path.exists():
                                            logger.info(f"使用配置路径读取日历: {config_path}")
                                            with open(config_path, "r") as f:
                                                dates = [
                                                    line.strip()
                                                    for line in f
                                                    if line.strip()
                                                ]
                                            import pandas as pd

                                            calendar_dates = pd.to_datetime(
                                                dates, format="%Y%m%d"
                                            )
                                            logger.info(
                                                f"成功从配置路径读取日历，包含 {len(calendar_dates)} 个交易日"
                                            )
                                            return calendar_dates
                                    except Exception as config_error:
                                        logger.warning(f"使用配置路径也失败: {config_error}")
                        # 如果无法修复，抛出原始错误
                        raise

                CalendarProvider.load_calendar = _patched_load_calendar
                logger.debug("已 patch CalendarProvider.load_calendar 方法以修复路径问题")
        except Exception as cal_patch_error:
            logger.debug(f"无法 patch CalendarProvider: {cal_patch_error}")
    except Exception:
        pass  # 如果无法 patch，继续执行

except ImportError as e:
    error_msg = str(e)
    missing_module = None

    # 检测缺失的模块
    if "setuptools_scm" in error_msg:
        missing_module = "setuptools_scm"
    elif "ruamel" in error_msg or "ruamel.yaml" in error_msg:
        missing_module = "ruamel.yaml"
    elif "cvxpy" in error_msg:
        missing_module = "cvxpy"

    if missing_module:
        logger.warning(
            f"Qlib缺少依赖 {missing_module}: {e}\n"
            f"解决方法: pip install {missing_module}\n"
            f"或运行修复脚本: ./fix_qlib_dependencies.sh"
        )
    else:
        logger.warning(f"Qlib未安装或Alpha158不可用: {e}")
    QLIB_AVAILABLE = False
    ALPHA158_AVAILABLE = False
    Alpha158DL = None
    Alpha158Handler = None
    QlibDataLoader = None

from ...core.config import settings
from ..data.simple_data_service import SimpleDataService
from ..prediction.technical_indicators import TechnicalIndicatorCalculator

# 全局Qlib初始化状态（跨实例共享）
_QLIB_GLOBAL_INITIALIZED = False


class FactorCache:
    """因子计算结果缓存 - 优化版"""

    def __init__(self, cache_dir: str = "./data/qlib_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存配置
        self.max_cache_size = 50  # 最大缓存文件数
        self.default_ttl = timedelta(hours=24)  # 默认缓存过期时间

        # 内存缓存层
        self.memory_cache = {}
        self.max_memory_cache_size = 10  # 最大内存缓存项数
        self.memory_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

        logger.info(f"因子缓存初始化: {self.cache_dir}, 内存缓存大小: {self.max_memory_cache_size}")

    def get_cache_key(
        self, stock_codes: List[str], date_range: Tuple[datetime, datetime]
    ) -> str:
        """生成缓存键 - 优化版"""
        # 对股票代码排序，确保相同股票集合生成相同的缓存键
        sorted_codes = sorted(stock_codes)
        codes_str = "_".join(sorted_codes)
        # 使用更高效的哈希算法
        codes_hash = hashlib.sha1(codes_str.encode()).hexdigest()[:12]
        start_str = date_range[0].strftime("%Y%m%d")
        end_str = date_range[1].strftime("%Y%m%d")
        return f"alpha_{codes_hash}_{start_str}_{end_str}"

    def get_cached_factors(self, cache_key: str) -> Optional[pd.DataFrame]:
        """获取缓存的因子数据 - 优先从内存缓存获取"""
        # 1. 先从内存缓存获取
        if cache_key in self.memory_cache:
            cache_item = self.memory_cache[cache_key]
            factors = cache_item["data"]
            timestamp = cache_item["timestamp"]

            # 检查内存缓存是否过期
            if datetime.now() - timestamp < self.default_ttl:
                self.memory_cache_stats["hits"] += 1
                logger.debug(f"内存缓存命中: {cache_key}, 数据量: {len(factors)}")
                return factors
            else:
                # 内存缓存过期，删除
                del self.memory_cache[cache_key]
                self.memory_cache_stats["misses"] += 1
                logger.debug(f"内存缓存过期: {cache_key}")
        else:
            self.memory_cache_stats["misses"] += 1

        # 2. 从磁盘缓存获取
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            try:
                # 检查文件是否过期
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - file_time > self.default_ttl:
                    logger.debug(f"磁盘缓存已过期: {cache_key}")
                    cache_file.unlink()
                    return None

                factors = pd.read_parquet(cache_file)
                logger.info(f"磁盘缓存命中: {cache_key}, 数据量: {len(factors)}")

                # 将数据加载到内存缓存
                self._add_to_memory_cache(cache_key, factors)

                return factors
            except Exception as e:
                logger.warning(f"读取磁盘缓存失败: {e}")
                # 删除损坏的缓存文件
                try:
                    cache_file.unlink()
                except:
                    pass
        return None

    def save_factors(self, cache_key: str, factors: pd.DataFrame):
        """保存因子数据到缓存 - 同时保存到内存和磁盘"""
        try:
            # 1. 保存到内存缓存
            self._add_to_memory_cache(cache_key, factors)

            # 2. 保存到磁盘缓存
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            # 优化：使用更快的压缩方式
            factors.to_parquet(cache_file, compression="snappy")

            # 清理旧缓存
            self._cleanup_old_cache()

            logger.info(f"因子数据缓存成功: {cache_key}, 数据量: {len(factors)}")
        except Exception as e:
            logger.warning(f"保存因子缓存失败: {e}")

    def _add_to_memory_cache(self, cache_key: str, factors: pd.DataFrame):
        """添加数据到内存缓存"""
        # 检查内存缓存大小
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
            self.memory_cache_stats["evictions"] += 1
            logger.debug(f"内存缓存淘汰: {oldest_key}")

        # 添加到内存缓存
        self.memory_cache[cache_key] = {"data": factors, "timestamp": datetime.now()}

    def _cleanup_old_cache(self):
        """清理旧缓存文件"""
        try:
            cache_files = list(self.cache_dir.glob("*.parquet"))
            if len(cache_files) <= self.max_cache_size:
                return

            # 按修改时间排序，删除最旧的文件
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            files_to_remove = len(cache_files) - self.max_cache_size

            for i in range(files_to_remove):
                cache_files[i].unlink()
                logger.debug(f"删除旧缓存文件: {cache_files[i].name}")

        except Exception as e:
            logger.warning(f"清理缓存失败: {e}")

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        # 计算磁盘缓存文件数
        try:
            disk_cache_count = len(list(self.cache_dir.glob("*.parquet")))
        except:
            disk_cache_count = 0

        return {
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": disk_cache_count,
            "memory_cache_hits": self.memory_cache_stats["hits"],
            "memory_cache_misses": self.memory_cache_stats["misses"],
            "memory_cache_evictions": self.memory_cache_stats["evictions"],
            "max_memory_cache_size": self.max_memory_cache_size,
            "max_disk_cache_size": self.max_cache_size,
        }

    def clear_cache(self, memory_only: bool = False):
        """清除缓存"""
        # 清除内存缓存
        self.memory_cache.clear()
        self.memory_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        logger.info("内存缓存已清除")

        # 清除磁盘缓存
        if not memory_only:
            try:
                for cache_file in self.cache_dir.glob("*.parquet"):
                    cache_file.unlink()
                logger.info("磁盘缓存已清除")
            except Exception as e:
                logger.warning(f"清除磁盘缓存失败: {e}")


import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


class Alpha158Calculator:
    """Alpha158因子计算器 - 使用Qlib内置的Alpha158实现"""

    def __init__(self):
        self.factor_cache = FactorCache()
        self.max_workers = min(mp.cpu_count(), 8)  # 最多使用8个进程

        # 使用Qlib内置的Alpha158配置
        if ALPHA158_AVAILABLE and Alpha158DL is not None:
            # 获取标准Alpha158因子配置
            default_config = {
                "kbar": {},
                "price": {
                    "windows": [0],
                    "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
                },
                "rolling": {},
            }
            try:
                # 处理Qlib不同版本返回值差异
                config_result = Alpha158DL.get_feature_config(default_config)
                if isinstance(config_result, tuple):
                    if len(config_result) >= 2:
                        self.alpha_fields, self.alpha_names = (
                            config_result[0],
                            config_result[1],
                        )
                    else:
                        # 兼容旧版本，假设返回的是(alpha_fields, alpha_names)
                        self.alpha_fields, self.alpha_names = config_result
                else:
                    # 处理非元组返回值
                    logger.warning(
                        f"Alpha158DL.get_feature_config返回非预期类型: {type(config_result)}"
                    )
                    self.alpha_fields, self.alpha_names = [], []
                logger.info(
                    f"Alpha158计算器初始化，使用Qlib内置Alpha158，支持 {len(self.alpha_fields)} 个因子"
                )
            except Exception as e:
                logger.warning(f"获取Alpha158配置失败: {e}，使用简化版本")
                self.alpha_fields, self.alpha_names = [], []
        else:
            # 回退到简化版本
            self.alpha_fields = []
            self.alpha_names = []
            logger.warning("Qlib内置Alpha158不可用，将使用简化版本")
            logger.info(f"Alpha158计算器初始化，支持 0 个因子（需要Qlib支持）")

    def _calculate_factors_for_stock(
        self, stock_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """为单个股票计算Alpha158因子"""
        try:
            # 确保数据有正确的列
            required_cols = ["$close", "$high", "$low", "$volume", "$open"]
            missing_cols = [
                col for col in required_cols if col not in stock_data.columns
            ]
            if missing_cols:
                logger.warning(f"股票 {stock_code} 缺少必要列: {missing_cols}，无法计算Alpha158因子")
                return pd.DataFrame(index=stock_data.index)

            factors = pd.DataFrame(index=stock_data.index)

            # 为了兼容性，使用pandas实现表达式计算
            # 这里实现一个简化版本的因子计算
            # 实际项目中可以根据需要扩展

            # 计算一些基本因子
            close = stock_data["$close"]
            high = stock_data["$high"]
            low = stock_data["$low"]
            volume = stock_data["$volume"]
            open_ = stock_data["$open"]

            # 计算常用因子
            factors["RET1"] = close.pct_change(1)
            factors["RET5"] = close.pct_change(5)
            factors["RET20"] = close.pct_change(20)
            factors["VOL1"] = volume.pct_change(1)
            factors["VOL5"] = volume.pct_change(5)
            factors["VOL20"] = volume.pct_change(20)
            factors["HL"] = (high - low) / close
            factors["OC"] = (open_ - close) / close
            factors["HLC"] = (high - low) / close

            # 计算移动平均线
            factors["MA5"] = close.rolling(5).mean()
            factors["MA20"] = close.rolling(20).mean()
            factors["MA60"] = close.rolling(60).mean()

            # 计算标准差
            factors["STD5"] = close.rolling(5).std()
            factors["STD20"] = close.rolling(20).std()

            # 计算动量指标
            factors["RSI14"] = self._calculate_rsi(close, 14)

            # 填充缺失值
            factors = factors.fillna(0)

            return factors

        except Exception as e:
            logger.error(f"计算股票 {stock_code} 的因子失败: {e}")
            return pd.DataFrame(index=stock_data.index)

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """计算RSI指标"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    async def calculate_alpha_factors(
        self,
        qlib_data: pd.DataFrame,
        stock_codes: List[str],
        date_range: Tuple[datetime, datetime],
        use_cache: bool = True,
        force_expression_engine: bool = False,
    ) -> pd.DataFrame:
        """
        计算Alpha158因子 - 使用Qlib内置的Alpha158 handler计算158个标准因子

        优先使用Alpha158 handler，如果不可用则使用表达式引擎计算

        Args:
            qlib_data: Qlib格式的数据
            stock_codes: 股票代码列表
            date_range: 日期范围
            use_cache: 是否使用缓存
            force_expression_engine: 是否强制使用表达式引擎（预计算时使用，因为文件还未保存）
        """
        if not QLIB_AVAILABLE or not ALPHA158_AVAILABLE:
            logger.warning("Qlib或Alpha158不可用，跳过Alpha因子计算")
            return pd.DataFrame(index=qlib_data.index)

        # 检查是否有Alpha158配置
        if len(self.alpha_fields) == 0 or len(self.alpha_names) == 0:
            logger.warning("Alpha158配置不可用，跳过Alpha因子计算")
            return pd.DataFrame(index=qlib_data.index)

        # 尝试从缓存获取
        if use_cache:
            cache_key = self.factor_cache.get_cache_key(stock_codes, date_range)
            cached_factors = self.factor_cache.get_cached_factors(cache_key)
            if cached_factors is not None:
                logger.info(f"从缓存加载Alpha158因子: {len(cached_factors.columns)} 个因子")
                return cached_factors

        try:
            logger.info(
                f"开始计算Alpha158因子: {len(stock_codes)} 只股票, 目标因子数: {len(self.alpha_names)}, 强制表达式引擎: {force_expression_engine}"
            )

            # 检查数据结构
            if qlib_data.empty:
                logger.warning("输入数据为空，无法计算因子")
                return pd.DataFrame(index=qlib_data.index)

            # 方法1：尝试使用Alpha158 handler（需要数据在Qlib系统中）
            # 注意：在预计算时，文件可能已保存，优先使用handler确保所有因子都能计算
            if not force_expression_engine:
                try:
                    alpha_factors = await self._calculate_using_alpha158_handler(
                        stock_codes, date_range
                    )
                    # 要求所有158个因子都能计算出来（用户明确要求）
                    if not alpha_factors.empty and len(alpha_factors.columns) >= 158:
                        logger.info(
                            f"使用Alpha158 handler计算完成: {len(alpha_factors.columns)} 个因子"
                        )
                        # 缓存结果
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
                            f"使用Alpha158 handler计算因子数不足: {factor_count}，期望158个，尝试使用表达式引擎"
                        )
                        # Handler失败，继续尝试表达式引擎
                except Exception as e:
                    logger.debug(f"使用Alpha158 handler失败: {e}，尝试使用表达式引擎")
            else:
                logger.debug("强制使用表达式引擎（预计算模式，文件未保存）")

            # 方法2：使用表达式引擎计算（基于Alpha158因子定义）
            logger.info("使用表达式引擎计算Alpha158因子")
            alpha_factors = await self._calculate_using_expression_engine(
                qlib_data, stock_codes, date_range
            )

            if not alpha_factors.empty:
                factor_count = len(alpha_factors.columns)
                logger.info(
                    f"Alpha158因子计算完成: {len(alpha_factors)} 条记录, {factor_count} 个因子"
                )
                # 检查因子数量
                if factor_count < 158:
                    logger.warning(
                        f"表达式引擎计算的因子数不足: {factor_count} < 158，缺失 {158 - factor_count} 个因子"
                    )
                # 缓存结果
                if use_cache:
                    cache_key = self.factor_cache.get_cache_key(stock_codes, date_range)
                    self.factor_cache.save_factors(cache_key, alpha_factors)
                return alpha_factors
            else:
                logger.warning("没有计算出任何因子")
                return pd.DataFrame(index=qlib_data.index)

        except Exception as e:
            logger.error(f"Alpha因子计算失败: {e}", exc_info=True)
            # 如果计算失败，尝试回退到简化版本
            logger.warning("回退到简化版Alpha因子计算")
            try:
                return await self._calculate_simplified_alpha_factors(qlib_data)
            except Exception as e2:
                logger.error(f"简化版Alpha因子计算也失败: {e2}")
                return pd.DataFrame(index=qlib_data.index)

    async def _calculate_using_alpha158_handler(
        self, stock_codes: List[str], date_range: Tuple[datetime, datetime]
    ) -> pd.DataFrame:
        """
        使用Qlib内置的Alpha158 handler计算158个标准因子

        注意：这需要数据已经在Qlib系统中
        """
        try:
            if Alpha158Handler is None:
                raise ValueError("Alpha158Handler不可用")

            # 在创建 handler 之前，确保日历文件路径正确
            # Qlib 内部可能会在路径拼接时出现问题，需要修复
            try:
                from pathlib import Path

                from qlib.config import C

                # 获取当前的 mount_path
                if hasattr(C, "dpm") and hasattr(C.dpm, "get_mount_path"):
                    mount_path = C.dpm.get_mount_path("day")
                    if mount_path:
                        calendar_file = Path(mount_path) / "calendars" / "day.txt"
                        # 如果日历文件存在但路径格式有问题，尝试修复
                        if not calendar_file.exists():
                            # 尝试使用绝对路径
                            calendar_file = (
                                Path(settings.QLIB_DATA_PATH).resolve()
                                / "calendars"
                                / "day.txt"
                            )
                            if calendar_file.exists():
                                # 如果文件存在但 Qlib 找不到，可能是路径格式问题
                                # 尝试通过符号链接或其他方式修复
                                logger.debug(f"日历文件存在但 Qlib 找不到，路径: {calendar_file}")
            except Exception as cal_check_error:
                logger.debug(f"检查日历文件路径时出错: {cal_check_error}")

            start_date, end_date = date_range

            # 将股票代码转换为Qlib数据文件名，并过滤不存在的标的
            logger.info(f"[Alpha158] 开始查找Qlib数据文件，股票代码: {stock_codes}")

            # 解析路径
            qlib_data_path_raw = Path(settings.QLIB_DATA_PATH)
            logger.debug(
                f"[Alpha158] 原始QLIB_DATA_PATH: {settings.QLIB_DATA_PATH}, 类型: {type(settings.QLIB_DATA_PATH)}"
            )
            logger.debug(
                f"[Alpha158] 解析后的qlib_data_path: {qlib_data_path_raw}, 绝对路径: {qlib_data_path_raw.resolve()}"
            )

            qlib_features_dir = qlib_data_path_raw.resolve() / "features" / "day"
            qlib_bin_dir = qlib_data_path_raw.resolve() / "features"
            logger.info(f"[Alpha158] Qlib数据目录路径: {qlib_features_dir}")
            logger.info(f"[Alpha158] 路径是绝对路径: {qlib_features_dir.is_absolute()}")

            # 检查目录是否存在
            dir_exists = qlib_features_dir.exists()
            dir_is_dir = qlib_features_dir.is_dir() if dir_exists else False
            logger.info(f"[Alpha158] 目录存在: {dir_exists}, 是目录: {dir_is_dir}")

            if not dir_exists:
                logger.error(f"[Alpha158] Qlib数据目录不存在: {qlib_features_dir}")
                raise ValueError(f"Qlib数据目录不存在: {qlib_features_dir}")

            # 预先获取所有可用的文件列表，避免重复检查
            # 注意：在多进程环境中，可能需要重新读取文件列表
            try:
                # 确保路径是绝对路径
                qlib_features_dir_abs = qlib_features_dir.resolve()
                qlib_bin_dir_abs = qlib_bin_dir.resolve()
                logger.debug(f"[Alpha158] 绝对路径: {qlib_features_dir_abs}")

                # 使用 glob 获取文件列表
                logger.debug(f"[Alpha158] 开始使用glob查找*.parquet文件...")
                glob_pattern = qlib_features_dir_abs / "*.parquet"
                logger.debug(f"[Alpha158] glob模式: {glob_pattern}")

                all_glob_files = list(qlib_features_dir_abs.glob("*.parquet"))
                logger.info(f"[Alpha158] glob找到 {len(all_glob_files)} 个parquet文件")

                # 验证文件确实存在且可读
                parquet_files = []
                sh_count = 0
                sz_count = 0
                for f in all_glob_files:
                    try:
                        exists = f.exists()
                        size = f.stat().st_size if exists else 0
                        # 验证文件存在且大小大于0（确保文件已完全写入）
                        if exists and size > 0:
                            parquet_files.append(f)
                            # 统计SH和SZ文件数量
                            if "_SH" in f.stem:
                                sh_count += 1
                            elif "_SZ" in f.stem:
                                sz_count += 1
                        else:
                            logger.debug(
                                f"[Alpha158] 跳过无效文件: {f.name} (exists={exists}, size={size})"
                            )
                    except (OSError, FileNotFoundError) as e:
                        logger.debug(f"[Alpha158] 无法访问文件 {f.name}: {e}")
                        # 忽略无法访问的文件
                        pass

                available_files = {f.stem for f in parquet_files}

                # 兼容Qlib bin格式：features/day/<instrument>/*.day.bin
                try:
                    qlib_bin_dir_abs = qlib_bin_dir.resolve()
                    bin_dirs = [p for p in qlib_bin_dir_abs.iterdir() if p.is_dir()]
                    for d in bin_dirs:
                        if list(d.glob("*.day.bin")):
                            available_files.add(d.name)
                except Exception:
                    pass

                logger.info(
                    f"[Alpha158] 有效文件数: {len(available_files)}, SH文件: {sh_count}, SZ文件: {sz_count}, 示例文件: {list(available_files)[:5]}"
                )

                # 检查特定股票代码的文件
                if stock_codes:
                    test_code = stock_codes[0]
                    test_candidates = [
                        test_code.upper().replace(".", "_"),
                        test_code.upper(),
                    ]
                    if "." in test_code.upper():
                        try:
                            sym, exch = test_code.upper().split(".")
                            test_candidates.extend([f"{sym}_{exch}", f"{exch}{sym}"])
                        except ValueError:
                            pass

                    available_map = {c.lower(): c for c in available_files}
                    found_in_set = [
                        c for c in test_candidates if c.lower() in available_map
                    ]
                    logger.info(
                        f"[Alpha158] 测试代码 {test_code}: 候选={test_candidates}, 在集合中={found_in_set}"
                    )

            except Exception as e:
                logger.error(
                    f"[Alpha158] 无法读取Qlib数据目录: {e}, 路径: {qlib_features_dir}",
                    exc_info=True,
                )
                available_files = set()

                # 调试：检查特定文件是否存在
                if stock_codes:
                    test_code = stock_codes[0]
                    test_candidates = [
                        test_code.upper().replace(".", "_"),
                        test_code.upper(),
                    ]
                    if "." in test_code.upper():
                        try:
                            sym, exch = test_code.upper().split(".")
                            test_candidates.extend([f"{sym}_{exch}", f"{exch}{sym}"])
                        except ValueError:
                            pass

                    available_map = {c.lower(): c for c in available_files}
                    found_in_set = [
                        c for c in test_candidates if c.lower() in available_map
                    ]
                    logger.debug(
                        f"测试代码 {test_code}: 候选={test_candidates}, 在集合中={found_in_set}, 集合示例={list(available_files)[:5]}"
                    )
            except Exception as e:
                logger.warning(f"无法读取Qlib数据目录: {e}, 路径: {qlib_features_dir}")
                available_files = set()

            qlib_bin_dir_abs = qlib_bin_dir.resolve()

            available_map = {c.lower(): c for c in available_files}

            instrument_map = {}  # 原始代码 -> 文件名
            resolved_instruments = []  # 传递给handler的instrument名称（使用标准格式）

            for code in stock_codes:
                logger.info(f"[Alpha158] 处理股票代码: {code}")
                raw_code = str(code).strip()
                norm_code = raw_code.upper()
                candidates = [norm_code]

                # 生成所有可能的文件名格式
                if "." in norm_code:
                    candidates.append(norm_code.replace(".", "_"))
                    try:
                        sym, exch = norm_code.split(".")
                        candidates.append(f"{exch}{sym}")  # 000001.SZ -> SZ000001
                        candidates.append(f"{sym}_{exch}")  # 000001.SZ -> 000001_SZ
                    except ValueError as e:
                        logger.debug(f"[Alpha158] 分割股票代码失败: {norm_code}, 错误: {e}")
                        pass

                if len(norm_code) >= 8 and norm_code[:2] in ("SZ", "SH"):
                    sym = norm_code[2:]
                    exch = norm_code[:2]
                    candidates.append(f"{sym}_{exch}")  # SZ000001 -> 000001_SZ

                if norm_code.isdigit() and len(norm_code) == 6:
                    # 纯数字代码，尝试沪深两种后缀
                    candidates.append(f"{norm_code}_SZ")
                    candidates.append(f"{norm_code}_SH")

                if "_" in norm_code:
                    parts = norm_code.split("_")
                    if len(parts) == 2:
                        candidates.append(
                            f"{parts[1]}{parts[0]}"
                        )  # SZ_000001 -> SZ000001

                # 去重候选列表
                candidates = list(dict.fromkeys(candidates))  # 保持顺序的去重
                logger.info(f"[Alpha158] 股票代码 {raw_code} 的候选文件名: {candidates}")

                # 选择第一个存在的数据文件
                # 首先从预先获取的文件列表中查找
                selected = None
                matching_candidates = []
                logger.debug(
                    f"[Alpha158] 在available_files集合中查找 (集合大小: {len(available_files)})"
                )
                for cand in candidates:
                    in_set = cand.lower() in available_map
                    logger.debug(f"[Alpha158]   候选 {cand}: 在集合中={in_set}")
                    if in_set:
                        matching_candidates.append(cand)
                        if selected is None:
                            selected = available_map.get(cand.lower(), cand)
                            logger.info(f"[Alpha158] 在集合中找到: {raw_code} -> {selected}")

                # 如果预先获取的列表中没有找到，直接检查文件系统（处理时序问题）
                if selected is None:
                    logger.warning(f"[Alpha158] 在集合中未找到，开始直接文件系统检查...")
                    # 确保使用绝对路径
                    qlib_features_dir_abs = qlib_features_dir.resolve()
                    logger.debug(f"[Alpha158] 使用绝对路径进行文件检查: {qlib_features_dir_abs}")

                    # 尝试多次检查（处理文件系统缓存和写入延迟）
                    import os
                    import time

                    max_retries = 5  # 增加重试次数
                    retry_delay = 0.2  # 200ms

                    for attempt in range(max_retries):
                        logger.debug(f"[Alpha158] 文件系统检查尝试 {attempt + 1}/{max_retries}")
                        for cand in candidates:
                            file_path = qlib_features_dir_abs / f"{cand}.parquet"
                            os_path = str(file_path)
                            bin_dir = qlib_bin_dir_abs / cand.lower()
                            bin_exists = bin_dir.is_dir() and any(
                                bin_dir.glob("*.day.bin")
                            )
                            logger.debug(f"[Alpha158]   检查文件: {file_path}")

                            # 方法1: Path.exists()
                            path_exists = file_path.exists()

                            # 方法2: os.path.exists()
                            os_exists = os.path.exists(os_path)

                            # 方法3: os.path.isfile()
                            os_isfile = os.path.isfile(os_path) if os_exists else False

                            logger.debug(
                                f"[Alpha158]     Path.exists(): {path_exists}, os.path.exists(): {os_exists}, os.path.isfile(): {os_isfile}"
                            )

                            # 如果任一方法返回True，尝试获取文件大小
                            if path_exists or os_exists:
                                try:
                                    # 尝试获取文件大小
                                    if path_exists:
                                        stat_result = file_path.stat()
                                        size = stat_result.st_size
                                    else:
                                        size = os.path.getsize(os_path)

                                    logger.debug(f"[Alpha158]     文件大小: {size} 字节")

                                    if size > 0:
                                        selected = cand
                                        # 更新 available_files 集合，避免重复检查
                                        available_files.add(cand)
                                        logger.info(
                                            f"[Alpha158] ✓ 通过直接文件检查找到: {raw_code} -> {cand}.parquet (路径: {file_path}, 大小: {size})"
                                        )
                                        break
                                    else:
                                        logger.warning(f"[Alpha158]     文件大小为0，等待重试...")
                                except (OSError, FileNotFoundError) as e:
                                    logger.debug(f"[Alpha158]     无法获取文件状态: {e}")
                                    # 即使获取状态失败，如果文件存在，也尝试使用
                                    if os_isfile:
                                        logger.warning(
                                            f"[Alpha158]     文件存在但无法获取状态，尝试使用: {cand}"
                                        )
                                        selected = cand
                                        available_files.add(cand)
                                        break
                            elif bin_exists:
                                selected = cand.lower()
                                available_files.add(selected)
                                logger.info(
                                    f"[Alpha158] ✓ 通过bin目录检查找到: {raw_code} -> {selected} (路径: {bin_dir})"
                                )
                                break

                        if selected is not None:
                            break

                        # 如果还没找到，等待一小段时间后重试（处理文件写入延迟）
                        if attempt < max_retries - 1:
                            logger.debug(f"[Alpha158] 等待 {retry_delay} 秒后重试...")
                            time.sleep(retry_delay)
                            retry_delay *= 1.5  # 线性增长（而不是指数）

                if selected is not None:
                    instrument_map[code] = selected  # 保存文件名映射
                    # Handler使用与数据文件/目录一致的instrument名称
                    handler_instrument = selected
                    resolved_instruments.append(handler_instrument)
                    logger.info(
                        f"[Alpha158] ✓ 找到Qlib数据文件: {raw_code} -> {selected}, handler使用: {handler_instrument}"
                    )
                else:
                    logger.error(f"[Alpha158] ✗ 未找到Qlib数据文件: {raw_code}")
                    # 更详细的调试信息
                    sample_matching = (
                        [
                            f
                            for f in list(available_files)[:10]
                            if raw_code.split(".")[0] in f
                        ]
                        if available_files
                        else []
                    )
                    # 检查文件是否真的不存在（使用绝对路径）
                    qlib_features_dir_abs = qlib_features_dir.resolve()
                    direct_check_results = {}

                    logger.error(f"[Alpha158] 详细诊断信息:")
                    logger.error(f"[Alpha158]   候选文件名: {candidates}")
                    logger.error(f"[Alpha158]   可用文件数: {len(available_files)}")
                    logger.error(f"[Alpha158]   匹配的候选: {matching_candidates}")
                    logger.error(f"[Alpha158]   目录路径: {qlib_features_dir_abs}")
                    logger.error(f"[Alpha158]   目录存在: {qlib_features_dir_abs.exists()}")
                    logger.error(
                        f"[Alpha158]   目录是目录: {qlib_features_dir_abs.is_dir()}"
                    )

                    # 使用多种方法检查文件
                    import os

                    for cand in candidates[:3]:  # 只检查前3个候选，避免太多IO
                        file_path = qlib_features_dir_abs / f"{cand}.parquet"
                        os_path = str(file_path)

                        # 方法1: Path.exists()
                        path_exists = file_path.exists()

                        # 方法2: os.path.exists()
                        os_exists = os.path.exists(os_path)

                        # 方法3: os.path.isfile()
                        os_isfile = os.path.isfile(os_path) if os_exists else False

                        # 方法4: 尝试打开文件
                        can_open = False
                        try:
                            with open(os_path, "rb") as f:
                                can_open = True
                                file_size = os.path.getsize(os_path)
                        except (OSError, FileNotFoundError) as e:
                            file_size = -1
                            open_error = str(e)

                        direct_check_results[cand] = {
                            "path_exists": path_exists,
                            "os_exists": os_exists,
                            "os_isfile": os_isfile,
                            "can_open": can_open,
                            "file_size": file_size if can_open else None,
                            "error": open_error if not can_open else None,
                        }

                        logger.error(f"[Alpha158]   文件 {cand}.parquet:")
                        logger.error(f"[Alpha158]     Path.exists(): {path_exists}")
                        logger.error(f"[Alpha158]     os.path.exists(): {os_exists}")
                        logger.error(f"[Alpha158]     os.path.isfile(): {os_isfile}")
                        logger.error(f"[Alpha158]     可以打开: {can_open}")
                        if can_open:
                            logger.error(f"[Alpha158]     文件大小: {file_size} 字节")
                        else:
                            logger.error(
                                f"[Alpha158]     错误: {open_error if 'open_error' in locals() else 'N/A'}"
                            )

                        if can_open and file_size > 0:
                            # 如果找到了，更新并选择
                            selected = cand
                            available_files.add(cand)
                            logger.info(
                                f"[Alpha158] ✓ 在错误处理中发现文件: {raw_code} -> {cand}.parquet"
                            )
                            break

                    if selected is None:
                        logger.error(f"[Alpha158]   相关文件示例: {sample_matching}")
                        logger.error(f"[Alpha158]   直接文件检查结果: {direct_check_results}")

            if not resolved_instruments:
                # 提供更详细的错误信息
                sample_files = list(available_files)[:5] if available_files else []
                raise ValueError(
                    f"Qlib数据文件不存在，无法使用Alpha158 handler\n"
                    f"  data_path: {qlib_features_dir}\n"
                    f"  目录存在: {qlib_features_dir.exists()}\n"
                    f"  可用文件数: {len(available_files)}\n"
                    f"  请求的股票代码: {stock_codes}\n"
                    f"  示例文件: {sample_files}"
                )

            # 在创建handler之前，先使用Qlib的D API验证数据是否可以加载
            # 这有助于诊断为什么handler返回空数据
            try:
                logger.debug(
                    f"[Alpha158] 使用D API验证数据可访问性，instruments: {resolved_instruments[:1] if resolved_instruments else []}"
                )
                if resolved_instruments:
                    test_instrument = resolved_instruments[0]
                    test_data = D.features(
                        instruments=[test_instrument],
                        fields=["$open", "$close", "$high", "$low", "$volume"],
                        start_time=start_date.strftime("%Y-%m-%d"),
                        end_time=end_date.strftime("%Y-%m-%d"),
                        freq="day",
                    )
                    logger.debug(
                        f"[Alpha158] D.features()返回: type={type(test_data)}, empty={test_data.empty if hasattr(test_data, 'empty') else 'N/A'}, shape={test_data.shape if hasattr(test_data, 'shape') else 'N/A'}"
                    )
                    if test_data.empty:
                        logger.warning(
                            f"[Alpha158] D.features()返回空数据，可能的原因：数据格式不对或Qlib无法识别文件"
                        )
                    else:
                        logger.info(f"[Alpha158] D.features()成功加载数据: {test_data.shape}")
            except Exception as d_api_error:
                logger.warning(f"[Alpha158] D API验证失败: {d_api_error}，但这不影响handler尝试")

            # 确保qlib已初始化（handler需要）
            try:
                import qlib
                from qlib.config import REG_CN

                # 检查是否已初始化
                if not hasattr(qlib, "_initialized") or not qlib._initialized:
                    logger.debug("[Alpha158] 初始化qlib...")
                    qlib.init(
                        provider_uri=str(
                            Path(settings.QLIB_DATA_PATH).resolve().as_posix()
                        ),
                        region=REG_CN,
                        mount_path=str(
                            Path(settings.QLIB_DATA_PATH).resolve().as_posix()
                        ),
                    )
                    logger.debug("[Alpha158] qlib初始化完成")
            except Exception as init_error:
                logger.warning(f"[Alpha158] qlib初始化失败: {init_error}，尝试继续使用handler")

            # 创建Alpha158 handler
            handler = Alpha158Handler(
                instruments=resolved_instruments,
                start_time=start_date.strftime("%Y-%m-%d"),
                end_time=end_date.strftime("%Y-%m-%d"),
                fit_start_time=start_date.strftime("%Y-%m-%d"),
                fit_end_time=end_date.strftime("%Y-%m-%d"),
            )

            # 获取158个标准因子
            logger.debug(
                f"[Alpha158] 调用handler.fetch()，instruments: {resolved_instruments}, 日期范围: {start_date} 到 {end_date}"
            )
            alpha_factors = handler.fetch()

            logger.debug(
                f"[Alpha158] handler.fetch()返回: type={type(alpha_factors)}, empty={alpha_factors.empty if hasattr(alpha_factors, 'empty') else 'N/A'}, shape={alpha_factors.shape if hasattr(alpha_factors, 'shape') else 'N/A'}"
            )

            if alpha_factors is not None and not alpha_factors.empty:
                # 将Qlib内部的instrument名称映射回原始股票代码，便于和qlib_data对齐
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
                    logger.debug(f"映射Alpha158 instrument名称失败: {map_error}")
                logger.info(
                    f"Alpha158 handler计算完成: {alpha_factors.shape}, 因子数: {len(alpha_factors.columns)}"
                )
                return alpha_factors
            else:
                # 添加更详细的诊断信息
                # 尝试读取实际文件，检查数据格式
                try:
                    import pandas as pd

                    test_file = qlib_features_dir / f"{resolved_instruments[0]}.parquet"
                    bin_dir = qlib_bin_dir / str(resolved_instruments[0]).lower()
                    if test_file.exists():
                        test_data = pd.read_parquet(test_file)
                        logger.warning(
                            f"[Alpha158] handler返回空数据，诊断信息：\n"
                            f"  文件路径: {test_file}\n"
                            f"  文件存在: {test_file.exists()}\n"
                            f"  文件大小: {test_file.stat().st_size if test_file.exists() else 0} 字节\n"
                            f"  数据形状: {test_data.shape if 'test_data' in locals() else 'N/A'}\n"
                            f"  数据列名: {list(test_data.columns) if 'test_data' in locals() and not test_data.empty else 'N/A'}\n"
                            f"  日期范围: {test_data.index.get_level_values(1).min() if isinstance(test_data.index, pd.MultiIndex) and test_data.index.nlevels >= 2 else test_data.index.min() if 'test_data' in locals() and not test_data.empty else 'N/A'} 到 {test_data.index.get_level_values(1).max() if isinstance(test_data.index, pd.MultiIndex) and test_data.index.nlevels >= 2 else test_data.index.max() if 'test_data' in locals() and not test_data.empty else 'N/A'}\n"
                            f"  请求日期范围: {start_date} 到 {end_date}\n"
                            f"  instrument名称: {resolved_instruments}"
                        )
                    elif bin_dir.is_dir() and list(bin_dir.glob("*.day.bin")):
                        logger.warning(
                            f"[Alpha158] handler返回空数据，诊断信息：\n"
                            f"  bin目录: {bin_dir}\n"
                            f"  bin文件数: {len(list(bin_dir.glob('*.day.bin')))}\n"
                            f"  请求日期范围: {start_date} 到 {end_date}\n"
                            f"  instrument名称: {resolved_instruments}"
                        )
                except Exception as diag_error:
                    logger.warning(f"[Alpha158] 诊断信息获取失败: {diag_error}")

                raise ValueError("Alpha158 handler返回空数据")

        except Exception as e:
            error_msg = str(e)
            # 如果是日历文件路径问题，提供更详细的错误信息
            if "calendar not exists" in error_msg or "calendar" in error_msg.lower():
                calendar_path = (
                    Path(settings.QLIB_DATA_PATH).resolve() / "calendars" / "day.txt"
                )
                logger.warning(
                    f"使用Alpha158 handler计算失败（日历文件路径问题）: {e}\n"
                    f"日历文件应该位于: {calendar_path}\n"
                    f"文件是否存在: {calendar_path.exists()}\n"
                    f"将回退到表达式引擎计算"
                )
            else:
                logger.warning(f"使用Alpha158 handler计算失败: {e}")
            raise

    async def _calculate_using_expression_engine(
        self,
        qlib_data: pd.DataFrame,
        stock_codes: List[str],
        date_range: Tuple[datetime, datetime],
    ) -> pd.DataFrame:
        """
        使用Qlib表达式引擎计算Alpha158因子

        基于Alpha158DL.get_feature_config()获取的因子表达式
        """
        try:
            # 检查数据结构
            if (
                isinstance(qlib_data.index, pd.MultiIndex)
                and qlib_data.index.nlevels == 2
            ):
                # MultiIndex: (stock_code, date)
                logger.info("使用MultiIndex数据结构，按股票分组计算")

                # 按股票分割数据
                stock_groups = {}
                for stock_code in stock_codes:
                    try:
                        stock_data = qlib_data.xs(stock_code, level=0, drop_level=False)
                        if not stock_data.empty:
                            stock_groups[stock_code] = stock_data
                    except KeyError:
                        logger.warning(f"股票 {stock_code} 不在数据中")
                        continue
            else:
                logger.warning("数据格式不支持，无法使用表达式引擎计算")
                return pd.DataFrame()

            # 使用表达式引擎计算因子
            # 由于Qlib表达式引擎需要数据在Qlib系统中，我们使用pandas实现表达式计算
            # 这里使用表达式解析和pandas计算

            factors_list = []

            # 使用多进程并行计算
            if len(stock_groups) > 1 and self.max_workers > 1:
                logger.info(f"使用 {self.max_workers} 个进程并行计算因子")

                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}

                    # 提交任务
                    for stock_code, stock_data in stock_groups.items():
                        future = executor.submit(
                            self._calculate_alpha_factors_from_expressions,
                            stock_data,
                            stock_code,
                        )
                        futures[future] = stock_code

                    # 收集结果
                    for future in as_completed(futures):
                        stock_code = futures[future]
                        try:
                            stock_factors = future.result()
                            if not stock_factors.empty:
                                factors_list.append(stock_factors)
                                logger.debug(
                                    f"完成股票 {stock_code} 的因子计算: {len(stock_factors.columns)} 个因子"
                                )
                        except Exception as e:
                            logger.error(f"计算股票 {stock_code} 的因子时发生错误: {e}")
            else:
                # 单进程计算
                logger.info("使用单进程计算因子")
                for stock_code, stock_data in stock_groups.items():
                    stock_factors = self._calculate_alpha_factors_from_expressions(
                        stock_data, stock_code
                    )
                    if not stock_factors.empty:
                        factors_list.append(stock_factors)
                        logger.debug(
                            f"完成股票 {stock_code} 的因子计算: {len(stock_factors.columns)} 个因子"
                        )

            # 合并所有因子
            if factors_list:
                alpha_factors = pd.concat(factors_list)
                return alpha_factors
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"使用表达式引擎计算失败: {e}")
            raise

    def _calculate_alpha_factors_from_expressions(
        self, stock_data: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """
        从Alpha158表达式计算因子（使用pandas实现）

        使用Qlib的alpha_fields和alpha_names，通过表达式解析计算全部158个因子
        """
        try:
            # 确保数据有正确的列
            required_cols = ["$close", "$high", "$low", "$volume", "$open"]
            missing_cols = [
                col for col in required_cols if col not in stock_data.columns
            ]
            if missing_cols:
                logger.warning(f"股票 {stock_code} 缺少必要列: {missing_cols}，无法计算Alpha158因子")
                return pd.DataFrame(index=stock_data.index)

            factors = pd.DataFrame(index=stock_data.index)

            # 如果有完整的alpha_fields和alpha_names，使用表达式解析计算全部158个因子
            if (
                len(self.alpha_fields) > 0
                and len(self.alpha_names) > 0
                and len(self.alpha_fields) == len(self.alpha_names)
            ):
                logger.info(
                    f"股票 {stock_code} 使用表达式解析计算全部 {len(self.alpha_fields)} 个Alpha158因子"
                )

                success_count = 0
                fail_count = 0

                # 遍历所有因子表达式
                failed_expressions = []  # 记录失败的表达式，用于调试
                total_factors = len(self.alpha_fields)
                for idx, (field_expr, factor_name) in enumerate(
                    zip(self.alpha_fields, self.alpha_names)
                ):
                    # 每10个因子输出一次进度
                    if idx % 10 == 0:
                        logger.info(
                            f"股票 {stock_code} 计算进度: {idx+1}/{total_factors} ({((idx+1)/total_factors*100):.1f}%)"
                        )
                    try:
                        # 使用表达式评估器计算因子
                        factor_series = self._evaluate_qlib_expression(
                            stock_data, field_expr
                        )
                        if factor_series is not None and len(factor_series) > 0:
                            # 检查是否有有效值（不是全部NaN）
                            valid_count = factor_series.notna().sum()
                            if valid_count > 0:
                                # 有有效值，填充NaN为0（对于滚动窗口函数，前几个值可能是NaN）
                                factors[factor_name] = factor_series.fillna(0)
                                success_count += 1
                            else:
                                # 全部是NaN，可能是数据不足或计算错误
                                logger.debug(
                                    f"因子 {factor_name} ({idx+1}/{len(self.alpha_fields)}) 全部为NaN: {field_expr}"
                                )
                                factors[factor_name] = 0
                                fail_count += 1
                                failed_expressions.append(
                                    (factor_name, field_expr, "全部为NaN")
                                )
                        else:
                            # 如果表达式解析失败，填充0
                            logger.debug(
                                f"因子 {factor_name} ({idx+1}/{len(self.alpha_fields)}) 表达式解析返回空: {field_expr}"
                            )
                            factors[factor_name] = 0
                            fail_count += 1
                            failed_expressions.append((factor_name, field_expr, "返回空"))
                    except Exception as e:
                        logger.debug(
                            f"计算因子 {factor_name} ({idx+1}/{len(self.alpha_fields)}) 失败: {e}, 表达式: {field_expr}"
                        )
                        # 失败时填充0
                        factors[factor_name] = 0
                        fail_count += 1
                        failed_expressions.append((factor_name, field_expr, str(e)))

                # 如果失败数量较多，记录前20个失败的表达式用于调试
                if fail_count > 0 and len(failed_expressions) > 0:
                    logger.warning(f"股票 {stock_code} 失败的因子表达式（前20个）:")
                    for name, expr, error in failed_expressions[:20]:
                        logger.warning(f"  - {name}: {expr} (错误: {error})")

                logger.info(
                    f"股票 {stock_code} 表达式解析完成: 成功 {success_count} 个, 失败 {fail_count} 个, 总计 {len(factors.columns)} 个因子"
                )
            else:
                # 回退到简化版本（47个核心因子）
                logger.warning(f"Alpha158配置不可用，使用简化版本（47个核心因子）")
                close = stock_data["$close"]
                high = stock_data["$high"]
                low = stock_data["$low"]
                volume = stock_data["$volume"]
                open_ = stock_data["$open"]

                # 基础价格因子
                factors["KMID"] = (close - open_) / open_
                factors["KLEN"] = (high - low) / close
                factors["KUP"] = (high - close) / close
                factors["KLOW"] = (close - low) / close

                # 价格收益率（不同周期）
                for period in [1, 2, 3, 5, 10, 20, 30, 60]:
                    factors[f"RET{period}"] = close.pct_change(period)

                # 移动平均
                for period in [5, 10, 20, 30, 60]:
                    factors[f"MA{period}"] = close.rolling(period).mean()

                # 标准差
                for period in [5, 10, 20, 30, 60]:
                    factors[f"STD{period}"] = close.rolling(period).std()

                # 最大值/最小值
                for period in [5, 10, 20, 30, 60]:
                    factors[f"MAX{period}"] = close.rolling(period).max()
                    factors[f"MIN{period}"] = close.rolling(period).min()

                # 量价相关性
                for period in [5, 10, 20, 30, 60]:
                    factors[f"CORR{period}"] = close.rolling(period).corr(volume)

                # 成交量因子
                for period in [5, 10, 20, 30, 60]:
                    factors[f"VMA{period}"] = volume.rolling(period).mean()
                    factors[f"VSTD{period}"] = volume.rolling(period).std()

            # 填充缺失值（使用新API）
            factors = factors.bfill().fillna(0)

            logger.debug(f"股票 {stock_code} 最终计算了 {len(factors.columns)} 个因子")

            return factors

        except Exception as e:
            logger.error(f"计算股票 {stock_code} 的因子失败: {e}", exc_info=True)
            return pd.DataFrame(index=stock_data.index)

    async def _calculate_qlib_alpha158_factors(
        self, data: pd.DataFrame, stock_codes: List[str]
    ) -> pd.DataFrame:
        """使用Qlib内置的Alpha158计算因子"""
        if data.empty or len(self.alpha_fields) == 0:
            return pd.DataFrame(index=data.index)

        try:
            # 确保数据有正确的列
            required_cols = ["$close", "$high", "$low", "$volume", "$open"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"缺少必要列: {missing_cols}，无法计算Alpha158因子")
                return pd.DataFrame(index=data.index)

            # 使用QlibDataLoader来计算Alpha158因子
            # 创建因子配置：[(expression, name), ...]
            factor_config = list(zip(self.alpha_fields, self.alpha_names))

            # 创建QlibDataLoader
            loader_config = {
                "config": {
                    "feature": factor_config,
                }
            }

            # 使用QlibDataLoader的load_group_df方法计算因子
            # 注意：这需要数据已经在Qlib的数据系统中
            # 由于我们使用的是DataFrame，需要直接计算表达式

            # 方法：使用Qlib的表达式引擎直接计算
            # 但Qlib的表达式引擎需要数据在Qlib系统中
            # 所以我们使用pandas实现表达式计算（基于Qlib的表达式定义）

            factors = pd.DataFrame(index=data.index)

            # 为了兼容性，我们使用pandas实现表达式计算
            # 这是一个折中方案：使用Qlib的因子定义，但用pandas计算
            logger.info(f"使用Qlib Alpha158因子定义计算 {len(self.alpha_fields)} 个因子")

            # 由于Qlib表达式引擎需要数据在Qlib系统中，我们使用表达式解析和pandas计算
            # 这里先实现一个简化版本，直接使用QlibDataLoader（如果可能）

            # 尝试使用QlibDataLoader
            try:
                loader = QlibDataLoader(config=loader_config["config"])

                # 准备数据：需要将DataFrame转换为Qlib可以使用的格式
                # 由于QlibDataLoader需要从Qlib数据系统加载，我们需要另一种方法

                # 方法：直接使用pandas计算表达式（基于Qlib表达式定义）
                # 这需要解析Qlib表达式并转换为pandas操作
                # 为了简化，我们使用一个表达式计算库或自己实现

                # 暂时使用简化实现，但使用Qlib定义的因子名称
                logger.info("使用Qlib Alpha158因子定义，通过pandas计算")
                # 注意：这里调用的是Alpha158Calculator类中的方法，需要按股票分组计算
                # 由于_calculate_alpha_factors_from_expressions需要(stock_data, stock_code)参数
                # 我们需要按股票分组调用
                factors_list = []
                for stock_code in stock_codes:
                    try:
                        if isinstance(data.index, pd.MultiIndex):
                            stock_data = data.xs(stock_code, level=0, drop_level=False)
                        else:
                            stock_data = data.copy()

                        if not stock_data.empty:
                            stock_factors = (
                                self._calculate_alpha_factors_from_expressions(
                                    stock_data, stock_code
                                )
                            )
                            if not stock_factors.empty:
                                factors_list.append(stock_factors)
                    except Exception as e:
                        logger.warning(f"计算股票 {stock_code} 的因子失败: {e}")
                        continue

                if factors_list:
                    return pd.concat(factors_list)
                else:
                    return pd.DataFrame(index=data.index)

            except Exception as e:
                logger.warning(f"使用QlibDataLoader失败: {e}，使用表达式计算")
                # 回退到按股票分组计算
                factors_list = []
                for stock_code in stock_codes:
                    try:
                        if isinstance(data.index, pd.MultiIndex):
                            stock_data = data.xs(stock_code, level=0, drop_level=False)
                        else:
                            stock_data = data.copy()

                        if not stock_data.empty:
                            stock_factors = (
                                self._calculate_alpha_factors_from_expressions(
                                    stock_data, stock_code
                                )
                            )
                            if not stock_factors.empty:
                                factors_list.append(stock_factors)
                    except Exception as e2:
                        logger.warning(f"计算股票 {stock_code} 的因子失败: {e2}")
                        continue

                if factors_list:
                    return pd.concat(factors_list)
                else:
                    return pd.DataFrame(index=data.index)

        except Exception as e:
            logger.error(f"Qlib Alpha158因子计算失败: {e}", exc_info=True)
            return pd.DataFrame(index=data.index)

    def _evaluate_qlib_expression(
        self, data: pd.DataFrame, expression: str
    ) -> Optional[pd.Series]:
        """
        评估Qlib表达式（使用pandas实现）

        支持Alpha158中使用的所有Qlib函数，包括嵌套函数和复杂表达式
        """
        try:
            import re

            # 创建一个数据副本用于计算（避免修改原始数据）
            calc_data = data.copy()

            # 标准化列名：确保基础列有$前缀
            # 如果列名没有$前缀，添加$前缀
            column_mapping = {}
            for col in ["open", "high", "low", "close", "volume", "vwap"]:
                if col in calc_data.columns and f"${col}" not in calc_data.columns:
                    column_mapping[col] = f"${col}"
            if column_mapping:
                calc_data = calc_data.rename(columns=column_mapping)

            # 确保$vwap列存在（如果不存在则使用$close替代）
            if "$vwap" not in calc_data.columns:
                if "$close" in calc_data.columns:
                    calc_data["$vwap"] = calc_data["$close"]
                elif "close" in calc_data.columns:
                    calc_data["$vwap"] = calc_data["close"]

            # 使用递归方式处理嵌套函数，从内到外
            # 先处理最内层的函数，然后逐步向外处理

            # 步骤1: 处理Ref函数（最基础，可能被其他函数使用）
            # 注意：这里直接替换变量，因为Ref函数需要立即使用变量
            # 但需要确保变量名已经标准化（有$前缀）
            def replace_ref(match):
                var = match.group(1)
                n = int(match.group(2))
                # 确保变量名有$前缀（calc_data中的列名应该有$前缀）
                return f"calc_data['${var}'].shift({n})"

            expr = expression
            # 添加最大迭代次数防止无限循环
            max_ref_iterations = 50
            ref_iteration = 0
            while (
                re.search(r"Ref\(\$(\w+),\s*(\d+)\)", expr)
                and ref_iteration < max_ref_iterations
            ):
                expr = re.sub(r"Ref\(\$(\w+),\s*(\d+)\)", replace_ref, expr)
                ref_iteration += 1

            # 步骤1.5: 修复Ref函数替换后可能产生的嵌套calc_data
            # 如果Ref函数中的$var在后续变量替换时又被替换，会产生嵌套
            # 这里先修复这种情况
            var_names = ["close", "open", "high", "low", "volume", "vwap"]
            for var_name in var_names:
                # 修复 calc_data['calc_data['$var']'].shift(...) 格式
                nested_pattern = (
                    rf"calc_data\['calc_data\['\${var_name}'\]'\]\.shift\(([^)]+)\)"
                )
                if re.search(nested_pattern, expr):
                    expr = re.sub(
                        nested_pattern, rf"calc_data['\${var_name}'].shift(\1)", expr
                    )
                # 更简单的修复：直接替换嵌套的calc_data（在任何上下文中）
                # 需要处理多种嵌套情况
                nested_patterns = [
                    f"calc_data['calc_data['${var_name}']']",  # 基本嵌套
                    f"calc_data['calc_data['${var_name}']'].shift",  # 带shift的嵌套
                ]
                for nested_pattern in nested_patterns:
                    if nested_pattern in expr:
                        expr = expr.replace(nested_pattern, f"calc_data['${var_name}']")

            # 步骤2: 处理Log函数（可能在Corr等函数内部）
            def replace_log(match):
                inner = match.group(1)
                # 处理 Log($var+1) 或 Log($var-1) 等形式
                if "+" in inner or "-" in inner:
                    var_match = re.search(r"\$(\w+)", inner)
                    if var_match:
                        var = var_match.group(1)
                        # 提取常数部分（支持科学计数法如1e-12）
                        const_match = re.search(
                            r"([+-]\s*(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))", inner
                        )
                        if const_match:
                            const = const_match.group(1).replace(" ", "")
                            return f"np.log(calc_data['${var}']{const})"
                # 简单情况：Log($var) 或 Log(表达式)
                var_match = re.search(r"\$(\w+)", inner)
                if var_match:
                    var = var_match.group(1)
                    return f"np.log(calc_data['${var}'])"
                # 如果inner已经是表达式，直接包装
                return f"np.log({inner})"

            # 添加最大迭代次数防止无限循环
            max_log_iterations = 50
            log_iteration = 0
            while (
                re.search(r"Log\(([^)]+)\)", expr)
                and log_iteration < max_log_iterations
            ):
                expr = re.sub(r"Log\(([^)]+)\)", replace_log, expr)
                log_iteration += 1

            # 步骤3: 处理Abs函数
            def replace_abs(match):
                inner = match.group(1)
                return f"np.abs({inner})"

            # 添加最大迭代次数防止无限循环
            max_abs_iterations = 50
            abs_iteration = 0
            while (
                re.search(r"Abs\(([^)]+)\)", expr)
                and abs_iteration < max_abs_iterations
            ):
                expr = re.sub(r"Abs\(([^)]+)\)", replace_abs, expr)
                abs_iteration += 1

            # 步骤4: 保留Greater/Less，交由运行时函数处理，返回Series以支持rolling
            def _as_series(x, ref):
                if isinstance(x, pd.Series):
                    return x
                if np.isscalar(x):
                    return pd.Series([x] * len(ref), index=ref.index)
                if isinstance(x, np.ndarray):
                    return pd.Series(x, index=ref.index)
                return pd.Series(x, index=ref.index)

            def Greater(a, b=0):
                """Greater函数：返回布尔Series，表示a > b"""
                if isinstance(a, pd.Series) or isinstance(b, pd.Series):
                    ref = a if isinstance(a, pd.Series) else b
                    a_s = _as_series(a, ref)
                    b_s = _as_series(b, ref)
                    # 返回布尔Series，True表示a > b，False表示a <= b
                    return (a_s > b_s).astype(float)
                return (
                    float(a > b)
                    if np.isscalar(a) and np.isscalar(b)
                    else np.maximum(a, b)
                )

            def Less(a, b=0):
                """Less函数：返回布尔Series，表示a < b"""
                if isinstance(a, pd.Series) or isinstance(b, pd.Series):
                    ref = a if isinstance(a, pd.Series) else b
                    a_s = _as_series(a, ref)
                    b_s = _as_series(b, ref)
                    # 返回布尔Series，True表示a < b，False表示a >= b
                    return (a_s < b_s).astype(float)
                return (
                    float(a < b)
                    if np.isscalar(a) and np.isscalar(b)
                    else np.minimum(a, b)
                )

            def Sum(x, n):
                """Sum函数：对Series进行滚动求和"""
                # 确保x是Series
                if not isinstance(x, pd.Series):
                    x_s = _as_series(x, calc_data)
                else:
                    x_s = x
                # 确保索引对齐
                if not x_s.index.equals(calc_data.index):
                    x_s = x_s.reindex(calc_data.index)
                # 如果是布尔类型，转换为数值类型（True->1, False->0）
                if x_s.dtype == bool:
                    x_s = x_s.astype(float)
                return x_s.rolling(int(n)).sum()

            def Mean(x, n):
                """Mean函数：对Series进行滚动均值"""
                if not isinstance(x, pd.Series):
                    x_s = _as_series(x, calc_data)
                else:
                    x_s = x
                if not x_s.index.equals(calc_data.index):
                    x_s = x_s.reindex(calc_data.index)
                if x_s.dtype == bool:
                    x_s = x_s.astype(float)
                return x_s.rolling(int(n)).mean()

            def Std(x, n):
                """Std函数：对Series进行滚动标准差"""
                if not isinstance(x, pd.Series):
                    x_s = _as_series(x, calc_data)
                else:
                    x_s = x
                if not x_s.index.equals(calc_data.index):
                    x_s = x_s.reindex(calc_data.index)
                if x_s.dtype == bool:
                    x_s = x_s.astype(float)
                return x_s.rolling(int(n)).std()

            # 步骤5: 处理IdxMax和IdxMin函数（优先于Max/Min，避免IdxMax被Max替换）
            # 注意：这些函数已经在返回值中包含了calc_data['$var']，所以不需要在步骤14再次替换
            # IdxMax返回最大值在窗口中的位置（从右往左，0到n-1），然后除以n归一化
            # IdxMin返回最小值在窗口中的位置（从右往左，0到n-1），然后除以n归一化
            def replace_idxmax(match):
                var = match.group(1)
                n = int(match.group(2))
                # 计算最大值在窗口中的位置（从右往左，0到n-1），然后除以n
                # argmax返回从左往右的第一个最大值位置，我们需要从右往左的位置
                # 注意：raw=True时x是numpy数组，需要使用pd.isna()或np.isnan()
                return f"(calc_data['${var}'].rolling({n}).apply(lambda x: (len(x) - 1 - x.argmax()) / {n} if len(x) == {n} and not np.isnan(x).all() else np.nan, raw=True))"

            def replace_idxmin(match):
                var = match.group(1)
                n = int(match.group(2))
                # 计算最小值在窗口中的位置（从右往左，0到n-1），然后除以n
                # 注意：raw=True时x是numpy数组，需要使用pd.isna()或np.isnan()
                return f"(calc_data['${var}'].rolling({n}).apply(lambda x: (len(x) - 1 - x.argmin()) / {n} if len(x) == {n} and not np.isnan(x).all() else np.nan, raw=True))"

            # 添加最大迭代次数防止无限循环
            max_idxmax_iterations = 50
            idxmax_iteration = 0
            while (
                re.search(r"IdxMax\(\$(\w+),\s*(\d+)\)", expr)
                and idxmax_iteration < max_idxmax_iterations
            ):
                expr = re.sub(r"IdxMax\(\$(\w+),\s*(\d+)\)", replace_idxmax, expr)
                idxmax_iteration += 1
            max_idxmin_iterations = 50
            idxmin_iteration = 0
            while (
                re.search(r"IdxMin\(\$(\w+),\s*(\d+)\)", expr)
                and idxmin_iteration < max_idxmin_iterations
            ):
                expr = re.sub(r"IdxMin\(\$(\w+),\s*(\d+)\)", replace_idxmin, expr)
                idxmin_iteration += 1

            # 步骤6: 处理单变量滚动函数（Max, Min, Mean, Std等）
            # 注意：Max/Min函数可能有两种用法：
            # 1. Max($var, n) - 滚动窗口的最大值
            # 2. Max($var, scalar) - Series和标量的逐元素最大值（运行时处理）
            # 这里先处理滚动窗口的情况，标量情况由运行时函数处理

            # 运行时Max/Min函数（处理Series和标量的情况）
            def Max(a, b):
                """Max函数：如果两个参数都是Series或一个是Series一个是标量，返回逐元素最大值"""
                if isinstance(a, pd.Series) or isinstance(b, pd.Series):
                    ref = a if isinstance(a, pd.Series) else b
                    a_s = _as_series(a, ref)
                    b_s = _as_series(b, ref)
                    return pd.concat([a_s, b_s], axis=1).max(axis=1)
                return (
                    max(a, b) if np.isscalar(a) and np.isscalar(b) else np.maximum(a, b)
                )

            def Min(a, b):
                """Min函数：如果两个参数都是Series或一个是Series一个是标量，返回逐元素最小值"""
                if isinstance(a, pd.Series) or isinstance(b, pd.Series):
                    ref = a if isinstance(a, pd.Series) else b
                    a_s = _as_series(a, ref)
                    b_s = _as_series(b, ref)
                    return pd.concat([a_s, b_s], axis=1).min(axis=1)
                return (
                    min(a, b) if np.isscalar(a) and np.isscalar(b) else np.minimum(a, b)
                )

            def replace_max(match):
                var = match.group(1)
                n = int(match.group(2))
                return f"calc_data['${var}'].rolling({n}).max()"

            def replace_min(match):
                var = match.group(1)
                n = int(match.group(2))
                return f"calc_data['${var}'].rolling({n}).min()"

            def replace_mean_var(match):
                var = match.group(1)
                n = int(match.group(2))
                return f"calc_data['${var}'].rolling({n}).mean()"

            def replace_std(match):
                var = match.group(1)
                n = int(match.group(2))
                return f"calc_data['${var}'].rolling({n}).std()"

            # 添加最大迭代次数防止无限循环
            max_max_iterations = 50
            max_iteration = 0
            while (
                re.search(r"Max\(\$(\w+),\s*(\d+)\)", expr)
                and max_iteration < max_max_iterations
            ):
                expr = re.sub(r"Max\(\$(\w+),\s*(\d+)\)", replace_max, expr)
                max_iteration += 1
            max_min_iterations = 50
            min_iteration = 0
            while (
                re.search(r"Min\(\$(\w+),\s*(\d+)\)", expr)
                and min_iteration < max_min_iterations
            ):
                expr = re.sub(r"Min\(\$(\w+),\s*(\d+)\)", replace_min, expr)
                min_iteration += 1
            max_mean_var_iterations = 50
            mean_var_iteration = 0
            while (
                re.search(r"Mean\(\$(\w+),\s*(\d+)\)", expr)
                and mean_var_iteration < max_mean_var_iterations
            ):
                expr = re.sub(r"Mean\(\$(\w+),\s*(\d+)\)", replace_mean_var, expr)
                mean_var_iteration += 1
            max_std_var_iterations = 50
            std_var_iteration = 0
            while (
                re.search(r"Std\(\$(\w+),\s*(\d+)\)", expr)
                and std_var_iteration < max_std_var_iterations
            ):
                expr = re.sub(r"Std\(\$(\w+),\s*(\d+)\)", replace_std, expr)
                std_var_iteration += 1

            # 步骤7: 处理Corr函数（两个变量）
            def replace_corr(match):
                var1_expr = match.group(1)
                var2_expr = match.group(2)
                n = int(match.group(3))
                # 如果var1或var2是表达式，需要先处理
                # 简化处理：假设都是$var格式
                var1_match = re.search(r"\$(\w+)", var1_expr)
                var2_match = re.search(r"\$(\w+)", var2_expr)
                if var1_match and var2_match:
                    var1 = var1_match.group(1)
                    var2 = var2_match.group(1)
                    return (
                        f"calc_data['${var1}'].rolling({n}).corr(calc_data['${var2}'])"
                    )
                # 如果包含表达式，需要更复杂的处理
                return (
                    f"pd.Series({var1_expr}).rolling({n}).corr(pd.Series({var2_expr}))"
                )

            # 添加最大迭代次数防止无限循环
            max_corr_iterations = 50
            corr_iteration = 0
            while (
                re.search(r"Corr\(([^,]+),\s*([^,]+),\s*(\d+)\)", expr)
                and corr_iteration < max_corr_iterations
            ):
                expr = re.sub(
                    r"Corr\(([^,]+),\s*([^,]+),\s*(\d+)\)", replace_corr, expr
                )
                corr_iteration += 1

            # 步骤8: Sum函数现在由运行时函数处理，不需要在表达式解析时替换
            # Sum函数保留为运行时调用，由eval时的Sum函数处理

            # 步骤9: 处理Mean函数（用于嵌套表达式，如Mean(Abs(...), n)）
            # 使用类似Sum函数的括号匹配方法处理嵌套Mean函数
            def find_mean_and_replace(expr_str):
                """找到Mean函数并替换，处理嵌套括号"""

                def find_matching_paren(s, start_pos):
                    count = 0
                    i = start_pos
                    while i < len(s):
                        if s[i] == "(":
                            count += 1
                        elif s[i] == ")":
                            count -= 1
                            if count == 0:
                                return i
                        i += 1
                    return -1

                mean_positions = [m.start() for m in re.finditer(r"Mean\(", expr_str)]
                if not mean_positions:
                    return expr_str, False

                for pos in reversed(mean_positions):
                    end_pos = find_matching_paren(expr_str, pos + 5)
                    if end_pos > 0:
                        comma_pos = expr_str.rfind(",", pos + 5, end_pos)
                        if comma_pos > 0:
                            inner_expr = expr_str[pos + 5 : comma_pos].strip()
                            n_str = expr_str[comma_pos + 1 : end_pos].strip()
                            try:
                                n = int(n_str)
                                # 检查是否是简单的$var格式（已经在步骤6处理过）
                                if re.match(
                                    r"calc_data\[\'\$\w+\'\]", inner_expr.strip()
                                ):
                                    continue  # 跳过，已经在步骤6处理过
                                # 对于复杂表达式（包括比较操作符、函数调用等），使用pd.Series包装
                                replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).mean()"
                                new_expr = (
                                    expr_str[:pos]
                                    + replacement
                                    + expr_str[end_pos + 1 :]
                                )
                                return new_expr, True
                            except ValueError:
                                pass
                return expr_str, False

            # 注意：Mean函数需要在变量替换之后处理，所以这里先跳过
            # Mean函数将在步骤15（变量替换后）处理
            # 但需要先定义find_mean_and_replace函数，以便在步骤15使用
            # find_mean_and_replace函数已在上面定义

            # 步骤10: 处理Std函数（用于嵌套表达式，如Std(Abs(...), n)）
            # 使用类似Sum函数的括号匹配方法处理嵌套Std函数
            def find_std_and_replace(expr_str):
                """找到Std函数并替换，处理嵌套括号"""

                def find_matching_paren(s, start_pos):
                    """找到匹配的右括号位置"""
                    count = 0
                    i = start_pos
                    while i < len(s):
                        if s[i] == "(":
                            count += 1
                        elif s[i] == ")":
                            count -= 1
                            if count == 0:
                                return i
                        i += 1
                    return -1

                # 查找所有Std(的位置
                std_positions = [m.start() for m in re.finditer(r"Std\(", expr_str)]
                if not std_positions:
                    return expr_str, False

                # 从后往前处理，避免位置偏移
                for pos in reversed(std_positions):
                    # 找到匹配的右括号
                    end_pos = find_matching_paren(expr_str, pos + 4)
                    if end_pos > 0:
                        # 找到最后一个逗号（分隔参数和n的）
                        comma_pos = expr_str.rfind(",", pos + 4, end_pos)
                        if comma_pos > 0:
                            inner_expr = expr_str[pos + 4 : comma_pos].strip()
                            n_str = expr_str[comma_pos + 1 : end_pos].strip()
                            try:
                                n = int(n_str)
                                # 检查是否是简单的$var格式（已经在步骤6处理过）
                                if re.match(
                                    r"calc_data\[\'\$\w+\'\]", inner_expr.strip()
                                ):
                                    continue  # 跳过，已经在步骤6处理过
                                # 对于复杂表达式，使用pd.Series包装
                                replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).std()"
                                new_expr = (
                                    expr_str[:pos]
                                    + replacement
                                    + expr_str[end_pos + 1 :]
                                )
                                return new_expr, True
                            except ValueError:
                                pass
                return expr_str, False

            # 递归处理嵌套的Std函数（从内到外）
            # 注意：需要在变量替换之前处理，但变量可能还没有替换
            # 所以需要先进行基本的变量替换（只替换独立的$var，不在函数参数中的）
            max_iterations_std = 20
            iteration = 0
            prev_expr = ""
            while iteration < max_iterations_std:
                if expr == prev_expr:
                    break
                prev_expr = expr
                new_expr, replaced = find_std_and_replace(expr)
                if not replaced:
                    break
                expr = new_expr
                iteration += 1

            # 步骤11: 处理Quantile函数
            def replace_quantile(match):
                var = match.group(1)
                n = int(match.group(2))
                q = float(match.group(3))
                return f"calc_data['${var}'].rolling({n}).quantile({q})"

            # 添加最大迭代次数防止无限循环
            max_quantile_iterations = 50
            quantile_iteration = 0
            while (
                re.search(r"Quantile\(\$(\w+),\s*(\d+),\s*([\d.]+)\)", expr)
                and quantile_iteration < max_quantile_iterations
            ):
                expr = re.sub(
                    r"Quantile\(\$(\w+),\s*(\d+),\s*([\d.]+)\)", replace_quantile, expr
                )
                quantile_iteration += 1

            # 步骤12: 处理Rank函数
            def replace_rank(match):
                var = match.group(1)
                n = int(match.group(2))
                # 使用raw=False，x是Series，可以使用iloc
                return f"calc_data['${var}'].rolling({n}).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == {n} and not pd.Series(x).isna().all() else np.nan, raw=False)"

            max_rank_iterations = 50
            rank_iteration = 0
            while (
                re.search(r"Rank\(\$(\w+),\s*(\d+)\)", expr)
                and rank_iteration < max_rank_iterations
            ):
                expr = re.sub(r"Rank\(\$(\w+),\s*(\d+)\)", replace_rank, expr)
                rank_iteration += 1

            # 步骤13: 处理Slope, Rsquare, Resi函数（如果需要）
            def replace_slope(match):
                var = match.group(1)
                n = int(match.group(2))
                return f"calc_data['${var}'].rolling({n}).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == {n} else np.nan, raw=True)"

            def replace_rsquare(match):
                var = match.group(1)
                n = int(match.group(2))
                # 使用raw=False，x是Series，可以使用iloc
                return f"calc_data['${var}'].rolling({n}).apply(lambda x: 1 - np.var(x.values - np.linspace(x.iloc[0] if len(x) > 0 else 0, x.iloc[-1] if len(x) > 0 else 0, len(x))) / (np.var(x.values) + 1e-8) if len(x) == {n} and np.var(x.values) > 0 else 0, raw=False)"

            def replace_resi(match):
                var = match.group(1)
                n = int(match.group(2))
                # 使用raw=False，x是Series，可以使用iloc
                return f"calc_data['${var}'].rolling({n}).apply(lambda x: x.iloc[-1] - (np.polyfit(range(len(x)), x.values, 1)[0] * (len(x) - 1) + np.polyfit(range(len(x)), x.values, 1)[1]) if len(x) == {n} else np.nan, raw=False)"

            # 添加最大迭代次数防止无限循环
            max_slope_iterations = 50
            slope_iteration = 0
            while (
                re.search(r"Slope\(\$(\w+),\s*(\d+)\)", expr)
                and slope_iteration < max_slope_iterations
            ):
                expr = re.sub(r"Slope\(\$(\w+),\s*(\d+)\)", replace_slope, expr)
                slope_iteration += 1
            max_rsquare_iterations = 50
            rsquare_iteration = 0
            while (
                re.search(r"Rsquare\(\$(\w+),\s*(\d+)\)", expr)
                and rsquare_iteration < max_rsquare_iterations
            ):
                expr = re.sub(r"Rsquare\(\$(\w+),\s*(\d+)\)", replace_rsquare, expr)
                rsquare_iteration += 1
            max_resi_iterations = 50
            resi_iteration = 0
            while (
                re.search(r"Resi\(\$(\w+),\s*(\d+)\)", expr)
                and resi_iteration < max_resi_iterations
            ):
                expr = re.sub(r"Resi\(\$(\w+),\s*(\d+)\)", replace_resi, expr)
                resi_iteration += 1

            # 步骤14: 替换变量引用（最后处理，但在Mean/Std处理之前）
            # 只替换那些还没有被处理的$var（不在calc_data['$var']中的）
            # 使用更安全的方法：先替换所有独立的$var，但要避免替换已经在calc_data['$var']中的
            var_names = ["close", "open", "high", "low", "volume", "vwap"]
            for var_name in var_names:
                replacement = f"calc_data['${var_name}']"
                # 检查是否还有未处理的$var
                if f"${var_name}" in expr:
                    # 先修复可能存在的嵌套calc_data（从之前的错误替换中恢复）
                    # 将calc_data['calc_data['$var']']替换为calc_data['$var']
                    # 需要处理多种嵌套情况
                    # 情况1: calc_data['calc_data['$var']...']（基本嵌套）
                    nested_pattern1 = rf"calc_data\['calc_data\['\${var_name}'\]'\]"
                    if nested_pattern1 in expr:
                        expr = expr.replace(nested_pattern1, replacement)
                    # 情况2: calc_data['calc_data['$var'].shift(...)']（带shift的嵌套）
                    nested_pattern2 = (
                        rf"calc_data\['calc_data\['\${var_name}'\]\.shift\(([^)]+)\)'\]"
                    )
                    if re.search(nested_pattern2, expr):
                        expr = re.sub(
                            nested_pattern2, rf"{replacement}.shift(\1)", expr
                        )
                    # 情况3: 在比较操作符中的嵌套（如 calc_data['$close']>calc_data['calc_data['$close']'].shift(1)）
                    # 这种情况在变量替换时产生，需要特别处理
                    nested_pattern3 = (
                        rf"calc_data\['calc_data\['\${var_name}'\]'\]\.shift\(([^)]+)\)"
                    )
                    if re.search(nested_pattern3, expr):
                        expr = re.sub(
                            nested_pattern3, rf"{replacement}.shift(\1)", expr
                        )
                    # 情况3: 修复在Ref函数中出现的嵌套（Ref($close, 1) -> calc_data['$close'].shift(1)，但$close又被替换了一次）
                    # 这种情况会在变量替换时产生 calc_data['calc_data['$close']'].shift(1)
                    # 需要在变量替换之前就修复Ref函数中的嵌套
                    nested_pattern3 = (
                        rf"calc_data\['calc_data\['\${var_name}'\]'\]\.shift\(([^)]+)\)"
                    )
                    if re.search(nested_pattern3, expr):
                        expr = re.sub(
                            nested_pattern3, rf"{replacement}.shift(\1)", expr
                        )

                    # 使用更精确的替换：只替换不在calc_data['$var']中的$var
                    # 使用固定宽度的lookbehind（只检查calc_data['）
                    pattern = rf"(?<!calc_data\[\')\${var_name}(?!\'\])"
                    # 多次替换直到没有变化（处理嵌套情况）
                    max_replace_iterations = 10
                    for iteration in range(max_replace_iterations):
                        new_expr = re.sub(pattern, replacement, expr)
                        # 每次替换后，立即修复可能产生的嵌套（在继续之前）
                        # 修复 calc_data['calc_data['$var']'] 格式
                        if f"calc_data['calc_data['${var_name}']']" in new_expr:
                            new_expr = new_expr.replace(
                                f"calc_data['calc_data['${var_name}']']", replacement
                            )
                        # 修复 calc_data['calc_data['$var']'].shift(...) 格式
                        nested_shift_pattern = rf"calc_data\['calc_data\['\${var_name}'\]'\]\.shift\(([^)]+)\)"
                        if re.search(nested_shift_pattern, new_expr):
                            new_expr = re.sub(
                                nested_shift_pattern,
                                rf"{replacement}.shift(\1)",
                                new_expr,
                            )
                        if new_expr == expr:
                            # 检查是否还有未替换的$var（可能在函数参数中）
                            if (
                                f"${var_name}" in new_expr
                                and f"calc_data['${var_name}']" in new_expr
                            ):
                                # 还有未替换的$var，尝试更激进的替换
                                # 只保留已经在calc_data中的，其他全部替换
                                temp_expr = new_expr.replace(
                                    f"calc_data['${var_name}']", f"__TEMP_{var_name}__"
                                )
                                temp_expr = temp_expr.replace(
                                    f"${var_name}", replacement
                                )
                                new_expr = temp_expr.replace(
                                    f"__TEMP_{var_name}__", f"calc_data['${var_name}']"
                                )
                                # 再次修复嵌套
                                if f"calc_data['calc_data['${var_name}']']" in new_expr:
                                    new_expr = new_expr.replace(
                                        f"calc_data['calc_data['${var_name}']']",
                                        replacement,
                                    )
                                if re.search(nested_shift_pattern, new_expr):
                                    new_expr = re.sub(
                                        nested_shift_pattern,
                                        rf"{replacement}.shift(\1)",
                                        new_expr,
                                    )
                            if new_expr == expr:
                                break
                        expr = new_expr
                    # 继续处理下一个变量，不break

            # 步骤15: 处理Mean和Std函数（在变量替换后，确保所有变量都已替换）
            # 处理包含比较操作符或复杂表达式的Mean/Std函数
            # Mean函数 - 处理包含比较操作符的表达式，如 Mean($close>Ref($close, 1), 5)
            # 使用直接的字符串替换方法，确保所有Mean函数都被替换
            def find_matching_paren_simple(s, start_pos):
                count = 0
                i = start_pos
                while i < len(s):
                    if s[i] == "(":
                        count += 1
                    elif s[i] == ")":
                        count -= 1
                        if count == 0:
                            return i
                    i += 1
                return -1

            # 先修复嵌套的calc_data（在Mean函数处理之前）
            var_names = ["close", "open", "high", "low", "volume", "vwap"]
            for var_name in var_names:
                # 修复 calc_data['calc_data['$var']'] 格式（在任何上下文中）
                # 需要处理多种嵌套情况
                nested_patterns = [
                    f"calc_data['calc_data['${var_name}']']",  # 基本嵌套
                    f"calc_data['calc_data['${var_name}']'].shift",  # 带shift的嵌套
                ]
                for nested_pattern in nested_patterns:
                    if nested_pattern in expr:
                        expr = expr.replace(nested_pattern, f"calc_data['${var_name}']")

            # 处理所有Mean函数
            max_mean_iterations = 20
            mean_iteration = 0
            while "Mean(" in expr and mean_iteration < max_mean_iterations:
                mean_positions = [m.start() for m in re.finditer(r"Mean\(", expr)]
                if not mean_positions:
                    break

                # 从后往前处理，避免位置偏移
                replaced_any = False
                for pos in reversed(mean_positions):
                    end_pos = find_matching_paren_simple(expr, pos + 5)
                    if end_pos > 0:
                        comma_pos = expr.rfind(",", pos + 5, end_pos)
                        if comma_pos > 0:
                            inner_expr = expr[pos + 5 : comma_pos].strip()
                            n_str = expr[comma_pos + 1 : end_pos].strip()
                            try:
                                n = int(n_str)
                                # 检查是否是简单的$var格式（已经在步骤6处理过）
                                # 简单格式：calc_data['$var']（不包含操作符或函数调用）
                                inner_expr_stripped = inner_expr.strip()
                                is_simple = re.match(
                                    r"^calc_data\[\'\$\w+\'\]$", inner_expr_stripped
                                )
                                # 如果不是简单格式，或者包含比较操作符、函数调用等，都需要处理
                                # 注意：即使is_simple为True，如果包含比较操作符，也需要处理
                                has_comparison = any(
                                    op in inner_expr_stripped
                                    for op in [">", "<", ">=", "<=", "==", "!="]
                                )
                                if not is_simple or has_comparison:
                                    # 对于复杂表达式（包括比较操作符、函数调用等），使用pd.Series包装
                                    replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).mean()"
                                    expr = (
                                        expr[:pos] + replacement + expr[end_pos + 1 :]
                                    )
                                    replaced_any = True
                                    break
                            except (ValueError, Exception) as e:
                                # 如果解析失败，尝试强制替换
                                try:
                                    n = int(n_str)
                                    replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).mean()"
                                    expr = (
                                        expr[:pos] + replacement + expr[end_pos + 1 :]
                                    )
                                    replaced_any = True
                                    break
                                except:
                                    pass

                if not replaced_any and mean_positions:
                    # 如果无法替换，可能是inner_expr格式问题，尝试强制替换
                    # 找到第一个Mean函数，强制替换
                    pos = mean_positions[0]
                    end_pos = find_matching_paren_simple(expr, pos + 5)
                    if end_pos > 0:
                        comma_pos = expr.rfind(",", pos + 5, end_pos)
                        if comma_pos > 0:
                            inner_expr = expr[pos + 5 : comma_pos].strip()
                            n_str = expr[comma_pos + 1 : end_pos].strip()
                            try:
                                n = int(n_str)
                                # 强制替换，不管格式
                                replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).mean()"
                                expr = expr[:pos] + replacement + expr[end_pos + 1 :]
                                replaced_any = True
                            except ValueError:
                                pass

                if not replaced_any:
                    break
                mean_iteration += 1

            # Std函数 - 处理包含复杂表达式的Std函数
            # 使用直接的字符串替换方法，确保所有Std函数都被替换
            # 处理所有Std函数
            max_std_iterations = 20
            std_iteration = 0
            while "Std(" in expr and std_iteration < max_std_iterations:
                std_positions = [m.start() for m in re.finditer(r"Std\(", expr)]
                if not std_positions:
                    break

                # 从后往前处理，避免位置偏移
                replaced_any = False
                for pos in reversed(std_positions):
                    end_pos = find_matching_paren_simple(expr, pos + 4)
                    if end_pos > 0:
                        comma_pos = expr.rfind(",", pos + 4, end_pos)
                        if comma_pos > 0:
                            inner_expr = expr[pos + 4 : comma_pos].strip()
                            n_str = expr[comma_pos + 1 : end_pos].strip()
                            try:
                                n = int(n_str)
                                # 检查是否是简单的$var格式（已经在步骤6处理过）
                                # 简单格式：calc_data['$var']（不包含操作符或函数调用）
                                inner_expr_stripped = inner_expr.strip()
                                is_simple = re.match(
                                    r"^calc_data\[\'\$\w+\'\]$", inner_expr_stripped
                                )
                                # 如果不是简单格式，或者包含比较操作符、函数调用等，都需要处理
                                # 注意：即使is_simple为True，如果包含比较操作符，也需要处理
                                has_comparison = any(
                                    op in inner_expr_stripped
                                    for op in [">", "<", ">=", "<=", "==", "!="]
                                )
                                if not is_simple or has_comparison:
                                    # 对于复杂表达式，使用pd.Series包装
                                    replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).std()"
                                    expr = (
                                        expr[:pos] + replacement + expr[end_pos + 1 :]
                                    )
                                    replaced_any = True
                                    break
                            except (ValueError, Exception) as e:
                                # 如果解析失败，尝试强制替换
                                try:
                                    n = int(n_str)
                                    replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).std()"
                                    expr = (
                                        expr[:pos] + replacement + expr[end_pos + 1 :]
                                    )
                                    replaced_any = True
                                    break
                                except:
                                    pass

                if not replaced_any and std_positions:
                    # 如果无法替换，可能是inner_expr格式问题，尝试强制替换
                    # 找到第一个Std函数，强制替换
                    pos = std_positions[0]
                    end_pos = find_matching_paren_simple(expr, pos + 4)
                    if end_pos > 0:
                        comma_pos = expr.rfind(",", pos + 4, end_pos)
                        if comma_pos > 0:
                            inner_expr = expr[pos + 4 : comma_pos].strip()
                            n_str = expr[comma_pos + 1 : end_pos].strip()
                            try:
                                n = int(n_str)
                                # 强制替换，不管格式
                                replacement = f"pd.Series({inner_expr}, index=calc_data.index).rolling({n}).std()"
                                expr = expr[:pos] + replacement + expr[end_pos + 1 :]
                                replaced_any = True
                            except ValueError:
                                pass

                if not replaced_any:
                    break
                std_iteration += 1

            # 在评估前，最后检查并修复Mean/Std函数（确保所有都被替换）
            # 如果还有Mean/Std函数，强制替换
            if "Mean(" in expr or "Std(" in expr:
                logger.debug(f"表达式评估前仍有未替换的Mean/Std函数，交由运行时函数处理: {expr[:200]}...")

            # 评估表达式
            result = eval(
                expr,
                {
                    "np": np,
                    "pd": pd,
                    "calc_data": calc_data,
                    "Greater": Greater,
                    "Less": Less,
                    "Sum": Sum,
                    "Mean": Mean,
                    "Std": Std,
                    "Max": Max,
                    "Min": Min,
                },
            )

            # 如果是Series，直接返回；如果是DataFrame，取最后一列或第一列
            if isinstance(result, pd.DataFrame):
                if len(result.columns) == 1:
                    return result.iloc[:, 0]
                else:
                    return result.iloc[:, -1]
            elif isinstance(result, pd.Series):
                return result
            else:
                # 如果是标量或其他类型，尝试转换为Series
                return pd.Series([result] * len(calc_data), index=calc_data.index)

        except Exception as e:
            # 记录失败信息（限制数量避免日志过多）
            error_msg = str(e)
            # 只记录前几个详细错误，其他只记录简要信息
            if not hasattr(self, "_detailed_error_count"):
                self._detailed_error_count = 0
            if self._detailed_error_count < 5:
                logger.warning(f"表达式评估失败: {expression[:100]}... 错误: {error_msg}")
                logger.debug(f"失败表达式(转换后): {expr[:200]}...")
                logger.debug(
                    f"可用列: {list(calc_data.columns)[:20]}{'...' if len(calc_data.columns) > 20 else ''}"
                )
                import traceback

                logger.debug(traceback.format_exc())
                self._detailed_error_count += 1
            return None

    async def _calculate_simplified_alpha_factors(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """计算简化版Alpha因子"""
        if data.empty:
            return pd.DataFrame()

        # 确保数据有正确的列
        required_cols = ["$close", "$high", "$low", "$volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"缺少必要列: {missing_cols}")
            return pd.DataFrame(index=data.index)

        factors = pd.DataFrame(index=data.index)

        try:
            # 价格收益率因子
            for period in [5, 10, 20, 30]:
                factors[f"RESI{period}"] = data["$close"].pct_change(periods=period)

            # 移动平均因子
            for period in [5, 10, 20, 30]:
                factors[f"MA{period}"] = data["$close"].rolling(period).mean()

            # 标准差因子
            for period in [5, 10, 20, 30]:
                factors[f"STD{period}"] = data["$close"].rolling(period).std()

            # 成交量标准差因子
            for period in [5, 10, 20, 30]:
                factors[f"VSTD{period}"] = data["$volume"].rolling(period).std()

            # 相关性因子（价格与成交量）
            for period in [5, 10, 20, 30]:
                factors[f"CORR{period}"] = (
                    data["$close"].rolling(period).corr(data["$volume"])
                )

            # 最高价因子
            for period in [5, 10, 20, 30]:
                factors[f"MAX{period}"] = data["$high"].rolling(period).max()

            # 最低价因子
            for period in [5, 10, 20, 30]:
                factors[f"MIN{period}"] = data["$low"].rolling(period).min()

            # 分位数比率因子
            for period in [5, 10, 20, 30]:
                q80 = data["$close"].rolling(period).quantile(0.8)
                q20 = data["$close"].rolling(period).quantile(0.2)
                factors[f"QTLU{period}"] = q80 / (q20 + 1e-8)  # 避免除零

            # 填充无穷大和NaN值
            factors = factors.replace([np.inf, -np.inf], np.nan)
            factors = factors.ffill().fillna(0)

            logger.debug(f"计算了 {len(factors.columns)} 个Alpha因子")
            return factors

        except Exception as e:
            logger.error(f"简化Alpha因子计算失败: {e}")
            return pd.DataFrame(index=data.index)


class EnhancedQlibDataProvider:
    """增强版Qlib数据提供器"""

    def __init__(self, data_service: Optional[SimpleDataService] = None):
        self.data_service = data_service or SimpleDataService()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.alpha_calculator = Alpha158Calculator()

        # Qlib初始化状态
        self._qlib_initialized = False

        logger.info("增强版Qlib数据提供器初始化完成")

    async def initialize_qlib(self):
        """初始化Qlib环境"""
        global _QLIB_GLOBAL_INITIALIZED

        # 如果全局已初始化，直接返回
        if _QLIB_GLOBAL_INITIALIZED or not QLIB_AVAILABLE:
            self._qlib_initialized = True
            return

        try:
            # 使用本地路径作为 provider_uri，避免 Qlib 将 ":" 拼进 data_path 导致日历路径出错
            # 使用配置中的QLIB_DATA_PATH，如果不存在则创建
            qlib_data_path = Path(settings.QLIB_DATA_PATH).resolve()
            qlib_data_path.mkdir(parents=True, exist_ok=True)

            # 确保交易日历文件存在（Qlib Alpha158 handler需要）
            try:
                from app.services.data.qlib_calendar_generator import (
                    QlibCalendarGenerator,
                )

                calendar_generator = QlibCalendarGenerator()
                calendar_generator.ensure_calendar_exists()
            except Exception as cal_error:
                logger.warning(f"生成交易日历文件失败: {cal_error}，Alpha158 handler可能无法使用")

            # 准备mount_path和provider_uri配置
            # qlib.init()内部会调用C.set()重置配置，所以需要通过参数传递
            # 确保所有路径都是绝对路径的字符串，避免Path对象传递
            # 注意：使用 .as_posix() 确保使用正斜杠，避免路径转义字符问题
            # Qlib 内部可能会在路径拼接时出现问题，使用 POSIX 格式可以避免
            qlib_data_path_str = qlib_data_path.resolve().as_posix()
            mount_path_config = {
                "day": qlib_data_path_str,
                "1min": qlib_data_path_str,
            }

            # 记录路径信息用于调试
            logger.debug(f"Qlib mount_path配置: {mount_path_config}")
            logger.debug(f"交易日历文件路径: {qlib_data_path_str}/calendars/day.txt")

            provider_uri_config = {
                "day": qlib_data_path_str,
                "1min": qlib_data_path_str,
            }

            # 在调用 qlib.init() 之前，尝试修复 C.dpm 配置中的 Path 对象
            # 这是一个临时解决方案，用于处理 qlib 内部配置系统的 Path 对象问题
            try:
                # 如果 C.dpm 存在且包含 Path 对象，尝试转换为字符串
                if hasattr(C, "dpm") and hasattr(C.dpm, "get_data_uri"):
                    # 获取所有频率的数据 URI，并确保它们是字符串
                    for freq in ["day", "1min"]:
                        try:
                            data_uri = C.dpm.get_data_uri(freq)
                            if isinstance(data_uri, (list, tuple)):
                                # 如果返回列表，确保所有元素都是字符串
                                data_uri = [
                                    str(item) if isinstance(item, Path) else item
                                    for item in data_uri
                                ]
                        except Exception:
                            pass
            except Exception:
                pass  # 忽略配置修复错误

            # 通过kwargs传递配置，避免被C.set()重置
            # 注意：provider_uri作为字典传递时，会覆盖字符串形式的provider_uri
            # 设置 auto_mount=False 避免 qlib 内部处理 NFS mount 时的 Path 对象问题
            qlib.init(
                region=REG_CN,
                provider_uri=provider_uri_config,
                mount_path=mount_path_config,
                auto_mount=False,
            )

            # 修复日历文件路径问题
            # Qlib 内部可能会在路径拼接时出现问题（出现 :\/ 或路径末尾的 :\）
            # 这通常是因为 Qlib 内部使用了错误的路径拼接方式
            # 我们需要在初始化后立即修复 C.dpm 中的路径配置
            try:
                from qlib.config import C

                calendar_file = Path(qlib_data_path_str) / "calendars" / "day.txt"
                if calendar_file.exists():
                    logger.debug(f"交易日历文件已确认存在: {calendar_file}")

                # 强制修复 C.dpm.data_path（无论日历文件是否存在）
                if hasattr(C, "dpm") and hasattr(C.dpm, "data_path"):
                    data_path = C.dpm.data_path
                    if isinstance(data_path, dict):
                        # 清理路径的函数
                        def clean_path_value(path_val):
                            """清理路径值"""
                            if isinstance(path_val, Path):
                                path_str = str(path_val)
                            elif isinstance(path_val, str):
                                path_str = path_val
                            else:
                                return path_val
                            # 清理路径末尾的异常字符
                            fixed = (
                                path_str.rstrip(":\\")
                                .rstrip(":/")
                                .rstrip("\\")
                                .rstrip("/")
                            )
                            # 如果路径中有 :\/，替换为 /
                            fixed = fixed.replace(":\/", "/").replace(":\\/", "/")
                            return fixed

                        # 修复字典中的每个路径
                        fixed_data_path = {}
                        needs_fix = False
                        for freq, path_val in data_path.items():
                            path_str = str(path_val)
                            fixed_path = clean_path_value(path_val)
                            fixed_data_path[freq] = fixed_path
                            # 检查是否需要修复
                            if (
                                ":\\" in path_str
                                or ":\/" in path_str
                                or path_str.endswith(":\\")
                                or path_str.endswith(":/")
                            ):
                                needs_fix = True
                                logger.info(
                                    f"检测到 data_path[{freq}] 路径问题: {repr(path_str)} -> {fixed_path}"
                                )

                        # 如果检测到问题，强制修复
                        if needs_fix:
                            # 方法1: 直接设置
                            try:
                                C.dpm.data_path = fixed_data_path
                                logger.info(f"✓ 已修复 C.dpm.data_path: {fixed_data_path}")
                            except Exception as set_error:
                                # 方法2: 通过 __dict__ 设置
                                try:
                                    if hasattr(C.dpm, "__dict__"):
                                        C.dpm.__dict__["data_path"] = fixed_data_path
                                        logger.info(f"✓ 通过 __dict__ 修复 data_path")
                                    else:
                                        # 方法3: 通过 setattr
                                        setattr(C.dpm, "data_path", fixed_data_path)
                                        logger.info(f"✓ 通过 setattr 修复 data_path")
                                except Exception as set_error2:
                                    logger.warning(
                                        f"✗ 无法修复 data_path: {set_error}, {set_error2}"
                                    )
                                    # 方法4: 尝试替换整个 dpm 对象（最后的手段）
                                    try:
                                        # 创建一个新的 DataPathManager 对象（如果可能）
                                        logger.warning("尝试其他方法修复 data_path...")
                                    except Exception:
                                        pass
                        else:
                            logger.debug("data_path 路径格式正确，无需修复")
            except Exception as cal_setup_error:
                logger.warning(f"修复 data_path 时出错: {cal_setup_error}，但不影响主要功能")
            _QLIB_GLOBAL_INITIALIZED = True
            self._qlib_initialized = True
            logger.info("Qlib环境初始化成功")
        except TypeError as e:
            # 处理 qlib 内部的 Path 对象问题
            if "expected str instance, PosixPath found" in str(e):
                # 这是一个已知的 qlib bug，与 C.dpm.get_data_uri() 返回 Path 对象有关
                # 尝试使用 monkey patch 或绕过初始化
                logger.warning(f"检测到 qlib Path 对象问题，尝试替代初始化方式: {e}")
                try:
                    # 尝试不传递 mount_path，让 qlib 使用默认值
                    qlib.init(
                        region=REG_CN,
                        provider_uri=str(qlib_data_path.resolve()),
                        auto_mount=False,
                    )
                    _QLIB_GLOBAL_INITIALIZED = True
                    self._qlib_initialized = True
                    logger.info("Qlib环境初始化成功（使用替代方式）")
                except Exception as e2:
                    logger.error(f"替代初始化方式也失败: {e2}")
                    raise e  # 抛出原始错误
            else:
                raise
        except Exception as e:
            error_msg = str(e)
            # 如果Qlib已经初始化，忽略错误并标记为已初始化
            if (
                "reinitialize" in error_msg.lower()
                or "already activated" in error_msg.lower()
            ):
                logger.warning(f"Qlib已经初始化，跳过重新初始化: {error_msg}")
                _QLIB_GLOBAL_INITIALIZED = True
                self._qlib_initialized = True
                return
            logger.error(f"Qlib初始化失败: {e}")
            raise

    async def prepare_qlib_dataset(
        self,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        include_alpha_factors: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """准备Qlib标准格式的数据集"""
        logger.info(
            f"准备Qlib数据集: {len(stock_codes)} 只股票, Alpha因子: {include_alpha_factors}"
        )

        # 1. 获取基础特征数据
        base_features = await self._prepare_base_features(
            stock_codes, start_date, end_date
        )

        if base_features.empty:
            logger.warning("基础特征数据为空")
            return pd.DataFrame()

        # 2. 转换为Qlib标准格式
        qlib_data = self._convert_to_qlib_format(base_features)

        # 3. 计算Alpha158因子（如果需要）
        if include_alpha_factors and QLIB_AVAILABLE:
            try:
                alpha_factors = await self.alpha_calculator.calculate_alpha_factors(
                    qlib_data, stock_codes, (start_date, end_date), use_cache
                )

                if not alpha_factors.empty:
                    # 合并Alpha因子
                    qlib_data = pd.concat([qlib_data, alpha_factors], axis=1)
                    logger.info(f"成功添加 {len(alpha_factors.columns)} 个Alpha因子")
            except Exception as e:
                logger.error(f"Alpha因子计算失败: {e}")

        logger.info(f"========== Qlib数据集准备完成 ==========")
        logger.info(f"记录数: {len(qlib_data)}")
        logger.info(f"特征数: {len(qlib_data.columns)}")
        logger.info(f"数据集形状: {qlib_data.shape}")
        logger.info(f"数据维度数: {qlib_data.ndim}")
        logger.info(f"索引类型: {type(qlib_data.index).__name__}")
        if isinstance(qlib_data.index, pd.MultiIndex):
            logger.info(f"MultiIndex级别数: {qlib_data.index.nlevels}")
            logger.info(f"MultiIndex级别名称: {qlib_data.index.names}")
        logger.info(
            f"特征列表: {list(qlib_data.columns[:20])}{'...' if len(qlib_data.columns) > 20 else ''}"
        )
        logger.info(f"缺失值总数: {qlib_data.isnull().sum().sum()}")
        logger.info(f"数据类型统计: {qlib_data.dtypes.value_counts().to_dict()}")
        logger.info(f"==========================================")
        if not qlib_data.empty:
            logger.info(
                f"数据统计: 缺失值={qlib_data.isnull().sum().sum()}, 数据类型={qlib_data.dtypes.value_counts().to_dict()}"
            )
        return qlib_data

    async def _prepare_base_features(
        self, stock_codes: List[str], start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """准备基础特征数据"""
        all_features = []

        for stock_code in stock_codes:
            try:
                # 优先从本地文件加载数据
                from app.core.config import settings
                from app.services.data.stock_data_loader import StockDataLoader

                loader = StockDataLoader(data_root=settings.DATA_ROOT_PATH)
                stock_data = loader.load_stock_data(
                    stock_code, start_date=start_date, end_date=end_date
                )

                # 如果本地没有数据，尝试从远端服务获取
                if stock_data.empty:
                    logger.info(f"本地无数据，从远端服务获取: {stock_code}")
                    stock_data_list = await self.data_service.get_stock_data(
                        stock_code, start_date, end_date
                    )

                    if not stock_data_list or len(stock_data_list) == 0:
                        logger.warning(f"股票 {stock_code} 无数据")
                        continue

                    # 转换为DataFrame格式
                    stock_data = pd.DataFrame(
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
                    stock_data = stock_data.set_index("date")

                # 确保数据有正确的列名
                if stock_data.empty:
                    logger.warning(f"股票 {stock_code} 无数据")
                    continue

                # 计算技术指标
                indicators = await self.indicator_calculator.calculate_all_indicators(
                    stock_data
                )

                # 合并数据
                if not indicators.empty:
                    features = stock_data.merge(
                        indicators, left_index=True, right_index=True, how="left"
                    )
                else:
                    features = stock_data.copy()

                features["stock_code"] = stock_code

                # 确保有date列
                if "date" not in features.columns and isinstance(
                    features.index, pd.DatetimeIndex
                ):
                    features = features.reset_index()
                    features.rename(columns={"index": "date"}, inplace=True)
                elif "date" not in features.columns:
                    features = features.reset_index()
                    if "date" not in features.columns:
                        features["date"] = features.index

                # 添加基本面特征
                features = self._add_fundamental_features(features)

                all_features.append(features)

            except Exception as e:
                logger.error(f"处理股票 {stock_code} 特征时出错: {e}")
                continue

        if not all_features:
            logger.warning("没有成功处理任何股票数据")
            return pd.DataFrame()

        # 合并所有股票的特征
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_features = combined_features.sort_values(["stock_code", "date"])

        logger.info(f"基础特征准备完成: {len(stock_codes)} 只股票, {len(combined_features)} 条记录")
        return combined_features

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
        """处理缺失值"""
        df_filled = df.copy()

        # 基础价格数据：前向填充
        price_cols = ["$open", "$high", "$low", "$close", "$volume"]
        for col in price_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].ffill()

        # 技术指标：使用0填充（因为计算窗口不足时为NaN是正常的）
        indicator_cols = [col for col in df_filled.columns if col not in price_cols]
        for col in indicator_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(0)

        # 记录缺失值处理情况
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.debug(f"处理缺失值: {missing_counts[missing_counts > 0].to_dict()}")

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
        """创建Qlib模型配置"""
        base_config = {
            "class": "LGBModel",  # 默认使用LightGBM
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        }

        # 根据模型类型调整配置
        if model_type.lower() == "lightgbm":
            base_config["class"] = "LGBModel"
            base_config["module_path"] = "qlib.contrib.model.gbdt"
        elif model_type.lower() == "xgboost":
            base_config["class"] = "XGBModel"
            base_config["module_path"] = "qlib.contrib.model.xgboost"
        elif model_type.lower() == "mlp":
            base_config["class"] = "DNNModelPytorch"
            base_config["module_path"] = "qlib.contrib.model.pytorch_nn"

        # 合并用户提供的超参数
        if hyperparameters:
            base_config["kwargs"].update(hyperparameters)

        return base_config

    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            cache_dir = self.alpha_calculator.factor_cache.cache_dir
            cache_files = list(cache_dir.glob("*.parquet"))

            total_size = sum(f.stat().st_size for f in cache_files)

            return {
                "cache_files": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "cache_dir": str(cache_dir),
                "qlib_available": QLIB_AVAILABLE,
                "qlib_initialized": self._qlib_initialized,
            }
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {
                "cache_files": 0,
                "total_size_mb": 0,
                "cache_dir": "unknown",
                "qlib_available": QLIB_AVAILABLE,
                "qlib_initialized": self._qlib_initialized,
            }

    async def clear_cache(self):
        """清空缓存"""
        try:
            cache_dir = self.alpha_calculator.factor_cache.cache_dir
            cache_files = list(cache_dir.glob("*.parquet"))

            for cache_file in cache_files:
                cache_file.unlink()

            logger.info(f"清空缓存完成，删除 {len(cache_files)} 个文件")
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            raise

    async def get_qlib_model_predictions(
        self,
        model_config: Dict[str, Any],
        dataset: pd.DataFrame,
        prediction_horizon: int = 5,
    ) -> pd.DataFrame:
        """使用Qlib模型进行预测"""
        if not QLIB_AVAILABLE:
            logger.warning("Qlib不可用，无法进行预测")
            return pd.DataFrame()

        try:
            await self.initialize_qlib()

            # 创建Qlib模型实例
            from qlib.utils import init_instance_by_config

            model = init_instance_by_config(model_config)

            # 训练模型（这里应该使用已训练的模型，但为了演示先快速训练）
            logger.info("开始Qlib模型训练...")
            model.fit(dataset)

            # 进行预测
            logger.info("开始Qlib模型预测...")
            predictions = model.predict(dataset)

            # 转换预测结果格式
            if isinstance(predictions, pd.Series):
                predictions = predictions.to_frame("prediction")
            elif not isinstance(predictions, pd.DataFrame):
                predictions = pd.DataFrame(predictions, columns=["prediction"])

            logger.info(f"Qlib预测完成: {len(predictions)} 条预测结果")
            return predictions

        except Exception as e:
            logger.error(f"Qlib模型预测失败: {e}")
            return pd.DataFrame()

    async def validate_and_fix_qlib_format(
        self, data: pd.DataFrame
    ) -> Tuple[bool, pd.DataFrame]:
        """验证并修复Qlib数据格式"""
        try:
            logger.info("开始验证和修复Qlib数据格式")

            # 1. 基本格式检查
            if data.empty:
                logger.warning("数据为空")
                return False, data

            # 2. 索引格式检查和修复
            if not isinstance(data.index, pd.MultiIndex):
                logger.info("修复MultiIndex格式")
                data = self._ensure_multiindex_format(data)

            # 3. 索引层级检查
            if len(data.index.names) != 2:
                logger.warning(f"索引层级数不正确: {len(data.index.names)}")
                return False, data

            # 4. 必要列检查
            required_cols = ["$open", "$high", "$low", "$close", "$volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                logger.warning(f"缺少必要的列: {missing_cols}")
                # 尝试从其他列推导
                data = self._fix_missing_columns(data, missing_cols)

                # 再次检查
                still_missing = [
                    col for col in required_cols if col not in data.columns
                ]
                if still_missing:
                    logger.error(f"无法修复缺少的列: {still_missing}")
                    return False, data

            # 5. 数据类型检查和修复
            data = self._fix_data_types(data)

            # 6. 数据质量检查
            quality_issues = self._check_data_quality(data)
            if quality_issues:
                logger.warning(f"数据质量问题: {quality_issues}")
                data = self._fix_data_quality_issues(data, quality_issues)

            # 7. 最终验证
            is_valid = await self.validate_qlib_data_format(data)

            logger.info(f"Qlib格式验证和修复完成: 有效={is_valid}, 数据形状={data.shape}")
            return is_valid, data

        except Exception as e:
            logger.error(f"Qlib格式验证和修复失败: {e}")
            return False, data

    def _fix_missing_columns(
        self, data: pd.DataFrame, missing_cols: List[str]
    ) -> pd.DataFrame:
        """修复缺少的列"""
        data_fixed = data.copy()

        # 尝试从相似列名推导
        column_alternatives = {
            "$open": ["open", "Open", "OPEN"],
            "$high": ["high", "High", "HIGH"],
            "$low": ["low", "Low", "LOW"],
            "$close": ["close", "Close", "CLOSE", "adj_close", "Adj_Close"],
            "$volume": ["volume", "Volume", "VOLUME", "vol", "Vol"],
        }

        for missing_col in missing_cols:
            if missing_col in column_alternatives:
                alternatives = column_alternatives[missing_col]
                for alt in alternatives:
                    if alt in data_fixed.columns:
                        data_fixed[missing_col] = data_fixed[alt]
                        logger.info(f"从 {alt} 推导出 {missing_col}")
                        break
                else:
                    # 如果无法推导，使用默认值
                    if missing_col == "$volume":
                        data_fixed[missing_col] = 1000000  # 默认成交量
                    else:
                        # 对于价格列，使用close价格作为默认值
                        if "$close" in data_fixed.columns:
                            data_fixed[missing_col] = data_fixed["$close"]
                        else:
                            data_fixed[missing_col] = 100.0  # 默认价格

                    logger.warning(f"使用默认值填充 {missing_col}")

        return data_fixed

    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """修复数据类型"""
        data_fixed = data.copy()

        # 价格列应该是数值类型
        price_cols = ["$open", "$high", "$low", "$close"]
        for col in price_cols:
            if col in data_fixed.columns:
                data_fixed[col] = pd.to_numeric(data_fixed[col], errors="coerce")

        # 成交量应该是整数类型
        if "$volume" in data_fixed.columns:
            data_fixed["$volume"] = pd.to_numeric(
                data_fixed["$volume"], errors="coerce"
            )
            data_fixed["$volume"] = data_fixed["$volume"].fillna(0).astype("int64")

        # 其他数值列
        for col in data_fixed.columns:
            if col not in price_cols + ["$volume"]:
                if data_fixed[col].dtype == "object":
                    data_fixed[col] = pd.to_numeric(data_fixed[col], errors="coerce")

        return data_fixed

    def _check_data_quality(self, data: pd.DataFrame) -> List[str]:
        """检查数据质量问题"""
        issues = []

        # 检查价格逻辑
        if all(col in data.columns for col in ["$open", "$high", "$low", "$close"]):
            # 最高价应该 >= 最低价
            invalid_high_low = (data["$high"] < data["$low"]).sum()
            if invalid_high_low > 0:
                issues.append(f"high < low: {invalid_high_low} 条记录")

            # 价格应该为正数
            for col in ["$open", "$high", "$low", "$close"]:
                negative_prices = (data[col] <= 0).sum()
                if negative_prices > 0:
                    issues.append(f"{col} <= 0: {negative_prices} 条记录")

        # 检查成交量
        if "$volume" in data.columns:
            negative_volume = (data["$volume"] < 0).sum()
            if negative_volume > 0:
                issues.append(f"负成交量: {negative_volume} 条记录")

        # 检查缺失值
        missing_counts = data.isnull().sum()
        critical_missing = missing_counts[missing_counts > len(data) * 0.1]  # 超过10%缺失
        if not critical_missing.empty:
            issues.append(f"高缺失率列: {critical_missing.to_dict()}")

        return issues

    def _fix_data_quality_issues(
        self, data: pd.DataFrame, issues: List[str]
    ) -> pd.DataFrame:
        """修复数据质量问题"""
        data_fixed = data.copy()

        # 修复价格逻辑问题
        if all(
            col in data_fixed.columns for col in ["$open", "$high", "$low", "$close"]
        ):
            # 修复 high < low 的问题
            invalid_mask = data_fixed["$high"] < data_fixed["$low"]
            if invalid_mask.sum() > 0:
                # 交换high和low
                data_fixed.loc[invalid_mask, ["$high", "$low"]] = data_fixed.loc[
                    invalid_mask, ["$low", "$high"]
                ].values
                logger.info(f"修复了 {invalid_mask.sum()} 条 high < low 的记录")

            # 修复负价格
            for col in ["$open", "$high", "$low", "$close"]:
                negative_mask = data_fixed[col] <= 0
                if negative_mask.sum() > 0:
                    # 使用前一个有效值填充
                    data_fixed.loc[negative_mask, col] = data_fixed[col].ffill()
                    # 如果还有负值，使用均值
                    still_negative = data_fixed[col] <= 0
                    if still_negative.sum() > 0:
                        mean_price = data_fixed[col][data_fixed[col] > 0].mean()
                        data_fixed.loc[still_negative, col] = mean_price
                    logger.info(f"修复了 {negative_mask.sum()} 条 {col} <= 0 的记录")

        # 修复负成交量
        if "$volume" in data_fixed.columns:
            negative_mask = data_fixed["$volume"] < 0
            if negative_mask.sum() > 0:
                data_fixed.loc[negative_mask, "$volume"] = 0
                logger.info(f"修复了 {negative_mask.sum()} 条负成交量记录")

        # 处理高缺失率列
        missing_counts = data_fixed.isnull().sum()
        high_missing_cols = missing_counts[
            missing_counts > len(data_fixed) * 0.5
        ].index  # 超过50%缺失

        for col in high_missing_cols:
            if col in ["$open", "$high", "$low", "$close"]:
                # 价格列使用前向填充
                data_fixed[col] = data_fixed[col].ffill().bfill()
            elif col == "$volume":
                # 成交量使用0填充
                data_fixed[col] = data_fixed[col].fillna(0)
            else:
                # 其他列使用0填充
                data_fixed[col] = data_fixed[col].fillna(0)

        return data_fixed

    async def validate_qlib_data_format(self, data: pd.DataFrame) -> bool:
        try:
            # 检查索引格式
            if not isinstance(data.index, pd.MultiIndex):
                logger.warning("数据索引不是MultiIndex格式")
                return False

            # 检查索引层级名称
            if len(data.index.names) != 2:
                logger.warning("数据索引应该有两个层级")
                return False

            # 检查必要的列
            required_cols = ["$close", "$high", "$low", "$open", "$volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"缺少必要的列: {missing_cols}")
                return False

            # 检查数据类型
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    logger.warning(f"列 {col} 不是数值类型")
                    return False

            logger.info("Qlib数据格式验证通过")
            return True

        except Exception as e:
            logger.error(f"Qlib数据格式验证失败: {e}")
            return False

    async def convert_dataframe_to_qlib(
        self, df: pd.DataFrame, validate: bool = True, fix_issues: bool = True
    ) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:
        """
        将DataFrame转换为Qlib格式的主要接口

        Args:
            df: 输入的DataFrame
            validate: 是否验证格式
            fix_issues: 是否自动修复问题

        Returns:
            (is_valid, converted_df, conversion_info)
        """
        conversion_info = {
            "input_shape": df.shape,
            "input_columns": list(df.columns),
            "conversion_steps": [],
            "issues_found": [],
            "issues_fixed": [],
        }

        try:
            logger.info(f"开始Qlib格式转换: 输入 {df.shape}")

            # 1. 基本转换
            converted_df = self._convert_to_qlib_format(df)
            conversion_info["conversion_steps"].append("基本格式转换")
            conversion_info["output_shape"] = converted_df.shape
            conversion_info["output_columns"] = list(converted_df.columns)

            # 2. 验证和修复（如果需要）
            if validate or fix_issues:
                is_valid, fixed_df = await self.validate_and_fix_qlib_format(
                    converted_df
                )
                conversion_info["is_valid_before_fix"] = is_valid

                if fix_issues and not is_valid:
                    converted_df = fixed_df
                    conversion_info["conversion_steps"].append("问题修复")

                    # 再次验证
                    is_valid, _ = await self.validate_and_fix_qlib_format(converted_df)
                    conversion_info["is_valid_after_fix"] = is_valid
            else:
                is_valid = True

            # 3. 最终统计
            conversion_info["final_shape"] = converted_df.shape
            conversion_info["final_columns"] = list(converted_df.columns)
            conversion_info["memory_usage_mb"] = (
                converted_df.memory_usage(deep=True).sum() / 1024 / 1024
            )

            logger.info(f"Qlib格式转换完成: {conversion_info['final_shape']}, 有效={is_valid}")
            return is_valid, converted_df, conversion_info

        except Exception as e:
            logger.error(f"Qlib格式转换失败: {e}")
            conversion_info["error"] = str(e)
            return False, df, conversion_info

    async def get_qlib_format_example(self) -> Dict[str, Any]:
        """获取Qlib格式示例和说明"""
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
