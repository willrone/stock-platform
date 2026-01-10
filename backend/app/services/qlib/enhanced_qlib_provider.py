"""
增强版Qlib数据提供器

基于现有QlibDataProvider，添加Alpha158因子计算和缓存机制
"""

import asyncio
import hashlib
import json
from loguru import logger
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd

# 检测Qlib可用性
try:
    import qlib
    from qlib.config import REG_CN, C
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.filter import NameDFilter, ExpressionDFilter
    from qlib.utils import init_instance_by_config
    QLIB_AVAILABLE = True
except ImportError:
    logger.warning("Qlib未安装，某些功能将不可用")
    QLIB_AVAILABLE = False

from ..data.simple_data_service import SimpleDataService
from ..prediction.technical_indicators import TechnicalIndicatorCalculator
from ...core.config import settings

# 全局Qlib初始化状态（跨实例共享）
_QLIB_GLOBAL_INITIALIZED = False


class FactorCache:
    """因子计算结果缓存"""
    
    def __init__(self, cache_dir: str = "./data/qlib_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存配置
        self.max_cache_size = 50  # 最大缓存文件数
        self.default_ttl = timedelta(hours=24)  # 默认缓存过期时间
        
        logger.info(f"因子缓存初始化: {self.cache_dir}")
    
    def get_cache_key(self, stock_codes: List[str], date_range: Tuple[datetime, datetime]) -> str:
        """生成缓存键"""
        codes_hash = hashlib.md5('_'.join(sorted(stock_codes)).encode()).hexdigest()[:8]
        start_str = date_range[0].strftime('%Y%m%d')
        end_str = date_range[1].strftime('%Y%m%d')
        return f"alpha_{codes_hash}_{start_str}_{end_str}"
    
    def get_cached_factors(self, cache_key: str) -> Optional[pd.DataFrame]:
        """获取缓存的因子数据"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            try:
                # 检查文件是否过期
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - file_time > self.default_ttl:
                    logger.debug(f"缓存文件已过期: {cache_key}")
                    cache_file.unlink()
                    return None
                
                factors = pd.read_parquet(cache_file)
                logger.info(f"命中因子缓存: {cache_key}, 数据量: {len(factors)}")
                return factors
            except Exception as e:
                logger.warning(f"读取因子缓存失败: {e}")
                # 删除损坏的缓存文件
                try:
                    cache_file.unlink()
                except:
                    pass
        return None
    
    def save_factors(self, cache_key: str, factors: pd.DataFrame):
        """保存因子数据到缓存"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            factors.to_parquet(cache_file)
            
            # 清理旧缓存
            self._cleanup_old_cache()
            
            logger.info(f"因子数据缓存成功: {cache_key}, 数据量: {len(factors)}")
        except Exception as e:
            logger.warning(f"保存因子缓存失败: {e}")
    
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


class Alpha158Calculator:
    """Alpha158因子计算器"""
    
    def __init__(self):
        self.factor_cache = FactorCache()
        
        # Alpha158因子表达式（简化版本）
        self.alpha_expressions = {
            # 价格相关因子
            'RESI5': '($close-Ref($close,5))/Ref($close,5)',
            'RESI10': '($close-Ref($close,10))/Ref($close,10)',
            'RESI20': '($close-Ref($close,20))/Ref($close,20)',
            'RESI30': '($close-Ref($close,30))/Ref($close,30)',
            
            # 移动平均因子
            'MA5': 'Mean($close,5)',
            'MA10': 'Mean($close,10)',
            'MA20': 'Mean($close,20)',
            'MA30': 'Mean($close,30)',
            
            # 标准差因子
            'STD5': 'Std($close,5)',
            'STD10': 'Std($close,10)',
            'STD20': 'Std($close,20)',
            'STD30': 'Std($close,30)',
            
            # 成交量因子
            'VSTD5': 'Std($volume,5)',
            'VSTD10': 'Std($volume,10)',
            'VSTD20': 'Std($volume,20)',
            'VSTD30': 'Std($volume,30)',
            
            # 相关性因子
            'CORR5': 'Corr($close,$volume,5)',
            'CORR10': 'Corr($close,$volume,10)',
            'CORR20': 'Corr($close,$volume,20)',
            'CORR30': 'Corr($close,$volume,30)',
            
            # 最高最低价因子
            'MAX5': 'Max($high,5)',
            'MAX10': 'Max($high,10)',
            'MAX20': 'Max($high,20)',
            'MAX30': 'Max($high,30)',
            
            'MIN5': 'Min($low,5)',
            'MIN10': 'Min($low,10)',
            'MIN20': 'Min($low,20)',
            'MIN30': 'Min($low,30)',
            
            # 量价比因子
            'QTLU5': 'Quantile($close,5,0.8)/Quantile($close,5,0.2)',
            'QTLU10': 'Quantile($close,10,0.8)/Quantile($close,10,0.2)',
            'QTLU20': 'Quantile($close,20,0.8)/Quantile($close,20,0.2)',
            'QTLU30': 'Quantile($close,30,0.8)/Quantile($close,30,0.2)',
        }
        
        logger.info(f"Alpha158计算器初始化，支持 {len(self.alpha_expressions)} 个因子")
    
    async def calculate_alpha_factors(
        self,
        qlib_data: pd.DataFrame,
        stock_codes: List[str],
        date_range: Tuple[datetime, datetime],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """计算Alpha158因子"""
        if not QLIB_AVAILABLE:
            logger.warning("Qlib不可用，跳过Alpha因子计算")
            return pd.DataFrame(index=qlib_data.index)
        
        # 尝试从缓存获取
        if use_cache:
            cache_key = self.factor_cache.get_cache_key(stock_codes, date_range)
            cached_factors = self.factor_cache.get_cached_factors(cache_key)
            if cached_factors is not None:
                return cached_factors
        
        try:
            logger.info(f"开始计算Alpha158因子: {len(stock_codes)} 只股票")
            
            # 使用简化的因子计算（不依赖Qlib的复杂表达式引擎）
            alpha_factors = await self._calculate_simplified_alpha_factors(qlib_data)
            
            # 缓存结果
            if use_cache and not alpha_factors.empty:
                cache_key = self.factor_cache.get_cache_key(stock_codes, date_range)
                self.factor_cache.save_factors(cache_key, alpha_factors)
            
            logger.info(f"Alpha158因子计算完成: {len(alpha_factors)} 条记录, {len(alpha_factors.columns)} 个因子")
            return alpha_factors
            
        except Exception as e:
            logger.error(f"Alpha因子计算失败: {e}")
            return pd.DataFrame(index=qlib_data.index)
    
    async def _calculate_simplified_alpha_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算简化版Alpha因子"""
        if data.empty:
            return pd.DataFrame()
        
        # 确保数据有正确的列
        required_cols = ['$close', '$high', '$low', '$volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"缺少必要列: {missing_cols}")
            return pd.DataFrame(index=data.index)
        
        factors = pd.DataFrame(index=data.index)
        
        try:
            # 价格收益率因子
            for period in [5, 10, 20, 30]:
                factors[f'RESI{period}'] = data['$close'].pct_change(periods=period)
            
            # 移动平均因子
            for period in [5, 10, 20, 30]:
                factors[f'MA{period}'] = data['$close'].rolling(period).mean()
            
            # 标准差因子
            for period in [5, 10, 20, 30]:
                factors[f'STD{period}'] = data['$close'].rolling(period).std()
            
            # 成交量标准差因子
            for period in [5, 10, 20, 30]:
                factors[f'VSTD{period}'] = data['$volume'].rolling(period).std()
            
            # 相关性因子（价格与成交量）
            for period in [5, 10, 20, 30]:
                factors[f'CORR{period}'] = data['$close'].rolling(period).corr(data['$volume'])
            
            # 最高价因子
            for period in [5, 10, 20, 30]:
                factors[f'MAX{period}'] = data['$high'].rolling(period).max()
            
            # 最低价因子
            for period in [5, 10, 20, 30]:
                factors[f'MIN{period}'] = data['$low'].rolling(period).min()
            
            # 分位数比率因子
            for period in [5, 10, 20, 30]:
                q80 = data['$close'].rolling(period).quantile(0.8)
                q20 = data['$close'].rolling(period).quantile(0.2)
                factors[f'QTLU{period}'] = q80 / (q20 + 1e-8)  # 避免除零
            
            # 填充无穷大和NaN值
            factors = factors.replace([np.inf, -np.inf], np.nan)
            factors = factors.fillna(method='ffill').fillna(0)
            
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
            # 在使用memory://模式时，需要先设置mount_path和provider_uri，否则qlib会报错
            # 使用配置中的QLIB_DATA_PATH，如果不存在则创建
            qlib_data_path = Path(settings.QLIB_DATA_PATH).resolve()
            qlib_data_path.mkdir(parents=True, exist_ok=True)
            
            # 准备mount_path和provider_uri配置
            # qlib.init()内部会调用C.set()重置配置，所以需要通过参数传递
            mount_path_config = {
                "day": str(qlib_data_path),
                "1min": str(qlib_data_path),
            }
            
            provider_uri_config = {
                "day": "memory://",
                "1min": "memory://",
            }
            
            # 使用内存模式，通过kwargs传递配置，避免被C.set()重置
            # 注意：provider_uri作为字典传递时，会覆盖字符串形式的provider_uri
            qlib.init(
                region=REG_CN,
                provider_uri=provider_uri_config,
                mount_path=mount_path_config
            )
            _QLIB_GLOBAL_INITIALIZED = True
            self._qlib_initialized = True
            logger.info("Qlib环境初始化成功")
        except Exception as e:
            error_msg = str(e)
            # 如果Qlib已经初始化，忽略错误并标记为已初始化
            if "reinitialize" in error_msg.lower() or "already activated" in error_msg.lower():
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
        use_cache: bool = True
    ) -> pd.DataFrame:
        """准备Qlib标准格式的数据集"""
        logger.info(f"准备Qlib数据集: {len(stock_codes)} 只股票, Alpha因子: {include_alpha_factors}")
        
        # 1. 获取基础特征数据
        base_features = await self._prepare_base_features(stock_codes, start_date, end_date)
        
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
        logger.info(f"特征列表: {list(qlib_data.columns[:20])}{'...' if len(qlib_data.columns) > 20 else ''}")
        logger.info(f"缺失值总数: {qlib_data.isnull().sum().sum()}")
        logger.info(f"数据类型统计: {qlib_data.dtypes.value_counts().to_dict()}")
        logger.info(f"==========================================")
        if not qlib_data.empty:
            logger.info(f"数据统计: 缺失值={qlib_data.isnull().sum().sum()}, 数据类型={qlib_data.dtypes.value_counts().to_dict()}")
        return qlib_data
    
    async def _prepare_base_features(
        self,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """准备基础特征数据"""
        all_features = []
        
        for stock_code in stock_codes:
            try:
                # 优先从本地文件加载数据
                from app.services.data.stock_data_loader import StockDataLoader
                from app.core.config import settings
                
                loader = StockDataLoader(data_root=settings.DATA_ROOT_PATH)
                stock_data = loader.load_stock_data(stock_code, start_date=start_date, end_date=end_date)
                
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
                    stock_data = pd.DataFrame([{
                        'date': item.date,
                        'open': item.open,
                        'high': item.high,
                        'low': item.low,
                        'close': item.close,
                        'volume': item.volume
                    } for item in stock_data_list])
                    stock_data = stock_data.set_index('date')
                
                # 确保数据有正确的列名
                if stock_data.empty:
                    logger.warning(f"股票 {stock_code} 无数据")
                    continue
                
                # 计算技术指标
                indicators = await self.indicator_calculator.calculate_all_indicators(stock_data)
                
                # 合并数据
                if not indicators.empty:
                    features = stock_data.merge(indicators, left_index=True, right_index=True, how='left')
                else:
                    features = stock_data.copy()
                
                features['stock_code'] = stock_code
                
                # 确保有date列
                if 'date' not in features.columns and isinstance(features.index, pd.DatetimeIndex):
                    features = features.reset_index()
                    features.rename(columns={'index': 'date'}, inplace=True)
                elif 'date' not in features.columns:
                    features = features.reset_index()
                    if 'date' not in features.columns:
                        features['date'] = features.index
                
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
        combined_features = combined_features.sort_values(['stock_code', 'date'])
        
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
                df.index.names = ['instrument', 'datetime']
                return df
            else:
                logger.warning(f"MultiIndex层级数不正确: {len(df.index.names)}")
        
        # 需要创建MultiIndex
        if 'stock_code' in df.columns and 'date' in df.columns:
            # 确保date列是datetime类型
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # 设置MultiIndex
            df_indexed = df.set_index(['stock_code', 'date'])
            df_indexed.index.names = ['instrument', 'datetime']
            return df_indexed
        
        elif isinstance(df.index, pd.DatetimeIndex) and 'stock_code' in df.columns:
            # 日期在索引中，股票代码在列中
            df_reset = df.reset_index()
            df_reset.rename(columns={'index': 'date'}, inplace=True)
            df_reset['date'] = pd.to_datetime(df_reset['date'])
            df_indexed = df_reset.set_index(['stock_code', 'date'])
            df_indexed.index.names = ['instrument', 'datetime']
            return df_indexed
        
        else:
            logger.warning("无法创建MultiIndex，缺少必要的股票代码或日期信息")
            return df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名为Qlib格式"""
        # Qlib标准列名映射
        column_mapping = {
            # 基础OHLCV数据
            'open': '$open',
            'high': '$high',
            'low': '$low',
            'close': '$close',
            'volume': '$volume',
            'adj_close': '$close',  # 如果有复权价格，使用它作为收盘价
            
            # 技术指标（保持原名或添加前缀）
            'MA5': 'MA5',
            'MA10': 'MA10',
            'MA20': 'MA20',
            'MA60': 'MA60',
            'EMA': 'EMA20',
            'WMA': 'WMA20',
            'RSI': 'RSI14',
            'MACD': 'MACD',
            'MACD_SIGNAL': 'MACD_SIGNAL',
            'MACD_HISTOGRAM': 'MACD_HIST',
            'BOLLINGER_UPPER': 'BOLL_UPPER',
            'BOLLINGER_MIDDLE': 'BOLL_MIDDLE',
            'BOLLINGER_LOWER': 'BOLL_LOWER',
            'ATR': 'ATR14',
            'VWAP': 'VWAP',
            'OBV': 'OBV',
            'STOCH_K': 'STOCH_K',
            'STOCH_D': 'STOCH_D',
            'WILLIAMS_R': 'WILLIAMS_R',
            'CCI': 'CCI20',
            'KDJ_K': 'KDJ_K',
            'KDJ_D': 'KDJ_D',
            'KDJ_J': 'KDJ_J',
            
            # 基本面特征
            'price_change': 'RET1',
            'price_change_5d': 'RET5',
            'price_change_20d': 'RET20',
            'volume_change': 'VOLUME_RET1',
            'volume_ma_ratio': 'VOLUME_MA_RATIO',
            'volatility_5d': 'VOLATILITY5',
            'volatility_20d': 'VOLATILITY20',
            'price_position': 'PRICE_POSITION',
        }
        
        # 只重命名存在的列
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_renamed = df.rename(columns=existing_mapping)
        
        # 确保基础OHLCV列存在
        required_base_cols = ['$open', '$high', '$low', '$close', '$volume']
        missing_base_cols = [col for col in required_base_cols if col not in df_renamed.columns]
        
        if missing_base_cols:
            logger.warning(f"缺少基础OHLCV列: {missing_base_cols}")
        
        logger.debug(f"列名标准化完成: {len(existing_mapping)} 个列被重命名")
        return df_renamed
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型以节省内存"""
        df_optimized = df.copy()
        
        # 价格相关列使用float32
        price_cols = ['$open', '$high', '$low', '$close']
        for col in price_cols:
            if col in df_optimized.columns:
                df_optimized[col] = pd.to_numeric(df_optimized[col], errors='coerce').astype('float32')
        
        # 成交量使用int64（可能很大）
        if '$volume' in df_optimized.columns:
            df_optimized['$volume'] = pd.to_numeric(df_optimized['$volume'], errors='coerce').astype('int64')
        
        # 技术指标使用float32
        indicator_cols = [col for col in df_optimized.columns if col not in price_cols + ['$volume']]
        for col in indicator_cols:
            if df_optimized[col].dtype in ['float64', 'object']:
                df_optimized[col] = pd.to_numeric(df_optimized[col], errors='coerce').astype('float32')
        
        logger.debug("数据类型优化完成")
        return df_optimized
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        df_filled = df.copy()
        
        # 基础价格数据：前向填充
        price_cols = ['$open', '$high', '$low', '$close', '$volume']
        for col in price_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(method='ffill')
        
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
            df_sorted = df_sorted[~df_sorted.index.duplicated(keep='last')]
            logger.warning(f"去重后: {len(df_sorted)}")
        
        return df_sorted
    
    def _add_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基本面特征"""
        # 价格变化率
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        df['price_change_20d'] = df['close'].pct_change(periods=20)
        
        # 成交量变化率
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 波动率
        df['volatility_5d'] = df['price_change'].rolling(5).std()
        df['volatility_20d'] = df['price_change'].rolling(20).std()
        
        # 价格位置
        df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (
            df['high'].rolling(20).max() - df['low'].rolling(20).min()
        )
        
        return df
    
    async def create_qlib_model_config(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any]
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
                "num_threads": 20
            }
        }
        
        # 根据模型类型调整配置
        if model_type.lower() == 'lightgbm':
            base_config["class"] = "LGBModel"
            base_config["module_path"] = "qlib.contrib.model.gbdt"
        elif model_type.lower() == 'xgboost':
            base_config["class"] = "XGBModel"
            base_config["module_path"] = "qlib.contrib.model.xgboost"
        elif model_type.lower() == 'mlp':
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
                "qlib_initialized": self._qlib_initialized
            }
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {
                "cache_files": 0,
                "total_size_mb": 0,
                "cache_dir": "unknown",
                "qlib_available": QLIB_AVAILABLE,
                "qlib_initialized": self._qlib_initialized
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
        prediction_horizon: int = 5
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
                predictions = predictions.to_frame('prediction')
            elif not isinstance(predictions, pd.DataFrame):
                predictions = pd.DataFrame(predictions, columns=['prediction'])
            
            logger.info(f"Qlib预测完成: {len(predictions)} 条预测结果")
            return predictions
            
        except Exception as e:
            logger.error(f"Qlib模型预测失败: {e}")
            return pd.DataFrame()
    
    async def validate_and_fix_qlib_format(self, data: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
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
            required_cols = ['$open', '$high', '$low', '$close', '$volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.warning(f"缺少必要的列: {missing_cols}")
                # 尝试从其他列推导
                data = self._fix_missing_columns(data, missing_cols)
                
                # 再次检查
                still_missing = [col for col in required_cols if col not in data.columns]
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
    
    def _fix_missing_columns(self, data: pd.DataFrame, missing_cols: List[str]) -> pd.DataFrame:
        """修复缺少的列"""
        data_fixed = data.copy()
        
        # 尝试从相似列名推导
        column_alternatives = {
            '$open': ['open', 'Open', 'OPEN'],
            '$high': ['high', 'High', 'HIGH'],
            '$low': ['low', 'Low', 'LOW'],
            '$close': ['close', 'Close', 'CLOSE', 'adj_close', 'Adj_Close'],
            '$volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol']
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
                    if missing_col == '$volume':
                        data_fixed[missing_col] = 1000000  # 默认成交量
                    else:
                        # 对于价格列，使用close价格作为默认值
                        if '$close' in data_fixed.columns:
                            data_fixed[missing_col] = data_fixed['$close']
                        else:
                            data_fixed[missing_col] = 100.0  # 默认价格
                    
                    logger.warning(f"使用默认值填充 {missing_col}")
        
        return data_fixed
    
    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """修复数据类型"""
        data_fixed = data.copy()
        
        # 价格列应该是数值类型
        price_cols = ['$open', '$high', '$low', '$close']
        for col in price_cols:
            if col in data_fixed.columns:
                data_fixed[col] = pd.to_numeric(data_fixed[col], errors='coerce')
        
        # 成交量应该是整数类型
        if '$volume' in data_fixed.columns:
            data_fixed['$volume'] = pd.to_numeric(data_fixed['$volume'], errors='coerce')
            data_fixed['$volume'] = data_fixed['$volume'].fillna(0).astype('int64')
        
        # 其他数值列
        for col in data_fixed.columns:
            if col not in price_cols + ['$volume']:
                if data_fixed[col].dtype == 'object':
                    data_fixed[col] = pd.to_numeric(data_fixed[col], errors='coerce')
        
        return data_fixed
    
    def _check_data_quality(self, data: pd.DataFrame) -> List[str]:
        """检查数据质量问题"""
        issues = []
        
        # 检查价格逻辑
        if all(col in data.columns for col in ['$open', '$high', '$low', '$close']):
            # 最高价应该 >= 最低价
            invalid_high_low = (data['$high'] < data['$low']).sum()
            if invalid_high_low > 0:
                issues.append(f"high < low: {invalid_high_low} 条记录")
            
            # 价格应该为正数
            for col in ['$open', '$high', '$low', '$close']:
                negative_prices = (data[col] <= 0).sum()
                if negative_prices > 0:
                    issues.append(f"{col} <= 0: {negative_prices} 条记录")
        
        # 检查成交量
        if '$volume' in data.columns:
            negative_volume = (data['$volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"负成交量: {negative_volume} 条记录")
        
        # 检查缺失值
        missing_counts = data.isnull().sum()
        critical_missing = missing_counts[missing_counts > len(data) * 0.1]  # 超过10%缺失
        if not critical_missing.empty:
            issues.append(f"高缺失率列: {critical_missing.to_dict()}")
        
        return issues
    
    def _fix_data_quality_issues(self, data: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """修复数据质量问题"""
        data_fixed = data.copy()
        
        # 修复价格逻辑问题
        if all(col in data_fixed.columns for col in ['$open', '$high', '$low', '$close']):
            # 修复 high < low 的问题
            invalid_mask = data_fixed['$high'] < data_fixed['$low']
            if invalid_mask.sum() > 0:
                # 交换high和low
                data_fixed.loc[invalid_mask, ['$high', '$low']] = data_fixed.loc[invalid_mask, ['$low', '$high']].values
                logger.info(f"修复了 {invalid_mask.sum()} 条 high < low 的记录")
            
            # 修复负价格
            for col in ['$open', '$high', '$low', '$close']:
                negative_mask = data_fixed[col] <= 0
                if negative_mask.sum() > 0:
                    # 使用前一个有效值填充
                    data_fixed.loc[negative_mask, col] = data_fixed[col].fillna(method='ffill')
                    # 如果还有负值，使用均值
                    still_negative = data_fixed[col] <= 0
                    if still_negative.sum() > 0:
                        mean_price = data_fixed[col][data_fixed[col] > 0].mean()
                        data_fixed.loc[still_negative, col] = mean_price
                    logger.info(f"修复了 {negative_mask.sum()} 条 {col} <= 0 的记录")
        
        # 修复负成交量
        if '$volume' in data_fixed.columns:
            negative_mask = data_fixed['$volume'] < 0
            if negative_mask.sum() > 0:
                data_fixed.loc[negative_mask, '$volume'] = 0
                logger.info(f"修复了 {negative_mask.sum()} 条负成交量记录")
        
        # 处理高缺失率列
        missing_counts = data_fixed.isnull().sum()
        high_missing_cols = missing_counts[missing_counts > len(data_fixed) * 0.5].index  # 超过50%缺失
        
        for col in high_missing_cols:
            if col in ['$open', '$high', '$low', '$close']:
                # 价格列使用前向填充
                data_fixed[col] = data_fixed[col].fillna(method='ffill').fillna(method='bfill')
            elif col == '$volume':
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
            required_cols = ['$close', '$high', '$low', '$open', '$volume']
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
        self,
        df: pd.DataFrame,
        validate: bool = True,
        fix_issues: bool = True
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
            "issues_fixed": []
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
                is_valid, fixed_df = await self.validate_and_fix_qlib_format(converted_df)
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
            conversion_info["memory_usage_mb"] = converted_df.memory_usage(deep=True).sum() / 1024 / 1024
            
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
                "example": "('000001.SZ', '2023-01-01')"
            },
            "required_columns": {
                "$open": "开盘价",
                "$high": "最高价", 
                "$low": "最低价",
                "$close": "收盘价",
                "$volume": "成交量"
            },
            "optional_columns": {
                "technical_indicators": "技术指标 (RSI, MACD, etc.)",
                "alpha_factors": "Alpha因子 (RESI5, MA10, etc.)",
                "fundamental_features": "基本面特征 (RET1, VOLATILITY5, etc.)"
            },
            "data_types": {
                "prices": "float32",
                "volume": "int64", 
                "indicators": "float32"
            },
            "quality_requirements": [
                "价格必须为正数",
                "最高价 >= 最低价",
                "成交量 >= 0",
                "无重复的时间戳",
                "按时间排序"
            ]
        }