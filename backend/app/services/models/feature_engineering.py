"""
数据准备和特征工程模块

提供模块化的数据加载、特征计算和数据预处理功能，
支持多模态特征工程和时间序列数据处理。
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from loguru import logger

# 导入统一的错误处理机制
try:
    from app.core.error_handler import DataError, ErrorSeverity, ErrorContext, handle_async_exception
except ImportError:
    logger.warning("错误处理模块未找到，使用默认错误处理")
    DataError = Exception
    ErrorSeverity = None
    ErrorContext = None
    handle_async_exception = lambda func: func

# 导入数据服务和指标计算
try:
    from ..data.simple_data_service import SimpleDataService
    from ..prediction.technical_indicators import TechnicalIndicatorCalculator
except ImportError:
    logger.warning("数据服务或指标计算模块未找到")
    SimpleDataService = None
    TechnicalIndicatorCalculator = None

# 导入Qlib相关模块
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


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_service: SimpleDataService, data_root: str):
        self.data_service = data_service
        self.data_root = data_root
    
    @handle_async_exception
    async def load_stock_data(
        self, 
        stock_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """加载股票数据"""
        try:
            # 优先从本地文件加载数据
            from app.services.data.stock_data_loader import StockDataLoader
            
            loader = StockDataLoader(data_root=self.data_root)
            stock_data = loader.load_stock_data(stock_code, start_date=start_date, end_date=end_date)
            
            # 如果本地没有数据，尝试从远端服务获取
            if stock_data.empty:
                logger.info(f"本地无数据，从远端服务获取: {stock_code}")
                stock_data_list = await self.data_service.get_stock_data(
                    stock_code, start_date, end_date
                )
                
                if not stock_data_list or len(stock_data_list) == 0:
                    logger.warning(f"股票 {stock_code} 无数据")
                    return pd.DataFrame()
                
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
            
            return stock_data
            
        except Exception as e:
            logger.error(f"加载股票数据失败: {stock_code}, 错误: {e}")
            return pd.DataFrame()


class FeatureCalculator:
    """特征计算器"""
    
    def __init__(self):
        self.indicator_calculator = TechnicalIndicatorCalculator() if TechnicalIndicatorCalculator else None
    
    @handle_async_exception
    async def calculate_technical_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        if not self.indicator_calculator:
            logger.warning("TechnicalIndicatorCalculator不可用，跳过技术指标计算")
            return pd.DataFrame()
        
        try:
            indicators = await self.indicator_calculator.calculate_all_indicators(stock_data)
            return indicators
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return pd.DataFrame()
    
    def calculate_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基本面特征"""
        try:
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
        except Exception as e:
            logger.error(f"计算基本面特征失败: {e}")
            return df
    
    def calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算额外特征"""
        try:
            # 价格动量
            df['momentum_5d'] = df['close'].pct_change(periods=5)
            df['momentum_20d'] = df['close'].pct_change(periods=20)
            
            # 成交量动量
            df['volume_momentum_5d'] = df['volume'].pct_change(periods=5)
            df['volume_momentum_20d'] = df['volume'].pct_change(periods=20)
            
            # 价格范围
            df['price_range'] = df['high'] - df['low']
            df['price_range_ratio'] = df['price_range'] / df['close']
            
            # 开盘/收盘关系
            df['open_close_ratio'] = df['open'] / df['close']
            df['high_close_ratio'] = df['high'] / df['close']
            df['low_close_ratio'] = df['low'] / df['close']
            
            return df
        except Exception as e:
            logger.error(f"计算额外特征失败: {e}")
            return df


class FeatureEngineer:
    """特征工程师"""
    
    def __init__(self, data_service: SimpleDataService, data_root: str):
        self.data_loader = DataLoader(data_service, data_root)
        self.feature_calculator = FeatureCalculator()
    
    @handle_async_exception
    async def prepare_features(
        self, 
        stock_codes: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        准备多模态特征集
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含所有特征的DataFrame
        """
        all_features = []
        
        for stock_code in stock_codes:
            try:
                # 加载股票数据
                stock_data = await self.data_loader.load_stock_data(
                    stock_code, start_date, end_date
                )
                
                if stock_data.empty:
                    logger.warning(f"股票 {stock_code} 无数据")
                    continue
                
                # 计算技术指标
                indicators = await self.feature_calculator.calculate_technical_indicators(stock_data)
                
                # 合并数据（使用索引合并，因为两者都使用日期作为索引）
                if not indicators.empty:
                    features = stock_data.merge(indicators, left_index=True, right_index=True, how='left')
                else:
                    features = stock_data.copy()
                
                # 添加股票代码
                features['stock_code'] = stock_code
                
                # 确保有date列用于后续排序（如果索引是日期，将其作为列）
                if 'date' not in features.columns and isinstance(features.index, pd.DatetimeIndex):
                    features = features.reset_index()
                    features.rename(columns={'index': 'date'}, inplace=True)
                elif 'date' not in features.columns:
                    # 如果索引不是日期，尝试从索引名称推断
                    features = features.reset_index()
                    if 'date' not in features.columns:
                        # 如果还是没有date列，使用索引作为date
                        features['date'] = features.index
                
                # 添加基本面特征
                features = self.feature_calculator.calculate_fundamental_features(features)
                
                # 添加额外特征
                features = self.feature_calculator.calculate_additional_features(features)
                
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"处理股票 {stock_code} 特征时出错: {e}")
                continue
        
        if not all_features:
            raise DataError(
                message="没有成功处理任何股票数据",
                severity=ErrorSeverity.HIGH
            )
        
        # 合并所有股票的特征
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_features = combined_features.sort_values(['stock_code', 'date'])
        
        logger.info(f"成功准备了 {len(stock_codes)} 只股票的特征数据，共 {len(combined_features)} 条记录")
        return combined_features


class QlibFeatureProvider:
    """Qlib特征提供器"""
    
    def __init__(self):
        self.qlib_initialized = False
    
    @handle_async_exception
    async def initialize_qlib(self):
        """初始化Qlib环境"""
        if not QLIB_AVAILABLE:
            logger.warning("Qlib未安装，跳过初始化")
            return False
        
        try:
            # 使用本地路径作为 provider_uri，避免 Qlib 将 ":" 拼进 data_path 导致日历路径出错
            from app.core.config import settings
            
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
                "day": str(qlib_data_path),
                "1min": str(qlib_data_path),
            }
            
            # 通过kwargs传递配置，避免被C.set()重置
            # 注意：provider_uri作为字典传递时，会覆盖字符串形式的provider_uri
            qlib.init(
                region=REG_CN,
                provider_uri=provider_uri_config,
                mount_path=mount_path_config
            )
            
            self.qlib_initialized = True
            logger.info("Qlib环境初始化成功")
            return True
        except Exception as e:
            logger.error(f"Qlib初始化失败: {e}")
            return False
    
    @handle_async_exception
    async def get_qlib_features(
        self, 
        stock_codes: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """获取Qlib特征"""
        if not self.qlib_initialized:
            logger.warning("Qlib未初始化，跳过特征获取")
            return pd.DataFrame()
        
        try:
            # 这里可以实现基于Qlib的特征获取逻辑
            # 为了简化，暂时返回空DataFrame
            logger.info("Qlib特征获取功能待实现")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取Qlib特征失败: {e}")
            return pd.DataFrame()


class DataPreprocessor:
    """数据预处理器"""
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        try:
            # 填充缺失值
            df = df.fillna(0)
            
            # 移除异常值
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in ['stock_code', 'date']:
                    # 使用IQR方法检测异常值
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            return df
        except Exception as e:
            logger.error(f"清理数据失败: {e}")
            return df
    
    @staticmethod
    def normalize_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """归一化特征"""
        try:
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            df[feature_columns] = scaler.fit_transform(df[feature_columns])
            
            return df
        except Exception as e:
            logger.error(f"归一化特征失败: {e}")
            return df
