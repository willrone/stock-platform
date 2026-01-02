"""
特征工程管道

协调特征计算、存储和管理
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from loguru import logger

import pandas as pd

from .feature_store import FeatureStore, FeatureType, FeatureMetadata
from ..prediction.technical_indicators import TechnicalIndicatorCalculator
from ..data.simple_data_service import SimpleDataService


class FeaturePipeline:
    """特征工程管道"""
    
    def __init__(self, feature_store: Optional[FeatureStore] = None):
        self.feature_store = feature_store or FeatureStore()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.data_service = SimpleDataService()
        
        # 注册的特征计算器
        self._feature_calculators: Dict[str, callable] = {}
        
        # 初始化标志
        self._initialized = False
        
        logger.info("特征工程管道初始化")
    
    async def initialize(self):
        """初始化特征管道"""
        if self._initialized:
            return
        
        try:
            # 初始化特征存储
            await self.feature_store.initialize()
            
            # 注册内置特征计算器
            await self._register_builtin_features()
            
            self._initialized = True
            logger.info("特征工程管道初始化完成")
        
        except Exception as e:
            logger.error(f"特征工程管道初始化失败: {e}")
            raise
    
    async def on_data_sync_complete(
        self,
        stock_code: str,
        date_range: Tuple[datetime, datetime],
        sync_type: str = "incremental"
    ):
        """数据同步完成回调"""
        logger.info(f"收到数据同步完成通知: {stock_code}, 日期范围: {date_range}, 类型: {sync_type}")
        
        try:
            # 使相关缓存失效
            await self.feature_store.invalidate_cache(
                stock_codes=[stock_code]
            )
            
            # 如果是增量同步，计算新的特征
            if sync_type == "incremental":
                await self._calculate_incremental_features(stock_code, date_range)
            
            logger.info(f"数据同步后处理完成: {stock_code}")
        
        except Exception as e:
            logger.error(f"数据同步后处理失败 {stock_code}: {e}")
    
    async def calculate_features(
        self,
        stock_codes: List[str],
        start_date: datetime,
        end_date: datetime,
        feature_names: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """计算特征"""
        if not self._initialized:
            await self.initialize()
        
        # 如果没有指定特征，使用所有已注册的特征
        if feature_names is None:
            feature_names = list(self._feature_calculators.keys())
        
        # 尝试从缓存获取
        if use_cache:
            cached_data = await self.feature_store.get_cached_features(
                stock_codes, (start_date, end_date), feature_names
            )
            if cached_data is not None:
                logger.info(f"使用缓存特征数据: {len(cached_data)} 条记录")
                return cached_data
        
        # 计算特征
        logger.info(f"开始计算特征: {len(stock_codes)} 只股票, {len(feature_names)} 个特征")
        
        all_features = []
        
        for stock_code in stock_codes:
            try:
                # 获取股票数据
                stock_data = await self._get_stock_data(stock_code, start_date, end_date)
                if stock_data.empty:
                    logger.warning(f"股票 {stock_code} 无数据")
                    continue
                
                # 计算特征
                features = await self._calculate_stock_features(
                    stock_code, stock_data, feature_names
                )
                
                if not features.empty:
                    features['stock_code'] = stock_code
                    all_features.append(features)
            
            except Exception as e:
                logger.error(f"计算股票 {stock_code} 特征失败: {e}")
                continue
        
        if not all_features:
            logger.warning("没有成功计算任何特征")
            return pd.DataFrame()
        
        # 合并所有特征
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # 缓存结果
        if use_cache and not combined_features.empty:
            await self.feature_store.cache_features(
                stock_codes, (start_date, end_date), combined_features, feature_names
            )
        
        logger.info(f"特征计算完成: {len(combined_features)} 条记录")
        return combined_features
    
    async def register_feature_calculator(
        self,
        feature_name: str,
        calculator: callable,
        feature_type: FeatureType = FeatureType.CUSTOM,
        dependencies: Optional[List[str]] = None,
        update_frequency: str = "daily",
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """注册特征计算器"""
        # 注册计算器
        self._feature_calculators[feature_name] = calculator
        
        # 注册元数据
        await self.feature_store.register_feature(
            feature_name=feature_name,
            feature_type=feature_type,
            calculation_method=calculator.__name__ if hasattr(calculator, '__name__') else str(calculator),
            dependencies=dependencies or [],
            update_frequency=update_frequency,
            description=description,
            parameters=parameters
        )
        
        logger.info(f"特征计算器注册成功: {feature_name}")
    
    async def get_available_features(self) -> List[FeatureMetadata]:
        """获取可用特征列表"""
        return await self.feature_store.list_features()
    
    async def get_feature_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """获取特征元数据"""
        return await self.feature_store.get_feature_metadata(feature_name)
    
    async def invalidate_feature_cache(
        self,
        stock_codes: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ):
        """使特征缓存失效"""
        await self.feature_store.invalidate_cache(stock_codes, feature_names)
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取管道统计信息"""
        cache_stats = await self.feature_store.get_cache_stats()
        
        return {
            "registered_features": len(self._feature_calculators),
            "cache_stats": cache_stats,
            "initialized": self._initialized
        }
    
    async def _register_builtin_features(self):
        """注册内置特征计算器"""
        # 技术指标特征 - 扩展支持更多指标
        technical_indicators = [
            # 移动平均线
            ("MA5", self._calculate_ma5, "5日移动平均线"),
            ("MA10", self._calculate_ma10, "10日移动平均线"),
            ("MA20", self._calculate_ma20, "20日移动平均线"),
            ("MA60", self._calculate_ma60, "60日移动平均线"),
            ("EMA", self._calculate_ema, "指数移动平均线"),
            ("WMA", self._calculate_wma, "加权移动平均线"),
            
            # 动量指标
            ("RSI", self._calculate_rsi, "相对强弱指数"),
            ("STOCH_K", self._calculate_stoch_k, "随机指标K值"),
            ("STOCH_D", self._calculate_stoch_d, "随机指标D值"),
            ("WILLIAMS_R", self._calculate_williams_r, "威廉指标"),
            ("CCI", self._calculate_cci, "商品通道指数"),
            
            # 趋势指标
            ("MACD", self._calculate_macd, "MACD指标"),
            ("MACD_SIGNAL", self._calculate_macd_signal, "MACD信号线"),
            ("MACD_HISTOGRAM", self._calculate_macd_histogram, "MACD柱状图"),
            ("BOLLINGER_UPPER", self._calculate_bb_upper, "布林带上轨"),
            ("BOLLINGER_MIDDLE", self._calculate_bb_middle, "布林带中轨"),
            ("BOLLINGER_LOWER", self._calculate_bb_lower, "布林带下轨"),
            
            # 波动率指标
            ("ATR", self._calculate_atr, "平均真实波幅"),
            
            # 成交量指标
            ("VWAP", self._calculate_vwap, "成交量加权平均价格"),
            ("OBV", self._calculate_obv, "能量潮"),
            
            # 复合指标
            ("KDJ_K", self._calculate_kdj_k, "KDJ指标K值"),
            ("KDJ_D", self._calculate_kdj_d, "KDJ指标D值"),
            ("KDJ_J", self._calculate_kdj_j, "KDJ指标J值"),
        ]
        
        for feature_name, calculator, description in technical_indicators:
            await self.register_feature_calculator(
                feature_name=feature_name,
                calculator=calculator,
                feature_type=FeatureType.TECHNICAL_INDICATOR,
                dependencies=["open", "high", "low", "close", "volume"],
                description=description
            )
        
        # 基本面特征
        fundamental_features = [
            ("price_change", self._calculate_price_change, "价格变化率"),
            ("price_change_5d", self._calculate_price_change_5d, "5日价格变化率"),
            ("price_change_20d", self._calculate_price_change_20d, "20日价格变化率"),
            ("volume_change", self._calculate_volume_change, "成交量变化率"),
            ("volume_ma_ratio", self._calculate_volume_ma_ratio, "成交量均线比率"),
            ("volatility_5d", self._calculate_volatility_5d, "5日波动率"),
            ("volatility_20d", self._calculate_volatility_20d, "20日波动率"),
            ("price_position", self._calculate_price_position, "价格位置"),
        ]
        
        for feature_name, calculator, description in fundamental_features:
            await self.register_feature_calculator(
                feature_name=feature_name,
                calculator=calculator,
                feature_type=FeatureType.FUNDAMENTAL,
                dependencies=["open", "high", "low", "close", "volume"],
                description=description
            )
        
        logger.info(f"内置特征注册完成: {len(technical_indicators) + len(fundamental_features)} 个特征")
    
    async def _get_stock_data(
        self,
        stock_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """获取股票数据"""
        try:
            # 优先从本地文件加载
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
                
                if stock_data_list and len(stock_data_list) > 0:
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
            logger.error(f"获取股票数据失败 {stock_code}: {e}")
            return pd.DataFrame()
    
    async def _calculate_stock_features(
        self,
        stock_code: str,
        stock_data: pd.DataFrame,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """计算单只股票的特征"""
        # 首先计算所有技术指标
        indicators_df = await self.indicator_calculator.calculate_all_indicators(stock_data)
        
        # 合并原始数据和技术指标
        if not indicators_df.empty:
            combined_data = stock_data.merge(indicators_df, left_index=True, right_index=True, how='left')
        else:
            combined_data = stock_data.copy()
        
        # 添加基本面特征
        combined_data = self._add_fundamental_features(combined_data)
        
        # 重置索引，确保有date列
        if 'date' not in combined_data.columns:
            combined_data = combined_data.reset_index()
            if 'index' in combined_data.columns:
                combined_data.rename(columns={'index': 'date'}, inplace=True)
        
        # 只返回请求的特征
        available_features = [f for f in feature_names if f in combined_data.columns]
        if available_features:
            result = combined_data[['date'] + available_features].copy()
            return result
        
        return pd.DataFrame()
    
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
    
    async def _calculate_incremental_features(
        self,
        stock_code: str,
        date_range: Tuple[datetime, datetime]
    ):
        """计算增量特征"""
        try:
            # 获取需要更新的特征
            features_to_update = await self.feature_store.list_features(
                update_frequency="daily"
            )
            
            if not features_to_update:
                return
            
            feature_names = [f.feature_name for f in features_to_update]
            
            # 计算特征
            features_df = await self.calculate_features(
                stock_codes=[stock_code],
                start_date=date_range[0],
                end_date=date_range[1],
                feature_names=feature_names,
                use_cache=False  # 增量更新不使用缓存
            )
            
            if not features_df.empty:
                logger.info(f"增量特征计算完成: {stock_code}, {len(features_df)} 条记录")
        
        except Exception as e:
            logger.error(f"增量特征计算失败 {stock_code}: {e}")
    
    # 技术指标计算器方法 - 扩展版本
    async def _calculate_ma5(self, data: pd.DataFrame) -> pd.Series:
        """计算5日移动平均线"""
        return data['close'].rolling(5).mean()
    
    async def _calculate_ma10(self, data: pd.DataFrame) -> pd.Series:
        """计算10日移动平均线"""
        return data['close'].rolling(10).mean()
    
    async def _calculate_ma20(self, data: pd.DataFrame) -> pd.Series:
        """计算20日移动平均线"""
        return data['close'].rolling(20).mean()
    
    async def _calculate_ma60(self, data: pd.DataFrame) -> pd.Series:
        """计算60日移动平均线"""
        return data['close'].rolling(60).mean()
    
    async def _calculate_ema(self, data: pd.DataFrame) -> pd.Series:
        """计算指数移动平均线"""
        return data['close'].ewm(span=20).mean()
    
    async def _calculate_wma(self, data: pd.DataFrame) -> pd.Series:
        """计算加权移动平均线"""
        # 简化实现，使用pandas的rolling和apply
        def wma_func(x):
            weights = range(1, len(x) + 1)
            return sum(x * weights) / sum(weights)
        
        return data['close'].rolling(20).apply(wma_func, raw=True)
    
    async def _calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """计算RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def _calculate_stoch_k(self, data: pd.DataFrame) -> pd.Series:
        """计算随机指标K值"""
        low_min = data['low'].rolling(14).min()
        high_max = data['high'].rolling(14).max()
        return 100 * (data['close'] - low_min) / (high_max - low_min)
    
    async def _calculate_stoch_d(self, data: pd.DataFrame) -> pd.Series:
        """计算随机指标D值"""
        stoch_k = await self._calculate_stoch_k(data)
        return stoch_k.rolling(3).mean()
    
    async def _calculate_williams_r(self, data: pd.DataFrame) -> pd.Series:
        """计算威廉指标"""
        high_max = data['high'].rolling(14).max()
        low_min = data['low'].rolling(14).min()
        return -100 * (high_max - data['close']) / (high_max - low_min)
    
    async def _calculate_cci(self, data: pd.DataFrame) -> pd.Series:
        """计算商品通道指数"""
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean())
        return (tp - sma_tp) / (0.015 * mad)
    
    async def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """计算平均真实波幅"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14).mean()
    
    async def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量加权平均价格"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
    async def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """计算能量潮"""
        price_change = data['close'].diff()
        volume_direction = data['volume'].where(price_change > 0, -data['volume']).where(price_change != 0, 0)
        return volume_direction.cumsum()
    
    async def _calculate_kdj_k(self, data: pd.DataFrame) -> pd.Series:
        """计算KDJ指标K值"""
        low_min = data['low'].rolling(9).min()
        high_max = data['high'].rolling(9).max()
        rsv = 100 * (data['close'] - low_min) / (high_max - low_min)
        
        # 计算K值（使用指数平滑）
        k_values = pd.Series(index=data.index, dtype=float)
        for i in range(len(data)):
            if i == 0 or pd.isna(rsv.iloc[i]):
                k_values.iloc[i] = rsv.iloc[i] if not pd.isna(rsv.iloc[i]) else None
            else:
                prev_k = k_values.iloc[i-1] if not pd.isna(k_values.iloc[i-1]) else rsv.iloc[i]
                k_values.iloc[i] = (2/3) * prev_k + (1/3) * rsv.iloc[i]
        
        return k_values
    
    async def _calculate_kdj_d(self, data: pd.DataFrame) -> pd.Series:
        """计算KDJ指标D值"""
        k_values = await self._calculate_kdj_k(data)
        
        # 计算D值（K值的指数平滑）
        d_values = pd.Series(index=data.index, dtype=float)
        for i in range(len(data)):
            if i == 0 or pd.isna(k_values.iloc[i]):
                d_values.iloc[i] = k_values.iloc[i] if not pd.isna(k_values.iloc[i]) else None
            else:
                prev_d = d_values.iloc[i-1] if not pd.isna(d_values.iloc[i-1]) else k_values.iloc[i]
                d_values.iloc[i] = (2/3) * prev_d + (1/3) * k_values.iloc[i]
        
        return d_values
    
    async def _calculate_kdj_j(self, data: pd.DataFrame) -> pd.Series:
        """计算KDJ指标J值"""
        k_values = await self._calculate_kdj_k(data)
        d_values = await self._calculate_kdj_d(data)
        return 3 * k_values - 2 * d_values
    
    async def _calculate_macd(self, data: pd.DataFrame) -> pd.Series:
        """计算MACD"""
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        return ema12 - ema26
    
    async def _calculate_macd_signal(self, data: pd.DataFrame) -> pd.Series:
        """计算MACD信号线"""
        macd = await self._calculate_macd(data)
        return macd.ewm(span=9).mean()
    
    async def _calculate_macd_histogram(self, data: pd.DataFrame) -> pd.Series:
        """计算MACD柱状图"""
        macd = await self._calculate_macd(data)
        signal = await self._calculate_macd_signal(data)
        return macd - signal
    
    async def _calculate_bb_upper(self, data: pd.DataFrame) -> pd.Series:
        """计算布林带上轨"""
        ma20 = data['close'].rolling(20).mean()
        std20 = data['close'].rolling(20).std()
        return ma20 + (2 * std20)
    
    async def _calculate_bb_middle(self, data: pd.DataFrame) -> pd.Series:
        """计算布林带中轨"""
        return data['close'].rolling(20).mean()
    
    async def _calculate_bb_lower(self, data: pd.DataFrame) -> pd.Series:
        """计算布林带下轨"""
        ma20 = data['close'].rolling(20).mean()
        std20 = data['close'].rolling(20).std()
        return ma20 - (2 * std20)
    
    # 基本面特征计算器方法
    async def _calculate_price_change(self, data: pd.DataFrame) -> pd.Series:
        """计算价格变化率"""
        return data['close'].pct_change()
    
    async def _calculate_price_change_5d(self, data: pd.DataFrame) -> pd.Series:
        """计算5日价格变化率"""
        return data['close'].pct_change(periods=5)
    
    async def _calculate_price_change_20d(self, data: pd.DataFrame) -> pd.Series:
        """计算20日价格变化率"""
        return data['close'].pct_change(periods=20)
    
    async def _calculate_volume_change(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量变化率"""
        return data['volume'].pct_change()
    
    async def _calculate_volume_ma_ratio(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量均线比率"""
        return data['volume'] / data['volume'].rolling(20).mean()
    
    async def _calculate_volatility_5d(self, data: pd.DataFrame) -> pd.Series:
        """计算5日波动率"""
        return data['close'].pct_change().rolling(5).std()
    
    async def _calculate_volatility_20d(self, data: pd.DataFrame) -> pd.Series:
        """计算20日波动率"""
        return data['close'].pct_change().rolling(20).std()
    
    async def _calculate_price_position(self, data: pd.DataFrame) -> pd.Series:
        """计算价格位置"""
        return (data['close'] - data['low'].rolling(20).min()) / (
            data['high'].rolling(20).max() - data['low'].rolling(20).min()
        )

# 全局特征管道实例
feature_pipeline = FeaturePipeline()