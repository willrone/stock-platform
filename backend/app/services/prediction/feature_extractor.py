"""
特征提取器 - 从股票数据中提取技术指标和统计特征
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import talib
from loguru import logger
from backend.app.core.error_handler import DataError, ErrorSeverity, ErrorContext


@dataclass
class FeatureConfig:
    """特征配置"""
    technical_indicators: List[str]
    statistical_features: List[str]
    time_windows: List[int]  # 时间窗口（天数）
    custom_features: Optional[List[str]] = None
    cache_enabled: bool = True


class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """简单移动平均线"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """指数移动平均线"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """相对强弱指数"""
        return pd.Series(talib.RSI(prices.values, timeperiod=window), index=prices.index)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD指标"""
        macd, macd_signal, macd_hist = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return {
            'macd': pd.Series(macd, index=prices.index),
            'macd_signal': pd.Series(macd_signal, index=prices.index),
            'macd_hist': pd.Series(macd_hist, index=prices.index)
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """布林带"""
        upper, middle, lower = talib.BBANDS(prices.values, timeperiod=window, nbdevup=std_dev, nbdevdn=std_dev)
        return {
            'bb_upper': pd.Series(upper, index=prices.index),
            'bb_middle': pd.Series(middle, index=prices.index),
            'bb_lower': pd.Series(lower, index=prices.index)
        }
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """随机指标"""
        slowk, slowd = talib.STOCH(high.values, low.values, close.values, 
                                  fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
        return {
            'stoch_k': pd.Series(slowk, index=close.index),
            'stoch_d': pd.Series(slowd, index=close.index)
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """平均真实波幅"""
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=window)
        return pd.Series(atr, index=close.index)
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """平均方向性指数"""
        adx = talib.ADX(high.values, low.values, close.values, timeperiod=window)
        return pd.Series(adx, index=close.index)
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """商品通道指数"""
        cci = talib.CCI(high.values, low.values, close.values, timeperiod=window)
        return pd.Series(cci, index=close.index)
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """威廉指标"""
        willr = talib.WILLR(high.values, low.values, close.values, timeperiod=window)
        return pd.Series(willr, index=close.index)


class StatisticalFeatures:
    """统计特征计算器"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """计算不同周期的收益率"""
        features = {}
        for period in periods:
            features[f'return_{period}d'] = prices.pct_change(periods=period)
        return features
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
        """计算不同窗口的波动率"""
        returns = prices.pct_change()
        features = {}
        for window in windows:
            features[f'volatility_{window}d'] = returns.rolling(window=window).std() * np.sqrt(252)
        return features
    
    @staticmethod
    def calculate_momentum(prices: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
        """计算动量指标"""
        features = {}
        for window in windows:
            features[f'momentum_{window}d'] = prices / prices.shift(window) - 1
        return features
    
    @staticmethod
    def calculate_price_ratios(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算价格比率特征"""
        features = {}
        if all(col in data.columns for col in ['high', 'low', 'close', 'open']):
            features['high_low_ratio'] = data['high'] / data['low']
            features['close_open_ratio'] = data['close'] / data['open']
            features['high_close_ratio'] = data['high'] / data['close']
            features['low_close_ratio'] = data['low'] / data['close']
        return features
    
    @staticmethod
    def calculate_volume_features(data: pd.DataFrame, windows: List[int]) -> Dict[str, pd.Series]:
        """计算成交量特征"""
        features = {}
        if 'volume' in data.columns:
            volume = data['volume']
            for window in windows:
                features[f'volume_sma_{window}d'] = volume.rolling(window=window).mean()
                features[f'volume_ratio_{window}d'] = volume / volume.rolling(window=window).mean()
        return features
    
    @staticmethod
    def calculate_statistical_moments(prices: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
        """计算统计矩特征"""
        returns = prices.pct_change()
        features = {}
        
        for window in windows:
            rolling_returns = returns.rolling(window=window)
            features[f'skewness_{window}d'] = rolling_returns.skew()
            features[f'kurtosis_{window}d'] = rolling_returns.kurt()
            features[f'mean_{window}d'] = rolling_returns.mean()
            features[f'std_{window}d'] = rolling_returns.std()
        
        return features


class FeatureCache:
    """特征缓存管理器"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.access_times: Dict[str, datetime] = {}
    
    def get_cache_key(self, stock_code: str, start_date: datetime, end_date: datetime, 
                     feature_config: FeatureConfig) -> str:
        """生成缓存键"""
        config_hash = hash(str(sorted(feature_config.__dict__.items())))
        return f"{stock_code}_{start_date.date()}_{end_date.date()}_{config_hash}"
    
    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        """获取缓存的特征"""
        if cache_key in self.cache:
            self.access_times[cache_key] = datetime.utcnow()
            return self.cache[cache_key]['features']
        return None
    
    def set(self, cache_key: str, features: pd.DataFrame):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[cache_key] = {
            'features': features.copy(),
            'created_at': datetime.utcnow()
        }
        self.access_times[cache_key] = datetime.utcnow()
    
    def _evict_oldest(self):
        """清理最旧的缓存项"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()


class FeatureExtractor:
    """特征提取器主类"""
    
    def __init__(self, cache_enabled: bool = True):
        self.technical_indicators = TechnicalIndicators()
        self.statistical_features = StatisticalFeatures()
        self.cache = FeatureCache() if cache_enabled else None
        
        # 默认特征配置
        self.default_config = FeatureConfig(
            technical_indicators=[
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_12', 'ema_26',
                'rsi_14', 'macd', 'bb_20', 'stoch_14',
                'atr_14', 'adx_14', 'cci_14', 'willr_14'
            ],
            statistical_features=[
                'returns', 'volatility', 'momentum', 
                'price_ratios', 'volume_features', 'statistical_moments'
            ],
            time_windows=[5, 10, 20, 50],
            cache_enabled=True
        )
    
    def extract_features(self, stock_code: str, data: pd.DataFrame, 
                        config: Optional[FeatureConfig] = None) -> pd.DataFrame:
        """
        提取股票特征
        
        Args:
            stock_code: 股票代码
            data: 股票数据 (包含 open, high, low, close, volume 列)
            config: 特征配置
            
        Returns:
            包含所有特征的DataFrame
        """
        if config is None:
            config = self.default_config
        
        try:
            # 检查缓存
            if self.cache and config.cache_enabled:
                start_date = data.index.min()
                end_date = data.index.max()
                cache_key = self.cache.get_cache_key(stock_code, start_date, end_date, config)
                cached_features = self.cache.get(cache_key)
                if cached_features is not None:
                    logger.info(f"使用缓存特征: {stock_code}")
                    return cached_features
            
            # 验证数据
            self._validate_data(data)
            
            # 提取特征
            features = pd.DataFrame(index=data.index)
            
            # 技术指标特征
            tech_features = self._extract_technical_indicators(data, config)
            features = pd.concat([features, tech_features], axis=1)
            
            # 统计特征
            stat_features = self._extract_statistical_features(data, config)
            features = pd.concat([features, stat_features], axis=1)
            
            # 自定义特征
            if config.custom_features:
                custom_features = self._extract_custom_features(data, config.custom_features)
                features = pd.concat([features, custom_features], axis=1)
            
            # 清理特征
            features = self._clean_features(features)
            
            # 缓存结果
            if self.cache and config.cache_enabled:
                self.cache.set(cache_key, features)
            
            logger.info(f"特征提取完成: {stock_code}, 特征数量: {len(features.columns)}")
            return features
            
        except Exception as e:
            error_context = ErrorContext(stock_code=stock_code)
            raise DataError(
                message=f"特征提取失败: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=error_context,
                original_exception=e
            )
    
    def _validate_data(self, data: pd.DataFrame):
        """验证输入数据"""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise DataError(
                message=f"缺少必需的数据列: {missing_columns}",
                severity=ErrorSeverity.HIGH
            )
        
        if len(data) < 50:
            raise DataError(
                message=f"数据量不足，至少需要50个交易日，当前: {len(data)}",
                severity=ErrorSeverity.MEDIUM
            )
        
        # 检查数据质量
        for col in required_columns:
            if data[col].isnull().sum() > len(data) * 0.1:
                raise DataError(
                    message=f"列 {col} 缺失值过多: {data[col].isnull().sum()}/{len(data)}",
                    severity=ErrorSeverity.MEDIUM
                )
    
    def _extract_technical_indicators(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """提取技术指标特征"""
        features = pd.DataFrame(index=data.index)
        
        high, low, close, open_price = data['high'], data['low'], data['close'], data['open']
        
        for indicator in config.technical_indicators:
            try:
                if indicator.startswith('sma_'):
                    window = int(indicator.split('_')[1])
                    features[indicator] = self.technical_indicators.calculate_sma(close, window)
                
                elif indicator.startswith('ema_'):
                    window = int(indicator.split('_')[1])
                    features[indicator] = self.technical_indicators.calculate_ema(close, window)
                
                elif indicator.startswith('rsi_'):
                    window = int(indicator.split('_')[1])
                    features[indicator] = self.technical_indicators.calculate_rsi(close, window)
                
                elif indicator == 'macd':
                    macd_features = self.technical_indicators.calculate_macd(close)
                    for key, series in macd_features.items():
                        features[key] = series
                
                elif indicator.startswith('bb_'):
                    window = int(indicator.split('_')[1])
                    bb_features = self.technical_indicators.calculate_bollinger_bands(close, window)
                    for key, series in bb_features.items():
                        features[key] = series
                
                elif indicator.startswith('stoch_'):
                    k_period = int(indicator.split('_')[1])
                    stoch_features = self.technical_indicators.calculate_stochastic(high, low, close, k_period)
                    for key, series in stoch_features.items():
                        features[key] = series
                
                elif indicator.startswith('atr_'):
                    window = int(indicator.split('_')[1])
                    features[indicator] = self.technical_indicators.calculate_atr(high, low, close, window)
                
                elif indicator.startswith('adx_'):
                    window = int(indicator.split('_')[1])
                    features[indicator] = self.technical_indicators.calculate_adx(high, low, close, window)
                
                elif indicator.startswith('cci_'):
                    window = int(indicator.split('_')[1])
                    features[indicator] = self.technical_indicators.calculate_cci(high, low, close, window)
                
                elif indicator.startswith('willr_'):
                    window = int(indicator.split('_')[1])
                    features[indicator] = self.technical_indicators.calculate_williams_r(high, low, close, window)
                
            except Exception as e:
                logger.warning(f"计算技术指标 {indicator} 失败: {e}")
                continue
        
        return features
    
    def _extract_statistical_features(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """提取统计特征"""
        features = pd.DataFrame(index=data.index)
        close = data['close']
        
        for feature_type in config.statistical_features:
            try:
                if feature_type == 'returns':
                    return_features = self.statistical_features.calculate_returns(close, [1, 5, 10, 20])
                    for key, series in return_features.items():
                        features[key] = series
                
                elif feature_type == 'volatility':
                    vol_features = self.statistical_features.calculate_volatility(close, config.time_windows)
                    for key, series in vol_features.items():
                        features[key] = series
                
                elif feature_type == 'momentum':
                    momentum_features = self.statistical_features.calculate_momentum(close, config.time_windows)
                    for key, series in momentum_features.items():
                        features[key] = series
                
                elif feature_type == 'price_ratios':
                    ratio_features = self.statistical_features.calculate_price_ratios(data)
                    for key, series in ratio_features.items():
                        features[key] = series
                
                elif feature_type == 'volume_features' and 'volume' in data.columns:
                    volume_features = self.statistical_features.calculate_volume_features(data, config.time_windows)
                    for key, series in volume_features.items():
                        features[key] = series
                
                elif feature_type == 'statistical_moments':
                    moment_features = self.statistical_features.calculate_statistical_moments(close, config.time_windows)
                    for key, series in moment_features.items():
                        features[key] = series
                
            except Exception as e:
                logger.warning(f"计算统计特征 {feature_type} 失败: {e}")
                continue
        
        return features
    
    def _extract_custom_features(self, data: pd.DataFrame, custom_features: List[str]) -> pd.DataFrame:
        """提取自定义特征"""
        features = pd.DataFrame(index=data.index)
        
        for feature in custom_features:
            try:
                if feature == 'price_change_acceleration':
                    # 价格变化加速度
                    returns = data['close'].pct_change()
                    features[feature] = returns.diff()
                
                elif feature == 'volume_price_trend':
                    # 成交量价格趋势
                    if 'volume' in data.columns:
                        features[feature] = (data['close'].pct_change() * data['volume']).rolling(10).mean()
                
                elif feature == 'intraday_intensity':
                    # 日内强度指标
                    features[feature] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
                
                elif feature == 'price_position':
                    # 价格位置指标
                    features[feature] = (data['close'] - data['low'].rolling(20).min()) / (data['high'].rolling(20).max() - data['low'].rolling(20).min())
                
            except Exception as e:
                logger.warning(f"计算自定义特征 {feature} 失败: {e}")
                continue
        
        return features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """清理特征数据"""
        # 移除全为NaN的列
        features = features.dropna(axis=1, how='all')
        
        # 处理无穷值
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # 前向填充缺失值
        features = features.fillna(method='ffill')
        
        # 后向填充剩余缺失值
        features = features.fillna(method='bfill')
        
        # 移除仍有缺失值的行
        features = features.dropna()
        
        return features
    
    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """计算特征重要性"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # 确保特征和目标对齐
            aligned_features, aligned_target = features.align(target, join='inner', axis=0)
            
            if len(aligned_features) == 0:
                raise DataError("特征和目标数据无法对齐")
            
            # 计算互信息
            importance_scores = mutual_info_regression(
                aligned_features.fillna(0), 
                aligned_target.fillna(0)
            )
            
            importance = pd.Series(
                importance_scores, 
                index=aligned_features.columns
            ).sort_values(ascending=False)
            
            return importance
            
        except Exception as e:
            logger.warning(f"计算特征重要性失败: {e}")
            return pd.Series(index=features.columns, dtype=float)
    
    def select_features(self, features: pd.DataFrame, target: pd.Series, 
                       top_k: int = 50) -> List[str]:
        """选择最重要的特征"""
        importance = self.get_feature_importance(features, target)
        return importance.head(top_k).index.tolist()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": len(self.cache.cache),
            "max_size": self.cache.max_size,
            "hit_rate": len(self.cache.access_times) / max(len(self.cache.cache), 1)
        }