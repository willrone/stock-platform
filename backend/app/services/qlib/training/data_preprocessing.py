"""
数据预处理模块

包含异常值处理和特征标准化功能
"""

from typing import List

import numpy as np
import pandas as pd
from loguru import logger


class OutlierHandler:
    """异常值处理器 - 对收益率标签进行Winsorize处理"""
    
    def __init__(self, method: str = "winsorize", lower_percentile: float = 0.01, upper_percentile: float = 0.99):
        """
        初始化异常值处理器
        
        Args:
            method: 处理方法，'winsorize' 或 'clip'
            lower_percentile: 下分位数（用于Winsorize）
            upper_percentile: 上分位数（用于Winsorize）
        """
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
    
    def handle_label_outliers(self, data: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """
        处理标签中的异常值
        
        Args:
            data: 数据DataFrame
            label_col: 标签列名
        
        Returns:
            处理后的DataFrame
        """
        if label_col not in data.columns:
            return data
        
        data_processed = data.copy()
        label_values = data_processed[label_col]
        
        # 移除NaN和无穷值
        valid_mask = pd.notna(label_values) & np.isfinite(label_values)
        if not valid_mask.any():
            logger.warning(f"标签列 {label_col} 没有有效值")
            return data_processed
        
        valid_labels = label_values[valid_mask]
        
        if self.method == "winsorize":
            # Winsorize方法：将极端值截断到分位数
            lower_bound = valid_labels.quantile(self.lower_percentile)
            upper_bound = valid_labels.quantile(self.upper_percentile)
            
            # 记录异常值数量
            outliers_lower = (label_values < lower_bound).sum()
            outliers_upper = (label_values > upper_bound).sum()
            
            if outliers_lower > 0 or outliers_upper > 0:
                logger.info(
                    f"标签异常值处理: 下界={lower_bound:.6f} (异常值={outliers_lower}), "
                    f"上界={upper_bound:.6f} (异常值={outliers_upper})"
                )
            
            # 截断到分位数
            data_processed[label_col] = label_values.clip(
                lower=lower_bound, upper=upper_bound
            )
            
        elif self.method == "clip":
            # Clip方法：使用Z-score方法检测异常值
            mean = valid_labels.mean()
            std = valid_labels.std()
            
            if std > 0:
                z_scores = np.abs((label_values - mean) / std)
                # 使用3倍标准差作为阈值
                threshold = 3.0
                outliers = z_scores > threshold
                
                if outliers.sum() > 0:
                    logger.info(
                        f"标签异常值处理: 使用Z-score方法，检测到 {outliers.sum()} 个异常值"
                    )
                    # 将异常值截断到阈值
                    data_processed.loc[outliers, label_col] = np.sign(
                        label_values[outliers] - mean
                    ) * threshold * std + mean
        
        # 处理极端价格变化（可能是除权除息）
        # 如果收益率超过50%，标记为可疑
        extreme_mask = np.abs(data_processed[label_col]) > 0.5
        if extreme_mask.sum() > 0:
            logger.warning(
                f"检测到 {extreme_mask.sum()} 个极端收益率（>50%），可能是���权除息，已处理"
            )
        
        return data_processed


class RobustFeatureScaler:
    """鲁棒特征标准化器（时间序列安全）"""
    
    def __init__(self):
        try:
            from sklearn.preprocessing import RobustScaler
            self.RobustScaler = RobustScaler
        except ImportError:
            logger.warning("sklearn不可用，特征标准化将使用简单标准化")
            self.RobustScaler = None
        
        self.scalers = {}
        self.fitted = False
        self.feature_cols = None
    
    def fit_transform(
        self, data: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """按时间序列方式标准化（避免未来信息泄漏）"""
        if self.RobustScaler is None:
            logger.warning("sklearn不可用，跳过特征标准化")
            return data
        
        data_scaled = data.copy()
        self.feature_cols = feature_cols
        
        # 确保数据按时间排序
        if isinstance(data.index, pd.MultiIndex):
            data_scaled = data_scaled.sort_index()
        elif isinstance(data.index, pd.DatetimeIndex):
            data_scaled = data_scaled.sort_index()
        
        for col in feature_cols:
            if col not in data_scaled.columns:
                continue
            
            # 跳过标签列和非数值列
            if col == "label" or not pd.api.types.is_numeric_dtype(data_scaled[col]):
                continue
            
            try:
                scaler = self.RobustScaler()
                # 只使用历史数据拟合（时间序列安全）
                col_values = data_scaled[col].values.reshape(-1, 1)
                # 移除NaN值进行拟合
                valid_mask = ~np.isnan(col_values.flatten())
                if valid_mask.sum() > 0:
                    scaler.fit(col_values[valid_mask])
                    # 转换所有值（包括NaN，NaN会保持为NaN）
                    data_scaled[col] = scaler.transform(col_values).flatten()
                    self.scalers[col] = scaler
                else:
                    logger.warning(f"列 {col} 全为NaN，跳过标准化")
            except Exception as e:
                logger.warning(f"标准化列 {col} 时出错: {e}，跳过该列")
                continue
        
        self.fitted = True
        logger.info(f"特征标准化完成，标准化了 {len(self.scalers)} 个特征列")
        return data_scaled
    
    def transform(
        self, data: pd.DataFrame, feature_cols: List[str] = None
    ) -> pd.DataFrame:
        """转换新数据"""
        if not self.fitted:
            raise ValueError("Scaler尚未拟合，请先调用fit_transform")
        
        if feature_cols is None:
            feature_cols = self.feature_cols
        
        if feature_cols is None:
            logger.warning("未��定特征列，返回原始数据")
            return data
        
        if self.RobustScaler is None:
            return data
        
        data_scaled = data.copy()
        
        for col in feature_cols:
            if col not in data_scaled.columns or col not in self.scalers:
                continue
            
            try:
                col_values = data_scaled[col].values.reshape(-1, 1)
                data_scaled[col] = self.scalers[col].transform(col_values).flatten()
            except Exception as e:
                logger.warning(f"转换列 {col} 时出错: {e}，保持原值")
                continue
        
        return data_scaled
