"""
数据预处理模块

包含异常值处理、CSRankNorm 标签变换和特征标准化功能
"""

from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm


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
                f"检测到 {extreme_mask.sum()} 个极端收益率（>50%），可能是权除息，已处理"
            )
        
        return data_processed


# CSRankNorm 常量
CS_RANK_MIN_STOCKS = 10
CS_RANK_CLIP_LOWER = 0.001
CS_RANK_CLIP_UPPER = 0.999


class CSRankNormTransformer:
    """截面排名标准化（Cross-Sectional Rank Normalization）

    每个交易日截面内：rank → percentile → clip → ppf(inverse normal)
    等价于 experiments/unified/label_transform.py 的实现。
    """

    def transform(self, data: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """
        对标签列做截面排名标准化

        Args:
            data: MultiIndex (instrument, datetime) 的 DataFrame
            label_col: 标签列名

        Returns:
            标签列已替换为 CSRankNorm 值的 DataFrame
        """
        if label_col not in data.columns:
            return data

        df = data.copy()
        date_level = self._resolve_date_level(df)
        if date_level is None:
            logger.warning("CSRankNorm: 无法确定日期层级，跳过")
            return df

        # Reset debug stats
        CSRankNormTransformer._debug_stats = {"total": 0, "too_few": 0, "all_same": 0, "ok": 0}

        if isinstance(df.index, pd.MultiIndex):
            # Debug: log index structure
            level0_sample = df.index.get_level_values(0)[:3].tolist()
            level1_sample = df.index.get_level_values(1)[:3].tolist()
            n_unique_0 = df.index.get_level_values(0).nunique()
            n_unique_1 = df.index.get_level_values(1).nunique()
            logger.info(
                f"CSRankNorm index debug: level0_sample={level0_sample}, level1_sample={level1_sample}, "
                f"n_unique_level0={n_unique_0}, n_unique_level1={n_unique_1}, date_level={date_level}"
            )
            # Debug: sample a few days' label distributions
            grouped = df.groupby(level=date_level)[label_col]
            sample_days = list(grouped.groups.keys())[:3]
            for day in sample_days:
                day_labels = grouped.get_group(day)
                logger.info(
                    f"CSRankNorm day={day}: n={len(day_labels)}, nunique={day_labels.nunique()}, "
                    f"min={day_labels.min():.6f}, max={day_labels.max():.6f}, "
                    f"values_sample={day_labels.head(5).tolist()}"
                )
            df[label_col] = df.groupby(level=date_level)[label_col].transform(
                self._rank_norm_single_day,
            )
        else:
            df[label_col] = df.groupby(date_level)[label_col].transform(
                self._rank_norm_single_day,
            )

        stats = CSRankNormTransformer._debug_stats
        logger.info(
            f"CSRankNorm debug: total_days={stats['total']}, "
            f"too_few={stats['too_few']}, all_same={stats['all_same']}, ok={stats['ok']}"
        )

        valid = df[label_col].notna().sum()
        logger.info(
            f"CSRankNorm 完成: {valid}/{len(df)} 有效, "
            f"均值={df[label_col].mean():.4f}, 标准差={df[label_col].std():.4f}",
        )
        return df

    _debug_stats = {"total": 0, "too_few": 0, "all_same": 0, "ok": 0}

    @classmethod
    def _rank_norm_single_day(cls, series: pd.Series) -> pd.Series:
        """单日截面: rank → percentile → clip → inverse normal"""
        cls._debug_stats["total"] += 1
        valid = series.dropna()
        if len(valid) < CS_RANK_MIN_STOCKS:
            cls._debug_stats["too_few"] += 1
            return pd.Series(np.nan, index=series.index)

        # 全重复值无法产生有意义的排名
        if valid.nunique() <= 1:
            cls._debug_stats["all_same"] += 1
            return pd.Series(0.0, index=series.index)

        cls._debug_stats["ok"] += 1
        ranked = series.rank(method="average", na_option="keep")
        n_valid = valid.count()
        percentile = (ranked - 0.5) / n_valid
        percentile = percentile.clip(CS_RANK_CLIP_LOWER, CS_RANK_CLIP_UPPER)
        return pd.Series(norm.ppf(percentile), index=series.index)

    @staticmethod
    def _resolve_date_level(df: pd.DataFrame):
        """确定日期层级名称或位置索引

        自动检测 MultiIndex 中哪一层是日期：
        - 优先按 index.names 匹配 'datetime' / 'date'
        - 若 names 为 None，按 dtype 检测（datetime64 类型的层即为日期层）
        """
        if isinstance(df.index, pd.MultiIndex):
            # 1. 按名称匹配
            for name in ("datetime", "date"):
                if name in df.index.names:
                    return name
            # 2. 按 dtype 检测日期层
            for i in range(df.index.nlevels):
                level_values = df.index.get_level_values(i)
                if pd.api.types.is_datetime64_any_dtype(level_values):
                    return df.index.names[i] if df.index.names[i] is not None else i
            # 3. 兜底：无法确定
            return None
        for name in ("date", "datetime"):
            if name in df.columns:
                return name
        return None


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
            logger.warning("未指定特征列，返回原始数据")
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


class CrossSectionalNeutralizer:
    """截面中性化处理器"""

    def __init__(self):
        pass

    def transform(self, data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        对指定特征进行截面中性化（减去截面均值）
        
        Args:
            data: 输入数据 DataFrame
            feature_cols: 需要中性化的特征列名列表
        
        Returns:
            中性化后的 DataFrame
        """
        if not feature_cols:
            return data

        df = data.copy()
        
        # 确定日期列 —— 复用 CSRankNormTransformer 的检测逻辑
        date_level = CSRankNormTransformer._resolve_date_level(df)
        
        if date_level is None:
            logger.warning("无法确定日期列，跳过截面中性化")
            return df
            
        logger.info(f"开始截面中性化，特征数: {len(feature_cols)}, 日期列: {date_level}")
        
        # 只处理存在的特征列
        valid_features = [c for c in feature_cols if c in df.columns]
        if not valid_features:
            return df
            
        try:
            # 批量计算均值并相减（比逐列循环快）
            if isinstance(df.index, pd.MultiIndex):
                # MultiIndex groupby
                means = df.groupby(level=date_level)[valid_features].transform('mean')
            else:
                # Column groupby
                means = df.groupby(date_level)[valid_features].transform('mean')
            
            # 执行中性化
            df[valid_features] = df[valid_features] - means
            logger.info("截面中性化完成")
            
        except Exception as e:
            logger.error(f"截面中性化失败: {e}")
            # 降级为逐列处理
            for col in valid_features:
                try:
                    if isinstance(df.index, pd.MultiIndex):
                        mean = df.groupby(level=date_level)[col].transform('mean')
                    else:
                        mean = df.groupby(date_level)[col].transform('mean')
                    df[col] = df[col] - mean
                except Exception as sub_e:
                    logger.warning(f"特征 {col} 中性化失败: {sub_e}")
        
        return df
