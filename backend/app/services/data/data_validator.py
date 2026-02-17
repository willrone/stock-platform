"""
数据验证和清洗服务
实现数据质量检查、异常数据过滤和数据清洗功能
"""

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class ValidationLevel(Enum):
    """验证级别"""

    STRICT = "strict"  # 严格验证，拒绝任何异常数据
    MODERATE = "moderate"  # 中等验证，修复可修复的异常
    LENIENT = "lenient"  # 宽松验证，只过滤明显错误的数据


class ValidationRule(Enum):
    """验证规则"""

    POSITIVE_PRICES = "positive_prices"
    PRICE_RELATIONSHIPS = "price_relationships"
    VOLUME_VALIDATION = "volume_validation"
    DATE_CONTINUITY = "date_continuity"
    PRICE_VOLATILITY = "price_volatility"
    DUPLICATE_RECORDS = "duplicate_records"
    MISSING_VALUES = "missing_values"
    OUTLIER_DETECTION = "outlier_detection"


@dataclass
class ValidationResult:
    """验证结果"""

    is_valid: bool
    cleaned_data: Optional[pd.DataFrame]
    issues_found: List[Dict[str, Any]]
    records_removed: int
    records_modified: int
    quality_score: float  # 0-1之间的质量评分


@dataclass
class ValidationIssue:
    """验证问题"""

    rule: ValidationRule
    severity: str  # 'error', 'warning', 'info'
    message: str
    affected_records: int
    details: Dict[str, Any]


class DataValidator:
    """数据验证器"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.enabled_rules = self._get_default_rules()

        # 验证阈值配置
        self.thresholds = {
            "max_daily_change": 0.5,  # 最大单日涨跌幅50%
            "min_price": 0.01,  # 最小价格
            "max_price": 10000.0,  # 最大价格
            "min_volume": 0,  # 最小成交量
            "max_volume": 1e12,  # 最大成交量
            "outlier_std_threshold": 3.0,  # 异常值标准差阈值
            "missing_data_threshold": 0.1,  # 缺失数据阈值10%
        }

    def _get_default_rules(self) -> List[ValidationRule]:
        """获取默认验证规则"""
        if self.validation_level == ValidationLevel.STRICT:
            return list(ValidationRule)
        elif self.validation_level == ValidationLevel.MODERATE:
            return [
                ValidationRule.POSITIVE_PRICES,
                ValidationRule.PRICE_RELATIONSHIPS,
                ValidationRule.VOLUME_VALIDATION,
                ValidationRule.DUPLICATE_RECORDS,
                ValidationRule.MISSING_VALUES,
                ValidationRule.PRICE_VOLATILITY,
            ]
        else:  # LENIENT
            return [
                ValidationRule.POSITIVE_PRICES,
                ValidationRule.PRICE_RELATIONSHIPS,
                ValidationRule.DUPLICATE_RECORDS,
                ValidationRule.MISSING_VALUES,
            ]

    def validate_stock_data(
        self, df: pd.DataFrame, stock_code: str
    ) -> ValidationResult:
        """验证股票数据"""
        if df is None or df.empty:
            return ValidationResult(
                is_valid=False,
                cleaned_data=None,
                issues_found=[],
                records_removed=0,
                records_modified=0,
                quality_score=0.0,
            )

        original_count = len(df)
        cleaned_df = df.copy()
        issues = []
        records_modified = 0

        # 执行各项验证规则
        for rule in self.enabled_rules:
            try:
                result = self._apply_validation_rule(cleaned_df, rule, stock_code)
                cleaned_df = result["data"]
                issues.extend(result["issues"])
                records_modified += result["modified_count"]

            except Exception as e:
                logger.error(f"验证规则 {rule.value} 执行失败: {e}")
                issues.append(
                    {
                        "rule": rule.value,
                        "severity": "error",
                        "message": f"验证规则执行失败: {str(e)}",
                        "affected_records": 0,
                        "details": {},
                    }
                )

        records_removed = original_count - len(cleaned_df)
        quality_score = self._calculate_quality_score(
            original_count, records_removed, issues
        )

        is_valid = quality_score >= 0.7  # 质量评分阈值

        return ValidationResult(
            is_valid=is_valid,
            cleaned_data=cleaned_df if len(cleaned_df) > 0 else None,
            issues_found=issues,
            records_removed=records_removed,
            records_modified=records_modified,
            quality_score=quality_score,
        )

    def _apply_validation_rule(
        self, df: pd.DataFrame, rule: ValidationRule, stock_code: str
    ) -> Dict[str, Any]:
        """应用单个验证规则"""
        issues = []
        modified_count = 0

        if rule == ValidationRule.POSITIVE_PRICES:
            df, rule_issues, rule_modified = self._validate_positive_prices(
                df, stock_code
            )
        elif rule == ValidationRule.PRICE_RELATIONSHIPS:
            df, rule_issues, rule_modified = self._validate_price_relationships(
                df, stock_code
            )
        elif rule == ValidationRule.VOLUME_VALIDATION:
            df, rule_issues, rule_modified = self._validate_volume(df, stock_code)
        elif rule == ValidationRule.DATE_CONTINUITY:
            df, rule_issues, rule_modified = self._validate_date_continuity(
                df, stock_code
            )
        elif rule == ValidationRule.PRICE_VOLATILITY:
            df, rule_issues, rule_modified = self._validate_price_volatility(
                df, stock_code
            )
        elif rule == ValidationRule.DUPLICATE_RECORDS:
            df, rule_issues, rule_modified = self._validate_duplicates(df, stock_code)
        elif rule == ValidationRule.MISSING_VALUES:
            df, rule_issues, rule_modified = self._validate_missing_values(
                df, stock_code
            )
        elif rule == ValidationRule.OUTLIER_DETECTION:
            df, rule_issues, rule_modified = self._detect_outliers(df, stock_code)
        else:
            rule_issues = []
            rule_modified = 0

        issues.extend(rule_issues)
        modified_count += rule_modified

        return {"data": df, "issues": issues, "modified_count": modified_count}

    def _validate_positive_prices(
        self, df: pd.DataFrame, stock_code: str
    ) -> Tuple[pd.DataFrame, List[Dict], int]:
        """验证价格为正数"""
        issues = []
        modified_count = 0

        price_columns = ["open", "high", "low", "close"]

        for col in price_columns:
            if col in df.columns:
                invalid_mask = (df[col] <= 0) | (df[col] > self.thresholds["max_price"])
                invalid_count = invalid_mask.sum()

                if invalid_count > 0:
                    issues.append(
                        {
                            "rule": ValidationRule.POSITIVE_PRICES.value,
                            "severity": "error",
                            "message": f"发现无效价格数据: {col}",
                            "affected_records": invalid_count,
                            "details": {
                                "column": col,
                                "min_value": df[col].min(),
                                "max_value": df[col].max(),
                            },
                        }
                    )

                    # 移除无效记录
                    df = df[~invalid_mask]

        return df, issues, modified_count

    def _validate_price_relationships(
        self, df: pd.DataFrame, stock_code: str
    ) -> Tuple[pd.DataFrame, List[Dict], int]:
        """验证价格关系的合理性"""
        issues = []
        modified_count = 0

        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            # 检查高价 >= 低价
            invalid_high_low = df["high"] < df["low"]
            if invalid_high_low.any():
                issues.append(
                    {
                        "rule": ValidationRule.PRICE_RELATIONSHIPS.value,
                        "severity": "error",
                        "message": "发现高价低于低价的记录",
                        "affected_records": invalid_high_low.sum(),
                        "details": {},
                    }
                )
                df = df[~invalid_high_low]

            # 检查开盘价和收盘价是否在高低价范围内
            invalid_open = (df["open"] > df["high"]) | (df["open"] < df["low"])
            invalid_close = (df["close"] > df["high"]) | (df["close"] < df["low"])

            if invalid_open.any():
                issues.append(
                    {
                        "rule": ValidationRule.PRICE_RELATIONSHIPS.value,
                        "severity": "error",
                        "message": "发现开盘价超出高低价范围的记录",
                        "affected_records": invalid_open.sum(),
                        "details": {},
                    }
                )
                df = df[~invalid_open]

            if invalid_close.any():
                issues.append(
                    {
                        "rule": ValidationRule.PRICE_RELATIONSHIPS.value,
                        "severity": "error",
                        "message": "发现收盘价超出高低价范围的记录",
                        "affected_records": invalid_close.sum(),
                        "details": {},
                    }
                )
                df = df[~invalid_close]

        return df, issues, modified_count

    def _validate_volume(
        self, df: pd.DataFrame, stock_code: str
    ) -> Tuple[pd.DataFrame, List[Dict], int]:
        """验证成交量"""
        issues = []
        modified_count = 0

        if "volume" in df.columns:
            invalid_volume = (df["volume"] < self.thresholds["min_volume"]) | (
                df["volume"] > self.thresholds["max_volume"]
            )

            if invalid_volume.any():
                issues.append(
                    {
                        "rule": ValidationRule.VOLUME_VALIDATION.value,
                        "severity": "error",
                        "message": "发现无效成交量数据",
                        "affected_records": invalid_volume.sum(),
                        "details": {
                            "min_volume": df["volume"].min(),
                            "max_volume": df["volume"].max(),
                        },
                    }
                )
                df = df[~invalid_volume]

        return df, issues, modified_count

    def _validate_date_continuity(
        self, df: pd.DataFrame, stock_code: str
    ) -> Tuple[pd.DataFrame, List[Dict], int]:
        """验证日期连续性"""
        issues = []
        modified_count = 0

        if "date" in df.columns and len(df) > 1:
            df_sorted = df.sort_values("date")
            date_diff = df_sorted["date"].diff()

            # 检查是否有重复日期
            duplicate_dates = df_sorted["date"].duplicated()
            if duplicate_dates.any():
                issues.append(
                    {
                        "rule": ValidationRule.DATE_CONTINUITY.value,
                        "severity": "warning",
                        "message": "发现重复日期",
                        "affected_records": duplicate_dates.sum(),
                        "details": {},
                    }
                )

            # 检查异常的日期间隔（超过30天）
            large_gaps = date_diff > timedelta(days=30)
            if large_gaps.any():
                issues.append(
                    {
                        "rule": ValidationRule.DATE_CONTINUITY.value,
                        "severity": "info",
                        "message": "发现较大的日期间隔",
                        "affected_records": large_gaps.sum(),
                        "details": {
                            "max_gap_days": date_diff.max().days
                            if date_diff.max()
                            else 0
                        },
                    }
                )

        return df, issues, modified_count

    def _validate_price_volatility(
        self, df: pd.DataFrame, stock_code: str
    ) -> Tuple[pd.DataFrame, List[Dict], int]:
        """验证价格波动合理性"""
        issues = []
        modified_count = 0

        if "close" in df.columns and len(df) > 1:
            df_sorted = df.sort_values("date")
            price_change = df_sorted["close"].pct_change().abs()

            extreme_change = price_change > self.thresholds["max_daily_change"]
            if extreme_change.any():
                issues.append(
                    {
                        "rule": ValidationRule.PRICE_VOLATILITY.value,
                        "severity": "warning",
                        "message": f"发现极端价格波动（>{self.thresholds['max_daily_change']:.1%}）",
                        "affected_records": extreme_change.sum(),
                        "details": {
                            "max_change": price_change.max(),
                            "avg_change": price_change.mean(),
                        },
                    }
                )

                # 根据验证级别决定是否移除
                if self.validation_level == ValidationLevel.STRICT:
                    df = df_sorted[~extreme_change]

        return df, issues, modified_count

    def _validate_duplicates(
        self, df: pd.DataFrame, stock_code: str
    ) -> Tuple[pd.DataFrame, List[Dict], int]:
        """验证重复记录"""
        issues = []
        modified_count = 0

        if "date" in df.columns:
            duplicate_mask = df.duplicated(subset=["date"], keep="first")
            duplicate_count = duplicate_mask.sum()

            if duplicate_count > 0:
                issues.append(
                    {
                        "rule": ValidationRule.DUPLICATE_RECORDS.value,
                        "severity": "warning",
                        "message": "发现重复记录",
                        "affected_records": duplicate_count,
                        "details": {},
                    }
                )

                # 移除重复记录
                df = df[~duplicate_mask]

        return df, issues, modified_count

    def _validate_missing_values(
        self, df: pd.DataFrame, stock_code: str
    ) -> Tuple[pd.DataFrame, List[Dict], int]:
        """验证缺失值"""
        issues = []
        modified_count = 0

        required_columns = ["date", "open", "high", "low", "close", "volume"]

        for col in required_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_ratio = missing_count / len(df)

                if missing_count > 0:
                    severity = (
                        "error"
                        if missing_ratio > self.thresholds["missing_data_threshold"]
                        else "warning"
                    )

                    issues.append(
                        {
                            "rule": ValidationRule.MISSING_VALUES.value,
                            "severity": severity,
                            "message": f"发现缺失值: {col}",
                            "affected_records": missing_count,
                            "details": {"column": col, "missing_ratio": missing_ratio},
                        }
                    )

                    # 移除有缺失值的记录
                    df = df.dropna(subset=[col])

        return df, issues, modified_count

    def _detect_outliers(
        self, df: pd.DataFrame, stock_code: str
    ) -> Tuple[pd.DataFrame, List[Dict], int]:
        """检测异常值"""
        issues = []
        modified_count = 0

        numeric_columns = ["open", "high", "low", "close", "volume"]

        for col in numeric_columns:
            if col in df.columns:
                # 使用Z-score方法检测异常值
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.thresholds["outlier_std_threshold"]
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    issues.append(
                        {
                            "rule": ValidationRule.OUTLIER_DETECTION.value,
                            "severity": "info",
                            "message": f"发现异常值: {col}",
                            "affected_records": outlier_count,
                            "details": {
                                "column": col,
                                "max_z_score": z_scores.max(),
                                "outlier_ratio": outlier_count / len(df),
                            },
                        }
                    )

                    # 根据验证级别决定是否移除异常值
                    if self.validation_level == ValidationLevel.STRICT:
                        df = df[~outliers]

        return df, issues, modified_count

    def _calculate_quality_score(
        self, original_count: int, removed_count: int, issues: List[Dict]
    ) -> float:
        """计算数据质量评分"""
        if original_count == 0:
            return 0.0

        # 基础分数：基于保留的数据比例
        retention_ratio = (original_count - removed_count) / original_count
        base_score = retention_ratio

        # 根据问题严重程度扣分
        error_penalty = sum(1 for issue in issues if issue["severity"] == "error") * 0.1
        warning_penalty = (
            sum(1 for issue in issues if issue["severity"] == "warning") * 0.05
        )

        quality_score = max(0.0, base_score - error_penalty - warning_penalty)

        return min(1.0, quality_score)

    def get_validation_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """获取验证摘要"""
        return {
            "is_valid": result.is_valid,
            "quality_score": result.quality_score,
            "records_processed": len(result.cleaned_data)
            if result.cleaned_data is not None
            else 0,
            "records_removed": result.records_removed,
            "records_modified": result.records_modified,
            "issues_by_severity": {
                "error": len(
                    [i for i in result.issues_found if i["severity"] == "error"]
                ),
                "warning": len(
                    [i for i in result.issues_found if i["severity"] == "warning"]
                ),
                "info": len(
                    [i for i in result.issues_found if i["severity"] == "info"]
                ),
            },
            "validation_level": self.validation_level.value,
            "enabled_rules": [rule.value for rule in self.enabled_rules],
        }
