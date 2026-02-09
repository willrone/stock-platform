"""
数据质量验证器

提供股票数据的质量验证功能，包括：
- 缺失值检测
- 异常值检测
- 时间序列连续性检查
- 数据一致性验证
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


class ValidationSeverity(Enum):
    """验证问题严重程度"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """验证问题记录

    Attributes:
        severity: 问题严重程度
        category: 问题类别
        message: 问题描述
        affected_rows: 受影响的行数
        affected_columns: 受影响的列
        details: 详细信息
    """

    severity: ValidationSeverity
    category: str
    message: str
    affected_rows: int = 0
    affected_columns: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """验证报告

    Attributes:
        is_valid: 数据是否通过验证
        total_rows: 总行数
        total_columns: 总列数
        issues: 问题列表
        summary: 摘要统计
    """

    is_valid: bool
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        """添加问题"""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.is_valid = False

    def get_issues_by_severity(
        self, severity: ValidationSeverity
    ) -> List[ValidationIssue]:
        """按严重程度获取问题"""
        return [i for i in self.issues if i.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "issues": [
                {
                    "severity": i.severity.value,
                    "category": i.category,
                    "message": i.message,
                    "affected_rows": i.affected_rows,
                    "affected_columns": i.affected_columns,
                }
                for i in self.issues
            ],
            "summary": self.summary,
        }


class DataQualityValidator:
    """数据质量验证器

    提供全面的数据质量检查功能。

    Attributes:
        missing_threshold: 缺失值比例阈值
        outlier_std_threshold: 异常值标准差阈值
        max_price_change: 最大允许价格变化比例
        required_columns: 必需的列
    """

    # 默认必需列
    DEFAULT_REQUIRED_COLUMNS = ["open", "close", "high", "low", "volume"]

    # 带 Qlib 前缀的必需列
    QLIB_REQUIRED_COLUMNS = ["$open", "$close", "$high", "$low", "$volume"]

    def __init__(
        self,
        missing_threshold: float = 0.1,
        outlier_std_threshold: float = 5.0,
        max_price_change: float = 0.2,
        required_columns: Optional[List[str]] = None,
    ):
        """初始化数据质量验证器

        Args:
            missing_threshold: 缺失值比例阈值，超过此比例将报告警告
            outlier_std_threshold: 异常值标准差阈值
            max_price_change: 单日最大允许价格变化比例（默认 20%）
            required_columns: 必需的列列表
        """
        self.missing_threshold = missing_threshold
        self.outlier_std_threshold = outlier_std_threshold
        self.max_price_change = max_price_change
        self.required_columns = required_columns

    def validate(
        self,
        df: pd.DataFrame,
        check_missing: bool = True,
        check_outliers: bool = True,
        check_continuity: bool = True,
        check_consistency: bool = True,
        check_price_logic: bool = True,
    ) -> ValidationReport:
        """执行完整的数据质量验证

        Args:
            df: 待验证的 DataFrame
            check_missing: 是否检查缺失值
            check_outliers: 是否检查异常值
            check_continuity: 是否检查时间序列连续性
            check_consistency: 是否检查数据一致性
            check_price_logic: 是否检查价格逻辑

        Returns:
            验证报告
        """
        report = ValidationReport(
            is_valid=True,
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        if df.empty:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="empty_data",
                    message="数据为空",
                )
            )
            return report

        # 检查必需列
        self._check_required_columns(df, report)

        # 检查缺失值
        if check_missing:
            self._check_missing_values(df, report)

        # 检查异常值
        if check_outliers:
            self._check_outliers(df, report)

        # 检查时间序列连续性
        if check_continuity:
            self._check_time_continuity(df, report)

        # 检查数据一致性
        if check_consistency:
            self._check_data_consistency(df, report)

        # 检查价格逻辑
        if check_price_logic:
            self._check_price_logic(df, report)

        # 生成摘要
        report.summary = self._generate_summary(df, report)

        return report

    def _get_required_columns(self, df: pd.DataFrame) -> List[str]:
        """获取必需列列表，自动检测是否使用 Qlib 前缀"""
        if self.required_columns:
            return self.required_columns

        # 检测是否使用 Qlib 前缀
        has_qlib_prefix = any(col.startswith("$") for col in df.columns)
        if has_qlib_prefix:
            return self.QLIB_REQUIRED_COLUMNS
        return self.DEFAULT_REQUIRED_COLUMNS

    def _check_required_columns(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """检查必需列是否存在"""
        required = self._get_required_columns(df)
        missing_cols = [col for col in required if col not in df.columns]

        if missing_cols:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="missing_columns",
                    message=f"缺少必需列: {missing_cols}",
                    affected_columns=missing_cols,
                )
            )

    def _check_missing_values(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """检查缺失值"""
        missing_stats = df.isnull().sum()
        missing_ratio = missing_stats / len(df)

        for col in df.columns:
            ratio = missing_ratio[col]
            count = missing_stats[col]

            if count == 0:
                continue

            if ratio > self.missing_threshold:
                severity = ValidationSeverity.WARNING
                if ratio > 0.5:
                    severity = ValidationSeverity.ERROR

                report.add_issue(
                    ValidationIssue(
                        severity=severity,
                        category="missing_values",
                        message=f"列 '{col}' 缺失值比例过高: {ratio:.2%}",
                        affected_rows=int(count),
                        affected_columns=[col],
                        details={"missing_ratio": ratio, "missing_count": int(count)},
                    )
                )
            elif count > 0:
                report.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="missing_values",
                        message=f"列 '{col}' 存在 {count} 个缺失值 ({ratio:.2%})",
                        affected_rows=int(count),
                        affected_columns=[col],
                        details={"missing_ratio": ratio, "missing_count": int(count)},
                    )
                )

    def _check_outliers(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """检查异常值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) < 10:
                continue

            mean = values.mean()
            std = values.std()

            if std == 0:
                continue

            z_scores = np.abs((values - mean) / std)
            outlier_mask = z_scores > self.outlier_std_threshold
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                outlier_ratio = outlier_count / len(values)
                severity = (
                    ValidationSeverity.WARNING
                    if outlier_ratio < 0.01
                    else ValidationSeverity.ERROR
                )

                report.add_issue(
                    ValidationIssue(
                        severity=severity,
                        category="outliers",
                        message=f"列 '{col}' 检测到 {outlier_count} 个异常值 "
                        f"(>{self.outlier_std_threshold}σ)",
                        affected_rows=int(outlier_count),
                        affected_columns=[col],
                        details={
                            "outlier_count": int(outlier_count),
                            "outlier_ratio": outlier_ratio,
                            "mean": float(mean),
                            "std": float(std),
                            "threshold": self.outlier_std_threshold,
                        },
                    )
                )

    def _check_time_continuity(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """检查时间序列连续性"""
        # 获取日期索引
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values(0)
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"])
        else:
            # 无法检查时间连续性
            return

        dates = pd.Series(dates).drop_duplicates().sort_values()

        if len(dates) < 2:
            return

        # 计算日期间隔
        date_diffs = dates.diff().dropna()

        # 检测异常间隔（超过 5 个交易日）
        max_gap = timedelta(days=7)  # 考虑周末和节假日
        large_gaps = date_diffs[date_diffs > max_gap]

        if len(large_gaps) > 0:
            gap_details = []
            for idx in large_gaps.index:
                if idx > 0:
                    prev_date = dates.iloc[dates.index.get_loc(idx) - 1]
                    curr_date = dates.iloc[dates.index.get_loc(idx)]
                    gap_details.append(
                        {
                            "from": str(prev_date.date()) if hasattr(prev_date, 'date') else str(prev_date),
                            "to": str(curr_date.date()) if hasattr(curr_date, 'date') else str(curr_date),
                            "days": large_gaps[idx].days,
                        }
                    )

            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="time_continuity",
                    message=f"检测到 {len(large_gaps)} 个时间间隔异常（>7天）",
                    affected_rows=len(large_gaps),
                    details={"gaps": gap_details[:10]},  # 只显示前 10 个
                )
            )

    def _check_data_consistency(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """检查数据一致性"""
        # 检查无穷值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                report.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="infinite_values",
                        message=f"列 '{col}' 包含 {inf_count} 个无穷值",
                        affected_rows=int(inf_count),
                        affected_columns=[col],
                    )
                )

        # 检查负值（价格和成交量不应为负）
        price_cols = self._find_price_columns(df)
        for col in price_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    report.add_issue(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="negative_values",
                            message=f"列 '{col}' 包含 {neg_count} 个负值",
                            affected_rows=int(neg_count),
                            affected_columns=[col],
                        )
                    )

    def _check_price_logic(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """检查价格逻辑"""
        # 获取价格列
        open_col = self._find_column(df, ["open", "$open"])
        close_col = self._find_column(df, ["close", "$close"])
        high_col = self._find_column(df, ["high", "$high"])
        low_col = self._find_column(df, ["low", "$low"])

        if not all([open_col, close_col, high_col, low_col]):
            return

        # 检查 high >= low
        invalid_hl = df[high_col] < df[low_col]
        invalid_hl_count = invalid_hl.sum()
        if invalid_hl_count > 0:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="price_logic",
                    message=f"检测到 {invalid_hl_count} 行 high < low",
                    affected_rows=int(invalid_hl_count),
                    affected_columns=[high_col, low_col],
                )
            )

        # 检查 high >= open, close 且 low <= open, close
        invalid_high = (df[high_col] < df[open_col]) | (df[high_col] < df[close_col])
        invalid_high_count = invalid_high.sum()
        if invalid_high_count > 0:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="price_logic",
                    message=f"检测到 {invalid_high_count} 行 high 小于 open 或 close",
                    affected_rows=int(invalid_high_count),
                    affected_columns=[high_col, open_col, close_col],
                )
            )

        invalid_low = (df[low_col] > df[open_col]) | (df[low_col] > df[close_col])
        invalid_low_count = invalid_low.sum()
        if invalid_low_count > 0:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="price_logic",
                    message=f"检测到 {invalid_low_count} 行 low 大于 open 或 close",
                    affected_rows=int(invalid_low_count),
                    affected_columns=[low_col, open_col, close_col],
                )
            )

        # 检查价格变化是否过大（可能是除权除息未处理）
        if close_col in df.columns:
            close_pct_change = df[close_col].pct_change().abs()
            extreme_changes = close_pct_change > self.max_price_change
            extreme_count = extreme_changes.sum()

            if extreme_count > 0:
                report.add_issue(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="extreme_price_change",
                        message=f"检测到 {extreme_count} 行价格变化超过 "
                        f"{self.max_price_change:.0%}，可能是除权除息",
                        affected_rows=int(extreme_count),
                        affected_columns=[close_col],
                        details={
                            "threshold": self.max_price_change,
                            "max_change": float(close_pct_change.max()),
                        },
                    )
                )

    def _find_column(
        self, df: pd.DataFrame, candidates: List[str]
    ) -> Optional[str]:
        """查找列名"""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _find_price_columns(self, df: pd.DataFrame) -> List[str]:
        """查找价格相关列"""
        price_patterns = [
            "open",
            "close",
            "high",
            "low",
            "volume",
            "$open",
            "$close",
            "$high",
            "$low",
            "$volume",
        ]
        return [col for col in df.columns if col.lower() in price_patterns]

    def _generate_summary(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> Dict[str, Any]:
        """生成验证摘要"""
        return {
            "total_issues": len(report.issues),
            "errors": len(report.get_issues_by_severity(ValidationSeverity.ERROR)),
            "warnings": len(report.get_issues_by_severity(ValidationSeverity.WARNING)),
            "info": len(report.get_issues_by_severity(ValidationSeverity.INFO)),
            "data_shape": {"rows": len(df), "columns": len(df.columns)},
            "missing_total": int(df.isnull().sum().sum()),
            "missing_ratio": float(df.isnull().sum().sum() / df.size)
            if df.size > 0
            else 0,
        }

    def fix_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """修复缺失值

        Args:
            df: 输入 DataFrame
            method: 填充方法，'ffill'（前向填充）、'bfill'（后向填充）、
                   'mean'（均值填充）、'median'（中位数填充）、'drop'（删除）
            columns: 要处理的列，None 表示所有列

        Returns:
            处理后的 DataFrame
        """
        df = df.copy()
        cols = columns or df.columns.tolist()

        if method == "ffill":
            df[cols] = df[cols].ffill()
        elif method == "bfill":
            df[cols] = df[cols].bfill()
        elif method == "mean":
            for col in cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            for col in cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
        elif method == "drop":
            df = df.dropna(subset=cols)
        else:
            raise ValueError(f"不支持的填充方法: {method}")

        logger.info(f"缺失值处理完成，方法: {method}")
        return df

    def fix_outliers(
        self,
        df: pd.DataFrame,
        method: str = "winsorize",
        columns: Optional[List[str]] = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ) -> pd.DataFrame:
        """修复异常值

        Args:
            df: 输入 DataFrame
            method: 处理方法，'winsorize'（截断）、'clip'（裁剪到阈值）、'nan'（设为 NaN）
            columns: 要处理的列
            lower_percentile: 下分位数
            upper_percentile: 上分位数

        Returns:
            处理后的 DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in df.columns:
                continue

            values = df[col]
            lower_bound = values.quantile(lower_percentile)
            upper_bound = values.quantile(upper_percentile)

            if method == "winsorize":
                df[col] = values.clip(lower=lower_bound, upper=upper_bound)
            elif method == "clip":
                mean = values.mean()
                std = values.std()
                if std > 0:
                    lower_clip = mean - self.outlier_std_threshold * std
                    upper_clip = mean + self.outlier_std_threshold * std
                    df[col] = values.clip(lower=lower_clip, upper=upper_clip)
            elif method == "nan":
                mask = (values < lower_bound) | (values > upper_bound)
                df.loc[mask, col] = np.nan
            else:
                raise ValueError(f"不支持的异常值处理方法: {method}")

        logger.info(f"异常值处理完成，方法: {method}")
        return df
