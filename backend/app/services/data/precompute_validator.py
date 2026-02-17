"""
预计算数据验证机制
验证预计算数据的完整性、质量和一致性
"""

from __future__ import annotations  # 延迟评估类型注解

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    """验证结果"""

    is_valid: bool
    missing_stocks: List[str]
    missing_dates: List[datetime]
    missing_indicators: List[str]
    quality_issues: List[str]
    consistency_issues: List[str]
    summary: str

    def to_dict(self) -> "Dict[str, Any]":
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "missing_stocks": [s for s in self.missing_stocks],
            "missing_dates": [d.isoformat() for d in self.missing_dates],
            "missing_indicators": self.missing_indicators,
            "quality_issues": self.quality_issues,
            "consistency_issues": self.consistency_issues,
            "summary": self.summary,
        }


class PrecomputeValidator:
    """预计算数据验证器"""

    # 必需的指标类别
    REQUIRED_INDICATOR_CATEGORIES = {
        "base": ["$open", "$high", "$low", "$close", "$volume"],
        "technical": ["MA5", "MA10", "MA20", "RSI14", "MACD"],
        "alpha": ["alpha_001"],  # 至少有一个Alpha因子
        "fundamental": ["price_change", "volatility_5d"],
    }

    def __init__(self):
        """初始化验证器"""

    def validate_precomputed_data(
        self,
        qlib_data: pd.DataFrame,
        expected_stocks: List[str],
        expected_date_range: Tuple[datetime, datetime],
        expected_indicators: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        验证预计算数据的完整性和正确性

        Args:
            qlib_data: Qlib格式的预计算数据（MultiIndex: stock_code, date）
            expected_stocks: 期望的股票列表
            expected_date_range: 期望的日期范围 (start_date, end_date)
            expected_indicators: 期望的指标列表（可选）

        Returns:
            ValidationResult对象
        """
        try:
            missing_stocks = []
            missing_dates = []
            missing_indicators = []
            quality_issues = []
            consistency_issues = []

            # 1. 检查数据完整性
            if qlib_data.empty:
                return ValidationResult(
                    is_valid=False,
                    missing_stocks=expected_stocks,
                    missing_dates=[],
                    missing_indicators=expected_indicators or [],
                    quality_issues=["数据为空"],
                    consistency_issues=[],
                    summary="预计算数据为空",
                )

            # 检查MultiIndex结构
            if (
                not isinstance(qlib_data.index, pd.MultiIndex)
                or qlib_data.index.nlevels != 2
            ):
                return ValidationResult(
                    is_valid=False,
                    missing_stocks=[],
                    missing_dates=[],
                    missing_indicators=[],
                    quality_issues=["数据格式错误：不是MultiIndex (stock_code, date)"],
                    consistency_issues=[],
                    summary="数据格式不正确",
                )

            # 获取实际数据中的股票和日期
            actual_stocks = set(qlib_data.index.get_level_values(0).unique())
            actual_dates = set(qlib_data.index.get_level_values(1).unique())
            expected_stocks_set = set(expected_stocks)

            # 检查缺失的股票
            missing_stocks = list(expected_stocks_set - actual_stocks)

            # 检查缺失的日期（检查日期范围）
            start_date, end_date = expected_date_range
            expected_dates = pd.date_range(start=start_date, end=end_date, freq="D")
            # 只检查交易日（排除周末）
            expected_dates = expected_dates[expected_dates.weekday < 5]
            missing_dates = [d for d in expected_dates if d not in actual_dates]

            # 检查缺失的指标
            if expected_indicators:
                actual_indicators = set(qlib_data.columns)
                missing_indicators = [
                    ind for ind in expected_indicators if ind not in actual_indicators
                ]

            # 2. 数据质量检查
            quality_issues = self._check_data_quality(qlib_data)

            # 3. 数据一致性检查
            consistency_issues = self._check_data_consistency(qlib_data)

            # 生成摘要
            is_valid = (
                len(missing_stocks) == 0
                and len(missing_dates) == 0
                and len(missing_indicators) == 0
                and len(quality_issues) == 0
                and len(consistency_issues) == 0
            )

            summary_parts = []
            if missing_stocks:
                summary_parts.append(f"缺失 {len(missing_stocks)} 只股票")
            if missing_dates:
                summary_parts.append(f"缺失 {len(missing_dates)} 个交易日")
            if missing_indicators:
                summary_parts.append(f"缺失 {len(missing_indicators)} 个指标")
            if quality_issues:
                summary_parts.append(f"{len(quality_issues)} 个质量问题")
            if consistency_issues:
                summary_parts.append(f"{len(consistency_issues)} 个一致性问题")

            summary = (
                "验证通过"
                if is_valid
                else "; ".join(summary_parts)
                if summary_parts
                else "未知问题"
            )

            return ValidationResult(
                is_valid=is_valid,
                missing_stocks=missing_stocks,
                missing_dates=missing_dates,
                missing_indicators=missing_indicators,
                quality_issues=quality_issues,
                consistency_issues=consistency_issues,
                summary=summary,
            )

        except Exception as e:
            logger.error(f"验证预计算数据失败: {e}")
            return ValidationResult(
                is_valid=False,
                missing_stocks=[],
                missing_dates=[],
                missing_indicators=[],
                quality_issues=[f"验证过程出错: {str(e)}"],
                consistency_issues=[],
                summary=f"验证失败: {str(e)}",
            )

    def _check_data_quality(self, qlib_data: pd.DataFrame) -> List[str]:
        """
        检查数据质量

        Args:
            qlib_data: Qlib格式数据

        Returns:
            质量问题列表
        """
        issues = []

        try:
            # 检查缺失值
            missing_counts = qlib_data.isnull().sum()
            high_missing_cols = missing_counts[
                missing_counts > len(qlib_data) * 0.1
            ]  # 缺失超过10%
            if len(high_missing_cols) > 0:
                issues.append(f"以下列缺失值过多: {list(high_missing_cols.index)}")

            # 检查异常值（价格列）
            price_cols = ["$open", "$high", "$low", "$close"]
            for col in price_cols:
                if col in qlib_data.columns:
                    # 检查负值
                    negative_count = (qlib_data[col] < 0).sum()
                    if negative_count > 0:
                        issues.append(f"{col} 有 {negative_count} 个负值")

                    # 检查异常大的值（可能是数据错误）
                    if qlib_data[col].max() > 1e6:
                        issues.append(f"{col} 有异常大的值（最大值: {qlib_data[col].max()}）")

            # 检查成交量
            if "$volume" in qlib_data.columns:
                negative_volume = (qlib_data["$volume"] < 0).sum()
                if negative_volume > 0:
                    issues.append(f"$volume 有 {negative_volume} 个负值")

            # 检查指标合理性（例如RSI应该在0-100之间）
            if "RSI14" in qlib_data.columns:
                invalid_rsi = (
                    (qlib_data["RSI14"] < 0) | (qlib_data["RSI14"] > 100)
                ).sum()
                if invalid_rsi > 0:
                    issues.append(f"RSI14 有 {invalid_rsi} 个无效值（应在0-100之间）")

            # 检查价格逻辑（high >= low, high >= close, low <= close等）
            if all(col in qlib_data.columns for col in ["$high", "$low", "$close"]):
                invalid_high_low = (qlib_data["$high"] < qlib_data["$low"]).sum()
                if invalid_high_low > 0:
                    issues.append(f"有 {invalid_high_low} 条记录 high < low")

                invalid_high_close = (qlib_data["$high"] < qlib_data["$close"]).sum()
                if invalid_high_close > 0:
                    issues.append(f"有 {invalid_high_close} 条记录 high < close")

                invalid_low_close = (qlib_data["$low"] > qlib_data["$close"]).sum()
                if invalid_low_close > 0:
                    issues.append(f"有 {invalid_low_close} 条记录 low > close")

        except Exception as e:
            logger.warning(f"数据质量检查出错: {e}")
            issues.append(f"质量检查过程出错: {str(e)}")

        return issues

    def _check_data_consistency(self, qlib_data: pd.DataFrame) -> List[str]:
        """
        检查数据一致性

        Args:
            qlib_data: Qlib格式数据

        Returns:
            一致性问题列表
        """
        issues = []

        try:
            # 检查索引排序
            if not qlib_data.index.is_monotonic_increasing:
                issues.append("索引未按 (stock_code, date) 排序")

            # 检查重复索引
            if qlib_data.index.duplicated().any():
                issues.append("存在重复的 (stock_code, date) 索引")

            # 检查每只股票的数据连续性（可选，可能因为停牌等原因有缺失）
            # 这里只检查是否有明显的日期跳跃（超过30天）
            stock_level = qlib_data.index.get_level_values(0)
            date_level = qlib_data.index.get_level_values(1)

            for stock_code in stock_level.unique():
                stock_dates = date_level[stock_level == stock_code]
                if len(stock_dates) > 1:
                    date_diffs = pd.Series(stock_dates).diff().dt.days
                    large_gaps = date_diffs[date_diffs > 30]
                    if len(large_gaps) > 0:
                        # 这可能是正常的（停牌等），只记录警告
                        logger.debug(f"股票 {stock_code} 有 {len(large_gaps)} 个超过30天的日期间隔")

        except Exception as e:
            logger.warning(f"数据一致性检查出错: {e}")
            issues.append(f"一致性检查过程出错: {str(e)}")

        return issues

    def validate_single_stock(
        self,
        qlib_data: pd.DataFrame,
        stock_code: str,
        expected_indicators: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        验证单只股票的预计算数据

        Args:
            qlib_data: Qlib格式数据（可以是多股票或单股票）
            stock_code: 股票代码
            expected_indicators: 期望的指标列表

        Returns:
            ValidationResult对象
        """
        try:
            # 提取单股票数据
            if isinstance(qlib_data.index, pd.MultiIndex):
                try:
                    stock_data = qlib_data.xs(stock_code, level=0, drop_level=False)
                except KeyError:
                    return ValidationResult(
                        is_valid=False,
                        missing_stocks=[stock_code],
                        missing_dates=[],
                        missing_indicators=expected_indicators or [],
                        quality_issues=[],
                        consistency_issues=[],
                        summary=f"股票 {stock_code} 不在数据中",
                    )
            else:
                stock_data = qlib_data

            # 检查指标
            missing_indicators = []
            if expected_indicators:
                actual_indicators = set(stock_data.columns)
                missing_indicators = [
                    ind for ind in expected_indicators if ind not in actual_indicators
                ]

            # 质量检查
            quality_issues = self._check_data_quality(stock_data)

            # 一致性检查
            consistency_issues = self._check_data_consistency(stock_data)

            is_valid = (
                len(missing_indicators) == 0
                and len(quality_issues) == 0
                and len(consistency_issues) == 0
            )

            summary = "验证通过" if is_valid else f"股票 {stock_code} 验证失败"

            return ValidationResult(
                is_valid=is_valid,
                missing_stocks=[],
                missing_dates=[],
                missing_indicators=missing_indicators,
                quality_issues=quality_issues,
                consistency_issues=consistency_issues,
                summary=summary,
            )

        except Exception as e:
            logger.error(f"验证单股票数据失败: {e}")
            return ValidationResult(
                is_valid=False,
                missing_stocks=[stock_code],
                missing_dates=[],
                missing_indicators=[],
                quality_issues=[f"验证过程出错: {str(e)}"],
                consistency_issues=[],
                summary=f"验证失败: {str(e)}",
            )
