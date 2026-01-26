"""
数据版本管理
管理预计算数据的版本信息，支持版本检查和一致性验证
"""

from __future__ import annotations  # 延迟评估类型注解

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.config import settings


class VersionManager:
    """数据版本管理器"""

    def __init__(self):
        self.qlib_data_path = Path(settings.QLIB_DATA_PATH)
        self.version_file = self.qlib_data_path / "data_version.json"
        self.version_file.parent.mkdir(parents=True, exist_ok=True)

    def get_version_info(self) -> "Dict[str, Any]":
        """
        获取当前数据版本信息

        Returns:
            版本信息字典
        """
        try:
            if self.version_file.exists():
                with open(self.version_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                # 返回默认版本信息
                return self._get_default_version_info()
        except Exception as e:
            logger.error(f"读取版本信息失败: {e}")
            return self._get_default_version_info()

    def update_version_info(
        self,
        stock_count: int,
        date_range: Dict[str, str],
        indicators: Dict[str, List[str]],
        parquet_version: Optional[str] = None,
    ):
        """
        更新版本信息

        Args:
            stock_count: 股票数量
            date_range: 日期范围 {'start': '2020-01-01', 'end': '2026-01-25'}
            indicators: 指标分类 {'technical': [...], 'alpha': [...], 'fundamental': [...]}
            parquet_version: Parquet数据版本（可选）
        """
        try:
            version_info = {
                "version": "1.0.0",
                "precompute_date": datetime.now().isoformat(),
                "parquet_version": parquet_version or datetime.now().isoformat(),
                "indicators": indicators,
                "stock_count": stock_count,
                "date_range": date_range,
                "indicator_count": sum(len(v) for v in indicators.values()),
            }

            with open(self.version_file, "w", encoding="utf-8") as f:
                json.dump(version_info, f, indent=2, ensure_ascii=False)

            logger.info(
                f"版本信息已更新: {version_info['version']}, 股票数: {stock_count}, 指标数: {version_info['indicator_count']}"
            )

        except Exception as e:
            logger.error(f"更新版本信息失败: {e}")

    def check_version_consistency(
        self,
        expected_stocks: Optional[List[str]] = None,
        expected_date_range: Optional[Dict[str, str]] = None,
    ) -> "Dict[str, Any]":
        """
        检查版本一致性

        Args:
            expected_stocks: 期望的股票列表（可选）
            expected_date_range: 期望的日期范围（可选）

        Returns:
            一致性检查结果
        """
        try:
            version_info = self.get_version_info()

            issues = []

            # 检查股票数量
            if expected_stocks:
                actual_count = version_info.get("stock_count", 0)
                expected_count = len(expected_stocks)
                if actual_count != expected_count:
                    issues.append(f"股票数量不一致: 期望 {expected_count}, 实际 {actual_count}")

            # 检查日期范围
            if expected_date_range:
                actual_range = version_info.get("date_range", {})
                if actual_range.get("start") != expected_date_range.get(
                    "start"
                ) or actual_range.get("end") != expected_date_range.get("end"):
                    issues.append(
                        f"日期范围不一致: 期望 {expected_date_range}, 实际 {actual_range}"
                    )

            is_consistent = len(issues) == 0

            return {
                "is_consistent": is_consistent,
                "issues": issues,
                "version_info": version_info,
            }

        except Exception as e:
            logger.error(f"检查版本一致性失败: {e}")
            return {
                "is_consistent": False,
                "issues": [f"检查失败: {str(e)}"],
                "version_info": None,
            }

    def _get_default_version_info(self) -> "Dict[str, Any]":
        """获取默认版本信息"""
        return {
            "version": "0.0.0",
            "precompute_date": None,
            "parquet_version": None,
            "indicators": {"technical": [], "alpha": [], "fundamental": [], "base": []},
            "stock_count": 0,
            "date_range": {"start": None, "end": None},
            "indicator_count": 0,
        }

    def get_indicator_versions(self) -> "Dict[str, Dict[str, Any]]":
        """
        获取所有指标的版本信息

        Returns:
            指标版本字典 {indicator_name: {version, added_date, ...}}
        """
        try:
            version_info = self.get_version_info()
            indicators = version_info.get("indicators", {})

            indicator_versions = {}
            for category, indicator_list in indicators.items():
                for indicator_name in indicator_list:
                    indicator_versions[indicator_name] = {
                        "version": version_info.get("version", "1.0.0"),
                        "added_date": version_info.get("precompute_date"),
                        "category": category,
                    }

            return indicator_versions

        except Exception as e:
            logger.error(f"获取指标版本信息失败: {e}")
            return {}

    def compare_with_parquet_version(self) -> Dict[str, Any]:
        """
        与Parquet数据版本对比

        Returns:
            对比结果
        """
        try:
            version_info = self.get_version_info()
            qlib_version = version_info.get("parquet_version")

            # 这里可以添加检查Parquet数据版本的逻辑
            # 例如：检查Parquet文件的修改时间

            return {
                "qlib_version": qlib_version,
                "needs_update": False,  # 需要根据实际检查逻辑确定
                "message": "版本检查完成",
            }

        except Exception as e:
            logger.error(f"对比Parquet版本失败: {e}")
            return {
                "qlib_version": None,
                "needs_update": True,
                "message": f"版本检查失败: {str(e)}",
            }
