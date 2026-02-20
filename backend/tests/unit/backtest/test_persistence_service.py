"""
持久化服务测试 (P3)

测试 persistence_service 中的纯函数工具。
主要的 save_backtest_results 等方法依赖数据库连接，此处仅测试
_safe_float、_safe_int、_to_python_type、_ensure_datetime 等可独立测试的工具函数。

源码：app/services/backtest/persistence/persistence_service.py
"""

import pytest
import numpy as np
from datetime import datetime

from app.services.backtest.persistence.persistence_service import (
    _safe_float,
    _safe_int,
    _to_python_type,
    _ensure_datetime,
)


class TestSafeFloat:
    """_safe_float 安全浮点转换测试"""

    def test_normal_float(self):
        """正常浮点数直接返回"""
        assert _safe_float(3.14) == 3.14

    def test_int_to_float(self):
        """整数转浮点"""
        assert _safe_float(42) == 42.0

    def test_none_returns_default(self):
        """None 返回默认值"""
        assert _safe_float(None) == 0.0
        assert _safe_float(None, default=-1.0) == -1.0

    def test_nan_returns_default(self):
        """NaN 返回默认值"""
        assert _safe_float(float("nan")) == 0.0

    def test_inf_returns_default(self):
        """Inf 返回默认值"""
        assert _safe_float(float("inf")) == 0.0
        assert _safe_float(float("-inf")) == 0.0

    def test_numpy_float(self):
        """numpy 浮点类型正确转换"""
        assert _safe_float(np.float64(2.5)) == 2.5

    def test_numpy_int(self):
        """numpy 整数类型正确转换"""
        assert _safe_float(np.int64(10)) == 10.0

    def test_numpy_nan(self):
        """numpy NaN 返回默认值"""
        assert _safe_float(np.float64("nan")) == 0.0

    def test_string_returns_default(self):
        """不可转换的字符串返回默认值"""
        assert _safe_float("abc") == 0.0


class TestSafeInt:
    """_safe_int 安全整数转换测试"""

    def test_normal_int(self):
        assert _safe_int(42) == 42

    def test_float_to_int(self):
        assert _safe_int(3.7) == 3

    def test_none_returns_default(self):
        assert _safe_int(None) == 0
        assert _safe_int(None, default=-1) == -1

    def test_numpy_int(self):
        assert _safe_int(np.int64(100)) == 100

    def test_numpy_float(self):
        assert _safe_int(np.float64(5.9)) == 5

    def test_string_returns_default(self):
        assert _safe_int("abc") == 0


class TestToPythonType:
    """_to_python_type 类型递归转换测试"""

    def test_numpy_int(self):
        """numpy 整数转 Python int"""
        result = _to_python_type(np.int64(42))
        assert result == 42
        assert type(result) is int

    def test_numpy_float(self):
        """numpy 浮点转 Python float"""
        result = _to_python_type(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert type(result) is float

    def test_numpy_array(self):
        """numpy 数组转 Python list"""
        result = _to_python_type(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_dict_recursive(self):
        """字典递归转换"""
        data = {"a": np.int64(1), "b": np.float64(2.5)}
        result = _to_python_type(data)
        assert result == {"a": 1, "b": 2.5}

    def test_list_recursive(self):
        """列表递归转换"""
        data = [np.int64(1), np.float64(2.5), "hello"]
        result = _to_python_type(data)
        assert result == [1, 2.5, "hello"]

    def test_datetime_to_isoformat(self):
        """datetime 转 ISO 格式字符串"""
        dt = datetime(2023, 6, 15, 10, 30, 0)
        result = _to_python_type(dt)
        assert result == "2023-06-15T10:30:00"

    def test_plain_types_passthrough(self):
        """普通 Python 类型直接返回"""
        assert _to_python_type("hello") == "hello"
        assert _to_python_type(42) == 42
        assert _to_python_type(True) is True


class TestEnsureDatetime:
    """_ensure_datetime 日期转换测试"""

    def test_none_returns_none(self):
        assert _ensure_datetime(None) is None

    def test_datetime_passthrough(self):
        dt = datetime(2023, 6, 15)
        assert _ensure_datetime(dt) == dt

    def test_string_iso_format(self):
        """ISO 格式字符串转 datetime"""
        result = _ensure_datetime("2023-06-15T10:30:00")
        assert isinstance(result, datetime)
        assert result.year == 2023
        assert result.month == 6

    def test_string_date_only(self):
        """纯日期字符串转 datetime"""
        result = _ensure_datetime("2023-06-15")
        assert isinstance(result, datetime)
        assert result.year == 2023

    def test_pandas_timestamp(self):
        """pandas Timestamp 转 datetime"""
        import pandas as pd
        ts = pd.Timestamp("2023-06-15 10:30:00")
        result = _ensure_datetime(ts)
        assert isinstance(result, datetime)
        assert result.year == 2023
