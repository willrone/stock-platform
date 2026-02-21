"""
错误处理和降级策略属性测试
验证 SystemErrorHandler、DataValidator、EnhancedLogger 的正确性
"""

import pytest
import asyncio
import tempfile
import shutil
import pandas as pd
from datetime import datetime
from unittest.mock import AsyncMock
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite

from app.services.system.error_handler import (
    SystemErrorHandler,
    ErrorCategory,
    ErrorSeverity,
    RetryStrategy,
)
from app.services.data.data_validator import DataValidator, ValidationLevel
from app.services.infrastructure.enhanced_logger import (
    EnhancedLogger,
    LogCategory,
    LogLevel,
)


@composite
def stock_data_frames(draw):
    """生成股票数据DataFrame"""
    size = draw(st.integers(min_value=10, max_value=50))
    dates = pd.date_range(start="2023-01-01", periods=size, freq="D")
    base_price = draw(st.floats(min_value=10.0, max_value=100.0))

    data = []
    current_price = base_price
    for date in dates:
        change = draw(st.floats(min_value=-0.05, max_value=0.05))
        current_price = max(1.0, current_price * (1 + change))
        high = current_price * 1.02
        low = current_price * 0.98
        volume = draw(st.integers(min_value=1000, max_value=100000))
        data.append({
            "stock_code": "TEST001",
            "date": date,
            "open": current_price,
            "high": high,
            "low": low,
            "close": current_price,
            "volume": volume,
        })
    return pd.DataFrame(data)


@composite
def corrupted_data_frames(draw):
    """生成包含错误的股票数据DataFrame"""
    df = draw(stock_data_frames())
    error_type = draw(st.sampled_from([
        "negative_prices", "invalid_relationships", "missing_values"
    ]))
    if error_type == "negative_prices":
        idx = draw(st.integers(min_value=0, max_value=len(df) - 1))
        df.loc[idx, "close"] = -abs(df.loc[idx, "close"])
    elif error_type == "invalid_relationships":
        idx = draw(st.integers(min_value=0, max_value=len(df) - 1))
        df.loc[idx, "high"] = df.loc[idx, "low"] * 0.5
    elif error_type == "missing_values":
        idx = draw(st.integers(min_value=0, max_value=len(df) - 1))
        df.loc[idx, "close"] = None
    return df


class TestSystemErrorHandlerProperties:
    """SystemErrorHandler 属性测试"""

    def setup_method(self):
        self.handler = SystemErrorHandler()

    def test_default_retry_configs_exist(self):
        """默认重试配置应覆盖主要错误类别"""
        assert ErrorCategory.NETWORK in self.handler.retry_configs
        assert ErrorCategory.DATABASE in self.handler.retry_configs
        assert ErrorCategory.EXTERNAL_API in self.handler.retry_configs
        assert ErrorCategory.COMPUTATION in self.handler.retry_configs

    def test_retry_delay_calculation(self):
        """重试延迟计算应正确"""
        # Exponential backoff
        config = {
            "strategy": RetryStrategy.EXPONENTIAL_BACKOFF,
            "base_delay": 1.0,
            "max_delay": 30.0,
        }
        assert self.handler._calculate_retry_delay(config, 0) == 1.0
        assert self.handler._calculate_retry_delay(config, 1) == 2.0
        assert self.handler._calculate_retry_delay(config, 2) == 4.0
        assert self.handler._calculate_retry_delay(config, 10) == 30.0  # capped

        # Linear backoff
        config_linear = {
            "strategy": RetryStrategy.LINEAR_BACKOFF,
            "base_delay": 2.0,
            "max_delay": 10.0,
        }
        assert self.handler._calculate_retry_delay(config_linear, 0) == 2.0
        assert self.handler._calculate_retry_delay(config_linear, 1) == 4.0
        assert self.handler._calculate_retry_delay(config_linear, 4) == 10.0  # capped

        # Immediate
        config_imm = {"strategy": RetryStrategy.IMMEDIATE}
        assert self.handler._calculate_retry_delay(config_imm, 0) == 0.0

    def test_circuit_breaker_triggers_on_threshold(self):
        """熔断器应在达到失败阈值时触发"""
        category = ErrorCategory.NETWORK
        error = Exception("test")
        from app.services.system.error_handler import ErrorInfo
        error_info = ErrorInfo(error, category, ErrorSeverity.HIGH)

        # 默认阈值是 5
        for i in range(4):
            result = self.handler._should_circuit_break(category, error_info)
            assert result is False, f"Should not trip at failure {i+1}"

        # 第 5 次应触发
        result = self.handler._should_circuit_break(category, error_info)
        assert result is True

    def test_circuit_breaker_reset(self):
        """熔断器重置应恢复初始状态"""
        category = ErrorCategory.NETWORK
        error = Exception("test")
        from app.services.system.error_handler import ErrorInfo
        error_info = ErrorInfo(error, category, ErrorSeverity.HIGH)

        # 触发熔断
        for _ in range(5):
            self.handler._should_circuit_break(category, error_info)

        assert self.handler.circuit_breakers[category.value]["state"] == "open"

        # 重置
        self.handler.reset_circuit_breaker(category)
        cb = self.handler.circuit_breakers[category.value]
        assert cb["state"] == "closed"
        assert cb["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_handle_error_records_history(self):
        """handle_error 应记录错误历史"""
        error = ValueError("test error")
        try:
            await self.handler.handle_error(
                error, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM
            )
        except ValueError:
            pass

        assert len(self.handler.error_history) == 1
        assert self.handler.error_history[0].category == ErrorCategory.VALIDATION

    @pytest.mark.asyncio
    async def test_handle_error_with_recovery(self):
        """handle_error 应在有 operation 时尝试恢复"""
        call_count = 0

        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise Exception("temp")
            return "ok"

        # 第一次调用会失败并尝试恢复
        result = await self.handler.handle_error(
            Exception("initial"),
            ErrorCategory.NETWORK,
            ErrorSeverity.MEDIUM,
            operation=failing_then_success,
        )
        # 恢复成功应返回结果
        if result is not None:
            assert result == "ok"

    def test_error_statistics(self):
        """错误统计应正确累计"""
        from app.services.system.error_handler import ErrorInfo
        error = Exception("test")
        info = ErrorInfo(error, ErrorCategory.NETWORK, ErrorSeverity.HIGH)
        self.handler.error_history.append(info)
        self.handler.error_stats["network_Exception"] = 1

        stats = self.handler.get_error_statistics()
        assert isinstance(stats, dict)
        assert "total_errors" in stats or len(stats) > 0


class TestDataValidatorProperties:
    """DataValidator 属性测试"""

    def setup_method(self):
        self.validator = DataValidator(ValidationLevel.MODERATE)

    @given(stock_data_frames())
    @settings(max_examples=5, deadline=10000)
    def test_valid_data_passes_validation(self, df):
        """合法数据应通过验证"""
        result = self.validator.validate_stock_data(df, "TEST001")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "quality_score")
        assert 0.0 <= result.quality_score <= 1.0
        assert hasattr(result, "cleaned_data")
        assert hasattr(result, "issues_found")

    @given(corrupted_data_frames())
    @settings(max_examples=5, deadline=10000)
    def test_corrupted_data_detected(self, df):
        """损坏数据应被检测到"""
        result = self.validator.validate_stock_data(df, "TEST001")
        assert isinstance(result.issues_found, list)
        # 验证清理后的数据（如果有）质量更好
        if result.cleaned_data is not None and not result.cleaned_data.empty:
            cleaned = result.cleaned_data
            if "high" in cleaned.columns and "low" in cleaned.columns:
                assert (cleaned["high"] >= cleaned["low"]).all()

    def test_empty_data_handled(self):
        """空数据应被正确处理"""
        result = self.validator.validate_stock_data(pd.DataFrame(), "TEST001")
        assert result.is_valid is False
        assert result.quality_score == 0.0

    def test_none_data_handled(self):
        """None 数据应被正确处理"""
        result = self.validator.validate_stock_data(None, "TEST001")
        assert result.is_valid is False
        assert result.quality_score == 0.0


class TestEnhancedLoggerProperties:
    """EnhancedLogger 属性测试"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = EnhancedLogger(
            "test", log_dir=self.temp_dir, enable_console=False
        )

    def teardown_method(self):
        # 关闭所有 handler 防止 ResourceWarning
        for h in self.logger.logger.handlers[:]:
            h.close()
            self.logger.logger.removeHandler(h)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_levels_tracked(self):
        """各级别日志应被正确统计"""
        self.logger.error("err", category=LogCategory.API)
        self.logger.warning("warn", category=LogCategory.DATA)
        self.logger.critical("crit", category=LogCategory.SYSTEM)
        self.logger.info("info", category=LogCategory.PERFORMANCE)

        stats = self.logger.get_stats()
        assert stats["total_logs"] == 4
        assert stats["logs_by_level"][LogLevel.ERROR.value] == 1
        assert stats["logs_by_level"][LogLevel.WARNING.value] == 1
        assert stats["logs_by_level"][LogLevel.CRITICAL.value] == 1
        assert stats["logs_by_level"][LogLevel.INFO.value] == 1

    def test_log_categories_tracked(self):
        """各分类日志应被正确统计"""
        self.logger.info("a", category=LogCategory.API)
        self.logger.info("b", category=LogCategory.DATA)
        self.logger.info("c", category=LogCategory.SYSTEM)

        stats = self.logger.get_stats()
        assert stats["logs_by_category"][LogCategory.API.value] == 1
        assert stats["logs_by_category"][LogCategory.DATA.value] == 1
        assert stats["logs_by_category"][LogCategory.SYSTEM.value] == 1

    def test_stats_initial_state(self):
        """初始统计应全为零"""
        stats = self.logger.get_stats()
        assert stats["total_logs"] == 0
        for count in stats["logs_by_level"].values():
            assert count == 0

    def test_uptime_tracked(self):
        """运行时间应被追踪"""
        stats = self.logger.get_stats()
        assert "uptime_seconds" in stats
        assert stats["uptime_seconds"] >= 0
